"""Score hypothesis RTTMs under No Collar / Selective Collar / Traditional Collar.

    python evaluate.py --systems nemotron vbx --datasets msdwild voxconverse

Reads   <rttm-root>/<system>/<dataset>/<file_id>.rttm      (default benchmark/outputs/rttm)
Writes  <results-dir>/per_file/<system>_<dataset>.csv       one row per file x condition
        <results-dir>/per_file/<system>_<dataset>_status.csv  one row per ground-truth file
        <results-dir>/cache/<system>_<dataset>.jsonl         resumable per-file cache
        <results-dir>/metadata.json                          conventions, versions, coverage

Re-running is cheap: a (file, condition) result is reused when the reference RTTM, the
hypothesis RTTM and the scoring configuration are unchanged (content hash).
"""
from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import multiprocessing as mp
import os
import sys
import time
from pathlib import Path

import pandas as pd

from common import (CONDITIONS, DATASETS, DEFAULT_RESULTS_DIR, DEFAULT_RTTM_ROOT, HARNESS_VERSION,
                    find_perl, parse_rttm)

COLUMNS = [
    "system", "dataset", "file_id", "condition",
    "DER", "MS", "FA", "SC", "JER", "CDER", "SER", "BER",
    "DER_pyannote", "DER_pyannote_merged", "DER_crop_check",
    "ref_speech_scored_s", "ref_speech_scored_pyannote_s", "ref_speech_total_s",
    "ref_removed_s", "ref_removed_pct", "ref_removed_timeline_s",
    "ref_support_s", "ref_overlap_s",
    "n_ref_segments", "n_ref_segments_erased", "ref_erased_segments_s", "n_hyp_segments",
    "n_spk_ref", "n_spk_hyp",
    "altered_starts", "altered_ends", "total_boundaries", "boundaries_forgiven",
    "BER_ref_part", "BER_fa_mean",
    "time_total_s", "time_core_s", "time_der_pyannote_s",
    "trad_half_width_s", "selective_collar_s", "selective_variant", "der_backend", "errors",
]


def sha1_file(path) -> str:
    return hashlib.sha1(Path(path).read_bytes()).hexdigest()


def cond_config(cond, cfg):
    """The part of the configuration a condition's result depends on (cache key)."""
    c = {"v": HARNESS_VERSION, "der_backend": cfg["der_backend"], "cond": cond}
    if cond == "selective":
        c.update(collar=cfg["selective_collar"], variant=cfg["selective_variant"])
    if cond == "traditional":
        c.update(half_width=cfg["trad_half_width"])
    return c


def cache_key(ref_hash, hyp_hash, cond, cfg):
    blob = json.dumps([ref_hash, hyp_hash, cond_config(cond, cfg)], sort_keys=True)
    return hashlib.sha1(blob.encode()).hexdigest()


def load_cache(path):
    cache = {}
    if path.exists():
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                try:
                    rec = json.loads(line)
                    cache[rec["key"]] = rec["row"]
                except Exception:  # noqa: BLE001  (truncated last line after a crash)
                    continue
    return cache


def _work(task):
    from scoring import evaluate_file
    try:
        return task, evaluate_file(task), None
    except Exception as ex:  # noqa: BLE001
        return task, None, f"{type(ex).__name__}: {ex}"


def run_pair(system, dataset, args, cfg, pool):
    info = DATASETS[dataset]
    gt_dir = Path(info["gt_dir"])
    hyp_dir = Path(args.rttm_root) / system / dataset
    out_dir = Path(args.results_dir)
    (out_dir / "per_file").mkdir(parents=True, exist_ok=True)
    (out_dir / "cache").mkdir(parents=True, exist_ok=True)
    cov = {"system": system, "dataset": dataset, "hyp_dir": str(hyp_dir)}
    if not hyp_dir.is_dir():
        print(f"[{system}/{dataset}] no hypothesis directory ({hyp_dir}) - skipped")
        cov["status"] = "no hypothesis directory"
        return cov

    gt_ids = sorted(p.stem for p in gt_dir.glob("*.rttm"))
    excluded = [i for i in gt_ids if i in info["exclude"]]
    ids = [i for i in gt_ids if i not in info["exclude"]]
    if args.limit:
        ids = ids[: args.limit]
    cache_path = out_dir / "cache" / f"{system}_{dataset}.jsonl"
    cache = {} if args.force else load_cache(cache_path)

    status, rows, tasks = {}, [], []
    for fid in ids:
        ref_path, hyp_path = gt_dir / f"{fid}.rttm", hyp_dir / f"{fid}.rttm"
        if not hyp_path.exists():
            status[fid] = "missing_hypothesis"
            continue
        try:
            if not parse_rttm(hyp_path):
                status[fid] = "empty_hypothesis"
                continue
            rh, hh = sha1_file(ref_path), sha1_file(hyp_path)
        except OSError as ex:
            status[fid] = f"unreadable: {ex}"
            continue
        keys = {c: cache_key(rh, hh, c, cfg) for c in CONDITIONS}
        todo = [c for c in CONDITIONS if keys[c] not in cache]
        rows.extend(cache[keys[c]] for c in CONDITIONS if keys[c] in cache)
        if todo:
            tasks.append(dict(system=system, dataset=dataset, file_id=fid, ref_path=str(ref_path),
                              hyp_path=str(hyp_path), conditions=todo, cfg=cfg, keys=keys))
        else:
            status[fid] = "ok"

    n_cached = sum(1 for v in status.values() if v == "ok")
    print(f"[{system}/{dataset}] {len(ids)} reference files: {n_cached} cached, {len(tasks)} to score, "
          f"{sum(v == 'missing_hypothesis' for v in status.values())} missing, "
          f"{sum(v == 'empty_hypothesis' for v in status.values())} empty", flush=True)
    t0 = time.time()
    if tasks:
        it = pool.imap_unordered(_work, tasks, chunksize=1) if pool else map(_work, tasks)
        with open(cache_path, "a", encoding="utf-8") as cf:
            for n, (task, res, err) in enumerate(it, 1):
                fid = task["file_id"]
                if err:
                    status[fid] = f"scoring_failed: {err}"
                    print(f"  ! {fid}: {err}", flush=True)
                else:
                    status[fid] = "ok"
                    for c, row in res.items():
                        rows.append(row)
                        cf.write(json.dumps({"key": task["keys"][c], "file_id": fid,
                                             "condition": c, "row": row}) + "\n")
                    cf.flush()
                if n % 25 == 0 or n == len(tasks):
                    print(f"  {n}/{len(tasks)} scored ({time.time() - t0:.0f}s)", flush=True)

    df = pd.DataFrame(rows)
    if len(df):
        df = df[df["file_id"].map(lambda f: status.get(f) == "ok")]
        for c in COLUMNS:
            if c not in df.columns:
                df[c] = pd.NA
        df["condition"] = pd.Categorical(df["condition"], CONDITIONS, ordered=True)
        df = df[COLUMNS].sort_values(["file_id", "condition"])
    else:
        df = pd.DataFrame(columns=COLUMNS)
    csv_path = out_dir / "per_file" / f"{system}_{dataset}.csv"
    df.to_csv(csv_path, index=False, float_format="%.6g")
    st = pd.DataFrame([{"system": system, "dataset": dataset, "file_id": f, "status": status[f]} for f in ids]
                      + [{"system": system, "dataset": dataset, "file_id": f,
                          "status": "excluded_by_protocol"} for f in excluded])
    st.to_csv(out_dir / "per_file" / f"{system}_{dataset}_status.csv", index=False)

    counts = st["status"].map(lambda s: s.split(":")[0]).value_counts().to_dict()
    cov.update(status="ok", n_reference_files=len(ids), counts=counts, per_file_csv=str(csv_path),
               n_rows=int(len(df)), n_rows_with_errors=int((df["errors"].fillna("") != "").sum()) if len(df) else 0)
    print(f"[{system}/{dataset}] wrote {csv_path}  ({counts})", flush=True)
    return cov


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--systems", nargs="+", required=True)
    ap.add_argument("--datasets", nargs="+", default=list(DATASETS), choices=list(DATASETS))
    ap.add_argument("--rttm-root", default=str(DEFAULT_RTTM_ROOT))
    ap.add_argument("--results-dir", default=str(DEFAULT_RESULTS_DIR))
    ap.add_argument("--trad-half-width", type=float, default=0.25,
                    help="Traditional collar HALF-width in seconds: scoring is suppressed within "
                         "+/- this value of every reference boundary (md-eval/dscore '-c'). "
                         "Default 0.25 (a 0.5 s window). pyannote.metrics collar=0.25 corresponds to 0.125.")
    ap.add_argument("--selective-collar", type=float, default=0.25,
                    help="Selective collar size c* (snapping tolerance eps = c*/2). Default 0.25.")
    ap.add_argument("--selective-variant", choices=["reference", "corrected"], default="reference",
                    help="'reference' = unmodified published class (default); 'corrected' = opt-in fixes.")
    ap.add_argument("--der-backend", choices=["mdeval", "pyannote"], default="mdeval",
                    help="DER/MS/FA/SC implementation (default: NIST md-eval via dscore-ovl; needs perl).")
    ap.add_argument("--perl", default=None, help="path to perl (default: auto-detect / $SELCOLLAR_PERL)")
    ap.add_argument("--workers", type=int, default=max(1, min(8, (os.cpu_count() or 2) // 2)))
    ap.add_argument("--limit", type=int, default=0, help="only the first N files per dataset (debug)")
    ap.add_argument("--force", action="store_true", help="ignore the cache")
    args = ap.parse_args(argv)

    perl = find_perl(args.perl)
    if args.der_backend == "mdeval" and not perl:
        sys.exit("perl not found (needed by md-eval). Install Git for Windows / Strawberry Perl, pass "
                 "--perl <path>, or use --der-backend pyannote.")
    cfg = dict(trad_half_width=args.trad_half_width, selective_collar=args.selective_collar,
               selective_variant=args.selective_variant, der_backend=args.der_backend, perl=perl)

    pool = mp.Pool(args.workers) if args.workers > 1 else None
    coverage = []
    try:
        for system in args.systems:
            for dataset in args.datasets:
                coverage.append(run_pair(system, dataset, args, cfg, pool))
    finally:
        if pool:
            pool.close()
            pool.join()

    import numpy, scipy, pyannote.core, pyannote.metrics  # noqa: E401
    meta_path = Path(args.results_dir) / "metadata.json"
    meta = json.loads(meta_path.read_text(encoding="utf-8")) if meta_path.exists() else {}
    meta.update({
        "harness_version": HARNESS_VERSION,
        "last_run": dt.datetime.now().isoformat(timespec="seconds"),
        "rttm_root": str(args.rttm_root),
        "config": {k: v for k, v in cfg.items()},
        "collar_convention": {
            "traditional_half_width_s": args.trad_half_width,
            "traditional_window_total_s": 2 * args.trad_half_width,
            "description": "Traditional collar: no scoring within +/- half_width seconds of every "
                           "reference segment start and end; overlapped speech is scored (no skip-overlap).",
            "DER_MS_FA_SC": ("NIST md-eval-22.pl (dscore-ovl) with -c <half_width> (md-eval collar = half-width)"
                             if args.der_backend == "mdeval" else
                             "pyannote.metrics DiarizationErrorRate(collar=2*half_width)"),
            "JER_CDER_SER_BER_traditional": "reference AND hypothesis cropped to the scored region "
                                            "(outside the +/- half_width zones), then scored with zero collar",
            "DER_pyannote": "pyannote.metrics DiarizationErrorRate(collar=2*half_width, skip_overlap=False); "
                            "pyannote 'collar' is the TOTAL width. One track per RTTM line (as in the "
                            "reference selective-collar code), so same-speaker overlaps count twice.",
            "DER_pyannote_merged": "same, after merging same-speaker overlapping segments (md-eval behaviour)",
            "selective": f"snap with c*={args.selective_collar} (eps={args.selective_collar / 2}), "
                         f"variant={args.selective_variant}, then score against the untouched reference at collar 0",
            "no_collar": "collar 0.0",
            "MS_FA_SC": "per-file component time / per-file scored reference speaker time",
            "time_rounding": "RTTM start/duration rounded to 3 decimals as in the reference class",
        },
        "excluded_ids": {d: sorted(DATASETS[d]["exclude"]) for d in DATASETS},
        "versions": {"python": sys.version.split()[0], "numpy": numpy.__version__, "scipy": scipy.__version__,
                     "pandas": pd.__version__,
                     "pyannote.core": getattr(pyannote.core, "__version__", "?"),
                     "pyannote.metrics": getattr(pyannote.metrics, "__version__", "?")},
        "vendored": {
            "selective_collar": "github.com/shahmananv13/selective_collar @ "
                                + (Path(__file__).parent / "vendor/selective_collar_ref/COMMIT").read_text().strip(),
            "DiarizationMetricInOne": "github.com/liutaocode/DiarizationMetricInOne @ 418ad7f6c387a6bbd4d28f6a210bd18354c645e7",
            "dscore-ovl": "github.com/liutaocode/dscore-ovl @ de8fb127025c3da9e796bcdb2d18c68d63bd310a",
            "CDER_Metric": "github.com/SpeechClub/CDER_Metric @ b509fd71e2f2df190cf738707663e9ca4e7618c3",
            "BER": "github.com/X-LANCE/BER @ 8dbb7ca9d38778fc4dd57d8ae2e8f010b3d35739",
        },
    })
    cov = {f"{c['system']}/{c['dataset']}": c for c in meta.get("coverage", [])}
    cov.update({f"{c['system']}/{c['dataset']}": c for c in coverage})
    meta["coverage"] = list(cov.values())
    meta_path.parent.mkdir(parents=True, exist_ok=True)
    meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print(f"metadata -> {meta_path}")


if __name__ == "__main__":
    main()
