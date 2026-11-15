"""Per-file scoring under the three collar conditions.

Metric implementations (all from the vendored DiarizationMetricInOne stack, called
in-process instead of through its subprocess wrapper ``run_for_all.py``):

  DER, MS, FA, SC : NIST md-eval-22.pl as shipped in dscore-ovl (perl), overlap scored.
                    md-eval ``-c`` is a HALF-width: scoring is suppressed within
                    +/- c seconds of every reference boundary.
  JER             : dscore-ovl ``scorelib.metrics.jer`` (10 ms frames).
  CDER            : CDER_Metric ``CSSDErrorRate`` (+ its ``rttm_read_cder`` merging).
  SER, BER        : X-LANCE BER ``score.py``.
  DER_pyannote    : pyannote.metrics DiarizationErrorRate cross-check. pyannote's
                    ``collar`` is the TOTAL width, so the equivalent call is
                    ``collar = 2 * half_width``.

Conditions:
  no_collar   : everything scored on (reference, hypothesis), collar 0.
  selective   : hypothesis snapped by the Selective Collar (c* = 0.25, eps = c*/2),
                then everything scored against the untouched reference, collar 0.
  traditional : DER/MS/FA/SC from md-eval with ``-c half_width`` (native collar).
                JER/CDER/SER/BER have no exclusion-zone collar of their own, so (as the
                paper describes: "traditional collars ... shrink the boundaries of both
                the reference and hypothesis") reference and hypothesis are both cropped
                to the scored region (everything outside +/- half_width of a reference
                boundary) and scored with zero collar. ``DER_crop_check`` is md-eval DER
                (collar 0) on that same cropped pair, for consistency checking.
"""
from __future__ import annotations

import bisect
import contextlib
import importlib.util
import io
import math
import os
import shutil
import subprocess
import sys
import tempfile
import time
import warnings
from pathlib import Path

import numpy as np

from common import DMIO_DIR, parse_rttm, write_rttm

NAN = float("nan")
_STATE = {}


# ----------------------------------------------------------------- vendored imports
def _vendor():
    """Lazy import of the vendored scorers (once per process)."""
    if _STATE:
        return _STATE
    warnings.filterwarnings("ignore")
    for sub in ("dscore-ovl", "CDER_Metric"):
        p = str(DMIO_DIR / sub)
        if p not in sys.path:
            sys.path.insert(0, p)
    from scorelib import metrics as ds_metrics            # noqa: E402
    from scorelib import turn as ds_turn                  # noqa: E402
    from scorelib import uem as ds_uem                    # noqa: E402
    from scorelib import rttm as ds_rttm                  # noqa: E402
    from scorelib.score import turns_to_frames            # noqa: E402
    ds_turn.warn = lambda *a, **k: None                   # silence per-turn merge warnings
    from diarization import CSSDErrorRate                 # CDER_Metric  # noqa: E402
    from pre_process import rttm_read_cder                # CDER_Metric  # noqa: E402
    spec = importlib.util.spec_from_file_location("xlance_ber_score", DMIO_DIR / "BER" / "score.py")
    ber = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(ber)
    from pyannote.core import Annotation, Segment, Timeline
    from pyannote.metrics.diarization import DiarizationErrorRate
    _STATE.update(ds_metrics=ds_metrics, ds_turn=ds_turn, ds_uem=ds_uem, ds_rttm=ds_rttm,
                  turns_to_frames=turns_to_frames, CSSDErrorRate=CSSDErrorRate,
                  rttm_read_cder=rttm_read_cder, ber=ber, Annotation=Annotation,
                  Segment=Segment, Timeline=Timeline, PyDER=DiarizationErrorRate,
                  mdeval=str(DMIO_DIR / "dscore-ovl" / "scorelib" / "md-eval-22.pl"))
    return _STATE


# ------------------------------------------------------------------ interval helpers
def merge_intervals(iv):
    out = []
    for s, e in sorted(iv):
        if out and s <= out[-1][1]:
            if e > out[-1][1]:
                out[-1][1] = e
        else:
            out.append([s, e])
    return [(s, e) for s, e in out]


def collar_zones(ref_segs, half_width):
    """Merged exclusion zones: +/- half_width around every reference start and end."""
    if half_width <= 0:
        return []
    return merge_intervals([(b - half_width, b + half_width) for s, e, _ in ref_segs for b in (s, e)])


def subtract(seg, zones, zone_starts):
    """Parts of (s, e) outside the merged, sorted ``zones``."""
    s, e = seg
    out = []
    i = max(bisect.bisect_right(zone_starts, s) - 1, 0)
    cur = s
    while i < len(zones) and zones[i][0] < e:
        zs, ze = zones[i]
        if ze > cur:
            if zs > cur:
                out.append((cur, min(zs, e)))
            cur = max(cur, ze)
        i += 1
    if cur < e:
        out.append((cur, e))
    return out


def crop_outside(segs, zones):
    """Crop labelled segments to the region outside ``zones`` (3-decimal rounding)."""
    if not zones:
        return list(segs)
    zs = [z[0] for z in zones]
    out = []
    for s, e, k in segs:
        for a, b in subtract((s, e), zones, zs):
            a, b = round(a, 3), round(b, 3)
            if b > a:
                out.append((a, b, k))
    return out


def speaker_time(segs):
    """Total speaker time with same-speaker overlaps merged (the DER denominator)."""
    by = {}
    for s, e, k in segs:
        by.setdefault(k, []).append((s, e))
    return sum(e - s for iv in by.values() for s, e in merge_intervals(iv))


def timeline_stats(segs):
    """(speech support duration, duration where >= 2 distinct speakers overlap)."""
    by = {}
    for s, e, k in segs:
        by.setdefault(k, []).append((s, e))
    ev = []
    for iv in by.values():
        for s, e in merge_intervals(iv):
            ev.append((s, 1))
            ev.append((e, -1))
    ev.sort()
    support = overlap = 0.0
    n, prev = 0, None
    for t, d in ev:
        if prev is not None and n >= 1:
            support += t - prev
            if n >= 2:
                overlap += t - prev
        n += d
        prev = t
    return support, overlap


# ------------------------------------------------------------------------- scorers
def _extent(ref, hyp):
    allsegs = list(ref) + list(hyp)
    return min(s for s, _, _ in allsegs), max(e for _, e, _ in allsegs)


def mdeval_der(ref, hyp, uri, workdir, collar_half_width, perl, tag):
    """DER via NIST md-eval-22 (dscore-ovl pipeline: gen UEM from extents, merge turns)."""
    V = _vendor()
    Turn = V["ds_turn"].Turn
    rt = V["ds_turn"].merge_turns([Turn(s, offset=e, speaker_id=k, file_id=uri) for s, e, k in ref])
    st = V["ds_turn"].merge_turns([Turn(s, offset=e, speaker_id=k, file_id=uri) for s, e, k in hyp])
    uem = V["ds_uem"].gen_uem(rt, st)
    rf, sf, uf = (os.path.join(workdir, f"{tag}_{n}") for n in ("ref.rttm", "sys.rttm", "all.uem"))
    V["ds_rttm"].write_rttm(rf, rt)
    V["ds_rttm"].write_rttm(sf, st)
    V["ds_uem"].write_uem(uf, uem)
    cmd = [perl, V["mdeval"], "-af", "-r", rf, "-s", sf, "-c", str(collar_half_width), "-u", uf]
    proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    out = proc.stdout.decode("utf-8", errors="replace")
    M = V["ds_metrics"]
    try:
        scored = float(M.SCORED_SPEAKER_REO.findall(out)[-1])
        miss = float(M.MISS_SPEAKER_REO.findall(out)[-1])
        fa = float(M.FA_SPEAKER_REO.findall(out)[-1])
        conf = float(M.ERROR_SPEAKER_REO.findall(out)[-1])
    except IndexError:
        raise RuntimeError("md-eval produced no scores: " + out[-300:].replace("\n", " | "))
    return {"scored": scored, "miss": miss, "fa": fa, "conf": conf}


def merge_same_speaker(segs):
    """Merge overlapping/touching segments of the same speaker (what dscore does before md-eval)."""
    by = {}
    for s, e, k in segs:
        by.setdefault(k, []).append((s, e))
    return sorted((s, e, k) for k, iv in by.items() for s, e in merge_intervals(iv))


def pyannote_der(ref, hyp, uri, collar_total, merge=False):
    """pyannote.metrics DER. ``merge=False`` builds annotations exactly like the reference
    selective-collar code (one track per RTTM line), so same-speaker overlapping segments
    count twice; ``merge=True`` merges them first (md-eval/dscore behaviour)."""
    V = _vendor()
    if merge:
        ref, hyp = merge_same_speaker(ref), merge_same_speaker(hyp)
    Annotation, Segment, Timeline = V["Annotation"], V["Segment"], V["Timeline"]

    def ann(segs):
        a = Annotation(uri=uri)
        for i, (s, e, k) in enumerate(segs):
            a[Segment(s, e), i] = k
        return a
    lo, hi = _extent(ref, hyp)
    d = V["PyDER"](collar=collar_total, skip_overlap=False)(
        ann(ref), ann(hyp), uem=Timeline([Segment(lo, hi)]), detailed=True)
    return {"scored": d["total"], "miss": d["missed detection"], "fa": d["false alarm"],
            "conf": d["confusion"]}


def rates(c):
    t = c["scored"]
    if not t or t <= 0:
        return {"DER": NAN, "MS": NAN, "FA": NAN, "SC": NAN}
    return {"DER": (c["miss"] + c["fa"] + c["conf"]) / t, "MS": c["miss"] / t,
            "FA": c["fa"] / t, "SC": c["conf"] / t}


def dscore_jer(ref, hyp, uri):
    V = _vendor()
    Turn = V["ds_turn"].Turn
    rt = [Turn(s, offset=e, speaker_id=k, file_id=uri) for s, e, k in ref]
    st = [Turn(s, offset=e, speaker_id=k, file_id=uri) for s, e, k in hyp]
    regions = [_extent(ref, hyp)]
    rl = V["turns_to_frames"](rt, regions, step=0.010)
    sl = V["turns_to_frames"](st, regions, step=0.010)
    cm = V["ds_metrics"].contingency_matrix(rl, sl)
    f2j, _ = V["ds_metrics"].jer({uri: rl.sum(axis=0)}, {uri: sl.sum(axis=0)}, {uri: cm}, 0)
    return f2j[uri] / 100.0


def cder(ref_path, hyp_path, uri):
    V = _vendor()
    r = V["rttm_read_cder"](ref_path)
    h = V["rttm_read_cder"](hyp_path)
    return float(V["CSSDErrorRate"]()(r[uri][1], h[uri][1]))


def ber_ser(ref_path, hyp_path, detail_path):
    V = _vendor()
    argv = sys.argv
    sys.argv = ["score.py", "-s", hyp_path, "-r", ref_path, "-d", detail_path]
    try:
        with contextlib.redirect_stdout(io.StringIO()):
            jer, ser, ref_part, fa_dur, fa_seg, fa_mean, ber = V["ber"].main()
    finally:
        sys.argv = argv
    return {"SER": ser, "BER": ber, "BER_ref_part": ref_part, "BER_fa_mean": fa_mean}


def score_all(ref, hyp, uri, workdir, tag, cfg, der_collar_half_width=0.0,
              seg_ref=None, seg_hyp=None):
    """All metrics for one (ref, hyp) pair. ``seg_ref``/``seg_hyp`` (default: ref/hyp) are
    the versions handed to JER/CDER/SER/BER (the cropped pair in the traditional condition)."""
    out, errs = {}, []
    seg_ref = ref if seg_ref is None else seg_ref
    seg_hyp = hyp if seg_hyp is None else seg_hyp

    # DER + components
    try:
        if cfg["der_backend"] == "mdeval":
            c = mdeval_der(ref, hyp, uri, workdir, der_collar_half_width, cfg["perl"], tag)
        else:
            c = pyannote_der(ref, hyp, uri, 2 * der_collar_half_width)
        out.update(rates(c))
        out["ref_speech_scored_s"] = c["scored"]
    except Exception as ex:  # noqa: BLE001
        errs.append(f"DER:{type(ex).__name__}:{ex}")
        out.update(DER=NAN, MS=NAN, FA=NAN, SC=NAN, ref_speech_scored_s=NAN)

    # pyannote cross-check (also the "core" timing reported for no_collar / traditional)
    t0 = time.perf_counter()
    try:
        pc = pyannote_der(ref, hyp, uri, 2 * der_collar_half_width)
        out["DER_pyannote"] = rates(pc)["DER"]
        out["ref_speech_scored_pyannote_s"] = pc["scored"]
    except Exception as ex:  # noqa: BLE001
        errs.append(f"DER_pyannote:{type(ex).__name__}:{ex}")
        out["DER_pyannote"] = NAN
        out["ref_speech_scored_pyannote_s"] = NAN
    out["time_der_pyannote_s"] = time.perf_counter() - t0
    try:
        out["DER_pyannote_merged"] = rates(
            pyannote_der(ref, hyp, uri, 2 * der_collar_half_width, merge=True))["DER"]
    except Exception as ex:  # noqa: BLE001
        errs.append(f"DER_pyannote_merged:{type(ex).__name__}:{ex}")
        out["DER_pyannote_merged"] = NAN

    # segment-level / frame-level metrics
    if not seg_ref or not seg_hyp:
        errs.append("empty reference after collar" if not seg_ref else "empty hypothesis after collar")
        out.update(JER=NAN, CDER=NAN, SER=NAN, BER=NAN, BER_ref_part=NAN, BER_fa_mean=NAN)
    else:
        rp = os.path.join(workdir, f"{tag}_segref.rttm")
        hp = os.path.join(workdir, f"{tag}_seghyp.rttm")
        write_rttm(rp, seg_ref, uri)
        write_rttm(hp, seg_hyp, uri)
        try:
            out["JER"] = dscore_jer(seg_ref, seg_hyp, uri)
        except Exception as ex:  # noqa: BLE001
            errs.append(f"JER:{type(ex).__name__}:{ex}")
            out["JER"] = NAN
        try:
            out["CDER"] = cder(rp, hp, uri)
        except Exception as ex:  # noqa: BLE001
            errs.append(f"CDER:{type(ex).__name__}:{ex}")
            out["CDER"] = NAN
        try:
            out.update(ber_ser(rp, hp, os.path.join(workdir, f"{tag}_ber_detail.txt")))
        except Exception as ex:  # noqa: BLE001
            errs.append(f"BER:{type(ex).__name__}:{ex}")
            out.update(SER=NAN, BER=NAN, BER_ref_part=NAN, BER_fa_mean=NAN)
    out["errors"] = "; ".join(errs)
    return out


# ---------------------------------------------------------------------- file worker
def evaluate_file(task):
    """Score one file under the requested conditions. Returns {condition: row dict}.

    task = dict(system, dataset, file_id, ref_path, hyp_path, conditions, cfg)
    cfg  = dict(trad_half_width, selective_collar, selective_variant, der_backend, perl)
    """
    from selective import snap
    _vendor()
    cfg = task["cfg"]
    uri = task["file_id"]
    h = float(cfg["trad_half_width"])
    workdir = tempfile.mkdtemp(prefix="selcollar_")
    rows = {}
    try:
        ref = parse_rttm(task["ref_path"])
        hyp = parse_rttm(task["hyp_path"])
        if not ref or not hyp:
            raise ValueError("empty reference" if not ref else "empty hypothesis")
        # normalised copies (uri = file id, 3 decimals) used by every scorer
        ref_n = os.path.join(workdir, "ref_norm.rttm")
        hyp_n = os.path.join(workdir, "hyp_norm.rttm")
        write_rttm(ref_n, ref, uri)
        write_rttm(hyp_n, hyp, uri)

        zones = collar_zones(ref, h)
        zstarts = [z[0] for z in zones]
        ref_crop = crop_outside(ref, zones)
        erased = [(s, e) for s, e, _ in ref if not subtract((s, e), zones, zstarts)]
        total_spk = speaker_time(ref)
        support, overlap = timeline_stats(ref)
        support_c, _ = timeline_stats(ref_crop) if ref_crop else (0.0, 0.0)
        base = {
            "system": task["system"], "dataset": task["dataset"], "file_id": uri,
            "n_ref_segments": len(ref), "n_hyp_segments": len(hyp),
            "n_spk_ref": len({k for _, _, k in ref}), "n_spk_hyp": len({k for _, _, k in hyp}),
            "total_boundaries": 2 * len(ref),
            "ref_speech_total_s": total_spk, "ref_support_s": support, "ref_overlap_s": overlap,
            "trad_half_width_s": h, "selective_collar_s": float(cfg["selective_collar"]),
            "selective_variant": cfg["selective_variant"], "der_backend": cfg["der_backend"],
        }
        untouched = {"ref_removed_s": 0.0, "ref_removed_pct": 0.0, "ref_removed_timeline_s": 0.0,
                     "n_ref_segments_erased": 0, "ref_erased_segments_s": 0.0}

        for cond in task["conditions"]:
            t0 = time.perf_counter()
            row = dict(base, condition=cond)
            if cond == "no_collar":
                row.update(score_all(ref, hyp, uri, workdir, "nc", cfg))
                row.update(untouched, altered_starts=0, altered_ends=0, boundaries_forgiven=0,
                           time_core_s=row["time_der_pyannote_s"])
            elif cond == "selective":
                ts = time.perf_counter()
                sr = snap(ref_n, hyp_n, collar=float(cfg["selective_collar"]), uri=uri,
                          variant=cfg["selective_variant"])
                t_snap = time.perf_counter() - ts
                row.update(score_all(ref, sr.segments, uri, workdir, "sel", cfg))
                row.update(untouched, altered_starts=sr.altered_starts, altered_ends=sr.altered_ends,
                           boundaries_forgiven=sr.altered_starts + sr.altered_ends,
                           time_core_s=t_snap)
                if sr.total_starts != len(ref):
                    row["errors"] = (row["errors"] + "; " if row["errors"] else "") + "boundary count mismatch"
            elif cond == "traditional":
                hyp_crop = crop_outside(hyp, zones)
                row.update(score_all(ref, hyp, uri, workdir, "trad", cfg, der_collar_half_width=h,
                                     seg_ref=ref_crop, seg_hyp=hyp_crop))
                removed = total_spk - speaker_time(ref_crop)
                row.update(
                    ref_removed_s=removed,
                    ref_removed_pct=100.0 * removed / total_spk if total_spk > 0 else NAN,
                    ref_removed_timeline_s=support - support_c,
                    n_ref_segments_erased=len(erased),
                    ref_erased_segments_s=sum(e - s for s, e in erased),
                    altered_starts=0, altered_ends=0, boundaries_forgiven=2 * len(ref),
                    time_core_s=row["time_der_pyannote_s"])
                # consistency check: md-eval at collar 0 on the cropped pair
                try:
                    if cfg["der_backend"] == "mdeval" and ref_crop and hyp_crop:
                        row["DER_crop_check"] = rates(mdeval_der(
                            ref_crop, hyp_crop, uri, workdir, 0.0, cfg["perl"], "tradchk"))["DER"]
                except Exception as ex:  # noqa: BLE001
                    row["errors"] = (row["errors"] + "; " if row["errors"] else "") + f"cropcheck:{ex}"
            else:
                raise ValueError(cond)
            row.setdefault("DER_crop_check", NAN)
            row["time_total_s"] = time.perf_counter() - t0
            rows[cond] = {k: (None if isinstance(v, float) and math.isnan(v) else
                              (float(v) if isinstance(v, (np.floating,)) else
                               int(v) if isinstance(v, (np.integer,)) else v))
                          for k, v in row.items()}
    finally:
        shutil.rmtree(workdir, ignore_errors=True)
    return rows
