"""Write a synthetic "system" (jittered ground truth) for pipeline testing.

    python make_synthetic.py --out <temp_rttm_root> --system synth_a --n-files 20

Output layout mirrors the real one: <out>/<system>/<dataset>/<file_id>.rttm.
Never point --out at benchmark/outputs/rttm.
"""
from __future__ import annotations

import argparse
import random
from pathlib import Path

from common import DATASETS, DEFAULT_RTTM_ROOT, parse_rttm


def jitter_segments(segs, rng, jitter=0.2, p_confuse=0.05, p_drop=0.03, p_insert=0.03):
    """Jitter boundaries uniformly in +/- jitter; confuse/drop/insert a few segments."""
    spk = sorted({k for _, _, k in segs})
    names = {k: f"hyp{i}" for i, k in enumerate(spk)}
    out = []
    for s, e, k in segs:
        if rng.random() < p_drop:
            continue
        s2 = max(0.0, s + rng.uniform(-jitter, jitter))
        e2 = e + rng.uniform(-jitter, jitter)
        if e2 - s2 < 0.05:
            continue
        lab = names[k]
        if rng.random() < p_confuse and len(spk) > 1:
            lab = names[rng.choice([x for x in spk if x != k])]
        out.append((round(s2, 3), round(e2, 3), lab))
        if rng.random() < p_insert:
            out.append((round(e2 + 0.3, 3), round(e2 + 0.3 + rng.uniform(0.2, 1.0), 3), "hyp_fa"))
    return sorted(out)


def make(out_root, system, datasets, n_files, seed=0, jitter=0.2, empty_first=False):
    rng = random.Random(seed)
    written = {}
    for ds in datasets:
        info = DATASETS[ds]
        ids = sorted(p.stem for p in Path(info["gt_dir"]).glob("*.rttm") if p.stem not in info["exclude"])
        ids = ids[:: max(1, len(ids) // n_files)][:n_files] if n_files else ids
        d = Path(out_root) / system / ds
        d.mkdir(parents=True, exist_ok=True)
        for j, fid in enumerate(ids):
            segs = parse_rttm(Path(info["gt_dir"]) / f"{fid}.rttm")
            hyp = [] if (empty_first and j == 0) else jitter_segments(segs, rng, jitter)
            with open(d / f"{fid}.rttm", "w", encoding="utf-8", newline="\n") as f:
                for s, e, k in hyp:
                    f.write(f"SPEAKER {fid} 1 {s:.3f} {e - s:.3f} <NA> <NA> {k} <NA> <NA>\n")
        written[ds] = ids
    return written


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", required=True)
    ap.add_argument("--system", default="synth")
    ap.add_argument("--datasets", nargs="+", default=list(DATASETS))
    ap.add_argument("--n-files", type=int, default=20)
    ap.add_argument("--jitter", type=float, default=0.2)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--empty-first", action="store_true", help="write an empty RTTM for the first file")
    a = ap.parse_args()
    if Path(a.out).resolve() == DEFAULT_RTTM_ROOT.resolve():
        raise SystemExit("refusing to write synthetic data into the real RTTM root")
    w = make(a.out, a.system, a.datasets, a.n_files, a.seed, a.jitter, a.empty_first)
    print({k: len(v) for k, v in w.items()})
