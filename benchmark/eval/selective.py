"""Selective Collar: thin wrapper around the vendored reference class + opt-in corrected variant.

``variant="reference"`` (default) runs the *unmodified* ``clsSelectiveCollar`` from
https://github.com/shahmananv13/selective_collar (vendored in
``vendor/selective_collar_ref/selective_collar.py``) so results stay comparable with the
published paper. A fresh instance is created per file because the reference class keeps
its zones and counters as instance state.

``variant="corrected"`` (opt-in, ``evaluate.py --selective-variant corrected``) is a
re-implementation that removes the reference-code quirks listed in README.md:
  * every reference segment keeps its own zone (no dict-key collisions),
  * a boundary inside several matching zones snaps to the *nearest* reference boundary
    (the reference snaps to the last matching zone in iteration order),
  * each hypothesis boundary is counted at most once,
  * a snap that would produce an empty/inverted segment (start >= end) is skipped.
"""
from __future__ import annotations

import importlib.util
from dataclasses import dataclass

from common import SELREF_DIR, parse_rttm, parse_rttm_text

_REF_MOD = None


def reference_module():
    global _REF_MOD
    if _REF_MOD is None:
        spec = importlib.util.spec_from_file_location(
            "selective_collar_ref", SELREF_DIR / "selective_collar.py")
        _REF_MOD = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(_REF_MOD)
    return _REF_MOD


@dataclass
class SnapResult:
    segments: list          # snapped hypothesis [(start, end, speaker)]
    rttm_text: str          # snapped hypothesis as RTTM text
    altered_starts: int
    altered_ends: int
    total_starts: int       # reference starts considered (= reference segments)
    total_ends: int


def snap_reference(ref_rttm, hyp_rttm, collar: float = 0.25, uri: str | None = None) -> SnapResult:
    """Run the vendored reference implementation, untouched."""
    cls = reference_module().clsSelectiveCollar
    sc = cls(collar_size=collar, uri=uri)
    text = sc.return_selective_collar_rttm(str(ref_rttm), str(hyp_rttm))
    return SnapResult(parse_rttm_text(text), text, sc.altered_starts, sc.altered_ends,
                      sc.traditional_start, sc.traditional_end)


def snap_corrected(ref_rttm, hyp_rttm, collar: float = 0.25, uri: str | None = None) -> SnapResult:
    """Opt-in corrected variant (see module docstring). Not used for paper-comparable numbers."""
    from pyannote.core import Annotation, Segment
    from pyannote.metrics.diarization import DiarizationErrorRate

    eps = collar / 2
    ref = parse_rttm(ref_rttm)
    hyp = parse_rttm(hyp_rttm)

    def to_ann(segs):
        ann = Annotation(uri=uri)
        for i, (s, e, k) in enumerate(segs):
            ann[Segment(s, e), i] = k
        return ann

    mapping = DiarizationErrorRate().optimal_mapping(to_ann(ref), to_ann(hyp))
    starts, ends = {}, {}
    for s, e, k in ref:
        starts.setdefault(k, []).append(s)
        ends.setdefault(k, []).append(e)

    def nearest(cands, t):
        best = None
        for c in cands:
            d = abs(c - t)
            if d <= eps + 1e-12 and (best is None or d < abs(best - t)):
                best = c
        return best

    out, a_s, a_e = [], 0, 0
    for s, e, k in hyp:
        rk = mapping.get(k)
        ns, ne = s, e
        if rk is not None:
            cs = nearest(starts.get(rk, ()), s)
            ce = nearest(ends.get(rk, ()), e)
            if cs is not None:
                ns = cs
            if ce is not None:
                ne = ce
            if ne <= ns:            # would erase the segment: keep the original times
                ns, ne = s, e
        a_s += ns != s
        a_e += ne != e
        out.append((ns, ne, k))
    text = "".join(
        f"SPEAKER {uri} 1 {s:.3f} {round(e - s, 3):.3f} <NA> <NA> {k} <NA> <NA>\n" for s, e, k in out)
    return SnapResult(out, text, a_s, a_e, len(ref), len(ref))


def snap(ref_rttm, hyp_rttm, collar: float = 0.25, uri: str | None = None,
         variant: str = "reference") -> SnapResult:
    if variant == "reference":
        return snap_reference(ref_rttm, hyp_rttm, collar, uri)
    if variant == "corrected":
        return snap_corrected(ref_rttm, hyp_rttm, collar, uri)
    raise ValueError(f"unknown selective variant: {variant}")
