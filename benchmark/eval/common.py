"""Shared configuration, paths, constants and RTTM helpers for the evaluation harness."""
from __future__ import annotations

import os
import shutil
from pathlib import Path

HARNESS_VERSION = "1.0"

EVAL_DIR = Path(__file__).resolve().parent
BENCH_DIR = EVAL_DIR.parent
REPO_ROOT = BENCH_DIR.parent
VENDOR_DIR = EVAL_DIR / "vendor"
DMIO_DIR = VENDOR_DIR / "DiarizationMetricInOne"
SELREF_DIR = VENDOR_DIR / "selective_collar_ref"

DEFAULT_RTTM_ROOT = BENCH_DIR / "outputs" / "rttm"
DEFAULT_RESULTS_DIR = BENCH_DIR / "results"

DATASETS = {
    "msdwild": {
        "label": "MSDWild",
        "gt_dir": REPO_ROOT / "Dataset" / "MSDWild" / "Ground Truth RTTM",
        "exclude": {"00017", "00018"},
    },
    "voxconverse": {
        "label": "VoxConverse",
        "gt_dir": REPO_ROOT / "Dataset" / "VoxConverse" / "Ground Truth RTTM",
        "exclude": set(),
    },
}

# Condition keys (CSV values), in the order used by the paper's tables.
CONDITIONS = ["no_collar", "selective", "traditional"]
CONDITION_LABELS = {
    "no_collar": "No Collar",
    "selective": "Selective Collar",
    "traditional": "Traditional Collar",
}
# Colour convention of the published paper (matplotlib tab10 blue/orange/green).
CONDITION_COLORS = {
    "no_collar": "#1f77b4",
    "selective": "#ff7f0e",
    "traditional": "#2ca02c",
}

MAIN_METRICS = ["DER", "JER", "CDER", "SER", "BER"]
COMPONENTS = ["MS", "FA", "SC"]
TABLE_METRICS = MAIN_METRICS + COMPONENTS

PUBLISHED_SYSTEM = "pyannote_3.1 (published)"
PUBLISHED_NOTE = (
    "Values copied from Table 5 of Vinod et al., IEEE Access 14 (2026), "
    "doi:10.1109/ACCESS.2026.3729661 (Pyannote 3.1; per-file mean +/- SD). "
    "Not recomputed by this harness."
)
# (mean, sd); sd is None where the paper does not report it (MS/FA/SC).
PUBLISHED = {
    "msdwild": {
        "n_files": 488,
        "no_collar": {"DER": (0.308, 0.203), "JER": (0.416, 0.223), "CDER": (0.624, 0.401),
                      "SER": (0.372, 0.223), "BER": (0.478, 0.316),
                      "MS": (0.086, None), "FA": (0.074, None), "SC": (0.149, None)},
        "selective": {"DER": (0.294, 0.204), "JER": (0.404, 0.227), "CDER": (0.609, 0.403),
                      "SER": (0.364, 0.222), "BER": (0.468, 0.317),
                      "MS": (0.080, None), "FA": (0.067, None), "SC": (0.147, None)},
        "traditional": {"DER": (0.247, 0.201), "JER": (0.378, 0.236), "CDER": (0.711, 0.419),
                        "SER": (0.293, 0.178), "BER": (0.414, 0.299),
                        "MS": (0.060, None), "FA": (0.042, None), "SC": (0.145, None)},
    },
    "voxconverse": {
        "n_files": 232,
        "no_collar": {"DER": (0.123, 0.093), "JER": (0.312, 0.234), "CDER": (0.813, 0.787),
                      "SER": (0.258, 0.159), "BER": (0.316, 0.247),
                      "MS": (0.036, None), "FA": (0.040, None), "SC": (0.047, None)},
        "selective": {"DER": (0.117, 0.091), "JER": (0.307, 0.235), "CDER": (0.806, 0.789),
                      "SER": (0.252, 0.158), "BER": (0.311, 0.248),
                      "MS": (0.034, None), "FA": (0.036, None), "SC": (0.047, None)},
        "traditional": {"DER": (0.095, 0.082), "JER": (0.292, 0.236), "CDER": (0.839, 0.797),
                        "SER": (0.202, 0.125), "BER": (0.278, 0.232),
                        "MS": (0.028, None), "FA": (0.025, None), "SC": (0.043, None)},
    },
}


def dataset_label(ds: str) -> str:
    return DATASETS.get(ds, {}).get("label", ds)


# --------------------------------------------------------------------------- RTTM
def parse_rttm(path) -> list[tuple[float, float, str]]:
    """Parse an RTTM file exactly like the reference ``clsSelectiveCollar.parse_rttm``.

    start and duration are rounded to 3 decimals, end = round(start + duration, 3).
    Only ``SPEAKER`` records are kept (the reference class assumes every non-blank
    line is one); zero/negative-duration records are dropped.
    Returns a list of (start, end, speaker) in file order.
    """
    segs = []
    with open(path, "r", encoding="utf-8", errors="replace") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < 8 or parts[0] != "SPEAKER":
                continue
            start = round(float(parts[3]), 3)
            dur = round(float(parts[4]), 3)
            end = round(start + dur, 3)
            if end <= start:
                continue
            segs.append((start, end, parts[7]))
    return segs


def parse_rttm_text(text: str) -> list[tuple[float, float, str]]:
    segs = []
    for line in text.splitlines():
        parts = line.strip().split()
        if len(parts) < 8 or parts[0] != "SPEAKER":
            continue
        start = round(float(parts[3]), 3)
        dur = round(float(parts[4]), 3)
        end = round(start + dur, 3)
        if end <= start:
            continue
        segs.append((start, end, parts[7]))
    return segs


def write_rttm(path, segs, uri: str) -> None:
    """Write normalised single-space RTTM (3 decimals), as all vendored scorers expect."""
    with open(path, "w", encoding="utf-8", newline="\n") as f:
        for s, e, spk in segs:
            f.write(f"SPEAKER {uri} 1 {s:.3f} {round(e - s, 3):.3f} <NA> <NA> {spk} <NA> <NA>\n")


def find_perl(explicit: str | None = None) -> str | None:
    """Locate a perl interpreter for NIST md-eval (Git for Windows ships one)."""
    cands = [explicit, os.environ.get("SELCOLLAR_PERL"), shutil.which("perl"),
             r"C:\Program Files\Git\usr\bin\perl.exe",
             r"C:\Program Files (x86)\Git\usr\bin\perl.exe",
             os.path.expandvars(r"%LOCALAPPDATA%\Programs\Git\usr\bin\perl.exe"),
             r"C:\Strawberry\perl\bin\perl.exe"]
    for c in cands:
        if c and Path(c).exists():
            return str(c)
    return None
