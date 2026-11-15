"""Self-checks: vendored Selective Collar, collar conventions, and the end-to-end pipeline."""
import random
import subprocess
import sys
import tempfile
from pathlib import Path

import pandas as pd
import pytest

from common import CONDITIONS, DATASETS, SELREF_DIR, find_perl, parse_rttm, parse_rttm_text, write_rttm
from selective import snap

EVAL_DIR = Path(__file__).resolve().parents[1]
GT = SELREF_DIR / "ground_truth.rttm"
PRED = SELREF_DIR / "raw_prediction.rttm"
CFG = dict(trad_half_width=0.25, selective_collar=0.25, selective_variant="reference",
           der_backend="mdeval", perl=find_perl())
needs_perl = pytest.mark.skipif(CFG["perl"] is None, reason="perl (md-eval) not available")


def test_reference_sample_counts_and_output():
    """Public repo sample: altered starts 5/8, altered ends 6/8, identical snapped RTTM."""
    r = snap(GT, PRED, collar=0.25, uri="sample_meeting")
    assert (r.altered_starts, r.total_starts) == (5, 8)
    assert (r.altered_ends, r.total_ends) == (6, 8)
    expected = parse_rttm_text((SELREF_DIR / "selective_collar_output.rttm").read_text())
    assert sorted(r.segments) == sorted(expected)


def test_corrected_variant_agrees_on_clean_sample():
    a = snap(GT, PRED, collar=0.25, uri="sample_meeting", variant="reference")
    b = snap(GT, PRED, collar=0.25, uri="sample_meeting", variant="corrected")
    assert sorted(a.segments) == sorted(b.segments)
    assert (a.altered_starts, a.altered_ends) == (b.altered_starts, b.altered_ends)


def _jittered(tmp, jitter, seed=3):
    rng = random.Random(seed)
    ref = parse_rttm(GT)
    names = {k: f"h{i}" for i, k in enumerate(sorted({k for _, _, k in ref}))}
    hyp = [(round(max(0, s + rng.uniform(-jitter, jitter)), 3), round(e + rng.uniform(-jitter, jitter), 3), names[k])
           for s, e, k in ref]
    hp = Path(tmp) / "sample_meeting.rttm"
    write_rttm(hp, hyp, "sample_meeting")
    return hp


@needs_perl
def test_selective_between_traditional_and_no_collar(tmp_path):
    """Hypothesis = ground truth + jitter: Traditional <= Selective <= No Collar (DER)."""
    from scoring import evaluate_file
    hp = _jittered(tmp_path, jitter=0.2)
    rows = evaluate_file(dict(system="t", dataset="sample", file_id="sample_meeting", ref_path=str(GT),
                              hyp_path=str(hp), conditions=CONDITIONS, cfg=CFG))
    nc, sel, tr = (rows[c] for c in CONDITIONS)
    assert not any(rows[c]["errors"] for c in CONDITIONS)
    assert tr["DER"] <= sel["DER"] <= nc["DER"]
    assert sel["DER"] < nc["DER"] and nc["DER"] > 0
    assert 0 < sel["boundaries_forgiven"] <= tr["boundaries_forgiven"] == 16
    # the selective collar never touches the reference; the traditional collar shrinks it
    assert sel["ref_speech_scored_s"] == pytest.approx(nc["ref_speech_scored_s"])
    assert tr["ref_speech_scored_s"] < nc["ref_speech_scored_s"]
    assert tr["ref_removed_s"] == pytest.approx(nc["ref_speech_scored_s"] - tr["ref_speech_scored_s"], abs=1e-3)


@needs_perl
def test_collar_conventions_agree(tmp_path):
    """md-eval -c h (half-width) == pyannote collar=2h (total width) == zero-collar DER on cropped pair."""
    from scoring import evaluate_file
    hp = _jittered(tmp_path, jitter=0.4, seed=7)
    for h in (0.125, 0.25):
        rows = evaluate_file(dict(system="t", dataset="sample", file_id="sample_meeting", ref_path=str(GT),
                                  hyp_path=str(hp), conditions=["no_collar", "traditional"],
                                  cfg=dict(CFG, trad_half_width=h)))
        tr, nc = rows["traditional"], rows["no_collar"]
        assert tr["DER"] == pytest.approx(tr["DER_pyannote"], abs=2e-3)
        assert tr["DER"] == pytest.approx(tr["DER_crop_check"], abs=2e-3)
        assert nc["DER"] == pytest.approx(nc["DER_pyannote"], abs=2e-3)
        # 8 segments x 2 boundaries x h seconds of speech removed (no overlapping zones in the sample)
        assert tr["ref_removed_s"] == pytest.approx(16 * h, abs=1e-6)


@needs_perl
@pytest.mark.skipif(not all(Path(d["gt_dir"]).is_dir() for d in DATASETS.values()),
                    reason="ground-truth RTTMs not available")
def test_end_to_end_synthetic_pipeline():
    """evaluate -> summarize -> make_figures on two synthetic systems (20 files per dataset)."""
    with tempfile.TemporaryDirectory(prefix="selcollar_e2e_") as tmp:
        rttm, res = Path(tmp) / "rttm", Path(tmp) / "results"

        def run(*args):
            p = subprocess.run([sys.executable, *args], cwd=EVAL_DIR, capture_output=True, text=True)
            assert p.returncode == 0, p.stdout[-2000:] + p.stderr[-2000:]
            return p.stdout

        run("make_synthetic.py", "--out", str(rttm), "--system", "synth_a", "--n-files", "20", "--seed", "1")
        run("make_synthetic.py", "--out", str(rttm), "--system", "synth_b", "--n-files", "20", "--seed", "2",
            "--jitter", "0.08", "--empty-first")
        common = ["--rttm-root", str(rttm), "--results-dir", str(res)]
        run("evaluate.py", "--systems", "synth_a", "synth_b", "--workers", "4", *common)
        out2 = run("evaluate.py", "--systems", "synth_a", "synth_b", "--workers", "1", *common)
        assert "20 cached, 0 to score" in out2          # resumable cache
        run("summarize.py", "--results-dir", str(res))
        run("make_figures.py", "--results-dir", str(res))

        df = pd.read_csv(res / "per_file" / "synth_a_msdwild.csv", dtype={"file_id": str})
        assert len(df) == 60 and set(df["condition"]) == set(CONDITIONS)
        assert not {"00017", "00018"} & set(df["file_id"])
        m = df.groupby("condition")["DER"].mean()
        assert m["traditional"] < m["selective"] < m["no_collar"]
        st = pd.read_csv(res / "per_file" / "synth_b_msdwild_status.csv")
        assert (st["status"] == "empty_hypothesis").sum() == 1
        common_files = pd.read_csv(res / "tables" / "common_files.csv")
        assert (common_files["dataset"] == "msdwild").sum() == 19
        for f in ("table5_summary.csv", "table5_summary.tex", "table6_paired_tests.csv", "table6_paired_tests.tex",
                  "forgiveness_behaviour.csv", "forgiveness_behaviour.tex", "system_ranking.csv",
                  "dropped_files.csv", "timing.csv"):
            assert (res / "tables" / f).stat().st_size > 0, f
        for f in ("fig_boxplots_all", "fig_boxplots_msdwild", "fig_der_decomposition", "fig_intervention_density",
                  "fig_data_ablation", "fig_der_by_system"):
            for ext in ("pdf", "png"):
                assert (res / "figures" / f"{f}.{ext}").stat().st_size > 0, f
