# Selective Collar benchmark: evaluation harness

Scores diarization hypothesis RTTMs from any number of systems against ground truth under
**No Collar**, **Traditional Collar** and **Selective Collar**, and produces per-file CSVs,
summary tables, paired tests and paper-ready figures.

## Setup (once)

The venv lives outside OneDrive. From `benchmark\eval\` in PowerShell:

```powershell
$env:UV_PROJECT_ENVIRONMENT = "C:\Users\manan\.venvs\selcollar_eval"
uv sync --python 3.12
```

DER is computed with NIST `md-eval-22.pl`, which needs `perl`. Git for Windows ships one and it is
found automatically (`C:\Program Files\Git\usr\bin\perl.exe`); otherwise pass `--perl <path>`.

## Run

```powershell
$env:UV_PROJECT_ENVIRONMENT = "C:\Users\manan\.venvs\selcollar_eval"
cd C:\Users\manan\OneDrive\Desktop\selective_collar\benchmark\eval

uv run --no-sync python evaluate.py --systems nemotron vbx --datasets msdwild voxconverse
uv run --no-sync python summarize.py
uv run --no-sync python make_figures.py --labels vbx=VBx nemotron=Nemotron
uv run --no-sync python -m pytest          # self-checks (about 1 minute)
```

`evaluate.py` is resumable: results are cached per file and condition, keyed by the content of
the reference and hypothesis RTTMs and the scoring configuration. Re-running after more
hypotheses arrive only scores what is new or changed. `--force` ignores the cache.

Paper-matching traditional collar (see "Collar conventions"), written to a separate tree:

```powershell
uv run --no-sync python evaluate.py --systems nemotron vbx --trad-half-width 0.125 --results-dir ..\results\alt_trad_hw0.125
uv run --no-sync python summarize.py    --results-dir ..\results\alt_trad_hw0.125
uv run --no-sync python make_figures.py --results-dir ..\results\alt_trad_hw0.125
```

Useful options: `--workers N`, `--rttm-root`, `--results-dir`, `--selective-variant corrected`,
`--der-backend pyannote`, `--limit N`; `summarize.py --systems ...`; `make_figures.py
--file-set all --no-published`.

## Inputs

- Ground truth: `Dataset\MSDWild\Ground Truth RTTM\<id>.rttm`, `Dataset\VoxConverse\Ground Truth RTTM\<id>.rttm`.
  MSDWild ids `00017` and `00018` are always excluded.
- Hypotheses: `benchmark\outputs\rttm\<system>\<dataset>\<file_id>.rttm`. A missing or empty
  file is recorded in `<system>_<dataset>_status.csv` and not scored.

## Outputs (`benchmark\results\`)

| Path | Content |
| :--- | :--- |
| `per_file\<system>_<dataset>.csv` | one row per file x condition (columns below) |
| `per_file\<system>_<dataset>_status.csv` | every ground-truth file: ok / missing / empty / excluded / failed |
| `metadata.json` | configuration, collar conventions, versions, vendored commits, coverage |
| `tables\table5_summary.{csv,tex}` | mean +/- SD per system x dataset x condition, plus published Pyannote 3.1 rows |
| `tables\table6_paired_tests.{csv,tex}` | Shapiro-Wilk on paired differences, Wilcoxon signed-rank, direction, means |
| `tables\forgiveness_behaviour.{csv,tex}`, `system_ranking.csv` | boundaries snapped, reference removed, DER gaps, ranking changes |
| `tables\common_files.csv`, `dropped_files.csv`, `timing.csv` | common file set, every dropped file with its reason, scoring time |
| `figures\fig_*.{pdf,png}` | boxplots, DER decomposition, intervention density, data ablation, DER by system |

Tables carry `file_set`: `common` (files where every included system has a non-empty hypothesis
and all metrics are defined in all conditions; used for LaTeX and all cross-system comparisons),
`all` (every scored file of that system) and `published` (copied from the paper).

Per-file columns: `DER MS FA SC JER CDER SER BER` (fractions; MS/FA/SC relative to the scored
reference speaker time of that file and condition), `DER_pyannote`, `DER_pyannote_merged`,
`DER_crop_check` (cross-checks), `ref_speech_scored_s` (DER denominator), `ref_speech_total_s`,
`ref_removed_s` / `ref_removed_pct` / `ref_removed_timeline_s` (reference removed by the
traditional collar, speaker-time and timeline), `ref_overlap_s`, `n_ref_segments`,
`n_ref_segments_erased`, `ref_erased_segments_s`, `n_hyp_segments`, `n_spk_ref`, `n_spk_hyp`,
`altered_starts`, `altered_ends`, `total_boundaries`, `boundaries_forgiven`, `time_total_s`
(all metrics), `time_core_s` (pyannote DER only; for Selective, the snapping call of the
reference class), configuration columns and `errors`.

## Metric implementations

All from the vendored DiarizationMetricInOne stack (`vendor\DiarizationMetricInOne`), called
in-process rather than through `run_for_all.py` (which shells out three times per file and uses
POSIX paths):

| Metric | Implementation |
| :--- | :--- |
| DER, MS, FA, SC | NIST `md-eval-22.pl` from dscore-ovl; overlap always scored |
| JER | dscore-ovl `scorelib.metrics.jer`, 10 ms frames |
| CDER | CDER_Metric `CSSDErrorRate` with its `rttm_read_cder` pre-processing |
| SER, BER | X-LANCE BER `score.py` |
| DER_pyannote | pyannote.metrics 4.0.0 `DiarizationErrorRate` (cross-check only) |

## Collar conventions

- **No Collar**: collar 0.0 everywhere.
- **Selective**: hypothesis snapped by the vendored reference class with c* = 0.25 (eps = 0.125),
  then all metrics against the untouched reference at collar 0.0.
- **Traditional**: `--trad-half-width h` (default 0.25) means no scoring within +/- h seconds of
  every reference start and end.
  - DER/MS/FA/SC: `md-eval -c h` (md-eval/dscore collar is a half-width).
  - JER/CDER/SER/BER have no exclusion-zone collar, so reference and hypothesis are both cropped
    to the scored region and scored at collar 0 (the paper: "traditional collars ... shrink the
    boundaries of both the reference and hypothesis").
  - `DER_pyannote` uses `collar = 2h` (pyannote's collar is the total width).
  - `DER_crop_check` is md-eval at collar 0 on the cropped pair.

**The published paper's Traditional Collar appears to be +/-0.125 s, not +/-0.25 s.** Table 4 of
the paper ("Unnec. Segs Lost" and "Unnec. Secs Lost" for bpzsc, bxcfq, gtnjb, ibrnm, oqwpd,
vdlvr, vuewy) is reproduced exactly from the ground truth with h = 0.125 (3 segments / 0.83 s
for bpzsc, 1 / 0.48 s for bxcfq, ...) and not with h = 0.25 (31 segments / 10.89 s for bpzsc).
This matches the commented-out code in the public repo, `DiarizationErrorRate(collar=0.25)`,
where pyannote's collar is the total window. The default here stays at h = 0.25 as the paper
text describes ("0.5 s window"); use `--trad-half-width 0.125` to compare against the published
Pyannote 3.1 Traditional row.

Known differences between the DER columns:
- md-eval decides the speaker mapping on the full file and then applies the collar; pyannote
  maps after removing the collar. A few files differ under Traditional for that reason
  (`DER` vs `DER_pyannote`); `DER_crop_check` equals pyannote.
- `DER_pyannote` builds one track per RTTM line like the reference class, so same-speaker
  overlapping hypothesis segments count twice; md-eval and `DER_pyannote_merged` merge them.

## Selective Collar: vendored reference and its quirks

`vendor\selective_collar_ref\selective_collar.py` is the unmodified public class
(`clsSelectiveCollar`). `selective.py` calls it with a fresh instance per file. Behaviour that
is kept as is (default `--selective-variant reference`):

1. Zones are stored in a dict keyed by `"{start+eps}_{end-eps}"`, so two reference segments with
   identical start and end (for example two speakers) share one entry and the earlier one is lost.
2. A hypothesis boundary inside several matching zones is snapped once per zone in iteration
   order: the last zone wins (not the nearest), and `altered_starts` / `altered_ends` can count
   one boundary more than once.
3. A snap can make start >= end; pyannote then drops the segment silently.
4. Snapping can create overlapping same-speaker hypothesis segments; with one track per line
   pyannote counts them twice.
5. Zones and counters are instance state, so re-using an instance across files mixes files.
6. `parse_rttm` treats every non-blank line as a SPEAKER record and ignores the file id.
7. The scan is O(N_hyp x N_ref), not the O((N_hyp + N_ref) log N_ref) stated in the paper.
8. Times are rounded to 3 decimals on read and on write. The harness applies the same rounding
   to the reference and hypothesis in all three conditions.

`--selective-variant corrected` is an opt-in re-implementation without quirks 1 to 4 (nearest
zone, one count per boundary, no inverted segments). It is not used for any default output.

## Tests

`tests\test_selfcheck.py`: public-repo sample (altered starts 5/8, ends 6/8, identical snapped
RTTM); corrected variant agrees on the sample; jittered ground truth gives Traditional <=
Selective <= No Collar; md-eval `-c h` = pyannote `collar=2h` = zero-collar on the cropped pair;
end-to-end evaluate -> summarize -> make_figures on two synthetic systems written to a temp
directory (`make_synthetic.py`).
