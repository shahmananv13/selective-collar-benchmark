# Selective Collar: multi-model benchmark

Benchmark code, system outputs and scores for the multi-model evaluation of the
**Selective Collar**, a speaker-diarization evaluation protocol that forgives a predicted boundary
only when it lies within the tolerance window *and* the mapped speaker identity is correct.

- Selective Collar algorithm and reference code: https://github.com/shahmananv13/selective_collar
- Original paper: S. M. Vinod *et al.*, "Beyond Uniform Forgiveness: Introducing the Selective Collar to
  Mitigate Evaluation Bias in Speaker Diarization," *IEEE Access*, vol. 14, pp. 136563-136580, 2026,
  doi: 10.1109/ACCESS.2026.3729661.

This repository carries out the multi-model validation listed there as future work: two further
diarization systems are scored under three collar conditions and compared with the published
Pyannote 3.1 results.

## What was evaluated

| | |
|---|---|
| Datasets | VoxConverse test (232 files), MSDWild `few.val` (488 files; 00017 and 00018 excluded as in the original paper) |
| Systems scored here | VBx (ResNet101 x-vectors + AHC + VB-HMM) with BUT energy VAD; NVIDIA Nemotron-3-Diarization (Streaming-Sortformer-style, via NeMo) |
| Baseline | Pyannote 3.1, published aggregate values from the original paper (not re-scored here) |
| Conditions | No Collar (0 s); Traditional Collar; Selective Collar (c* = 0.25 s, tolerance 0.125 s, then scored at 0 s) |
| Metrics | DER with MS / FA / SC, JER, CDER, SER, BER |

### Collar convention (read this before comparing numbers)

The main results use a Traditional Collar of **+/-0.125 s** around each reference boundary (a 0.25 s
window; `collar=0.25` in pyannote.metrics). This is the window passed to the Selective Collar and the
setting that reproduces the published Pyannote 3.1 numbers. NIST md-eval, and therefore the VoxConverse
and MSDWild protocols, apply **+/-0.25 s**. Results for that wider setting are in `benchmark/results/`
(top level); the main-setting results are in `benchmark/results/alt_trad_hw0.125/`.

## Headline results (per-file mean DER)

| System | Dataset | No Collar | Selective | Traditional | Boundaries snapped |
|---|---|---|---|---|---|
| VBx | MSDWild | 0.459 | 0.452 | 0.412 | 14.8% |
| VBx | VoxConverse | 0.225 | 0.221 | 0.199 | 19.9% |
| Pyannote 3.1 (published) | MSDWild | 0.308 | 0.294 | 0.247 | n/a |
| Pyannote 3.1 (published) | VoxConverse | 0.123 | 0.117 | 0.095 | n/a |
| Nemotron-3 | MSDWild | 0.178 | 0.154 | 0.116 | 55.3% |
| Nemotron-3 | VoxConverse | 0.082 | 0.072 | 0.061 | 61.3% |

Caveats: one system per architectural family; the VBx pipeline uses an untuned energy VAD and is well
below tuned VBx cascades; Nemotron-3 was trained on the VoxConverse test set (MSDWild is the held-out
comparison) and outputs at most eight speakers (61 VoxConverse files have more).

## Layout

```
benchmark/
  nemotron/     Nemotron-3-Diarization runner (own environment, README with setup)
  vbx/          VBx + energy VAD runner (own environment, README with setup and code patches)
  eval/         evaluation harness: three conditions, all metrics, tables, statistical tests, figures
  outputs/
    rttm/<system>/<dataset>/<file_id>.rttm    hypotheses produced by the two systems
    runtime/                                  per-file audio duration and inference time, run logs
  results/
    alt_trad_hw0.125/   main setting (Traditional +/-0.125 s): per-file CSVs, tables, figures
    per_file/, tables/, figures/              Traditional +/-0.25 s variant
  literature/   literature review notes and verified BibTeX
```

## Reproducing

Each component has its own Python 3.12 environment (managed with `uv`); see the README in each folder.

1. **Data** (not included): download VoxConverse test audio and RTTM, and MSDWild `few.val` audio and
   RTTM, from their official sources. The scripts expect
   `Dataset/<VoxConverse|MSDWild>/Raw Audio Files/*.wav` and
   `Dataset/<VoxConverse|MSDWild>/Ground Truth RTTM/<file_id>.rttm` next to `benchmark/`.
2. **Model weights** (not included):
   - VBx: the ResNet101 16 kHz x-vector weights (`final.onnx`, `raw_81.pth`) exceed GitHub's file limit.
     Get them from https://github.com/BUTSpeechFIT/VBx (`VBx/models/ResNet101_16kHz/nnet/`, split zip
     parts) and place them in `benchmark/vbx/VBx/models/ResNet101_16kHz/nnet/`. The PLDA and transform
     files are included.
   - Nemotron-3-Diarization: downloaded from Hugging Face (`nvidia/Nemotron-3-Diarization`) on first run.
3. **Inference**: `benchmark/nemotron/run_nemotron.py` and `benchmark/vbx/run_vbx.py` (resumable).
4. **Scoring**, from `benchmark/eval/`:
   ```
   python evaluate.py --systems nemotron vbx --datasets msdwild voxconverse --trad-half-width 0.125 --results-dir ../results/alt_trad_hw0.125
   python summarize.py --results-dir ../results/alt_trad_hw0.125
   ```
   The hypotheses in `benchmark/outputs/rttm/` allow steps 1-3 to be skipped except for the ground truth.

The Selective Collar is run through the unmodified reference class (`clsSelectiveCollar`), vendored at
`benchmark/eval/vendor/selective_collar_ref/`. An opt-in corrected variant (nearest-zone snapping) exists
in `benchmark/eval/selective.py`; it gives the same DER to three decimals and is not used for reported numbers.

## Third-party code

- `benchmark/vbx/VBx/`: BUT VBx (Apache-2.0), https://github.com/BUTSpeechFIT/VBx, with the small patches
  listed in `benchmark/vbx/README.md`.
- `benchmark/eval/vendor/DiarizationMetricInOne/`: https://github.com/liutaocode/DiarizationMetricInOne
  (bundles dscore-ovl / NIST md-eval, CDER_Metric and BER; see the licenses inside).
- `benchmark/nemotron/vendor/wheels/`: NVIDIA NeMo wheel built from the NeMo repository (Apache-2.0).

Datasets and model weights keep their own licenses and are not redistributed here.
