# VBx runner (selective-collar benchmark)

Conventional clustering-based diarization baseline: BUT energy VAD ->
ResNet101 16 kHz x-vectors -> AHC -> VB-HMM (VBx, Landini et al. 2022).
Self-contained copy of the code that was previously run from `VBX\`.

## Layout

```
benchmark/vbx/
  run_vbx.py            CLI runner (resumable, per-file timing)
  energy_vad/           BUT energy VAD (energy_VAD.py, gmm.py)   <- VBX\energy_vad
  VBx/                  BUT VBx python package                   <- VBX\VBx\VBx\VBx\VBx (upstream 57466e6)
    models/resnet.py
    models/ResNet101_16kHz/{plda,transform.h5,nnet/final.onnx,nnet/raw_81.pth}
  pyproject.toml
```

Model files are included (about 200 MB). `raw_81.pth` is the unzipped
concatenation of upstream `raw_81.pth.zip.part*` (verified byte-identical);
`final.onnx` is the upstream ONNX export of the same network. If they are ever
missing, fetch `VBx/models/ResNet101_16kHz` from https://github.com/BUTSpeechFIT/VBx
and run `cat raw_81.pth.zip.part* > raw_81.pth.zip && unzip raw_81.pth.zip`.

## Setup (Windows, Git Bash; venv lives outside OneDrive)

```bash
uv venv --python 3.12 "C:/Users/manan/.venvs/selcollar_vbx"
PY="C:/Users/manan/.venvs/selcollar_vbx/Scripts/python.exe"
uv pip install --python "$PY" torch==2.11.0 --index-url https://download.pytorch.org/whl/cu128
uv pip install --python "$PY" numpy==2.5.4 scipy==1.18.1 soundfile==0.14.0 soxr==1.1.0 \
    h5py==3.16.0 fastcluster==1.3.0 kaldi_io==0.9.8 numexpr==2.14.2
# optional, only for --backend onnx:
uv pip install --python "$PY" onnxruntime-gpu==1.32.0
```

The venv used for the paper run was built with the `uv pip` commands above.
`pyproject.toml` pins the same versions
(`UV_PROJECT_ENVIRONMENT="C:/Users/manan/.venvs/selcollar_vbx" uv sync` inside
`benchmark/vbx`), but that route was not exercised.

## Run

```bash
cd C:/Users/manan/OneDrive/Desktop/selective_collar
"$PY" -W ignore benchmark/vbx/run_vbx.py msdwild voxconverse --vb-workers 8 \
    > benchmark/outputs/runtime/vbx_run.log 2>&1
```

Re-running the same command resumes (files with an existing RTTM are skipped)
and retries failures. Options: `--ids a,b`, `--limit N`, `--force`,
`--batch-size` (default 32), `--backend {pytorch,onnx}`, `--device {cuda,cpu}`,
`--prep-workers`, `--vb-workers`.

Outputs:

- `benchmark/outputs/rttm/vbx/<dataset>/<file_id>.rttm`
  (`SPEAKER <file_id> 1 <start:.3f> <dur:.3f> <NA> <NA> <spk> <NA> <NA>`, written atomically)
- `benchmark/outputs/runtime/vbx_<dataset>.csv` (`file_id,audio_seconds,inference_seconds`)
- `benchmark/outputs/runtime/vbx_<dataset>_stages.csv` (per-stage seconds, x-vector / segment / speaker counts)
- `benchmark/outputs/runtime/vbx_empty.txt` (ids with an empty RTTM), `vbx_<dataset>_failed.txt`
- cache (outside OneDrive): `~/.cache/selective_collar/vbx_wav16k/<dataset>/*.wav` and
  `~/.cache/selective_collar/vbx_work/<dataset>/{lab,exp,rttm_raw}/` (lab, ark, seg, raw VBx RTTM)

`inference_seconds` = energy VAD + x-vector extraction (wav read, fbank, GPU
forward) + AHC/VB-HMM for that file. It excludes the copy/resample into the wav
cache and model loading. The CPU stages run in parallel worker processes, so
per-file times include some contention; wall-clock totals are in `vbx_run.log`.

MSDWild ids 00017 and 00018 are skipped (excluded from the study).

## Configuration (same as the original `VBX\` scripts)

- Audio: 16 kHz mono 16-bit wav (both datasets already are; files are copied, others would be down-mixed and resampled with soxr HQ).
- Energy VAD (`energy_VAD.py`): threshold 0.5, median window 50 frames (median threshold 5), 25 ms window / 10 ms shift, 3-Gaussian GMM with 5 re-alignments, dither 8.0, seed 3.
- x-vectors: ResNet101 16 kHz, 64-dim fbank (25 ms / 10 ms, 20-7600 Hz), floating CMN (150/149), window 144 frames (1.44 s), jump 24 frames (0.24 s), embedding dim 256.
- VB-HMM (`vbhmm.py`): `--init AHC+VB --threshold -0.015 --lda-dim 128 --Fa 0.3 --Fb 17 --loopP 0.99`, `transform.h5` + `plda` of ResNet101_16kHz, defaults `--target-energy 1.0 --init-smoothing 5.0`, 40 VB iterations.

## Differences from the original scripts / patches to the copied code

1. **x-vector backend**: the original used `--backend onnx` (`final.onnx`). That
   graph has a fixed batch size of 1, so `run_vbx.py` defaults to the PyTorch
   backend with `raw_81.pth` (same network and weights) on the GPU (torch cu128,
   fp32) and embeds windows in batches of 32. Feature extraction, segmentation
   and key naming are a verbatim port of `VBx/predict.py` into
   `XvectorExtractor.extract_file`. ONNX vs PyTorch embeddings were compared on
   random inputs: max abs difference 2.7e-4 (cosine > 0.999999). `--backend onnx`
   is still available.
2. `VBx/vbhmm.py`: package-relative imports; body wrapped in `main(argv=None)`;
   `write_output` receives `file_name` explicitly (as in the user's
   `End to End VBx` copy); a recording with a single x-vector is assigned one
   speaker instead of crashing in the 2-GMM calibration / linkage.
3. `VBx/diarization_lib.py`: `read_xvector_timing_dict` wraps `np.loadtxt` in
   `np.atleast_2d` (segments files with one row).
4. `VBx/predict.py`: package-relative imports and `main(argv=None)` only
   (kept for reference; not used by `run_vbx.py`).
5. `energy_vad/energy_VAD.py`: `from . import gmm`, missing `import logging`
   added, new `vad_file(wav, lab, threshold, median_window_length)` = the
   per-file loop body of `main()` (same computation and `.lab` format),
   `__main__` guard restored.
6. RTTM post-formatting in `run_vbx.py`: VBx writes `%f` times; they are
   rewritten with 3 decimals and the plain file stem as recording id
   (the old runner produced `$name.wav`-style ids). Zero-duration lines are dropped.
7. No numpy/scipy alias fixes were needed (code runs unchanged on numpy 2.5 /
   scipy 1.18 / Python 3.12). `kaldi_io` prints a harmless KALDI_ROOT warning.
