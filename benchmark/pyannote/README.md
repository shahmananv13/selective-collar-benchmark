# pyannote 3.1 runner (selective-collar benchmark)

Runs the pre-trained `pyannote/speaker-diarization-3.1` pipeline (Hugging Face,
default hyperparameters, no fine-tuning, no `num_speakers` / `min_speakers` /
`max_speakers`) over MSDWild few.val and VoxConverse test and writes one RTTM
per file.

```
benchmark/pyannote/
  run_pyannote.py        CLI runner (resumable, per-file timing, atomic RTTM writes)
  compat.py              compatibility shims (imported before pyannote.audio)
  pyproject.toml         direct dependency pins
  requirements.lock.txt  full `uv pip freeze` of the venv used for the paper run
```

## Setup (Windows, Git Bash; venv lives outside OneDrive)

Prerequisites: `hf auth login` done (token in `~/.cache/huggingface/token`), and
the user conditions of the gated repos `pyannote/speaker-diarization-3.1` and
`pyannote/segmentation-3.0` accepted with that account. The token is never read
or printed by the scripts; `huggingface_hub` picks it up from the cache.

```bash
uv venv --python 3.12 "C:/Users/manan/.venvs/selcollar_pyannote"
PY="C:/Users/manan/.venvs/selcollar_pyannote/Scripts/python.exe"
uv pip install --python "$PY" torch==2.11.0 torchaudio==2.11.0 --index-url https://download.pytorch.org/whl/cu128
uv pip install --python "$PY" --extra-index-url https://download.pytorch.org/whl/cu128 --index-strategy unsafe-best-match \
    "torch==2.11.0+cu128" "torchaudio==2.11.0+cu128" \
    "pyannote.audio==3.3.2" "pyannote.core==5.0.0" "pyannote.database==5.1.3" \
    "pyannote.pipeline==3.0.1" "pyannote.metrics==3.2.1" \
    "huggingface_hub<1.0" "scikit-learn<1.9" matplotlib soundfile soxr
```

These are the commands the venv was built with. Exact resolved versions are in
`requirements.lock.txt` (reinstall with `uv pip install --python "$PY" -r
requirements.lock.txt --extra-index-url https://download.pytorch.org/whl/cu128
--index-strategy unsafe-best-match`). `pyproject.toml` pins the same direct
dependencies, but the `uv sync` route was not exercised.

Key versions: Python 3.12.13, pyannote.audio 3.3.2, torch 2.11.0+cu128,
torchaudio 2.11.0+cu128, lightning 2.6.6, speechbrain 1.1.2, numpy 2.5.4,
huggingface_hub 0.36.2, pyannote.core 5.0.0, scikit-learn 1.8.0.

## Run

```bash
cd C:/Users/manan/OneDrive/Desktop/selective_collar
"$PY" -W ignore benchmark/pyannote/run_pyannote.py --check        # versions + hyperparameters only
"$PY" -W ignore benchmark/pyannote/run_pyannote.py msdwild voxconverse \
    > benchmark/outputs/runtime/pyannote_run.log 2>&1
```

Re-running the same command resumes (files with an existing RTTM are skipped),
which also retries earlier failures. Options: `--ids a,b`, `--limit N`,
`--force`, `--retry-failed`, `--device {cuda,cpu}`.

Outputs:

- `benchmark/outputs/rttm/pyannote/<dataset>/<file_id>.rttm`
  (`SPEAKER <file_id> 1 <start:.3f> <dur:.3f> <NA> <NA> <SPEAKER_xx> <NA> <NA>`, temp file then rename).
  The pipeline's output annotation is written as is (no merging or smoothing).
- `benchmark/outputs/runtime/pyannote_<dataset>.csv` (`file_id,audio_seconds,inference_seconds`)
- `benchmark/outputs/runtime/pyannote_empty.txt` (one file id per line; the RTTM of such a file is empty)
- `benchmark/outputs/runtime/pyannote_failed.txt` (`dataset<TAB>file_id<TAB>error`; entries are removed when a retry succeeds)
- `benchmark/outputs/runtime/pyannote_run.log` (configuration dump, per-file lines, per-dataset totals)

`inference_seconds` is the wall-clock time of the `pipeline(...)` call
(segmentation + embedding + clustering, GPU synchronised before and after). It
excludes audio decoding, model loading and one untimed 15 s warm-up call.

## Configuration actually loaded (checked at every start-up, run aborts on mismatch)

- segmentation `pyannote/segmentation-3.0` (powerset, 10 s windows, step 0.1 = 1 s hop, batch 32)
- embedding `pyannote/wespeaker-voxceleb-resnet34-LM` (batch 32, `embedding_exclude_overlap: true`)
- clustering `AgglomerativeClustering`, method `centroid`, threshold `0.7045654963945799`, `min_cluster_size` 12
- segmentation `min_duration_off` 0.0
- both models on `cuda:0`

Audio: decoded with soundfile, down-mixed to mono by channel averaging, resampled
to 16 kHz with soxr HQ only if needed (both datasets already are 16 kHz mono),
passed as `{"waveform": float32 tensor (1, T), "sample_rate": 16000}`.

## Compatibility shims (`compat.py`)

pyannote.audio 3.3.2 predates torch 2.6 / torchaudio 2.9 / numpy 2. None of the
shims touches weights or hyperparameters.

1. `torchaudio.AudioMetaData`, `torchaudio.info`, `torchaudio.list_audio_backends`,
   `torchaudio.set_audio_backend`, `torchaudio.get_audio_backend` were removed in
   torchaudio 2.9; pyannote.audio 3.3.2 references them at import time. They are
   replaced by small soundfile-based stand-ins. Real decoding never goes through
   them because audio is passed in memory.
2. `torch.load` defaults to `weights_only=True` since torch 2.6; the pyannote
   checkpoints pickle extra objects, so the default is switched back to `False`
   (only official pyannote checkpoints are loaded).
3. numpy 2 removed `np.NaN`, `np.NAN`, `np.Inf`, `np.infty`; re-added as aliases for
   pyannote.core 5.0.0 / pyannote.metrics 3.2.1.
4. Version pins instead of code changes: `huggingface_hub<1.0` (3.x passes
   `use_auth_token`), pyannote.core/database/pipeline/metrics pinned to the 3.3.2
   era (the 6.x/4.x releases that uv resolves by default drop matplotlib and are
   meant for pyannote.audio 4), `scikit-learn<1.9` (Smart App Control), and
   `matplotlib` installed explicitly.
5. Environment defaults set in-process only: `HF_HUB_DISABLE_XET=1`,
   `HF_HUB_DISABLE_SYMLINKS_WARNING=1`.
