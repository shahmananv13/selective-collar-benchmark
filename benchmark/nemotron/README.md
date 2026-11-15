# Nemotron-3-Diarization runner (selective-collar benchmark)

Runs `nvidia/Nemotron-3-Diarization` (Streaming Sortformer, max 8 speakers) through
NeMo `SortformerEncLabelModel` and writes one RTTM per audio file.

## Setup (Git Bash, Windows, RTX 50xx)

```bash
cd "/c/Users/manan/OneDrive/Desktop/selective_collar/benchmark/nemotron"
export UV_PROJECT_ENVIRONMENT='C:\Users\manan\.venvs\selcollar_nemotron'   # venv outside OneDrive
uv sync --python 3.12
uv run --no-sync python -c "import torch, nemo; print(torch.__version__, torch.cuda.is_available(), nemo.__version__)"
```

- `vendor/wheels/nemo_toolkit-3.1.0+1688cc3d6-py3-none-any.whl` is NeMo git main @1688cc3d6
  (PyPI 3.0.0 cannot load this model). torch/torchaudio come from the cu128 index.
- Model weights are read from the local HF cache (`HF_HUB_OFFLINE=1` is set by the script;
  export `HF_HUB_OFFLINE=0` to download on a fresh machine).

## Run

```bash
export UV_PROJECT_ENVIRONMENT='C:\Users\manan\.venvs\selcollar_nemotron'
uv run --no-sync python run_nemotron.py voxconverse msdwild          # resumable: existing RTTMs are skipped
uv run --no-sync python run_nemotron.py voxconverse --ids aepyx aorju
uv run --no-sync python run_nemotron.py msdwild voxconverse --retry-failed
```

Options: `--force` (redo existing), `--limit N`, `--ids ...`, `--retry-failed`.

## Inputs / outputs

- Audio: `Dataset/MSDWild/Raw Audio Files/*.wav` (ids 00017, 00018 skipped) and
  `Dataset/VoxConverse/Raw Audio Files/*.wav`.
- 16 kHz mono PCM16 cache: `~/.cache/selective_collar/nemotron_wav16k/<dataset>/`.
- RTTM: `benchmark/outputs/rttm/nemotron/<dataset>/<file_id>.rttm`
  (`SPEAKER <file_id> 1 <start> <dur> <NA> <NA> speaker_<k> <NA> <NA>`, written atomically).
- Runtime: `benchmark/outputs/runtime/nemotron_<dataset>.csv` (`file_id,audio_seconds,inference_seconds`;
  inference = wall clock of `model.diarize()` for that file, model already on GPU and warmed up,
  audio conversion excluded).
- `benchmark/outputs/runtime/nemotron_empty.txt` (files with no segments -> empty RTTM),
  `nemotron_failed.txt` (crashed files, no RTTM), `nemotron_run.log`.

## Model configuration

- Offline-style streaming config from the model card: `chunk_len=340`, `chunk_right_context=40`,
  `fifo_len=40`, `spkcache_update_period=300`; `spkcache_len=264`, `chunk_left_context=0` (checkpoint defaults);
  80 ms frames (subsampling 8).
- `batch_size=1`, no `postprocessing_yaml` -> NeMo default post-processing
  (onset 0.5, offset 0.5, pad_onset/pad_offset 0, min_duration_on/off 0).
- Windows workarounds (inside `run_nemotron.py`): temp-manifest lock shim, eager FlexAttention (no Triton).
