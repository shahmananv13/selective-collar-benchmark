"""Compatibility shims so pyannote.audio 3.3.2 runs on torch/torchaudio 2.11 (cu128), numpy 2.x.

Import this module BEFORE importing pyannote.audio. Nothing here changes the
model, its weights or any pipeline hyperparameter; it only restores APIs that
newer torch / torchaudio / numpy removed or changed.
"""
from __future__ import annotations

import os
import warnings
from dataclasses import dataclass

os.environ.setdefault("HF_HUB_DISABLE_XET", "1")
os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")

import numpy as np
import torch
import torchaudio

APPLIED: list[str] = []

# --- numpy 2 removed aliases (used by older pyannote.* / speechbrain code) ----
for _alias, _target in (("NaN", "nan"), ("NAN", "nan"), ("Inf", "inf"), ("infty", "inf")):
    if not hasattr(np, _alias):
        setattr(np, _alias, getattr(np, _target))
        APPLIED.append(f"numpy.{_alias} alias")

# --- torchaudio >= 2.9 removed the backend / metadata API -------------------
# pyannote.audio 3.3.2 references these at import time (io.py type hints and
# backend selection). Audio is decoded with soundfile in run_pyannote.py and
# passed in memory, so these are never used for real decoding of our files.
if not hasattr(torchaudio, "AudioMetaData"):

    @dataclass
    class AudioMetaData:  # minimal stand-in for the removed class
        sample_rate: int
        num_frames: int
        num_channels: int
        bits_per_sample: int = 0
        encoding: str = "UNKNOWN"

    torchaudio.AudioMetaData = AudioMetaData
    APPLIED.append("torchaudio.AudioMetaData stand-in")

if not hasattr(torchaudio, "list_audio_backends"):
    torchaudio.list_audio_backends = lambda: ["soundfile"]
    APPLIED.append("torchaudio.list_audio_backends stub")

if not hasattr(torchaudio, "set_audio_backend"):
    torchaudio.set_audio_backend = lambda backend=None: None
    APPLIED.append("torchaudio.set_audio_backend stub")

if not hasattr(torchaudio, "get_audio_backend"):
    torchaudio.get_audio_backend = lambda: "soundfile"
    APPLIED.append("torchaudio.get_audio_backend stub")

if not hasattr(torchaudio, "info"):

    def _info(path, backend=None, **kwargs):
        import soundfile as sf

        i = sf.info(str(path))
        return torchaudio.AudioMetaData(
            sample_rate=i.samplerate, num_frames=i.frames, num_channels=i.channels
        )

    torchaudio.info = _info
    APPLIED.append("torchaudio.info via soundfile")

# --- torch >= 2.6: torch.load defaults to weights_only=True ------------------
# The pyannote checkpoints (lightning .ckpt / pytorch_model.bin from the
# official pyannote repos) pickle non-tensor objects; load them the pre-2.6 way.
_orig_torch_load = torch.load


def _torch_load(*args, **kwargs):
    if kwargs.get("weights_only") is None:
        kwargs["weights_only"] = False
    return _orig_torch_load(*args, **kwargs)


torch.load = _torch_load
APPLIED.append("torch.load weights_only=False default")

warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)
