"""Run NVIDIA Nemotron-3-Diarization (Streaming Sortformer, via NeMo) over the
selective-collar benchmark datasets and write one RTTM per audio file.

    python run_nemotron.py msdwild voxconverse
    python run_nemotron.py voxconverse --ids aepyx aorju      # subset
    python run_nemotron.py msdwild --retry-failed             # only ids in nemotron_failed.txt

Resumable: a file whose RTTM already exists is skipped (use --force to redo).

Model config = the model card's offline-style setting (30.4 s input buffer):
chunk_len=340, chunk_right_context=40, fifo_len=40, spkcache_update_period=300,
spkcache_len = checkpoint default. Post-processing = NeMo defaults (no
postprocessing_yaml passed to diarize()).
"""
from __future__ import annotations

import argparse
import csv
import gc
import importlib.util
import os
import re
import shutil
import sys
import tempfile
import time
import traceback
import types
from pathlib import Path

os.environ.setdefault("HF_HUB_DISABLE_XET", "1")  # hf_xet stalls on this machine
os.environ.setdefault("HF_HUB_OFFLINE", "1")      # weights are already cached; no token configured

import numpy as np
import soundfile as sf
import soxr
import torch

MODEL_ID = os.getenv("DIAR_MODEL_ID", "nvidia/Nemotron-3-Diarization")
SAMPLE_RATE = 16000
STREAMING = dict(chunk_len=340, chunk_right_context=40, fifo_len=40, spkcache_update_period=300)

HERE = Path(__file__).resolve().parent
REPO = HERE.parent.parent
OUT = REPO / "benchmark" / "outputs"
RTTM_ROOT = OUT / "rttm" / "nemotron"
RUNTIME_DIR = OUT / "runtime"
EMPTY_LIST = RUNTIME_DIR / "nemotron_empty.txt"
FAILED_LIST = RUNTIME_DIR / "nemotron_failed.txt"

CACHE_DIR = Path(os.getenv("SELCOLLAR_CACHE", Path.home() / ".cache" / "selective_collar"))
WAV_CACHE = CACHE_DIR / "nemotron_wav16k"
NEMO_TMP = CACHE_DIR / "nemotron_tmp"

DATASETS = {
    "msdwild": dict(audio=REPO / "Dataset" / "MSDWild" / "Raw Audio Files", skip={"00017", "00018"}),
    "voxconverse": dict(audio=REPO / "Dataset" / "VoxConverse" / "Raw Audio Files", skip=set()),
}


def log(*a):
    print(time.strftime("%H:%M:%S"), *a, flush=True)


# --------------------------------------------------------------------------- NeMo Windows patches
def _tempfile_shim():
    NEMO_TMP.mkdir(parents=True, exist_ok=True)

    def _td(*args, **kwargs):
        kwargs.setdefault("dir", str(NEMO_TMP))
        kwargs["ignore_cleanup_errors"] = True
        return tempfile.TemporaryDirectory(*args, **kwargs)

    shim = types.SimpleNamespace(**{k: getattr(tempfile, k) for k in dir(tempfile) if not k.startswith("__")})
    shim.TemporaryDirectory = _td
    return shim


def apply_nemo_patches() -> None:
    """1) temp-manifest lock (WinError 32) in diarize(); 2) FlexAttention without Triton."""
    from nemo.collections.asr.parts.mixins import diarization

    if not getattr(diarization, "_selcollar_patched", False):
        diarization.tempfile = _tempfile_shim()
        diarization._selcollar_patched = True
    if importlib.util.find_spec("triton") is None:
        from torch.nn.attention.flex_attention import flex_attention
        from nemo.collections.asr.modules import transformer_encoder_utils as tu

        tu._get_flex_attention = lambda x: flex_attention


def cleanup_tmp() -> None:
    gc.collect()
    if NEMO_TMP.exists():
        for p in NEMO_TMP.iterdir():
            shutil.rmtree(p, ignore_errors=True)


# --------------------------------------------------------------------------- audio
def ensure_wav(src: Path, cache_dir: Path) -> tuple[Path, float]:
    """16 kHz mono PCM16 copy in the cache (converted once). Returns (path, seconds)."""
    cache_dir.mkdir(parents=True, exist_ok=True)
    dst = cache_dir / f"{src.stem}.wav"
    if dst.exists() and dst.stat().st_size > 44:
        info = sf.info(str(dst))
        return dst, info.frames / info.samplerate
    data, sr = sf.read(str(src), dtype="float32", always_2d=True)
    mono = data.mean(axis=1)
    if sr != SAMPLE_RATE:
        mono = soxr.resample(mono, sr, SAMPLE_RATE, quality="HQ")
    mono = np.ascontiguousarray(mono, dtype=np.float32)
    tmp = dst.with_suffix(".tmp.wav")
    sf.write(str(tmp), mono, SAMPLE_RATE, subtype="PCM_16")
    tmp.replace(dst)
    return dst, len(mono) / SAMPLE_RATE


# --------------------------------------------------------------------------- model
_SEG_RE = re.compile(r"^\s*([0-9.]+)\s+([0-9.]+)\s+(\S+)\s*$")


def parse_segment(seg) -> tuple[float, float, str]:
    if isinstance(seg, str):
        m = _SEG_RE.match(seg)
        if not m:
            raise ValueError(f"Unrecognised diarization segment: {seg!r}")
        start, end, spk = float(m.group(1)), float(m.group(2)), m.group(3)
    else:
        start, end, spk = float(seg[0]), float(seg[1]), seg[2]
    if not isinstance(spk, str):
        spk = f"speaker_{int(spk)}"
    return start, end, spk


def load_model():
    from nemo.collections.asr.models import SortformerEncLabelModel

    apply_nemo_patches()
    model = SortformerEncLabelModel.from_pretrained(MODEL_ID, map_location="cpu")
    model.eval()
    m = model.sortformer_modules
    for k, v in STREAMING.items():
        setattr(m, k, v)
    model._check_streaming_parameters()
    log("streaming params:", {k: getattr(m, k, None) for k in
                              ("chunk_len", "chunk_left_context", "chunk_right_context", "fifo_len",
                               "spkcache_len", "spkcache_update_period", "subsampling_factor")})
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA not available")
    model.cuda()
    log("model on", next(model.parameters()).device, torch.cuda.get_device_name(0))
    return model


@torch.inference_mode()
def diarize(model, wav: Path) -> list[tuple[float, float, str]]:
    try:
        out = model.diarize(audio=[str(wav)], batch_size=1, verbose=False)
    finally:
        cleanup_tmp()
    segs = out[0] if isinstance(out, (list, tuple)) else out
    parsed = [parse_segment(s) for s in segs]
    parsed = [s for s in parsed if s[1] > s[0]]
    parsed.sort(key=lambda x: (x[0], x[1]))
    return parsed


# --------------------------------------------------------------------------- outputs
def write_rttm_atomic(path: Path, file_id: str, segs) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    with open(tmp, "w", encoding="utf-8", newline="\n") as f:
        for start, end, spk in segs:
            f.write(f"SPEAKER {file_id} 1 {start:.3f} {end - start:.3f} <NA> <NA> {spk} <NA> <NA>\n")
    for attempt in range(5):  # OneDrive can briefly hold the file
        try:
            os.replace(tmp, path)
            return
        except PermissionError:
            time.sleep(0.5 * (attempt + 1))
    os.replace(tmp, path)


def _read_ids(path: Path) -> list[str]:
    if not path.exists():
        return []
    return [l.strip() for l in path.read_text(encoding="utf-8").splitlines() if l.strip()]


def _set_listed(path: Path, file_id: str, present: bool) -> None:
    ids = _read_ids(path)
    if present and file_id not in ids:
        ids.append(file_id)
    elif not present and file_id in ids:
        ids = [i for i in ids if i != file_id]
    else:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(i + "\n" for i in ids), encoding="utf-8")


def append_runtime(dataset: str, file_id: str, audio_s: float, infer_s: float) -> None:
    path = RUNTIME_DIR / f"nemotron_{dataset}.csv"
    path.parent.mkdir(parents=True, exist_ok=True)
    rows = []
    if path.exists():
        with open(path, newline="", encoding="utf-8") as f:
            rows = [r for r in csv.reader(f)][1:]
    rows = [r for r in rows if r and r[0] != file_id]  # one row per file (latest run wins)
    rows.append([file_id, f"{audio_s:.3f}", f"{infer_s:.3f}"])
    tmp = path.with_name(path.name + ".tmp")
    with open(tmp, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["file_id", "audio_seconds", "inference_seconds"])
        w.writerows(rows)
    for attempt in range(5):
        try:
            os.replace(tmp, path)
            return
        except PermissionError:
            time.sleep(0.5 * (attempt + 1))
    os.replace(tmp, path)


# --------------------------------------------------------------------------- main
def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("datasets", nargs="+", choices=sorted(DATASETS))
    ap.add_argument("--ids", nargs="*", help="only these file ids")
    ap.add_argument("--limit", type=int, help="process at most N pending files per dataset")
    ap.add_argument("--force", action="store_true", help="redo files whose RTTM exists")
    ap.add_argument("--retry-failed", action="store_true", help="only ids listed in nemotron_failed.txt")
    args = ap.parse_args()

    log("torch", torch.__version__, "cuda", torch.version.cuda)
    model = None
    warmed = False
    n_fail = 0
    for ds in args.datasets:
        cfg = DATASETS[ds]
        files = sorted(p for p in cfg["audio"].glob("*.wav") if p.stem not in cfg["skip"])
        if args.ids:
            files = [p for p in files if p.stem in set(args.ids)]
        if args.retry_failed:
            failed = set(_read_ids(FAILED_LIST))
            files = [p for p in files if p.stem in failed]
        rttm_dir = RTTM_ROOT / ds
        pending = [p for p in files if args.force or not (rttm_dir / f"{p.stem}.rttm").exists()]
        if args.limit:
            pending = pending[: args.limit]
        log(f"[{ds}] {len(files)} selected, {len(files) - len(pending)} already done, {len(pending)} to run")
        if pending and model is None:
            model = load_model()
        if pending and not warmed:
            # untimed warm-up (CUDA init / first-call overhead) so per-file timings are steady-state
            try:
                diarize(model, ensure_wav(pending[0], WAV_CACHE / ds)[0])
            except Exception as e:
                log("warm-up failed (continuing):", e)
            warmed = True
        for i, src in enumerate(pending, 1):
            fid = src.stem
            try:
                wav, dur = ensure_wav(src, WAV_CACHE / ds)
                torch.cuda.reset_peak_memory_stats()
                t0 = time.perf_counter()
                segs = diarize(model, wav)
                torch.cuda.synchronize()
                dt = time.perf_counter() - t0
                write_rttm_atomic(rttm_dir / f"{fid}.rttm", fid, segs)
                append_runtime(ds, fid, dur, dt)
                _set_listed(EMPTY_LIST, fid, present=not segs)
                _set_listed(FAILED_LIST, fid, present=False)
                log(f"[{ds}] {i}/{len(pending)} {fid} audio={dur:.1f}s infer={dt:.2f}s rtf={dt / max(dur, 1e-9):.4f} "
                    f"segs={len(segs)} spk={len({s[2] for s in segs})} "
                    f"peak={torch.cuda.max_memory_allocated() / 2**20:.0f}MB")
            except KeyboardInterrupt:
                raise
            except Exception as e:  # log and continue (e.g. CUDA OOM)
                n_fail += 1
                log(f"[{ds}] {i}/{len(pending)} {fid} FAILED: {type(e).__name__}: {e}")
                traceback.print_exc()
                sys.stderr.flush()
                _set_listed(FAILED_LIST, fid, present=True)
                gc.collect()
                torch.cuda.empty_cache()
    log(f"done, {n_fail} failure(s)")
    return 1 if n_fail else 0


if __name__ == "__main__":
    sys.exit(main())
