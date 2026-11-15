"""pyannote/speaker-diarization-3.1 RTTM runner for the selective-collar benchmark.

    python run_pyannote.py msdwild voxconverse            # resumable full run
    python run_pyannote.py msdwild --ids 00017,00018      # selected files
    python run_pyannote.py voxconverse --limit 5 --force  # redo first 5
    python run_pyannote.py --check                        # print versions / hyperparameters only

The pipeline is used exactly as published (default hyperparameters, no
num_speakers / min_speakers / max_speakers). Audio is decoded with soundfile,
down-mixed to mono, resampled to 16 kHz if needed and passed in memory.
"""
from __future__ import annotations

import argparse
import csv
import os
import sys
import time
import traceback
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import compat  # noqa: E402  (must precede pyannote.audio)

import numpy as np  # noqa: E402
import soundfile as sf  # noqa: E402
import torch  # noqa: E402

REPO = HERE.parent.parent
OUT = REPO / "benchmark" / "outputs"
RTTM_DIR = OUT / "rttm" / "pyannote"
RUNTIME_DIR = OUT / "runtime"
EMPTY_LIST = RUNTIME_DIR / "pyannote_empty.txt"
FAILED_LIST = RUNTIME_DIR / "pyannote_failed.txt"

DATASETS = {
    "msdwild": REPO / "Dataset" / "MSDWild" / "Raw Audio Files",
    "voxconverse": REPO / "Dataset" / "VoxConverse" / "Raw Audio Files",
}

PIPELINE_ID = "pyannote/speaker-diarization-3.1"
SAMPLE_RATE = 16000

# values stated in the journal paper / model card, checked at start-up
EXPECTED = {
    "segmentation": "pyannote/segmentation-3.0",
    "embedding": "pyannote/wespeaker-voxceleb-resnet34-LM",
    "clustering": "AgglomerativeClustering",
    "threshold": 0.7045654963945799,
    "min_cluster_size": 12,
    "method": "centroid",
    "min_duration_off": 0.0,
}


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def load_pipeline(device: torch.device):
    import yaml
    from huggingface_hub import hf_hub_download
    from huggingface_hub.utils import GatedRepoError, HfHubHTTPError
    from pyannote.audio import Pipeline

    # explicit gated-repo probe so that a missing licence acceptance is reported by repo name
    for repo, fname in ((PIPELINE_ID, "config.yaml"), (EXPECTED["segmentation"], "config.yaml")):
        try:
            hf_hub_download(repo, fname)
        except GatedRepoError as e:
            raise SystemExit(
                f"GATED: accept the user conditions of https://hf.co/{repo} with the logged-in "
                f"Hugging Face account ({type(e).__name__})."
            )
        except HfHubHTTPError as e:
            code = getattr(getattr(e, "response", None), "status_code", None)
            raise SystemExit(f"Hugging Face download of {repo}/{fname} failed (HTTP {code}).")

    with open(hf_hub_download(PIPELINE_ID, "config.yaml"), "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    pipeline = Pipeline.from_pretrained(PIPELINE_ID)  # token picked up from ~/.cache/huggingface
    if pipeline is None:
        raise SystemExit(f"Pipeline.from_pretrained returned None: check gating of {PIPELINE_ID}.")
    pipeline.to(device)
    return pipeline, config


def describe(pipeline, config) -> dict:
    import lightning
    import pyannote.audio
    import speechbrain
    import torchaudio

    inst = pipeline.parameters(instantiated=True)
    seg_model = pipeline._segmentation.model
    info = {
        "pyannote.audio": pyannote.audio.__version__,
        "torch": torch.__version__,
        "torchaudio": torchaudio.__version__,
        "lightning": lightning.__version__,
        "speechbrain": speechbrain.__version__,
        "numpy": np.__version__,
        "cuda_available": torch.cuda.is_available(),
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "segmentation_device": str(next(seg_model.parameters()).device),
        "embedding_device": str(pipeline._embedding.device),
        "config.segmentation": config["pipeline"]["params"]["segmentation"],
        "config.embedding": config["pipeline"]["params"]["embedding"],
        "config.clustering": config["pipeline"]["params"]["clustering"],
        "config.params": config.get("params"),
        "loaded.embedding": pipeline.embedding,
        "loaded.clustering": pipeline.klustering,
        "loaded.params": inst,
        "segmentation_step": pipeline.segmentation_step,
        "segmentation_duration": pipeline._segmentation.duration,
        "segmentation_batch_size": pipeline.segmentation_batch_size,
        "embedding_batch_size": pipeline.embedding_batch_size,
        "embedding_exclude_overlap": pipeline.embedding_exclude_overlap,
        "powerset": bool(seg_model.specifications.powerset),
        "shims": compat.APPLIED,
    }
    problems = []
    if info["config.segmentation"] != EXPECTED["segmentation"]:
        problems.append("segmentation model")
    if info["loaded.embedding"] != EXPECTED["embedding"]:
        problems.append("embedding model")
    if info["loaded.clustering"] != EXPECTED["clustering"]:
        problems.append("clustering class")
    c = inst["clustering"]
    if abs(c["threshold"] - EXPECTED["threshold"]) > 1e-9:
        problems.append("clustering threshold")
    if c["min_cluster_size"] != EXPECTED["min_cluster_size"]:
        problems.append("min_cluster_size")
    if c["method"] != EXPECTED["method"]:
        problems.append("clustering method")
    if inst["segmentation"]["min_duration_off"] != EXPECTED["min_duration_off"]:
        problems.append("min_duration_off")
    info["hyperparameter_check"] = "OK" if not problems else "MISMATCH: " + ", ".join(problems)
    return info


def load_audio(path: Path) -> tuple[torch.Tensor, float]:
    data, sr = sf.read(str(path), dtype="float32", always_2d=True)  # (frames, channels)
    mono = data.mean(axis=1) if data.shape[1] > 1 else data[:, 0]
    if sr != SAMPLE_RATE:
        import soxr

        mono = soxr.resample(mono, sr, SAMPLE_RATE, quality="HQ").astype(np.float32)
    mono = np.ascontiguousarray(mono, dtype=np.float32)
    return torch.from_numpy(mono).unsqueeze(0), mono.shape[0] / SAMPLE_RATE


def rttm_lines(annotation, file_id: str) -> list[str]:
    lines = []
    for segment, _, label in annotation.itertracks(yield_label=True):
        lines.append(
            f"SPEAKER {file_id} 1 {segment.start:.3f} {segment.duration:.3f} <NA> <NA> {label} <NA> <NA>"
        )
    return lines


def atomic_write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + f".tmp{os.getpid()}")
    with open(tmp, "w", encoding="utf-8", newline="\n") as f:
        f.write(text)
        f.flush()
        os.fsync(f.fileno())
    for attempt in range(5):  # OneDrive may briefly lock the target
        try:
            os.replace(tmp, path)
            return
        except PermissionError:
            if attempt == 4:
                raise
            time.sleep(0.5 * (attempt + 1))


def append_runtime(dataset: str, file_id: str, audio_s: float, infer_s: float) -> None:
    path = RUNTIME_DIR / f"pyannote_{dataset}.csv"
    new = not path.exists()
    with open(path, "a", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        if new:
            w.writerow(["file_id", "audio_seconds", "inference_seconds"])
        w.writerow([file_id, f"{audio_s:.3f}", f"{infer_s:.3f}"])


def dedupe_runtime(dataset: str) -> None:
    """Keep the last row per file_id (a file may be re-run with --force)."""
    path = RUNTIME_DIR / f"pyannote_{dataset}.csv"
    if not path.exists():
        return
    with open(path, "r", encoding="utf-8", newline="") as f:
        rows = list(csv.reader(f))
    if not rows:
        return
    last = {r[0]: r for r in rows[1:] if r}
    if len(last) == len(rows) - 1:
        return
    text = ",".join(rows[0]) + "\n" + "".join(",".join(r) + "\n" for r in last.values())
    atomic_write(path, text)


def read_empty() -> list[str]:
    if not EMPTY_LIST.exists():
        return []
    return [l.strip() for l in EMPTY_LIST.read_text(encoding="utf-8").splitlines() if l.strip()]


def read_failed() -> dict[tuple[str, str], str]:
    out = {}
    if FAILED_LIST.exists():
        for l in FAILED_LIST.read_text(encoding="utf-8").splitlines():
            p = l.split("\t", 2)
            if len(p) >= 2:
                out[(p[0], p[1])] = p[2] if len(p) > 2 else ""
    return out


def write_failed(failed: dict[tuple[str, str], str]) -> None:
    text = "".join(f"{ds}\t{fid}\t{err}\n" for (ds, fid), err in sorted(failed.items()))
    atomic_write(FAILED_LIST, text)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("datasets", nargs="*", choices=[*DATASETS, []], help="msdwild and/or voxconverse")
    ap.add_argument("--ids", help="comma-separated file ids (default: all)")
    ap.add_argument("--limit", type=int, help="only the first N files per dataset")
    ap.add_argument("--force", action="store_true", help="re-run files whose RTTM already exists")
    ap.add_argument("--retry-failed", action="store_true", help="only ids listed in pyannote_failed.txt")
    ap.add_argument("--device", default="cuda", choices=["cuda", "cpu"])
    ap.add_argument("--check", action="store_true", help="load the pipeline, print configuration and exit")
    args = ap.parse_args()
    if not args.datasets and not args.check:
        ap.error("give at least one dataset (or --check)")

    if args.device == "cuda" and not torch.cuda.is_available():
        raise SystemExit("CUDA requested but torch.cuda.is_available() is False")
    device = torch.device(args.device)

    t0 = time.perf_counter()
    pipeline, config = load_pipeline(device)
    info = describe(pipeline, config)
    log(f"pipeline {PIPELINE_ID} loaded on {device} in {time.perf_counter() - t0:.1f}s")
    for k, v in info.items():
        log(f"  {k}: {v}")
    if info["hyperparameter_check"] != "OK":
        raise SystemExit("loaded pipeline does not match the expected configuration: " + info["hyperparameter_check"])
    if args.check:
        return 0

    # untimed warm-up (cuDNN / CUDA initialisation) so the first file's timing is not inflated
    g = torch.Generator().manual_seed(0)
    with torch.inference_mode():
        pipeline({"waveform": 0.01 * torch.randn(1, 15 * SAMPLE_RATE, generator=g), "sample_rate": SAMPLE_RATE})

    RUNTIME_DIR.mkdir(parents=True, exist_ok=True)
    failed = read_failed()
    empty = read_empty()
    total_fail = 0

    for ds in args.datasets:
        wavs = sorted(DATASETS[ds].glob("*.wav"))
        if args.ids:
            wanted = {s.strip() for s in args.ids.split(",") if s.strip()}
            wavs = [w for w in wavs if w.stem in wanted]
            missing = wanted - {w.stem for w in wavs}
            if missing:
                log(f"[{ds}] ids not found: {sorted(missing)}")
        if args.retry_failed:
            wavs = [w for w in wavs if (ds, w.stem) in failed]
        if args.limit:
            wavs = wavs[: args.limit]
        out_dir = RTTM_DIR / ds
        out_dir.mkdir(parents=True, exist_ok=True)
        todo = [w for w in wavs if args.force or not (out_dir / f"{w.stem}.rttm").exists()]
        log(f"[{ds}] {len(wavs)} selected, {len(wavs) - len(todo)} already done, {len(todo)} to run")

        ds_audio = ds_infer = 0.0
        n_ok = n_fail = 0
        ds_start = time.perf_counter()
        for i, wav in enumerate(todo, 1):
            fid = wav.stem
            try:
                waveform, audio_s = load_audio(wav)
                if device.type == "cuda":
                    torch.cuda.synchronize()
                t = time.perf_counter()
                with torch.inference_mode():
                    annotation = pipeline({"waveform": waveform, "sample_rate": SAMPLE_RATE, "uri": fid})
                if device.type == "cuda":
                    torch.cuda.synchronize()
                infer_s = time.perf_counter() - t

                lines = rttm_lines(annotation, fid)
                atomic_write(out_dir / f"{fid}.rttm", "".join(l + "\n" for l in lines))
                append_runtime(ds, fid, audio_s, infer_s)
                if not lines:
                    if fid not in empty:
                        empty.append(fid)
                        atomic_write(EMPTY_LIST, "".join(e + "\n" for e in empty))
                elif fid in empty:
                    empty.remove(fid)
                    atomic_write(EMPTY_LIST, "".join(e + "\n" for e in empty))
                if (ds, fid) in failed:
                    del failed[(ds, fid)]
                    write_failed(failed)
                ds_audio += audio_s
                ds_infer += infer_s
                n_ok += 1
                log(
                    f"[{ds}] {i}/{len(todo)} {fid} audio={audio_s:.1f}s infer={infer_s:.2f}s "
                    f"rtf={infer_s / max(audio_s, 1e-9):.4f} speakers={len(annotation.labels())} "
                    f"segments={len(lines)}{' EMPTY' if not lines else ''}"
                )
            except KeyboardInterrupt:
                raise
            except Exception as e:  # keep going, record the failure
                n_fail += 1
                err = f"{type(e).__name__}: {e}".replace("\t", " ").replace("\n", " ")[:500]
                failed[(ds, fid)] = err
                write_failed(failed)
                log(f"[{ds}] {i}/{len(todo)} {fid} FAILED {err}")
                traceback.print_exc()
                if device.type == "cuda":
                    torch.cuda.empty_cache()
        dedupe_runtime(ds)
        total_fail += n_fail
        log(
            f"[{ds}] done: ok={n_ok} failed={n_fail} audio={ds_audio / 3600:.3f}h "
            f"inference={ds_infer:.1f}s rtf={ds_infer / max(ds_audio, 1e-9):.5f} "
            f"wall={time.perf_counter() - ds_start:.1f}s"
        )
    if device.type == "cuda":
        log(f"peak GPU memory allocated: {torch.cuda.max_memory_allocated() / 2**20:.0f} MiB")
    return 1 if total_fail else 0


if __name__ == "__main__":
    sys.exit(main())
