#!/usr/bin/env python
"""VBx diarization runner for the selective-collar benchmark.

Pipeline per file (same logic / hyper-parameters as the original experiment
scripts in VBX/, see README.md):

  audio -> 16 kHz mono 16-bit wav (cache)
        -> BUT energy VAD (threshold 0.5, median window 50)      -> .lab
        -> ResNet101 16 kHz x-vectors (1.44 s window, 0.24 s hop) -> .ark/.seg
        -> AHC + VB-HMM (VBx)                                     -> .rttm

Usage:
    python run_vbx.py msdwild voxconverse
    python run_vbx.py voxconverse --ids aepyx,aggyz      # subset
Resumable: files whose output RTTM already exists are skipped.
"""
import os

# Keep the CPU worker processes single-threaded (they are parallelised by file).
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import argparse
import csv
import shutil
import sys
import time
import traceback
import warnings
from multiprocessing import Pool
from pathlib import Path

import numpy as np

warnings.filterwarnings("ignore", category=SyntaxWarning)

HERE = Path(__file__).resolve().parent
REPO = HERE.parent.parent
sys.path.insert(0, str(HERE))

MODEL_DIR = HERE / "VBx" / "models" / "ResNet101_16kHz"
OUT_ROOT = REPO / "benchmark" / "outputs"
RTTM_ROOT = OUT_ROOT / "rttm" / "vbx"
RUNTIME_DIR = OUT_ROOT / "runtime"
CACHE_ROOT = Path.home() / ".cache" / "selective_collar"
WAV_CACHE = CACHE_ROOT / "vbx_wav16k"
WORK_CACHE = CACHE_ROOT / "vbx_work"

DATASETS = {
    "msdwild": {
        "audio": REPO / "Dataset" / "MSDWild" / "Raw Audio Files",
        "skip": {"00017", "00018"},  # excluded from the study
    },
    "voxconverse": {
        "audio": REPO / "Dataset" / "VoxConverse" / "Raw Audio Files",
        "skip": set(),
    },
}

# ---- configuration (identical to the original VBX/ experiment scripts) ------
VAD_THRESHOLD = 0.5
VAD_MEDIAN_WINDOW = 50
SEG_LEN = 144   # frames (1.44 s)
SEG_JUMP = 24   # frames (0.24 s)
VBHMM_ARGS = ["--init", "AHC+VB", "--threshold", "-0.015", "--lda-dim", "128",
              "--Fa", "0.3", "--Fb", "17", "--loopP", "0.99"]


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def atomic_write_text(path, text):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + f".tmp{os.getpid()}")
    with open(tmp, "w", newline="\n") as f:
        f.write(text)
    os.replace(tmp, path)


# ----------------------------------------------------------------------------
# Stage 1 (CPU worker): audio conversion + energy VAD
# ----------------------------------------------------------------------------
def to_wav16k(src, dst):
    """Write `src` as 16 kHz mono PCM16 wav at `dst` (atomic). Returns seconds."""
    import soundfile as sf
    info = sf.info(str(src))
    if dst.exists():
        try:
            return sf.info(str(dst)).duration
        except Exception:
            dst.unlink()
    dst.parent.mkdir(parents=True, exist_ok=True)
    tmp = dst.with_name(dst.name + f".tmp{os.getpid()}")
    if info.samplerate == 16000 and info.channels == 1 and info.subtype == "PCM_16" and info.format == "WAV":
        shutil.copyfile(src, tmp)  # already in target format
    else:
        import soxr
        x, sr = sf.read(str(src), dtype="float32", always_2d=True)
        x = x.mean(axis=1)
        if sr != 16000:
            x = soxr.resample(x, sr, 16000, quality="HQ")
        sf.write(str(tmp), np.clip(x, -1.0, 1.0), 16000, subtype="PCM_16", format="WAV")
    os.replace(tmp, dst)
    return sf.info(str(dst)).duration


def prep_job(task):
    dataset, fid, src = task
    try:
        from energy_vad.energy_VAD import vad_file
        wav = WAV_CACHE / dataset / f"{fid}.wav"
        lab = WORK_CACHE / dataset / "lab" / f"{fid}.lab"
        lab.parent.mkdir(parents=True, exist_ok=True)
        dur = to_wav16k(Path(src), wav)
        t0 = time.perf_counter()
        labels = vad_file(str(wav), str(lab), threshold=VAD_THRESHOLD,
                          median_window_length=VAD_MEDIAN_WINDOW)
        return dict(fid=fid, ok=True, audio_seconds=dur, vad_s=time.perf_counter() - t0,
                    n_lab=len(labels))
    except Exception:
        return dict(fid=fid, ok=False, err=traceback.format_exc())


# ----------------------------------------------------------------------------
# Stage 2 (main process, GPU): x-vector extraction
# Feature extraction and segmentation are a verbatim port of VBx/predict.py;
# the only change is that fixed-length windows are embedded in batches.
# ----------------------------------------------------------------------------
class XvectorExtractor:
    def __init__(self, backend="pytorch", device="cuda", batch_size=32):
        self.backend = backend
        self.batch_size = batch_size
        if backend == "pytorch":
            import torch
            from VBx.models.resnet import ResNet101
            self.torch = torch
            if device == "cuda" and not torch.cuda.is_available():
                raise RuntimeError("CUDA requested but torch.cuda.is_available() is False")
            self.device = torch.device(device)
            model = ResNet101(feat_dim=64, embed_dim=256).to(self.device)
            ckpt = torch.load(str(MODEL_DIR / "nnet" / "raw_81.pth"), map_location=self.device)
            model.load_state_dict(ckpt["state_dict"], strict=False)
            self.model = model.eval()
            self.desc = f"pytorch raw_81.pth on {self.device}" + (
                f" ({torch.cuda.get_device_name(0)})" if device == "cuda" else "")
        else:  # onnx (batch dimension of final.onnx is fixed to 1)
            import onnxruntime as ort
            so = ort.SessionOptions()
            so.log_severity_level = 3
            providers = (["CUDAExecutionProvider"] if device == "cuda" else []) + ["CPUExecutionProvider"]
            self.model = ort.InferenceSession(str(MODEL_DIR / "nnet" / "final.onnx"), so, providers=providers)
            self.input_name = self.model.get_inputs()[0].name
            self.label_name = self.model.get_outputs()[0].name
            self.batch_size = 1
            self.desc = f"onnx final.onnx providers={self.model.get_providers()}"

    def embed(self, chunks):
        """chunks: list of (T, 64) float32 arrays -> list of 256-d vectors."""
        out = [None] * len(chunks)
        if self.backend == "onnx":
            for i, d in enumerate(chunks):
                out[i] = self.model.run([self.label_name], {
                    self.input_name: d.astype(np.float32).transpose()[np.newaxis, :, :]})[0].squeeze()
            return out
        torch = self.torch
        by_len = {}
        for i, d in enumerate(chunks):
            by_len.setdefault(len(d), []).append(i)
        with torch.no_grad():
            for _, idx in by_len.items():
                for b in range(0, len(idx), self.batch_size):
                    sel = idx[b:b + self.batch_size]
                    data = torch.from_numpy(np.stack([chunks[i] for i in sel])).to(self.device)
                    data = torch.transpose(data, 1, 2).contiguous()
                    emb = self.model(data).float().cpu().numpy()
                    for j, i in enumerate(sel):
                        out[i] = emb[j]
        return out

    def extract_file(self, fn, wav_path, lab_path, ark_path, seg_path):
        """Port of the per-file loop of VBx/predict.py. Returns number of x-vectors."""
        import kaldi_io
        import soundfile as sf
        from VBx import features

        signal, samplerate = sf.read(str(wav_path))
        assert samplerate == 16000, samplerate
        labs = []
        with open(lab_path) as f:
            for line in f:
                p = line.split()
                if len(p) >= 2:
                    labs.append((float(p[0]), float(p[1])))
        labs = (np.array(labs, dtype=float).reshape(-1, 2) * samplerate).astype(int)

        noverlap = 240
        winlen = 400
        window = features.povey_window(winlen)
        fbank_mx = features.mel_fbank_mx(winlen, samplerate, NUMCHANS=64, LOFREQ=20.0, HIFREQ=7600, htk_bug=False)
        LC = 150
        RC = 149
        seg_len, seg_jump = SEG_LEN, SEG_JUMP

        np.random.seed(3)  # for reproducibility
        signal = features.add_dither((signal * 2 ** 15).astype(int))

        items = []  # (key, seg_start, seg_end, data)
        for segnum in range(len(labs)):
            seg = signal[labs[segnum, 0]:labs[segnum, 1]]
            if seg.shape[0] > 0.01 * samplerate:  # process segment only if longer than 0.01s
                # Mirror noverlap//2 initial and final samples
                seg = np.r_[seg[noverlap // 2 - 1::-1], seg, seg[-1:-winlen // 2 - 1:-1]]
                fea = features.fbank_htk(seg, window, noverlap, fbank_mx, USEPOWER=True, ZMEANSOURCE=True)
                fea = features.cmvn_floating_kaldi(fea, LC, RC, norm_vars=False).astype(np.float32)

                slen = len(fea)
                start = -seg_jump

                for start in range(0, slen - seg_len, seg_jump):
                    data = fea[start:start + seg_len]
                    key = f'{fn}_{segnum:04}-{start:08}-{(start + seg_len):08}'
                    seg_start = round(labs[segnum, 0] / float(samplerate) + start / 100.0, 3)
                    seg_end = round(labs[segnum, 0] / float(samplerate) + start / 100.0 + seg_len / 100.0, 3)
                    items.append((key, seg_start, seg_end, data))

                if slen - start - seg_jump >= 10:
                    data = fea[start + seg_jump:slen]
                    key = f'{fn}_{segnum:04}-{(start + seg_jump):08}-{slen:08}'
                    seg_start = round(labs[segnum, 0] / float(samplerate) + (start + seg_jump) / 100.0, 3)
                    seg_end = round(labs[segnum, 1] / float(samplerate), 3)
                    items.append((key, seg_start, seg_end, data))

        xvecs = self.embed([it[3] for it in items])
        n = 0
        tmp_ark, tmp_seg = str(ark_path) + ".tmp", str(seg_path) + ".tmp"
        with open(tmp_seg, 'w', newline="\n") as seg_file, open(tmp_ark, 'wb') as ark_file:
            for (key, seg_start, seg_end, _), xvector in zip(items, xvecs):
                if np.isnan(xvector).any():
                    log(f'NaN found, not processing: {key}')
                else:
                    seg_file.write(f'{key} {fn} {seg_start} {seg_end}\n')
                    kaldi_io.write_vec_flt(ark_file, xvector.astype(np.float32), key=key)
                    n += 1
        os.replace(tmp_ark, ark_path)
        os.replace(tmp_seg, seg_path)
        return n


# ----------------------------------------------------------------------------
# Stage 3 (CPU worker): AHC + VB-HMM, RTTM formatting
# ----------------------------------------------------------------------------
def format_rttm(raw_path, fid):
    lines, speakers = [], set()
    with open(raw_path) as f:
        for line in f:
            p = line.split()
            if len(p) < 8 or p[0] != "SPEAKER":
                continue
            start, dur, spk = float(p[3]), float(p[4]), p[7]
            if dur <= 0:
                continue
            speakers.add(spk)
            lines.append(f"SPEAKER {fid} 1 {start:.3f} {dur:.3f} <NA> <NA> {spk} <NA> <NA>\n")
    return "".join(lines), len(lines), len(speakers)


def vb_job(task):
    dataset, fid, ark, seg, raw_dir, out_rttm = task
    try:
        import contextlib
        import io
        from VBx import vbhmm
        t0 = time.perf_counter()
        with contextlib.redirect_stdout(io.StringIO()):
            vbhmm.main(VBHMM_ARGS + [
                "--out-rttm-dir", raw_dir, "--xvec-ark-file", ark, "--segments-file", seg,
                "--xvec-transform", str(MODEL_DIR / "transform.h5"),
                "--plda-file", str(MODEL_DIR / "plda")])
        text, n_lines, n_spk = format_rttm(os.path.join(raw_dir, f"{fid}.rttm"), fid)
        atomic_write_text(out_rttm, text)
        return dict(fid=fid, ok=True, vb_s=time.perf_counter() - t0, n_lines=n_lines, n_spk=n_spk)
    except Exception:
        return dict(fid=fid, ok=False, err=traceback.format_exc())


# ----------------------------------------------------------------------------
# Book-keeping
# ----------------------------------------------------------------------------
def append_csv(path, header, row):
    new = not path.exists()
    with open(path, "a", newline="") as f:
        w = csv.writer(f)
        if new:
            w.writerow(header)
        w.writerow(row)


def dedupe_csv(path):
    """Keep the last row per file_id (re-runs append)."""
    if not path.exists():
        return
    with open(path, newline="") as f:
        rows = list(csv.reader(f))
    if not rows:
        return
    header, last = rows[0], {}
    for r in rows[1:]:
        if r:
            last[r[0]] = r
    tmp = path.with_name(path.name + ".tmp")
    with open(tmp, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(header)
        w.writerows(last[k] for k in sorted(last))
    os.replace(tmp, path)


def add_to_list(path, item):
    items = set(path.read_text().split()) if path.exists() else set()
    items.add(item)
    atomic_write_text(path, "".join(f"{i}\n" for i in sorted(items)))


def run_dataset(name, args, extractor):
    cfg = DATASETS[name]
    out_dir = RTTM_ROOT / name
    out_dir.mkdir(parents=True, exist_ok=True)
    RUNTIME_DIR.mkdir(parents=True, exist_ok=True)
    work = WORK_CACHE / name
    for sub in ("lab", "exp", "rttm_raw"):
        (work / sub).mkdir(parents=True, exist_ok=True)
    csv_path = RUNTIME_DIR / f"vbx_{name}.csv"
    stage_csv = RUNTIME_DIR / f"vbx_{name}_stages.csv"
    empty_path = RUNTIME_DIR / "vbx_empty.txt"
    failed_path = RUNTIME_DIR / f"vbx_{name}_failed.txt"

    files = sorted(cfg["audio"].glob("*.wav"))
    ids = {f.stem: f for f in files if f.stem not in cfg["skip"]}
    if args.ids:
        want = set(args.ids.split(","))
        ids = {k: v for k, v in ids.items() if k in want}
    todo = [(name, fid, str(p)) for fid, p in ids.items()
            if args.force or not (out_dir / f"{fid}.rttm").exists()]
    if args.limit:
        todo = todo[:args.limit]
    log(f"[{name}] {len(ids)} files selected, {len(ids) - len(todo)} already done, {len(todo)} to process")
    if not todo:
        return

    meta, failed = {}, {}
    n_done = 0
    t_start = time.perf_counter()
    audio_done = 0.0

    def finish(fid, vb_s, n_lines, n_spk):
        nonlocal n_done, audio_done
        m = meta.pop(fid)
        total = m["vad_s"] + m["xvec_s"] + vb_s
        append_csv(csv_path, ["file_id", "audio_seconds", "inference_seconds"],
                   [fid, f"{m['audio_seconds']:.3f}", f"{total:.3f}"])
        append_csv(stage_csv, ["file_id", "audio_seconds", "vad_seconds", "xvector_seconds", "vbhmm_seconds",
                               "n_xvectors", "n_segments", "n_speakers"],
                   [fid, f"{m['audio_seconds']:.3f}", f"{m['vad_s']:.3f}", f"{m['xvec_s']:.3f}", f"{vb_s:.3f}",
                    m["n_xvec"], n_lines, n_spk])
        if n_lines == 0:
            add_to_list(empty_path, fid)
        n_done += 1
        audio_done += m["audio_seconds"]
        el = time.perf_counter() - t_start
        log(f"[{name}] {n_done}/{len(todo)} {fid}: audio {m['audio_seconds']:.1f}s vad {m['vad_s']:.2f}s "
            f"xvec {m['xvec_s']:.2f}s ({m['n_xvec']} xv) vb {vb_s:.2f}s -> {n_lines} segs, {n_spk} spk | "
            f"elapsed {el / 60:.1f} min, overall RTF {el / max(audio_done, 1e-9):.4f}")

    def on_vb(res):
        if res["ok"]:
            finish(res["fid"], res["vb_s"], res["n_lines"], res["n_spk"])
        else:
            meta.pop(res["fid"], None)
            failed[res["fid"]] = "vbhmm: " + res["err"]
            log(f"[{name}] FAILED (vbhmm) {res['fid']}:\n{res['err']}")

    with Pool(args.prep_workers) as prep_pool, Pool(args.vb_workers) as vb_pool:
        pending = []
        for res in prep_pool.imap_unordered(prep_job, todo):
            fid = res["fid"]
            if not res["ok"]:
                failed[fid] = "prep/vad: " + res["err"]
                log(f"[{name}] FAILED (prep/vad) {fid}:\n{res['err']}")
                continue
            try:
                ark = work / "exp" / f"{fid}.ark"
                seg = work / "exp" / f"{fid}.seg"
                t0 = time.perf_counter()
                n_xvec = extractor.extract_file(fid, WAV_CACHE / name / f"{fid}.wav",
                                                work / "lab" / f"{fid}.lab", ark, seg)
                res["xvec_s"] = time.perf_counter() - t0
                res["n_xvec"] = n_xvec
                meta[fid] = res
                if n_xvec == 0:
                    atomic_write_text(out_dir / f"{fid}.rttm", "")
                    finish(fid, 0.0, 0, 0)
                else:
                    pending.append(vb_pool.apply_async(
                        vb_job, ((name, fid, str(ark), str(seg), str(work / "rttm_raw"),
                                  str(out_dir / f"{fid}.rttm")),)))
            except Exception:
                meta.pop(fid, None)
                failed[fid] = "xvector: " + traceback.format_exc()
                log(f"[{name}] FAILED (xvector) {fid}:\n{failed[fid]}")
            # collect finished VB jobs
            still = []
            for p in pending:
                if p.ready():
                    on_vb(p.get())
                else:
                    still.append(p)
            pending = still
        for p in pending:
            on_vb(p.get())

    dedupe_csv(csv_path)
    dedupe_csv(stage_csv)
    atomic_write_text(failed_path, "".join(f"{k}\n" for k in sorted(failed)))
    el = time.perf_counter() - t_start
    log(f"[{name}] finished: {n_done} ok, {len(failed)} failed, {audio_done / 3600:.2f} h audio "
        f"in {el / 60:.1f} min wall (RTF {el / max(audio_done, 1e-9):.4f})")
    if failed:
        log(f"[{name}] failed ids: {sorted(failed)} (re-run the same command to retry)")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("datasets", nargs="+", choices=sorted(DATASETS))
    ap.add_argument("--ids", default=None, help="comma separated subset of file ids")
    ap.add_argument("--limit", type=int, default=0, help="process at most N files per dataset")
    ap.add_argument("--force", action="store_true", help="re-process files whose RTTM exists")
    ap.add_argument("--backend", choices=["pytorch", "onnx"], default="pytorch")
    ap.add_argument("--device", choices=["cuda", "cpu"], default="cuda")
    ap.add_argument("--batch-size", type=int, default=32, help="x-vector batch size (pytorch backend)")
    ap.add_argument("--prep-workers", type=int, default=4, help="processes for audio conversion + VAD")
    ap.add_argument("--vb-workers", type=int, default=6, help="processes for AHC + VB-HMM")
    args = ap.parse_args()

    extractor = XvectorExtractor(args.backend, args.device, args.batch_size)
    log(f"x-vector extractor: {extractor.desc}, batch size {extractor.batch_size}")
    for name in args.datasets:
        run_dataset(name, args, extractor)
    if args.backend == "pytorch" and args.device == "cuda":
        log(f"peak GPU memory allocated by torch: {extractor.torch.cuda.max_memory_allocated() / 2 ** 20:.0f} MiB")


if __name__ == "__main__":
    main()
