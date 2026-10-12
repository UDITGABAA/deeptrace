"""
Dense face extraction for FF++ TRAIN actors only (V12 data).

V11 used 10 frames per video. Here we decode N (default 32) evenly spaced frames per video
straight from the extracted .mp4 files (no full frames written to disk) and crop faces with the
IDENTICAL pipeline of scripts/crop_ffpp_faces.py.

Leakage safety: only videos whose actors are ALL in the 540 train actors of
utils.actor_splits.get_actor_disjoint_split_3way are processed. Val (60) and test (400) actors
keep their existing 10-frame crops, so V11 vs V12 val/test comparisons are exact.

Output: D:\\datasets\\ffpp_c23\\faces256_dense\\<manip>\\<video>_frame_XXXX.jpg
        manifests/ffpp_c23_faces256_dense_train_manifest.csv
"""
import sys
import json
import argparse
import logging
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader

sys.path.insert(0, str(Path(__file__).parent.parent))

from scripts.crop_ffpp_faces import DET_LONG_SIDE, MIN_PROB, JPEG_QUALITY, square_crop
from utils.actor_splits import get_actor_disjoint_split_3way, _extract_actor_ids

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("dense_faces")

VIDEO_ROOT = Path(r"D:\datasets\ffpp_c23\extracted\FaceForensics++_C23")
COHORTS = ["Deepfakes", "FaceSwap", "FaceShifter", "Face2Face", "NeuralTextures"]


class VideoFrames(Dataset):
    def __init__(self, jobs, n_frames):
        self.jobs, self.n = jobs, n_frames

    def __len__(self):
        return len(self.jobs)

    def __getitem__(self, i):
        vid, manip, path = self.jobs[i]
        cap = cv2.VideoCapture(str(path))
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        frames = []
        for idx in np.linspace(0, max(total - 1, 0), self.n).astype(int):
            cap.set(cv2.CAP_PROP_POS_FRAMES, int(idx))
            ok, bgr = cap.read()
            if ok:
                frames.append((int(idx), cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)))
        cap.release()
        return vid, manip, frames


def _identity(x):
    return x  # keep numpy frames (default_convert would turn them into tensors)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n_frames", type=int, default=32)
    ap.add_argument("--out_root", default=r"D:\datasets\ffpp_c23\faces256_dense")
    ap.add_argument("--out_manifest", default="manifests/ffpp_c23_faces256_dense_train_manifest.csv")
    ap.add_argument("--limit_videos", type=int, default=0)
    args = ap.parse_args()

    sp = get_actor_disjoint_split_3way(pd.read_csv("manifests/ffpp_c23_faces256_manifest.csv"), "filepath")
    train_actors = sp["train_actors"]
    jobs = [(v, "real", VIDEO_ROOT / "real" / f"{v}.mp4") for v in sorted(sp["train_reals"]["video_id"].unique())]
    for m in COHORTS:
        jobs += [(v, m, VIDEO_ROOT / "fake" / m / f"{v}.mp4") for v in sorted(sp["train_fakes"][m]["video_id"].unique())]
    for v, m, p in jobs:  # hard safety check
        a, b = _extract_actor_ids(v)
        assert a in train_actors and b in train_actors, f"non-train actor video {v}"
        assert p.exists(), f"missing video {p}"
    if args.limit_videos:
        jobs = jobs[: args.limit_videos]
    out_root = Path(args.out_root)
    done = {f.name.split("_frame_")[0] + "|" + f.parent.name for f in out_root.glob("*/*.jpg")} if out_root.exists() else set()
    n_before = len(jobs)
    jobs = [j for j in jobs if f"{j[0]}|{j[1]}" not in done]  # resumable: skip videos already cropped
    logger.info(f"Resuming: {n_before - len(jobs)} videos already done, {len(jobs)} to go")
    logger.info(f"{len(jobs)} train-actor videos x {args.n_frames} frames")

    from facenet_pytorch import MTCNN
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    mtcnn = MTCNN(keep_all=True, device=device, post_process=False)
    rows, stats = [], {}
    loader = DataLoader(VideoFrames(jobs, args.n_frames), batch_size=None, num_workers=2, prefetch_factor=1, collate_fn=_identity)  # full-res frames: keep few in flight
    for k, (vid, manip, frames) in enumerate(loader):
        st = stats.setdefault(manip, {"frames": 0, "kept": 0})
        if not frames:
            continue
        rgb0 = frames[0][1]
        h, w = rgb0.shape[:2]
        s = DET_LONG_SIDE / max(h, w)
        small = np.stack([cv2.resize(f, (round(w * s), round(h * s)), interpolation=cv2.INTER_AREA) for _, f in frames])
        boxes, probs = mtcnn.detect(small)
        for (idx, rgb), bx, pr in zip(frames, boxes, probs):
            st["frames"] += 1
            if bx is None or pr is None or np.nanmax(pr) < MIN_PROB:
                continue
            j = int(np.nanargmax(pr))
            face = square_crop(rgb, bx[j] / s)
            dst = out_root / manip / f"{vid}_frame_{idx:04d}.jpg"
            dst.parent.mkdir(parents=True, exist_ok=True)
            cv2.imwrite(str(dst), cv2.cvtColor(face, cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, JPEG_QUALITY])
            st["kept"] += 1
            rows.append({"filepath": str(dst), "label": 0 if manip == "real" else 1,
                         "manipulation_type": manip, "video_id": vid, "det_prob": round(float(pr[j]), 4)})
        if k % 100 == 0:
            logger.info(f"{k}/{len(jobs)} videos, {len(rows)} crops")

    # manifest from DISK (includes videos finished by earlier, interrupted runs)
    det = {r["filepath"]: r["det_prob"] for r in rows}
    files = sorted(out_root.glob("*/*.jpg"))
    pd.DataFrame({
        "filepath": [str(f) for f in files],
        "label": [0 if f.parent.name == "real" else 1 for f in files],
        "manipulation_type": [f.parent.name for f in files],
        "video_id": [f.name.split("_frame_")[0] for f in files],
        "det_prob": [det.get(str(f)) for f in files],
    }).to_csv(args.out_manifest, index=False)
    Path(args.out_manifest).with_suffix(".summary.json").write_text(json.dumps(
        {"n_frames_per_video": args.n_frames, "n_videos": len(jobs), "per_class": stats}, indent=2))
    logger.info(f"Wrote {len(rows)} crops -> {args.out_manifest}; {json.dumps(stats)}")


if __name__ == "__main__":
    main()
