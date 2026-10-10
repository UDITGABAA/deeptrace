"""
Face-crop FF++ c23 frames so FF++ matches the Kaggle input domain (tight 256x256 face crops).

Why: the raw FF++ manifest stores FULL video frames (640x480 .. 1920x1080). Resizing those to
160x160 shrinks the face to a few dozen pixels, destroys blending artefacts, and makes the
source dataset trivially identifiable (see audit_v10_claude/AUDIT_REPORT.md section 3.3).

Every frame (real or fake) goes through the IDENTICAL pipeline:
  MTCNN on a 640px-long-side copy -> highest-confidence face -> square box (scale 1.3x)
  -> crop from the full-res frame (edge-replicate padding if out of bounds)
  -> resize 256x256 (INTER_AREA) -> JPEG quality 75 (same quant table as the source frames).
Frames with no face (prob < 0.90) are dropped and counted per class in the summary.

Output:
  D:\\datasets\\ffpp_c23\\faces256\\<manipulation>\\<frame>.jpg
  manifests/ffpp_c23_faces256_manifest.csv  (filepath,label,manipulation_type,src_filepath,det_prob,face_px)

Usage:
  venv\\Scripts\\python.exe scripts\\crop_ffpp_faces.py [--include_dfd] [--limit N]
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

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("crop_ffpp_faces")

DET_LONG_SIDE = 640
BOX_SCALE = 1.3
OUT_SIZE = 256
JPEG_QUALITY = 75
MIN_PROB = 0.90


class FrameDataset(Dataset):
    def __init__(self, paths):
        self.paths = paths

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, i):
        bgr = cv2.imread(self.paths[i], cv2.IMREAD_COLOR)
        if bgr is None:
            return i, None, None, 1.0
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        h, w = rgb.shape[:2]
        s = DET_LONG_SIDE / max(h, w)
        small = cv2.resize(rgb, (round(w * s), round(h * s)), interpolation=cv2.INTER_AREA)
        return i, rgb, small, s


def collate(batch):
    return batch


def square_crop(rgb, box):
    x1, y1, x2, y2 = box
    cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
    side = max(x2 - x1, y2 - y1) * BOX_SCALE
    half = side / 2
    l, t, r, b = int(round(cx - half)), int(round(cy - half)), int(round(cx + half)), int(round(cy + half))
    h, w = rgb.shape[:2]
    pad = max(0, -l, -t, r - w, b - h)
    if pad:
        rgb = cv2.copyMakeBorder(rgb, pad, pad, pad, pad, cv2.BORDER_REPLICATE)
        l, t, r, b = l + pad, t + pad, r + pad, b + pad
    crop = rgb[t:b, l:r]
    return cv2.resize(crop, (OUT_SIZE, OUT_SIZE), interpolation=cv2.INTER_AREA)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", default="manifests/ffpp_c23_manifest.csv")
    ap.add_argument("--out_root", default=r"D:\datasets\ffpp_c23\faces256")
    ap.add_argument("--out_manifest", default="manifests/ffpp_c23_faces256_manifest.csv")
    ap.add_argument("--include_dfd", action="store_true")
    ap.add_argument("--limit", type=int, default=0, help="smoke test: only first N frames per class")
    ap.add_argument("--batch", type=int, default=16)
    args = ap.parse_args()

    from facenet_pytorch import MTCNN
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    mtcnn = MTCNN(keep_all=True, device=device, post_process=False)

    df = pd.read_csv(args.manifest)
    if not args.include_dfd:
        df = df[df["manipulation_type"] != "DeepFakeDetection"]
    if args.limit:
        df = df.groupby("manipulation_type").head(args.limit)
    df = df.reset_index(drop=True)
    out_root = Path(args.out_root)

    paths = df["filepath"].tolist()
    loader = DataLoader(FrameDataset(paths), batch_size=args.batch, num_workers=4, collate_fn=collate)

    rows, stats = [], {}
    done = 0
    for batch in loader:
        # MTCNN batches need equal sizes -> group by small-image shape
        groups = {}
        for item in batch:
            i, rgb, small, s = item
            if rgb is None:
                continue
            groups.setdefault(small.shape, []).append(item)
        results = {}
        for shape, items in groups.items():
            arr = np.stack([it[2] for it in items])
            boxes, probs = mtcnn.detect(arr)
            for it, bx, pr in zip(items, boxes, probs):
                results[it[0]] = (it, bx, pr)

        for i in [it[0] for it in batch]:
            r = df.iloc[i]
            m = r["manipulation_type"]
            st = stats.setdefault(m, {"total": 0, "kept": 0, "no_face": 0, "unreadable": 0})
            st["total"] += 1
            if i not in results:
                st["unreadable"] += 1
                continue
            (_, rgb, _, s), bx, pr = results[i]
            if bx is None or pr is None or np.nanmax(pr) < MIN_PROB:
                st["no_face"] += 1
                continue
            k = int(np.nanargmax(pr))
            box = bx[k] / s
            face = square_crop(rgb, box)
            dst = out_root / m / Path(r["filepath"]).name
            dst.parent.mkdir(parents=True, exist_ok=True)
            cv2.imwrite(str(dst), cv2.cvtColor(face, cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, JPEG_QUALITY])
            st["kept"] += 1
            rows.append({
                "filepath": str(dst), "label": int(r["label"]), "manipulation_type": m,
                "src_filepath": r["filepath"], "det_prob": round(float(pr[k]), 4),
                "face_px": round(float(max(box[2] - box[0], box[3] - box[1])), 1),
            })
        done += len(batch)
        if done % 2000 < args.batch:
            logger.info(f"{done}/{len(paths)} frames processed")

    out = pd.DataFrame(rows)
    Path(args.out_manifest).parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(args.out_manifest, index=False)
    for m, st in stats.items():
        st["no_face_rate"] = round(st["no_face"] / max(1, st["total"]), 4)
    summary_path = Path(args.out_manifest).with_suffix(".summary.json")
    with open(summary_path, "w") as f:
        json.dump({"params": {"det_long_side": DET_LONG_SIDE, "box_scale": BOX_SCALE, "out_size": OUT_SIZE,
                              "jpeg_quality": JPEG_QUALITY, "min_prob": MIN_PROB}, "per_class": stats}, f, indent=2)
    logger.info(f"Wrote {len(out)} crops -> {args.out_manifest}")
    logger.info(json.dumps(stats, indent=2))


if __name__ == "__main__":
    main()
