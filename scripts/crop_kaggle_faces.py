"""
MTCNN-crop the Kaggle real-vs-fake images with the SAME pipeline as FF++ (scripts/crop_ffpp_faces.py)
and the web predictor (inference/v11_predictor.py).

Why: V11 trained on raw Kaggle 256x256 images but the app re-crops every upload with MTCNN.
That mismatch cost ~0.06 AUC on Kaggle test through the app path (0.9645 -> 0.905,
checkpoints/v11_clean/app_threshold.json). Cropping every source identically removes it.

Output: data/kaggle_realfake/faces256/{train,valid,test}/{real,fake}/<name>.jpg
        manifests/kaggle_faces256_manifest.csv  (filepath,src_filepath,split,label,det_prob)
"""
import sys
import json
import argparse
import random
import logging
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader

sys.path.insert(0, str(Path(__file__).parent.parent))

from scripts.crop_ffpp_faces import DET_LONG_SIDE, MIN_PROB, JPEG_QUALITY, square_crop

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("crop_kaggle_faces")
SRC = Path("data/kaggle_realfake/real_vs_fake/real-vs-fake")


class Imgs(Dataset):
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
        small = cv2.resize(rgb, (round(w * s), round(h * s)), interpolation=cv2.INTER_AREA)  # same as predictor + FF++ cropper
        return i, rgb, small, s


def _collate(batch):
    return batch


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out_root", default="data/kaggle_realfake/faces256")
    ap.add_argument("--out_manifest", default="manifests/kaggle_faces256_manifest.csv")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--train_per_class", type=int, default=25000, help="0 = all train images")
    ap.add_argument("--valid_from", default="checkpoints/v11_clean/data_manifest.json",
                    help="only crop the Kaggle valid images used as val by this run (empty = all)")
    args = ap.parse_args()

    jobs = []
    rng = random.Random(42)
    val_keep = None
    if args.valid_from:  # crop exactly the validation images an earlier run used
        val_keep = {str(Path(v[0])) for v in json.load(open(args.valid_from))["val"] if v[2].startswith("kaggle")}
    for split in ("train", "valid", "test"):
        for cls, y in (("real", 0), ("fake", 1)):
            paths = [str(p) for p in sorted((SRC / split / cls).glob("*.jpg"))]
            if split == "train" and args.train_per_class:
                paths = sorted(rng.sample(paths, min(args.train_per_class, len(paths))))
            if split == "valid" and val_keep is not None:
                paths = [p for p in paths if str(Path(p)) in val_keep]
            jobs += [(p, split, cls, y) for p in paths]
    if args.limit:
        jobs = jobs[:: max(1, len(jobs) // args.limit)][: args.limit]
    out_root = Path(args.out_root)
    all_jobs = jobs
    jobs = [j for j in jobs if not (out_root / j[1] / j[2] / Path(j[0]).name).exists()]  # resumable
    logger.info(f"{len(jobs)} Kaggle images")

    from facenet_pytorch import MTCNN
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    mtcnn = MTCNN(keep_all=True, device=device, post_process=False)
    rows, stats = [], {}
    loader = DataLoader(Imgs([j[0] for j in jobs]), batch_size=64, num_workers=2, prefetch_factor=1, collate_fn=_collate)
    for b, batch in enumerate(loader):
        groups = {}
        for it in batch:
            if it[1] is not None:
                groups.setdefault(it[2].shape, []).append(it)
        res = {}
        for _, items in groups.items():
            bx, pr = mtcnn.detect(np.stack([it[2] for it in items]))
            for it, b_, p_ in zip(items, bx, pr):
                res[it[0]] = (it, b_, p_)
        for it in batch:
            i = it[0]
            src, split, cls, y = jobs[i]
            st = stats.setdefault(f"{split}/{cls}", {"total": 0, "kept": 0})
            st["total"] += 1
            if i not in res:
                continue
            (_, rgb, _, s), bx, pr = res[i]
            if bx is None or pr is None or np.nanmax(pr) < MIN_PROB:
                continue
            k = int(np.nanargmax(pr))
            face = square_crop(rgb, bx[k] / s)
            dst = out_root / split / cls / Path(src).name
            dst.parent.mkdir(parents=True, exist_ok=True)
            cv2.imwrite(str(dst), cv2.cvtColor(face, cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, JPEG_QUALITY])
            st["kept"] += 1
            rows.append({"filepath": str(dst), "src_filepath": src, "split": split, "label": y,
                         "det_prob": round(float(pr[k]), 4)})
        if b % 200 == 0:
            logger.info(f"{(b + 1) * 64}/{len(jobs)}")

    # manifest from DISK so resumed runs include earlier crops (det_prob only for this run's crops)
    det = {r["filepath"]: r["det_prob"] for r in rows}
    out = []
    for src, split, cls, y in all_jobs:
        dst = out_root / split / cls / Path(src).name
        if dst.exists():
            out.append({"filepath": str(dst), "src_filepath": src, "split": split, "label": y, "det_prob": det.get(str(dst))})
    pd.DataFrame(out).to_csv(args.out_manifest, index=False)
    stats = {}
    for sp_, cl_ in [(a, b) for a in ("train", "valid", "test") for b in ("real", "fake")]:  # over ALL jobs
        tot = sum(1 for j in all_jobs if j[1] == sp_ and j[2] == cl_)
        kept = sum(1 for r in out if r["split"] == sp_ and r["label"] == (0 if cl_ == "real" else 1))
        stats[f"{sp_}/{cl_}"] = {"total": tot, "kept": kept}
    for st in stats.values():
        st["drop_rate"] = round(1 - st["kept"] / max(1, st["total"]), 4)
    Path(args.out_manifest).with_suffix(".summary.json").write_text(json.dumps(stats, indent=2))
    logger.info(f"Wrote {len(rows)} crops; {json.dumps(stats)}")


if __name__ == "__main__":
    main()
