"""
Baselines re-trained on EXACTLY the DeepTrace V12 Kaggle split, plus a cross-dataset test on FF++.

  --arch mesonet   : Meso-4 (Afchar et al., 2018), Adam lr 1e-3, 160 px
  --arch xception  : XceptionNet (Chollet 2017; FF++ baseline of Rossler et al., 2019), ImageNet init, AdamW 1e-4

Data (identical to V12's Kaggle part):
  train = Kaggle train MTCNN crops      (manifests/kaggle_faces256_manifest.csv, split=train)
  val   = V12's Kaggle validation list   (checkpoints/v12_dense/data_manifest.json, kaggle groups)
  test  = Kaggle test MTCNN crops (in-domain) + FF++ c23 test actors' face crops (CROSS-DATASET, never trained on)
Threshold fit on Kaggle val only (balanced accuracy) and applied unchanged to both test sets.

  venv\\Scripts\\python.exe scripts\\train_baseline_same_split.py --arch mesonet
"""
import sys
import json
import argparse
import logging
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from PIL import Image
from sklearn.metrics import roc_auc_score, brier_score_loss
from torch.utils.data import Dataset, DataLoader

sys.path.insert(0, str(Path(__file__).parent.parent))

from datasets.transforms import get_train_transforms, get_val_transforms
from utils.actor_splits import get_actor_disjoint_split
from utils.leakage_guard import assert_no_content_overlap

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("baseline_same_split")
COHORTS = ["Deepfakes", "FaceSwap", "FaceShifter", "Face2Face", "NeuralTextures"]


class DS(Dataset):
    def __init__(self, paths, labels, tf):
        self.paths, self.labels, self.tf = paths, labels, tf

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, i):
        x = self.tf(image=np.array(Image.open(self.paths[i]).convert("RGB")))["image"]
        return x, torch.tensor(self.labels[i], dtype=torch.float32)


def build_model(arch):
    if arch == "mesonet":
        from models.mesonet import Meso4
        return Meso4(), 160, (lambda m: torch.optim.Adam(m.parameters(), lr=1e-3))
    import timm
    m = timm.create_model("legacy_xception", pretrained=True, num_classes=1)
    return m, 160, (lambda mm: torch.optim.AdamW(mm.parameters(), lr=1e-4, weight_decay=1e-4))


@torch.no_grad()
def predict(model, paths, size, device):
    dl = DataLoader(DS(paths, [0] * len(paths), get_val_transforms(size)), batch_size=128, num_workers=2)
    model.eval()
    out = []
    for x, _ in dl:
        with torch.autocast(device_type="cuda", enabled=device.type == "cuda"):
            out.append(model(x.to(device)).float().reshape(-1).cpu().numpy())
    return np.concatenate(out)


def bal_thr(y, p):
    grid = np.linspace(0.02, 0.98, 97)
    ba = [0.5 * (np.mean(p[y == 1] >= t) + np.mean(p[y == 0] < t)) for t in grid]
    return float(grid[int(np.argmax(ba))])


def ece(y, p, bins=15):
    conf = np.where(p >= 0.5, p, 1 - p); pred = (p >= 0.5).astype(int)
    e, edges = 0.0, np.linspace(0.5, 1, bins + 1)
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (conf >= lo) & (conf <= hi) if lo == 0.5 else (conf > lo) & (conf <= hi)
        if m.any():
            e += m.mean() * abs((pred[m] == y[m]).mean() - conf[m].mean())
    return round(float(e), 4)


def report(y, z, thr):
    y = np.asarray(y); p = 1 / (1 + np.exp(-np.clip(z, -20, 20))); pr = p >= thr
    tp, fn = int(np.sum(pr & (y == 1))), int(np.sum(~pr & (y == 1)))
    tn, fp = int(np.sum(~pr & (y == 0))), int(np.sum(pr & (y == 0)))
    return {"n": int(len(y)), "auc": round(float(roc_auc_score(y, z)), 4),
            "recall_fake": round(tp / max(1, tp + fn), 4), "specificity_real": round(tn / max(1, tn + fp), 4),
            "precision_fake": round(tp / max(1, tp + fp), 4),
            "bal_acc": round(0.5 * (tp / max(1, tp + fn) + tn / max(1, tn + fp)), 4),
            "brier": round(float(brier_score_loss(y, p)), 4), "ece": ece(y, p),
            "confusion": {"tp": tp, "fn": fn, "tn": tn, "fp": fp}, "threshold": round(thr, 4)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arch", choices=["mesonet", "xception"], default="mesonet")
    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--batch_size", type=int, default=64)
    ap.add_argument("--val_manifest", default="checkpoints/v12_dense/data_manifest.json")
    args = ap.parse_args()
    torch.manual_seed(42); np.random.seed(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    out_dir = Path("results/baselines_same_split"); out_dir.mkdir(parents=True, exist_ok=True)

    km = pd.read_csv("manifests/kaggle_faces256_manifest.csv")
    tr = km[km.split == "train"]
    val = [(p, y) for p, y, g in json.load(open(args.val_manifest))["val"] if g.startswith("kaggle")]
    kt = km[km.split == "test"].sort_values("filepath")
    sp = get_actor_disjoint_split(pd.read_csv("manifests/ffpp_c23_faces256_manifest.csv"), "filepath")
    ff = pd.concat([sp["test_reals"].assign(manip="real", y=0)] +
                   [sp["test_fakes"][m].assign(manip=m, y=1) for m in COHORTS], ignore_index=True)
    guard = assert_no_content_overlap(tr.filepath.tolist(), {
        "val": [p for p, _ in val], "kaggle_test": kt.filepath.tolist(), "ffpp_test": ff.filepath.tolist()})

    model, size, make_opt = build_model(args.arch)
    model.to(device)
    opt = make_opt(model)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")
    loader = DataLoader(DS(tr.filepath.tolist(), tr.label.tolist(), get_train_transforms(size)),
                        batch_size=args.batch_size, shuffle=True, num_workers=2, drop_last=True)
    lossf = nn.BCEWithLogitsLoss()
    vp, vy = [p for p, _ in val], np.array([y for _, y in val])
    best, hist = -1.0, []
    ckpt = out_dir / f"{args.arch}_best.pth"
    for ep in range(1, args.epochs + 1):
        model.train()
        tot = 0.0
        for i, (x, y) in enumerate(loader):
            x, y = x.to(device), y.to(device)
            with torch.autocast(device_type="cuda", enabled=device.type == "cuda"):
                loss = lossf(model(x).reshape(-1).float(), y)
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward(); scaler.step(opt); scaler.update()
            tot += loss.item()
        sched.step()
        vz = predict(model, vp, size, device)
        vauc = float(roc_auc_score(vy, vz))
        hist.append({"epoch": ep, "train_loss": round(tot / (i + 1), 4), "val_auc": round(vauc, 4)})
        logger.info(f"[{args.arch}] epoch {ep}: {hist[-1]}")
        if vauc > best:
            best = vauc
            torch.save(model.state_dict(), ckpt)

    model.load_state_dict(torch.load(ckpt, map_location=device))
    vz = predict(model, vp, size, device)
    thr = bal_thr(vy, 1 / (1 + np.exp(-vz)))
    res = {"arch": args.arch, "image_size": size, "epochs": args.epochs, "history": hist,
           "best_val_auc": round(best, 4), "threshold_from_kaggle_val": round(thr, 4), "leakage_guard": guard,
           "val_kaggle": report(vy, vz, thr)}
    kz = predict(model, kt.filepath.tolist(), size, device)
    res["kaggle_test"] = report(kt.label.values, kz, thr)
    ff["logit"] = predict(model, ff.filepath.tolist(), size, device)
    reals = ff[ff.y == 0]
    res["ffpp_cross_dataset"] = {m: report(pd.concat([reals, ff[ff.manip == m]]).y.values,
                                           pd.concat([reals, ff[ff.manip == m]]).logit.values, thr) for m in COHORTS}
    res["ffpp_cross_dataset"]["all_cohorts"] = report(ff.y.values, ff.logit.values, thr)
    res["ffpp_cross_dataset"]["macro_auc"] = round(float(np.mean([res["ffpp_cross_dataset"][m]["auc"] for m in COHORTS])), 4)
    (out_dir / f"{args.arch}_results.json").write_text(json.dumps(res, indent=2))
    logger.info(f"[{args.arch}] Kaggle test {res['kaggle_test']['auc']} | FF++ cross macro {res['ffpp_cross_dataset']['macro_auc']}")


if __name__ == "__main__":
    main()
