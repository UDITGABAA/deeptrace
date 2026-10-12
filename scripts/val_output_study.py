"""
Output-quality decisions made on VALIDATION only (never test):
  1. hflip test-time augmentation: keep it only if it improves val (FF++ macro AUC up, Kaggle AUC not down >0.002)
  2. temperature scaling: one scalar T fit on val logits (min NLL) so probabilities are calibrated
  3. decision thresholds re-fit on val for the chosen mode

Writes <checkpoint_dir>/output_calibration.json, read by scripts/evaluate_v11_honest.py.

  venv\\Scripts\\python.exe scripts\\val_output_study.py --checkpoint checkpoints/v11_clean/best_model.pth
"""
import sys
import json
import argparse
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from scipy.optimize import minimize_scalar
from sklearn.metrics import roc_auc_score
from torch.utils.data import Dataset, DataLoader

sys.path.insert(0, str(Path(__file__).parent.parent))

from datasets.transforms import get_val_transforms
from models.detector import DeepfakeDetector
from scripts.evaluate_v10_full_battery import V10EvalDetector
from utils.device import AMPContext

COHORTS = ["Deepfakes", "FaceSwap", "FaceShifter", "Face2Face", "NeuralTextures"]


class DS(Dataset):
    def __init__(self, paths, size):
        self.paths, self.tf = paths, get_val_transforms(size)

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, i):
        return self.tf(image=np.array(Image.open(self.paths[i]).convert("RGB")))["image"]


@torch.no_grad()
def logits(model, paths, size, device):
    z, zf = [], []
    for x in DataLoader(DS(paths, size), batch_size=64, num_workers=0):
        x = x.to(device)
        with AMPContext(device=device, enabled=True):
            z.append(model(x).float().cpu().numpy())
            zf.append(model(torch.flip(x, dims=[3])).float().cpu().numpy())
    return np.concatenate(z), np.concatenate(zf)


def bal_thr(y, p):
    grid = np.linspace(0.02, 0.98, 97)
    ba = [0.5 * (np.mean(p[y == 1] >= t) + np.mean(p[y == 0] < t)) for t in grid]
    return round(float(grid[int(np.argmax(ba))]), 4), round(float(max(ba)), 4)


def ece(y, p, bins=15):
    conf = np.where(p >= 0.5, p, 1 - p); pred = (p >= 0.5).astype(int)
    e = 0.0
    for lo, hi in zip(np.linspace(0.5, 1, bins + 1)[:-1], np.linspace(0.5, 1, bins + 1)[1:]):
        m = (conf >= lo) & (conf <= hi) if lo == 0.5 else (conf > lo) & (conf <= hi)
        if m.any():
            e += m.mean() * abs((pred[m] == y[m]).mean() - conf[m].mean())
    return round(float(e), 4)


def metrics(y, g, z):
    out = {"kaggle_auc": round(float(roc_auc_score(y[np.char.startswith(g, "kaggle")], z[np.char.startswith(g, "kaggle")])), 4)}
    aucs = {}
    for c in COHORTS:
        sel = (g == "ffpp_real") | (g == f"ffpp_{c}")
        aucs[c] = round(float(roc_auc_score(y[sel], z[sel])), 4)
    out["ffpp_cohort_auc"] = aucs
    out["ffpp_macro_auc"] = round(float(np.mean(list(aucs.values()))), 4)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", default="checkpoints/v11_clean/best_model.pth")
    ap.add_argument("--val_manifest", default=None,
                    help="score another run's val list (comparison only; nothing is written)")
    args = ap.parse_args()
    ck = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    size = int(ck.get("args", {}).get("image_size", 160))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = V10EvalDetector(DeepfakeDetector())
    model.load_state_dict(ck["model_state_dict"], strict=True)
    model.to(device).eval()

    val = json.load(open(args.val_manifest or Path(args.checkpoint).parent / "data_manifest.json"))["val"]
    paths = [v[0] for v in val]; y = np.array([v[1] for v in val]); g = np.array([v[2] for v in val])
    z, zf = logits(model, paths, size, device)
    zt = 0.5 * (z + zf)

    plain, tta = metrics(y, g, z), metrics(y, g, zt)
    use_tta = tta["ffpp_macro_auc"] > plain["ffpp_macro_auc"] and tta["kaggle_auc"] >= plain["kaggle_auc"] - 0.002
    zc = zt if use_tta else z

    # temperature: minimise NLL on val
    def nll(T):
        p = np.clip(1 / (1 + np.exp(-zc / T)), 1e-7, 1 - 1e-7)
        return -np.mean(y * np.log(p) + (1 - y) * np.log(1 - p))
    T = float(minimize_scalar(nll, bounds=(0.05, 20), method="bounded").x)
    p0, pT = 1 / (1 + np.exp(-zc)), 1 / (1 + np.exp(-zc / T))

    f = np.char.startswith(g, "ffpp"); k = ~f
    thr_ffpp, ba_ffpp = bal_thr(y[f], pT[f])
    thr_kaggle, ba_kaggle = bal_thr(y[k], pT[k])
    out = {
        "checkpoint": args.checkpoint, "image_size": size, "decided_on": "validation only",
        "val_plain": plain, "val_tta": tta, "use_tta": bool(use_tta),
        "temperature": round(T, 4),
        "val_ece_before": ece(y, p0), "val_ece_after(in-sample)": ece(y, pT),
        "val_nll_before": round(float(nll(1.0)), 4), "val_nll_after": round(float(nll(T)), 4),
        "thresholds": {"ffpp": thr_ffpp, "kaggle": thr_kaggle},
        "val_bal_acc_at_thresholds": {"ffpp": ba_ffpp, "kaggle": ba_kaggle},
    }
    if args.val_manifest:  # comparison mode: do not overwrite this checkpoint's calibration
        out["val_manifest"] = args.val_manifest
    else:
        (Path(args.checkpoint).parent / "output_calibration.json").write_text(json.dumps(out, indent=2))
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
