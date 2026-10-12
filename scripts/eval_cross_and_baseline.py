"""
Evaluation only (no training) of the two extra models for the V12 report:
  * DeepTrace trained on Kaggle ONLY  (checkpoints/v12_kaggle_only/best_model.pth) -> cross-dataset on FF++
  * MesoNet-4 baseline on the identical Kaggle split (results/baselines_same_split/mesonet_best.pth)
Both: threshold fit on Kaggle validation only; tested on Kaggle test crops and FF++ c23 test actors (never trained on).
Writes results/cross_dataset_and_baseline.json
"""
import sys
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset, DataLoader

sys.path.insert(0, str(Path(__file__).parent.parent))

from datasets.transforms import get_val_transforms
from models.detector import DeepfakeDetector
from models.mesonet import Meso4
from scripts.evaluate_v10_full_battery import V10EvalDetector
from scripts.train_baseline_same_split import report, bal_thr
from utils.actor_splits import get_actor_disjoint_split

COHORTS = ["Deepfakes", "FaceSwap", "FaceShifter", "Face2Face", "NeuralTextures"]


class DS(Dataset):
    def __init__(self, paths, size):
        self.paths, self.tf = paths, get_val_transforms(size)

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, i):
        return self.tf(image=np.array(Image.open(self.paths[i]).convert("RGB")))["image"]


@torch.no_grad()
def predict(model, paths, size, device):
    out = []
    for x in DataLoader(DS(paths, size), batch_size=128, num_workers=2):
        with torch.autocast(device_type="cuda", enabled=device.type == "cuda"):
            out.append(model(x.to(device)).float().reshape(-1).cpu().numpy())
    return np.concatenate(out)


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    km = pd.read_csv("manifests/kaggle_faces256_manifest.csv")
    kt = km[km.split == "test"].sort_values("filepath")
    val = [(p, y) for p, y, g in json.load(open("checkpoints/v12_dense/data_manifest.json"))["val"] if g.startswith("kaggle")]
    vp, vy = [p for p, _ in val], np.array([y for _, y in val])
    sp = get_actor_disjoint_split(pd.read_csv("manifests/ffpp_c23_faces256_manifest.csv"), "filepath")
    ff = pd.concat([sp["test_reals"].assign(manip="real", y=0)] +
                   [sp["test_fakes"][m].assign(manip=m, y=1) for m in COHORTS], ignore_index=True)

    models = {}
    ck = torch.load("checkpoints/v12_kaggle_only/best_model.pth", map_location="cpu", weights_only=False)
    dt = V10EvalDetector(DeepfakeDetector()); dt.load_state_dict(ck["model_state_dict"], strict=True)
    models["deeptrace_kaggle_only"] = (dt, int(ck["args"]["image_size"]), {"epochs_trained": ck["epoch"], "val_auc": ck["metrics"]["kaggle_val_auc"]})
    ms = Meso4(); ms.load_state_dict(torch.load("results/baselines_same_split/mesonet_best.pth", map_location="cpu"))
    models["mesonet_baseline"] = (ms, 160, {"epochs_trained": "3 of 10 (stopped for time)"})

    out = {"note": "Evaluation only. Both models trained on Kaggle only and stopped early for time "
                   "(DeepTrace-Kaggle-only: 1 epoch; MesoNet: 3 of 10 epochs) -> results are directional, "
                   "not converged. Threshold fit on Kaggle validation and applied unchanged to FF++."}
    for name, (m, size, meta) in models.items():
        m.to(device).eval()
        vz = predict(m, vp, size, device)
        thr = bal_thr(vy, 1 / (1 + np.exp(-vz)))
        r = {"meta": meta, "threshold_from_kaggle_val": round(thr, 4), "val_kaggle": report(vy, vz, thr)}
        r["kaggle_test"] = report(kt.label.values, predict(m, kt.filepath.tolist(), size, device), thr)
        ff["z"] = predict(m, ff.filepath.tolist(), size, device)
        reals = ff[ff.y == 0]
        r["ffpp_cross_dataset"] = {c: report(pd.concat([reals, ff[ff.manip == c]]).y.values,
                                             pd.concat([reals, ff[ff.manip == c]]).z.values, thr) for c in COHORTS}
        r["ffpp_cross_dataset"]["all"] = report(ff.y.values, ff.z.values, thr)
        r["ffpp_cross_dataset"]["macro_auc"] = round(float(np.mean([r["ffpp_cross_dataset"][c]["auc"] for c in COHORTS])), 4)
        out[name] = r
        m.cpu()
        print(name, "kaggle", r["kaggle_test"]["auc"], "ffpp macro", r["ffpp_cross_dataset"]["macro_auc"], flush=True)
    Path("results/cross_dataset_and_baseline.json").write_text(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
