"""
T3 — Run the V10 (and V9) checkpoint over every evaluation pool once and cache
raw logits + per-sample metadata to results/t3_logits_<ckpt>.pkl.  All
statistical tests in t4 work from this cache so they are cheap to re-run.

Uses the project's OWN eval model class, transforms and perturbation functions
(scripts/evaluate_v10_full_battery.py, scripts/evaluate_robustness.py) so the
numbers are directly comparable with the reported ones.
"""
import sys, os, json, random, argparse, time
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image

ROOT = Path(__file__).resolve().parent.parent
os.chdir(ROOT); sys.path.insert(0, str(ROOT))
OUT = Path(__file__).parent / "results"; OUT.mkdir(exist_ok=True)

from scripts.evaluate_v10_full_battery import V10EvalDetector
from scripts.evaluate_robustness import apply_jpeg, apply_blur, apply_noise, apply_downscale
from models.detector import DeepfakeDetector
from utils.checkpoint import load_checkpoint
from utils.device import AMPContext
from utils.actor_splits import get_actor_disjoint_split, _extract_video_id
from datasets.transforms import get_val_transforms

PERTURB = {
    "pristine": None,
    "jpeg_q50": lambda im: apply_jpeg(im, 50),
    "jpeg_q30": lambda im: apply_jpeg(im, 30),
    "noise_s025": lambda im: apply_noise(im, 0.25),
    "blur_k7": lambda im: apply_blur(im, 7),
    "hflip": lambda im: np.ascontiguousarray(im[:, ::-1]),
}


class DS(Dataset):
    def __init__(self, paths, perturb):
        self.paths, self.perturb, self.tf = paths, perturb, get_val_transforms(160)

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, i):
        try:
            im = np.array(Image.open(self.paths[i]).convert("RGB"))
            ok = 1
        except Exception:
            im = np.full((160, 160, 3), 128, np.uint8); ok = 0
        if self.perturb is not None:
            np.random.seed(i)
            im = PERTURB[self.perturb](im)
        return self.tf(image=im)["image"], ok


def build_pools():
    rng = random.Random(1234)
    pools = []  # (pool, path, label, extra dict)
    K = Path("data/kaggle_realfake/real_vs_fake/real-vs-fake")
    for c, y in [("real", 0), ("fake", 1)]:
        for p in (K / "test" / c).glob("*.jpg"):
            pools.append(("kaggle_test", str(p), y, {}))
        allv = list((K / "valid" / c).glob("*.jpg"))
        for p in allv[:1000]:
            pools.append(("kaggle_valid_v10val", str(p), y, {}))
        for p in rng.sample(allv[1000:], 1000):
            pools.append(("kaggle_valid_unused", str(p), y, {}))
        allt = list((K / "train" / c).glob("*.jpg"))
        for p in rng.sample(allt[:10000], 1000):
            pools.append(("kaggle_train_v10used", str(p), y, {}))
        for p in rng.sample(allt[10000:], 1000):
            pools.append(("kaggle_train_v10unused", str(p), y, {}))

    df = pd.read_csv("manifests/ffpp_c23_manifest.csv")
    sp = get_actor_disjoint_split(df, "filepath")
    for p in sp["test_reals"]["filepath"]:
        pools.append(("ffpp_test", p, 0, {"manip": "real"}))
    for m, d in sp["test_fakes"].items():
        for p in d["filepath"]:
            pools.append(("ffpp_test", p, 1, {"manip": m}))
    tr = sp["train_reals"]["filepath"].tolist()
    for p in rng.sample(tr, 2000):
        pools.append(("ffpp_train", p, 0, {"manip": "real"}))
    for m, d in sp["train_fakes"].items():
        for p in rng.sample(d["filepath"].tolist(), 400):
            pools.append(("ffpp_train", p, 1, {"manip": m}))
    for m, d in sp["discarded_fakes"].items():
        for p in rng.sample(d["filepath"].tolist(), 200):
            pools.append(("ffpp_mixedpair", p, 1, {"manip": m}))
    dfd = sp["dfd_excluded"]["filepath"].tolist()
    for p in rng.sample(dfd, 1000):
        pools.append(("ffpp_dfd", p, 1, {"manip": "DFD"}))

    N = Path("data/_new_dataset_extracted/Test")
    for c, y in [("Real", 0), ("Fake", 1)]:
        for p in (N / c).glob("*"):
            pools.append(("newds_test", str(p), y, {}))
    for c, y in [("real", 0), ("fake", 1)]:
        for p in (Path("test_data") / c).glob("*"):
            if p.is_file():
                pools.append(("test_data", str(p), y, {}))

    rows = [dict(pool=a, path=b, label=c, **d) for a, b, c, d in pools]
    df = pd.DataFrame(rows)
    df["video"] = df.apply(lambda r: _extract_video_id(r.path) if r.pool.startswith("ffpp") else r.path, axis=1)
    return df


def robustness_pool_paths():
    """Exact pool used by scripts/evaluate_robustness.py."""
    K = Path("data/kaggle_realfake/real_vs_fake/real-vs-fake/test")
    s = [(str(p), 0, "kaggle") for p in list((K / "real").glob("*.jpg"))[:1000]]
    s += [(str(p), 1, "kaggle") for p in list((K / "fake").glob("*.jpg"))[:1000]]
    sp = get_actor_disjoint_split(pd.read_csv("manifests/ffpp_c23_manifest.csv"), "filepath")
    s += [(p, 0, "ffpp") for p in sp["test_reals"]["filepath"].tolist()[:1000]]
    for m in ["Deepfakes", "FaceSwap", "FaceShifter", "Face2Face", "NeuralTextures"]:
        s += [(p, 1, "ffpp") for p in sp["test_fakes"][m]["filepath"].tolist()[:200]]
    return pd.DataFrame(s, columns=["path", "label", "src"])


@torch.no_grad()
def infer(model, paths, perturb, device, bs=64):
    dl = DataLoader(DS(paths, perturb), batch_size=bs, shuffle=False, num_workers=4, pin_memory=True)
    out, oks = [], []
    for x, ok in dl:
        with AMPContext(device=device, enabled=True):
            lg = model(x.to(device, non_blocking=True))
        out.append(lg.float().cpu().numpy()); oks.append(ok.numpy())
    return np.concatenate(out), np.concatenate(oks)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="checkpoints/v10_robust/best_model.pth")
    ap.add_argument("--tag", default="v10")
    ap.add_argument("--pools", default="all")
    ap.add_argument("--robustness", action="store_true")
    a = ap.parse_args()
    device = torch.device("cuda")
    model = V10EvalDetector(DeepfakeDetector())
    load_checkpoint(a.ckpt, model, device=device)
    model.to(device).eval()

    df = build_pools()
    if a.pools != "all":
        df = df[df.pool.isin(a.pools.split(","))].reset_index(drop=True)
    t = time.time()
    df["logit"], df["ok"] = infer(model, df.path.tolist(), None, device)
    print(f"[{a.tag}] main pools: {len(df)} imgs in {time.time()-t:.0f}s")
    # flip-consistency on FF++ test + Kaggle test subset
    sub = df[df.pool.isin(["ffpp_test", "kaggle_test", "test_data"])].index
    if len(sub):
        df.loc[sub, "logit_hflip"], _ = infer(model, df.loc[sub, "path"].tolist(), "hflip", device)
    df.to_pickle(OUT / f"t3_logits_{a.tag}.pkl")

    if a.robustness:
        rp = robustness_pool_paths()
        for name in ["pristine", "jpeg_q50", "jpeg_q30", "noise_s025", "blur_k7"]:
            t = time.time()
            rp[name], _ = infer(model, rp.path.tolist(), name if name != "pristine" else None, device)
            print(f"[{a.tag}] robustness {name}: {time.time()-t:.0f}s")
        rp.to_pickle(OUT / f"t3_robustness_{a.tag}.pkl")
        # also the whole FF++ test set under JPEG q50 / q30 (cross-domain robustness proper)
        ft = df[df.pool == "ffpp_test"][["path", "label", "manip", "video"]].copy()
        for name in ["jpeg_q50", "jpeg_q30"]:
            ft[name], _ = infer(model, ft.path.tolist(), name, device)
        ft.to_pickle(OUT / f"t3_ffpp_test_perturbed_{a.tag}.pkl")
    print("done")


if __name__ == "__main__":
    main()
