"""
DeepTrace V11: leak-free retrain (fixes from audit_v10_claude/AUDIT_REPORT.md).

Same architecture as V10 (V10RobustDetector, 119.7M params) so results are comparable.
What changes:
  1. CLEAN INIT  — ImageNet EfficientNet + OpenAI CLIP only. No warm start from V1-V10,
                   which all descend from runs trained on Kaggle test fakes / FF++ test actors.
                   (--warm_start is only accepted for checkpoints carrying clean_lineage=True.)
  2. FACE CROPS  — FF++ uses manifests/ffpp_c23_faces256_manifest.csv (scripts/crop_ffpp_faces.py)
                   instead of full video frames.
  3. 3-WAY SPLIT — utils.actor_splits.get_actor_disjoint_split_3way: 540 train / 60 val / 400 test
                   actors. Test actors are identical to V9/V10 and are NEVER loaded here.
  4. GUARDS      — MD5 content check (train vs Kaggle valid/test + FF++ val/test) and actor check
                   run before training; the run aborts on any overlap.
  5. SBI FIXED   — self-blending is applied to REAL faces and relabelled FAKE (V10 blended fakes).
  6. BALANCED    — WeightedRandomSampler: 50% real / 50% fake, equal mass per FF++ cohort,
                   plain BCE (no 2.0-3.5x fake weights, no focal).
  7. SELECTION   — checkpoint chosen on val actors only; decision thresholds are fit on val and
                   stored in the checkpoint so evaluation can use a FROZEN threshold.

Usage:
  venv\\Scripts\\python.exe scripts\\train_v11_clean.py                 # full run
  venv\\Scripts\\python.exe scripts\\train_v11_clean.py --smoke         # 2 tiny epochs
"""
import sys
import json
import math
import random
import logging
import argparse
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from PIL import Image
from sklearn.metrics import roc_auc_score

sys.path.insert(0, str(Path(__file__).parent.parent))

from models.detector import DeepfakeDetector
from scripts.train_v10_robust import V10RobustDetector, compute_spectral_orthogonality_loss
from datasets.transforms import get_train_transforms, get_val_transforms
from utils.actor_splits import get_actor_disjoint_split_3way, _extract_video_id, _extract_actor_ids
from utils.checkpoint import load_checkpoint
from utils.device import AMPContext, get_grad_scaler
from utils.leakage_guard import assert_no_content_overlap, assert_actor_disjoint, assert_clean_lineage

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("v11_clean_train")

SEED = 42
KAGGLE = Path("data/kaggle_realfake/real_vs_fake/real-vs-fake")
FACES_MANIFEST = Path("manifests/ffpp_c23_faces256_manifest.csv")
COHORTS = ["Deepfakes", "FaceSwap", "FaceShifter", "Face2Face", "NeuralTextures"]
# manipulation head classes (5): 0 real, 1 GAN, 2 blend/swap (incl. SBI), 3 autoencoder, 4 reenactment
MANIP_IDX = {"real": 0, "kaggle_fake": 1, "FaceSwap": 2, "FaceShifter": 2, "sbi": 2,
             "Deepfakes": 3, "Face2Face": 4, "NeuralTextures": 4}
# sampling mass per group (sums to 1.0): real 0.5 / fake 0.5
GROUP_MASS = {"kaggle_real": 0.20, "ffpp_real": 0.30, "kaggle_fake": 0.15, "sbi": 0.10,
              **{f"ffpp_{m}": 0.25 / len(COHORTS) for m in COHORTS}}


# ─── Self-Blended Images (on REAL faces → label FAKE) ───────────────────────────
def self_blend(img: np.ndarray, rng: random.Random) -> np.ndarray:
    """Blend a slightly transformed copy of a real face into itself through a soft elliptical mask."""
    h, w = img.shape[:2]
    src = img.astype(np.float32)
    src = src * np.array([rng.uniform(0.9, 1.1) for _ in range(3)], np.float32) + \
        np.array([rng.uniform(-12, 12) for _ in range(3)], np.float32)
    s = rng.uniform(0.96, 1.04)
    tx, ty = rng.uniform(-0.02, 0.02) * w, rng.uniform(-0.02, 0.02) * h
    M = np.array([[s, 0, (1 - s) * w / 2 + tx], [0, s, (1 - s) * h / 2 + ty]], np.float32)
    src = cv2.warpAffine(src, M, (w, h), borderMode=cv2.BORDER_REFLECT)
    if rng.random() < 0.5:
        src = cv2.GaussianBlur(src, (3, 3), 0)
    mask = np.zeros((h, w), np.float32)
    center = (int(w / 2 + rng.uniform(-0.04, 0.04) * w), int(h * rng.uniform(0.50, 0.58)))
    axes = (int(w * rng.uniform(0.22, 0.32)), int(h * rng.uniform(0.28, 0.40)))
    cv2.ellipse(mask, center, axes, rng.uniform(-10, 10), 0, 360, 1.0, -1)
    k = int(w * rng.uniform(0.04, 0.12)) * 2 + 1
    mask = cv2.GaussianBlur(mask, (k, k), 0)[..., None]
    return np.clip(mask * src + (1 - mask) * img.astype(np.float32), 0, 255).astype(np.uint8)


class V11Dataset(Dataset):
    """samples: list of dicts {path, label, manip, sbi, group}"""

    def __init__(self, samples, transform, train: bool):
        self.samples, self.transform, self.train = samples, transform, train

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        s = self.samples[idx]
        img = np.array(Image.open(s["path"]).convert("RGB"))
        label = s["label"]
        if s["sbi"]:
            img = self_blend(img, random.Random(random.getrandbits(32)))
            label = 1
        x = self.transform(image=img)["image"]
        return {"image": x, "label": torch.tensor(label, dtype=torch.float32),
                "manip": torch.tensor(s["manip"], dtype=torch.long)}


# ─── Data ───────────────────────────────────────────────────────────────────────
def build_data(max_kaggle: int, smoke: bool, dense_manifest: str = None, val_from: str = None,
               kaggle_faces_manifest: str = None, kaggle_only: bool = False):
    rng = random.Random(SEED)
    train, val = [], []

    def add(lst, path, label, manip, group, sbi=False):
        lst.append({"path": str(path), "label": label, "manip": MANIP_IDX[manip], "group": group, "sbi": sbi})

    # Kaggle: train split only for training; valid split for val; test split is never loaded here
    kaggle_map = None
    if kaggle_faces_manifest:  # MTCNN-cropped Kaggle images (scripts/crop_kaggle_faces.py) — same preprocessing as FF++
        km = pd.read_csv(kaggle_faces_manifest)
        kaggle_map = {str(Path(a)): b for a, b in zip(km.src_filepath, km.filepath)}
        for cls, y, man in [("real", 0, "real"), ("fake", 1, "kaggle_fake")]:
            crops = sorted(km[(km.split == "train") & (km.label == y)].filepath)
            for p in rng.sample(crops, min(max_kaggle, len(crops))):
                add(train, p, y, man, f"kaggle_{cls}")
    for cls, y, man in ([] if kaggle_faces_manifest else [("real", 0, "real"), ("fake", 1, "kaggle_fake")]):
        paths = sorted((KAGGLE / "train" / cls).glob("*.jpg"))
        for p in rng.sample(paths, min(max_kaggle, len(paths))):
            add(train, p, y, man, f"kaggle_{cls}")
        vpaths = sorted((KAGGLE / "valid" / cls).glob("*.jpg"))
        for p in rng.sample(vpaths, 1000):
            add(val, p, y, man, f"kaggle_{cls}")

    # FF++ face crops, 3-way actor split (test actors are not even materialised in this function)
    if not FACES_MANIFEST.exists():
        raise FileNotFoundError(f"{FACES_MANIFEST} missing — run scripts/crop_ffpp_faces.py first")
    df = pd.read_csv(FACES_MANIFEST)
    sp = get_actor_disjoint_split_3way(df, filepath_col="filepath")
    if kaggle_only:  # cross-dataset experiment: NO FF++ data at all in train or val (sp only used by the guards)
        sp_used = {"train_reals": sp["train_reals"].iloc[0:0], "val_reals": sp["val_reals"].iloc[0:0],
                   "train_fakes": {m: sp["train_fakes"][m].iloc[0:0] for m in COHORTS},
                   "val_fakes": {m: sp["val_fakes"][m].iloc[0:0] for m in COHORTS}}
    else:
        sp_used = sp
    for p in sp_used["train_reals"]["filepath"]:
        add(train, p, 0, "real", "ffpp_real")
    for p in sp_used["val_reals"]["filepath"]:
        add(val, p, 0, "real", "ffpp_real")
    for m in COHORTS:
        for p in sp_used["train_fakes"][m]["filepath"]:
            add(train, p, 1, m, f"ffpp_{m}")
        for p in sp_used["val_fakes"][m]["filepath"]:
            add(val, p, 1, m, f"ffpp_{m}")

    # Optional extra frames (scripts/extract_ffpp_dense_train_faces.py) — TRAIN actors only, hard-checked
    if dense_manifest and not kaggle_only:
        dd = pd.read_csv(dense_manifest, dtype={"video_id": str})
        for vid in dd["video_id"].astype(str).unique():
            a, b = _extract_actor_ids(vid)
            if not (a in sp["train_actors"] and b in sp["train_actors"]):
                raise RuntimeError(f"LEAKAGE: dense manifest contains non-train actor video {vid}")
        for p, m in zip(dd["filepath"], dd["manipulation_type"]):
            add(train, p, 0 if m == "real" else 1, m, "ffpp_real" if m == "real" else f"ffpp_{m}")
        logger.info(f"Added {len(dd)} dense train-actor crops from {dense_manifest}")

    # Optional: reuse an earlier run's exact validation list (keeps val identical across experiments)
    if val_from:
        group2manip = {"kaggle_real": "real", "kaggle_fake": "kaggle_fake", "ffpp_real": "real",
                       **{f"ffpp_{m}": m for m in COHORTS}}
        val = []
        for path, label, group in json.load(open(val_from))["val"]:
            if kaggle_only and group.startswith("ffpp"):
                continue
            if group.startswith("kaggle") and kaggle_map is not None:
                if str(Path(path)) not in kaggle_map:  # MTCNN found no face -> cannot be scored by the app either
                    continue
                path = kaggle_map[str(Path(path))]
            add(val, path, label, group2manip[group], group)
        logger.info(f"Validation list loaded from {val_from} ({len(val)} items)")

    # SBI entries: every training REAL can also be drawn as a self-blended FAKE
    reals = [s for s in train if s["label"] == 0]
    train += [{**s, "sbi": True, "manip": MANIP_IDX["sbi"], "group": "sbi"} for s in reals]

    if smoke:
        rng.shuffle(train); rng.shuffle(val)
        train, val = train[:600], val[:300]
    return train, val, sp


def run_guards(train, val, sp):
    """Abort if training data overlaps any evaluation data (content or actor)."""
    train_paths = sorted({s["path"] for s in train})
    test_ffpp = list(sp["test_reals"]["filepath"]) + [p for m in COHORTS for p in sp["test_fakes"][m]["filepath"]]
    kaggle_test = [str(p) for c in ("real", "fake") for p in sorted((KAGGLE / "test" / c).glob("*.jpg"))]
    kf = Path("manifests/kaggle_faces256_manifest.csv")
    if kf.exists():  # also guard against the cropped versions of the Kaggle test images
        km = pd.read_csv(kf)
        kaggle_test += km[km.split == "test"].filepath.tolist()
    report = {
        "md5": assert_no_content_overlap(train_paths, {
            "val": sorted({s["path"] for s in val}), "kaggle_test": kaggle_test, "ffpp_test": test_ffpp}),
        "actors": assert_actor_disjoint([p for p in train_paths if "ffpp" in p.lower()], {
            "val": [s["path"] for s in val if "ffpp" in s["path"].lower()], "test": test_ffpp}),
    }
    return report


def make_sampler(train, epoch_size):
    sizes = {}
    for s in train:
        sizes[s["group"]] = sizes.get(s["group"], 0) + 1
    w = [GROUP_MASS[s["group"]] / sizes[s["group"]] for s in train]
    return WeightedRandomSampler(torch.tensor(w, dtype=torch.double), num_samples=epoch_size, replacement=True)


# ─── Model ──────────────────────────────────────────────────────────────────────
class V11Detector(V10RobustDetector):
    """V10 architecture; fixes the multitask key (V10 read 'manipulation_type_logits', the head
    returns 'manipulation_logits', so V10's manipulation loss was silently always zero)."""

    def forward(self, images, return_spectral_details: bool = False):
        out = super().forward(images, return_spectral_details=return_spectral_details)
        if out["multitask_logits"] is None:
            out["multitask_logits"] = self._last_manip_logits  # captured by _capture_manip_logits hook
        return out


def _capture_manip_logits(model: V11Detector):
    def hook(_m, _inp, outp):
        model._last_manip_logits = outp.get("manipulation_logits")
    model._last_manip_logits = None
    model.detection_head.register_forward_hook(hook)


def configure_trainable(model):
    """Partial unfreeze (as V3): EfficientNet blocks 4-6 + head, CLIP blocks 10-11; new modules fully trainable."""
    groups = {"effnet": [], "clip": [], "new": []}
    for enc in (model.spatial_encoder, model.frequency_encoder):
        for n, p in enc.named_parameters():
            p.requires_grad = any(f"blocks.{i}." in n for i in (4, 5, 6)) or "conv_head" in n or "bn2" in n
            if p.requires_grad:
                groups["effnet"].append(p)
    # CLIP visual stays frozen: at inference CLIPAlignment never runs the ViT (the fused "CLIP token"
    # is spatial_projection(EfficientNet features)); CLIP only supplies an alignment target in training.
    for n, p in model.clip_alignment.named_parameters():
        trainable = n.startswith("spatial_projection") or n.startswith("clip_projection")
        p.requires_grad = trainable
        if trainable:
            groups["clip"].append(p)
    for mod in (model.spectral_combiner, model.fusion, model.detection_head):
        for p in mod.parameters():
            p.requires_grad = True
            groups["new"].append(p)
    n_tr = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"Trainable params: {n_tr/1e6:.2f}M / {sum(p.numel() for p in model.parameters())/1e6:.2f}M")
    return groups


@torch.no_grad()
def predict(model, loader, device):
    model.eval()
    logits = []
    for b in loader:
        with AMPContext(device=device, enabled=True):
            out = model(b["image"].to(device, non_blocking=True), return_spectral_details=False)
        logits.append(out["binary_logit"].float().cpu().numpy())
    return np.nan_to_num(np.concatenate(logits), nan=0.0)


def balanced_acc_threshold(y, p):
    best_t, best = 0.5, -1.0
    for t in np.linspace(0.02, 0.98, 97):
        pred = p >= t
        ba = 0.5 * (np.mean(pred[y == 1]) + np.mean(~pred[y == 0]))
        if ba > best:
            best, best_t = ba, float(t)
    return round(best_t, 4), round(float(best), 4)


def val_metrics(val, logits):
    y = np.array([s["label"] for s in val])
    g = np.array([s["group"] for s in val])
    p = 1 / (1 + np.exp(-np.clip(logits, -20, 20)))
    m = {}
    k = np.isin(g, ["kaggle_real", "kaggle_fake"])
    m["kaggle_val_auc"] = round(float(roc_auc_score(y[k], p[k])), 4)
    cohort_aucs = {}
    for c in COHORTS:
        sel = (g == "ffpp_real") | (g == f"ffpp_{c}")
        if len(np.unique(y[sel])) == 2:
            cohort_aucs[c] = round(float(roc_auc_score(y[sel], p[sel])), 4)
    m["ffpp_val_cohort_auc"] = cohort_aucs
    m["ffpp_val_macro_auc"] = round(float(np.mean(list(cohort_aucs.values()))), 4) if cohort_aucs else 0.0
    f = np.char.startswith(g.astype(str), "ffpp")
    m["threshold_kaggle"], m["kaggle_val_bal_acc"] = balanced_acc_threshold(y[k], p[k])
    if not f.any():  # Kaggle-only (cross-dataset) run: no FF++ information may be used, FF++ gets the Kaggle threshold
        m["threshold_ffpp"], m["ffpp_val_bal_acc"], m["ffpp_val_bal_acc@0.5"] = m["threshold_kaggle"], None, None
        m["selection_score"] = m["kaggle_val_auc"]
        return m
    m["threshold_ffpp"], m["ffpp_val_bal_acc"] = balanced_acc_threshold(y[f], p[f])
    m["ffpp_val_bal_acc@0.5"] = round(float(0.5 * (np.mean(p[f][y[f] == 1] >= 0.5) + np.mean(p[f][y[f] == 0] < 0.5))), 4)
    # selection score: FF++ is the hard, generalisation-relevant part; Kaggle must not collapse
    m["selection_score"] = round(0.75 * m["ffpp_val_macro_auc"] + 0.25 * m["kaggle_val_auc"], 4)
    return m


# ─── Main ───────────────────────────────────────────────────────────────────────
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=15)
    ap.add_argument("--epoch_size", type=int, default=30000, help="samples drawn per epoch")
    ap.add_argument("--batch_size", type=int, default=32)
    ap.add_argument("--accumulation", type=int, default=2)
    ap.add_argument("--max_kaggle", type=int, default=10000)
    ap.add_argument("--dense_manifest", default=None, help="extra train-actor FF++ crops (V12)")
    ap.add_argument("--val_from", default=None, help="data_manifest.json of an earlier run: reuse its exact val list")
    ap.add_argument("--image_size", type=int, default=160)
    ap.add_argument("--kaggle_faces_manifest", default=None, help="manifests/kaggle_faces256_manifest.csv (V12)")
    ap.add_argument("--kaggle_only", action="store_true", help="cross-dataset run: train/val on Kaggle only, FF++ never seen")
    ap.add_argument("--patience", type=int, default=4)
    ap.add_argument("--out_dir", default="checkpoints/v11_clean")
    ap.add_argument("--warm_start", default=None, help="only checkpoints with clean_lineage=True are accepted")
    ap.add_argument("--resume", default=None, help="continue a paused run from its last.pth (same args, same out_dir)")
    ap.add_argument("--num_workers", type=int, default=3)
    ap.add_argument("--val_workers", type=int, default=0,
                    help="0 = validate in-process; extra Windows workers each reserve GBs of commit memory (MemoryError seen)")
    ap.add_argument("--pin_memory", action="store_true", help="off by default: CUDA 'resource already mapped' crash on Windows")
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()
    if args.smoke:
        args.epochs, args.epoch_size, args.out_dir = 2, 256, "checkpoints/_v11_smoke"

    random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    out_dir = Path(args.out_dir); out_dir.mkdir(parents=True, exist_ok=True)

    train, val, sp = build_data(args.max_kaggle, args.smoke, args.dense_manifest, args.val_from,
                                args.kaggle_faces_manifest, args.kaggle_only)
    guard_report = run_guards(train, val, sp)
    with open(out_dir / "data_manifest.json", "w") as f:
        json.dump({"train": [(s["path"], s["label"], s["group"], s["sbi"]) for s in train],
                   "val": [(s["path"], s["label"], s["group"]) for s in val],
                   "val_actors": sorted(sp["val_actors"]), "guards": guard_report}, f)
    logger.info(f"Train entries: {len(train)} (incl. SBI views) | Val: {len(val)}")

    model = V11Detector(DeepfakeDetector())
    _capture_manip_logits(model)
    lineage = {"init": "imagenet_efficientnet_b0 + openai_clip_vit_b32", "warm_start": None}
    if args.warm_start:
        assert_clean_lineage(args.warm_start)
        load_checkpoint(args.warm_start, model, device=device)
        lineage["warm_start"] = args.warm_start
    model.to(device)
    groups = configure_trainable(model)
    opt = torch.optim.AdamW([
        {"params": groups["effnet"], "lr": 3e-5, "weight_decay": 1e-4},
        {"params": groups["clip"], "lr": 1e-5, "weight_decay": 1e-4},
        {"params": groups["new"], "lr": 1e-4, "weight_decay": 1e-4},
    ])
    base_lrs = [g["lr"] for g in opt.param_groups]
    steps_per_epoch = math.ceil(args.epoch_size / args.batch_size / args.accumulation)
    total_steps, warmup = steps_per_epoch * args.epochs, steps_per_epoch  # 1 epoch warmup then cosine

    def set_lr(step):
        f = step / warmup if step < warmup else 0.5 * (1 + math.cos(math.pi * (step - warmup) / max(1, total_steps - warmup)))
        for g, lr in zip(opt.param_groups, base_lrs):
            g["lr"] = lr * max(f, 0.01)

    scaler = get_grad_scaler(device=device)
    train_loader = DataLoader(V11Dataset(train, get_train_transforms(args.image_size), True), batch_size=args.batch_size,
                              sampler=make_sampler(train, args.epoch_size), num_workers=args.num_workers,
                              pin_memory=args.pin_memory, persistent_workers=args.num_workers > 0, drop_last=True)
    val_loader = DataLoader(V11Dataset(val, get_val_transforms(args.image_size), False), batch_size=64, shuffle=False,
                            num_workers=args.val_workers, pin_memory=args.pin_memory, persistent_workers=args.val_workers > 0)

    best, bad_epochs, history, opt_step, start_epoch = -1.0, 0, [], 0, 1
    if args.resume:
        assert_clean_lineage(args.resume)
        ck = torch.load(args.resume, map_location="cpu", weights_only=False)
        model.load_state_dict(ck["model_state_dict"])
        lineage = ck.get("lineage", lineage)
        if "optimizer_state_dict" in ck:  # older last.pth files (epochs 1-5 of the first run) lack it
            opt.load_state_dict(ck["optimizer_state_dict"])
            scaler.load_state_dict(ck["scaler_state_dict"])
        else:
            logger.warning("Resume checkpoint has no optimizer state: AdamW moments restart (LR schedule is kept)")
        start_epoch = ck["epoch"] + 1
        opt_step = ck["epoch"] * steps_per_epoch
        hist_file = out_dir / "training_history.json"
        if hist_file.exists():
            history = json.load(open(hist_file))["history"]
            history = [h for h in history if h["epoch"] <= ck["epoch"]]
        if history:
            best_idx = max(range(len(history)), key=lambda i: history[i]["selection_score"])
            best, bad_epochs = history[best_idx]["selection_score"], len(history) - 1 - best_idx
        logger.info(f"Resumed from {args.resume}: next epoch {start_epoch}, best so far {best}, opt_step {opt_step}")

    for epoch in range(start_epoch, args.epochs + 1):
        model.train()
        tot, n = 0.0, 0
        opt.zero_grad(set_to_none=True)
        for step, b in enumerate(train_loader):
            x = b["image"].to(device, non_blocking=True)
            y = b["label"].to(device, non_blocking=True)
            mi = b["manip"].to(device, non_blocking=True)
            with AMPContext(device=device, enabled=True):
                out = model(x, return_spectral_details=True)
                l_bce = F.binary_cross_entropy_with_logits(out["binary_logit"], y)
                l_man = F.cross_entropy(out["multitask_logits"], mi) if out["multitask_logits"] is not None else 0.0
                l_ort = compute_spectral_orthogonality_loss(out.get("branch_features") or {})
                l_clip = out.get("clip_loss")
                l_clip = l_clip if l_clip is not None else 0.0
                loss = (l_bce + 0.15 * l_man + 0.10 * l_ort + 0.05 * l_clip) / args.accumulation
            scaler.scale(loss).backward()
            if (step + 1) % args.accumulation == 0:
                set_lr(opt_step)
                scaler.unscale_(opt)
                torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], 1.0)
                scaler.step(opt); scaler.update(); opt.zero_grad(set_to_none=True)
                opt_step += 1
            tot += loss.item() * args.accumulation; n += 1
            if step % 200 == 0:
                logger.info(f"epoch {epoch} step {step}/{len(train_loader)} loss {tot/max(n,1):.4f}")

        m = val_metrics(val, predict(model, val_loader, device))
        m.update({"epoch": epoch, "train_loss": round(tot / max(n, 1), 4)})
        history.append(m)
        logger.info(f"Epoch {epoch}: {json.dumps(m)}")

        state = {"epoch": epoch, "model_state_dict": model.state_dict(), "metrics": m,
                 "clean_lineage": True, "lineage": lineage, "val_actors": sorted(sp["val_actors"]),
                 "thresholds": {"ffpp": m["threshold_ffpp"], "kaggle": m["threshold_kaggle"]},
                 "args": vars(args)}
        torch.save({**state, "optimizer_state_dict": opt.state_dict(), "scaler_state_dict": scaler.state_dict()},
                   out_dir / "last.pth")
        if m["selection_score"] > best:
            best, bad_epochs = m["selection_score"], 0
            torch.save(state, out_dir / "best_model.pth")
            logger.info(f"New best (selection_score={best}) saved")
        else:
            bad_epochs += 1
        with open(out_dir / "training_history.json", "w") as f:
            json.dump({"best_selection_score": best, "history": history}, f, indent=2)
        if bad_epochs >= args.patience:
            logger.info(f"Early stop: no val improvement for {args.patience} epochs")
            break
    logger.info(f"Done. Best val selection score {best}")


if __name__ == "__main__":
    main()
