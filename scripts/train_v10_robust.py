"""
DeepTrace V10: Robust Multi-Source Forensics Training Engine.

Key Upgrades over V7/V9:
1. Persistent JPEG Compression Augmentation (p=0.7 standalone) in training pipeline.
2. Balanced Multi-Source Training Pool: Kaggle In-Domain (10k real + 10k fake) +
   FF++ 100% Actor-Disjoint (600 train actors, 5 manipulation cohorts: Deepfakes,
   Face2Face, FaceSwap, FaceShifter, NeuralTextures) + On-The-Fly SBI Boundaries.
3. Strict Actor-Disjoint Validation: 400 held-out test actors + in-domain validation.
4. Tri-Objective Robustness Loss (Boundary Focal Loss + Spectral Orthogonal Loss +
   Multi-Task Manipulation Loss + CLIP Semantic Loss).
5. Isolated checkpoint output: checkpoints/v10_robust/best_model.pth.
"""

import os
import sys
import math
import random
import logging
import argparse
import json
from pathlib import Path
import numpy as np
import pandas as pd
from PIL import Image, ImageFilter
import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import accuracy_score, roc_auc_score, f1_score

sys.path.insert(0, str(Path(__file__).parent.parent))

from models.detector import DeepfakeDetector
from models.spectral_branches import SOTASpectralCombiner
from datasets.transforms import get_train_transforms, get_val_transforms
from utils.checkpoint import load_checkpoint, save_checkpoint
from utils.device import get_device, AMPContext, get_grad_scaler
from utils.actor_splits import get_actor_disjoint_split

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("v10_robust_train")


class V10RobustDetector(nn.Module):
    """
    V10 Robust Detector Architecture:
    Spatial (EfficientNet-B0) + DCT Frequency (EfficientNet-B0) +
    SOTA Multi-Spectral Combiner (FFT SPR, Wavelet Packet, SRM/Gabor, LSGN) +
    OpenCLIP ViT Semantic Projection + Cross-Attention Fusion + Multi-Task Head.
    """
    def __init__(self, base_detector: DeepfakeDetector):
        super().__init__()
        self.spatial_encoder = base_detector.spatial_encoder
        self.frequency_encoder = base_detector.frequency_encoder
        self.clip_alignment = base_detector.clip_alignment
        self.fusion = base_detector.fusion
        self.detection_head = base_detector.detection_head

        self.spectral_combiner = SOTASpectralCombiner(
            dct_in_dim=1280,
            branch_dim=320,
            output_dim=1280,
            gradient_checkpointing=True,
        )

    def forward(self, images: torch.Tensor, return_spectral_details: bool = False):
        spatial_feat = self.spatial_encoder(images)
        dct_feat = self.frequency_encoder(images)
        spectral_out = self.spectral_combiner(images, dct_feat, return_branches=return_spectral_details)

        if return_spectral_details:
            combined_spectral_feat = spectral_out["combined"]
            branch_dict = spectral_out["branches"]
            gate_weights = spectral_out["gate_weights"]
        else:
            combined_spectral_feat = spectral_out
            branch_dict = None
            gate_weights = None

        clip_result = self.clip_alignment(spatial_feat, images, compute_alignment_loss=True)
        clip_proj = clip_result["spatial_projected"]
        clip_loss = clip_result["alignment_loss"]

        fused_feat = self.fusion(
            spatial_features=spatial_feat,
            frequency_features=combined_spectral_feat,
            temporal_features=None,
            physiology_features=None,
            clip_features=clip_proj,
        )

        predictions = self.detection_head(fused_feat)
        binary_logit = predictions["binary_logit"].squeeze(-1)

        out = {
            "binary_logit": binary_logit,
            "clip_loss": clip_loss,
            "multitask_logits": predictions.get("manipulation_type_logits", None),
        }
        if return_spectral_details:
            out["branch_features"] = branch_dict
            out["gate_weights"] = gate_weights
        return out


def generate_dynamic_sbi_seam(img_np: np.ndarray, p_blend: float = 0.5) -> np.ndarray:
    """Generate dynamic self-blended image boundary artifact on numpy RGB image."""
    if random.random() > p_blend:
        return img_np
    h, w, _ = img_np.shape
    cx, cy = w // 2, h // 2
    rw, rh = int(w * random.uniform(0.35, 0.65)), int(h * random.uniform(0.35, 0.65))
    x1, y1 = max(0, cx - rw // 2), max(0, cy - rh // 2)
    x2, y2 = min(w, cx + rw // 2), min(h, cy + rh // 2)

    crop = img_np[y1:y2, x1:x2].copy()
    if random.random() < 0.5:
        crop = cv2.GaussianBlur(crop, (5, 5), random.uniform(1.0, 3.0))

    # Soft alpha feathering
    mask = np.zeros((y2 - y1, x2 - x1), dtype=np.float32)
    for px in range(x2 - x1):
        for py in range(y2 - y1):
            dx = min(px, x2 - x1 - px) / max(1.0, (x2 - x1) * 0.15)
            dy = min(py, y2 - y1 - py) / max(1.0, (y2 - y1) * 0.15)
            mask[py, px] = min(1.0, dx, dy)

    mask_3ch = np.repeat(mask[:, :, np.newaxis], 3, axis=2)
    blended = img_np.copy()
    blended[y1:y2, x1:x2] = (crop * mask_3ch + blended[y1:y2, x1:x2] * (1.0 - mask_3ch)).astype(np.uint8)
    return blended


class V10Dataset(Dataset):
    """
    V10 Multi-Source Forensic Dataset:
    Loads images and applies Albumentations transform (with heavy JPEG compression).
    """
    def __init__(self, samples, transform, is_training: bool = True):
        self.samples = samples
        self.transform = transform
        self.is_training = is_training

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, binary_label, manip_idx, sample_weight = self.samples[idx]
        try:
            img = Image.open(path).convert("RGB")
            img_np = np.array(img)
        except Exception:
            img_np = np.full((160, 160, 3), 128, dtype=np.uint8)

        if self.is_training and binary_label == 1 and random.random() < 0.30:
            img_np = generate_dynamic_sbi_seam(img_np, p_blend=0.80)

        augmented = self.transform(image=img_np)
        tensor_img = augmented["image"]

        return {
            "image": tensor_img,
            "binary_label": torch.tensor(binary_label, dtype=torch.float32),
            "manip_label": torch.tensor(manip_idx, dtype=torch.long),
            "weight": torch.tensor(sample_weight, dtype=torch.float32),
        }


def compute_spectral_orthogonality_loss(branch_dict: dict) -> torch.Tensor:
    if not branch_dict:
        return torch.tensor(0.0)
    feats = [F.normalize(feat, p=2, dim=-1) for feat in branch_dict.values()]
    ortho_loss = torch.tensor(0.0, device=feats[0].device)
    pairs = 0
    for i in range(len(feats)):
        for j in range(i + 1, len(feats)):
            cosine_sim = (feats[i] * feats[j]).sum(dim=-1).abs().mean()
            ortho_loss = ortho_loss + cosine_sim
            pairs += 1
    return ortho_loss / max(pairs, 1)


def build_v10_dataset(max_kaggle: int = 10000):
    logger.info("Constructing V10 Robust Multi-Source Dataset...")
    train_samples = []
    val_samples = []

    # 1. In-domain Kaggle anchor (StyleGAN vs Real) - capped at max_kaggle
    k_train = Path("data/kaggle_realfake/real_vs_fake/real-vs-fake/train")
    k_val = Path("data/kaggle_realfake/real_vs_fake/real-vs-fake/valid")

    k_real_train = list((k_train / "real").glob("*.jpg"))[:max_kaggle]
    k_fake_train = list((k_train / "fake").glob("*.jpg"))[:max_kaggle]
    for p in k_real_train:
        train_samples.append((str(p), 0, 0, 1.0))
    for p in k_fake_train:
        train_samples.append((str(p), 1, 1, 1.0))

    for p in list((k_val / "real").glob("*.jpg"))[:1000]:
        val_samples.append((str(p), 0, 0, 1.0))
    for p in list((k_val / "fake").glob("*.jpg"))[:1000]:
        val_samples.append((str(p), 1, 1, 1.0))

    logger.info(f"Kaggle contribution: {len(k_real_train)} real + {len(k_fake_train)} fake")

    # 2. FaceForensics++ strictly actor-disjoint (100% leak-free)
    ffpp_csv = Path("manifests/ffpp_c23_manifest.csv")
    if ffpp_csv.exists():
        df_ffpp = pd.read_csv(ffpp_csv)
        col = "filepath" if "filepath" in df_ffpp.columns else "image_path"
        split = get_actor_disjoint_split(df_ffpp, filepath_col=col)

        # Train reals: frames from 600 train actors
        train_real_paths = split["train_reals"][col].tolist()
        for p in train_real_paths:
            train_samples.append((str(p), 0, 0, 1.2))

        # Test/Val reals: frames from 400 held-out test actors
        test_real_paths = split["test_reals"][col].tolist()
        for p in test_real_paths[:1000]:
            val_samples.append((str(p), 0, 0, 1.0))

        manip_specs = [
            ("FaceSwap", 2, 3.5),
            ("FaceShifter", 2, 3.5),
            ("Deepfakes", 3, 2.0),
            ("Face2Face", 4, 2.5),
            ("NeuralTextures", 4, 2.0),
        ]

        for manip, c_idx, w_val in manip_specs:
            if manip in split["train_fakes"]:
                train_fake_paths = split["train_fakes"][manip][col].tolist()
                for p in train_fake_paths:
                    train_samples.append((str(p), 1, c_idx, w_val))
            if manip in split["test_fakes"]:
                test_fake_paths = split["test_fakes"][manip][col].tolist()
                for p in test_fake_paths[:400]:
                    val_samples.append((str(p), 1, c_idx, 1.0))

    random.shuffle(train_samples)
    random.shuffle(val_samples)
    logger.info(f"V10 Total Dataset: {len(train_samples)} Train | {len(val_samples)} Val")
    return train_samples, val_samples


def main():
    parser = argparse.ArgumentParser(description="DeepTrace V10 Robust Training")
    parser.add_argument("--epochs", type=int, default=20, help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=32, help="Mini-batch size")
    parser.add_argument("--accumulation", type=int, default=2, help="Gradient accumulation steps")
    parser.add_argument("--max_kaggle", type=int, default=10000, help="Max Kaggle samples per class")
    parser.add_argument("--lr", type=float, default=1e-5, help="Base learning rate")
    args = parser.parse_args()

    device = get_device()
    logger.info(f"=== Starting DeepTrace V10 Robust Training on {device} ===")
    logger.info(f"Config: Epochs={args.epochs}, BatchSize={args.batch_size}, Accumulation={args.accumulation}, BaseLR={args.lr}")

    out_dir = Path("checkpoints/v10_robust")
    out_dir.mkdir(parents=True, exist_ok=True)
    results_dir = Path("results/benchmark_eval_v10")
    results_dir.mkdir(parents=True, exist_ok=True)

    base_detector = DeepfakeDetector()
    model = V10RobustDetector(base_detector)

    # Warm start priority: v9_actor_disjoint -> v7_sota_spectral -> base
    ckpt_v9 = "checkpoints/v9_actor_disjoint/best_model.pth"
    ckpt_v7 = "checkpoints/v7_sota_spectral/best_model.pth"

    if Path(ckpt_v9).exists():
        logger.info(f"Warm-starting from V9 Actor-Disjoint: {ckpt_v9}")
        load_checkpoint(ckpt_v9, model, device=device)
    elif Path(ckpt_v7).exists():
        logger.info(f"Warm-starting from V7 Multi-Spectral: {ckpt_v7}")
        load_checkpoint(ckpt_v7, model, device=device)
    else:
        logger.info("Initializing from scratch / base detector")

    model.to(device)

    # Differential learning rates with decay
    optimizer = torch.optim.AdamW(
        [
            {"params": model.spectral_combiner.parameters(), "lr": args.lr * 10, "weight_decay": 1e-4},
            {"params": model.spatial_encoder.parameters(), "lr": args.lr * 1.5, "weight_decay": 1e-4},
            {"params": model.frequency_encoder.parameters(), "lr": args.lr * 1.5, "weight_decay": 1e-4},
            {"params": model.fusion.parameters(), "lr": args.lr * 5, "weight_decay": 1e-4},
            {"params": model.detection_head.parameters(), "lr": args.lr * 5, "weight_decay": 1e-4},
            {"params": model.clip_alignment.parameters(), "lr": args.lr * 0.2, "weight_decay": 1e-4},
        ]
    )
    scaler = get_grad_scaler(device=device)

    train_transforms = get_train_transforms(image_size=160)
    val_transforms = get_val_transforms(image_size=160)

    train_samples, val_samples = build_v10_dataset(max_kaggle=args.max_kaggle)
    train_loader = DataLoader(
        V10Dataset(train_samples, train_transforms, is_training=True),
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=0,  # safe multiprocessing on Windows
        pin_memory=(device.type == "cuda"),
    )
    val_loader = DataLoader(
        V10Dataset(val_samples, val_transforms, is_training=False),
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=(device.type == "cuda"),
    )

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-7)
    bce_loss_fn = nn.BCEWithLogitsLoss(reduction="none")
    ce_loss_fn = nn.CrossEntropyLoss()

    best_val_auc = 0.0
    training_history = []

    for epoch in range(1, args.epochs + 1):
        model.train()
        train_loss, train_steps = 0.0, 0
        optimizer.zero_grad(set_to_none=True)

        for step, batch in enumerate(train_loader):
            imgs = batch["image"].to(device, non_blocking=True)
            labels = batch["binary_label"].to(device, non_blocking=True)
            manips = batch["manip_label"].to(device, non_blocking=True)
            weights = batch["weight"].to(device, non_blocking=True)

            with AMPContext(device=device, enabled=True):
                out = model(imgs, return_spectral_details=True)
                raw_bce = bce_loss_fn(out["binary_logit"], labels)
                pt = torch.exp(-raw_bce)
                focal_bce = ((1.0 - pt) ** 2.0) * raw_bce * weights
                loss_bce = focal_bce.mean()

                loss_ortho = compute_spectral_orthogonality_loss(out.get("branch_features", {}))
                loss_clip = out.get("clip_loss", torch.tensor(0.0, device=device))
                loss_manip = ce_loss_fn(out["multitask_logits"], manips) if out["multitask_logits"] is not None else torch.tensor(0.0, device=device)

                total_loss = (loss_bce + 0.10 * loss_ortho + 0.05 * loss_clip + 0.15 * loss_manip) / args.accumulation

            scaler.scale(total_loss).backward()

            if (step + 1) % args.accumulation == 0 or (step + 1) == len(train_loader):
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)

            train_loss += total_loss.item() * args.accumulation
            train_steps += 1

        scheduler.step()

        # Validation
        model.eval()
        all_preds, all_labels = [], []
        with torch.no_grad():
            for batch in val_loader:
                imgs = batch["image"].to(device, non_blocking=True)
                labels = batch["binary_label"].cpu().numpy()
                with AMPContext(device=device, enabled=True):
                    out = model(imgs, return_spectral_details=False)
                    logits = np.nan_to_num(out["binary_logit"].float().cpu().numpy(), nan=0.0)
                all_preds.extend(logits)
                all_labels.extend(labels)

        preds_arr = np.nan_to_num(np.array(all_preds), nan=0.0)
        probs = 1.0 / (1.0 + np.exp(-np.clip(preds_arr, -20.0, 20.0)))
        probs = np.nan_to_num(probs, nan=0.5)
        val_auc = float(roc_auc_score(all_labels, probs))
        val_acc = float(accuracy_score(all_labels, probs >= 0.5))
        avg_train_loss = train_loss / max(1, train_steps)

        logger.info(f"Epoch {epoch}/{args.epochs} -> Train Loss: {avg_train_loss:.4f} | Val AUC: {val_auc:.4f} | Val Acc: {val_acc*100:.2f}%")

        epoch_stat = {
            "epoch": epoch,
            "train_loss": avg_train_loss,
            "val_auc": val_auc,
            "val_acc": val_acc,
        }
        training_history.append(epoch_stat)

        # Save last checkpoint
        save_checkpoint(
            model=model,
            optimizer=optimizer,
            epoch=epoch,
            metrics={"val_auc": val_auc, "val_acc": val_acc},
            path=str(out_dir / "last.pth"),
        )

        if val_auc > best_val_auc:
            best_val_auc = val_auc
            save_checkpoint(
                model=model,
                optimizer=optimizer,
                epoch=epoch,
                metrics={"val_auc": val_auc, "val_acc": val_acc},
                path=str(out_dir / "best_model.pth"),
            )
            logger.info(f"🏆 New Best V10 Robust Model Saved (Val AUC = {val_auc:.4f})")

        # Save training history JSON
        with open(results_dir / "v10_training_history.json", "w") as f:
            json.dump({"best_val_auc": best_val_auc, "history": training_history}, f, indent=2)

    logger.info(f"=== DeepTrace V10 Robust Training Complete! Best Val AUC: {best_val_auc:.4f} ===")


if __name__ == "__main__":
    main()
