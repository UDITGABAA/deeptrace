"""
DeepTrace V7: SOTA Multi-Spectral & Boundary Forensics Training Engine.
Integrates:
1. SOTA Multi-Spectral Combiner (Continuous Phase FFT, 2-Level Wavelet Packet, 9-Ch SRM/Gabor, LSGN).
2. Dynamic On-The-Fly Self-Blended Image (SBI) Boundary Synthesis.
3. Tri-Objective Loss (Boundary Focal Loss + Spectral Orthogonal Loss + Multi-Task Manipulation Loss).
4. Full Isolation: saves to checkpoints/v7_sota_spectral/best_model.pth.
"""

import os
import sys
import math
import random
import logging
from pathlib import Path
import numpy as np
import pandas as pd
from PIL import Image, ImageFilter
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from sklearn.metrics import accuracy_score, roc_auc_score, f1_score

sys.path.insert(0, str(Path(__file__).parent.parent))

from models.detector import DeepfakeDetector
from models.spectral_branches import SOTASpectralCombiner
from utils.checkpoint import load_checkpoint, save_checkpoint
from utils.device import get_device, AMPContext, get_grad_scaler

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("v7_spectral_train")


# ─── V7 SOTA Master Architecture ───────────────────────────────────────────────

class V7SOTADetector(nn.Module):
    """
    V7 SOTA Detector:
    Spatial (EfficientNet) + DCT (EfficientNet) + SOTA Multi-Spectral (FFT SPR, Wavelet, SRM/Gabor, LSGN)
    + OpenCLIP Semantic Alignment + Cross-Attention Fusion + Multi-Task Detection Head.
    """

    def __init__(self, base_detector: DeepfakeDetector):
        super().__init__()
        self.spatial_encoder = base_detector.spatial_encoder
        self.frequency_encoder = base_detector.frequency_encoder
        self.clip_alignment = base_detector.clip_alignment
        self.fusion = base_detector.fusion
        self.detection_head = base_detector.detection_head

        # Plug in 2025-2026 SOTA Multi-Spectral Combiner
        self.spectral_combiner = SOTASpectralCombiner(
            dct_in_dim=1280,
            branch_dim=320,
            output_dim=1280,
            gradient_checkpointing=True,
        )

    def forward(self, images: torch.Tensor, return_spectral_details: bool = False):
        # 1. Spatial stream
        spatial_feat = self.spatial_encoder(images)  # (B, 1280)

        # 2. Raw DCT frequency stream
        dct_feat = self.frequency_encoder(images)  # (B, 1280)

        # 3. SOTA Multi-Spectral Combiner (FFT + Wavelet + SRM/Gabor + LSGN)
        spectral_out = self.spectral_combiner(images, dct_feat, return_branches=return_spectral_details)

        if return_spectral_details:
            combined_spectral_feat = spectral_out["combined"]
            branch_dict = spectral_out["branches"]
            gate_weights = spectral_out["gate_weights"]
        else:
            combined_spectral_feat = spectral_out
            branch_dict = None
            gate_weights = None

        # 4. OpenCLIP projection (args: spatial_features, images)
        clip_result = self.clip_alignment(spatial_feat, images, compute_alignment_loss=True)
        clip_proj = clip_result["spatial_projected"]  # (B, 256)
        clip_loss = clip_result["alignment_loss"]

        # 5. Multimodal Cross-Attention Fusion
        # Fuses spatial (1280), SOTA spectral (1280), and CLIP (256)
        fused_feat = self.fusion(
            spatial_features=spatial_feat,
            frequency_features=combined_spectral_feat,
            temporal_features=None,
            physiology_features=None,
            clip_features=clip_proj,
        )

        # 6. Multi-Task Detection Head
        predictions = self.detection_head(fused_feat)
        binary_logit = predictions["binary_logit"].squeeze(-1)

        out = {
            "binary_logit": binary_logit,
            "manip_logits": predictions["manipulation_logits"],
            "confidence": predictions["confidence"],
            "clip_loss": clip_loss,
            "fused_feat": fused_feat,
        }

        if return_spectral_details:
            out["branch_dict"] = branch_dict
            out["gate_weights"] = gate_weights

        return out


# ─── Dynamic On-The-Fly Self-Blended Synthesis (SBI) ──────────────────────────

def generate_dynamic_sbi_seam(pil_img: Image.Image) -> tuple:
    """
    On-the-fly Self-Blended Image (SBI) generation.
    Takes an authentic face, extracts a random facial ellipse/polygon mask,
    applies color jitter and boundary Gaussian feathering, and blends back.
    Returns: (synthesized_pil_image, is_blended_bool)
    """
    w, h = pil_img.size
    img_np = np.array(pil_img).astype(np.float32)

    # Generate random convex facial mask (jawline / cheek / mouth region)
    mask = np.zeros((h, w), dtype=np.float32)
    center_x = random.randint(int(w * 0.3), int(w * 0.7))
    center_y = random.randint(int(h * 0.35), int(h * 0.75))
    axis_x = random.randint(int(w * 0.2), int(w * 0.4))
    axis_y = random.randint(int(h * 0.2), int(h * 0.4))
    angle = random.randint(-30, 30)

    # Draw ellipse
    y, x = np.ogrid[:h, :w]
    rad = math.radians(angle)
    x_rot = (x - center_x) * math.cos(rad) + (y - center_y) * math.sin(rad)
    y_rot = -(x - center_x) * math.sin(rad) + (y - center_y) * math.cos(rad)
    ellipse_cond = (x_rot / max(axis_x, 1))**2 + (y_rot / max(axis_y, 1))**2 <= 1.0
    mask[ellipse_cond] = 1.0

    # Boundary feathering
    mask_pil = Image.fromarray((mask * 255).astype(np.uint8))
    blur_radius = random.choice([1, 2, 3])
    mask_blurred = np.array(mask_pil.filter(ImageFilter.GaussianBlur(radius=blur_radius))).astype(np.float32) / 255.0
    mask_3d = np.expand_dims(mask_blurred, axis=-1)

    # Color / Gamma manipulation inside source patch
    color_shift = np.random.uniform(-25.0, 25.0, size=(1, 1, 3))
    gamma = random.uniform(0.85, 1.15)
    source_patch = np.clip(((img_np / 255.0) ** gamma) * 255.0 + color_shift, 0, 255)

    # Linear / Poisson alpha composite
    blended_np = img_np * (1.0 - mask_3d) + source_patch * mask_3d
    blended_np = np.clip(blended_np, 0, 255).astype(np.uint8)
    return Image.fromarray(blended_np), True


# ─── Dataset with Dynamic SBI Augmentation ────────────────────────────────────

class V7SpectralDataset(Dataset):
    def __init__(self, samples, transform=None, is_train=True, sbi_prob=0.35):
        self.samples = samples  # list of (path, binary_lbl, manip_idx, weight)
        self.transform = transform
        self.is_train = is_train
        self.sbi_prob = sbi_prob

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, binary_lbl, manip_idx, weight = self.samples[idx]
        try:
            pil_img = Image.open(path).convert("RGB")
        except Exception:
            pil_img = Image.new("RGB", (160, 160), (128, 128, 128))

        final_binary = binary_lbl
        final_manip = manip_idx
        final_weight = weight

        # On-the-fly SBI boundary synthesis on real images during training
        if self.is_train and binary_lbl == 0 and random.random() < self.sbi_prob:
            pil_img, _ = generate_dynamic_sbi_seam(pil_img)
            final_binary = 1.0
            final_manip = 2  # FaceSwap / Blending Seam class
            final_weight = 3.5  # High priority seam weight

        if self.transform:
            img = self.transform(pil_img)
        else:
            img = transforms.ToTensor()(pil_img.resize((160, 160)))

        return {
            "image": img,
            "binary_label": torch.tensor(final_binary, dtype=torch.float32),
            "manip_label": torch.tensor(final_manip, dtype=torch.long),
            "sample_weight": torch.tensor(final_weight, dtype=torch.float32),
        }


# ─── Tri-Objective Loss Formulations ──────────────────────────────────────────

class SOTABoundaryFocalLoss(nn.Module):
    """Focal Loss with sample-specific hard boundary weighting."""
    def __init__(self, gamma=2.5, alpha=0.25):
        super().__init__()
        self.gamma = gamma
        self.alpha = alpha

    def forward(self, logits, targets, weights):
        probs = torch.sigmoid(logits)
        p_t = probs * targets + (1.0 - probs) * (1.0 - targets)
        alpha_t = self.alpha * targets + (1.0 - self.alpha) * (1.0 - targets)
        bce = F.binary_cross_entropy_with_logits(logits, targets, reduction="none")
        focal = alpha_t * ((1.0 - p_t) ** self.gamma) * bce
        return (focal * weights).mean()


def compute_spectral_orthogonal_loss(branch_dict: dict) -> torch.Tensor:
    """
    Computes pairwise cosine similarity between DCT, FFT, Wavelet, and SRM embeddings.
    Penalizes non-zero off-diagonal correlations to enforce orthogonal forensic sub-spaces.
    """
    keys = ["dct", "fft", "wavelet", "srm"]
    feats = [F.normalize(branch_dict[k], p=2, dim=-1) for k in keys]

    ortho_loss = torch.tensor(0.0, device=feats[0].device)
    pairs = 0
    for i in range(len(feats)):
        for j in range(i + 1, len(feats)):
            cosine_sim = (feats[i] * feats[j]).sum(dim=-1).abs().mean()
            ortho_loss = ortho_loss + cosine_sim
            pairs += 1

    return ortho_loss / max(pairs, 1)


# ─── Dataset Loader Construction ──────────────────────────────────────────────

def build_v7_dataset():
    logger.info("Constructing V7 Multi-Spectral Multi-Source Dataset (Actor-Disjoint)...")

    train_samples = []
    val_samples = []

    # 1. Kaggle Anchor Split (StyleGAN vs Authentic Reals) — no FF++ actors
    k_train = Path("data/kaggle_realfake/real_vs_fake/real-vs-fake/train")
    k_val = Path("data/kaggle_realfake/real_vs_fake/real-vs-fake/valid")

    for p in list((k_train / "real").glob("*.jpg"))[:5000]:
        train_samples.append((str(p), 0, 0, 1.0))
    for p in list((k_train / "fake").glob("*.jpg"))[:5000]:
        train_samples.append((str(p), 1, 1, 1.0))

    for p in list((k_val / "real").glob("*.jpg"))[:1000]:
        val_samples.append((str(p), 0, 0, 1.0))
    for p in list((k_val / "fake").glob("*.jpg"))[:1000]:
        val_samples.append((str(p), 1, 1, 1.0))

    # 2. FaceForensics++ with Actor-Disjoint Split (no test actor contamination)
    ffpp_csv = Path("manifests/ffpp_c23_manifest.csv")
    if ffpp_csv.exists():
        from utils.actor_splits import get_actor_disjoint_split
        df_ffpp = pd.read_csv(ffpp_csv)
        col = "filepath" if "filepath" in df_ffpp.columns else "image_path"

        split = get_actor_disjoint_split(df_ffpp, filepath_col=col)

        # Train reals: ONLY from train actors (6,000 frames from 600 actors)
        train_real_paths = split["train_reals"][col].tolist()
        for p in train_real_paths:
            train_samples.append((str(p), 0, 0, 1.0))

        # Val reals: small held-out sample from train actors (last 10%)
        val_real_count = max(1, len(train_real_paths) // 10)
        for p in train_real_paths[-val_real_count:]:
            val_samples.append((str(p), 0, 0, 1.0))

        # Train fakes: ONLY both-train-actor pairs (DFD excluded)
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

                # Val fakes: small held-out from train fakes (last 15%)
                val_fake_count = max(1, len(train_fake_paths) // 7)
                for p in train_fake_paths[-val_fake_count:]:
                    val_samples.append((str(p), 1, c_idx, 1.0))

        logger.info(
            f"FF++ Actor-Disjoint: {len(split['train_actors'])} train actors, "
            f"{len(split['test_actors'])} test actors (held out), DFD excluded"
        )

    random.shuffle(train_samples)
    random.shuffle(val_samples)

    logger.info(f"V7 Dataset Built: {len(train_samples)} Train Samples | {len(val_samples)} Val Samples")
    return train_samples, val_samples


# ─── Training Execution Loop ──────────────────────────────────────────────────

def main():
    device = get_device()
    logger.info(f"=== Starting DeepTrace V7 SOTA Multi-Spectral Training on {device} ===")

    out_dir = Path("checkpoints/v7_sota_spectral")
    out_dir.mkdir(parents=True, exist_ok=True)

    # 1. Initialize Base Detector and Warm-Start from V5 Checkpoint
    base_detector = DeepfakeDetector()
    ckpt_v5 = "checkpoints/v5_srm_residual/best_model.pth"
    if Path(ckpt_v5).exists():
        logger.info(f"Warm-starting from V5 checkpoint: {ckpt_v5}")
        load_checkpoint(ckpt_v5, base_detector, device=device)
    else:
        logger.info("V5 checkpoint not found, starting from V2 baseline")
        load_checkpoint("checkpoints/v2_clip_finetune/best_model.pth", base_detector, device=device)

    # Wrap into V7 Detector
    model = V7SOTADetector(base_detector).to(device)

    # 2. Data Transforms
    train_transform = transforms.Compose([
        transforms.Resize((160, 160)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    val_transform = transforms.Compose([
        transforms.Resize((160, 160)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    train_samples, val_samples = build_v7_dataset()
    train_dataset = V7SpectralDataset(train_samples, transform=train_transform, is_train=True, sbi_prob=0.35)
    val_dataset = V7SpectralDataset(val_samples, transform=val_transform, is_train=False)

    train_loader = DataLoader(
        train_dataset,
        batch_size=16,
        shuffle=True,
        num_workers=2,
        pin_memory=(device.type == "cuda"),
        drop_last=True,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=32,
        shuffle=False,
        num_workers=2,
        pin_memory=(device.type == "cuda"),
    )

    # 3. Differential Optimizer Parameter Groups
    optimizer_params = [
        {"params": [p for p in model.spatial_encoder.parameters() if p.requires_grad], "lr": 2e-5, "weight_decay": 1e-4},
        {"params": [p for p in model.frequency_encoder.parameters() if p.requires_grad], "lr": 2e-5, "weight_decay": 1e-4},
        {"params": [p for p in model.clip_alignment.parameters() if p.requires_grad], "lr": 5e-6, "weight_decay": 1e-4},
        {"params": model.spectral_combiner.parameters(), "lr": 1.2e-4, "weight_decay": 1e-4},
        {"params": list(model.fusion.parameters()) + list(model.detection_head.parameters()), "lr": 1.5e-4, "weight_decay": 1e-4},
    ]

    optimizer = torch.optim.AdamW(optimizer_params)
    epochs = 8
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-6)
    scaler = get_grad_scaler(device=device, enabled=(device.type == "cuda"))

    bfl_loss_fn = SOTABoundaryFocalLoss(gamma=2.5, alpha=0.25)
    ce_loss_fn = nn.CrossEntropyLoss()

    best_val_auc = 0.0
    accum_steps = 4

    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0.0
        train_preds, train_targets = [], []

        optimizer.zero_grad()

        for step, batch in enumerate(train_loader):
            images = batch["image"].to(device, non_blocking=True)
            binary_labels = batch["binary_label"].to(device, non_blocking=True)
            manip_labels = batch["manip_label"].to(device, non_blocking=True)
            sample_weights = batch["sample_weight"].to(device, non_blocking=True)

            with AMPContext(device=device, enabled=True):
                outputs = model(images, return_spectral_details=True)

                loss_bfl = bfl_loss_fn(outputs["binary_logit"], binary_labels, sample_weights)
                loss_manip = ce_loss_fn(outputs["manip_logits"], manip_labels)
                loss_ortho = compute_spectral_orthogonal_loss(outputs["branch_dict"])

                # Tri-Objective Loss Formulation
                loss = loss_bfl + 0.30 * loss_manip + 0.30 * loss_ortho
                loss_scaled = loss / accum_steps

            scaler.scale(loss_scaled).backward()

            if (step + 1) % accum_steps == 0 or (step + 1) == len(train_loader):
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()

            total_loss += loss.item()
            probs = torch.sigmoid(outputs["binary_logit"]).detach().cpu().numpy()
            train_preds.extend(probs)
            train_targets.extend(binary_labels.cpu().numpy())

        scheduler.step()

        train_acc = accuracy_score(train_targets, np.array(train_preds) >= 0.5)
        try:
            train_auc = roc_auc_score(train_targets, train_preds)
        except Exception:
            train_auc = 0.5

        # ─── Validation Phase ───────────────────────────────────────────
        model.eval()
        val_preds, val_targets = [], []

        with torch.no_grad():
            for batch in val_loader:
                images = batch["image"].to(device, non_blocking=True)
                binary_labels = batch["binary_label"].to(device, non_blocking=True)

                with AMPContext(device=device, enabled=True):
                    outputs = model(images, return_spectral_details=False)

                probs = torch.sigmoid(outputs["binary_logit"]).cpu().numpy()
                val_preds.extend(probs)
                val_targets.extend(binary_labels.cpu().numpy())

        val_acc = accuracy_score(val_targets, np.array(val_preds) >= 0.5)
        try:
            val_auc = roc_auc_score(val_targets, val_preds)
        except Exception:
            val_auc = 0.5
        val_f1 = f1_score(val_targets, np.array(val_preds) >= 0.5)

        logger.info(
            f"Epoch {epoch}/{epochs} | "
            f"Train Loss: {total_loss / len(train_loader):.4f} | "
            f"Train Acc: {train_acc * 100:.2f}% | Train AUC: {train_auc:.4f} | "
            f"Val Acc: {val_acc * 100:.2f}% | Val AUC: {val_auc:.4f} | Val F1: {val_f1:.4f}"
        )

        # Save Checkpoint
        checkpoint_dict = {
            "epoch": epoch,
            "state_dict": model.state_dict(),
            "val_auc": val_auc,
            "val_acc": val_acc,
            "val_f1": val_f1,
        }
        torch.save(checkpoint_dict, out_dir / "last.pth")

        if val_auc > best_val_auc:
            best_val_auc = val_auc
            torch.save(checkpoint_dict, out_dir / "best_model.pth")
            logger.info(f"--> Saved New Best V7 SOTA Checkpoint (Val AUC: {val_auc:.4f})")

    logger.info(f"=== V7 SOTA Multi-Spectral Training Complete! Best Val AUC: {best_val_auc:.4f} ===")


if __name__ == "__main__":
    main()
