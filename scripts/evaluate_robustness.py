"""
DeepTrace V10: Comprehensive Robustness Perturbation Evaluation Engine.
Evaluates model resilience under real-world social media and transmission distortions:
1. Multi-level JPEG Compression (Q=30, 50, 70, 90)
2. Gaussian Defocus Blur (kernel=3, 5, 7)
3. Zero-mean Gaussian Noise (sigma=0.05, 0.15, 0.25)
4. Multi-scale Downscale-Upscale (scale=0.5, 0.7, 0.9)

Saves detailed results to results/benchmark_eval_v10/robustness_results.json
"""

import sys
import json
import logging
import argparse
from pathlib import Path
import numpy as np
from PIL import Image
import cv2
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import roc_auc_score, accuracy_score, brier_score_loss

sys.path.insert(0, str(Path(__file__).parent.parent))

from models.detector import DeepfakeDetector
from models.spectral_branches import SOTASpectralCombiner
from utils.checkpoint import load_checkpoint
from utils.device import get_device, AMPContext
from utils.actor_splits import get_actor_disjoint_split
from datasets.transforms import get_val_transforms
import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("robustness_eval")


class V10EvalDetector(nn.Module):
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
            gradient_checkpointing=False,
        )

    def forward(self, images: torch.Tensor):
        spatial_feat = self.spatial_encoder(images)
        dct_feat = self.frequency_encoder(images)
        combined_spectral_feat = self.spectral_combiner(images, dct_feat, return_branches=False)

        clip_result = self.clip_alignment(spatial_feat, images, compute_alignment_loss=False)
        clip_proj = clip_result["spatial_projected"]

        fused_feat = self.fusion(
            spatial_features=spatial_feat,
            frequency_features=combined_spectral_feat,
            temporal_features=None,
            physiology_features=None,
            clip_features=clip_proj,
        )

        predictions = self.detection_head(fused_feat)
        return predictions["binary_logit"].squeeze(-1)


def apply_jpeg(img_np: np.ndarray, quality: int) -> np.ndarray:
    encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), quality]
    _, enc = cv2.imencode(".jpg", cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR), encode_param)
    dec = cv2.imdecode(enc, cv2.IMREAD_COLOR)
    return cv2.cvtColor(dec, cv2.COLOR_BGR2RGB)


def apply_blur(img_np: np.ndarray, k: int) -> np.ndarray:
    return cv2.GaussianBlur(img_np, (k, k), 0)


def apply_noise(img_np: np.ndarray, sigma: float) -> np.ndarray:
    noise = np.random.normal(0, sigma * 255.0, img_np.shape).astype(np.float32)
    noisy = np.clip(img_np.astype(np.float32) + noise, 0, 255).astype(np.uint8)
    return noisy


def apply_downscale(img_np: np.ndarray, scale: float) -> np.ndarray:
    h, w, _ = img_np.shape
    small = cv2.resize(img_np, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_LINEAR)
    return cv2.resize(small, (w, h), interpolation=cv2.INTER_LINEAR)


class PerturbedDataset(Dataset):
    def __init__(self, samples, perturbation_fn, base_transform):
        self.samples = samples
        self.perturbation_fn = perturbation_fn
        self.base_transform = base_transform

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        try:
            img = Image.open(path).convert("RGB")
            img_np = np.array(img)
        except Exception:
            img_np = np.full((160, 160, 3), 128, dtype=np.uint8)

        if self.perturbation_fn is not None:
            img_np = self.perturbation_fn(img_np)

        augmented = self.base_transform(image=img_np)
        return {
            "image": augmented["image"],
            "label": torch.tensor(label, dtype=torch.float32),
        }


def evaluate_perturbation(model, loader, device):
    model.eval()
    all_preds, all_labels = [], []
    with torch.no_grad():
        for batch in loader:
            imgs = batch["image"].to(device, non_blocking=True)
            labels = batch["label"].numpy()
            with AMPContext(device=device, enabled=True):
                logits = model(imgs)
                preds = np.nan_to_num(logits.float().cpu().numpy(), nan=0.0)
            all_preds.extend(preds)
            all_labels.extend(labels)

    preds_arr = np.nan_to_num(np.array(all_preds), nan=0.0)
    probs = 1.0 / (1.0 + np.exp(-np.clip(preds_arr, -20.0, 20.0)))
    probs = np.nan_to_num(probs, nan=0.5)

    auc = float(roc_auc_score(all_labels, probs))
    acc = float(accuracy_score(all_labels, probs >= 0.5))
    brier = float(brier_score_loss(all_labels, probs))
    return {"auc": auc, "accuracy": acc, "brier_score": brier, "count": len(all_labels)}


def main():
    parser = argparse.ArgumentParser(description="Evaluate V10 Robustness")
    parser.add_argument("--checkpoint", type=str, default="checkpoints/v10_robust/best_model.pth")
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--out_json", type=str, default="results/benchmark_eval_v10/robustness_results.json")
    args = parser.parse_args()

    device = get_device()
    logger.info(f"=== Starting Robustness Battery on {device} using {args.checkpoint} ===")

    base_detector = DeepfakeDetector()
    model = V10EvalDetector(base_detector)
    load_checkpoint(args.checkpoint, model, device=device)
    model.to(device)

    # 1. Build test samples (Kaggle Test + FF++ Actor-Disjoint Test)
    test_samples = []
    k_test = Path("data/kaggle_realfake/real_vs_fake/real-vs-fake/test")
    if k_test.exists():
        for p in list((k_test / "real").glob("*.jpg"))[:1000]:
            test_samples.append((str(p), 0))
        for p in list((k_test / "fake").glob("*.jpg"))[:1000]:
            test_samples.append((str(p), 1))

    ffpp_csv = Path("manifests/ffpp_c23_manifest.csv")
    if ffpp_csv.exists():
        df_ffpp = pd.read_csv(ffpp_csv)
        col = "filepath" if "filepath" in df_ffpp.columns else "image_path"
        split = get_actor_disjoint_split(df_ffpp, filepath_col=col)
        for p in split["test_reals"][col].tolist()[:1000]:
            test_samples.append((str(p), 0))
        for manip in ["Deepfakes", "FaceSwap", "FaceShifter", "Face2Face", "NeuralTextures"]:
            if manip in split["test_fakes"]:
                for p in split["test_fakes"][manip][col].tolist()[:200]:
                    test_samples.append((str(p), 1))

    logger.info(f"Robustness test pool: {len(test_samples)} samples (Balanced In-Domain + FF++ Cross-Domain)")

    base_transform = get_val_transforms(image_size=160)

    perturbation_suite = {
        "pristine": None,
        "jpeg_q90": lambda img: apply_jpeg(img, 90),
        "jpeg_q70": lambda img: apply_jpeg(img, 70),
        "jpeg_q50": lambda img: apply_jpeg(img, 50),
        "jpeg_q30": lambda img: apply_jpeg(img, 30),
        "blur_k3": lambda img: apply_blur(img, 3),
        "blur_k5": lambda img: apply_blur(img, 5),
        "blur_k7": lambda img: apply_blur(img, 7),
        "noise_s005": lambda img: apply_noise(img, 0.05),
        "noise_s015": lambda img: apply_noise(img, 0.15),
        "noise_s025": lambda img: apply_noise(img, 0.25),
        "downscale_s09": lambda img: apply_downscale(img, 0.9),
        "downscale_s07": lambda img: apply_downscale(img, 0.7),
        "downscale_s05": lambda img: apply_downscale(img, 0.5),
    }

    results = {}
    for name, p_fn in perturbation_suite.items():
        ds = PerturbedDataset(test_samples, p_fn, base_transform)
        loader = DataLoader(ds, batch_size=args.batch_size, shuffle=False, num_workers=0)
        metrics = evaluate_perturbation(model, loader, device)
        results[name] = metrics
        logger.info(f"[{name.upper():14s}] AUC: {metrics['auc']:.4f} | Acc: {metrics['accuracy']*100:.2f}% | Brier: {metrics['brier_score']:.4f}")

    out_path = Path(args.out_json)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)

    logger.info(f"=== Robustness Battery Complete! Results saved to {out_path} ===")


if __name__ == "__main__":
    main()
