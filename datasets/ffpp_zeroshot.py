"""
FF++ Zero-Shot Evaluation Loader.

For evaluating a model trained ONLY on Kaggle against FF++ test data.
This does NOT use the original FaceForensicsDataset (which has invalid split logic).

Expected input: Pre-extracted face crops from a Kaggle dataset like
  fatmairshad/faceforensics-extracted-dataset-c23

Directory structure (auto-detected):
  root/
    Original/ or original/ or real/
    Deepfakes/
    Face2Face/
    FaceSwap/
    NeuralTextures/
    FaceShifter/ (optional)

Unit of evaluation: Per-image for extracted crops.
If original videos are available, video-level aggregation should be done
at the evaluation script level, not here.
"""

import os
import logging
from pathlib import Path
from typing import Dict, List, Tuple

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset

from datasets.transforms import get_val_transforms, apply_dct_transform

logger = logging.getLogger(__name__)

# Canonical manipulation method names (case-insensitive matching)
REAL_NAMES = {"original", "real", "youtube"}
FAKE_METHODS = {"deepfakes", "face2face", "faceswap", "neuraltextures", "faceshifter"}
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


class FFppZeroShotDataset(Dataset):
    """
    Load FF++ pre-extracted face crops for zero-shot evaluation.

    Each sample includes:
    - image tensor (RGB, normalized)
    - dct tensor (DCT features, normalized)
    - label (0=real, 1=fake)
    - manipulation_type (str: "original", "Deepfakes", etc.)
    - source_path (str: original file path)

    Deduplication: Uses resolved paths to avoid Windows case-insensitive doubles.
    """

    def __init__(self, root_dir: str, image_size: int = 160, max_per_method: int = None):
        """
        Args:
            root_dir: Path to FF++ extracted faces root
            image_size: Resize target (default 160 to match Kaggle pipeline)
            max_per_method: Optional cap per manipulation method (for quick testing)
        """
        self.root = Path(root_dir)
        self.image_size = image_size
        self.transform = get_val_transforms(image_size)
        self.samples: List[Dict] = []

        self._discover_samples(max_per_method)
        logger.info(f"FFppZeroShotDataset: {len(self.samples)} images from {root_dir}")
        self._log_class_balance()

    def _discover_samples(self, max_per_method: int = None):
        """Enumerate on-disk directories, match to known method names."""
        seen_paths = set()

        if not self.root.exists():
            logger.warning(f"FF++ root does not exist: {self.root}")
            return

        for entry in os.listdir(self.root):
            entry_lower = entry.lower()
            entry_path = self.root / entry

            if not entry_path.is_dir():
                continue

            # Determine label and method
            if entry_lower in REAL_NAMES:
                label = 0
                method = "original"
            elif entry_lower in FAKE_METHODS:
                label = 1
                # Preserve canonical casing for reporting
                method = next((m for m in ["Deepfakes", "Face2Face", "FaceSwap",
                               "NeuralTextures", "FaceShifter"]
                               if m.lower() == entry_lower), entry)
            else:
                logger.info(f"Skipping unknown directory: {entry}")
                continue

            count = 0
            for f in sorted(entry_path.rglob("*")):
                if f.suffix.lower() not in IMAGE_EXTS:
                    continue

                resolved = str(f.resolve())
                if resolved in seen_paths:
                    continue
                seen_paths.add(resolved)

                self.samples.append({
                    "path": str(f),
                    "label": label,
                    "manipulation_type": method,
                })
                count += 1

                if max_per_method and count >= max_per_method:
                    break

            logger.info(f"  {method}: {count} images (label={label})")

    def _log_class_balance(self):
        """Log per-method sample counts."""
        from collections import Counter
        methods = Counter(s["manipulation_type"] for s in self.samples)
        total_real = sum(v for k, v in methods.items() if k == "original")
        total_fake = sum(v for k, v in methods.items() if k != "original")
        logger.info(f"  Total: {len(self.samples)} ({total_real} real, {total_fake} fake)")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        path = sample["path"]

        img = cv2.imread(path)
        if img is None:
            img = np.zeros((self.image_size, self.image_size, 3), dtype=np.uint8)

        # Unified preprocessing: resize → RGB → DCT (same as Kaggle pipeline)
        img = cv2.resize(img, (self.image_size, self.image_size))
        rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # DCT transform (same as GenericFolderDataset and Kaggle pipeline)
        dct = apply_dct_transform(img)
        dct_norm = (dct - dct.mean()) / (dct.std() + 1e-8)
        dct_t = torch.from_numpy(dct_norm).permute(2, 0, 1).float()
        dct_t = F.interpolate(
            dct_t.unsqueeze(0),
            size=(self.image_size, self.image_size),
            mode="bilinear",
            align_corners=False,
        ).squeeze(0)

        aug = self.transform(image=rgb)

        return {
            "image": aug["image"],
            "dct": dct_t,
            "label": torch.tensor(sample["label"], dtype=torch.float32),
            "manipulation_type": sample["manipulation_type"],
            "path": path,
        }
