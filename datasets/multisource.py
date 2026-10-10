"""
Multi-Source Balanced Dataset Loader for Phase 3 Training.
Combines Kaggle Real-vs-Fake with FaceForensics++ (FF++) manipulations.

Target Balanced Batch (e.g. Batch Size 64):
  - 16 samples: Kaggle Real (FFHQ)
  - 16 samples: Kaggle Fake (StyleGAN)
  - 16 samples: FF++ Real (Original pristine video frames)
  - 16 samples: FF++ Fake (4 Deepfakes, 4 Face2Face, 4 FaceSwap, 4 NeuralTextures)
"""

import random
from pathlib import Path
import cv2
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, Sampler

from datasets.transforms import get_train_transforms, get_val_transforms, apply_dct_transform


MANIPULATION_MAP = {
    "real": 0,
    "StyleGAN": 1,
    "Deepfakes": 2,
    "Face2Face": 3,
    "FaceSwap": 4,
    "NeuralTextures": 5,
}


class MultiSourceDataset(Dataset):
    """Dataset providing RGB and DCT tensors with binary and manipulation-type labels."""

    def __init__(self, image_records, image_size=160, is_train=True):
        """
        image_records: List of tuples (path, binary_label, manip_type_str)
        """
        self.records = image_records
        self.image_size = image_size
        self.is_train = is_train

        if is_train:
            self.transform = get_train_transforms(image_size)
        else:
            self.transform = get_val_transforms(image_size)

    def __len__(self):
        return len(self.records)

    def __getitem__(self, idx):
        img_path, binary_label, manip_name = self.records[idx]

        img = cv2.imread(str(img_path))
        if img is None:
            # Fallback to zeros if corrupted
            img = np.zeros((self.image_size, self.image_size, 3), dtype=np.uint8)

        img_resized = cv2.resize(img, (self.image_size, self.image_size))
        rgb = cv2.cvtColor(img_resized, cv2.COLOR_BGR2RGB)

        # Apply albumentations
        aug = self.transform(image=rgb)
        rgb_tensor = aug["image"]

        # Compute DCT tensor
        dct = apply_dct_transform(img_resized)
        dct_norm = (dct - dct.mean()) / (dct.std() + 1e-8)
        dct_tensor = torch.from_numpy(dct_norm).permute(2, 0, 1).float()
        dct_tensor = F.interpolate(
            dct_tensor.unsqueeze(0),
            size=(self.image_size, self.image_size),
            mode="bilinear",
            align_corners=False,
        ).squeeze(0)

        manip_label = MANIPULATION_MAP.get(manip_name, 0)

        return {
            "image": rgb_tensor,
            "dct": dct_tensor,
            "label": torch.tensor(binary_label, dtype=torch.float32),
            "manip_label": torch.tensor(manip_label, dtype=torch.long),
            "manip_name": manip_name,
            "path": str(img_path),
        }
