"""
Kaggle Real-vs-Fake image dataset loader.
Expected structure:
    data/kaggle_realfake/real_vs_fake/real-vs-fake/
        train/
            real/*.jpg
            fake/*.jpg
        valid/
            real/*.jpg
            fake/*.jpg
        test/
            real/*.jpg
            fake/*.jpg
"""

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset
from pathlib import Path
import logging

from datasets.transforms import get_train_transforms, get_val_transforms, apply_dct_transform

logger = logging.getLogger(__name__)


class KaggleRealFakeDataset(Dataset):
    """Simple image-based real/fake dataset from Kaggle."""

    MANIPULATION_TYPES = {
        "real": 0,
        "Deepfakes": 1,
    }

    # generator_type labels (used in Phase 1 GAN finetune + V2 multi-task)
    # -1 = real (ignored in loss), 0 = StyleGAN, 1 = Diffusion, 2 = Other GAN, 3 = Unknown fake
    GENERATOR_TYPES = {
        "real": -1,
        "stylegan": 0,
        "diffusion": 1,
        "other_gan": 2,
        "unknown": 3,
    }

    def __init__(
        self,
        root_dir: str,
        split: str = "train",
        image_size: int = 160,
        extra_fake_dirs: list = None,  # additional fake image dirs (e.g. D: GAN downloads)
        **kwargs,
    ):
        self.root_dir = Path(root_dir)
        self.split = split
        self.image_size = image_size
        self.extra_fake_dirs = [Path(d) for d in (extra_fake_dirs or [])]

        # Map split names: our "val" → folder "valid"
        folder_name = "valid" if split == "val" else split

        # Try to find the images directory
        # Structure: root_dir/real_vs_fake/real-vs-fake/{train,valid,test}/{real,fake}/
        img_root = self.root_dir / "real_vs_fake" / "real-vs-fake" / folder_name
        if not img_root.exists():
            # Try without the nested real_vs_fake
            img_root = self.root_dir / folder_name
        if not img_root.exists():
            logger.warning(f"Dataset path not found: {img_root}")
            self.samples = []
            return

        # Feature cache directory (precomputed frozen encoder outputs)
        self.cache_dir = self.root_dir / "feature_cache"
        self.use_cache = self.cache_dir.exists()
        if self.use_cache:
            logger.info(f"Feature cache found at {self.cache_dir} — skipping frozen encoders")
        else:
            logger.info("No feature cache found — encoders will run normally")

        # Set transforms
        if split == "train":
            self.transform = get_train_transforms(image_size)
        else:
            self.transform = get_val_transforms(image_size)

        # Load samples
        self.samples = []
        for label_name, label_id in [("real", 0), ("fake", 1)]:
            label_dir = img_root / label_name
            if not label_dir.exists():
                logger.warning(f"Missing folder: {label_dir}")
                continue
            for img_path in sorted(label_dir.glob("*")):
                if img_path.suffix.lower() in (".jpg", ".jpeg", ".png", ".bmp", ".webp"):
                    # Infer generator_type from filename
                    stem = img_path.stem.lower()
                    if label_id == 0:
                        gen_type = -1  # real
                    elif stem.startswith("stylegan"):
                        gen_type = 0  # StyleGAN
                    elif any(k in stem for k in ("diff", "ldm", "sd_", "stable")):
                        gen_type = 1  # Diffusion
                    elif any(k in stem for k in ("gan", "progan", "dcgan")):
                        gen_type = 2  # Other GAN
                    else:
                        gen_type = 3  # Unknown fake
                    self.samples.append({
                        "path": str(img_path),
                        "label": label_id,
                        "manipulation_type": "Deepfakes" if label_id == 1 else "real",
                        "generator_type": gen_type,
                    })

        logger.info(
            f"KaggleRealFakeDataset [{split}]: {len(self.samples)} images "
            f"({sum(1 for s in self.samples if s['label']==0)} real, "
            f"{sum(1 for s in self.samples if s['label']==1)} fake)"
        )

        # Load extra GAN images from additional dirs (e.g. D: drive downloads)
        # Only applied to train split — val/test use only the main dataset
        if self.extra_fake_dirs and split == "train":
            extra_count = 0
            for extra_dir in self.extra_fake_dirs:
                if not extra_dir.exists():
                    logger.warning(f"extra_fake_dir not found: {extra_dir}")
                    continue
                for img_path in sorted(extra_dir.glob("*")):
                    if img_path.suffix.lower() in (".jpg", ".jpeg", ".png", ".bmp", ".webp"):
                        stem = img_path.stem.lower()
                        if stem.startswith("stylegan"):
                            gen_type = 0
                        elif any(k in stem for k in ("diff", "ldm", "sd_", "stable")):
                            gen_type = 1
                        elif any(k in stem for k in ("gan", "progan", "dcgan")):
                            gen_type = 2
                        else:
                            gen_type = 3
                        self.samples.append({
                            "path": str(img_path),
                            "label": 1,
                            "manipulation_type": "Deepfakes",
                            "generator_type": gen_type,
                        })
                        extra_count += 1
            logger.info(f"  + {extra_count} extra GAN images from {len(self.extra_fake_dirs)} extra dir(s)")


    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx: int) -> dict:
        sample = self.samples[idx]
        label = sample["label"]
        manip_type = self.MANIPULATION_TYPES.get(sample["manipulation_type"], 1)

        # Read image
        image = cv2.imread(sample["path"])
        if image is None:
            image = np.zeros((self.image_size, self.image_size, 3), dtype=np.uint8)

        # Resize
        face = cv2.resize(image, (self.image_size, self.image_size))

        # DCT for frequency encoder
        dct = apply_dct_transform(face)

        # RGB + augmentation
        rgb = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
        augmented = self.transform(image=rgb)
        tensor = augmented["image"]

        # DCT tensor
        dct_normalized = (dct - dct.mean()) / (dct.std() + 1e-8)
        dct_tensor = torch.from_numpy(dct_normalized).permute(2, 0, 1).float()
        dct_tensor = torch.nn.functional.interpolate(
            dct_tensor.unsqueeze(0), size=(self.image_size, self.image_size), mode="bilinear",
            align_corners=False,
        ).squeeze(0)

        return {
            "image": tensor,
            "dct": dct_tensor,
            "label": torch.tensor(label, dtype=torch.float32),
            "manipulation_type": torch.tensor(manip_type, dtype=torch.long),
            "generator_type": torch.tensor(sample["generator_type"], dtype=torch.long),
            "path": sample["path"],
            "cached_freq": self._load_cache(sample["path"], "freq"),
            "cached_clip": self._load_cache(sample["path"], "clip"),
        }

    def _load_cache(self, img_path: str, key: str):
        """Load a cached feature tensor if available, else return None."""
        if not self.use_cache:
            return None
        import numpy as np
        stem = Path(img_path).stem
        cache_file = self.cache_dir / f"{stem}_{key}.npy"
        if cache_file.exists():
            return torch.from_numpy(np.load(str(cache_file)))
        return None
