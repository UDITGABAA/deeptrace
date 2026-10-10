"""
DFDC (DeepFake Detection Challenge) dataset loader.
Expected structure:
    data/DFDC/
        dfdc_train_part_0/
            *.mp4
            metadata.json    # {"filename.mp4": {"label": "FAKE"/"REAL", ...}}
        dfdc_train_part_1/
            ...
        dfdc_train_part_49/
            ...
"""

import json
from pathlib import Path
import logging
from datasets.base_dataset import BaseDeepfakeDataset

logger = logging.getLogger(__name__)


class DFDCDataset(BaseDeepfakeDataset):
    """DFDC dataset."""

    def __init__(self, root_dir: str, parts: list = None, **kwargs):
        """
        Args:
            parts: List of part indices to load, e.g. [0,1,2]. None = all available.
        """
        self.parts = parts
        super().__init__(root_dir, **kwargs)

    def _load_samples(self) -> list:
        samples = []
        dfdc_root = self.root_dir / "DFDC"

        # Find all part directories
        part_dirs = sorted(dfdc_root.glob("dfdc_train_part_*"))
        if self.parts is not None:
            part_dirs = [d for d in part_dirs if int(d.name.split("_")[-1]) in self.parts]

        for part_dir in part_dirs:
            metadata_path = part_dir / "metadata.json"
            if not metadata_path.exists():
                logger.warning(f"No metadata.json in {part_dir}, skipping")
                continue

            with open(metadata_path, "r") as f:
                metadata = json.load(f)

            for filename, info in metadata.items():
                vid_path = part_dir / filename
                if not vid_path.exists():
                    continue

                label_str = info.get("label", "REAL").upper()
                samples.append({
                    "path": str(vid_path),
                    "label": 1 if label_str == "FAKE" else 0,
                    "manipulation_type": "Deepfakes" if label_str == "FAKE" else "real",
                })

        # Split
        samples = self._apply_split(samples)
        return samples

    def _apply_split(self, samples: list) -> list:
        """Apply train/val/test split (80/10/10)."""
        n = len(samples)
        if self.split == "train":
            return samples[: int(n * 0.8)]
        elif self.split == "val":
            return samples[int(n * 0.8) : int(n * 0.9)]
        else:
            return samples[int(n * 0.9) :]
