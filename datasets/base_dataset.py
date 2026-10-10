"""
Abstract base dataset for deepfake detection.
Handles face extraction, frame sampling, and preprocessing.
"""

import torch
from torch.utils.data import Dataset
import cv2
import numpy as np
from pathlib import Path
from abc import ABC, abstractmethod
from typing import Optional, Tuple, List
import logging

from datasets.transforms import get_train_transforms, get_val_transforms, apply_dct_transform

logger = logging.getLogger(__name__)


class BaseDeepfakeDataset(Dataset, ABC):
    """
    Abstract base class for deepfake detection datasets.

    Subclasses must implement:
        _load_samples() -> list of (path, label, manipulation_type)
    """

    MANIPULATION_TYPES = {
        "real": 0,
        "Deepfakes": 1,
        "Face2Face": 2,
        "FaceSwap": 3,
        "NeuralTextures": 4,
    }

    def __init__(
        self,
        root_dir: str,
        split: str = "train",
        image_size: int = 160,
        num_frames: int = 8,
        mode: str = "image",  # "image" or "video"
        face_margin: float = 0.3,
    ):
        self.root_dir = Path(root_dir)
        self.split = split
        self.image_size = image_size
        self.num_frames = num_frames
        self.mode = mode
        self.face_margin = face_margin

        # Set transforms
        if split == "train":
            self.transform = get_train_transforms(image_size)
        else:
            self.transform = get_val_transforms(image_size)

        # Load samples
        self.samples = self._load_samples()
        logger.info(f"Loaded {len(self.samples)} {split} samples from {self.__class__.__name__}")

        # Face detector (lazy init)
        self._face_detector = None

    @abstractmethod
    def _load_samples(self) -> list:
        """
        Load dataset samples.
        Returns: list of dicts with keys: path, label (0=real, 1=fake), manipulation_type (str)
        """
        raise NotImplementedError

    @property
    def face_detector(self):
        """Lazy-init MTCNN face detector."""
        if self._face_detector is None:
            try:
                from facenet_pytorch import MTCNN
                self._face_detector = MTCNN(
                    image_size=self.image_size,
                    margin=int(self.image_size * self.face_margin),
                    select_largest=True,
                    post_process=False,
                    device="cpu",
                )
            except ImportError:
                logger.warning("facenet-pytorch not installed, face detection disabled")
        return self._face_detector

    def _extract_face(self, image: np.ndarray) -> np.ndarray:
        """Extract face region from image using MTCNN."""
        if self.face_detector is None:
            return cv2.resize(image, (self.image_size, self.image_size))

        try:
            from PIL import Image
            pil_img = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
            boxes, probs = self.face_detector.detect(pil_img)

            if boxes is not None and len(boxes) > 0:
                box = boxes[0].astype(int)
                x1, y1, x2, y2 = box
                h, w = image.shape[:2]
                # Add margin
                margin = int(max(x2 - x1, y2 - y1) * self.face_margin)
                x1 = max(0, x1 - margin)
                y1 = max(0, y1 - margin)
                x2 = min(w, x2 + margin)
                y2 = min(h, y2 + margin)
                face = image[y1:y2, x1:x2]
                return cv2.resize(face, (self.image_size, self.image_size))
        except Exception as e:
            logger.debug(f"Face detection failed: {e}")

        return cv2.resize(image, (self.image_size, self.image_size))

    def _sample_frames(self, video_path: str) -> List[np.ndarray]:
        """Sample evenly-spaced frames from a video."""
        cap = cv2.VideoCapture(str(video_path))
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        if total <= 0:
            cap.release()
            return []

        indices = np.linspace(0, total - 1, self.num_frames, dtype=int)
        frames = []
        for idx in indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ret, frame = cap.read()
            if ret:
                frames.append(frame)

        cap.release()
        return frames

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx: int) -> dict:
        sample = self.samples[idx]
        label = sample["label"]
        manip_type = self.MANIPULATION_TYPES.get(sample["manipulation_type"], 0)

        if self.mode == "video":
            return self._get_video_item(sample, label, manip_type)
        else:
            return self._get_image_item(sample, label, manip_type)

    def _get_image_item(self, sample: dict, label: int, manip_type: int) -> dict:
        """Process a single image."""
        path = sample["path"]

        if str(path).endswith((".mp4", ".avi", ".mov")):
            # Extract a single frame from video
            frames = self._sample_frames(path)
            if frames:
                image = frames[len(frames) // 2]  # middle frame
            else:
                image = np.zeros((self.image_size, self.image_size, 3), dtype=np.uint8)
        else:
            image = cv2.imread(str(path))
            if image is None:
                image = np.zeros((self.image_size, self.image_size, 3), dtype=np.uint8)

        # Extract face
        face = self._extract_face(image)

        # Generate DCT for frequency encoder
        dct = apply_dct_transform(face)

        # Apply augmentation/normalization
        rgb = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
        augmented = self.transform(image=rgb)
        tensor = augmented["image"]

        # DCT transform (normalize manually)
        dct_normalized = (dct - dct.mean()) / (dct.std() + 1e-8)
        dct_tensor = torch.from_numpy(dct_normalized).permute(2, 0, 1).float()
        # Resize DCT to match image size
        dct_tensor = torch.nn.functional.interpolate(
            dct_tensor.unsqueeze(0), size=(self.image_size, self.image_size), mode="bilinear"
        ).squeeze(0)

        return {
            "image": tensor,
            "dct": dct_tensor,
            "label": torch.tensor(label, dtype=torch.float32),
            "manipulation_type": torch.tensor(manip_type, dtype=torch.long),
            "path": str(path),
        }

    def _get_video_item(self, sample: dict, label: int, manip_type: int) -> dict:
        """Process a video (multiple frames)."""
        path = sample["path"]
        frames = self._sample_frames(path)

        if len(frames) < self.num_frames:
            # Pad with last frame or black
            while len(frames) < self.num_frames:
                frames.append(frames[-1] if frames else np.zeros((self.image_size, self.image_size, 3), dtype=np.uint8))

        processed_frames = []
        dct_frames = []

        for frame in frames[: self.num_frames]:
            face = self._extract_face(frame)
            dct = apply_dct_transform(face)

            rgb = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
            augmented = self.transform(image=rgb)
            processed_frames.append(augmented["image"])

            dct_normalized = (dct - dct.mean()) / (dct.std() + 1e-8)
            dct_tensor = torch.from_numpy(dct_normalized).permute(2, 0, 1).float()
            dct_tensor = torch.nn.functional.interpolate(
                dct_tensor.unsqueeze(0), size=(self.image_size, self.image_size), mode="bilinear"
            ).squeeze(0)
            dct_frames.append(dct_tensor)

        return {
            "frames": torch.stack(processed_frames),       # (T, 3, H, W)
            "dct_frames": torch.stack(dct_frames),         # (T, 3, H, W)
            "label": torch.tensor(label, dtype=torch.float32),
            "manipulation_type": torch.tensor(manip_type, dtype=torch.long),
            "path": str(path),
        }
