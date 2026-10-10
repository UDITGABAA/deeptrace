"""Image preprocessing and augmentation transforms."""

import albumentations as A
from albumentations.pytorch import ToTensorV2
import numpy as np
import cv2


def get_train_transforms(image_size: int = 160) -> A.Compose:
    """Training augmentations: heavy compression focus for robustness."""
    return A.Compose([
        A.Resize(image_size, image_size),
        A.HorizontalFlip(p=0.5),
        A.Rotate(limit=15, p=0.3),
        # JPEG compression applied independently (not inside OneOf)
        A.ImageCompression(quality_range=(30, 95), p=0.7),
        # Then also apply blur/noise separately
        A.OneOf([
            A.GaussianBlur(blur_limit=5, p=1.0),
            A.GaussNoise(std_range=(0.1, 0.5), p=1.0),
        ], p=0.3),
        A.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.05, p=0.3),
        A.CoarseDropout(num_holes_range=(1, 4), hole_height_range=(8, 16),
                        hole_width_range=(8, 16), p=0.2),
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ToTensorV2(),
    ])


def get_val_transforms(image_size: int = 160) -> A.Compose:
    """Validation/test transforms: resize + normalize only."""
    return A.Compose([
        A.Resize(image_size, image_size),
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ToTensorV2(),
    ])


def apply_dct_transform(image: np.ndarray) -> np.ndarray:
    """
    Apply DCT (Discrete Cosine Transform) to image.
    Converts to YCrCb and applies block-wise DCT.

    Args:
        image: BGR image as numpy array (H, W, 3)

    Returns:
        DCT coefficients as numpy array (H, W, 3)
    """
    # Convert to YCrCb color space
    ycrcb = cv2.cvtColor(image, cv2.COLOR_BGR2YCrCb).astype(np.float32)

    dct_channels = []
    for c in range(3):
        channel = ycrcb[:, :, c]
        # Resize to multiple of 8 for DCT blocks
        h, w = channel.shape
        h8 = (h // 8) * 8
        w8 = (w // 8) * 8
        channel = channel[:h8, :w8]

        # Block-wise 8x8 DCT
        dct_result = np.zeros_like(channel)
        for i in range(0, h8, 8):
            for j in range(0, w8, 8):
                block = channel[i : i + 8, j : j + 8]
                dct_result[i : i + 8, j : j + 8] = cv2.dct(block)

        dct_channels.append(dct_result)

    return np.stack(dct_channels, axis=-1)
