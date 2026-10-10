"""
Adversarial Robustness Augmentations.
Simulates real-world attacks: social media compression, rescaling, re-encoding, etc.
Applied during Stage 5 (robustness hardening) alongside standard augmentations.
"""

import albumentations as A
from albumentations.pytorch import ToTensorV2
import numpy as np
import cv2


def get_adversarial_transforms(image_size: int = 160) -> A.Compose:
    """
    Adversarial augmentations that simulate real-world distribution shifts.
    Use during Stage 5 (adversarial hardening) training.
    """
    return A.Compose([
        A.Resize(image_size, image_size),
        A.HorizontalFlip(p=0.5),
        A.Rotate(limit=15, p=0.3),

        # --- Adversarial augmentations ---

        # Double JPEG compression (simulates social media re-encoding)
        A.OneOf([
            A.Compose([
                A.ImageCompression(quality_range=(20, 50), p=1.0),
                A.ImageCompression(quality_range=(40, 70), p=1.0),
            ]),
            A.ImageCompression(quality_range=(15, 40), p=1.0),
        ], p=0.4),

        # Multi-scale resize (downscale then upscale — destroys artifacts)
        A.OneOf([
            A.Compose([
                A.Resize(int(image_size * 0.5), int(image_size * 0.5)),
                A.Resize(image_size, image_size),
            ]),
            A.Compose([
                A.Resize(int(image_size * 0.7), int(image_size * 0.7)),
                A.Resize(image_size, image_size),
            ]),
            A.Compose([
                A.Resize(int(image_size * 0.9), int(image_size * 0.9)),
                A.Resize(image_size, image_size),
            ]),
        ], p=0.3),

        # Noise injection (simulates camera noise and transmission noise)
        A.OneOf([
            A.GaussNoise(std_range=(0.1, 0.5), p=1.0),
            A.GaussNoise(std_range=(0.25, 0.75), p=1.0),  # heavy noise
            A.ISONoise(p=1.0),  # camera sensor noise
        ], p=0.3),

        # Blur (motion and defocus)
        A.OneOf([
            A.GaussianBlur(blur_limit=(3, 7), p=1.0),
            A.MotionBlur(blur_limit=7, p=1.0),
            A.MedianBlur(blur_limit=5, p=1.0),
        ], p=0.25),

        # Random crop + pad back (simulates cropping attacks)
        A.OneOf([
            A.RandomCrop(int(image_size * 0.85), int(image_size * 0.85), p=1.0),
            A.RandomCrop(int(image_size * 0.7), int(image_size * 0.7), p=1.0),
        ], p=0.2),
        A.Resize(image_size, image_size),  # resize back after potential crop

        # Color distortions (simulates different display/capture conditions)
        A.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.1, p=0.3),

        # Sharpen (some social media platforms apply sharpening)
        A.Sharpen(alpha=(0.2, 0.5), lightness=(0.5, 1.0), p=0.15),

        # CoarseDropout (simulates occlusion)
        A.CoarseDropout(num_holes_range=(1, 6), hole_height_range=(8, 20), hole_width_range=(8, 20), p=0.2),

        # Standard normalization
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ToTensorV2(),
    ])


def get_social_media_transform(image_size: int = 160) -> A.Compose:
    """Simulates typical social media pipeline: JPEG + resize + slight noise."""
    return A.Compose([
        A.Resize(image_size, image_size),
        A.ImageCompression(quality_range=(30, 75), p=0.8),
        A.Resize(int(image_size * 0.8), int(image_size * 0.8)),
        A.Resize(image_size, image_size),
        A.GaussNoise(std_range=(0.05, 0.15), p=0.3),
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ToTensorV2(),
    ])
