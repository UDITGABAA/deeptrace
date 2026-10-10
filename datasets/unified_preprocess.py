"""
Unified Preprocessing Pipeline for DeepTrace.

All datasets (Kaggle, FF++, Celeb-DF, DFDC) must go through this same pipeline
to ensure cross-dataset scores measure model generalization, not preprocessing differences.

Pipeline:
  1. Input: source image (pre-cropped face or raw frame)
  2. Face detection/crop: SKIP if already face-cropped (Kaggle, FF++ extracted faces)
     For raw video frames: use MTCNN with these settings:
       - min_face_size=40
       - thresholds=[0.6, 0.7, 0.7]
       - select largest face if multiple detected
       - fallback: center-crop to square if no face detected
  3. Resize: 160x160 (bilinear interpolation)
  4. RGB normalization: via Albumentations (ImageNet mean/std)
  5. DCT transform: apply_dct_transform() from datasets/transforms.py
     - Normalize: (dct - mean) / (std + 1e-8)
     - Resize DCT tensor to 160x160

This file provides the canonical preprocessing function used by all loaders.
"""

import cv2
import numpy as np
import torch
import torch.nn.functional as F

from datasets.transforms import get_val_transforms, apply_dct_transform


def preprocess_face_image(
    img_bgr: np.ndarray,
    image_size: int = 160,
    transform=None,
    is_pre_cropped: bool = True,
):
    """
    Canonical preprocessing for a single face image.

    Args:
        img_bgr: BGR image (numpy array)
        image_size: target size (default 160)
        transform: Albumentations val transform (created if None)
        is_pre_cropped: True for Kaggle/FF++ extracted faces (skip face detection)

    Returns:
        dict with:
        - image_tensor: (3, H, W) float tensor
        - dct_tensor: (3, H, W) float tensor
        - original_size: (H, W) tuple
    """
    if transform is None:
        transform = get_val_transforms(image_size)

    original_size = (img_bgr.shape[0], img_bgr.shape[1])

    # Step 1: Face crop (skip for pre-cropped datasets)
    if not is_pre_cropped:
        # For raw frames: center-crop to square as fallback
        # TODO: Add MTCNN face detection for video frame processing
        h, w = img_bgr.shape[:2]
        if h != w:
            s = min(h, w)
            y = (h - s) // 2
            x = (w - s) // 2
            img_bgr = img_bgr[y:y+s, x:x+s]

    # Step 2: Resize to target size
    img_resized = cv2.resize(img_bgr, (image_size, image_size))

    # Step 3: RGB conversion + normalization
    rgb = cv2.cvtColor(img_resized, cv2.COLOR_BGR2RGB)
    aug = transform(image=rgb)
    image_tensor = aug["image"]  # (3, H, W) normalized

    # Step 4: DCT transform + normalization
    dct = apply_dct_transform(img_resized)
    dct_norm = (dct - dct.mean()) / (dct.std() + 1e-8)
    dct_tensor = torch.from_numpy(dct_norm).permute(2, 0, 1).float()
    dct_tensor = F.interpolate(
        dct_tensor.unsqueeze(0),
        size=(image_size, image_size),
        mode="bilinear",
        align_corners=False,
    ).squeeze(0)

    return {
        "image_tensor": image_tensor,
        "dct_tensor": dct_tensor,
        "original_size": original_size,
    }


# ── Documentation ──────────────────────────────────────────────────
PREPROCESSING_SPEC = {
    "version": "1.0",
    "image_size": 160,
    "rgb_normalization": "ImageNet (mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225])",
    "dct_normalization": "per-image (x - mean) / (std + 1e-8)",
    "face_detection": {
        "pre_cropped_datasets": ["kaggle_realfake", "ffpp_extracted_c23", "celeb_df_extracted"],
        "action": "skip face detection, direct resize",
    },
    "raw_video_frames": {
        "detector": "MTCNN (to be implemented for video processing)",
        "face_selection": "largest face if multiple",
        "fallback": "center-crop to square",
    },
    "datasets_treated_as_pre_cropped": {
        "kaggle_realfake": "256x256 aligned faces, resize to 160x160",
        "ffpp_c23_extracted": "Pre-extracted face crops from Kaggle dataset, resize to 160x160",
        "celeb_df_extracted": "Pre-extracted face crops, resize to 160x160",
    },
    "tensor_equality_note": (
        "For Kaggle pre-cropped images, the old GenericFolderDataset path and this "
        "unified path produce identical tensors (same resize → RGB → DCT pipeline). "
        "Verified by construction: both call cv2.resize + apply_dct_transform with "
        "identical parameters."
    ),
}
