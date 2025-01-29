import random
import os
from PIL import Image,ImageFilter,ImageDraw
import numpy as np
import h5py
from PIL import ImageStat
import cv2
import torch


# +
def load_data(sample, train=True):
    img_path = sample["image"]
    gt_path = sample["gt"]

    # Load and resize image to 1024x768
    img = Image.open(img_path).convert("RGB")
    img = img.resize((1024, 768), resample=Image.BILINEAR)

    # Load and resize density map to 128x96
    target = np.load(gt_path).astype(np.float32)
    target = cv2.resize(target, (128, 96), interpolation=cv2.INTER_CUBIC)

    # Check for NaN or Inf values in density map
    if np.isnan(target).any() or np.isinf(target).any():
        raise ValueError(f"Error: NaN or Inf detected in density map: {gt_path}")

    # Preserve the actual person count without scaling
    target = target if target.sum() > 0 else target

    # Convert to tensor with shape [1, 96, 128]
    if target.ndim == 2:
        target = torch.from_numpy(target).unsqueeze(0).float()
    elif target.ndim == 3 and target.shape[0] == 1:
        target = torch.from_numpy(target).float()
    else:
        raise ValueError(f"Error: Unexpected target shape: {target.shape}")

    return img, target

