import os
import random
import torch
import numpy as np
from torch.utils.data import Dataset
from PIL import Image
from image import *
import torchvision.transforms.functional as F

class listDataset(Dataset):
    def __init__(self, root, shape=None, shuffle=True, transform=None, train=False, seen=0, batch_size=1, num_workers=4):
        if train:
            root = root * 4  # Data augmentation by repetition
        random.shuffle(root)

        self.nSamples = len(root)
        self.lines = root
        self.transform = transform
        self.train = train
        self.shape = shape
        self.seen = seen
        self.batch_size = batch_size
        self.num_workers = num_workers

    def __len__(self):
        return len(self.lines)

    def __getitem__(self, index):
        assert index < len(self), 'index range error'

        sample = self.lines[index]  # should be a dict with "image" and "gt"
        img, target = load_data(sample, self.train)

        if self.transform is not None:
            img = self.transform(img)  # Should be [3, H, W]

        # Ensure target is [1, H, W]
        if isinstance(target, np.ndarray):
            if target.ndim == 2:
                target = torch.from_numpy(target).unsqueeze(0).float()
            else:
                raise ValueError(f"Unexpected target shape: {target.shape}")
        elif isinstance(target, torch.Tensor) and target.ndim == 2:
            target = target.unsqueeze(0)

        return img, target



