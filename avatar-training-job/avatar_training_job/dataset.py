from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional
import cv2
from torch.utils.data import Dataset
import numpy as np
import os
import torch

TARGET_H = 1024
TARGET_W = 576

class SVDDataset(Dataset):
    def __init__(self, body_paths: List[Path], target_h: int = TARGET_H, target_w: int = TARGET_W):
        assert target_h % 8 == 0 and target_w % 8 == 0, "Target dimensions must be multiples of 8"
        self.body = sorted([p for p in body_paths if os.path.getsize(p) > 0])
        self.target_h = target_h
        self.target_w = target_w
        if not self.body:
            raise ValueError("Dataset is empty after filtering for valid files.")

    def __len__(self):
        return len(self.body)

    def __getitem__(self, idx):
        path = self.body[idx]
        img = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
        if img is None or img.size == 0:
            img = np.zeros((self.target_h, self.target_w, 3), dtype=np.uint8)
        else:
            if img.ndim == 2:
                img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
            elif img.shape[-1] == 4:
                img = img[:, :, :3]
            h, w = img.shape[:2]
            if h != self.target_h or w != self.target_w:
                img = cv2.resize(img, (self.target_w, self.target_h), interpolation=cv2.INTER_AREA)

        return {"body_bgr": torch.from_numpy(img.astype(np.uint8))}