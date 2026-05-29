from __future__ import annotations

import random
from pathlib import Path
from typing import Callable, Optional, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

# FER2013 官方 Usage 字段到自定义 split 的映射
SPLIT_MAP = {
    "train": "Training",
    "training": "Training",
    "validation": "PublicTest",
    "val": "PublicTest",
    "dev": "PublicTest",
    "test": "PrivateTest",
}


class FER2013CsvDataset(Dataset):
    """基于 fer2013 的 Dataset，支持原始 Usage 列或拆分后的独立 CSV。"""

    def __init__(
        self,
        csv_path: str | Path,
        split: str = "train",
        transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None,
        dtype: torch.dtype = torch.float32,
        normalize: bool = True,
        resize_to: int = 224,
        three_channels: bool = True,
        augment: bool = False,
        require_usage: bool = True,
    ) -> None:
        csv_path = Path(csv_path)
        if not csv_path.exists():
            raise FileNotFoundError(f"CSV 不存在: {csv_path}")

        df = pd.read_csv(csv_path)
        split_lower = split.lower()

        if "Usage" in df.columns:
            mapped = SPLIT_MAP.get(split_lower, split)
            if mapped not in {"Training", "PublicTest", "PrivateTest"}:
                raise ValueError("split 仅支持 'train'/'validation'/'test' 或原始名称")
            df = df[df["Usage"] == mapped].reset_index(drop=True)
        else:
            # 无 Usage 列时，直接使用整份 CSV。外部应分别传入 train/val/test CSV
            if require_usage and split_lower not in {"train", "validation", "val", "test"}:
                raise ValueError("自定义拆分 CSV 仅支持 'train', 'validation', 'test'")
            df = df.reset_index(drop=True)

        if "pixels" not in df.columns or "emotion" not in df.columns:
            raise KeyError("CSV 必须包含 'pixels' 与 'emotion' 列")

        self.pixels = df["pixels"]
        self.labels = df["emotion"].astype(int)
        self.transform = transform
        self.target_transform = target_transform
        self.dtype = dtype
        self.normalize = normalize
        self.resize_to = resize_to
        self.three_channels = three_channels
        self.augment = augment and (split_lower in {"train", "training"})

    def __len__(self) -> int:
        return len(self.pixels)

    def __getitem__(self, index: int):
        pixels_str = self.pixels.iloc[index]
        arr = np.fromstring(pixels_str, dtype=np.uint8, sep=" ")
        if arr.size != 48 * 48:
            raise ValueError(f"像素数量不匹配: got {arr.size} at index {index}")
        arr = arr.reshape(48, 48)
        img = torch.from_numpy(arr).unsqueeze(0).to(self.dtype)  # (1, 48, 48)
        if self.normalize:
            img = img / 255.0
        if self.resize_to and self.resize_to != 48:
            img = F.interpolate(
                img.unsqueeze(0),
                size=(self.resize_to, self.resize_to),
                mode="bilinear",
                align_corners=False,
            ).squeeze(0)

        if self.augment:
            if random.random() < 0.5:
                img = torch.flip(img, dims=[2])

        if self.three_channels:
            img = img.repeat(3, 1, 1)

        if self.transform is not None:
            img = self.transform(img)

        label = int(self.labels.iloc[index])
        if self.target_transform is not None:
            label = self.target_transform(label)
        return img, label


def create_fer_dataloaders(
    csv_path: str | Path,
    batch_size: int = 64,
    num_workers: int = 0,
    pin_memory: bool = False,
    val_csv_path: str | Path | None = None,
    test_csv_path: str | Path | None = None,
):
    csv_path = Path(csv_path)
    val_csv_path = Path(val_csv_path) if val_csv_path else csv_path
    test_csv_path = Path(test_csv_path) if test_csv_path else csv_path

    # 当 val/test 与 train CSV 不一致时，不需要 Usage 列
    require_usage = (
        csv_path == val_csv_path == test_csv_path
    )

    train_ds = FER2013CsvDataset(
        csv_path,
        split="train",
        augment=True,
        three_channels=True,
        resize_to=224,
        require_usage=require_usage,
    )
    val_ds = FER2013CsvDataset(
        val_csv_path,
        split="validation",
        augment=False,
        three_channels=True,
        resize_to=224,
        require_usage=require_usage,
    )
    test_ds = FER2013CsvDataset(
        test_csv_path,
        split="test",
        augment=False,
        three_channels=True,
        resize_to=224,
        require_usage=require_usage,
    )

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=pin_memory)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=pin_memory)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=pin_memory)
    return train_loader, val_loader, test_loader

