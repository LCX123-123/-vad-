"""
CMU-MOSEI（cmumosei）多模态数据集封装

数据源（你当前提供的 pkl）：
  - mosei_emotion_aligned_60.pkl:
      vision: [N, 60, 35]
      audio:  [N, 60, 74]
      text:   [N, 60, 300]
      labels: [N, 6]   (int64, 每维0/1，多标签二分类)
  - mosei_senti_data.pkl:
      vision: [N, 50, 35]
      audio:  [N, 50, 74]
      text:   [N, 50, 300]
      labels: [N, 1, 1] (float32，连续回归，范围约[-3,3])
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Literal, Tuple

import pickle
import numpy as np
import torch
from torch.utils.data import Dataset


TaskName = Literal["emotion", "senti"]


class MoseiMultimodalDataset(Dataset):
    def __init__(
        self,
        pkl_path: str,
        task: TaskName,
        split: Literal["train", "valid", "test"] = "train",
    ):
        self.pkl_path = str(pkl_path)
        self.task = task
        self.split = split

        if not Path(self.pkl_path).exists():
            raise FileNotFoundError(f"MOSEI pkl 不存在: {self.pkl_path}")

        with open(self.pkl_path, "rb") as f:
            obj = pickle.load(f)

        if split not in obj:
            raise KeyError(f"pkl 内缺少 split: {split}，可用: {list(obj.keys())}")

        data = obj[split]
        # 强制转成 numpy 数组视图（pickle 已经给了 numpy）
        self.vision = data["vision"]  # [N,T,35]
        self.audio = data["audio"]  # [N,T,74]
        self.text = data["text"]  # [N,T,300]
        self.labels = data["labels"]

        n = self.vision.shape[0]
        assert self.audio.shape[0] == n and self.text.shape[0] == n

        if task == "emotion":
            if self.labels.shape != (n, 6):
                raise ValueError(f"emotion labels shape 不符合预期: got {self.labels.shape}, expect {(n,6)}")
        elif task == "senti":
            if self.labels.shape[0] != n:
                raise ValueError(f"senti labels shape 不符合预期: got {self.labels.shape}")
        else:
            raise ValueError(f"未知 task: {task}")

    def __len__(self) -> int:
        return int(self.vision.shape[0])

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        vision = torch.from_numpy(self.vision[idx]).to(torch.float32)  # [T,35]
        audio = torch.from_numpy(self.audio[idx]).to(torch.float32)  # [T,74]
        text = torch.from_numpy(self.text[idx]).to(torch.float32)  # [T,300]

        # 清洗 NaN/Inf，避免回归 loss 出现 nan
        vision = torch.nan_to_num(vision, nan=0.0, posinf=0.0, neginf=0.0)
        audio = torch.nan_to_num(audio, nan=0.0, posinf=0.0, neginf=0.0)
        text = torch.nan_to_num(text, nan=0.0, posinf=0.0, neginf=0.0)

        if self.task == "emotion":
            label = torch.from_numpy(self.labels[idx]).to(torch.float32)  # [6]
            label = torch.nan_to_num(label, nan=0.0, posinf=0.0, neginf=0.0)
            return {"vision": vision, "audio": audio, "text": text, "labels": label}

        # senti
        label = self.labels[idx]
        # [1,1] -> scalar
        if isinstance(label, np.ndarray):
            label = float(label.reshape(-1)[0])
        else:
            label = float(label)
        label_t = torch.tensor(label, dtype=torch.float32)
        label_t = torch.nan_to_num(label_t, nan=0.0, posinf=0.0, neginf=0.0)
        return {"vision": vision, "audio": audio, "text": text, "labels": label_t}

