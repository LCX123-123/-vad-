"""
音频 VAD 数据集与数据加载工具
期望有一个 CSV 标注文件，例如 data/voice/audio_vad.csv：
    path,valence,arousal,dominance
    voice/xxx.wav,0.1,0.3,-0.2
path 为相对于 CSV 所在目录的相对路径或绝对路径。
"""

from pathlib import Path
from typing import Tuple, List

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
import torchaudio
import logging

logger = logging.getLogger(__name__)


class AudioVADDataset(Dataset):
    def __init__(
        self,
        csv_path: str,
        root_dir: str = None,
        sample_rate: int = 16000,
        max_duration: float = 6.0,
    ):
        """
        Args:
            csv_path: 带有 path,valence,arousal,dominance 的 CSV 文件
            root_dir: 音频根目录，若为 None 则使用 csv 所在目录
            sample_rate: 目标采样率
            max_duration: 最长保留秒数（多余截断，不足补零）
        """
        self.csv_path = Path(csv_path)
        self.root_dir = Path(root_dir) if root_dir else self.csv_path.parent
        self.sample_rate = sample_rate
        self.max_samples = int(sample_rate * max_duration)

        df = pd.read_csv(self.csv_path, encoding="utf-8")
        required_cols = ["path", "valence", "arousal", "dominance"]
        for c in required_cols:
            if c not in df.columns:
                raise KeyError(f"音频 VAD CSV 缺少列: {c}")

        self.paths = df["path"].astype(str).tolist()
        self.labels = df[["valence", "arousal", "dominance"]].astype(float).values.astype(
            np.float32
        )

        # 重采样器延迟创建，避免 orig_freq=None 触发内部 int(None) 错误
        self._resampler = None

        logger.info(
            f"加载音频 VAD 数据集: {csv_path}, 样本数: {len(self.paths)}, 采样率: {sample_rate}"
        )

    def __len__(self) -> int:
        return len(self.paths)

    def _load_waveform(self, idx: int) -> torch.Tensor:
        path = self.paths[idx]
        wav_path = Path(path)
        if not wav_path.is_absolute():
            wav_path = self.root_dir / wav_path
        if not wav_path.exists():
            raise FileNotFoundError(f"音频文件不存在: {wav_path}")

        waveform, sr = torchaudio.load(wav_path)
        # 转单声道
        if waveform.shape[0] > 1:
            waveform = waveform.mean(dim=0, keepdim=True)
        # 重采样
        if sr != self.sample_rate:
            if self._resampler is None or getattr(self._resampler, "orig_freq", None) != sr:
                self._resampler = torchaudio.transforms.Resample(
                    orig_freq=sr, new_freq=self.sample_rate
                )
            waveform = self._resampler(waveform)
        # 截断/填充到固定长度
        if waveform.shape[1] > self.max_samples:
            waveform = waveform[:, : self.max_samples]
        else:
            pad_len = self.max_samples - waveform.shape[1]
            if pad_len > 0:
                waveform = torch.nn.functional.pad(waveform, (0, pad_len))
        return waveform  # [1, T]

    def __getitem__(self, idx: int) -> dict:
        waveform = self._load_waveform(idx)
        label = torch.from_numpy(self.labels[idx])  # [3]
        return {"waveform": waveform, "vad": label}


def create_audio_vad_dataloaders(
    csv_path: str,
    batch_size: int = 16,
    num_workers: int = 4,
    sample_rate: int = 16000,
    max_duration: float = 6.0,
    val_ratio: float = 0.1,
    test_ratio: float = 0.2,
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """从单一 CSV 创建 train/val/test 三个 DataLoader。"""
    from sklearn.model_selection import train_test_split

    df = pd.read_csv(csv_path, encoding="utf-8")
    indices = np.arange(len(df))
    train_indices, test_indices = train_test_split(
        indices, test_size=test_ratio, random_state=42
    )
    train_indices, val_indices = train_test_split(
        train_indices, test_size=val_ratio / (1 - test_ratio), random_state=42
    )

    def save_subset(idxs: np.ndarray, subset_name: str) -> str:
        sub = df.iloc[idxs]
        out_path = Path(csv_path).with_name(
            Path(csv_path).stem + f"_{subset_name}.csv"
        )
        sub.to_csv(out_path, index=False, encoding="utf-8")
        return str(out_path)

    train_csv = save_subset(train_indices, "train")
    val_csv = save_subset(val_indices, "val")
    test_csv = save_subset(test_indices, "test")

    train_ds = AudioVADDataset(train_csv, sample_rate=sample_rate, max_duration=max_duration)
    val_ds = AudioVADDataset(val_csv, sample_rate=sample_rate, max_duration=max_duration)
    test_ds = AudioVADDataset(test_csv, sample_rate=sample_rate, max_duration=max_duration)

    def make_loader(ds: Dataset, shuffle: bool) -> DataLoader:
        return DataLoader(
            ds,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
            pin_memory=True,
        )

    return make_loader(train_ds, True), make_loader(val_ds, False), make_loader(
        test_ds, False
    )

