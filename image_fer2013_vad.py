from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Tuple

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.cuda.amp import autocast, GradScaler

from image_fer2013_dataset import FER2013CsvDataset, SPLIT_MAP
from image_fer2013_model import ResNet18RGB

# FER2013 类别索引到情绪名的映射（固定于数据集定义）
# 0=Angry,1=Disgust,2=Fear,3=Happy,4=Sad,5=Surprise,6=Neutral
INDEX_TO_EMO = {
	0: "angry",
	1: "disgust",
	2: "fear",
	3: "happy",
	4: "sad",
	5: "surprise",
	6: "neutral",
}

# 参考文献与常用经验的 VAD 近似值，范围在 [-1, 1]
# 这些值可按需微调或替换为更权威的表
EMO_TO_VAD: Dict[str, Tuple[float, float, float]] = {
	"angry": (-0.51, 0.59, 0.25),
	"disgust": (-0.60, 0.35, 0.11),
	"fear": (-0.64, 0.60, -0.43),
	"happy": (0.81, 0.51, 0.46),
	"sad": (-0.63, -0.27, -0.33),
	"surprise": (0.40, 0.67, -0.13),
	"neutral": (0.02, 0.01, 0.00),
}


class FER2013CsvVADDataset(FER2013CsvDataset):
	"""在原有 FER2013CsvDataset 基础上，将离散情绪映射到 VAD 三维目标。"""

	def __init__(
		self,
		csv_path: str | Path,
		split: str = "train",
		transform=None,
		target_transform=None,
		dtype=torch.float32,
		normalize: bool = True,
		resize_to: int = 224,
		three_channels: bool = True,
		augment: bool = False,
		require_usage: bool = True,
	) -> None:
		super().__init__(
			csv_path=csv_path,
			split=split,
			transform=transform,
			target_transform=target_transform,
			dtype=dtype,
			normalize=normalize,
			resize_to=resize_to,
			three_channels=three_channels,
			augment=augment,
			require_usage=require_usage,
		)

	def __getitem__(self, index: int):
		img, label = super().__getitem__(index)
		emo = INDEX_TO_EMO[int(label)]
		vad = torch.tensor(EMO_TO_VAD[emo], dtype=torch.float32)
		return img, vad


def create_vad_dataloaders(
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
	require_usage = (csv_path == val_csv_path == test_csv_path)

	train_ds = FER2013CsvVADDataset(csv_path, split="train", augment=True, three_channels=True, resize_to=224, require_usage=require_usage)
	val_ds = FER2013CsvVADDataset(val_csv_path, split="validation", augment=False, three_channels=True, resize_to=224, require_usage=require_usage)
	test_ds = FER2013CsvVADDataset(test_csv_path, split="test", augment=False, three_channels=True, resize_to=224, require_usage=require_usage)

	train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=pin_memory)
	val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=pin_memory)
	test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=pin_memory)
	return train_loader, val_loader, test_loader


@dataclass
class VADMetrics:
	mse: float
	mae: float
	rmse: float


class VADRegressor(nn.Module):
	"""ResNet18 回归 VAD（三输出）。"""

	def __init__(self, pretrained: bool = True) -> None:
		super().__init__()
		self.backbone = ResNet18RGB(num_classes=3, pretrained=pretrained)

	def forward(self, x: torch.Tensor) -> torch.Tensor:
		return self.backbone(x)


class VADTrainer:
	def __init__(
		self,
		model: nn.Module,
		train_loader: DataLoader,
		val_loader: DataLoader,
		test_loader: DataLoader,
		device: str = "cpu",
		learning_rate: float = 1e-3,
		weight_decay: float = 1e-4,
		save_dir: str | Path = "checkpoints",
	) -> None:
		self.model = model.to(device)
		self.train_loader = train_loader
		self.val_loader = val_loader
		self.test_loader = test_loader
		self.device = device
		self.criterion = nn.MSELoss()
		self.optimizer = optim.AdamW(self.model.parameters(), lr=learning_rate, weight_decay=weight_decay)
		self.scheduler = optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=50)
		self.scaler = GradScaler(enabled=(device == "cuda"))
		self.save_dir = Path(save_dir)
		self.save_dir.mkdir(parents=True, exist_ok=True)

	@torch.no_grad()
	def _reduce_metrics(self, sq_err_sum: float, abs_err_sum: float, n: int) -> VADMetrics:
		mse = sq_err_sum / n
		mae = abs_err_sum / n
		rmse = mse ** 0.5
		return VADMetrics(mse=mse, mae=mae, rmse=rmse)

	def _run_epoch(self, loader: DataLoader, train: bool) -> VADMetrics:
		if train:
			self.model.train()
		else:
			self.model.eval()
		sq_err_sum = 0.0
		abs_err_sum = 0.0
		num = 0
		for images, targets in loader:
			images = images.to(self.device)
			targets = targets.to(self.device)
			if train:
				self.optimizer.zero_grad()
			with torch.set_grad_enabled(train):
				with autocast(enabled=(self.device == "cuda")):
					preds = self.model(images)
					loss = self.criterion(preds, targets)
				if train:
					self.scaler.scale(loss).backward()
					self.scaler.step(self.optimizer)
					self.scaler.update()
			# 累计误差
			sq_err_sum += torch.sum((preds - targets) ** 2).item()
			abs_err_sum += torch.sum(torch.abs(preds - targets)).item()
			num += targets.numel()
		return self._reduce_metrics(sq_err_sum, abs_err_sum, num)

	def train(self, num_epochs: int = 20) -> None:
		best_mse = float("inf")
		best_path = self.save_dir / "best_image_vad_model.pt"
		for epoch in range(1, num_epochs + 1):
			train_m = self._run_epoch(self.train_loader, train=True)
			val_m = self._run_epoch(self.val_loader, train=False)
			print(
				f"Epoch {epoch}: train_mse={train_m.mse:.4f} train_mae={train_m.mae:.4f} "
				f"val_mse={val_m.mse:.4f} val_mae={val_m.mae:.4f}"
			)
			if val_m.mse < best_mse:
				best_mse = val_m.mse
				torch.save({"model_state_dict": self.model.state_dict()}, best_path)
			self.scheduler.step()
		print(f"Best val MSE: {best_mse:.4f}, saved at {best_path}")

	def evaluate(self) -> VADMetrics:
		m = self._run_epoch(self.test_loader, train=False)
		print(f"Test: mse={m.mse:.4f} mae={m.mae:.4f} rmse={m.rmse:.4f}")
		return m


