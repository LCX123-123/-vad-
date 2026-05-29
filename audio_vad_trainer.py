"""
音频 VAD 训练与评估
"""

from pathlib import Path
from typing import Dict

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import logging

logger = logging.getLogger(__name__)


class AudioVADTrainer:
    def __init__(
        self,
        model: nn.Module,
        train_loader,
        val_loader,
        test_loader,
        device: str = "cuda",
        learning_rate: float = 1e-4,
        weight_decay: float = 1e-4,
        save_dir: str = "checkpoints",
        log_dir: str = "logs_audio",
    ):
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.test_loader = test_loader
        self.device = device

        self.optimizer = optim.AdamW(
            self.model.parameters(), lr=learning_rate, weight_decay=weight_decay
        )
        self.criterion = nn.MSELoss()

        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)
        self.writer = SummaryWriter(log_dir=log_dir)

        self.best_val_loss = float("inf")
        self.best_state = None

    def _run_epoch(self, loader, train: bool) -> Dict[str, float]:
        if train:
            self.model.train()
        else:
            self.model.eval()

        total_loss = total_mse = total_mae = total_r2 = 0.0
        n_batches = 0

        loop = tqdm(loader, desc="训练中" if train else "验证中")
        for batch in loop:
            wave = batch["waveform"].to(self.device)  # [B,1,T]
            vad = batch["vad"].to(self.device)  # [B,3]

            if train:
                self.optimizer.zero_grad()

            pred = self.model(wave)  # [B,3]
            loss = self.criterion(pred, vad)

            if train:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                self.optimizer.step()

            with torch.no_grad():
                p = pred.detach().cpu().numpy()
                y = vad.detach().cpu().numpy()
                mse = mean_squared_error(y, p)
                mae = mean_absolute_error(y, p)
                try:
                    r2 = r2_score(y, p)
                except ValueError:
                    r2 = 0.0

            total_loss += loss.item()
            total_mse += mse
            total_mae += mae
            total_r2 += r2
            n_batches += 1

            loop.set_postfix(
                loss=f"{loss.item():.4f}",
                mse=f"{mse:.4f}",
                mae=f"{mae:.4f}",
                r2=f"{r2:.4f}",
            )

        return {
            "loss": total_loss / n_batches,
            "mse": total_mse / n_batches,
            "mae": total_mae / n_batches,
            "r2": total_r2 / n_batches,
        }

    def train(self, num_epochs: int = 10):
        for epoch in range(num_epochs):
            logger.info(f"Audio VAD Epoch {epoch+1}/{num_epochs}")
            train_metrics = self._run_epoch(self.train_loader, train=True)
            val_metrics = self._run_epoch(self.val_loader, train=False)

            for k, v in train_metrics.items():
                self.writer.add_scalar(f"train/{k}", v, epoch)
            for k, v in val_metrics.items():
                self.writer.add_scalar(f"val/{k}", v, epoch)

            logger.info(
                f"Train - Loss {train_metrics['loss']:.4f}, MSE {train_metrics['mse']:.4f}, "
                f"MAE {train_metrics['mae']:.4f}, R2 {train_metrics['r2']:.4f}"
            )
            logger.info(
                f"Val   - Loss {val_metrics['loss']:.4f}, MSE {val_metrics['mse']:.4f}, "
                f"MAE {val_metrics['mae']:.4f}, R2 {val_metrics['r2']:.4f}"
            )

            if val_metrics["loss"] < self.best_val_loss:
                self.best_val_loss = val_metrics["loss"]
                self.best_state = self.model.state_dict()
                torch.save(
                    {
                        "model_state_dict": self.model.state_dict(),
                        "val_loss": val_metrics["loss"],
                    },
                    self.save_dir / "best_audio_vad_model.pt",
                )
                logger.info("保存最佳 Audio VAD 模型")

        if self.best_state is not None:
            self.model.load_state_dict(self.best_state)
            logger.info("加载最佳 Audio VAD 模型用于评估")

    def evaluate(self) -> Dict[str, float]:
        self.model.eval()
        total_loss = total_mse = total_mae = total_r2 = 0.0
        n_batches = 0
        all_p, all_y = [], []

        loop = tqdm(self.test_loader, desc="评估中")
        for batch in loop:
            wave = batch["waveform"].to(self.device)
            vad = batch["vad"].to(self.device)
            with torch.no_grad():
                pred = self.model(wave)
                loss = self.criterion(pred, vad)
            p = pred.detach().cpu().numpy()
            y = vad.detach().cpu().numpy()
            all_p.append(p)
            all_y.append(y)

            mse = mean_squared_error(y, p)
            mae = mean_absolute_error(y, p)
            try:
                r2 = r2_score(y, p)
            except ValueError:
                r2 = 0.0

            total_loss += loss.item()
            total_mse += mse
            total_mae += mae
            total_r2 += r2
            n_batches += 1

        all_p = np.concatenate(all_p, axis=0)
        all_y = np.concatenate(all_y, axis=0)

        metrics = {
            "loss": total_loss / n_batches,
            "mse": total_mse / n_batches,
            "mae": total_mae / n_batches,
            "r2": total_r2 / n_batches,
        }
        logger.info(
            f"Test - Loss {metrics['loss']:.4f}, MSE {metrics['mse']:.4f}, "
            f"MAE {metrics['mae']:.4f}, R2 {metrics['r2']:.4f}"
        )
        return metrics

