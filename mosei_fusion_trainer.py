"""
MOSEI 多模态融合训练器
支持：
  - emotion: 6维多标签二分类（BCEWithLogitsLoss）
  - senti: 连续情感回归（MSELoss）
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Literal, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import f1_score
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import logging

from mosei_fusion_model import ConcatMLPFusionModel


logger = logging.getLogger(__name__)

TaskName = Literal["emotion", "senti"]


def _to_numpy(x):
    if isinstance(x, torch.Tensor):
        return x.detach().cpu().numpy()
    return x


class MoseiFusionTrainer:
    def __init__(
        self,
        task: TaskName,
        model: ConcatMLPFusionModel,
        train_loader,
        val_loader,
        test_loader,
        device: str,
        learning_rate: float = 2e-4,
        weight_decay: float = 1e-4,
        save_dir: str = "checkpoints",
        log_dir: str = "logs_mosei",
        max_grad_norm: float = 1.0,
    ):
        self.task = task
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.test_loader = test_loader
        self.device = device
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)
        self.log_dir = Path(log_dir)
        self.writer = SummaryWriter(log_dir=str(self.log_dir))
        self.max_grad_norm = max_grad_norm

        self.optimizer = optim.AdamW(
            self.model.parameters(), lr=learning_rate, weight_decay=weight_decay
        )

        if task == "emotion":
            self.criterion = nn.BCEWithLogitsLoss()
        else:
            self.criterion = nn.MSELoss()

        self.best_val_loss = float("inf")
        self.best_path = self.save_dir / f"best_mosei_{task}_model.pt"

    def _run_epoch(self, loader, train: bool) -> Dict[str, float]:
        if train:
            self.model.train()
        else:
            self.model.eval()

        total_loss = 0.0
        n_batches = 0

        all_preds = []
        all_labels = []

        loop = tqdm(loader, desc="训练中" if train else "验证中")
        for batch in loop:
            vision = batch["vision"].to(self.device)  # [B,T,35]
            audio = batch["audio"].to(self.device)  # [B,T,74]
            text = batch["text"].to(self.device)  # [B,T,300]
            labels = batch["labels"].to(self.device)

            if train:
                self.optimizer.zero_grad()

            with torch.set_grad_enabled(train):
                preds = self.model(vision, audio, text)
                loss = self.criterion(preds, labels)

                if train:
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)
                    self.optimizer.step()

            total_loss += float(loss.item())
            n_batches += 1

            all_preds.append(_to_numpy(preds))
            all_labels.append(_to_numpy(labels))

            loop.set_postfix(loss=f"{loss.item():.4f}")

        avg_loss = total_loss / max(1, n_batches)

        all_preds = np.concatenate(all_preds, axis=0)
        all_labels = np.concatenate(all_labels, axis=0)

        metrics: Dict[str, float] = {"loss": avg_loss}

        if self.task == "emotion":
            # preds: [N,6] logits
            probs = 1 / (1 + np.exp(-all_preds))
            y_pred = (probs >= 0.5).astype(np.int32)
            y_true = all_labels.astype(np.int32)
            # micro-F1 / macro-F1
            metrics["micro_f1"] = float(f1_score(y_true.reshape(-1), y_pred.reshape(-1), average="micro"))
            metrics["macro_f1"] = float(f1_score(y_true.reshape(-1), y_pred.reshape(-1), average="macro"))
            # per-dim F1
            per_dim = []
            for i in range(y_true.shape[1]):
                per_dim.append(float(f1_score(y_true[:, i], y_pred[:, i], average="binary", zero_division=0)))
            metrics["per_dim_f1"] = per_dim
        else:
            # preds: [N] regression
            y_pred = all_preds.reshape(-1)
            y_true = all_labels.reshape(-1)
            metrics["mae"] = float(np.mean(np.abs(y_true - y_pred)))
            # Pearson corr；方差为0时 corrcoef 可能给nan
            corr = np.corrcoef(y_true, y_pred)[0, 1] if y_true.size > 1 else 0.0
            metrics["pearson_r"] = float(0.0 if np.isnan(corr) else corr)

        return metrics

    def train(
        self,
        num_epochs: int = 10,
        early_stopping_patience: int = 5,
    ) -> Dict[str, float]:
        patience = 0

        for epoch in range(num_epochs):
            logger.info(f"MOSEI {self.task} Epoch {epoch+1}/{num_epochs}")

            train_metrics = self._run_epoch(self.train_loader, train=True)
            val_metrics = self._run_epoch(self.val_loader, train=False)

            self.writer.add_scalar(f"{self.task}/train_loss", train_metrics["loss"], epoch)
            self.writer.add_scalar(f"{self.task}/val_loss", val_metrics["loss"], epoch)

            # 保存 best
            if val_metrics["loss"] < self.best_val_loss:
                self.best_val_loss = val_metrics["loss"]
                torch.save(
                    {
                        "task": self.task,
                        "model_state_dict": self.model.state_dict(),
                    },
                    self.best_path,
                )
                patience = 0
                logger.info(f"保存 best: {self.best_path}")
            else:
                patience += 1

            logger.info(f"Train metrics: {train_metrics}")
            logger.info(f"Val metrics: {val_metrics}")

            if patience >= early_stopping_patience:
                logger.info("早停触发，结束训练")
                break

        self.writer.close()

        # 加载 best 并测试
        if self.best_path.exists():
            ckpt = torch.load(self.best_path, map_location=self.device, weights_only=False)
            self.model.load_state_dict(ckpt["model_state_dict"])

        return self.evaluate()

    @torch.no_grad()
    def evaluate(self) -> Dict[str, float]:
        test_metrics = self._run_epoch(self.test_loader, train=False)
        logger.info(f"Test metrics: {test_metrics}")

        out_path = self.save_dir / f"mosei_{self.task}_metrics.json"
        # 确保可json序列化
        import json

        with open(out_path, "w", encoding="utf-8") as f:
            json.dump({k: float(v) for k, v in test_metrics.items()}, f, ensure_ascii=False, indent=2)

        return test_metrics

