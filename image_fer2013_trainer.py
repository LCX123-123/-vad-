from __future__ import annotations

from pathlib import Path
from typing import Dict

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.cuda.amp import autocast, GradScaler


class ImageTrainer:
    def __init__(
        self,
        model: torch.nn.Module,
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
        # Label Smoothing 提升泛化
        self.criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
        self.optimizer = optim.AdamW(self.model.parameters(), lr=learning_rate, weight_decay=weight_decay)
        # 余弦退火调度
        self.scheduler = optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=50)
        # AMP
        self.scaler = GradScaler(enabled=(device == "cuda"))
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)

    def _run_epoch(self, loader: DataLoader, train: bool) -> Dict[str, float]:
        epoch_loss = 0.0
        correct = 0
        total = 0
        if train:
            self.model.train()
        else:
            self.model.eval()
        for images, labels in loader:
            images = images.to(self.device)
            labels = labels.to(self.device)
            if train:
                self.optimizer.zero_grad()
            with torch.set_grad_enabled(train):
                with autocast(enabled=(self.device == "cuda")):
                    logits = self.model(images)
                    loss = self.criterion(logits, labels)
                if train:
                    self.scaler.scale(loss).backward()
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
            epoch_loss += loss.item() * labels.size(0)
            preds = torch.argmax(logits, dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)
        return {"loss": epoch_loss / total, "acc": correct / total}

    def train(self, num_epochs: int = 10) -> None:
        best_acc = 0.0
        best_path = self.save_dir / "best_image_model.pt"
        for epoch in range(1, num_epochs + 1):
            train_metrics = self._run_epoch(self.train_loader, train=True)
            val_metrics = self._run_epoch(self.val_loader, train=False)
            print(f"Epoch {epoch}: train_loss={train_metrics['loss']:.4f} train_acc={train_metrics['acc']:.4f} val_loss={val_metrics['loss']:.4f} val_acc={val_metrics['acc']:.4f}")
            if val_metrics["acc"] > best_acc:
                best_acc = val_metrics["acc"]
                torch.save({"model_state_dict": self.model.state_dict()}, best_path)
            self.scheduler.step()
        print(f"Best val acc: {best_acc:.4f}, saved at {best_path}")

    def evaluate(self) -> Dict[str, float]:
        metrics = self._run_epoch(self.test_loader, train=False)
        print(f"Test: loss={metrics['loss']:.4f} acc={metrics['acc']:.4f}")
        return metrics


