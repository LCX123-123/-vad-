"""
情感维度模型训练器
支持多维度情感回归任务的训练、验证和评估
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import numpy as np
import logging
import os
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import time
from tqdm import tqdm

logger = logging.getLogger(__name__)


class EmotionDimensionTrainer:
    """
    情感维度模型训练器
    支持多维度情感回归任务的训练和评估
    """
    
    def __init__(
        self,
        model: nn.Module,
        train_loader,
        val_loader,
        test_loader,
        device: str = "cuda",
        learning_rate: float = 2e-5,
        weight_decay: float = 0.01,
        warmup_steps: int = 500,
        max_grad_norm: float = 1.0,
        save_dir: str = "checkpoints",
        log_dir: str = "logs"
    ):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.test_loader = test_loader
        self.device = device
        self.save_dir = Path(save_dir)
        self.log_dir = Path(log_dir)
        
        # 创建保存目录
        self.save_dir.mkdir(parents=True, exist_ok=True)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        
        # 设置设备
        self.model.to(device)
        
        # 优化器
        self.optimizer = optim.AdamW(
            self.model.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay
        )
        
        # 学习率调度器
        self.scheduler = optim.lr_scheduler.LinearLR(
            self.optimizer,
            start_factor=0.1,
            total_iters=warmup_steps
        )
        
        # 损失函数
        self.criterion = nn.MSELoss()
        
        # 训练历史
        self.history = {
            "train": {"loss": [], "mse": [], "mae": [], "r2": []},
            "val": {"loss": [], "mse": [], "mae": [], "r2": []}
        }
        
        # 最佳模型指标
        self.best_val_loss = float('inf')
        self.best_model_state = None
        
        # TensorBoard日志
        self.writer = SummaryWriter(log_dir=self.log_dir)
        
        logger.info(f"情感维度训练器初始化完成")
        logger.info(f"设备: {device}")
        logger.info(f"学习率: {learning_rate}")
        logger.info(f"权重衰减: {weight_decay}")
    
    def train_epoch(self) -> Dict[str, float]:
        """
        训练一个epoch
        
        Returns:
            metrics: 训练指标
        """
        self.model.train()
        total_loss = 0.0
        total_mse = 0.0
        total_mae = 0.0
        total_r2 = 0.0
        num_batches = 0
        
        # 获取情感维度列表
        emotion_dimensions = self.model.emotion_dimensions
        
        progress_bar = tqdm(self.train_loader, desc="训练中")
        
        for batch in progress_bar:
            # 移动数据到设备
            input_ids = batch['input_ids'].to(self.device)
            attention_mask = batch['attention_mask'].to(self.device)
            emotion_labels = batch['emotion_labels'].to(self.device)
            
            # 前向传播
            self.optimizer.zero_grad()
            predictions = self.model(input_ids, attention_mask)
            
            # 计算损失
            loss = self.criterion(predictions, emotion_labels)
            
            # 反向传播
            loss.backward()
            
            # 梯度裁剪
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            
            # 更新参数
            self.optimizer.step()
            self.scheduler.step()
            
            # 计算指标
            with torch.no_grad():
                mse = mean_squared_error(
                    emotion_labels.cpu().numpy(),
                    predictions.cpu().numpy()
                )
                mae = mean_absolute_error(
                    emotion_labels.cpu().numpy(),
                    predictions.cpu().numpy()
                )
                r2 = r2_score(
                    emotion_labels.cpu().numpy(),
                    predictions.cpu().numpy()
                )
            
            total_loss += loss.item()
            total_mse += mse
            total_mae += mae
            total_r2 += r2
            num_batches += 1
            
            # 更新进度条
            progress_bar.set_postfix({
                'loss': f'{loss.item():.4f}',
                'mse': f'{mse:.4f}',
                'mae': f'{mae:.4f}',
                'r2': f'{r2:.4f}'
            })
        
        # 计算平均指标
        avg_loss = total_loss / num_batches
        avg_mse = total_mse / num_batches
        avg_mae = total_mae / num_batches
        avg_r2 = total_r2 / num_batches
        
        return {
            "loss": avg_loss,
            "mse": avg_mse,
            "mae": avg_mae,
            "r2": avg_r2
        }
    
    def validate_epoch(self) -> Dict[str, float]:
        """
        验证一个epoch
        
        Returns:
            metrics: 验证指标
        """
        self.model.eval()
        total_loss = 0.0
        total_mse = 0.0
        total_mae = 0.0
        total_r2 = 0.0
        num_batches = 0
        
        # 获取情感维度列表
        emotion_dimensions = self.model.emotion_dimensions
        
        with torch.no_grad():
            for batch in tqdm(self.val_loader, desc="验证中"):
                # 移动数据到设备
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                emotion_labels = batch['emotion_labels'].to(self.device)
                
                # 前向传播
                predictions = self.model(input_ids, attention_mask)
                
                # 计算损失
                loss = self.criterion(predictions, emotion_labels)
                
                # 计算指标
                mse = mean_squared_error(
                    emotion_labels.cpu().numpy(),
                    predictions.cpu().numpy()
                )
                mae = mean_absolute_error(
                    emotion_labels.cpu().numpy(),
                    predictions.cpu().numpy()
                )
                r2 = r2_score(
                    emotion_labels.cpu().numpy(),
                    predictions.cpu().numpy()
                )
                
                total_loss += loss.item()
                total_mse += mse
                total_mae += mae
                total_r2 += r2
                num_batches += 1
        
        # 计算平均指标
        avg_loss = total_loss / num_batches
        avg_mse = total_mse / num_batches
        avg_mae = total_mae / num_batches
        avg_r2 = total_r2 / num_batches
        
        return {
            "loss": avg_loss,
            "mse": avg_mse,
            "mae": avg_mae,
            "r2": avg_r2
        }
    
    def train(self, num_epochs: int = 10, save_best: bool = True) -> Dict:
        """
        训练模型
        
        Args:
            num_epochs: 训练轮数
            save_best: 是否保存最佳模型
        
        Returns:
            history: 训练历史
        """
        logger.info(f"开始训练，共 {num_epochs} 个epoch")
        
        for epoch in range(num_epochs):
            logger.info(f"Epoch {epoch + 1}/{num_epochs}")
            
            # 训练
            train_metrics = self.train_epoch()
            
            # 验证
            val_metrics = self.validate_epoch()
            
            # 记录历史
            for key in train_metrics:
                self.history["train"][key].append(train_metrics[key])
                self.history["val"][key].append(val_metrics[key])
            
            # 记录到TensorBoard
            self.writer.add_scalar("Loss/Train", train_metrics["loss"], epoch)
            self.writer.add_scalar("Loss/Val", val_metrics["loss"], epoch)
            self.writer.add_scalar("MSE/Train", train_metrics["mse"], epoch)
            self.writer.add_scalar("MSE/Val", val_metrics["mse"], epoch)
            self.writer.add_scalar("MAE/Train", train_metrics["mae"], epoch)
            self.writer.add_scalar("MAE/Val", val_metrics["mae"], epoch)
            self.writer.add_scalar("R2/Train", train_metrics["r2"], epoch)
            self.writer.add_scalar("R2/Val", val_metrics["r2"], epoch)
            
            # 记录学习率
            self.writer.add_scalar("Learning_Rate", self.optimizer.param_groups[0]['lr'], epoch)
            
            # 打印指标
            logger.info(f"训练 - Loss: {train_metrics['loss']:.4f}, MSE: {train_metrics['mse']:.4f}, "
                       f"MAE: {train_metrics['mae']:.4f}, R2: {train_metrics['r2']:.4f}")
            logger.info(f"验证 - Loss: {val_metrics['loss']:.4f}, MSE: {val_metrics['mse']:.4f}, "
                       f"MAE: {val_metrics['mae']:.4f}, R2: {val_metrics['r2']:.4f}")
            
            # 保存最佳模型
            if save_best and val_metrics["loss"] < self.best_val_loss:
                self.best_val_loss = val_metrics["loss"]
                self.best_model_state = self.model.state_dict().copy()
                
                # 保存模型
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    'scheduler_state_dict': self.scheduler.state_dict(),
                    'val_loss': val_metrics["loss"],
                    'val_mse': val_metrics["mse"],
                    'val_mae': val_metrics["mae"],
                    'val_r2': val_metrics["r2"],
                    'emotion_dimensions': self.model.emotion_dimensions
                }, self.save_dir / "best_model.pt")
                
                logger.info(f"保存最佳模型，验证损失: {val_metrics['loss']:.4f}")
        
        # 加载最佳模型
        if save_best and self.best_model_state is not None:
            self.model.load_state_dict(self.best_model_state)
            logger.info("已加载最佳模型")
        
        # 关闭TensorBoard
        self.writer.close()
        
        logger.info("训练完成！")
        return self.history
    
    def evaluate(self) -> Dict[str, float]:
        """
        评估模型
        
        Returns:
            metrics: 评估指标
        """
        logger.info("开始评估模型...")
        
        self.model.eval()
        total_loss = 0.0
        total_mse = 0.0
        total_mae = 0.0
        total_r2 = 0.0
        num_batches = 0
        
        # 获取情感维度列表
        emotion_dimensions = self.model.emotion_dimensions
        
        # 存储所有预测和真实值
        all_predictions = []
        all_labels = []
        
        with torch.no_grad():
            for batch in tqdm(self.test_loader, desc="评估中"):
                # 移动数据到设备
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                emotion_labels = batch['emotion_labels'].to(self.device)
                
                # 前向传播
                predictions = self.model(input_ids, attention_mask)
                
                # 计算损失
                loss = self.criterion(predictions, emotion_labels)
                
                # 计算指标
                mse = mean_squared_error(
                    emotion_labels.cpu().numpy(),
                    predictions.cpu().numpy()
                )
                mae = mean_absolute_error(
                    emotion_labels.cpu().numpy(),
                    predictions.cpu().numpy()
                )
                r2 = r2_score(
                    emotion_labels.cpu().numpy(),
                    predictions.cpu().numpy()
                )
                
                total_loss += loss.item()
                total_mse += mse
                total_mae += mae
                total_r2 += r2
                num_batches += 1
                
                # 存储预测和真实值
                all_predictions.append(predictions.cpu().numpy())
                all_labels.append(emotion_labels.cpu().numpy())
        
        # 计算平均指标
        avg_loss = total_loss / num_batches
        avg_mse = total_mse / num_batches
        avg_mae = total_mae / num_batches
        avg_r2 = total_r2 / num_batches
        
        # 计算各维度的指标
        all_predictions = np.concatenate(all_predictions, axis=0)
        all_labels = np.concatenate(all_labels, axis=0)
        
        dimension_metrics = {}
        for i, dimension in enumerate(emotion_dimensions):
            dim_predictions = all_predictions[:, i]
            dim_labels = all_labels[:, i]
            
            dimension_metrics[dimension] = {
                "mse": mean_squared_error(dim_labels, dim_predictions),
                "mae": mean_absolute_error(dim_labels, dim_predictions),
                "r2": r2_score(dim_labels, dim_predictions)
            }
        
        # 保存评估结果
        evaluation_results = {
            "overall": {
                "loss": avg_loss,
                "mse": avg_mse,
                "mae": avg_mae,
                "r2": avg_r2
            },
            "dimensions": dimension_metrics
        }
        
        # 打印结果
        logger.info("评估结果:")
        logger.info(f"  总体 - Loss: {avg_loss:.4f}, MSE: {avg_mse:.4f}, "
                   f"MAE: {avg_mae:.4f}, R2: {avg_r2:.4f}")
        
        for dimension, metrics in dimension_metrics.items():
            logger.info(f"  {dimension} - MSE: {metrics['mse']:.4f}, "
                       f"MAE: {metrics['mae']:.4f}, R2: {metrics['r2']:.4f}")
        
        return evaluation_results
    
    def save_model(self, filepath: str):
        """
        保存模型
        
        Args:
            filepath: 保存路径
        """
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'emotion_dimensions': self.model.emotion_dimensions,
            'history': self.history
        }, filepath)
        
        logger.info(f"模型已保存到: {filepath}")
    
    def load_model(self, filepath: str):
        """
        加载模型
        
        Args:
            filepath: 模型路径
        """
        checkpoint = torch.load(filepath, map_location=self.device)
        
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        
        if 'history' in checkpoint:
            self.history = checkpoint['history']
        
        logger.info(f"模型已从 {filepath} 加载")


class MultiTaskEmotionTrainer(EmotionDimensionTrainer):
    """
    多任务情感模型训练器
    同时训练分类和回归任务
    """
    
    def __init__(
        self,
        model: nn.Module,
        train_loader,
        val_loader,
        test_loader,
        device: str = "cuda",
        learning_rate: float = 2e-5,
        weight_decay: float = 0.01,
        warmup_steps: int = 500,
        max_grad_norm: float = 1.0,
        save_dir: str = "checkpoints",
        log_dir: str = "logs",
        task_weights: Dict[str, float] = None
    ):
        super().__init__(
            model, train_loader, val_loader, test_loader,
            device, learning_rate, weight_decay, warmup_steps,
            max_grad_norm, save_dir, log_dir
        )
        
        # 设置任务权重
        if task_weights is None:
            self.task_weights = {
                "classification": 1.0,
                "regression": 1.0
            }
        else:
            self.task_weights = task_weights
        
        # 分类损失函数
        self.classification_criterion = nn.CrossEntropyLoss()
        
        logger.info(f"多任务训练器初始化完成")
        logger.info(f"任务权重: {self.task_weights}")
    
    def train_epoch(self) -> Dict[str, float]:
        """
        训练一个epoch（多任务）
        """
        self.model.train()
        total_loss = 0.0
        total_classification_loss = 0.0
        total_regression_loss = 0.0
        total_mse = 0.0
        total_mae = 0.0
        total_r2 = 0.0
        total_accuracy = 0.0
        num_batches = 0
        
        progress_bar = tqdm(self.train_loader, desc="训练中")
        
        for batch in progress_bar:
            # 移动数据到设备
            input_ids = batch['input_ids'].to(self.device)
            attention_mask = batch['attention_mask'].to(self.device)
            
            # 检查是否有分类标签
            if 'labels' in batch:
                classification_labels = batch['labels'].to(self.device)
            else:
                classification_labels = None
            
            # 检查是否有情感维度标签
            if 'emotion_labels' in batch:
                emotion_labels = batch['emotion_labels'].to(self.device)
            else:
                emotion_labels = None
            
            # 前向传播
            self.optimizer.zero_grad()
            
            if classification_labels is not None and emotion_labels is not None:
                # 多任务训练
                classification_logits, regression_predictions = self.model(input_ids, attention_mask)
                
                # 计算分类损失
                classification_loss = self.classification_criterion(classification_logits, classification_labels)
                
                # 计算回归损失
                regression_loss = self.criterion(regression_predictions, emotion_labels)
                
                # 总损失
                total_loss_batch = (self.task_weights["classification"] * classification_loss + 
                                  self.task_weights["regression"] * regression_loss)
                
                # 计算分类准确率
                predictions = torch.argmax(classification_logits, dim=1)
                accuracy = (predictions == classification_labels).float().mean()
                
            elif classification_labels is not None:
                # 仅分类任务
                classification_logits, _ = self.model(input_ids, attention_mask)
                classification_loss = self.classification_criterion(classification_logits, classification_labels)
                total_loss_batch = classification_loss
                regression_loss = 0.0
                
                predictions = torch.argmax(classification_logits, dim=1)
                accuracy = (predictions == classification_labels).float().mean()
                
            elif emotion_labels is not None:
                # 仅回归任务
                _, regression_predictions = self.model(input_ids, attention_mask)
                regression_loss = self.criterion(regression_predictions, emotion_labels)
                total_loss_batch = regression_loss
                classification_loss = 0.0
                accuracy = 0.0
            
            # 反向传播
            total_loss_batch.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.optimizer.step()
            self.scheduler.step()
            
            # 计算回归指标
            if emotion_labels is not None:
                mse = mean_squared_error(
                    emotion_labels.cpu().numpy(),
                    regression_predictions.cpu().numpy()
                )
                mae = mean_absolute_error(
                    emotion_labels.cpu().numpy(),
                    regression_predictions.cpu().numpy()
                )
                r2 = r2_score(
                    emotion_labels.cpu().numpy(),
                    regression_predictions.cpu().numpy()
                )
            else:
                mse = mae = r2 = 0.0
            
            total_loss += total_loss_batch.item()
            total_classification_loss += classification_loss if isinstance(classification_loss, float) else classification_loss.item()
            total_regression_loss += regression_loss if isinstance(regression_loss, float) else regression_loss.item()
            total_mse += mse
            total_mae += mae
            total_r2 += r2
            total_accuracy += accuracy
            num_batches += 1
            
            # 更新进度条
            progress_bar.set_postfix({
                'loss': f'{total_loss_batch.item():.4f}',
                'cls_loss': f'{classification_loss if isinstance(classification_loss, float) else classification_loss.item():.4f}',
                'reg_loss': f'{regression_loss if isinstance(regression_loss, float) else regression_loss.item():.4f}',
                'acc': f'{accuracy:.4f}',
                'mse': f'{mse:.4f}'
            })
        
        # 计算平均指标
        avg_loss = total_loss / num_batches
        avg_classification_loss = total_classification_loss / num_batches
        avg_regression_loss = total_regression_loss / num_batches
        avg_mse = total_mse / num_batches
        avg_mae = total_mae / num_batches
        avg_r2 = total_r2 / num_batches
        avg_accuracy = total_accuracy / num_batches
        
        return {
            "loss": avg_loss,
            "classification_loss": avg_classification_loss,
            "regression_loss": avg_regression_loss,
            "mse": avg_mse,
            "mae": avg_mae,
            "r2": avg_r2,
            "accuracy": avg_accuracy
        }


if __name__ == "__main__":
    # 测试训练器
    print("测试情感维度训练器...")
    
    # 这里需要实际的模型和数据加载器
    # 在实际使用中，这些会从外部传入
    print("训练器测试完成！")

