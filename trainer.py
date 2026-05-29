"""
训练和评估模块
支持Transformer模型的训练、验证和测试
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
import numpy as np
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix, classification_report
import time
import os
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import logging
import json
from tqdm import tqdm

logger = logging.getLogger(__name__)


class EarlyStopping:
    """早停机制"""
    
    def __init__(self, patience: int = 7, min_delta: float = 0.0, restore_best_weights: bool = True):
        self.patience = patience
        self.min_delta = min_delta
        self.restore_best_weights = restore_best_weights
        self.best_loss = None
        self.counter = 0
        self.best_weights = None
        
    def __call__(self, val_loss: float, model: nn.Module) -> bool:
        if self.best_loss is None:
            self.best_loss = val_loss
            self.save_checkpoint(model)
        elif val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.counter = 0
            self.save_checkpoint(model)
        else:
            self.counter += 1
            
        if self.counter >= self.patience:
            if self.restore_best_weights:
                model.load_state_dict(self.best_weights)
            return True
        return False
    
    def save_checkpoint(self, model: nn.Module):
        """保存最佳模型权重"""
        self.best_weights = model.state_dict().copy()


class ModelTrainer:
    """模型训练器"""
    
    def __init__(
        self,
        model: nn.Module,
        train_loader,
        val_loader,
        test_loader,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
        learning_rate: float = 2e-5,
        weight_decay: float = 0.01,
        warmup_steps: int = 500,
        max_grad_norm: float = 1.0,
        save_dir: str = "checkpoints"
    ):
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.test_loader = test_loader
        self.device = device
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)
        
        # 优化器设置
        self.optimizer = optim.AdamW(
            self.model.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay
        )
        
        # 学习率调度器
        total_steps = len(train_loader) * 10  # 假设训练10个epoch
        self.scheduler = optim.lr_scheduler.LinearLR(
            self.optimizer,
            start_factor=0.1,
            total_iters=warmup_steps
        )
        
        # 损失函数
        self.criterion = nn.CrossEntropyLoss()
        
        # 梯度裁剪
        self.max_grad_norm = max_grad_norm
        
        # TensorBoard日志
        self.writer = SummaryWriter(log_dir=str(self.save_dir / "logs"))
        
        # 早停
        self.early_stopping = EarlyStopping(patience=5)
        
        logger.info(f"训练器初始化完成，设备: {device}")
        logger.info(f"学习率: {learning_rate}, 权重衰减: {weight_decay}")
    
    def train_epoch(self, epoch: int) -> Dict[str, float]:
        """训练一个epoch"""
        self.model.train()
        total_loss = 0
        all_predictions = []
        all_labels = []
        
        progress_bar = tqdm(self.train_loader, desc=f"Epoch {epoch+1} [训练]")
        
        for batch_idx, batch in enumerate(progress_bar):
            # 数据移到设备
            input_ids = batch['input_ids'].to(self.device)
            attention_mask = batch['attention_mask'].to(self.device)
            labels = batch['labels'].to(self.device)
            
            # 前向传播
            self.optimizer.zero_grad()
            logits = self.model(input_ids, attention_mask)
            loss = self.criterion(logits, labels)
            
            # 反向传播
            loss.backward()
            
            # 梯度裁剪
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)
            
            self.optimizer.step()
            self.scheduler.step()
            
            # 统计
            total_loss += loss.item()
            predictions = torch.argmax(logits, dim=1)
            all_predictions.extend(predictions.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            
            # 更新进度条
            progress_bar.set_postfix({
                'loss': f'{loss.item():.4f}',
                'lr': f'{self.optimizer.param_groups[0]["lr"]:.2e}'
            })
        
        # 计算指标
        avg_loss = total_loss / len(self.train_loader)
        accuracy = accuracy_score(all_labels, all_predictions)
        
        return {
            'loss': avg_loss,
            'accuracy': accuracy
        }
    
    def validate_epoch(self, epoch: int) -> Dict[str, float]:
        """验证一个epoch"""
        self.model.eval()
        total_loss = 0
        all_predictions = []
        all_labels = []
        
        with torch.no_grad():
            progress_bar = tqdm(self.val_loader, desc=f"Epoch {epoch+1} [验证]")
            
            for batch in progress_bar:
                # 数据移到设备
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                labels = batch['labels'].to(self.device)
                
                # 前向传播
                logits = self.model(input_ids, attention_mask)
                loss = self.criterion(logits, labels)
                
                # 统计
                total_loss += loss.item()
                predictions = torch.argmax(logits, dim=1)
                all_predictions.extend(predictions.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
                
                progress_bar.set_postfix({'loss': f'{loss.item():.4f}'})
        
        # 计算指标
        avg_loss = total_loss / len(self.val_loader)
        accuracy = accuracy_score(all_labels, all_predictions)
        precision, recall, f1, _ = precision_recall_fscore_support(
            all_labels, all_predictions, average='weighted'
        )
        
        return {
            'loss': avg_loss,
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1': f1
        }
    
    def train(self, num_epochs: int = 10, save_best: bool = True) -> Dict[str, List[float]]:
        """完整训练流程"""
        logger.info(f"开始训练，共 {num_epochs} 个epoch")
        
        train_history = {
            'loss': [],
            'accuracy': []
        }
        val_history = {
            'loss': [],
            'accuracy': [],
            'precision': [],
            'recall': [],
            'f1': []
        }
        
        best_val_loss = float('inf')
        
        for epoch in range(num_epochs):
            start_time = time.time()
            
            # 训练
            train_metrics = self.train_epoch(epoch)
            train_history['loss'].append(train_metrics['loss'])
            train_history['accuracy'].append(train_metrics['accuracy'])
            
            # 验证
            val_metrics = self.validate_epoch(epoch)
            val_history['loss'].append(val_metrics['loss'])
            val_history['accuracy'].append(val_metrics['accuracy'])
            val_history['precision'].append(val_metrics['precision'])
            val_history['recall'].append(val_metrics['recall'])
            val_history['f1'].append(val_metrics['f1'])
            
            # 记录到TensorBoard
            self.writer.add_scalar('Loss/Train', train_metrics['loss'], epoch)
            self.writer.add_scalar('Loss/Val', val_metrics['loss'], epoch)
            self.writer.add_scalar('Accuracy/Train', train_metrics['accuracy'], epoch)
            self.writer.add_scalar('Accuracy/Val', val_metrics['accuracy'], epoch)
            self.writer.add_scalar('F1/Val', val_metrics['f1'], epoch)
            
            # 保存最佳模型
            if save_best and val_metrics['loss'] < best_val_loss:
                best_val_loss = val_metrics['loss']
                self.save_model(f"best_model_epoch_{epoch+1}.pt")
            
            # 早停检查
            if self.early_stopping(val_metrics['loss'], self.model):
                logger.info(f"早停触发，在第 {epoch+1} 个epoch停止训练")
                break
            
            epoch_time = time.time() - start_time
            logger.info(
                f"Epoch {epoch+1}/{num_epochs} - "
                f"训练损失: {train_metrics['loss']:.4f}, "
                f"训练准确率: {train_metrics['accuracy']:.4f}, "
                f"验证损失: {val_metrics['loss']:.4f}, "
                f"验证准确率: {val_metrics['accuracy']:.4f}, "
                f"验证F1: {val_metrics['f1']:.4f}, "
                f"时间: {epoch_time:.2f}s"
            )
        
        # 保存训练历史
        self.save_training_history(train_history, val_history)
        
        logger.info("训练完成！")
        return {
            'train': train_history,
            'val': val_history
        }
    
    def evaluate(self, data_loader=None) -> Dict[str, float]:
        """评估模型"""
        if data_loader is None:
            data_loader = self.test_loader
        
        self.model.eval()
        all_predictions = []
        all_labels = []
        total_loss = 0
        
        with torch.no_grad():
            for batch in tqdm(data_loader, desc="评估中"):
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                labels = batch['labels'].to(self.device)
                
                logits = self.model(input_ids, attention_mask)
                loss = self.criterion(logits, labels)
                
                total_loss += loss.item()
                predictions = torch.argmax(logits, dim=1)
                all_predictions.extend(predictions.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
        
        # 计算详细指标
        accuracy = accuracy_score(all_labels, all_predictions)
        precision, recall, f1, _ = precision_recall_fscore_support(
            all_labels, all_predictions, average='weighted'
        )
        
        # 混淆矩阵
        cm = confusion_matrix(all_labels, all_predictions)
        
        # 分类报告
        report = classification_report(
            all_labels, all_predictions,
            target_names=[f'Class_{i}' for i in range(len(np.unique(all_labels)))]
        )
        
        metrics = {
            'loss': total_loss / len(data_loader),
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'confusion_matrix': cm.tolist(),
            'classification_report': report
        }
        
        logger.info(f"评估结果:")
        logger.info(f"  准确率: {accuracy:.4f}")
        logger.info(f"  精确率: {precision:.4f}")
        logger.info(f"  召回率: {recall:.4f}")
        logger.info(f"  F1分数: {f1:.4f}")
        
        return metrics
    
    def save_model(self, filename: str):
        """保存模型"""
        save_path = self.save_dir / filename
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
        }, save_path)
        logger.info(f"模型已保存到: {save_path}")
    
    def load_model(self, filename: str):
        """加载模型"""
        load_path = self.save_dir / filename
        checkpoint = torch.load(load_path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        logger.info(f"模型已从 {load_path} 加载")
    
    def save_training_history(self, train_history: Dict, val_history: Dict):
        """保存训练历史"""
        history = {
            'train': train_history,
            'val': val_history
        }
        
        with open(self.save_dir / "training_history.json", 'w', encoding='utf-8') as f:
            json.dump(history, f, indent=2, ensure_ascii=False)
        
        logger.info("训练历史已保存")
    
    def close(self):
        """关闭TensorBoard写入器"""
        self.writer.close()


def train_model(
    model,
    train_loader,
    val_loader,
    test_loader,
    num_epochs: int = 10,
    learning_rate: float = 2e-5,
    device: str = None,
    save_dir: str = "checkpoints"
) -> ModelTrainer:
    """
    训练模型的便捷函数
    
    Args:
        model: 要训练的模型
        train_loader, val_loader, test_loader: 数据加载器
        num_epochs: 训练轮数
        learning_rate: 学习率
        device: 设备
        save_dir: 保存目录
    
    Returns:
        trainer: 训练器对象
    """
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # 创建训练器
    trainer = ModelTrainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        test_loader=test_loader,
        device=device,
        learning_rate=learning_rate,
        save_dir=save_dir
    )
    
    # 开始训练
    trainer.train(num_epochs=num_epochs)
    
    # 评估测试集
    test_metrics = trainer.evaluate()
    
    return trainer


if __name__ == "__main__":
    # 测试训练器
    print("测试训练器模块...")
    
    # 这里需要实际的模型和数据加载器来测试
    # 在实际使用中，这些会从其他模块导入
    print("训练器模块测试完成！")
