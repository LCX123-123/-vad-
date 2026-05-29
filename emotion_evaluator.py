"""
情感维度模型评估器
提供全面的情感维度模型评估指标和可视化功能
"""

import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    mean_squared_error, mean_absolute_error, r2_score,
    explained_variance_score, max_error
)
from typing import Dict, List, Tuple, Optional, Union
import logging
from pathlib import Path
import json

logger = logging.getLogger(__name__)


class EmotionDimensionEvaluator:
    """
    情感维度模型评估器
    提供全面的评估指标和可视化功能
    """
    
    def __init__(
        self,
        model: torch.nn.Module,
        emotion_dimensions: List[str],
        device: str = "cuda"
    ):
        self.model = model
        self.emotion_dimensions = emotion_dimensions
        self.device = device
        self.model.to(device)
        
        logger.info(f"情感维度评估器初始化完成")
        logger.info(f"情感维度: {emotion_dimensions}")
    
    def evaluate_dataset(
        self,
        data_loader,
        save_results: bool = True,
        results_path: str = "evaluation_results.json"
    ) -> Dict:
        """
        评估整个数据集
        
        Args:
            data_loader: 数据加载器
            save_results: 是否保存结果
            results_path: 结果保存路径
        
        Returns:
            results: 评估结果字典
        """
        logger.info("开始评估数据集...")
        
        self.model.eval()
        all_predictions = []
        all_labels = []
        all_texts = []
        
        with torch.no_grad():
            for batch in data_loader:
                # 移动数据到设备
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                emotion_labels = batch['emotion_labels'].to(self.device)
                
                # 前向传播
                predictions = self.model(input_ids, attention_mask)
                
                # 存储结果
                all_predictions.append(predictions.cpu().numpy())
                all_labels.append(emotion_labels.cpu().numpy())
                
                # 如果有文本信息，也存储
                if 'texts' in batch:
                    all_texts.extend(batch['texts'])
        
        # 合并所有结果
        all_predictions = np.concatenate(all_predictions, axis=0)
        all_labels = np.concatenate(all_labels, axis=0)
        
        # 计算评估指标
        results = self._calculate_metrics(all_predictions, all_labels)
        
        # 保存结果
        if save_results:
            self._save_results(results, results_path)
        
        logger.info("数据集评估完成")
        return results
    
    def _calculate_metrics(
        self,
        predictions: np.ndarray,
        labels: np.ndarray
    ) -> Dict:
        """
        计算评估指标
        
        Args:
            predictions: 预测值
            labels: 真实值
        
        Returns:
            metrics: 评估指标字典
        """
        results = {
            "overall": {},
            "dimensions": {},
            "correlations": {}
        }
        
        # 总体指标
        results["overall"] = {
            "mse": float(mean_squared_error(labels, predictions)),
            "rmse": float(np.sqrt(mean_squared_error(labels, predictions))),
            "mae": float(mean_absolute_error(labels, predictions)),
            "r2": float(r2_score(labels, predictions)),
            "explained_variance": float(explained_variance_score(labels, predictions)),
            # sklearn.max_error 不支持 multioutput，这里改为逐元素误差的全局最大值
            "max_error": float(np.max(np.abs(labels - predictions)))
        }
        
        # 各维度指标
        for i, dimension in enumerate(self.emotion_dimensions):
            dim_predictions = predictions[:, i]
            dim_labels = labels[:, i]
            
            results["dimensions"][dimension] = {
                "mse": float(mean_squared_error(dim_labels, dim_predictions)),
                "rmse": float(np.sqrt(mean_squared_error(dim_labels, dim_predictions))),
                "mae": float(mean_absolute_error(dim_labels, dim_predictions)),
                "r2": float(r2_score(dim_labels, dim_predictions)),
                "explained_variance": float(explained_variance_score(dim_labels, dim_predictions)),
                "max_error": float(np.max(np.abs(dim_labels - dim_predictions))),
                "mean_absolute_percentage_error": float(self._calculate_mape(dim_labels, dim_predictions)),
                "symmetric_mean_absolute_percentage_error": float(self._calculate_smape(dim_labels, dim_predictions))
            }
        
        # 计算维度间相关性
        results["correlations"] = self._calculate_correlations(predictions, labels)
        
        return results
    
    def _calculate_mape(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """
        计算平均绝对百分比误差。
        仅对 |y_true| >= 1e-6 的样本计算，避免接近零时的数值爆炸；单样本贡献上限 200%。
        """
        eps = 1e-6
        mask = np.abs(y_true) >= eps
        if not np.any(mask):
            return 0.0
        ratio = np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])
        ratio = np.minimum(ratio, 2.0)  # 单样本最多 200%，避免异常值拉高
        return float(np.mean(ratio) * 100)
    
    def _calculate_smape(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """
        计算对称平均绝对百分比误差。
        分母过小时用 epsilon 兜底，单样本贡献上限 100%，避免数值异常。
        """
        numerator = np.abs(y_true - y_pred)
        denominator = (np.abs(y_true) + np.abs(y_pred)) / 2.0
        eps = 1e-10
        denominator = np.maximum(denominator, eps)
        ratio = numerator / denominator
        ratio = np.minimum(ratio, 1.0)  # 单样本最多 100%
        return float(np.mean(ratio) * 100)
    
    def _calculate_correlations(
        self,
        predictions: np.ndarray,
        labels: np.ndarray
    ) -> Dict:
        """
        计算维度间相关性
        
        Args:
            predictions: 预测值
            labels: 真实值
        
        Returns:
            correlations: 相关性字典
        """
        correlations = {}
        
        # 预测值维度间相关性
        pred_corr = np.corrcoef(predictions.T)
        correlations["predictions"] = {
            "matrix": pred_corr.tolist(),
            "dimensions": self.emotion_dimensions
        }
        
        # 真实值维度间相关性
        true_corr = np.corrcoef(labels.T)
        correlations["labels"] = {
            "matrix": true_corr.tolist(),
            "dimensions": self.emotion_dimensions
        }
        
        # 预测值与真实值的相关性
        pred_true_corr = []
        for i in range(len(self.emotion_dimensions)):
            corr = np.corrcoef(predictions[:, i], labels[:, i])[0, 1]
            pred_true_corr.append(float(corr))
        
        correlations["prediction_accuracy"] = {
            "correlations": pred_true_corr,
            "dimensions": self.emotion_dimensions
        }
        
        return correlations
    
    def _save_results(self, results: Dict, filepath: str):
        """
        保存评估结果
        
        Args:
            results: 评估结果
            filepath: 保存路径
        """
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        
        logger.info(f"评估结果已保存到: {filepath}")
    
    def plot_predictions_vs_labels(
        self,
        predictions: np.ndarray,
        labels: np.ndarray,
        save_path: str = "predictions_vs_labels.png"
    ):
        """
        绘制预测值vs真实值散点图
        
        Args:
            predictions: 预测值
            labels: 真实值
            save_path: 保存路径
        """
        fig, axes = plt.subplots(1, len(self.emotion_dimensions), figsize=(5 * len(self.emotion_dimensions), 5))
        
        if len(self.emotion_dimensions) == 1:
            axes = [axes]
        
        for i, dimension in enumerate(self.emotion_dimensions):
            ax = axes[i]
            
            # 散点图
            ax.scatter(labels[:, i], predictions[:, i], alpha=0.6, s=20)
            
            # 完美预测线
            min_val = min(labels[:, i].min(), predictions[:, i].min())
            max_val = max(labels[:, i].max(), predictions[:, i].max())
            ax.plot([min_val, max_val], [min_val, max_val], 'r--', alpha=0.8, label='完美预测')
            
            # 设置标签和标题
            ax.set_xlabel(f'真实 {dimension}')
            ax.set_ylabel(f'预测 {dimension}')
            ax.set_title(f'{dimension} 预测 vs 真实')
            ax.legend()
            ax.grid(True, alpha=0.3)
            
            # 计算R²
            r2 = r2_score(labels[:, i], predictions[:, i])
            ax.text(0.05, 0.95, f'R² = {r2:.3f}', transform=ax.transAxes, 
                   bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info(f"预测vs真实值图已保存到: {save_path}")
    
    def plot_residuals(
        self,
        predictions: np.ndarray,
        labels: np.ndarray,
        save_path: str = "residuals.png"
    ):
        """
        绘制残差图
        
        Args:
            predictions: 预测值
            labels: 真实值
            save_path: 保存路径
        """
        fig, axes = plt.subplots(1, len(self.emotion_dimensions), figsize=(5 * len(self.emotion_dimensions), 5))
        
        if len(self.emotion_dimensions) == 1:
            axes = [axes]
        
        for i, dimension in enumerate(self.emotion_dimensions):
            ax = axes[i]
            
            # 计算残差
            residuals = predictions[:, i] - labels[:, i]
            
            # 残差散点图
            ax.scatter(labels[:, i], residuals, alpha=0.6, s=20)
            
            # 零线
            ax.axhline(y=0, color='r', linestyle='--', alpha=0.8)
            
            # 设置标签和标题
            ax.set_xlabel(f'真实 {dimension}')
            ax.set_ylabel(f'残差 ({dimension})')
            ax.set_title(f'{dimension} 残差图')
            ax.grid(True, alpha=0.3)
            
            # 计算残差统计
            mean_residual = np.mean(residuals)
            std_residual = np.std(residuals)
            ax.text(0.05, 0.95, f'均值: {mean_residual:.3f}\n标准差: {std_residual:.3f}', 
                   transform=ax.transAxes, bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info(f"残差图已保存到: {save_path}")
    
    def plot_correlation_heatmap(
        self,
        predictions: np.ndarray,
        labels: np.ndarray,
        save_path: str = "correlation_heatmap.png"
    ):
        """
        绘制相关性热力图
        
        Args:
            predictions: 预测值
            labels: 真实值
            save_path: 保存路径
        """
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        
        # 预测值相关性
        pred_corr = np.corrcoef(predictions.T)
        sns.heatmap(pred_corr, annot=True, cmap='coolwarm', center=0,
                   xticklabels=self.emotion_dimensions, yticklabels=self.emotion_dimensions,
                   ax=axes[0])
        axes[0].set_title('预测值维度间相关性')
        
        # 真实值相关性
        true_corr = np.corrcoef(labels.T)
        sns.heatmap(true_corr, annot=True, cmap='coolwarm', center=0,
                   xticklabels=self.emotion_dimensions, yticklabels=self.emotion_dimensions,
                   ax=axes[1])
        axes[1].set_title('真实值维度间相关性')
        
        # 预测值与真实值的相关性
        pred_true_corr = np.zeros((len(self.emotion_dimensions), len(self.emotion_dimensions)))
        for i in range(len(self.emotion_dimensions)):
            pred_true_corr[i, i] = np.corrcoef(predictions[:, i], labels[:, i])[0, 1]
        
        sns.heatmap(pred_true_corr, annot=True, cmap='coolwarm', center=0,
                   xticklabels=self.emotion_dimensions, yticklabels=self.emotion_dimensions,
                   ax=axes[2])
        axes[2].set_title('预测值与真实值相关性')
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info(f"相关性热力图已保存到: {save_path}")
    
    def plot_error_distribution(
        self,
        predictions: np.ndarray,
        labels: np.ndarray,
        save_path: str = "error_distribution.png"
    ):
        """
        绘制误差分布图
        
        Args:
            predictions: 预测值
            labels: 真实值
            save_path: 保存路径
        """
        fig, axes = plt.subplots(1, len(self.emotion_dimensions), figsize=(5 * len(self.emotion_dimensions), 5))
        
        if len(self.emotion_dimensions) == 1:
            axes = [axes]
        
        for i, dimension in enumerate(self.emotion_dimensions):
            ax = axes[i]
            
            # 计算误差
            errors = predictions[:, i] - labels[:, i]
            
            # 绘制直方图
            ax.hist(errors, bins=30, alpha=0.7, density=True, color='skyblue', edgecolor='black')
            
            # 添加正态分布拟合
            mu, sigma = np.mean(errors), np.std(errors)
            x = np.linspace(errors.min(), errors.max(), 100)
            y = (1 / (sigma * np.sqrt(2 * np.pi))) * np.exp(-0.5 * ((x - mu) / sigma) ** 2)
            ax.plot(x, y, 'r-', linewidth=2, label=f'正态分布拟合\nμ={mu:.3f}, σ={sigma:.3f}')
            
            # 设置标签和标题
            ax.set_xlabel(f'误差 ({dimension})')
            ax.set_ylabel('密度')
            ax.set_title(f'{dimension} 误差分布')
            ax.legend()
            ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info(f"误差分布图已保存到: {save_path}")
    
    def generate_comprehensive_report(
        self,
        data_loader,
        save_dir: str = "evaluation_report"
    ) -> Dict:
        """
        生成综合评估报告
        
        Args:
            data_loader: 数据加载器
            save_dir: 保存目录
        
        Returns:
            results: 评估结果
        """
        logger.info("生成综合评估报告...")
        
        # 创建保存目录
        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)
        
        # 评估数据集
        results = self.evaluate_dataset(data_loader, save_results=True, 
                                      results_path=save_dir / "evaluation_results.json")
        
        # 获取预测和真实值用于绘图
        self.model.eval()
        all_predictions = []
        all_labels = []
        
        with torch.no_grad():
            for batch in data_loader:
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                emotion_labels = batch['emotion_labels'].to(self.device)
                
                predictions = self.model(input_ids, attention_mask)
                
                all_predictions.append(predictions.cpu().numpy())
                all_labels.append(emotion_labels.cpu().numpy())
        
        all_predictions = np.concatenate(all_predictions, axis=0)
        all_labels = np.concatenate(all_labels, axis=0)
        
        # 生成各种图表
        self.plot_predictions_vs_labels(all_predictions, all_labels, 
                                      save_path=save_dir / "predictions_vs_labels.png")
        self.plot_residuals(all_predictions, all_labels, 
                          save_path=save_dir / "residuals.png")
        self.plot_correlation_heatmap(all_predictions, all_labels, 
                                    save_path=save_dir / "correlation_heatmap.png")
        self.plot_error_distribution(all_predictions, all_labels, 
                                   save_path=save_dir / "error_distribution.png")
        
        # 生成文本报告
        self._generate_text_report(results, save_dir / "evaluation_report.txt")
        
        logger.info(f"综合评估报告已生成到: {save_dir}")
        return results
    
    def _generate_text_report(self, results: Dict, filepath: str):
        """
        生成文本报告
        
        Args:
            results: 评估结果
            filepath: 保存路径
        """
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write("情感维度模型评估报告\n")
            f.write("=" * 50 + "\n\n")
            
            # 总体指标
            f.write("总体指标:\n")
            f.write("-" * 20 + "\n")
            for metric, value in results["overall"].items():
                f.write(f"{metric}: {value:.4f}\n")
            f.write("\n")
            
            # 各维度指标
            f.write("各维度指标:\n")
            f.write("-" * 20 + "\n")
            for dimension, metrics in results["dimensions"].items():
                f.write(f"\n{dimension}:\n")
                for metric, value in metrics.items():
                    f.write(f"  {metric}: {value:.4f}\n")
            
            # 相关性分析
            f.write("\n相关性分析:\n")
            f.write("-" * 20 + "\n")
            f.write("预测值与真实值相关性:\n")
            for i, dimension in enumerate(self.emotion_dimensions):
                corr = results["correlations"]["prediction_accuracy"]["correlations"][i]
                f.write(f"  {dimension}: {corr:.4f}\n")
        
        logger.info(f"文本报告已保存到: {filepath}")


if __name__ == "__main__":
    # 测试评估器
    print("测试情感维度评估器...")
    
    # 这里需要实际的模型和数据加载器
    # 在实际使用中，这些会从外部传入
    print("评估器测试完成！")

