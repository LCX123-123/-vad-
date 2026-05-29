"""
模型推理和预测模块
支持单文本预测、批量预测和交互式预测
"""

import torch
import torch.nn.functional as F
import numpy as np
from typing import List, Dict, Union, Optional
import logging
from pathlib import Path
import json
import pandas as pd
from sklearn.preprocessing import LabelEncoder

from transformer_model import TransformerClassifier
from data_loader import DataProcessor

logger = logging.getLogger(__name__)


class SentimentPredictor:
    """情感分析预测器"""
    
    def __init__(
        self,
        model_path: str,
        model_name: str = "bert-base-uncased",
        device: str = None,
        label_mapping: Optional[Dict] = None
    ):
        """
        初始化预测器
        
        Args:
            model_path: 模型权重文件路径
            model_name: 预训练模型名称
            device: 设备
            label_mapping: 标签映射字典
        """
        if device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device
        
        # 加载模型
        self.model = TransformerClassifier(
            model_name=model_name,
            num_classes=2,  # 默认二分类
            max_length=512
        )
        
        # 加载权重
        self.load_model(model_path)
        self.model.to(self.device)
        self.model.eval()
        
        # 标签映射
        if label_mapping is None:
            self.label_mapping = {0: "负面", 1: "正面"}
        else:
            self.label_mapping = label_mapping
        
        logger.info(f"预测器初始化完成，设备: {self.device}")
        logger.info(f"标签映射: {self.label_mapping}")
    
    def load_model(self, model_path: str):
        """加载模型权重"""
        # PyTorch 2.6 之后默认 weights_only=True 会限制可反序列化类型，
        # 为兼容现有 checkpoint，这里显式设置 weights_only=False
        checkpoint = torch.load(model_path, map_location=self.device, weights_only=False)
        
        if 'model_state_dict' in checkpoint:
            self.model.load_state_dict(checkpoint['model_state_dict'])
        else:
            self.model.load_state_dict(checkpoint)
        
        logger.info(f"模型权重已从 {model_path} 加载")
    
    def predict_single(self, text: str) -> Dict[str, Union[str, float]]:
        """
        预测单个文本的情感
        
        Args:
            text: 输入文本
        
        Returns:
            result: 包含预测结果和概率的字典
        """
        # 预处理文本
        if not isinstance(text, str) or not text.strip():
            return {
                'text': text,
                'prediction': '未知',
                'confidence': 0.0,
                'probabilities': {}
            }
        
        # 编码文本
        encoding = self.model.tokenizer(
            text.strip(),
            max_length=512,
            padding=True,
            truncation=True,
            return_tensors='pt'
        )
        
        # 移到设备
        input_ids = encoding['input_ids'].to(self.device)
        attention_mask = encoding['attention_mask'].to(self.device)
        
        # 预测
        with torch.no_grad():
            logits = self.model(input_ids, attention_mask)
            probabilities = F.softmax(logits, dim=1)
            prediction = torch.argmax(probabilities, dim=1).item()
            confidence = probabilities[0][prediction].item()
        
        # 构建结果
        result = {
            'text': text,
            'prediction': self.label_mapping.get(prediction, f'类别_{prediction}'),
            'confidence': confidence,
            'probabilities': {
                self.label_mapping.get(i, f'类别_{i}'): prob.item()
                for i, prob in enumerate(probabilities[0])
            }
        }
        
        return result
    
    def predict_batch(self, texts: List[str], batch_size: int = 32) -> List[Dict]:
        """
        批量预测文本情感
        
        Args:
            texts: 文本列表
            batch_size: 批处理大小
        
        Returns:
            results: 预测结果列表
        """
        results = []
        
        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i:i + batch_size]
            
            # 编码批次
            encoding = self.model.tokenizer(
                batch_texts,
                max_length=512,
                padding=True,
                truncation=True,
                return_tensors='pt'
            )
            
            # 移到设备
            input_ids = encoding['input_ids'].to(self.device)
            attention_mask = encoding['attention_mask'].to(self.device)
            
            # 预测
            with torch.no_grad():
                logits = self.model(input_ids, attention_mask)
                probabilities = F.softmax(logits, dim=1)
                predictions = torch.argmax(probabilities, dim=1)
                confidences = torch.max(probabilities, dim=1)[0]
            
            # 构建结果
            for j, text in enumerate(batch_texts):
                pred = predictions[j].item()
                conf = confidences[j].item()
                
                result = {
                    'text': text,
                    'prediction': self.label_mapping.get(pred, f'类别_{pred}'),
                    'confidence': conf,
                    'probabilities': {
                        self.label_mapping.get(k, f'类别_{k}'): prob.item()
                        for k, prob in enumerate(probabilities[j])
                    }
                }
                results.append(result)
        
        return results
    
    def predict_file(self, file_path: str, text_column: str = "text", output_path: str = None) -> pd.DataFrame:
        """
        预测文件中的文本
        
        Args:
            file_path: 输入文件路径
            text_column: 文本列名
            output_path: 输出文件路径
        
        Returns:
            df: 包含预测结果的DataFrame
        """
        # 读取文件
        if file_path.endswith('.csv'):
            df = pd.read_csv(file_path, encoding='utf-8')
        elif file_path.endswith('.xlsx'):
            df = pd.read_excel(file_path)
        else:
            raise ValueError("不支持的文件格式，请使用CSV或Excel文件")
        
        # 检查文本列
        if text_column not in df.columns:
            raise KeyError(f"未找到文本列: {text_column}")
        
        # 批量预测
        texts = df[text_column].astype(str).tolist()
        results = self.predict_batch(texts)
        
        # 添加预测结果到DataFrame
        df['prediction'] = [r['prediction'] for r in results]
        df['confidence'] = [r['confidence'] for r in results]
        
        # 保存结果
        if output_path:
            if output_path.endswith('.csv'):
                df.to_csv(output_path, index=False, encoding='utf-8')
            elif output_path.endswith('.xlsx'):
                df.to_excel(output_path, index=False)
            logger.info(f"预测结果已保存到: {output_path}")
        
        return df
    
    def interactive_predict(self):
        """交互式预测"""
        print("=== 情感分析预测器 ===")
        print("输入文本进行情感分析，输入 'quit' 退出")
        print(f"支持的标签: {list(self.label_mapping.values())}")
        print("-" * 50)
        
        while True:
            try:
                text = input("\n请输入文本: ").strip()
                
                if text.lower() in ['quit', 'exit', '退出']:
                    print("再见！")
                    break
                
                if not text:
                    print("请输入有效的文本")
                    continue
                
                # 预测
                result = self.predict_single(text)
                
                # 显示结果
                print(f"\n文本: {result['text']}")
                print(f"预测: {result['prediction']}")
                print(f"置信度: {result['confidence']:.4f}")
                print("概率分布:")
                for label, prob in result['probabilities'].items():
                    print(f"  {label}: {prob:.4f}")
                
            except KeyboardInterrupt:
                print("\n\n再见！")
                break
            except Exception as e:
                print(f"预测出错: {e}")


class ModelEvaluator:
    """模型评估器"""
    
    def __init__(self, predictor: SentimentPredictor):
        self.predictor = predictor
    
    def evaluate_dataset(self, test_loader, label_encoder: LabelEncoder) -> Dict:
        """
        评估模型在测试集上的性能
        
        Args:
            test_loader: 测试数据加载器
            label_encoder: 标签编码器
        
        Returns:
            metrics: 评估指标
        """
        self.predictor.model.eval()
        all_predictions = []
        all_labels = []
        
        with torch.no_grad():
            for batch in test_loader:
                input_ids = batch['input_ids'].to(self.predictor.device)
                attention_mask = batch['attention_mask'].to(self.predictor.device)
                labels = batch['labels'].to(self.predictor.device)
                
                logits = self.predictor.model(input_ids, attention_mask)
                predictions = torch.argmax(logits, dim=1)
                
                all_predictions.extend(predictions.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
        
        # 计算指标
        from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix
        
        accuracy = accuracy_score(all_labels, all_predictions)
        precision, recall, f1, _ = precision_recall_fscore_support(
            all_labels, all_predictions, average='weighted'
        )
        cm = confusion_matrix(all_labels, all_predictions)
        
        # 获取标签名称
        label_names = label_encoder.classes_
        
        metrics = {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'confusion_matrix': cm.tolist(),
            'label_names': label_names.tolist()
        }
        
        return metrics
    
    def analyze_errors(self, test_loader, label_encoder: LabelEncoder, num_examples: int = 10) -> List[Dict]:
        """
        分析预测错误的样本
        
        Args:
            test_loader: 测试数据加载器
            label_encoder: 标签编码器
            num_examples: 返回的错误样本数量
        
        Returns:
            error_examples: 错误样本列表
        """
        self.predictor.model.eval()
        error_examples = []
        
        with torch.no_grad():
            for batch in test_loader:
                input_ids = batch['input_ids'].to(self.predictor.device)
                attention_mask = batch['attention_mask'].to(self.predictor.device)
                labels = batch['labels'].to(self.predictor.device)
                
                logits = self.predictor.model(input_ids, attention_mask)
                predictions = torch.argmax(logits, dim=1)
                probabilities = F.softmax(logits, dim=1)
                
                # 找到预测错误的样本
                for i in range(len(labels)):
                    if predictions[i] != labels[i]:
                        # 解码文本（简化版本，实际可能需要更复杂的解码）
                        text_tokens = self.predictor.model.tokenizer.convert_ids_to_tokens(
                            input_ids[i].cpu().numpy()
                        )
                        text = self.predictor.model.tokenizer.convert_tokens_to_string(text_tokens)
                        
                        error_example = {
                            'text': text,
                            'true_label': label_encoder.inverse_transform([labels[i].item()])[0],
                            'predicted_label': label_encoder.inverse_transform([predictions[i].item()])[0],
                            'confidence': probabilities[i][predictions[i]].item()
                        }
                        error_examples.append(error_example)
                        
                        if len(error_examples) >= num_examples:
                            break
                
                if len(error_examples) >= num_examples:
                    break
        
        return error_examples


def create_predictor_from_checkpoint(
    checkpoint_path: str,
    model_name: str = "bert-base-uncased",
    device: str = None
) -> SentimentPredictor:
    """
    从检查点创建预测器
    
    Args:
        checkpoint_path: 检查点文件路径
        model_name: 预训练模型名称
        device: 设备
    
    Returns:
        predictor: 预测器对象
    """
    predictor = SentimentPredictor(
        model_path=checkpoint_path,
        model_name=model_name,
        device=device
    )
    
    return predictor


if __name__ == "__main__":
    # 测试预测器
    print("测试预测器模块...")
    
    # 这里需要实际的模型文件来测试
    # 在实际使用中，这些会从训练好的模型加载
    print("预测器模块测试完成！")
