"""
基于Transformer的NLP文本分类模型
支持BERT、RoBERTa等预训练模型进行情感分析
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import (
    AutoTokenizer, 
    AutoModel, 
    AutoConfig,
    BertTokenizer,
    BertModel,
    RobertaTokenizer,
    RobertaModel
)
from typing import Dict, List, Optional, Union
import logging

# 设置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class TransformerClassifier(nn.Module):
    """
    基于Transformer的文本分类器
    支持多种预训练模型：BERT、RoBERTa、DistilBERT等
    """
    
    def __init__(
        self,
        model_name: str = "bert-base-uncased",
        num_classes: int = 2,
        dropout_rate: float = 0.3,
        freeze_bert: bool = False,
        max_length: int = 512
    ):
        super(TransformerClassifier, self).__init__()
        
        self.model_name = model_name
        self.num_classes = num_classes
        self.max_length = max_length
        
        # 加载预训练模型和分词器
        self.config = AutoConfig.from_pretrained(model_name)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.bert = AutoModel.from_pretrained(model_name)
        
        # 如果冻结BERT参数
        if freeze_bert:
            for param in self.bert.parameters():
                param.requires_grad = False
            logger.info(f"已冻结 {model_name} 的参数")
        
        # 获取BERT隐藏层维度
        self.hidden_size = self.bert.config.hidden_size
        
        # 分类头
        self.dropout = nn.Dropout(dropout_rate)
        self.classifier = nn.Linear(self.hidden_size, num_classes)
        
        # 初始化分类器权重
        nn.init.xavier_uniform_(self.classifier.weight)
        nn.init.zeros_(self.classifier.bias)
        
        logger.info(f"初始化模型: {model_name}, 类别数: {num_classes}")
    
    def forward(self, input_ids, attention_mask=None, token_type_ids=None):
        """
        前向传播
        
        Args:
            input_ids: 输入token IDs [batch_size, seq_len]
            attention_mask: 注意力掩码 [batch_size, seq_len]
            token_type_ids: token类型IDs [batch_size, seq_len]
        
        Returns:
            logits: 分类logits [batch_size, num_classes]
        """
        # 通过BERT获取序列表示
        outputs = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids
        )
        
        # 使用[CLS] token的表示进行分类
        pooled_output = outputs.pooler_output
        
        # Dropout和分类
        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)
        
        return logits
    
    def encode_text(self, texts: List[str], batch_size: int = 32) -> torch.Tensor:
        """
        编码文本为BERT输入格式
        
        Args:
            texts: 文本列表
            batch_size: 批处理大小
        
        Returns:
            encoded: 编码后的输入字典
        """
        self.eval()
        all_input_ids = []
        all_attention_masks = []
        
        with torch.no_grad():
            for i in range(0, len(texts), batch_size):
                batch_texts = texts[i:i + batch_size]
                
                # 分词和编码
                encoded = self.tokenizer(
                    batch_texts,
                    max_length=self.max_length,
                    padding=True,
                    truncation=True,
                    return_tensors='pt'
                )
                
                all_input_ids.append(encoded['input_ids'])
                all_attention_masks.append(encoded['attention_mask'])
        
        # 合并所有批次
        input_ids = torch.cat(all_input_ids, dim=0)
        attention_masks = torch.cat(all_attention_masks, dim=0)
        
        return {
            'input_ids': input_ids,
            'attention_mask': attention_masks
        }
    
    def predict(self, texts: List[str], batch_size: int = 32) -> List[int]:
        """
        预测文本类别
        
        Args:
            texts: 待预测的文本列表
            batch_size: 批处理大小
        
        Returns:
            predictions: 预测的类别列表
        """
        self.eval()
        predictions = []
        
        # 编码文本
        encoded = self.encode_text(texts, batch_size)
        input_ids = encoded['input_ids']
        attention_mask = encoded['attention_mask']
        
        with torch.no_grad():
            for i in range(0, len(input_ids), batch_size):
                batch_input_ids = input_ids[i:i + batch_size]
                batch_attention_mask = attention_mask[i:i + batch_size]
                
                # 前向传播
                logits = self.forward(batch_input_ids, batch_attention_mask)
                
                # 获取预测类别
                batch_predictions = torch.argmax(logits, dim=1)
                predictions.extend(batch_predictions.cpu().numpy().tolist())
        
        return predictions
    
    def predict_proba(self, texts: List[str], batch_size: int = 32) -> torch.Tensor:
        """
        预测文本类别的概率分布
        
        Args:
            texts: 待预测的文本列表
            batch_size: 批处理大小
        
        Returns:
            probabilities: 概率分布 [num_texts, num_classes]
        """
        self.eval()
        all_probabilities = []
        
        # 编码文本
        encoded = self.encode_text(texts, batch_size)
        input_ids = encoded['input_ids']
        attention_mask = encoded['attention_mask']
        
        with torch.no_grad():
            for i in range(0, len(input_ids), batch_size):
                batch_input_ids = input_ids[i:i + batch_size]
                batch_attention_mask = attention_mask[i:i + batch_size]
                
                # 前向传播
                logits = self.forward(batch_input_ids, batch_attention_mask)
                
                # 计算概率
                probabilities = F.softmax(logits, dim=1)
                all_probabilities.append(probabilities)
        
        return torch.cat(all_probabilities, dim=0)


class MultiLayerTransformerClassifier(nn.Module):
    """
    多层Transformer分类器，支持更复杂的分类任务
    """
    
    def __init__(
        self,
        model_name: str = "bert-base-uncased",
        num_classes: int = 2,
        hidden_dims: List[int] = [768, 256, 64],
        dropout_rate: float = 0.3,
        freeze_bert: bool = False,
        max_length: int = 512
    ):
        super(MultiLayerTransformerClassifier, self).__init__()
        
        self.model_name = model_name
        self.num_classes = num_classes
        self.max_length = max_length
        
        # 加载预训练模型
        self.config = AutoConfig.from_pretrained(model_name)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.bert = AutoModel.from_pretrained(model_name)
        
        if freeze_bert:
            for param in self.bert.parameters():
                param.requires_grad = False
        
        # 构建多层分类器
        self.hidden_size = self.bert.config.hidden_size
        layers = []
        
        # 输入层
        prev_dim = self.hidden_size
        
        # 隐藏层
        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout_rate)
            ])
            prev_dim = hidden_dim
        
        # 输出层
        layers.append(nn.Linear(prev_dim, num_classes))
        
        self.classifier = nn.Sequential(*layers)
        
        logger.info(f"初始化多层模型: {model_name}, 隐藏层: {hidden_dims}")
    
    def forward(self, input_ids, attention_mask=None, token_type_ids=None):
        """前向传播"""
        outputs = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids
        )
        
        pooled_output = outputs.pooler_output
        logits = self.classifier(pooled_output)
        
        return logits


def create_model(
    model_name: str = "bert-base-uncased",
    num_classes: int = 2,
    model_type: str = "simple",
    **kwargs
) -> TransformerClassifier:
    """
    创建Transformer分类模型
    
    Args:
        model_name: 预训练模型名称
        num_classes: 分类类别数
        model_type: 模型类型 ("simple" 或 "multi_layer")
        **kwargs: 其他模型参数
    
    Returns:
        model: 初始化的模型
    """
    if model_type == "simple":
        return TransformerClassifier(
            model_name=model_name,
            num_classes=num_classes,
            **kwargs
        )
    elif model_type == "multi_layer":
        return MultiLayerTransformerClassifier(
            model_name=model_name,
            num_classes=num_classes,
            **kwargs
        )
    else:
        raise ValueError(f"不支持的模型类型: {model_type}")


# 预定义的模型配置
MODEL_CONFIGS = {
    "bert-base": {
        "model_name": "bert-base-uncased",
        "max_length": 512,
        "dropout_rate": 0.3
    },
    "bert-large": {
        "model_name": "bert-large-uncased", 
        "max_length": 512,
        "dropout_rate": 0.3
    },
    "roberta-base": {
        "model_name": "roberta-base",
        "max_length": 512,
        "dropout_rate": 0.3
    },
    "distilbert": {
        "model_name": "distilbert-base-uncased",
        "max_length": 512,
        "dropout_rate": 0.3
    }
}


if __name__ == "__main__":
    # 测试模型创建
    print("测试Transformer分类模型...")
    
    # 创建简单模型
    model = create_model(
        model_name="bert-base-uncased",
        num_classes=2,
        model_type="simple"
    )
    
    print(f"模型参数数量: {sum(p.numel() for p in model.parameters()):,}")
    print(f"可训练参数数量: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")
    
    # 测试预测
    test_texts = [
        "This movie is absolutely fantastic!",
        "I hate this boring film."
    ]
    
    predictions = model.predict(test_texts)
    print(f"预测结果: {predictions}")
    
    probabilities = model.predict_proba(test_texts)
    print(f"概率分布: {probabilities}")
