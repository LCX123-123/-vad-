"""
情感维度模型
支持多维度情感预测，包括Valence（效价）、Arousal（唤醒度）、Dominance（支配性）等维度
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import (
    AutoTokenizer, 
    AutoModel, 
    AutoConfig
)
from typing import Dict, List, Optional, Union, Tuple
import logging
import numpy as np

# 设置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class EmotionDimensionModel(nn.Module):
    """
    情感维度模型
    支持多维度情感预测，包括Valence、Arousal、Dominance等维度
    """
    
    def __init__(
        self,
        model_name: str = "bert-base-uncased",
        emotion_dimensions: List[str] = ["valence", "arousal", "dominance"],
        dimension_ranges: Dict[str, Tuple[float, float]] = None,
        hidden_dims: List[int] = [768, 256, 64],
        dropout_rate: float = 0.3,
        freeze_bert: bool = False,
        max_length: int = 512,
        use_attention_pooling: bool = True
    ):
        super(EmotionDimensionModel, self).__init__()
        
        self.model_name = model_name
        self.emotion_dimensions = emotion_dimensions
        self.num_dimensions = len(emotion_dimensions)
        self.max_length = max_length
        self.use_attention_pooling = use_attention_pooling
        
        # 设置维度范围，默认为[-1, 1]
        if dimension_ranges is None:
            self.dimension_ranges = {dim: (-1.0, 1.0) for dim in emotion_dimensions}
        else:
            self.dimension_ranges = dimension_ranges
        
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
        
        # 注意力池化层（可选）
        if use_attention_pooling:
            self.attention_pooling = nn.MultiheadAttention(
                embed_dim=self.hidden_size,
                num_heads=8,
                dropout=dropout_rate,
                batch_first=True
            )
        
        # 构建多层回归器
        layers = []
        prev_dim = self.hidden_size
        
        # 隐藏层
        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout_rate)
            ])
            prev_dim = hidden_dim
        
        # 创建分类器（隐藏层）
        self.classifier = nn.Sequential(*layers)
        
        # 输出层 - 每个情感维度一个输出
        self.regressors = nn.ModuleDict()
        for dimension in emotion_dimensions:
            self.regressors[dimension] = nn.Linear(prev_dim, 1)
        
        # 初始化权重
        self._init_weights()
        
        logger.info(f"初始化情感维度模型: {model_name}")
        logger.info(f"情感维度: {emotion_dimensions}")
        logger.info(f"维度范围: {self.dimension_ranges}")
    
    def _init_weights(self):
        """初始化模型权重"""
        for dimension in self.emotion_dimensions:
            nn.init.xavier_uniform_(self.regressors[dimension].weight)
            nn.init.zeros_(self.regressors[dimension].bias)
    
    def forward(self, input_ids, attention_mask=None, token_type_ids=None):
        """
        前向传播
        
        Args:
            input_ids: 输入token IDs [batch_size, seq_len]
            attention_mask: 注意力掩码 [batch_size, seq_len]
            token_type_ids: token类型IDs [batch_size, seq_len]
        
        Returns:
            predictions: 各维度预测值 [batch_size, num_dimensions]
        """
        # 通过BERT获取序列表示
        outputs = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids
        )
        
        # 获取序列表示
        sequence_output = outputs.last_hidden_state  # [batch_size, seq_len, hidden_size]
        
        if self.use_attention_pooling:
            # 使用注意力池化
            # 创建全局查询向量
            batch_size = sequence_output.size(0)
            global_query = torch.mean(sequence_output, dim=1, keepdim=True)  # [batch_size, 1, hidden_size]
            
            # 注意力池化
            pooled_output, _ = self.attention_pooling(
                query=global_query,
                key=sequence_output,
                value=sequence_output,
                key_padding_mask=~attention_mask.bool() if attention_mask is not None else None
            )
            pooled_output = pooled_output.squeeze(1)  # [batch_size, hidden_size]
        else:
            # 使用[CLS] token的表示
            pooled_output = outputs.pooler_output
        
        # 通过隐藏层
        hidden_output = pooled_output
        for layer in self.classifier[:-1]:  # 除了最后一层
            hidden_output = layer(hidden_output)
        
        # 预测各维度
        predictions = {}
        for dimension in self.emotion_dimensions:
            predictions[dimension] = self.regressors[dimension](hidden_output)
        
        # 合并所有维度的预测
        all_predictions = torch.cat([predictions[dim] for dim in self.emotion_dimensions], dim=1)
        
        return all_predictions
    
    def predict_dimensions(self, texts: List[str], batch_size: int = 32) -> Dict[str, List[float]]:
        """
        预测文本的情感维度
        
        Args:
            texts: 文本列表
            batch_size: 批处理大小
        
        Returns:
            predictions: 各维度的预测值
        """
        self.eval()
        all_predictions = {dim: [] for dim in self.emotion_dimensions}
        
        # 编码文本
        encoded = self.encode_text(texts, batch_size)
        input_ids = encoded['input_ids']
        attention_mask = encoded['attention_mask']
        
        # 设备
        device = next(self.parameters()).device
        
        with torch.no_grad():
            for i in range(0, len(input_ids), batch_size):
                batch_input_ids = input_ids[i:i + batch_size].to(device)
                batch_attention_mask = attention_mask[i:i + batch_size].to(device)
                
                # 前向传播
                predictions = self.forward(batch_input_ids, batch_attention_mask)
                
                # 分离各维度预测
                for j, dimension in enumerate(self.emotion_dimensions):
                    dim_predictions = predictions[:, j].cpu().numpy().tolist()
                    all_predictions[dimension].extend(dim_predictions)
        
        return all_predictions
    
    def encode_text(self, texts: List[str], batch_size: int = 32) -> Dict[str, torch.Tensor]:
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
    
    def get_emotion_profile(self, text: str) -> Dict[str, float]:
        """
        获取文本的情感轮廓
        
        Args:
            text: 输入文本
        
        Returns:
            emotion_profile: 情感轮廓字典
        """
        predictions = self.predict_dimensions([text])
        
        emotion_profile = {}
        for dimension in self.emotion_dimensions:
            emotion_profile[dimension] = predictions[dimension][0]
        
        return emotion_profile
    
    def interpret_emotion(self, emotion_profile: Dict[str, float]) -> str:
        """
        解释情感轮廓
        
        Args:
            emotion_profile: 情感轮廓字典
        
        Returns:
            interpretation: 情感解释文本
        """
        interpretations = []
        
        for dimension, value in emotion_profile.items():
            if dimension == "valence":
                if value > 0.5:
                    interpretations.append("非常积极")
                elif value > 0:
                    interpretations.append("积极")
                elif value > -0.5:
                    interpretations.append("消极")
                else:
                    interpretations.append("非常消极")
            
            elif dimension == "arousal":
                if value > 0.5:
                    interpretations.append("高唤醒")
                elif value > 0:
                    interpretations.append("中等唤醒")
                elif value > -0.5:
                    interpretations.append("低唤醒")
                else:
                    interpretations.append("极低唤醒")
            
            elif dimension == "dominance":
                if value > 0.5:
                    interpretations.append("高支配性")
                elif value > 0:
                    interpretations.append("中等支配性")
                elif value > -0.5:
                    interpretations.append("低支配性")
                else:
                    interpretations.append("极低支配性")
        
        return " | ".join(interpretations)


class MultiTaskEmotionModel(nn.Module):
    """
    多任务情感模型
    同时进行情感分类和情感维度预测
    """
    
    def __init__(
        self,
        model_name: str = "bert-base-uncased",
        num_classes: int = 2,
        emotion_dimensions: List[str] = ["valence", "arousal", "dominance"],
        hidden_dims: List[int] = [768, 256, 64],
        dropout_rate: float = 0.3,
        freeze_bert: bool = False,
        max_length: int = 512,
        task_weights: Dict[str, float] = None
    ):
        super(MultiTaskEmotionModel, self).__init__()
        
        self.model_name = model_name
        self.num_classes = num_classes
        self.emotion_dimensions = emotion_dimensions
        self.num_dimensions = len(emotion_dimensions)
        self.max_length = max_length
        
        # 设置任务权重
        if task_weights is None:
            self.task_weights = {
                "classification": 1.0,
                "regression": 1.0
            }
        else:
            self.task_weights = task_weights
        
        # 加载预训练模型
        self.config = AutoConfig.from_pretrained(model_name)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.bert = AutoModel.from_pretrained(model_name)
        
        if freeze_bert:
            for param in self.bert.parameters():
                param.requires_grad = False
        
        self.hidden_size = self.bert.config.hidden_size
        
        # 共享特征提取器
        layers = []
        prev_dim = self.hidden_size
        
        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout_rate)
            ])
            prev_dim = hidden_dim
        
        self.shared_layers = nn.Sequential(*layers)
        
        # 分类头
        self.classifier = nn.Linear(prev_dim, num_classes)
        
        # 回归头
        self.regressors = nn.ModuleDict()
        for dimension in emotion_dimensions:
            self.regressors[dimension] = nn.Linear(prev_dim, 1)
        
        # 初始化权重
        self._init_weights()
        
        logger.info(f"初始化多任务情感模型: {model_name}")
        logger.info(f"分类类别数: {num_classes}")
        logger.info(f"情感维度: {emotion_dimensions}")
    
    def _init_weights(self):
        """初始化模型权重"""
        nn.init.xavier_uniform_(self.classifier.weight)
        nn.init.zeros_(self.classifier.bias)
        
        for dimension in self.emotion_dimensions:
            nn.init.xavier_uniform_(self.regressors[dimension].weight)
            nn.init.zeros_(self.regressors[dimension].bias)
    
    def forward(self, input_ids, attention_mask=None, token_type_ids=None):
        """
        前向传播
        
        Returns:
            classification_logits: 分类logits
            regression_predictions: 回归预测
        """
        # 通过BERT获取序列表示
        outputs = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids
        )
        
        pooled_output = outputs.pooler_output
        
        # 共享特征提取
        shared_features = self.shared_layers(pooled_output)
        
        # 分类预测
        classification_logits = self.classifier(shared_features)
        
        # 回归预测
        regression_predictions = {}
        for dimension in self.emotion_dimensions:
            regression_predictions[dimension] = self.regressors[dimension](shared_features)
        
        # 合并回归预测
        all_regression = torch.cat([regression_predictions[dim] for dim in self.emotion_dimensions], dim=1)
        
        return classification_logits, all_regression


def create_emotion_dimension_model(
    model_name: str = "bert-base-uncased",
    emotion_dimensions: List[str] = ["valence", "arousal", "dominance"],
    dimension_ranges: Dict[str, Tuple[float, float]] = None,
    model_type: str = "dimension",
    **kwargs
) -> Union[EmotionDimensionModel, MultiTaskEmotionModel]:
    """
    创建情感维度模型
    
    Args:
        model_name: 预训练模型名称
        emotion_dimensions: 情感维度列表
        dimension_ranges: 各维度的取值范围
        model_type: 模型类型 ("dimension" 或 "multi_task")
        **kwargs: 其他模型参数
    
    Returns:
        model: 初始化的模型
    """
    if model_type == "dimension":
        return EmotionDimensionModel(
            model_name=model_name,
            emotion_dimensions=emotion_dimensions,
            dimension_ranges=dimension_ranges,
            **kwargs
        )
    elif model_type == "multi_task":
        return MultiTaskEmotionModel(
            model_name=model_name,
            emotion_dimensions=emotion_dimensions,
            **kwargs
        )
    else:
        raise ValueError(f"不支持的模型类型: {model_type}")


# 预定义的情感维度配置
EMOTION_DIMENSION_CONFIGS = {
    "vad": {
        "emotion_dimensions": ["valence", "arousal", "dominance"],
        "dimension_ranges": {
            "valence": (-1.0, 1.0),
            "arousal": (-1.0, 1.0),
            "dominance": (-1.0, 1.0)
        }
    },
    "va": {
        "emotion_dimensions": ["valence", "arousal"],
        "dimension_ranges": {
            "valence": (-1.0, 1.0),
            "arousal": (-1.0, 1.0)
        }
    },
    "ekman": {
        "emotion_dimensions": ["anger", "fear", "joy", "sadness", "surprise", "disgust"],
        "dimension_ranges": {
            "anger": (0.0, 1.0),
            "fear": (0.0, 1.0),
            "joy": (0.0, 1.0),
            "sadness": (0.0, 1.0),
            "surprise": (0.0, 1.0),
            "disgust": (0.0, 1.0)
        }
    }
}


if __name__ == "__main__":
    # 测试情感维度模型
    print("测试情感维度模型...")
    
    # 创建VAD模型
    model = create_emotion_dimension_model(
        model_name="bert-base-uncased",
        emotion_dimensions=["valence", "arousal", "dominance"],
        model_type="dimension"
    )
    
    print(f"模型参数数量: {sum(p.numel() for p in model.parameters()):,}")
    print(f"可训练参数数量: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")
    
    # 测试预测
    test_texts = [
        "This movie is absolutely fantastic!",
        "I hate this boring film.",
        "The weather is okay today."
    ]
    
    predictions = model.predict_dimensions(test_texts)
    print(f"预测结果: {predictions}")
    
    # 测试情感轮廓
    emotion_profile = model.get_emotion_profile(test_texts[0])
    interpretation = model.interpret_emotion(emotion_profile)
    print(f"情感轮廓: {emotion_profile}")
    print(f"情感解释: {interpretation}")
