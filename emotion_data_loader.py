"""
情感维度数据加载器
支持情感维度标签的加载、预处理和批处理
"""

import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from typing import List, Dict, Tuple, Optional, Union
import logging
import os
from pathlib import Path
import numpy as np

logger = logging.getLogger(__name__)


class EmotionDimensionDataset(Dataset):
    """
    情感维度数据集类
    支持多维度情感标签
    """
    
    def __init__(
        self,
        texts: List[str],
        emotion_labels: Dict[str, List[float]],
        tokenizer,
        max_length: int = 512,
        scaler: Optional[StandardScaler] = None
    ):
        self.texts = texts
        self.emotion_labels = emotion_labels
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.scaler = scaler
        
        # 验证数据长度
        for dimension, labels in emotion_labels.items():
            assert len(texts) == len(labels), f"文本和{dimension}标签数量不匹配"
        
        # 获取情感维度列表
        self.emotion_dimensions = list(emotion_labels.keys())
        self.num_dimensions = len(self.emotion_dimensions)
        
        logger.info(f"数据集大小: {len(texts)} 样本")
        logger.info(f"情感维度: {self.emotion_dimensions}")
    
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        text = str(self.texts[idx])
        
        # 分词和编码
        encoding = self.tokenizer(
            text,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        
        # 获取情感标签
        emotion_values = []
        for dimension in self.emotion_dimensions:
            emotion_values.append(self.emotion_labels[dimension][idx])
        
        emotion_tensor = torch.tensor(emotion_values, dtype=torch.float32)
        
        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'emotion_labels': emotion_tensor
        }


class EmotionDataProcessor:
    """
    情感数据处理器，负责加载、清洗和分割情感维度数据
    """
    
    def __init__(self, data_dir: str = "data"):
        self.data_dir = Path(data_dir)
        self.scalers = {}
        
    def load_emotion_data(
        self,
        file_path: str,
        text_column: str = "text",
        emotion_columns: List[str] = None
    ) -> pd.DataFrame:
        """
        加载情感维度数据
        
        Args:
            file_path: 数据文件路径
            text_column: 文本列名
            emotion_columns: 情感维度列名列表
        
        Returns:
            df: 加载的数据框
        """
        if not Path(file_path).exists():
            raise FileNotFoundError(f"数据文件不存在: {file_path}")
        
        df = pd.read_csv(file_path, encoding='utf-8')
        logger.info(f"加载数据: {file_path}, 形状: {df.shape}")
        
        # 检查必要的列
        if text_column not in df.columns:
            raise KeyError(f"未找到文本列: {text_column}")
        
        if emotion_columns is None:
            # 自动检测情感维度列
            emotion_columns = [col for col in df.columns if col != text_column]
        
        for col in emotion_columns:
            if col not in df.columns:
                raise KeyError(f"未找到情感维度列: {col}")
        
        # 去除空值
        df = df.dropna(subset=[text_column] + emotion_columns)
        logger.info(f"清洗后数据形状: {df.shape}")
        
        return df
    
    def preprocess_emotion_labels(
        self, 
        df: pd.DataFrame, 
        emotion_columns: List[str],
        normalize: bool = True
    ) -> Dict[str, List[float]]:
        """
        预处理情感标签
        
        Args:
            df: 数据框
            emotion_columns: 情感维度列名列表
            normalize: 是否标准化
        
        Returns:
            emotion_labels: 预处理后的情感标签字典
        """
        emotion_labels = {}
        
        for column in emotion_columns:
            labels = df[column].astype(float).tolist()
            
            if normalize:
                # 标准化到[-1, 1]范围
                labels = np.array(labels)
                min_val, max_val = labels.min(), labels.max()
                if max_val > min_val:
                    labels = 2 * (labels - min_val) / (max_val - min_val) - 1
                else:
                    labels = np.zeros_like(labels)
                
                # 保存标准化参数
                self.scalers[column] = {
                    'min': min_val,
                    'max': max_val,
                    'mean': labels.mean(),
                    'std': labels.std()
                }
            
            emotion_labels[column] = labels.tolist()
        
        logger.info(f"情感标签预处理完成，维度: {list(emotion_labels.keys())}")
        
        return emotion_labels
    
    def split_data(
        self,
        texts: List[str],
        emotion_labels: Dict[str, List[float]],
        test_size: float = 0.2,
        val_size: float = 0.1,
        random_state: int = 42,
        stratify: bool = False
    ) -> Tuple[List[str], List[str], List[str], Dict[str, List[float]], Dict[str, List[float]], Dict[str, List[float]]]:
        """
        分割数据集为训练集、验证集和测试集
        
        Args:
            texts: 文本列表
            emotion_labels: 情感标签字典
            test_size: 测试集比例
            val_size: 验证集比例
            random_state: 随机种子
            stratify: 是否分层采样（对于回归任务通常为False）
        
        Returns:
            train_texts, val_texts, test_texts, train_labels, val_labels, test_labels
        """
        # 使用索引进行数据分割
        indices = list(range(len(texts)))
        
        # 首先分割出测试集
        train_val_indices, test_indices = train_test_split(
            indices,
            test_size=test_size,
            random_state=random_state
        )
        
        # 再从训练+验证集中分割出验证集
        val_size_adjusted = val_size / (1 - test_size)
        
        train_indices, val_indices = train_test_split(
            train_val_indices,
            test_size=val_size_adjusted,
            random_state=random_state
        )
        
        # 根据索引分割数据
        train_texts = [texts[i] for i in train_indices]
        val_texts = [texts[i] for i in val_indices]
        test_texts = [texts[i] for i in test_indices]
        
        # 分割情感标签
        train_labels_dict = {}
        val_labels_dict = {}
        test_labels_dict = {}
        
        for dimension, labels in emotion_labels.items():
            train_labels_dict[dimension] = [labels[i] for i in train_indices]
            val_labels_dict[dimension] = [labels[i] for i in val_indices]
            test_labels_dict[dimension] = [labels[i] for i in test_indices]
        
        logger.info(f"数据分割完成:")
        logger.info(f"  训练集: {len(train_texts)} 样本")
        logger.info(f"  验证集: {len(val_texts)} 样本")
        logger.info(f"  测试集: {len(test_texts)} 样本")
        
        return train_texts, val_texts, test_texts, train_labels_dict, val_labels_dict, test_labels_dict
    
    def create_dataloaders(
        self,
        train_texts: List[str],
        train_labels: Dict[str, List[float]],
        val_texts: List[str],
        val_labels: Dict[str, List[float]],
        test_texts: List[str],
        test_labels: Dict[str, List[float]],
        tokenizer,
        max_length: int = 512,
        batch_size: int = 16,
        num_workers: int = 4
    ) -> Tuple[DataLoader, DataLoader, DataLoader]:
        """
        创建数据加载器
        
        Args:
            train_texts, val_texts, test_texts: 各数据集的文本
            train_labels, val_labels, test_labels: 各数据集的情感标签
            tokenizer: 分词器
            max_length: 最大序列长度
            batch_size: 批处理大小
            num_workers: 工作进程数
        
        Returns:
            train_loader, val_loader, test_loader
        """
        # 创建数据集
        train_dataset = EmotionDimensionDataset(
            train_texts, train_labels, tokenizer, max_length
        )
        val_dataset = EmotionDimensionDataset(
            val_texts, val_labels, tokenizer, max_length
        )
        test_dataset = EmotionDimensionDataset(
            test_texts, test_labels, tokenizer, max_length
        )
        
        # 创建数据加载器
        train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=True
        )
        
        val_loader = DataLoader(
            val_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=True
        )
        
        test_loader = DataLoader(
            test_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=True
        )
        
        logger.info(f"数据加载器创建完成，批处理大小: {batch_size}")
        
        return train_loader, val_loader, test_loader


def load_and_prepare_emotion_data(
    file_path: str,
    text_column: str = "text",
    emotion_columns: List[str] = None,
    test_size: float = 0.2,
    val_size: float = 0.1,
    batch_size: int = 16,
    max_length: int = 512,
    tokenizer=None,
    random_state: int = 42,
    normalize: bool = True
) -> Tuple[DataLoader, DataLoader, DataLoader, EmotionDataProcessor]:
    """
    一站式情感数据加载和准备函数
    
    Args:
        file_path: 数据文件路径
        text_column: 文本列名
        emotion_columns: 情感维度列名列表
        test_size: 测试集比例
        val_size: 验证集比例
        batch_size: 批处理大小
        max_length: 最大序列长度
        tokenizer: 分词器
        random_state: 随机种子
        normalize: 是否标准化情感标签
    
    Returns:
        train_loader, val_loader, test_loader, processor
    """
    # 创建数据处理器
    processor = EmotionDataProcessor()
    
    # 加载数据
    df = processor.load_emotion_data(file_path, text_column, emotion_columns)
    
    # 提取文本和情感标签
    texts = df[text_column].tolist()
    emotion_labels = processor.preprocess_emotion_labels(df, emotion_columns, normalize)
    
    # 分割数据
    train_texts, val_texts, test_texts, train_labels, val_labels, test_labels = processor.split_data(
        texts, emotion_labels, test_size, val_size, random_state
    )
    
    # 创建数据加载器
    train_loader, val_loader, test_loader = processor.create_dataloaders(
        train_texts, train_labels,
        val_texts, val_labels,
        test_texts, test_labels,
        tokenizer, max_length, batch_size
    )
    
    return train_loader, val_loader, test_loader, processor


def _auto_map_columns_for_emobank(df: pd.DataFrame) -> Dict[str, str]:
    """为EmoBank数据集自动映射列名到标准列名。

    期望输出列: text, valence, arousal, dominance

    可能的候选列（大小写不敏感，自动匹配）:
      - 文本: ["text", "sentence", "utterance", "content"]
      - 效价: ["valence", "v"]
      - 唤醒: ["arousal", "a"]
      - 支配: ["dominance", "d"]
    """
    lower_cols = {c.lower(): c for c in df.columns}

    def pick(candidates):
        for name in candidates:
            if name in lower_cols:
                return lower_cols[name]
        return None

    mapping = {}
    mapping["text"] = pick(["text", "sentence", "utterance", "content"]) or "text"
    mapping["valence"] = pick(["valence", "v"]) or "valence"
    mapping["arousal"] = pick(["arousal", "a"]) or "arousal"
    mapping["dominance"] = pick(["dominance", "d"]) or "dominance"

    # 校验存在性
    missing = [k for k, v in mapping.items() if v not in df.columns]
    if missing:
        raise KeyError(f"EmoBank列缺失，无法映射: {missing}；现有列: {list(df.columns)}")

    return mapping


def parse_emobank_csv(file_path: str,
                      save_normalized_path: Optional[str] = None) -> pd.DataFrame:
    """解析EmoBank CSV，标准化列名为 text,valence,arousal,dominance。

    Args:
        file_path: 原始EmoBank CSV路径
        save_normalized_path: 若提供，将保存标准化后的CSV

    Returns:
        标准化后的DataFrame，包含[text, valence, arousal, dominance]
    """
    if not Path(file_path).exists():
        raise FileNotFoundError(f"数据文件不存在: {file_path}")

    df = pd.read_csv(file_path, encoding="utf-8")
    logger.info(f"加载EmoBank: {file_path}, 形状: {df.shape}")

    mapping = _auto_map_columns_for_emobank(df)
    std_df = df[[mapping["text"], mapping["valence"], mapping["arousal"], mapping["dominance"]]].copy()
    std_df.columns = ["text", "valence", "arousal", "dominance"]

    # 去除空值并确保为float
    std_df = std_df.dropna(subset=["text", "valence", "arousal", "dominance"]).copy()
    for col in ["valence", "arousal", "dominance"]:
        std_df[col] = pd.to_numeric(std_df[col], errors="coerce")
    std_df = std_df.dropna(subset=["valence", "arousal", "dominance"]).copy()

    logger.info(f"EmoBank标准化后形状: {std_df.shape}")

    if save_normalized_path:
        os.makedirs(os.path.dirname(save_normalized_path), exist_ok=True)
        std_df.to_csv(save_normalized_path, index=False, encoding="utf-8")
        logger.info(f"已保存标准化EmoBank到: {save_normalized_path}")

    return std_df


def load_and_prepare_emobank(file_path: str,
                             test_size: float = 0.2,
                             val_size: float = 0.1,
                             batch_size: int = 16,
                             max_length: int = 512,
                             tokenizer=None,
                             random_state: int = 42,
                             normalize: bool = True,
                             save_normalized_path: Optional[str] = None
                             ) -> Tuple[DataLoader, DataLoader, DataLoader, EmotionDataProcessor]:
    """一站式加载EmoBank并返回数据加载器。

    该函数会：
      1) 解析并标准化EmoBank列名
      2) 调用现有预处理/分割/构建DataLoader流水线
    """
    std_df = parse_emobank_csv(file_path, save_normalized_path)

    processor = EmotionDataProcessor()
    texts = std_df["text"].tolist()
    emotion_labels = processor.preprocess_emotion_labels(std_df, ["valence", "arousal", "dominance"], normalize)

    train_texts, val_texts, test_texts, train_labels, val_labels, test_labels = processor.split_data(
        texts, emotion_labels, test_size, val_size, random_state
    )

    train_loader, val_loader, test_loader = processor.create_dataloaders(
        train_texts, train_labels,
        val_texts, val_labels,
        test_texts, test_labels,
        tokenizer, max_length, batch_size
    )

    return train_loader, val_loader, test_loader, processor

def create_sample_emotion_data(
    num_samples: int = 1000,
    emotion_dimensions: List[str] = ["valence", "arousal", "dominance"],
    save_path: str = "data/sample_emotion_data.csv"
) -> pd.DataFrame:
    """
    创建示例情感数据
    
    Args:
        num_samples: 样本数量
        emotion_dimensions: 情感维度列表
        save_path: 保存路径
    
    Returns:
        df: 示例数据框
    """
    # 生成示例文本
    positive_texts = [
        "This movie is absolutely fantastic!",
        "I love this amazing film!",
        "What a wonderful experience!",
        "This is the best thing ever!",
        "I'm so happy and excited!"
    ]
    
    negative_texts = [
        "This movie is terrible and boring.",
        "I hate this awful film.",
        "What a disappointing experience.",
        "This is the worst thing ever.",
        "I'm so sad and frustrated."
    ]
    
    neutral_texts = [
        "The weather is okay today.",
        "This is a normal day.",
        "Nothing special happened.",
        "The movie was average.",
        "I feel neutral about this."
    ]
    
    all_texts = positive_texts + negative_texts + neutral_texts
    
    # 生成示例数据
    texts = []
    emotion_data = {dim: [] for dim in emotion_dimensions}
    
    for i in range(num_samples):
        text = all_texts[i % len(all_texts)]
        texts.append(text)
        
        # 根据文本类型生成情感标签
        if "fantastic" in text or "love" in text or "wonderful" in text or "best" in text or "happy" in text:
            # 积极文本
            for dim in emotion_dimensions:
                if dim == "valence":
                    emotion_data[dim].append(np.random.uniform(0.5, 1.0))
                elif dim == "arousal":
                    emotion_data[dim].append(np.random.uniform(0.3, 1.0))
                elif dim == "dominance":
                    emotion_data[dim].append(np.random.uniform(0.2, 1.0))
                else:
                    emotion_data[dim].append(np.random.uniform(0.0, 1.0))
        
        elif "terrible" in text or "hate" in text or "awful" in text or "worst" in text or "sad" in text:
            # 消极文本
            for dim in emotion_dimensions:
                if dim == "valence":
                    emotion_data[dim].append(np.random.uniform(-1.0, -0.5))
                elif dim == "arousal":
                    emotion_data[dim].append(np.random.uniform(0.2, 0.8))
                elif dim == "dominance":
                    emotion_data[dim].append(np.random.uniform(-0.5, 0.2))
                else:
                    emotion_data[dim].append(np.random.uniform(-1.0, 0.0))
        
        else:
            # 中性文本
            for dim in emotion_dimensions:
                emotion_data[dim].append(np.random.uniform(-0.3, 0.3))
    
    # 创建数据框
    data = {"text": texts}
    data.update(emotion_data)
    df = pd.DataFrame(data)
    
    # 保存数据
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    df.to_csv(save_path, index=False, encoding='utf-8')
    
    logger.info(f"示例情感数据已保存到: {save_path}")
    logger.info(f"数据形状: {df.shape}")
    
    return df


if __name__ == "__main__":
    # 测试情感数据加载
    print("测试情感数据加载模块...")
    
    from transformers import AutoTokenizer
    
    # 创建示例数据
    sample_df = create_sample_emotion_data(
        num_samples=100,
        emotion_dimensions=["valence", "arousal", "dominance"],
        save_path="data/sample_emotion_data.csv"
    )
    
    # 创建分词器
    tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
    
    # 加载数据
    train_loader, val_loader, test_loader, processor = load_and_prepare_emotion_data(
        file_path="data/sample_emotion_data.csv",
        text_column="text",
        emotion_columns=["valence", "arousal", "dominance"],
        batch_size=8,
        tokenizer=tokenizer
    )
    
    # 测试一个批次
    for batch in train_loader:
        print(f"批次形状:")
        print(f"  input_ids: {batch['input_ids'].shape}")
        print(f"  attention_mask: {batch['attention_mask'].shape}")
        print(f"  emotion_labels: {batch['emotion_labels'].shape}")
        print(f"  情感标签示例: {batch['emotion_labels'][0]}")
        break
    
    print("情感数据加载测试完成！")
