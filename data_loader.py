"""
数据加载和预处理模块
支持IMDB情感分析数据集的加载、预处理和批处理
"""

import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from typing import List, Dict, Tuple, Optional
import logging
import os
from pathlib import Path

logger = logging.getLogger(__name__)


class IMDBDataset(Dataset):
    """
    IMDB情感分析数据集类
    """
    
    def __init__(
        self,
        texts: List[str],
        labels: List[int],
        tokenizer,
        max_length: int = 512,
        label_encoder: Optional[LabelEncoder] = None
    ):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.label_encoder = label_encoder
        
        # 验证数据长度
        assert len(texts) == len(labels), "文本和标签数量不匹配"
        
        logger.info(f"数据集大小: {len(texts)} 样本")
    
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        text = str(self.texts[idx])
        label = self.labels[idx]
        
        # 分词和编码
        encoding = self.tokenizer(
            text,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        
        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels': torch.tensor(label, dtype=torch.long)
        }


class DataProcessor:
    """
    数据处理器，负责加载、清洗和分割数据
    """
    
    def __init__(self, data_dir: str = "data"):
        self.data_dir = Path(data_dir)
        self.label_encoder = LabelEncoder()
        
    def load_imdb_data(
        self,
        language: str = "en",
        text_column: str = "text_clean",
        label_column: str = "sentiment"
    ) -> pd.DataFrame:
        """
        加载IMDB数据集
        
        Args:
            language: 语言类型 ("en" 或 "pt")
            text_column: 文本列名
            label_column: 标签列名
        
        Returns:
            df: 加载的数据框
        """
        if language == "en":
            file_path = self.data_dir / "imdb_en_clean.csv"
        elif language == "pt":
            file_path = self.data_dir / "imdb_pt_clean.csv"
        else:
            raise ValueError(f"不支持的语言: {language}")
        
        if not file_path.exists():
            raise FileNotFoundError(f"数据文件不存在: {file_path}")
        
        df = pd.read_csv(file_path, encoding='utf-8')
        logger.info(f"加载数据: {file_path}, 形状: {df.shape}")
        
        # 检查必要的列
        if text_column not in df.columns:
            raise KeyError(f"未找到文本列: {text_column}")
        if label_column not in df.columns:
            raise KeyError(f"未找到标签列: {label_column}")
        
        # 去除空值
        df = df.dropna(subset=[text_column, label_column])
        logger.info(f"清洗后数据形状: {df.shape}")
        
        return df
    
    def preprocess_labels(self, labels: pd.Series) -> List[int]:
        """
        预处理标签，转换为数值编码
        
        Args:
            labels: 原始标签序列
        
        Returns:
            encoded_labels: 编码后的标签列表
        """
        # 使用LabelEncoder进行编码
        encoded_labels = self.label_encoder.fit_transform(labels)
        
        # 记录标签映射
        label_mapping = dict(zip(
            self.label_encoder.classes_,
            range(len(self.label_encoder.classes_))
        ))
        logger.info(f"标签映射: {label_mapping}")
        
        return encoded_labels.tolist()
    
    def split_data(
        self,
        texts: List[str],
        labels: List[int],
        test_size: float = 0.2,
        val_size: float = 0.1,
        random_state: int = 42,
        stratify: bool = True
    ) -> Tuple[List[str], List[str], List[str], List[int], List[int], List[int]]:
        """
        分割数据集为训练集、验证集和测试集
        
        Args:
            texts: 文本列表
            labels: 标签列表
            test_size: 测试集比例
            val_size: 验证集比例
            random_state: 随机种子
            stratify: 是否分层采样
        
        Returns:
            train_texts, val_texts, test_texts, train_labels, val_labels, test_labels
        """
        # 首先分割出测试集
        if stratify:
            train_val_texts, test_texts, train_val_labels, test_labels = train_test_split(
                texts, labels,
                test_size=test_size,
                random_state=random_state,
                stratify=labels
            )
        else:
            train_val_texts, test_texts, train_val_labels, test_labels = train_test_split(
                texts, labels,
                test_size=test_size,
                random_state=random_state
            )
        
        # 再从训练+验证集中分割出验证集
        val_size_adjusted = val_size / (1 - test_size)  # 调整验证集比例
        
        if stratify:
            train_texts, val_texts, train_labels, val_labels = train_test_split(
                train_val_texts, train_val_labels,
                test_size=val_size_adjusted,
                random_state=random_state,
                stratify=train_val_labels
            )
        else:
            train_texts, val_texts, train_labels, val_labels = train_test_split(
                train_val_texts, train_val_labels,
                test_size=val_size_adjusted,
                random_state=random_state
            )
        
        logger.info(f"数据分割完成:")
        logger.info(f"  训练集: {len(train_texts)} 样本")
        logger.info(f"  验证集: {len(val_texts)} 样本")
        logger.info(f"  测试集: {len(test_texts)} 样本")
        
        return train_texts, val_texts, test_texts, train_labels, val_labels, test_labels
    
    def create_dataloaders(
        self,
        train_texts: List[str],
        train_labels: List[int],
        val_texts: List[str],
        val_labels: List[int],
        test_texts: List[str],
        test_labels: List[int],
        tokenizer,
        max_length: int = 512,
        batch_size: int = 16,
        num_workers: int = 4
    ) -> Tuple[DataLoader, DataLoader, DataLoader]:
        """
        创建数据加载器
        
        Args:
            train_texts, val_texts, test_texts: 各数据集的文本
            train_labels, val_labels, test_labels: 各数据集的标签
            tokenizer: 分词器
            max_length: 最大序列长度
            batch_size: 批处理大小
            num_workers: 工作进程数
        
        Returns:
            train_loader, val_loader, test_loader
        """
        # 创建数据集
        train_dataset = IMDBDataset(
            train_texts, train_labels, tokenizer, max_length, self.label_encoder
        )
        val_dataset = IMDBDataset(
            val_texts, val_labels, tokenizer, max_length, self.label_encoder
        )
        test_dataset = IMDBDataset(
            test_texts, test_labels, tokenizer, max_length, self.label_encoder
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


def load_and_prepare_data(
    data_dir: str = "data",
    language: str = "en",
    test_size: float = 0.2,
    val_size: float = 0.1,
    batch_size: int = 16,
    max_length: int = 512,
    tokenizer=None,
    random_state: int = 42
) -> Tuple[DataLoader, DataLoader, DataLoader, LabelEncoder]:
    """
    一站式数据加载和准备函数
    
    Args:
        data_dir: 数据目录
        language: 语言类型
        test_size: 测试集比例
        val_size: 验证集比例
        batch_size: 批处理大小
        max_length: 最大序列长度
        tokenizer: 分词器
        random_state: 随机种子
    
    Returns:
        train_loader, val_loader, test_loader, label_encoder
    """
    # 创建数据处理器
    processor = DataProcessor(data_dir)
    
    # 加载数据
    df = processor.load_imdb_data(language=language)
    
    # 提取文本和标签
    texts = df['text_clean'].tolist()
    labels = processor.preprocess_labels(df['sentiment'])
    
    # 分割数据
    train_texts, val_texts, test_texts, train_labels, val_labels, test_labels = processor.split_data(
        texts, labels, test_size, val_size, random_state
    )
    
    # 创建数据加载器
    train_loader, val_loader, test_loader = processor.create_dataloaders(
        train_texts, train_labels,
        val_texts, val_labels,
        test_texts, test_labels,
        tokenizer, max_length, batch_size
    )
    
    return train_loader, val_loader, test_loader, processor.label_encoder


if __name__ == "__main__":
    # 测试数据加载
    print("测试数据加载模块...")
    
    from transformers import AutoTokenizer
    
    # 创建分词器
    tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
    
    # 加载数据
    train_loader, val_loader, test_loader, label_encoder = load_and_prepare_data(
        data_dir="data",
        language="en",
        batch_size=8,
        tokenizer=tokenizer
    )
    
    # 测试一个批次
    for batch in train_loader:
        print(f"批次形状:")
        print(f"  input_ids: {batch['input_ids'].shape}")
        print(f"  attention_mask: {batch['attention_mask'].shape}")
        print(f"  labels: {batch['labels'].shape}")
        break
    
    print("数据加载测试完成！")
