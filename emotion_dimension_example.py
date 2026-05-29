"""
情感维度模型使用示例
演示如何使用情感维度模型进行训练、评估和预测
"""

import torch
import numpy as np
import pandas as pd
from transformers import AutoTokenizer
from pathlib import Path
import logging

# 导入情感维度模块
from emotion_dimension_model import create_emotion_dimension_model, EMOTION_DIMENSION_CONFIGS
from emotion_data_loader import load_and_prepare_emotion_data, create_sample_emotion_data
from emotion_trainer import EmotionDimensionTrainer
from emotion_evaluator import EmotionDimensionEvaluator

# 设置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def create_sample_data():
    """创建示例情感数据"""
    logger.info("创建示例情感数据...")
    
    # 创建示例数据
    sample_df = create_sample_emotion_data(
        num_samples=1000,
        emotion_dimensions=["valence", "arousal", "dominance"],
        save_path="data/sample_emotion_data.csv"
    )
    
    logger.info(f"示例数据已创建，形状: {sample_df.shape}")
    return sample_df


def train_emotion_dimension_model():
    """训练情感维度模型"""
    logger.info("开始训练情感维度模型...")
    
    # 创建示例数据
    create_sample_data()
    
    # 设置设备
    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info(f"使用设备: {device}")
    
    # 创建分词器
    tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
    
    # 加载数据
    train_loader, val_loader, test_loader, processor = load_and_prepare_emotion_data(
        file_path="data/sample_emotion_data.csv",
        text_column="text",
        emotion_columns=["valence", "arousal", "dominance"],
        test_size=0.2,
        val_size=0.1,
        batch_size=16,
        max_length=512,
        tokenizer=tokenizer,
        random_state=42,
        normalize=True
    )
    
    # 创建模型
    model = create_emotion_dimension_model(
        model_name="bert-base-uncased",
        emotion_dimensions=["valence", "arousal", "dominance"],
        dimension_ranges={
            "valence": (-1.0, 1.0),
            "arousal": (-1.0, 1.0),
            "dominance": (-1.0, 1.0)
        },
        model_type="dimension",
        hidden_dims=[768, 256, 64],
        dropout_rate=0.3,
        freeze_bert=False,
        max_length=512,
        use_attention_pooling=True
    )
    
    # 创建训练器
    trainer = EmotionDimensionTrainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        test_loader=test_loader,
        device=device,
        learning_rate=2e-5,
        weight_decay=0.01,
        warmup_steps=500,
        max_grad_norm=1.0,
        save_dir="checkpoints",
        log_dir="logs"
    )
    
    # 开始训练
    history = trainer.train(num_epochs=3, save_best=True)
    
    # 评估模型
    test_metrics = trainer.evaluate()
    
    logger.info("训练完成！")
    logger.info(f"测试集MSE: {test_metrics['overall']['mse']:.4f}")
    logger.info(f"测试集MAE: {test_metrics['overall']['mae']:.4f}")
    logger.info(f"测试集R²: {test_metrics['overall']['r2']:.4f}")
    
    return model, test_loader, processor


def evaluate_model(model, test_loader):
    """评估模型"""
    logger.info("开始评估模型...")
    
    # 设置设备
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # 创建评估器
    evaluator = EmotionDimensionEvaluator(
        model=model,
        emotion_dimensions=["valence", "arousal", "dominance"],
        device=device
    )
    
    # 生成综合评估报告
    results = evaluator.generate_comprehensive_report(
        test_loader, 
        save_dir="evaluation_report"
    )
    
    logger.info("评估完成！")
    return results


def predict_emotions(model, texts):
    """预测情感维度"""
    logger.info("开始预测情感维度...")
    
    # 设置设备
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)
    
    # 预测情感维度
    predictions = model.predict_dimensions(texts)
    
    # 显示结果
    for i, text in enumerate(texts):
        logger.info(f"文本: {text}")
        for dimension, value in predictions.items():
            logger.info(f"  {dimension}: {value[i]:.4f}")
        
        # 获取情感轮廓和解释
        emotion_profile = model.get_emotion_profile(text)
        interpretation = model.interpret_emotion(emotion_profile)
        logger.info(f"  情感解释: {interpretation}")
        logger.info("-" * 50)
    
    return predictions


def main():
    """主函数"""
    logger.info("情感维度模型示例开始...")
    
    try:
        # 训练模型
        model, test_loader, processor = train_emotion_dimension_model()
        
        # 评估模型
        results = evaluate_model(model, test_loader)
        
        # 预测示例文本
        test_texts = [
            "This movie is absolutely fantastic!",
            "I hate this boring film.",
            "The weather is okay today.",
            "I'm so excited about this new opportunity!",
            "This is the worst experience ever."
        ]
        
        predictions = predict_emotions(model, test_texts)
        
        logger.info("情感维度模型示例完成！")
        
    except Exception as e:
        logger.error(f"示例运行出错: {e}")
        raise


if __name__ == "__main__":
    main()

