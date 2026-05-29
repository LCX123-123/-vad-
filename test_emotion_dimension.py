"""
情感维度模型测试脚本
验证情感维度模型的基本功能
"""

import torch
import numpy as np
from transformers import AutoTokenizer
import logging

# 导入情感维度模块
from emotion_dimension_model import create_emotion_dimension_model, EMOTION_DIMENSION_CONFIGS
from emotion_data_loader import create_sample_emotion_data
from emotion_trainer import EmotionDimensionTrainer
from emotion_evaluator import EmotionDimensionEvaluator

# 设置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def test_model_creation():
    """测试模型创建"""
    logger.info("测试模型创建...")
    
    try:
        # 创建VAD模型
        model = create_emotion_dimension_model(
            model_name="bert-base-uncased",
            emotion_dimensions=["valence", "arousal", "dominance"],
            model_type="dimension"
        )
        
        logger.info(f"模型创建成功，参数数量: {sum(p.numel() for p in model.parameters()):,}")
        return model
        
    except Exception as e:
        logger.error(f"模型创建失败: {e}")
        return None


def test_data_creation():
    """测试数据创建"""
    logger.info("测试数据创建...")
    
    try:
        # 创建示例数据
        sample_df = create_sample_emotion_data(
            num_samples=100,
            emotion_dimensions=["valence", "arousal", "dominance"],
            save_path="data/test_emotion_data.csv"
        )
        
        logger.info(f"数据创建成功，形状: {sample_df.shape}")
        logger.info(f"数据列: {list(sample_df.columns)}")
        return sample_df
        
    except Exception as e:
        logger.error(f"数据创建失败: {e}")
        return None


def test_model_prediction(model):
    """测试模型预测"""
    logger.info("测试模型预测...")
    
    try:
        # 测试文本
        test_texts = [
            "This movie is absolutely fantastic!",
            "I hate this boring film.",
            "The weather is okay today."
        ]
        
        # 预测情感维度
        predictions = model.predict_dimensions(test_texts)
        
        logger.info("预测结果:")
        for i, text in enumerate(test_texts):
            logger.info(f"文本: {text}")
            for dimension, values in predictions.items():
                logger.info(f"  {dimension}: {values[i]:.4f}")
        
        # 测试情感轮廓
        emotion_profile = model.get_emotion_profile(test_texts[0])
        interpretation = model.interpret_emotion(emotion_profile)
        
        logger.info(f"情感轮廓: {emotion_profile}")
        logger.info(f"情感解释: {interpretation}")
        
        return True
        
    except Exception as e:
        logger.error(f"模型预测失败: {e}")
        return False


def test_configurations():
    """测试预定义配置"""
    logger.info("测试预定义配置...")
    
    try:
        # 测试VAD配置
        vad_config = EMOTION_DIMENSION_CONFIGS["vad"]
        logger.info(f"VAD配置: {vad_config}")
        
        # 测试VA配置
        va_config = EMOTION_DIMENSION_CONFIGS["va"]
        logger.info(f"VA配置: {va_config}")
        
        # 测试Ekman配置
        ekman_config = EMOTION_DIMENSION_CONFIGS["ekman"]
        logger.info(f"Ekman配置: {ekman_config}")
        
        return True
        
    except Exception as e:
        logger.error(f"配置测试失败: {e}")
        return False


def test_model_forward():
    """测试模型前向传播"""
    logger.info("测试模型前向传播...")
    
    try:
        # 创建模型
        model = create_emotion_dimension_model(
            model_name="bert-base-uncased",
            emotion_dimensions=["valence", "arousal", "dominance"],
            model_type="dimension"
        )
        
        # 创建分词器
        tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
        
        # 准备输入
        text = "This is a test sentence."
        encoding = tokenizer(
            text,
            max_length=512,
            padding=True,
            truncation=True,
            return_tensors='pt'
        )
        
        # 前向传播
        model.eval()
        with torch.no_grad():
            predictions = model(
                input_ids=encoding['input_ids'],
                attention_mask=encoding['attention_mask']
            )
        
        logger.info(f"前向传播成功，输出形状: {predictions.shape}")
        logger.info(f"预测值: {predictions[0].tolist()}")
        
        return True
        
    except Exception as e:
        logger.error(f"前向传播测试失败: {e}")
        return False


def main():
    """主测试函数"""
    logger.info("开始情感维度模型测试...")
    
    # 测试配置
    if not test_configurations():
        logger.error("配置测试失败")
        return
    
    # 测试数据创建
    sample_df = test_data_creation()
    if sample_df is None:
        logger.error("数据创建测试失败")
        return
    
    # 测试模型创建
    model = test_model_creation()
    if model is None:
        logger.error("模型创建测试失败")
        return
    
    # 测试前向传播
    if not test_model_forward():
        logger.error("前向传播测试失败")
        return
    
    # 测试模型预测
    if not test_model_prediction(model):
        logger.error("模型预测测试失败")
        return
    
    logger.info("所有测试通过！情感维度模型工作正常。")


if __name__ == "__main__":
    main()
