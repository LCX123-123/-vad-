"""
简单的情感维度模型测试
"""

import torch
import numpy as np
from transformers import AutoTokenizer

# 导入情感维度模块
from emotion_dimension_model import create_emotion_dimension_model, EMOTION_DIMENSION_CONFIGS

def test_basic_functionality():
    """测试基本功能"""
    print("开始测试情感维度模型...")
    
    try:
        # 测试配置
        print("1. 测试预定义配置...")
        vad_config = EMOTION_DIMENSION_CONFIGS["vad"]
        print(f"VAD配置: {vad_config}")
        
        # 测试模型创建
        print("2. 测试模型创建...")
        model = create_emotion_dimension_model(
            model_name="bert-base-uncased",
            emotion_dimensions=["valence", "arousal", "dominance"],
            model_type="dimension"
        )
        print(f"模型创建成功，参数数量: {sum(p.numel() for p in model.parameters()):,}")
        
        # 测试前向传播
        print("3. 测试前向传播...")
        tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
        text = "This is a test sentence."
        encoding = tokenizer(
            text,
            max_length=512,
            padding=True,
            truncation=True,
            return_tensors='pt'
        )
        
        model.eval()
        with torch.no_grad():
            predictions = model(
                input_ids=encoding['input_ids'],
                attention_mask=encoding['attention_mask']
            )
        
        print(f"前向传播成功，输出形状: {predictions.shape}")
        print(f"预测值: {predictions[0].tolist()}")
        
        # 测试预测功能
        print("4. 测试预测功能...")
        test_texts = ["This movie is great!", "I hate this film."]
        predictions = model.predict_dimensions(test_texts)
        
        for i, text in enumerate(test_texts):
            print(f"文本: {text}")
            for dimension, values in predictions.items():
                print(f"  {dimension}: {values[i]:.4f}")
        
        print("所有测试通过！情感维度模型工作正常。")
        return True
        
    except Exception as e:
        print(f"测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    test_basic_functionality()


