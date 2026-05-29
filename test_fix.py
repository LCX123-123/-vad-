"""
测试情感维度模型修复
"""

import torch
from transformers import AutoTokenizer
from emotion_dimension_model import create_emotion_dimension_model

def test_model_forward():
    """测试模型前向传播"""
    print("测试情感维度模型前向传播...")
    
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
        
        print(f"前向传播成功，输出形状: {predictions.shape}")
        print(f"预测值: {predictions[0].tolist()}")
        
        # 测试预测功能
        test_texts = ["This movie is great!", "I hate this film."]
        predictions = model.predict_dimensions(test_texts)
        
        print("预测结果:")
        for i, text in enumerate(test_texts):
            print(f"文本: {text}")
            for dimension, values in predictions.items():
                print(f"  {dimension}: {values[i]:.4f}")
        
        print("测试通过！模型修复成功。")
        return True
        
    except Exception as e:
        print(f"测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    test_model_forward()


