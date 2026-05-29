"""
使用示例脚本
演示如何使用基于Transformer的NLP分类系统
"""

import torch
from transformers import AutoTokenizer
from transformer_model import create_model
from data_loader import load_and_prepare_data
from trainer import train_model
from inference import SentimentPredictor

def quick_demo():
    """快速演示"""
    print("=== 基于Transformer的NLP分类系统演示 ===\n")
    
    # 1. 创建模型
    print("1. 创建Transformer模型...")
    model = create_model(
        model_name="bert-base-uncased",
        num_classes=2,
        model_type="simple",
        dropout_rate=0.3
    )
    print(f"模型参数数量: {sum(p.numel() for p in model.parameters()):,}")
    print(f"可训练参数数量: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}\n")
    
    # 2. 测试预测功能（使用随机权重）
    print("2. 测试预测功能...")
    test_texts = [
        "This movie is absolutely fantastic! I love it!",
        "I hate this boring film. It's terrible.",
        "The acting was okay, but the plot was confusing.",
        "Amazing cinematography and great performances!"
    ]
    
    # 创建分词器
    tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
    
    # 编码文本
    for text in test_texts:
        encoding = tokenizer(
            text,
            max_length=512,
            padding=True,
            truncation=True,
            return_tensors='pt'
        )
        
        # 前向传播（使用随机权重）
        with torch.no_grad():
            logits = model(encoding['input_ids'], encoding['attention_mask'])
            probabilities = torch.softmax(logits, dim=1)
            prediction = torch.argmax(probabilities, dim=1).item()
            confidence = probabilities[0][prediction].item()
        
        sentiment = "正面" if prediction == 1 else "负面"
        print(f"文本: {text}")
        print(f"预测: {sentiment} (置信度: {confidence:.4f})")
        print(f"概率分布: 负面={probabilities[0][0]:.4f}, 正面={probabilities[0][1]:.4f}\n")
    
    print("演示完成！")
    print("\n要开始实际训练，请运行:")
    print("python main.py --mode train")
    print("\n要使用交互式预测，请运行:")
    print("python main.py --mode predict --model_path checkpoints/best_model.pt --interactive")


def data_preview():
    """数据预览"""
    print("=== 数据预览 ===\n")
    
    try:
        import pandas as pd
        
        # 加载英文数据
        print("英文IMDB数据预览:")
        df_en = pd.read_csv("data/imdb_en_clean.csv", nrows=5)
        print(df_en.head())
        print(f"数据形状: {df_en.shape}\n")
        
        # 加载葡萄牙语数据
        print("葡萄牙语IMDB数据预览:")
        df_pt = pd.read_csv("data/imdb_pt_clean.csv", nrows=5)
        print(df_pt.head())
        print(f"数据形状: {df_pt.shape}\n")
        
    except FileNotFoundError:
        print("数据文件不存在，请先运行数据预处理脚本")
    except Exception as e:
        print(f"数据预览出错: {e}")


def model_comparison():
    """模型比较"""
    print("=== 模型比较 ===\n")
    
    models = [
        ("bert-base-uncased", "BERT基础模型"),
        ("distilbert-base-uncased", "DistilBERT轻量模型"),
        ("roberta-base", "RoBERTa基础模型")
    ]
    
    for model_name, description in models:
        try:
            model = create_model(
                model_name=model_name,
                num_classes=2,
                model_type="simple"
            )
            param_count = sum(p.numel() for p in model.parameters())
            trainable_count = sum(p.numel() for p in model.parameters() if p.requires_grad)
            
            print(f"{description} ({model_name}):")
            print(f"  总参数: {param_count:,}")
            print(f"  可训练参数: {trainable_count:,}")
            print(f"  模型大小: {param_count * 4 / 1024 / 1024:.1f} MB\n")
            
        except Exception as e:
            print(f"加载 {model_name} 失败: {e}\n")


if __name__ == "__main__":
    print("选择演示模式:")
    print("1. 快速演示")
    print("2. 数据预览")
    print("3. 模型比较")
    
    choice = input("请输入选择 (1-3): ").strip()
    
    if choice == "1":
        quick_demo()
    elif choice == "2":
        data_preview()
    elif choice == "3":
        model_comparison()
    else:
        print("无效选择，运行快速演示...")
        quick_demo()
