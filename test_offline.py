"""
离线测试脚本
在没有网络连接的情况下测试系统功能
"""

import torch
import torch.nn as nn
import logging
from pathlib import Path
import json

# 设置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class SimpleTokenizer:
    """简单的分词器，用于离线测试"""
    
    def __init__(self, vocab_size=10000):
        self.vocab_size = vocab_size
        self.pad_token_id = 0
        self.cls_token_id = 1
        self.sep_token_id = 2
        self.unk_token_id = 3
    
    def __call__(self, texts, max_length=512, padding=True, truncation=True, return_tensors='pt'):
        """简单的分词实现"""
        if isinstance(texts, str):
            texts = [texts]
        
        input_ids = []
        attention_masks = []
        
        for text in texts:
            # 简单的词汇映射（实际应用中需要更复杂的处理）
            words = text.lower().split()[:max_length-2]  # 保留CLS和SEP的位置
            
            # 转换为token IDs
            token_ids = [self.cls_token_id]
            for word in words:
                # 简单的哈希映射到词汇表
                token_id = hash(word) % (self.vocab_size - 4) + 4
                token_ids.append(token_id)
            token_ids.append(self.sep_token_id)
            
            # 截断或填充
            if len(token_ids) > max_length:
                token_ids = token_ids[:max_length]
                attention_mask = [1] * max_length
            else:
                attention_mask = [1] * len(token_ids)
                if padding:
                    token_ids.extend([self.pad_token_id] * (max_length - len(token_ids)))
                    attention_mask.extend([0] * (max_length - len(attention_mask)))
            
            input_ids.append(token_ids)
            attention_masks.append(attention_mask)
        
        result = {
            'input_ids': torch.tensor(input_ids),
            'attention_mask': torch.tensor(attention_masks)
        }
        
        return result


class SimpleTransformer(nn.Module):
    """简化的Transformer模型，用于离线测试"""
    
    def __init__(self, vocab_size=10000, hidden_size=768, num_classes=2, max_length=512):
        super().__init__()
        self.hidden_size = hidden_size
        self.max_length = max_length
        
        # 嵌入层
        self.embedding = nn.Embedding(vocab_size, hidden_size)
        self.position_embedding = nn.Embedding(max_length, hidden_size)
        
        # 简单的注意力机制
        self.attention = nn.MultiheadAttention(hidden_size, num_heads=8, batch_first=True)
        
        # 分类头
        self.classifier = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_size // 2, num_classes)
        )
        
        # 初始化权重
        self._init_weights()
    
    def _init_weights(self):
        """初始化权重"""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Embedding):
                nn.init.normal_(module.weight, mean=0, std=0.02)
    
    def forward(self, input_ids, attention_mask=None):
        batch_size, seq_len = input_ids.shape
        
        # 词嵌入
        token_embeddings = self.embedding(input_ids)
        
        # 位置嵌入
        positions = torch.arange(seq_len, device=input_ids.device).unsqueeze(0).expand(batch_size, -1)
        position_embeddings = self.position_embedding(positions)
        
        # 组合嵌入
        embeddings = token_embeddings + position_embeddings
        
        # 注意力机制
        if attention_mask is not None:
            # 转换attention_mask格式
            attn_mask = attention_mask == 0
        else:
            attn_mask = None
        
        attended, _ = self.attention(embeddings, embeddings, embeddings, key_padding_mask=attn_mask)
        
        # 使用[CLS] token进行分类
        cls_output = attended[:, 0, :]  # 取第一个token的输出
        
        # 分类
        logits = self.classifier(cls_output)
        
        return logits


def test_offline_system():
    """测试离线系统"""
    logger.info("=== 离线系统测试 ===")
    
    # 1. 创建简单分词器
    logger.info("1. 创建简单分词器...")
    tokenizer = SimpleTokenizer()
    
    # 2. 创建简单模型
    logger.info("2. 创建简单模型...")
    model = SimpleTransformer()
    
    # 3. 测试文本
    test_texts = [
        "This movie is absolutely fantastic!",
        "I hate this boring film.",
        "The acting was okay, but the plot was confusing.",
        "Amazing cinematography and great performances!"
    ]
    
    # 4. 测试预测
    logger.info("3. 测试预测功能...")
    model.eval()
    
    with torch.no_grad():
        for text in test_texts:
            # 分词
            encoding = tokenizer(text, max_length=128)
            input_ids = encoding['input_ids']
            attention_mask = encoding['attention_mask']
            
            # 预测
            logits = model(input_ids, attention_mask)
            probabilities = torch.softmax(logits, dim=1)
            prediction = torch.argmax(probabilities, dim=1).item()
            confidence = probabilities[0][prediction].item()
            
            sentiment = "正面" if prediction == 1 else "负面"
            print(f"文本: {text}")
            print(f"预测: {sentiment} (置信度: {confidence:.4f})")
            print(f"概率分布: 负面={probabilities[0][0]:.4f}, 正面={probabilities[0][1]:.4f}")
            print("-" * 50)
    
    # 5. 模型信息
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    logger.info(f"模型参数统计:")
    logger.info(f"  总参数: {total_params:,}")
    logger.info(f"  可训练参数: {trainable_params:,}")
    logger.info(f"  模型大小: {total_params * 4 / 1024 / 1024:.1f} MB")
    
    logger.info("离线测试完成！")


def create_demo_data():
    """创建演示数据"""
    logger.info("创建演示数据...")
    
    # 创建data目录
    data_dir = Path("data")
    data_dir.mkdir(exist_ok=True)
    
    # 创建简单的演示数据
    demo_data = [
        {"text_clean": "this movie is absolutely fantastic i love it", "sentiment": "pos"},
        {"text_clean": "i hate this boring film it is terrible", "sentiment": "neg"},
        {"text_clean": "the acting was okay but the plot was confusing", "sentiment": "neg"},
        {"text_clean": "amazing cinematography and great performances", "sentiment": "pos"},
        {"text_clean": "waste of time completely disappointed", "sentiment": "neg"},
        {"text_clean": "brilliant direction and outstanding cast", "sentiment": "pos"},
        {"text_clean": "boring and predictable storyline", "sentiment": "neg"},
        {"text_clean": "emotional and touching beautiful film", "sentiment": "pos"},
    ]
    
    # 保存为CSV
    import pandas as pd
    df = pd.DataFrame(demo_data)
    df.to_csv(data_dir / "demo_data.csv", index=False, encoding='utf-8')
    
    logger.info(f"演示数据已保存到: {data_dir / 'demo_data.csv'}")
    return str(data_dir / "demo_data.csv")


def show_network_solutions():
    """显示网络问题解决方案"""
    solutions = """
=== 网络连接问题解决方案 ===

1. 检查网络连接
   - 确保能够访问 https://huggingface.co
   - 检查防火墙设置
   - 尝试使用VPN

2. 手动下载模型
   ```bash
   # 创建模型目录
   mkdir -p models/bert-base-uncased
   cd models/bert-base-uncased
   
   # 使用git下载
   git clone https://huggingface.co/bert-base-uncased .
   ```

3. 使用代理
   ```bash
   export HF_ENDPOINT=https://hf-mirror.com
   ```

4. 离线模式
   - 使用本脚本进行测试
   - 创建简化的模型进行演示

5. 使用本地模型
   - 将模型文件放在 models/ 目录下
   - 确保包含 config.json 等必要文件

当前系统支持离线测试，可以运行:
python test_offline.py
"""
    print(solutions)


if __name__ == "__main__":
    print("选择测试模式:")
    print("1. 离线系统测试")
    print("2. 创建演示数据")
    print("3. 显示网络解决方案")
    
    choice = input("请输入选择 (1-3): ").strip()
    
    if choice == "1":
        test_offline_system()
    elif choice == "2":
        create_demo_data()
    elif choice == "3":
        show_network_solutions()
    else:
        print("无效选择，运行离线测试...")
        test_offline_system()
