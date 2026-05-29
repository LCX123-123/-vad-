"""
离线模式工具
处理网络连接问题和本地模型缓存
"""

import os
import requests
import logging
from pathlib import Path
from typing import Optional, Dict, Any
import json
import shutil

logger = logging.getLogger(__name__)


def check_internet_connection() -> bool:
    """检查网络连接"""
    try:
        response = requests.get("https://huggingface.co", timeout=5)
        return response.status_code == 200
    except:
        return False


def get_local_model_path(model_name: str, cache_dir: str = "models") -> Optional[str]:
    """获取本地模型路径"""
    cache_path = Path(cache_dir)
    model_path = cache_path / model_name
    
    if model_path.exists() and (model_path / "config.json").exists():
        return str(model_path)
    
    return None


def download_model_offline(model_name: str, cache_dir: str = "models") -> bool:
    """
    离线下载模型（需要手动下载）
    返回是否成功找到本地模型
    """
    cache_path = Path(cache_dir)
    cache_path.mkdir(exist_ok=True)
    
    model_path = cache_path / model_name
    
    if model_path.exists():
        logger.info(f"找到本地模型: {model_path}")
        return True
    
    logger.warning(f"未找到本地模型: {model_name}")
    logger.info("请手动下载模型文件到以下目录:")
    logger.info(f"  {model_path}")
    logger.info("需要的文件:")
    logger.info("  - config.json")
    logger.info("  - pytorch_model.bin 或 model.safetensors")
    logger.info("  - tokenizer.json")
    logger.info("  - tokenizer_config.json")
    logger.info("  - vocab.txt")
    
    return False


def create_minimal_model_config(model_name: str, cache_dir: str = "models"):
    """创建最小化的模型配置（用于测试）"""
    cache_path = Path(cache_dir)
    model_path = cache_path / model_name
    model_path.mkdir(parents=True, exist_ok=True)
    
    # 创建基本的config.json
    config = {
        "architectures": ["BertForSequenceClassification"],
        "attention_probs_dropout_prob": 0.1,
        "hidden_act": "gelu",
        "hidden_dropout_prob": 0.1,
        "hidden_size": 768,
        "initializer_range": 0.02,
        "intermediate_size": 3072,
        "layer_norm_eps": 1e-12,
        "max_position_embeddings": 512,
        "model_type": "bert",
        "num_attention_heads": 12,
        "num_hidden_layers": 12,
        "pad_token_id": 0,
        "type_vocab_size": 2,
        "vocab_size": 30522,
        "num_labels": 2,
        "id2label": {"0": "负面", "1": "正面"},
        "label2id": {"负面": 0, "正面": 1}
    }
    
    config_path = model_path / "config.json"
    with open(config_path, 'w', encoding='utf-8') as f:
        json.dump(config, f, indent=2, ensure_ascii=False)
    
    # 创建基本的tokenizer配置
    tokenizer_config = {
        "do_lower_case": True,
        "init_inputs": [],
        "model_max_length": 512,
        "name_or_path": model_name,
        "tokenizer_class": "BertTokenizer",
        "vocab_size": 30522
    }
    
    tokenizer_config_path = model_path / "tokenizer_config.json"
    with open(tokenizer_config_path, 'w', encoding='utf-8') as f:
        json.dump(tokenizer_config, f, indent=2, ensure_ascii=False)
    
    logger.info(f"已创建最小化模型配置: {model_path}")
    return str(model_path)


def setup_offline_mode(model_name: str = "bert-base-uncased", cache_dir: str = "models") -> str:
    """
    设置离线模式
    返回可用的模型路径
    """
    logger.info("设置离线模式...")
    
    # 检查网络连接
    if check_internet_connection():
        logger.info("网络连接正常，可以使用在线模式")
        return model_name
    
    logger.warning("网络连接不可用，切换到离线模式")
    
    # 检查本地缓存
    local_path = get_local_model_path(model_name, cache_dir)
    if local_path:
        logger.info(f"使用本地缓存模型: {local_path}")
        return local_path
    
    # 尝试下载或创建最小配置
    if download_model_offline(model_name, cache_dir):
        return get_local_model_path(model_name, cache_dir)
    else:
        logger.info("创建最小化模型配置用于测试...")
        return create_minimal_model_config(model_name, cache_dir)


def get_model_download_instructions():
    """获取模型下载说明"""
    instructions = """
=== 模型下载说明 ===

由于网络连接问题，您需要手动下载预训练模型。

方法1: 使用git clone
```bash
# 创建模型目录
mkdir -p models/bert-base-uncased
cd models/bert-base-uncased

# 克隆模型文件
git clone https://huggingface.co/bert-base-uncased .
```

方法2: 使用huggingface-hub
```bash
pip install huggingface-hub
python -c "
from huggingface_hub import snapshot_download
snapshot_download(repo_id='bert-base-uncased', local_dir='models/bert-base-uncased')
"
```

方法3: 使用transformers库
```bash
python -c "
from transformers import AutoTokenizer, AutoModel
tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')
model = AutoModel.from_pretrained('bert-base-uncased')
tokenizer.save_pretrained('models/bert-base-uncased')
model.save_pretrained('models/bert-base-uncased')
"
```

下载完成后，模型目录应包含以下文件:
- config.json
- pytorch_model.bin 或 model.safetensors
- tokenizer.json
- tokenizer_config.json
- vocab.txt
- tokenizer.json

然后重新运行训练命令。
"""
    return instructions


if __name__ == "__main__":
    # 测试离线模式设置
    print("测试离线模式设置...")
    
    # 检查网络
    if check_internet_connection():
        print("✅ 网络连接正常")
    else:
        print("❌ 网络连接不可用")
    
    # 设置离线模式
    model_path = setup_offline_mode()
    print(f"模型路径: {model_path}")
    
    # 显示下载说明
    print(get_model_download_instructions())
