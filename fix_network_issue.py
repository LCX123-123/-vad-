"""
网络问题修复脚本
解决Hugging Face模型下载问题
"""

import os
import sys
import subprocess
import requests
from pathlib import Path
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def check_network():
    """检查网络连接"""
    try:
        response = requests.get("https://huggingface.co", timeout=10)
        if response.status_code == 200:
            logger.info("✅ 网络连接正常")
            return True
        else:
            logger.warning(f"⚠️ 网络响应异常: {response.status_code}")
            return False
    except Exception as e:
        logger.error(f"❌ 网络连接失败: {e}")
        return False


def install_requirements():
    """安装必要的依赖"""
    logger.info("安装必要的依赖包...")
    
    packages = [
        "torch",
        "transformers",
        "pandas",
        "scikit-learn",
        "tqdm",
        "requests"
    ]
    
    for package in packages:
        try:
            subprocess.check_call([sys.executable, "-m", "pip", "install", package])
            logger.info(f"✅ {package} 安装成功")
        except subprocess.CalledProcessError:
            logger.error(f"❌ {package} 安装失败")


def download_model_manually():
    """手动下载模型"""
    logger.info("开始手动下载BERT模型...")
    
    # 创建模型目录
    model_dir = Path("models/bert-base-uncased")
    model_dir.mkdir(parents=True, exist_ok=True)
    
    # 使用transformers库下载
    try:
        from transformers import AutoTokenizer, AutoModel
        
        logger.info("下载tokenizer...")
        tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
        tokenizer.save_pretrained(str(model_dir))
        
        logger.info("下载模型...")
        model = AutoModel.from_pretrained("bert-base-uncased")
        model.save_pretrained(str(model_dir))
        
        logger.info(f"✅ 模型下载成功，保存到: {model_dir}")
        return True
        
    except Exception as e:
        logger.error(f"❌ 模型下载失败: {e}")
        return False


def create_simple_config():
    """创建简单的配置文件"""
    logger.info("创建简单配置...")
    
    # 创建模型目录
    model_dir = Path("models/bert-base-uncased")
    model_dir.mkdir(parents=True, exist_ok=True)
    
    # 创建config.json
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
    
    import json
    config_path = model_dir / "config.json"
    with open(config_path, 'w', encoding='utf-8') as f:
        json.dump(config, f, indent=2, ensure_ascii=False)
    
    logger.info(f"✅ 配置文件已创建: {config_path}")


def test_simple_model():
    """测试简单模型"""
    logger.info("测试简单模型...")
    
    try:
        from transformers import AutoTokenizer, AutoModel
        
        model_path = "models/bert-base-uncased"
        
        # 测试加载
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        model = AutoModel.from_pretrained(model_path)
        
        # 测试预测
        text = "This is a test sentence."
        inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True)
        
        with torch.no_grad():
            outputs = model(**inputs)
        
        logger.info("✅ 模型测试成功")
        return True
        
    except Exception as e:
        logger.error(f"❌ 模型测试失败: {e}")
        return False


def main():
    """主函数"""
    print("=== 网络问题修复脚本 ===\n")
    
    # 1. 检查网络
    print("1. 检查网络连接...")
    network_ok = check_network()
    
    # 2. 安装依赖
    print("\n2. 安装依赖包...")
    install_requirements()
    
    # 3. 尝试下载模型
    if network_ok:
        print("\n3. 下载模型...")
        download_success = download_model_manually()
        
        if download_success:
            print("\n4. 测试模型...")
            test_simple_model()
        else:
            print("\n3. 创建简单配置...")
            create_simple_config()
    else:
        print("\n3. 网络不可用，创建简单配置...")
        create_simple_config()
    
    # 4. 提供解决方案
    print("\n=== 解决方案 ===")
    print("如果问题仍然存在，请尝试以下方法:")
    print("\n方法1: 使用镜像源")
    print("export HF_ENDPOINT=https://hf-mirror.com")
    print("python main.py --mode train")
    
    print("\n方法2: 手动下载模型")
    print("mkdir -p models/bert-base-uncased")
    print("cd models/bert-base-uncased")
    print("git clone https://huggingface.co/bert-base-uncased .")
    
    print("\n方法3: 使用离线测试")
    print("python test_offline.py")
    
    print("\n方法4: 使用代理")
    print("pip install --proxy http://proxy:port transformers")
    
    print("\n修复完成！请重新运行训练命令。")


if __name__ == "__main__":
    main()
