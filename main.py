"""
主程序入口
支持训练、评估和推理的完整NLP分类任务流程
"""

import argparse
import json
import logging
import os
import sys
from pathlib import Path
from typing import Dict, Any

import torch
from transformers import AutoTokenizer

# 导入自定义模块
from transformer_model import create_model, MODEL_CONFIGS
from data_loader import load_and_prepare_data, DataProcessor
from trainer import train_model, ModelTrainer
from inference import SentimentPredictor, ModelEvaluator
from offline_utils import setup_offline_mode, check_internet_connection, get_model_download_instructions

# 导入情感维度模块
from emotion_dimension_model import create_emotion_dimension_model, EMOTION_DIMENSION_CONFIGS
from emotion_data_loader import (
    load_and_prepare_emotion_data,
    create_sample_emotion_data,
    load_and_prepare_emobank,
)
from emotion_trainer import EmotionDimensionTrainer, MultiTaskEmotionTrainer
from emotion_evaluator import EmotionDimensionEvaluator

# 设置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('nlp_classification.log', encoding='utf-8'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)


class Config:
    """配置管理类"""
    
    def __init__(self, config_path: str = None):
        self.config = self._load_default_config()
        
        if config_path and Path(config_path).exists():
            self._load_config(config_path)
    
    def _load_default_config(self) -> Dict[str, Any]:
        """加载默认配置"""
        return {
            # 数据配置
            "data": {
                "data_dir": "data",
                "language": "en",  # "en" 或 "pt"
                "test_size": 0.2,
                "val_size": 0.1,
                "batch_size": 16,
                "max_length": 512,
                "random_state": 42
            },
            
            # 模型配置
            "model": {
                "model_name": "bert-base-uncased",
                "model_type": "simple",  # "simple" 或 "multi_layer"
                "num_classes": 2,
                "dropout_rate": 0.3,
                "freeze_bert": False
            },
            
            # 情感维度配置
            "emotion_dimension": {
                "enabled": False,
                "model_type": "dimension",  # "dimension" 或 "multi_task"
                "emotion_dimensions": ["valence", "arousal", "dominance"],
                "dimension_ranges": {
                    "valence": (-1.0, 1.0),
                    "arousal": (-1.0, 1.0),
                    "dominance": (-1.0, 1.0)
                },
                "hidden_dims": [768, 256, 64],
                "use_attention_pooling": True,
                "normalize_labels": True
            },
            
            # 训练配置
            "training": {
                "num_epochs": 3,
                "learning_rate": 2e-5,
                "weight_decay": 0.01,
                "warmup_steps": 500,
                "max_grad_norm": 1.0,
                "early_stopping_patience": 5,
                "save_best": True
            },
            
            # 系统配置
            "system": {
                "device": "auto",  # "auto", "cuda", "cpu"
                "num_workers": 4,
                "save_dir": "checkpoints",
                "log_dir": "logs"
            }
        }
    
    def _load_config(self, config_path: str):
        """从文件加载配置"""
        with open(config_path, 'r', encoding='utf-8') as f:
            user_config = json.load(f)
        
        # 递归更新配置
        self._update_config(self.config, user_config)
        logger.info(f"已加载配置文件: {config_path}")
    
    def _update_config(self, base_config: Dict, user_config: Dict):
        """递归更新配置"""
        for key, value in user_config.items():
            if key in base_config and isinstance(base_config[key], dict) and isinstance(value, dict):
                self._update_config(base_config[key], value)
            else:
                base_config[key] = value
    
    def get(self, key_path: str, default=None):
        """获取配置值，支持点号分隔的路径"""
        keys = key_path.split('.')
        value = self.config
        
        try:
            for key in keys:
                value = value[key]
            return value
        except (KeyError, TypeError):
            return default
    
    def save(self, config_path: str):
        """保存配置到文件"""
        with open(config_path, 'w', encoding='utf-8') as f:
            json.dump(self.config, f, indent=2, ensure_ascii=False)
        logger.info(f"配置已保存到: {config_path}")


def setup_device(config: Config) -> str:
    """设置计算设备"""
    device_config = config.get("system.device", "auto")
    
    if device_config == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device = device_config
    
    logger.info(f"使用设备: {device}")
    if device == "cuda":
        logger.info(f"GPU信息: {torch.cuda.get_device_name()}")
        logger.info(f"GPU内存: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
    
    return device


def train_command(args, config: Config):
    """训练命令"""
    logger.info("开始训练模式")
    
    # 设置设备
    device = setup_device(config)
    
    # 检查网络连接并设置模型路径
    model_name = config.get("model.model_name", "bert-base-uncased")
    
    try:
        # 尝试在线模式
        if check_internet_connection():
            logger.info("网络连接正常，使用在线模式")
            actual_model_name = model_name
        else:
            logger.warning("网络连接不可用，切换到离线模式")
            actual_model_name = setup_offline_mode(model_name)
    except Exception as e:
        logger.error(f"模型加载失败: {e}")
        logger.info("请按照以下说明下载模型:")
        print(get_model_download_instructions())
        return
    
    # 创建分词器
    try:
        tokenizer = AutoTokenizer.from_pretrained(actual_model_name)
    except Exception as e:
        logger.error(f"分词器加载失败: {e}")
        logger.info("请按照以下说明下载模型:")
        print(get_model_download_instructions())
        return
    
    # 检查是否启用情感维度模型
    if config.get("emotion_dimension.enabled", False):
        train_emotion_dimension_command(args, config, device, actual_model_name, tokenizer)
    else:
        train_classification_command(args, config, device, actual_model_name, tokenizer)


def train_classification_command(args, config: Config, device: str, model_name: str, tokenizer):
    """训练分类模型"""
    logger.info("训练分类模型...")
    
    # 加载数据
    logger.info("加载数据...")
    train_loader, val_loader, test_loader, label_encoder = load_and_prepare_data(
        data_dir=config.get("data.data_dir"),
        language=config.get("data.language"),
        test_size=config.get("data.test_size"),
        val_size=config.get("data.val_size"),
        batch_size=config.get("data.batch_size"),
        max_length=config.get("data.max_length"),
        tokenizer=tokenizer,
        random_state=config.get("data.random_state")
    )
    
    # 创建模型
    logger.info("创建模型...")
    try:
        model = create_model(
            model_name=model_name,
            num_classes=config.get("model.num_classes"),
            model_type=config.get("model.model_type"),
            dropout_rate=config.get("model.dropout_rate"),
            freeze_bert=config.get("model.freeze_bert"),
            max_length=config.get("data.max_length")
        )
    except Exception as e:
        logger.error(f"模型创建失败: {e}")
        logger.info("请按照以下说明下载模型:")
        print(get_model_download_instructions())
        return
    
    # 创建训练器
    trainer = ModelTrainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        test_loader=test_loader,
        device=device,
        learning_rate=config.get("training.learning_rate"),
        weight_decay=config.get("training.weight_decay"),
        warmup_steps=config.get("training.warmup_steps"),
        max_grad_norm=config.get("training.max_grad_norm"),
        save_dir=config.get("system.save_dir")
    )
    
    # 开始训练
    num_epochs = config.get("training.num_epochs")
    history = trainer.train(num_epochs=num_epochs)
    
    # 评估测试集
    logger.info("评估测试集...")
    test_metrics = trainer.evaluate()
    
    # 保存标签编码器
    import pickle
    with open(Path(config.get("system.save_dir")) / "label_encoder.pkl", 'wb') as f:
        pickle.dump(label_encoder, f)
    
    # 保存配置
    config.save(Path(config.get("system.save_dir")) / "config.json")
    
    logger.info("训练完成！")
    logger.info(f"测试集准确率: {test_metrics['accuracy']:.4f}")
    logger.info(f"测试集F1分数: {test_metrics['f1']:.4f}")


def train_emotion_dimension_command(args, config: Config, device: str, model_name: str, tokenizer):
    """训练情感维度模型"""
    logger.info("训练情感维度模型...")
    
    # 获取情感维度配置
    emotion_config = config.get("emotion_dimension", {})
    emotion_dimensions = emotion_config.get("emotion_dimensions", ["valence", "arousal", "dominance"])
    model_type = emotion_config.get("model_type", "dimension")
    
    # 检查数据文件
    data_file = args.data_file if hasattr(args, 'data_file') and args.data_file else "data/sample_emotion_data.csv"
    
    if not Path(data_file).exists() and not getattr(args, 'emobank', False):
        logger.info("创建示例情感数据...")
        create_sample_emotion_data(
            num_samples=1000,
            emotion_dimensions=emotion_dimensions,
            save_path=data_file
        )
    
    # 加载情感数据
    if getattr(args, 'emobank', False):
        logger.info("使用EmoBank解析流程加载数据...")
        train_loader, val_loader, test_loader, processor = load_and_prepare_emobank(
            file_path=data_file,
            test_size=config.get("data.test_size"),
            val_size=config.get("data.val_size"),
            batch_size=config.get("data.batch_size"),
            max_length=config.get("data.max_length"),
            tokenizer=tokenizer,
            random_state=config.get("data.random_state"),
            normalize=emotion_config.get("normalize_labels", True),
            save_normalized_path=str(Path("data") / "emobank_std.csv")
        )
        # 强制维度为VAD
        emotion_dimensions = ["valence", "arousal", "dominance"]
    else:
        logger.info("加载情感数据...")
        train_loader, val_loader, test_loader, processor = load_and_prepare_emotion_data(
            file_path=data_file,
            text_column="text",
            emotion_columns=emotion_dimensions,
            test_size=config.get("data.test_size"),
            val_size=config.get("data.val_size"),
            batch_size=config.get("data.batch_size"),
            max_length=config.get("data.max_length"),
            tokenizer=tokenizer,
            random_state=config.get("data.random_state"),
            normalize=emotion_config.get("normalize_labels", True)
        )
    
    # 创建情感维度模型
    logger.info("创建情感维度模型...")
    try:
        model = create_emotion_dimension_model(
            model_name=model_name,
            emotion_dimensions=emotion_dimensions,
            dimension_ranges=emotion_config.get("dimension_ranges"),
            model_type=model_type,
            hidden_dims=emotion_config.get("hidden_dims", [768, 256, 64]),
            dropout_rate=config.get("model.dropout_rate"),
            freeze_bert=config.get("model.freeze_bert"),
            max_length=config.get("data.max_length"),
            use_attention_pooling=emotion_config.get("use_attention_pooling", True)
        )
    except Exception as e:
        logger.error(f"情感维度模型创建失败: {e}")
        return
    
    # 创建训练器
    trainer = EmotionDimensionTrainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        test_loader=test_loader,
        device=device,
        learning_rate=config.get("training.learning_rate"),
        weight_decay=config.get("training.weight_decay"),
        warmup_steps=config.get("training.warmup_steps"),
        max_grad_norm=config.get("training.max_grad_norm"),
        save_dir=config.get("system.save_dir"),
        log_dir=config.get("system.log_dir")
    )
    
    # 训练或跳过训练
    if getattr(args, 'evaluate_only', False):
        logger.info("跳过训练，直接进入评估阶段 (--evaluate_only)")
        # 从已有最佳模型加载权重，用于“单独评估”
        checkpoint_path = Path(config.get("system.save_dir")) / "best_model.pt"
        if checkpoint_path.exists():
            try:
                checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
                model.load_state_dict(checkpoint['model_state_dict'])
                logger.info(f"已从 {checkpoint_path} 加载已训练的情感维度模型权重，用于评估")
            except Exception as e:
                logger.error(f"加载情感维度模型权重失败，将使用当前模型参数进行评估: {e}")
        else:
            logger.warning(f"未找到已训练情感维度模型权重: {checkpoint_path}，将使用当前模型参数进行评估")
    else:
        num_epochs = config.get("training.num_epochs")
        history = trainer.train(num_epochs=num_epochs)
    
    # 评估测试集
    logger.info("评估测试集...")
    test_metrics = trainer.evaluate()
    
    # 保存配置
    config.save(Path(config.get("system.save_dir")) / "config.json")
    
    logger.info("情感维度模型训练完成！")
    logger.info(f"测试集MSE: {test_metrics['overall']['mse']:.4f}")
    logger.info(f"测试集MAE: {test_metrics['overall']['mae']:.4f}")
    logger.info(f"测试集R²: {test_metrics['overall']['r2']:.4f}")
    
    # 生成评估报告
    logger.info("生成评估报告...")
    evaluator = EmotionDimensionEvaluator(model, emotion_dimensions, device)
    evaluator.generate_comprehensive_report(test_loader, "evaluation_report")


def evaluate_command(args, config: Config):
    """评估命令"""
    logger.info("开始评估模式")
    
    # 检查模型文件
    model_path = args.model_path
    if not Path(model_path).exists():
        raise FileNotFoundError(f"模型文件不存在: {model_path}")
    
    # 设置设备
    device = setup_device(config)
    
    # 创建分词器
    model_name = config.get("model.model_name", "bert-base-uncased")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    # 加载数据
    logger.info("加载测试数据...")
    train_loader, val_loader, test_loader, label_encoder = load_and_prepare_data(
        data_dir=config.get("data.data_dir"),
        language=config.get("data.language"),
        test_size=config.get("data.test_size"),
        val_size=config.get("data.val_size"),
        batch_size=config.get("data.batch_size"),
        max_length=config.get("data.max_length"),
        tokenizer=tokenizer,
        random_state=config.get("data.random_state")
    )
    
    # 创建预测器
    predictor = SentimentPredictor(
        model_path=model_path,
        model_name=model_name,
        device=device
    )
    
    # 创建评估器
    evaluator = ModelEvaluator(predictor)
    
    # 评估模型
    metrics = evaluator.evaluate_dataset(test_loader, label_encoder)
    
    # 显示结果
    logger.info("评估结果:")
    logger.info(f"  准确率: {metrics['accuracy']:.4f}")
    logger.info(f"  精确率: {metrics['precision']:.4f}")
    logger.info(f"  召回率: {metrics['recall']:.4f}")
    logger.info(f"  F1分数: {metrics['f1']:.4f}")
    
    # 分析错误样本
    if args.analyze_errors:
        logger.info("分析错误样本...")
        error_examples = evaluator.analyze_errors(test_loader, label_encoder, num_examples=5)
        
        logger.info("错误样本示例:")
        for i, example in enumerate(error_examples):
            logger.info(f"  样本 {i+1}:")
            logger.info(f"    文本: {example['text'][:100]}...")
            logger.info(f"    真实标签: {example['true_label']}")
            logger.info(f"    预测标签: {example['predicted_label']}")
            logger.info(f"    置信度: {example['confidence']:.4f}")


def predict_command(args, config: Config):
    """预测命令"""
    logger.info("开始预测模式")
    
    # 检查模型文件
    model_path = args.model_path
    if not Path(model_path).exists():
        raise FileNotFoundError(f"模型文件不存在: {model_path}")
    
    # 设置设备
    device = setup_device(config)
    
    # 如果指定使用图像模型，走图像预测流程
    if getattr(args, 'image_model', False):
        logger.info("启用图像情感维度预测模式")
        from image_fer2013_vad import VADRegressor
        from PIL import Image
        import torch.nn.functional as F
        
        # 加载图像VAD模型
        model = VADRegressor(pretrained=False)
        checkpoint = torch.load(model_path, map_location=device, weights_only=False)
        model.load_state_dict(checkpoint['model_state_dict'])
        model.to(device)
        model.eval()
        
        def load_and_preprocess_image(image_path: str) -> torch.Tensor:
            """加载并预处理图像为模型输入格式"""
            img = Image.open(image_path).convert('L')  # 转为灰度
            # 转为numpy并reshape为48x48（FER2013原始尺寸）
            import numpy as np
            img = img.resize((48, 48), Image.BILINEAR)
            arr = np.array(img, dtype=np.uint8)
            # 转为tensor并归一化
            img_tensor = torch.from_numpy(arr).unsqueeze(0).float() / 255.0  # (1, 48, 48)
            # resize到224并复制为3通道（匹配训练时的预处理）
            img_tensor = F.interpolate(img_tensor.unsqueeze(0), size=(224, 224), mode="bilinear", align_corners=False).squeeze(0)
            img_tensor = img_tensor.repeat(3, 1, 1)  # (3, 224, 224)
            return img_tensor.unsqueeze(0)  # (1, 3, 224, 224)
        
        def interpret_vad(vad: Dict[str, float]) -> str:
            """解释VAD值"""
            v, a, d = vad.get('valence', 0), vad.get('arousal', 0), vad.get('dominance', 0)
            v_str = "非常积极" if v > 0.5 else "积极" if v > 0 else "中性" if v > -0.5 else "消极"
            a_str = "高唤醒" if a > 0.3 else "中等唤醒" if a > -0.3 else "低唤醒"
            d_str = "高支配性" if d > 0.3 else "中等支配性" if d > -0.3 else "低支配性"
            return f"{v_str} | {a_str} | {d_str}"
        
        if args.image_path:
            # 单图像预测
            img_tensor = load_and_preprocess_image(args.image_path).to(device)
            with torch.no_grad():
                pred = model(img_tensor).cpu().squeeze().numpy()
            vad = {"valence": float(pred[0]), "arousal": float(pred[1]), "dominance": float(pred[2])}
            print("预测情感维度:")
            for dim, val in vad.items():
                print(f"  {dim}: {val:.4f}")
            print(f"解释: {interpret_vad(vad)}")
        elif args.interactive:
            # 交互式图像预测
            print("进入图像情感维度交互预测，输入图像路径并回车。输入 'exit' 退出。")
            while True:
                try:
                    img_path = input("图像路径> ").strip()
                except (EOFError, KeyboardInterrupt):
                    print("\n已退出。")
                    break
                if img_path.lower() in {"exit", "quit", ":q"}:
                    print("已退出。")
                    break
                if not img_path:
                    continue
                if not Path(img_path).exists():
                    print(f"错误: 文件不存在: {img_path}")
                    continue
                try:
                    img_tensor = load_and_preprocess_image(img_path).to(device)
                    with torch.no_grad():
                        pred = model(img_tensor).cpu().squeeze().numpy()
                    vad = {"valence": float(pred[0]), "arousal": float(pred[1]), "dominance": float(pred[2])}
                    print("预测情感维度:")
                    for dim, val in vad.items():
                        print(f"  {dim}: {val:.4f}")
                    print(f"解释: {interpret_vad(vad)}")
                except Exception as e:
                    print(f"错误: 处理图像时出错: {e}")
        else:
            print("请指定预测方式: --image_path 或 --interactive")
        return
    
    # 如果指定使用音频模型，走音频 VAD 预测流程
    if getattr(args, 'audio_model', False):
        logger.info("启用音频情感维度预测模式")
        from audio_vad_model import AudioVADRegressor, interpret_audio_vad
        import torchaudio
        
        model = AudioVADRegressor()
        checkpoint = torch.load(model_path, map_location=device, weights_only=False)
        model.load_state_dict(checkpoint["model_state_dict"])
        model.to(device)
        model.eval()

        def load_and_preprocess_audio(audio_path: str, target_sr: int = 16000, max_duration: float = 6.0) -> torch.Tensor:
            """加载并预处理音频为模型输入格式 [1,1,T]"""
            wav, sr = torchaudio.load(audio_path)
            # 单声道
            if wav.shape[0] > 1:
                wav = wav.mean(dim=0, keepdim=True)
            # 重采样
            if sr != target_sr:
                resampler = torchaudio.transforms.Resample(orig_freq=sr, new_freq=target_sr)
                wav = resampler(wav)
            max_samples = int(target_sr * max_duration)
            if wav.shape[1] > max_samples:
                wav = wav[:, :max_samples]
            else:
                pad_len = max_samples - wav.shape[1]
                if pad_len > 0:
                    wav = torch.nn.functional.pad(wav, (0, pad_len))
            return wav.unsqueeze(0)  # [1,1,T]

        def predict_single_audio(path: str):
            if not Path(path).exists():
                print(f"错误: 音频文件不存在: {path}")
                return
            audio = load_and_preprocess_audio(path).to(device)
            with torch.no_grad():
                pred = model(audio).cpu().squeeze().numpy()
            vad = {"valence": float(pred[0]), "arousal": float(pred[1]), "dominance": float(pred[2])}
            print("预测情感维度:")
            for dim, val in vad.items():
                print(f"  {dim}: {val:.4f}")
            print(f"解释: {interpret_audio_vad(vad)}")

        if args.audio_path:
            predict_single_audio(args.audio_path)
        elif args.interactive:
            print("进入音频情感维度交互预测，输入音频路径并回车。输入 'exit' 退出。")
            while True:
                try:
                    audio_path = input("音频路径> ").strip()
                except (EOFError, KeyboardInterrupt):
                    print("\n已退出。")
                    break
                if audio_path.lower() in {"exit", "quit", ":q"}:
                    print("已退出。")
                    break
                if not audio_path:
                    continue
                predict_single_audio(audio_path)
        else:
            print("请指定预测方式: --audio_path 或 --interactive")
        return
    
    # 如果启用情感维度，则走情感维度交互/预测流程
    if config.get("emotion_dimension.enabled", False) or getattr(args, 'emotion_dimension', False):
        logger.info("启用情感维度预测模式")
        emotion_config = config.get("emotion_dimension", {})
        emotion_dimensions = emotion_config.get("emotion_dimensions", ["valence", "arousal", "dominance"])
        model_name = config.get("model.model_name", "bert-base-uncased")
        
        # 构建模型并加载权重
        from emotion_dimension_model import create_emotion_dimension_model
        model = create_emotion_dimension_model(
            model_name=model_name,
            emotion_dimensions=emotion_dimensions,
            model_type=emotion_config.get("model_type", "dimension"),
            hidden_dims=emotion_config.get("hidden_dims", [768, 256, 64]),
            dropout_rate=config.get("model.dropout_rate"),
            freeze_bert=False,
            max_length=config.get("data.max_length"),
            use_attention_pooling=emotion_config.get("use_attention_pooling", True)
        )
        checkpoint = torch.load(model_path, map_location=device, weights_only=False)
        model.load_state_dict(checkpoint['model_state_dict'])
        model.to(device)
        model.eval()
        
        def interactive_loop():
            print("进入情感维度交互预测，输入文本并回车。输入 'exit' 退出。")
            while True:
                try:
                    text = input("文本> ").strip()
                except (EOFError, KeyboardInterrupt):
                    print("\n已退出。")
                    break
                if text.lower() in {"exit", "quit", ":q"}:
                    print("已退出。")
                    break
                if not text:
                    continue
                with torch.no_grad():
                    preds = model.predict_dimensions([text])
                    profile = {dim: float(preds[dim][0]) for dim in emotion_dimensions}
                # 简要打印
                print("预测情感维度:")
                for dim in emotion_dimensions:
                    print(f"  {dim}: {profile[dim]:.4f}")
                # 解释
                try:
                    interp = model.interpret_emotion(profile)
                    print(f"解释: {interp}")
                except Exception:
                    pass
        
        if args.text:
            with torch.no_grad():
                preds = model.predict_dimensions([args.text])
                for dim in emotion_dimensions:
                    print(f"{dim}: {float(preds[dim][0]):.4f}")
        elif args.file:
            import pandas as pd
            text_col = args.text_column or "text"
            df = pd.read_csv(args.file, encoding='utf-8')
            if text_col not in df.columns:
                raise KeyError(f"文件中未找到文本列: {text_col}")
            texts = df[text_col].astype(str).tolist()
            preds = model.predict_dimensions(texts)
            for dim in emotion_dimensions:
                df[f"pred_{dim}"] = preds[dim]
            output_path = args.output if args.output else f"predictions_{Path(args.file).stem}.csv"
            df.to_csv(output_path, index=False, encoding='utf-8')
            print(f"预测完成，结果已保存到: {output_path}")
        elif args.interactive:
            interactive_loop()
        else:
            print("请指定预测方式: --text, --file 或 --interactive")
        return
    
    # 否则走原有分类预测流程
    model_name = config.get("model.model_name", "bert-base-uncased")
    predictor = SentimentPredictor(
        model_path=model_path,
        model_name=model_name,
        device=device
    )
    
    if args.text:
        # 单文本预测
        result = predictor.predict_single(args.text)
        print(f"\n文本: {result['text']}")
        print(f"预测: {result['prediction']}")
        print(f"置信度: {result['confidence']:.4f}")
        print("概率分布:")
        for label, prob in result['probabilities'].items():
            print(f"  {label}: {prob:.4f}")
    
    elif args.file:
        # 文件预测
        output_path = args.output if args.output else f"predictions_{Path(args.file).stem}.csv"
        df = predictor.predict_file(args.file, args.text_column, output_path)
        print(f"预测完成，结果已保存到: {output_path}")
        print(f"预测样本数: {len(df)}")
    
    elif args.interactive:
        # 交互式预测
        predictor.interactive_predict()
    
    else:
        print("请指定预测方式: --text, --file 或 --interactive")


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description="基于Transformer的NLP分类任务")
    parser.add_argument("--config", type=str, help="配置文件路径")
    parser.add_argument("--mode", type=str, choices=[
        "train",
        "evaluate",
        "predict",
        "train_image",
        "train_emotion_image",
        "train_audio",
        "evaluate_image_vad",
        "evaluate_audio_vad",
        "train_mosei_senti",
        "train_mosei_emotion",
        "predict_mosei_senti",
        "predict_mosei_emotion",
        "evaluate_mosei_senti",
        "evaluate_mosei_emotion",
        "export_mosei_senti",
        "export_mosei_emotion",
        "predict_video_fusion",
    ], 
                       required=True, help="运行模式")
    # 图像 VAD 训练参数
    parser.add_argument("--image_vad", action="store_true", help="使用图像回归 VAD（从 fer2013 映射）")
    parser.add_argument("--fer_vad_csv", type=str, help="fer2013.csv 路径（用于图像VAD训练，或单一CSV包含Usage列）")
    parser.add_argument("--fer_vad_val_csv", type=str, help="图像VAD验证集 CSV")
    parser.add_argument("--fer_vad_test_csv", type=str, help="图像VAD测试集 CSV")
    parser.add_argument("--image_vad_epochs", type=int, help="图像VAD训练轮数")
    # 图像 FER2013 参数
    parser.add_argument("--fer_csv", type=str, help="fer2013.csv 路径（用于图像训练，或单一CSV包含Usage列）")
    parser.add_argument("--fer_val_csv", type=str, help="验证集 CSV（若与train分开时使用）")
    parser.add_argument("--fer_test_csv", type=str, help="测试集 CSV（若与train分开时使用）")
    parser.add_argument("--image_epochs", type=int, help="图像训练轮数")
    parser.add_argument("--image_batch_size", type=int, help="图像批大小")
    # 音频 VAD 参数
    parser.add_argument("--audio_vad_csv", type=str, help="音频 VAD 标注 CSV 路径")
    parser.add_argument("--audio_epochs", type=int, help="音频 VAD 训练轮数")
    parser.add_argument("--audio_batch_size", type=int, help="音频批大小")
    # 评估报告输出目录（图像/音频 VAD）
    parser.add_argument("--report_dir", type=str, help="评估报告输出目录（默认自动根据模态选择）")

    # MOSEI 多模态融合参数
    parser.add_argument("--mosei_root", type=str, help="cmumosei 根目录", default="data/cmumosei")
    parser.add_argument("--mosei_epochs", type=int, help="MOSEI训练轮数")
    parser.add_argument("--mosei_batch_size", type=int, help="MOSEI训练批大小")
    parser.add_argument("--mosei_lr", type=float, help="MOSEI学习率")
    parser.add_argument("--mosei_hidden_dim", type=int, help="MOSEI融合隐藏维度", default=256)

    # MOSEI 预测参数（基于 pkl 的索引）
    parser.add_argument("--mosei_split", type=str, default="test", help="MOSEI split: train/valid/test")
    parser.add_argument("--mosei_index", type=int, help="要预测的样本索引（0-based）")
    parser.add_argument("--mosei_interactive", action="store_true", help="进入交互预测（输入 index，输入 exit 退出）")
    parser.add_argument("--mosei_pkl_senti", type=str, help="mosei_senti_data.pkl 路径", default=str(Path("data") / "cmumosei" / "mosei_senti_data.pkl"))
    parser.add_argument("--mosei_pkl_emotion", type=str, help="mosei_emotion_aligned_60.pkl 路径", default=str(Path("data") / "cmumosei" / "mosei_emotion_aligned_60.pkl"))

    # MOSEI 导出与评估参数
    parser.add_argument("--mosei_out_csv", type=str, default=str(Path("checkpoints") / "mosei_predictions.csv"))
    parser.add_argument("--mosei_export_limit", type=int, help="导出最多多少条样本（None=全量）")

    # 视频多模态融合预测参数
    parser.add_argument("--video_path", type=str, help="输入视频路径（多模态融合预测）")
    parser.add_argument("--video_fps", type=float, default=1.0, help="抽帧频率（fps）")
    parser.add_argument("--audio_chunk_seconds", type=float, default=6.0, help="音频分块时长（秒）")
    parser.add_argument("--text_model_path", type=str, help="文本VAD模型checkpoint路径",
                        default=str(Path("checkpoints") / "best_model.pt"))
    parser.add_argument("--image_model_path", type=str, help="图像VAD模型checkpoint路径",
                        default=str(Path("checkpoints") / "best_image_vad_model.pt"))
    parser.add_argument("--audio_model_path", type=str, help="音频VAD模型checkpoint路径",
                        default=str(Path("checkpoints") / "best_audio_vad_model.pt"))
    parser.add_argument("--subtitle_srt_path", type=str, help="字幕 srt 路径（可选；没有则会尝试从视频中抽取）")
    parser.add_argument("--text_weight", type=float, default=1.0, help="文本融合权重")
    parser.add_argument("--audio_weight", type=float, default=1.0, help="音频融合权重")
    parser.add_argument("--image_weight", type=float, default=1.0, help="图像融合权重")
    parser.add_argument("--ffmpeg_path", type=str, default="ffmpeg", help="ffmpeg.exe 完整路径（可选；默认从 PATH 查找）")
    
    # 训练参数
    parser.add_argument("--epochs", type=int, help="训练轮数")
    parser.add_argument("--batch_size", type=int, help="批处理大小")
    parser.add_argument("--learning_rate", type=float, help="学习率")
    
    # 情感维度参数
    parser.add_argument("--emotion_dimension", action="store_true", help="启用情感维度模型")
    parser.add_argument("--data_file", type=str, help="情感数据文件路径")
    parser.add_argument("--emotion_dimensions", nargs="+", default=["valence", "arousal", "dominance"], 
                       help="情感维度列表")
    parser.add_argument("--emobank", action="store_true", help="使用EmoBank解析流程 (自动映射列并标准化)")
    
    # 评估参数
    parser.add_argument("--model_path", type=str, help="模型文件路径")
    parser.add_argument("--analyze_errors", action="store_true", help="分析错误样本")
    
    # 预测参数
    parser.add_argument("--text", type=str, help="待预测的文本")
    parser.add_argument("--file", type=str, help="待预测的文件路径")
    parser.add_argument("--text_column", type=str, default="text", help="文本列名")
    parser.add_argument("--output", type=str, help="输出文件路径")
    parser.add_argument("--interactive", action="store_true", help="交互式预测")
    parser.add_argument("--image_model", action="store_true", help="使用图像模型进行预测（需要配合--model_path指定图像VAD模型）")
    parser.add_argument("--image_path", type=str, help="待预测的图像文件路径")
    parser.add_argument("--audio_model", action="store_true", help="使用音频模型进行预测（需要配合--model_path指定音频VAD模型）")
    parser.add_argument("--audio_path", type=str, help="待预测的音频文件路径")
    
    # 仅评估选项（用于跳过训练，直接评估与生成报告）
    parser.add_argument("--evaluate_only", action="store_true", help="仅评估与生成报告，跳过训练阶段")
    
    args = parser.parse_args()
    
    # 加载配置
    config = Config(args.config)
    
    # 覆盖命令行参数
    if args.epochs:
        config.config["training"]["num_epochs"] = args.epochs
    if args.batch_size:
        config.config["data"]["batch_size"] = args.batch_size
    if args.learning_rate:
        config.config["training"]["learning_rate"] = args.learning_rate
    
    # 覆盖情感维度参数
    if args.emotion_dimension:
        config.config["emotion_dimension"]["enabled"] = True
    if args.emotion_dimensions:
        config.config["emotion_dimension"]["emotion_dimensions"] = args.emotion_dimensions
    
    # 根据模式执行相应命令
    try:
        if args.mode == "train":
            train_command(args, config)
        elif args.mode == "evaluate":
            evaluate_command(args, config)
        elif args.mode == "predict":
            predict_command(args, config)
        elif args.mode == "train_image":
            # 图像训练入口
            from image_fer2013_dataset import create_fer_dataloaders
            from image_fer2013_model import ResNet18RGB
            from image_fer2013_trainer import ImageTrainer

            csv_path = args.fer_csv or str(Path("data") / "fer2013" / "fer2013.csv")
            if not Path(csv_path).exists():
                raise FileNotFoundError(f"未找到 fer2013.csv: {csv_path}")
            batch_size = args.image_batch_size or config.get("data.batch_size", 64)
            num_workers = config.get("system.num_workers", 4)
            train_loader, val_loader, test_loader = create_fer_dataloaders(
                csv_path,
                batch_size=batch_size,
                num_workers=num_workers,
                pin_memory=True,
                val_csv_path=args.fer_val_csv,
                test_csv_path=args.fer_test_csv,
            )
            device = setup_device(config)
            model = ResNet18RGB(num_classes=7, pretrained=True)
            trainer = ImageTrainer(
                model=model,
                train_loader=train_loader,
                val_loader=val_loader,
                test_loader=test_loader,
                device=device,
                learning_rate=config.get("training.learning_rate", 1e-3),
                weight_decay=config.get("training.weight_decay", 1e-4),
                save_dir=config.get("system.save_dir", "checkpoints"),
            )
            num_epochs = args.image_epochs or config.get("training.num_epochs", 10)
            trainer.train(num_epochs=num_epochs)
            trainer.evaluate()
        elif args.mode == "train_emotion_image":
            # 图像->VAD 训练入口
            from image_fer2013_vad import create_vad_dataloaders, VADRegressor, VADTrainer
            csv_path = args.fer_vad_csv or str(Path("data") / "fer2013" / "fer2013.csv")
            if not Path(csv_path).exists():
                raise FileNotFoundError(f"未找到 fer2013.csv: {csv_path}")
            batch_size = args.image_batch_size or config.get("data.batch_size", 64)
            num_workers = config.get("system.num_workers", 4)
            train_loader, val_loader, test_loader = create_vad_dataloaders(
                csv_path,
                batch_size=batch_size,
                num_workers=num_workers,
                pin_memory=True,
                val_csv_path=args.fer_vad_val_csv,
                test_csv_path=args.fer_vad_test_csv,
            )
            device = setup_device(config)
            model = VADRegressor(pretrained=True)
            trainer = VADTrainer(
                model=model,
                train_loader=train_loader,
                val_loader=val_loader,
                test_loader=test_loader,
                device=device,
                learning_rate=config.get("training.learning_rate", 1e-3),
                weight_decay=config.get("training.weight_decay", 1e-4),
                save_dir=config.get("system.save_dir", "checkpoints"),
            )
            num_epochs = args.image_vad_epochs or config.get("training.num_epochs", 20)
            trainer.train(num_epochs=num_epochs)
            trainer.evaluate()
        elif args.mode == "train_audio":
            from audio_vad_dataset import create_audio_vad_dataloaders
            from audio_vad_model import AudioVADRegressor
            from audio_vad_trainer import AudioVADTrainer

            csv_path = args.audio_vad_csv or str(Path("data") / "voice" / "audio_vad.csv")
            if not Path(csv_path).exists():
                raise FileNotFoundError(f"未找到音频 VAD 标注 CSV: {csv_path}")
            batch_size = args.audio_batch_size or config.get("data.batch_size", 16)
            num_workers = config.get("system.num_workers", 4)
            train_loader, val_loader, test_loader = create_audio_vad_dataloaders(
                csv_path,
                batch_size=batch_size,
                num_workers=num_workers,
            )
            device = setup_device(config)
            model = AudioVADRegressor()
            trainer = AudioVADTrainer(
                model=model,
                train_loader=train_loader,
                val_loader=val_loader,
                test_loader=test_loader,
                device=device,
                learning_rate=config.get("training.learning_rate", 1e-4),
                weight_decay=config.get("training.weight_decay", 1e-4),
                save_dir=config.get("system.save_dir", "checkpoints"),
                log_dir=str(Path(config.get("system.log_dir", "logs")) / "audio_vad"),
            )
            num_epochs = args.audio_epochs or config.get("training.num_epochs", 10)
            trainer.train(num_epochs=num_epochs)
            trainer.evaluate()
        elif args.mode == "evaluate_image_vad":
            # 图像 VAD：加载 checkpoint，跑 test，输出落盘评估报告
            from image_fer2013_vad import create_vad_dataloaders, VADRegressor
            from vad_regression_report import calculate_vad_metrics, run_regression_inference, save_vad_report

            csv_path = args.fer_vad_csv or str(Path("data") / "fer2013" / "fer2013.csv")
            if not Path(csv_path).exists():
                raise FileNotFoundError(f"未找到 fer2013.csv: {csv_path}")

            batch_size = args.image_batch_size or config.get("data.batch_size", 64)
            num_workers = config.get("system.num_workers", 4)
            _, _, test_loader = create_vad_dataloaders(
                csv_path,
                batch_size=batch_size,
                num_workers=num_workers,
                pin_memory=True,
                val_csv_path=args.fer_vad_val_csv,
                test_csv_path=args.fer_vad_test_csv,
            )

            device = setup_device(config)
            ckpt_path = args.model_path or str(Path(config.get("system.save_dir", "checkpoints")) / "best_image_vad_model.pt")
            if not Path(ckpt_path).exists():
                raise FileNotFoundError(f"未找到图像 VAD checkpoint: {ckpt_path}")

            model = VADRegressor(pretrained=False)
            ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
            state = ckpt.get("model_state_dict", ckpt)
            model.load_state_dict(state)

            preds, labels = run_regression_inference(
                model=model,
                data_loader=test_loader,
                device=device,
                batch_to_xy=lambda b: (b[0], b[1]),
            )
            metrics = calculate_vad_metrics(preds, labels, dims=["valence", "arousal", "dominance"])
            out_dir = args.report_dir or "evaluation_report_image"
            save_vad_report(
                report_dir=out_dir,
                metrics=metrics,
                extra={"checkpoint": ckpt_path, "csv_path": csv_path, "modality": "image"},
            )
            print(f"图像 VAD 评估报告已保存到: {out_dir}")
        elif args.mode == "evaluate_audio_vad":
            # 音频 VAD：加载 checkpoint，跑 test，输出落盘评估报告
            from audio_vad_dataset import create_audio_vad_dataloaders
            from audio_vad_model import AudioVADRegressor
            from vad_regression_report import calculate_vad_metrics, run_regression_inference, save_vad_report

            csv_path = args.audio_vad_csv or str(Path("data") / "voice" / "audio_vad.csv")
            if not Path(csv_path).exists():
                raise FileNotFoundError(f"未找到音频 VAD 标注 CSV: {csv_path}")

            batch_size = args.audio_batch_size or config.get("data.batch_size", 16)
            num_workers = config.get("system.num_workers", 4)
            _, _, test_loader = create_audio_vad_dataloaders(
                csv_path,
                batch_size=batch_size,
                num_workers=num_workers,
            )

            device = setup_device(config)
            ckpt_path = args.model_path or str(Path(config.get("system.save_dir", "checkpoints")) / "best_audio_vad_model.pt")
            if not Path(ckpt_path).exists():
                raise FileNotFoundError(f"未找到音频 VAD checkpoint: {ckpt_path}")

            model = AudioVADRegressor()
            ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
            state = ckpt.get("model_state_dict", ckpt)
            model.load_state_dict(state)

            preds, labels = run_regression_inference(
                model=model,
                data_loader=test_loader,
                device=device,
                batch_to_xy=lambda b: (b["waveform"], b["vad"]),
            )
            metrics = calculate_vad_metrics(preds, labels, dims=["valence", "arousal", "dominance"])
            out_dir = args.report_dir or "evaluation_report_audio"
            save_vad_report(
                report_dir=out_dir,
                metrics=metrics,
                extra={"checkpoint": ckpt_path, "csv_path": csv_path, "modality": "audio"},
            )
            print(f"音频 VAD 评估报告已保存到: {out_dir}")
        elif args.mode == "train_mosei_senti":
            from torch.utils.data import DataLoader
            from mosei_multimodal_dataset import MoseiMultimodalDataset
            from mosei_fusion_model import ConcatMLPFusionModel
            from mosei_fusion_trainer import MoseiFusionTrainer

            from pathlib import Path as _Path

            mosei_root = _Path(args.mosei_root)
            pkl_path = mosei_root / "mosei_senti_data.pkl"
            if not pkl_path.exists():
                raise FileNotFoundError(f"未找到: {pkl_path}")

            batch_size = args.mosei_batch_size or args.batch_size or config.get("data.batch_size", 16)
            num_epochs = args.mosei_epochs or config.get("training.num_epochs", 10)
            lr = args.mosei_lr or args.learning_rate or 2e-4

            train_ds = MoseiMultimodalDataset(str(pkl_path), task="senti", split="train")
            val_ds = MoseiMultimodalDataset(str(pkl_path), task="senti", split="valid")
            test_ds = MoseiMultimodalDataset(str(pkl_path), task="senti", split="test")

            # 注意：pkl 特征很大，强烈建议 num_workers=0，避免多进程复制内存
            train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=0, pin_memory=True)
            val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=0, pin_memory=True)
            test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=0, pin_memory=True)

            device = setup_device(config)
            model = ConcatMLPFusionModel(task="senti", hidden_dim=args.mosei_hidden_dim, dropout=0.2)
            trainer = MoseiFusionTrainer(
                task="senti",
                model=model,
                train_loader=train_loader,
                val_loader=val_loader,
                test_loader=test_loader,
                device=device,
                learning_rate=lr,
                weight_decay=config.get("training.weight_decay", 1e-4),
                save_dir=config.get("system.save_dir", "checkpoints"),
                log_dir=str(Path(config.get("system.log_dir", "logs")) / "mosei_senti"),
            )
            trainer.train(num_epochs=num_epochs)
            trainer.evaluate()
        elif args.mode == "train_mosei_emotion":
            from torch.utils.data import DataLoader
            from mosei_multimodal_dataset import MoseiMultimodalDataset
            from mosei_fusion_model import ConcatMLPFusionModel
            from mosei_fusion_trainer import MoseiFusionTrainer

            mosei_root = Path(args.mosei_root)
            pkl_path = mosei_root / "mosei_emotion_aligned_60.pkl"
            if not pkl_path.exists():
                raise FileNotFoundError(f"未找到: {pkl_path}")

            batch_size = args.mosei_batch_size or args.batch_size or config.get("data.batch_size", 16)
            num_epochs = args.mosei_epochs or config.get("training.num_epochs", 10)
            lr = args.mosei_lr or args.learning_rate or 2e-4

            train_ds = MoseiMultimodalDataset(str(pkl_path), task="emotion", split="train")
            val_ds = MoseiMultimodalDataset(str(pkl_path), task="emotion", split="valid")
            test_ds = MoseiMultimodalDataset(str(pkl_path), task="emotion", split="test")

            train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=0, pin_memory=True)
            val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=0, pin_memory=True)
            test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=0, pin_memory=True)

            device = setup_device(config)
            model = ConcatMLPFusionModel(task="emotion", hidden_dim=args.mosei_hidden_dim, dropout=0.2)
            trainer = MoseiFusionTrainer(
                task="emotion",
                model=model,
                train_loader=train_loader,
                val_loader=val_loader,
                test_loader=test_loader,
                device=device,
                learning_rate=lr,
                weight_decay=config.get("training.weight_decay", 1e-4),
                save_dir=config.get("system.save_dir", "checkpoints"),
                log_dir=str(Path(config.get("system.log_dir", "logs")) / "mosei_emotion"),
            )
            trainer.train(num_epochs=num_epochs)
            # 训练后跑一次测试，生成指标文件
            trainer.evaluate()
        elif args.mode == "evaluate_mosei_senti":
            from torch.utils.data import DataLoader
            from mosei_multimodal_dataset import MoseiMultimodalDataset
            from mosei_fusion_model import ConcatMLPFusionModel
            from mosei_fusion_trainer import MoseiFusionTrainer

            device = setup_device(config)
            if not args.model_path:
                raise ValueError("evaluate_mosei_senti 需要 --model_path 指定checkpoint")

            mosei_root = Path(args.mosei_root)
            pkl_path = Path(args.mosei_pkl_senti) if hasattr(args, "mosei_pkl_senti") and args.mosei_pkl_senti else (mosei_root / "mosei_senti_data.pkl")
            # 这里仍然按 trainer.evaluate() 的逻辑评估 test split
            test_ds = MoseiMultimodalDataset(str(pkl_path), task="senti", split=args.mosei_split)
            # 给 trainer 构造需要的 loader：train/val 用同一个 test split 占位即可
            train_ds = test_ds
            val_ds = test_ds
            train_loader = DataLoader(train_ds, batch_size=args.mosei_batch_size or 16, shuffle=False, num_workers=0, pin_memory=True)
            val_loader = DataLoader(val_ds, batch_size=args.mosei_batch_size or 16, shuffle=False, num_workers=0, pin_memory=True)
            test_loader = DataLoader(test_ds, batch_size=args.mosei_batch_size or 16, shuffle=False, num_workers=0, pin_memory=True)

            model = ConcatMLPFusionModel(task="senti", hidden_dim=args.mosei_hidden_dim, dropout=0.2)
            trainer = MoseiFusionTrainer(
                task="senti",
                model=model,
                train_loader=train_loader,
                val_loader=val_loader,
                test_loader=test_loader,
                device=device,
                learning_rate=args.mosei_lr or 2e-4,
                weight_decay=config.get("training.weight_decay", 1e-4),
                save_dir=config.get("system.save_dir", "checkpoints"),
                log_dir=str(Path(config.get("system.log_dir", "logs")) / "mosei_eval_senti"),
            )

            ckpt = torch.load(args.model_path, map_location=device, weights_only=False)
            state = ckpt.get("model_state_dict", ckpt)
            trainer.model.load_state_dict(state, strict=True)
            trainer.evaluate()
        elif args.mode == "evaluate_mosei_emotion":
            from torch.utils.data import DataLoader
            from mosei_multimodal_dataset import MoseiMultimodalDataset
            from mosei_fusion_model import ConcatMLPFusionModel
            from mosei_fusion_trainer import MoseiFusionTrainer

            device = setup_device(config)
            if not args.model_path:
                raise ValueError("evaluate_mosei_emotion 需要 --model_path 指定checkpoint")

            pkl_path = Path(args.mosei_pkl_emotion) if hasattr(args, "mosei_pkl_emotion") and args.mosei_pkl_emotion else (Path(args.mosei_root) / "mosei_emotion_aligned_60.pkl")
            test_ds = MoseiMultimodalDataset(str(pkl_path), task="emotion", split=args.mosei_split)
            train_ds = test_ds
            val_ds = test_ds
            train_loader = DataLoader(train_ds, batch_size=args.mosei_batch_size or 16, shuffle=False, num_workers=0, pin_memory=True)
            val_loader = DataLoader(val_ds, batch_size=args.mosei_batch_size or 16, shuffle=False, num_workers=0, pin_memory=True)
            test_loader = DataLoader(test_ds, batch_size=args.mosei_batch_size or 16, shuffle=False, num_workers=0, pin_memory=True)

            model = ConcatMLPFusionModel(task="emotion", hidden_dim=args.mosei_hidden_dim, dropout=0.2)
            trainer = MoseiFusionTrainer(
                task="emotion",
                model=model,
                train_loader=train_loader,
                val_loader=val_loader,
                test_loader=test_loader,
                device=device,
                learning_rate=args.mosei_lr or 2e-4,
                weight_decay=config.get("training.weight_decay", 1e-4),
                save_dir=config.get("system.save_dir", "checkpoints"),
                log_dir=str(Path(config.get("system.log_dir", "logs")) / "mosei_eval_emotion"),
            )

            ckpt = torch.load(args.model_path, map_location=device, weights_only=False)
            state = ckpt.get("model_state_dict", ckpt)
            trainer.model.load_state_dict(state, strict=True)
            trainer.evaluate()
        elif args.mode == "export_mosei_senti":
            from mosei_multimodal_infer import export_split_predictions

            if not args.model_path:
                raise ValueError("export_mosei_senti 需要 --model_path 指定checkpoint")

            device = setup_device(config)
            out_csv = args.mosei_out_csv
            export_split_predictions(
                task="senti",
                checkpoint_path=args.model_path,
                dataset_pkl_path=args.mosei_pkl_senti,
                split=args.mosei_split,
                device=device,
                out_csv=out_csv,
                hidden_dim=args.mosei_hidden_dim,
                dropout=0.2,
                batch_size=args.mosei_batch_size or 16,
                limit=args.mosei_export_limit,
            )
            print(f"已导出: {out_csv}")
        elif args.mode == "export_mosei_emotion":
            from mosei_multimodal_infer import export_split_predictions

            if not args.model_path:
                raise ValueError("export_mosei_emotion 需要 --model_path 指定checkpoint")

            device = setup_device(config)
            out_csv = args.mosei_out_csv
            export_split_predictions(
                task="emotion",
                checkpoint_path=args.model_path,
                dataset_pkl_path=args.mosei_pkl_emotion,
                split=args.mosei_split,
                device=device,
                out_csv=out_csv,
                hidden_dim=args.mosei_hidden_dim,
                dropout=0.2,
                batch_size=args.mosei_batch_size or 16,
                limit=args.mosei_export_limit,
            )
            print(f"已导出: {out_csv}")
        elif args.mode == "predict_video_fusion":
            if not args.video_path:
                raise ValueError("predict_video_fusion 需要 --video_path")
            from video_fusion_predictor import predict_vad_from_video, format_vad_result

            device = setup_device(config)
            results = predict_vad_from_video(
                video_path=args.video_path,
                device=device,
                text_model_path=args.text_model_path,
                image_model_path=args.image_model_path,
                audio_model_path=args.audio_model_path,
                ffmpeg_path=args.ffmpeg_path,
                video_fps=args.video_fps,
                max_frames=64,
                audio_chunk_seconds=args.audio_chunk_seconds,
                text_weight=args.text_weight,
                audio_weight=args.audio_weight,
                image_weight=args.image_weight,
                subtitle_srt_path=args.subtitle_srt_path,
            )

            print("\n=== 单模态结果 ===")
            print("[Text]\n" + format_vad_result(results.get("text", {})))
            print("\n[Audio]\n" + format_vad_result(results.get("audio", {})))
            print("\n[Image]\n" + format_vad_result(results.get("image", {})))

            print("\n=== 融合结果 ===")
            print(format_vad_result(results.get("fusion", {})))
        elif args.mode == "predict_mosei_senti":
            from mosei_multimodal_infer import predict_single_by_index

            device = setup_device(config)
            if not args.model_path:
                raise ValueError("predict_mosei_senti 需要 --model_path 指定已训练 checkpoint")

            if args.mosei_interactive:
                print("进入 MOSEI sentiment 交互预测（输入 index，0-based；exit退出）")
                while True:
                    try:
                        s = input("index> ").strip()
                    except (EOFError, KeyboardInterrupt):
                        print("\n已退出。")
                        break
                    if s.lower() in {"exit", "quit", ":q"}:
                        print("已退出。")
                        break
                    if not s:
                        continue
                    idx = int(s)
                    res = predict_single_by_index(
                        task="senti",
                        checkpoint_path=args.model_path,
                        dataset_pkl_path=args.mosei_pkl_senti,
                        split=args.mosei_split,
                        index=idx,
                        device=device,
                        hidden_dim=args.mosei_hidden_dim,
                        dropout=0.2,
                    )
                    print(f"split={res['split']}, index={res['index']}")
                    print(f"pred_score: {res['score']:.4f}")
                    print(f"interpretation: {res['interpretation']}")
                    print(f"gt_score: {res['gt_label']:.4f}")
            else:
                if args.mosei_index is None:
                    raise ValueError("predict_mosei_senti 需要 --mosei_index（或使用 --mosei_interactive）")
                res = predict_single_by_index(
                    task="senti",
                    checkpoint_path=args.model_path,
                    dataset_pkl_path=args.mosei_pkl_senti,
                    split=args.mosei_split,
                    index=args.mosei_index,
                    device=device,
                    hidden_dim=args.mosei_hidden_dim,
                    dropout=0.2,
                )
                print(f"split={res['split']}, index={res['index']}")
                print(f"pred_score: {res['score']:.4f}")
                print(f"interpretation: {res['interpretation']}")
                print(f"gt_score: {res['gt_label']:.4f}")
        elif args.mode == "predict_mosei_emotion":
            from mosei_multimodal_infer import predict_single_by_index

            device = setup_device(config)
            if not args.model_path:
                raise ValueError("predict_mosei_emotion 需要 --model_path 指定已训练 checkpoint")

            if args.mosei_interactive:
                print("进入 MOSEI emotion 交互预测（输入 index，0-based；exit退出）")
                while True:
                    try:
                        s = input("index> ").strip()
                    except (EOFError, KeyboardInterrupt):
                        print("\n已退出。")
                        break
                    if s.lower() in {"exit", "quit", ":q"}:
                        print("已退出。")
                        break
                    if not s:
                        continue
                    idx = int(s)
                    res = predict_single_by_index(
                        task="emotion",
                        checkpoint_path=args.model_path,
                        dataset_pkl_path=args.mosei_pkl_emotion,
                        split=args.mosei_split,
                        index=idx,
                        device=device,
                        hidden_dim=args.mosei_hidden_dim,
                        dropout=0.2,
                    )
                    print(f"split={res['split']}, index={res['index']}")
                    print("pred_probs:")
                    for k, v in res["probs"].items():
                        print(f"  {k}: {v:.4f}")
                    print(f"interpretation: {res['interpretation']}")
                    print("gt_labels:")
                    for k, v in res["gt_labels"].items():
                        print(f"  {k}: {int(v)}")
                    print("-" * 50)
            else:
                if args.mosei_index is None:
                    raise ValueError("predict_mosei_emotion 需要 --mosei_index（或使用 --mosei_interactive）")
                res = predict_single_by_index(
                    task="emotion",
                    checkpoint_path=args.model_path,
                    dataset_pkl_path=args.mosei_pkl_emotion,
                    split=args.mosei_split,
                    index=args.mosei_index,
                    device=device,
                    hidden_dim=args.mosei_hidden_dim,
                    dropout=0.2,
                )
                print(f"split={res['split']}, index={res['index']}")
                print("pred_probs:")
                for k, v in res["probs"].items():
                    print(f"  {k}: {v:.4f}")
                print(f"interpretation: {res['interpretation']}")
                print("gt_labels:")
                for k, v in res["gt_labels"].items():
                    print(f"  {k}: {int(v)}")
        else:
            raise ValueError(f"未实现的模式: {args.mode}")
    except Exception as e:
        logger.error(f"执行出错: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
