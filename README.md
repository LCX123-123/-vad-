# 基于Transformer的NLP分类任务

这是一个完整的基于Transformer架构的NLP文本分类项目，支持情感分析等文本分类任务。

## 功能特性

- 🚀 **多种预训练模型支持**: BERT、RoBERTa、DistilBERT等
- 📊 **完整的数据处理流程**: 数据加载、预处理、分割
- 🎯 **灵活的模型架构**: 简单分类器和多层分类器
- 📈 **训练监控**: TensorBoard可视化、早停机制
- 🔍 **模型评估**: 详细的性能指标和错误分析
- 💬 **交互式预测**: 支持单文本、批量文件和交互式预测
- ⚙️ **配置管理**: JSON配置文件，易于调整参数

## 项目结构

```
Natural Language/
├── data/                          # 数据目录
│   ├── imdb_en_clean.csv         # 英文IMDB数据
│   └── imdb_pt_clean.csv         # 葡萄牙语IMDB数据
├── checkpoints/                   # 模型检查点目录
├── logs/                         # 日志目录
├── transformer_model.py          # Transformer模型定义
├── data_loader.py                # 数据加载和预处理
├── trainer.py                    # 训练和评估模块
├── inference.py                  # 推理和预测模块
├── main.py                       # 主程序入口
├── config.json                   # 配置文件
├── requirements.txt              # 依赖包列表
└── README.md                     # 项目说明
```

## 安装依赖

```bash
pip install -r requirements.txt
```

## 使用方法

### 1. 训练模型

```bash
# 使用默认配置训练
python main.py --mode train

# 使用自定义配置训练
python main.py --mode train --config config.json

# 指定训练参数
python main.py --mode train --epochs 5 --batch_size 32 --learning_rate 3e-5
```

### 2. 评估模型

```bash
# 评估训练好的模型
python main.py --mode evaluate --model_path checkpoints/best_model_epoch_5.pt

# 评估并分析错误样本
python main.py --mode evaluate --model_path checkpoints/best_model_epoch_5.pt --analyze_errors
```

### 3. 预测文本

```bash
# 单文本预测
python main.py --mode predict --model_path checkpoints/best_model_epoch_5.pt --text "This movie is amazing!"

# 批量文件预测
python main.py --mode predict --model_path checkpoints/best_model_epoch_5.pt --file test_data.csv --output predictions.csv

# 交互式预测
python main.py --mode predict --model_path checkpoints/best_model_epoch_5.pt --interactive
```

## 配置说明

配置文件 `config.json` 包含以下主要参数：

### 数据配置
- `data_dir`: 数据目录路径
- `language`: 语言类型 ("en" 或 "pt")
- `batch_size`: 批处理大小
- `max_length`: 最大序列长度

### 模型配置
- `model_name`: 预训练模型名称
- `model_type`: 模型类型 ("simple" 或 "multi_layer")
- `num_classes`: 分类类别数
- `dropout_rate`: Dropout比例

### 训练配置
- `num_epochs`: 训练轮数
- `learning_rate`: 学习率
- `weight_decay`: 权重衰减
- `early_stopping_patience`: 早停耐心值

## 支持的预训练模型

- `bert-base-uncased`: BERT基础模型（推荐）
- `bert-large-uncased`: BERT大型模型
- `roberta-base`: RoBERTa基础模型
- `distilbert-base-uncased`: DistilBERT轻量模型

## 模型架构

### 简单分类器
- 基于预训练Transformer模型
- 使用[CLS] token进行分类
- 单层分类头

### 多层分类器
- 支持自定义隐藏层维度
- 多层全连接网络
- 更强的表达能力

## 训练监控

训练过程中会生成以下文件：
- `checkpoints/`: 模型检查点
- `logs/`: TensorBoard日志
- `training_history.json`: 训练历史记录
- `nlp_classification.log`: 训练日志

使用TensorBoard查看训练过程：
```bash
tensorboard --logdir checkpoints/logs
```

## 性能指标

模型评估包含以下指标：
- 准确率 (Accuracy)
- 精确率 (Precision)
- 召回率 (Recall)
- F1分数 (F1-Score)
- 混淆矩阵 (Confusion Matrix)

## 示例代码

### 快速开始

```python
from transformer_model import create_model
from data_loader import load_and_prepare_data
from trainer import train_model
from inference import SentimentPredictor

# 1. 创建模型
model = create_model(
    model_name="bert-base-uncased",
    num_classes=2,
    model_type="simple"
)

# 2. 加载数据
train_loader, val_loader, test_loader, label_encoder = load_and_prepare_data(
    data_dir="data",
    language="en",
    batch_size=16
)

# 3. 训练模型
trainer = train_model(
    model=model,
    train_loader=train_loader,
    val_loader=val_loader,
    test_loader=test_loader,
    num_epochs=5
)

# 4. 预测文本
predictor = SentimentPredictor("checkpoints/best_model.pt")
result = predictor.predict_single("This movie is fantastic!")
print(f"预测结果: {result['prediction']}")
```

## 注意事项

1. **GPU内存**: 如果遇到GPU内存不足，可以减小`batch_size`或使用`distilbert-base-uncased`等轻量模型
2. **数据格式**: 确保数据文件包含`text_clean`和`sentiment`列
3. **模型保存**: 训练过程中会自动保存最佳模型
4. **早停机制**: 验证损失不再下降时会自动停止训练

## 扩展功能

- 支持多语言情感分析
- 可扩展到其他文本分类任务
- 支持自定义数据预处理
- 支持模型集成和投票

## 许可证

MIT License

## 贡献

欢迎提交Issue和Pull Request！
