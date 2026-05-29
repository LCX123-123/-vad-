# 情感维度模型使用指南

## 概述

情感维度模型是一个基于Transformer的多维度情感分析系统，支持预测文本在多个情感维度上的数值，如Valence（效价）、Arousal（唤醒度）、Dominance（支配性）等。

## 主要特性

- **多维度情感预测**: 支持VAD（Valence-Arousal-Dominance）模型和自定义情感维度
- **灵活的模型架构**: 支持注意力池化和传统池化方式
- **完整的训练流程**: 包含训练、验证、评估和可视化功能
- **丰富的评估指标**: 提供MSE、MAE、R²等多种回归指标
- **可视化分析**: 自动生成预测vs真实值图、残差图、相关性热力图等

## 文件结构

```
├── emotion_dimension_model.py      # 情感维度模型定义
├── emotion_data_loader.py          # 情感数据加载器
├── emotion_trainer.py              # 情感维度训练器
├── emotion_evaluator.py            # 情感维度评估器
├── emotion_dimension_example.py    # 使用示例
└── EMOTION_DIMENSION_README.md     # 本文档
```

## 快速开始

### 1. 基本使用

```python
from emotion_dimension_model import create_emotion_dimension_model
from emotion_data_loader import load_and_prepare_emotion_data
from emotion_trainer import EmotionDimensionTrainer

# 创建模型
model = create_emotion_dimension_model(
    model_name="bert-base-uncased",
    emotion_dimensions=["valence", "arousal", "dominance"],
    model_type="dimension"
)

# 加载数据
train_loader, val_loader, test_loader, processor = load_and_prepare_emotion_data(
    file_path="data/emotion_data.csv",
    emotion_columns=["valence", "arousal", "dominance"]
)

# 训练模型
trainer = EmotionDimensionTrainer(model, train_loader, val_loader, test_loader)
history = trainer.train(num_epochs=10)
```

### 2. 使用命令行

```bash
# 训练情感维度模型
python main.py --mode train --emotion_dimension --data_file data/emotion_data.csv

# 评估模型
python main.py --mode evaluate --model_path checkpoints/best_model.pt

# 预测文本
python main.py --mode predict --model_path checkpoints/best_model.pt --text "This movie is great!"
```

### 3. 运行示例

```bash
python emotion_dimension_example.py
```

## 配置说明

### 情感维度配置

在`config.json`中添加以下配置：

```json
{
  "emotion_dimension": {
    "enabled": true,
    "model_type": "dimension",
    "emotion_dimensions": ["valence", "arousal", "dominance"],
    "dimension_ranges": {
      "valence": [-1.0, 1.0],
      "arousal": [-1.0, 1.0],
      "dominance": [-1.0, 1.0]
    },
    "hidden_dims": [768, 256, 64],
    "use_attention_pooling": true,
    "normalize_labels": true
  }
}
```

### 参数说明

- `enabled`: 是否启用情感维度模型
- `model_type`: 模型类型（"dimension" 或 "multi_task"）
- `emotion_dimensions`: 情感维度列表
- `dimension_ranges`: 各维度的取值范围
- `hidden_dims`: 隐藏层维度
- `use_attention_pooling`: 是否使用注意力池化
- `normalize_labels`: 是否标准化标签

## 数据格式

### 输入数据格式

CSV文件，包含以下列：
- `text`: 文本内容
- `valence`: 效价值（-1到1）
- `arousal`: 唤醒度值（-1到1）
- `dominance`: 支配性值（-1到1）

示例：
```csv
text,valence,arousal,dominance
"This movie is great!",0.8,0.6,0.7
"I hate this film.",-0.9,0.4,-0.3
"The weather is okay.",0.1,0.2,0.0
```

### 创建示例数据

```python
from emotion_data_loader import create_sample_emotion_data

# 创建示例数据
sample_df = create_sample_emotion_data(
    num_samples=1000,
    emotion_dimensions=["valence", "arousal", "dominance"],
    save_path="data/sample_emotion_data.csv"
)
```

## 模型架构

### 情感维度模型

```python
class EmotionDimensionModel(nn.Module):
    def __init__(self, model_name, emotion_dimensions, ...):
        # BERT编码器
        self.bert = AutoModel.from_pretrained(model_name)
        
        # 注意力池化（可选）
        self.attention_pooling = nn.MultiheadAttention(...)
        
        # 多层回归器
        self.regressors = nn.ModuleDict()
        for dimension in emotion_dimensions:
            self.regressors[dimension] = nn.Linear(hidden_dim, 1)
```

### 多任务模型

```python
class MultiTaskEmotionModel(nn.Module):
    def __init__(self, model_name, num_classes, emotion_dimensions, ...):
        # 共享特征提取器
        self.shared_layers = nn.Sequential(...)
        
        # 分类头
        self.classifier = nn.Linear(hidden_dim, num_classes)
        
        # 回归头
        self.regressors = nn.ModuleDict()
        for dimension in emotion_dimensions:
            self.regressors[dimension] = nn.Linear(hidden_dim, 1)
```

## 训练和评估

### 训练器

```python
trainer = EmotionDimensionTrainer(
    model=model,
    train_loader=train_loader,
    val_loader=val_loader,
    test_loader=test_loader,
    device=device,
    learning_rate=2e-5,
    weight_decay=0.01
)

# 开始训练
history = trainer.train(num_epochs=10)

# 评估模型
test_metrics = trainer.evaluate()
```

### 评估指标

- **MSE**: 均方误差
- **RMSE**: 均方根误差
- **MAE**: 平均绝对误差
- **R²**: 决定系数
- **MAPE**: 平均绝对百分比误差
- **SMAPE**: 对称平均绝对百分比误差

### 可视化

评估器会自动生成以下图表：
- 预测值vs真实值散点图
- 残差图
- 相关性热力图
- 误差分布图

## 预测和推理

### 单文本预测

```python
# 预测情感维度
predictions = model.predict_dimensions(["This movie is great!"])

# 获取情感轮廓
emotion_profile = model.get_emotion_profile("This movie is great!")

# 解释情感
interpretation = model.interpret_emotion(emotion_profile)
```

### 批量预测

```python
texts = ["Text 1", "Text 2", "Text 3"]
predictions = model.predict_dimensions(texts, batch_size=32)
```

## 预定义配置

### VAD模型

```python
from emotion_dimension_model import EMOTION_DIMENSION_CONFIGS

# 使用VAD配置
vad_config = EMOTION_DIMENSION_CONFIGS["vad"]
model = create_emotion_dimension_model(
    emotion_dimensions=vad_config["emotion_dimensions"],
    dimension_ranges=vad_config["dimension_ranges"]
)
```

### 可用配置

- `vad`: Valence-Arousal-Dominance模型
- `va`: Valence-Arousal模型
- `ekman`: Ekman六种基本情感模型

## 高级功能

### 自定义情感维度

```python
# 定义自定义情感维度
custom_dimensions = ["joy", "sadness", "anger", "fear"]
custom_ranges = {
    "joy": (0.0, 1.0),
    "sadness": (0.0, 1.0),
    "anger": (0.0, 1.0),
    "fear": (0.0, 1.0)
}

model = create_emotion_dimension_model(
    emotion_dimensions=custom_dimensions,
    dimension_ranges=custom_ranges
)
```

### 多任务学习

```python
# 创建多任务模型
model = create_emotion_dimension_model(
    model_type="multi_task",
    num_classes=2,  # 分类任务类别数
    emotion_dimensions=["valence", "arousal", "dominance"]
)

# 使用多任务训练器
trainer = MultiTaskEmotionTrainer(
    model=model,
    train_loader=train_loader,
    val_loader=val_loader,
    test_loader=test_loader,
    task_weights={"classification": 1.0, "regression": 1.0}
)
```

## 故障排除

### 常见问题

1. **内存不足**: 减少批处理大小或使用梯度累积
2. **训练不收敛**: 调整学习率或使用学习率调度器
3. **过拟合**: 增加dropout率或使用正则化

### 性能优化

1. **使用GPU**: 确保CUDA可用
2. **批处理**: 适当调整批处理大小
3. **模型压缩**: 使用DistilBERT等轻量级模型

## 扩展和定制

### 添加新的情感维度

1. 在配置中定义新维度
2. 更新数据格式
3. 重新训练模型

### 集成到现有系统

```python
# 加载训练好的模型
model = create_emotion_dimension_model(...)
model.load_state_dict(torch.load("checkpoints/best_model.pt"))

# 在应用中使用
def analyze_emotion(text):
    emotion_profile = model.get_emotion_profile(text)
    return emotion_profile
```

## 参考文献

1. Russell, J. A. (1980). A circumplex model of affect.
2. Bradley, M. M., & Lang, P. J. (1994). Measuring emotion: the self-assessment manikin and the semantic differential.
3. Devlin, J., et al. (2018). BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding.

## 许可证

本项目采用MIT许可证。

