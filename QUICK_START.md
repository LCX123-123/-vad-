# 快速启动指南

## 🚨 网络连接问题解决方案

如果您遇到 "We couldn't connect to 'https://huggingface.co'" 错误，请按照以下步骤解决：

### 方法1: 自动修复（推荐）

```bash
python fix_network_issue.py
```

这个脚本会自动：
- 检查网络连接
- 安装必要依赖
- 下载或创建模型配置
- 提供解决方案

### 方法2: 手动下载模型

```bash
# 创建模型目录
mkdir -p models/bert-base-uncased
cd models/bert-base-uncased

# 使用git下载模型
git clone https://huggingface.co/bert-base-uncased .

# 或者使用Python下载
python -c "
from transformers import AutoTokenizer, AutoModel
tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')
model = AutoModel.from_pretrained('bert-base-uncased')
tokenizer.save_pretrained('models/bert-base-uncased')
model.save_pretrained('models/bert-base-uncased')
"
```

### 方法3: 使用镜像源

```bash
# 设置Hugging Face镜像
export HF_ENDPOINT=https://hf-mirror.com

# 或者使用环境变量
set HF_ENDPOINT=https://hf-mirror.com  # Windows
export HF_ENDPOINT=https://hf-mirror.com  # Linux/Mac
```

### 方法4: 离线测试

```bash
# 运行离线测试脚本
python test_offline.py
```

## 🚀 正常使用流程

### 1. 安装依赖

```bash
pip install -r requirements.txt
```

### 2. 训练模型

```bash
# 使用默认配置训练
python main.py --mode train

# 使用自定义参数
python main.py --mode train --epochs 5 --batch_size 32
```

### 3. 评估模型

```bash
python main.py --mode evaluate --model_path checkpoints/best_model.pt
```

### 4. 预测文本

```bash
# 单文本预测
python main.py --mode predict --model_path checkpoints/best_model.pt --text "This movie is great!"

# 交互式预测
python main.py --mode predict --model_path checkpoints/best_model.pt --interactive
```

## 🔧 常见问题解决

### 问题1: 网络连接超时

**解决方案:**
- 检查网络连接
- 使用VPN或代理
- 使用镜像源
- 手动下载模型

### 问题2: GPU内存不足

**解决方案:**
- 减小batch_size: `--batch_size 8`
- 使用轻量模型: 修改config.json中的model_name为"distilbert-base-uncased"
- 使用CPU: 修改config.json中的device为"cpu"

### 问题3: 数据文件不存在

**解决方案:**
- 确保data目录下有imdb_en_clean.csv或imdb_pt_clean.csv
- 运行数据预处理脚本: `python imdb_pt_preprocess.py`

### 问题4: 模型加载失败

**解决方案:**
- 检查模型文件是否完整
- 重新下载模型
- 使用离线模式

## 📊 性能优化建议

### 1. 模型选择

- **bert-base-uncased**: 平衡性能和速度（推荐）
- **distilbert-base-uncased**: 更快的训练和推理
- **bert-large-uncased**: 更好的性能，但需要更多资源

### 2. 训练参数

```json
{
  "training": {
    "num_epochs": 3,        // 减少训练轮数
    "batch_size": 16,       // 根据GPU内存调整
    "learning_rate": 2e-5,  // 标准学习率
    "early_stopping_patience": 3  // 早停耐心值
  }
}
```

### 3. 系统配置

```json
{
  "system": {
    "device": "cuda",       // 使用GPU
    "num_workers": 4,       // 数据加载进程数
    "save_dir": "checkpoints"
  }
}
```

## 🎯 快速测试

### 测试系统是否正常工作

```bash
# 运行示例脚本
python example.py

# 选择选项1进行快速演示
```

### 测试离线功能

```bash
# 运行离线测试
python test_offline.py

# 选择选项1进行离线测试
```

## 📝 日志和调试

### 查看训练日志

```bash
# 查看日志文件
tail -f nlp_classification.log

# 使用TensorBoard查看训练过程
tensorboard --logdir checkpoints/logs
```

### 调试模式

```bash
# 设置详细日志
export PYTHONPATH=.
python -u main.py --mode train 2>&1 | tee training.log
```

## 🆘 获取帮助

如果仍然遇到问题，请：

1. 检查日志文件 `nlp_classification.log`
2. 运行 `python fix_network_issue.py` 进行自动修复
3. 查看错误信息和堆栈跟踪
4. 确保所有依赖包已正确安装

## 📚 更多资源

- [Hugging Face Transformers文档](https://huggingface.co/docs/transformers/)
- [PyTorch文档](https://pytorch.org/docs/)
- [项目README](README.md)
