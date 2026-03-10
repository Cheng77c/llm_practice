# Tiny-LLM 预训练项目

基于 LLaMA 架构的小型中文语言模型，支持预训练和 SFT 微调。

## 项目结构

```
llama/
├── src/                        # 核心模型代码
│   ├── __init__.py
│   ├── model.py                # Transformer 模型（RoPE + GQA + SwiGLU）
│   └── dataset.py              # PretrainDataset / SFTDataset
├── scripts/                    # 工具脚本
│   ├── download_dataset.py     # 下载数据集
│   ├── process_data.py         # 数据预处理
│   └── train_tokenizer.py      # 分词器训练与评估
├── configs/
│   └── pretrain_config.py      # 模型规格与超参数配置
├── tests/
│   └── test_model.py           # 模型单元测试
├── tokenizer_k/                # 训练好的分词器
├── train_model.py              # 预训练入口
├── requirements.txt
└── README.md
```

## 快速开始

### 1. 安装依赖
```bash
pip install -r requirements.txt
```

### 2. 下载数据集
```bash
python scripts/download_dataset.py
```

### 3. 数据预处理
```bash
python scripts/process_data.py --output_dir /share/cheng/llama_data
```

### 4. 训练分词器
```bash
python scripts/train_tokenizer.py --train --eval
```

### 5. 开始预训练
```bash
# 单卡
python train_model.py --device cuda:0 --batch_size 32

# 多卡（DataParallel）
python train_model.py --gpus 0,1,2,3 --batch_size 64

# 使用 SwanLab 记录实验
python train_model.py --use_swanlab
```

### 6. 运行测试
```bash
python tests/test_model.py
```

## 模型架构

| 参数 | 默认值（215M）|
|---|---|
| 隐藏维度 `dim` | 1024 |
| 层数 `n_layers` | 18 |
| 注意力头数 `n_heads` | 16 |
| KV 头数 `n_kv_heads` | 8 |
| 词表大小 `vocab_size` | 6144 |
| 最大序列长度 | 512 |

主要特性：
- **RoPE**：旋转位置编码
- **GQA**：分组查询注意力（减少 KV 缓存显存）
- **SwiGLU**：前馈网络激活函数
- **Flash Attention**：自动检测并启用

## 主要训练参数

| 参数 | 默认值 | 说明 |
|---|---|---|
| `--epochs` | 1 | 训练轮数 |
| `--batch_size` | 64 | 批次大小 |
| `--learning_rate` | 2e-4 | 初始学习率 |
| `--accumulation_steps` | 8 | 梯度累积步数 |
| `--dtype` | bfloat16 | 混合精度类型 |
| `--warmup_iters` | 0 | Warmup 迭代次数 |
| `--save_interval` | 1000 | 检查点保存间隔 |
