# -*- coding: utf-8 -*-
import os
# 设置缓存路径到 10PB 的共享盘，防止磁盘满
os.environ["HF_DATASETS_CACHE"] = "/share/cheng/hf_cache"

from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments, Trainer, default_data_collator
from datasets import load_dataset
from itertools import chain
logger = logging.getLogger(__name__)

# 1. 加载模型和分词器
model = AutoModelForCausalLM.from_pretrained("./models", trust_remote_code=True)
tokenizer = AutoTokenizer.from_pretrained("./models", trust_remote_code=True)

# 2. 加载预训练数据
ds = load_dataset('json', data_files='dataset/pretrain/mobvoi_seq_monkey_general_open_corpus.jsonl')
# 获取原始列名，用于后续删除
column_names = ds["train"].column_names 

# 3. 对数据集进行分词
def tokenize_function(examples):
    return tokenizer(examples["text"])

tokenized_datasets = ds.map(
    tokenize_function,
    batched=True,
    num_proc=10,
    remove_columns=column_names,
    load_from_cache_file=True,
    desc="Running tokenizer on dataset",
)

# 4. 数据打包 (Packing)
block_size = 2048
def group_texts(examples):
    concatenated_examples = {k: list(chain(*examples[k])) for k in examples.keys()}
    total_length = len(concatenated_examples[list(examples.keys())[0]])
    if total_length >= block_size:
        total_length = (total_length // block_size) * block_size
    result = {
        k: [t[i : i + block_size] for i in range(0, total_length, block_size)]
        for k, t in concatenated_examples.items()
    }
    result["labels"] = result["input_ids"].copy()
    return result

lm_datasets = tokenized_datasets.map(
    group_texts,
    batched=True,
    num_proc=10,
    load_from_cache_file=True,
    desc=f"Grouping texts in chunks of {block_size}",
    batch_size=40000,
)
train_dataset = lm_datasets["train"]

# 5. 配置训练参数
training_args = TrainingArguments(
    output_dir="/share/cheng/output", # 训练参数输出路径 (放在共享盘)
    per_device_train_batch_size=4,# 训练的 batch_size
    gradient_accumulation_steps=4,# 梯度累计步数
    logging_steps=10,# 打印 loss 的步数间隔
    num_train_epochs=1,# 训练的 epoch 数
    save_steps=100, # 保存模型参数的步数间隔
    learning_rate=1e-4,# 学习率
    gradient_checkpointing=True,# 开启梯度检查点
    fp16=True, # 开启半精度训练，更省显存，速度更快
    push_to_hub=False,
)

# 6. 开始训练
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    tokenizer=tokenizer,
    data_collator=default_data_collator
)

trainer.train()
