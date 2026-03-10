import logging
import json
import sys
import os
from dataclasses import dataclass, field
from typing import Optional, List, Dict
from tqdm import tqdm

import torch
from torch.utils.data import Dataset

import datasets
import transformers
from transformers import (
    AutoConfig,
    AutoModelForCausalLM,
    AutoTokenizer,
    HfArgumentParser,
    Trainer,
    TrainingArguments,
    set_seed,
)
from transformers.trainer_utils import get_last_checkpoint
import swanlab

# 超参类
@dataclass
class ModelArguments:
    model_name_or_path: Optional[str] = field(
        default=None,
        metadata={"help": "后训练使用，为预训练模型参数地址"},
    )

@dataclass
class DataTrainingArguments:
    train_files: Optional[str] = field(default=None, metadata={"help": "训练数据路径"})
    block_size: Optional[int] = field(
        default=2048,
        metadata={"help": "设置的文本块长度"},
    )
    preprocessing_num_workers: Optional[int] = field(
        default=None,
        metadata={"help": "预处理使用线程数."},
    )

def preprocess(sources, tokenizer, max_len):
    # 不同的 tokenizer 需要特别定义
    im_start_tokens = tokenizer("<|im_start|>").input_ids
    im_end_tokens = tokenizer("<|im_end|>").input_ids
    IGNORE_TOKEN_ID = -100 # 训练中需要忽略的标签（Hugging Face 默认是 -100）
    nl_tokens = tokenizer('\n').input_ids
    _system = tokenizer('system').input_ids + nl_tokens
    
    system_message = "You are a helpful assistant."
    roles = {"human": "<|im_start|>human", "assistant": "<|im_start|>assistant"}

    # 拼接多轮对话
    input_ids_list, targets_list = [], []
    # 多个样本
    for i in tqdm(range(len(sources)), desc="Tokenizing dataset"):
        # source 为一个多轮对话样本
        source = sources[i]
        # 从 user 开始
        if source[0]["from"] != "human":
            source = source[1:]
        # 分别是输入和输出
        input_id, target = [], []
        # system: 【BOS】system\nYou are a helpful assistant.【EOS】\n
        system = im_start_tokens + _system + tokenizer(system_message).input_ids + im_end_tokens + nl_tokens
        input_id += system
        # system 不需要拟合 (设为 -100)
        target += im_start_tokens + [IGNORE_TOKEN_ID] * (len(system)-3) + im_end_tokens + nl_tokens
        assert len(input_id) == len(target)
        
        # 依次拼接
        for j, sentence in enumerate(source):
            # sentence 为一轮对话
            role = roles.get(sentence["from"], "<|im_start|>human")
            # user：<|im_start|>human\ninstruction【EOS】\n
            # assistant：<|im_start|>assistant\nresponse【EOS】\n
            _input_id = tokenizer(role).input_ids + nl_tokens + \
                tokenizer(sentence["value"]).input_ids + im_end_tokens + nl_tokens
            input_id += _input_id
            
            if role == '<|im_start|>human':
                # user 不需要拟合
                _target = im_start_tokens + [IGNORE_TOKEN_ID] * (len(_input_id)-3) + im_end_tokens + nl_tokens
            elif role == '<|im_start|>assistant':
                # assistant 需要拟合
                _target = im_start_tokens + [IGNORE_TOKEN_ID] * len(tokenizer(role).input_ids) + \
                    _input_id[len(tokenizer(role).input_ids)+1:-2] + im_end_tokens + nl_tokens
            else:
                raise NotImplementedError(f"未知的 role: {role}")
            
            target += _target
            
        assert len(input_id) == len(target)
        
        # 最后进行截断或 PAD
        input_id = input_id[:max_len]
        target = target[:max_len]
        
        # 补齐到 max_len
        pad_len = max_len - len(input_id)
        if pad_len > 0:
            input_id += [tokenizer.pad_token_id] * pad_len
            target += [IGNORE_TOKEN_ID] * pad_len
            
        input_ids_list.append(input_id)
        targets_list.append(target)

    input_ids = torch.tensor(input_ids_list)
    targets = torch.tensor(targets_list)

    return dict(
        input_ids=input_ids,
        labels=targets,
        attention_mask=input_ids.ne(tokenizer.pad_token_id),
    )

class SupervisedDataset(Dataset):
    def __init__(self, raw_data, tokenizer, max_len: int):
        super(SupervisedDataset, self).__init__()
        # 加载并预处理数据
        sources = [example["conversations"] for example in raw_data]
        # preprocess 即上文定义的数据预处理逻辑
        data_dict = preprocess(sources, tokenizer, max_len)

        self.input_ids = data_dict["input_ids"]
        self.labels = data_dict["labels"]
        self.attention_mask = data_dict["attention_mask"]

    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, i) -> Dict[str, torch.Tensor]:
        return dict(
            input_ids=self.input_ids[i],
            labels=self.labels[i],
            attention_mask=self.attention_mask[i],
        )

def main():
    # 加载脚本参数
    parser = HfArgumentParser((ModelArguments, DataTrainingArguments, TrainingArguments))
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    # 初始化 SwanLab
    swanlab.init(project="sft", experiment_name="qwen-1.5b")

    # 设置日志
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
    )
    logger = logging.getLogger(__name__)

    # 将日志级别设置为 INFO
    transformers.utils.logging.set_verbosity_info()
    log_level = training_args.get_process_log_level()
    logger.setLevel(log_level)
    datasets.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.enable_default_handler()
    transformers.utils.logging.enable_explicit_format()

    # 训练整体情况记录
    logger.warning(
        f"Process rank: {training_args.local_rank}, device: {training_args.device}, n_gpu: {training_args.n_gpu}\n"
        + f"distributed training: {bool(training_args.local_rank != -1)}, 16-bits training: {training_args.fp16}"
    )
    logger.info(f"Training/evaluation parameters {training_args}")

    # 检查 checkpoint
    last_checkpoint = None
    if os.path.isdir(training_args.output_dir):
        last_checkpoint = get_last_checkpoint(training_args.output_dir)
        if last_checkpoint is None and len(os.listdir(training_args.output_dir)) > 0:
            raise ValueError(
                f"输出路径 ({training_args.output_dir}) 非空 "
            )
        elif last_checkpoint is not None and training_args.resume_from_checkpoint is None:
            logger.info(
                f"从 {last_checkpoint}恢复训练"
            )

    # 设置随机数种子.
    set_seed(training_args.seed)

    # 初始化 Tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_args.model_name_or_path, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    logger.info("完成 tokenizer 加载")

    # 初始化模型
    logger.warning("加载预训练模型")
    logger.info(f"模型参数地址：{model_args.model_name_or_path}")
    model = AutoModelForCausalLM.from_pretrained(model_args.model_name_or_path, trust_remote_code=True)
    n_params = sum({p.data_ptr(): p.numel() for p in model.parameters()}.values())
    logger.info(f"继承一个预训练模型 - Total size={n_params/2**20:.2f}M params")

    # 加载微调数据
    with open(data_args.train_files, 'r', encoding='utf-8') as f:
        # 演示用，限制加载前 10000 条
        lst = [json.loads(line) for line in f.readlines()[:10000]]
    logger.info("完成训练集加载")
    logger.info(f"训练集地址：{data_args.train_files}")
    logger.info(f'训练样本总数:{len(lst)}')

    train_dataset = SupervisedDataset(lst, tokenizer=tokenizer, max_len=data_args.block_size)

    logger.info("初始化 Trainer")
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        tokenizer=tokenizer
    )

    # 从 checkpoint 加载
    checkpoint = None
    if training_args.resume_from_checkpoint is not None:
        checkpoint = training_args.resume_from_checkpoint
    elif last_checkpoint is not None:
        checkpoint = last_checkpoint

    logger.info("开始训练")
    train_result = trainer.train(resume_from_checkpoint=checkpoint)
    trainer.save_model() 

if __name__ == "__main__":
    main()
