"""
数据集模块

包含：
- PretrainDataset: 预训练数据集，从 JSONL 文件中逐行读取文本
- SFTDataset: 监督微调数据集，支持多轮对话格式，按助手回复生成 loss mask
"""

import json
import os

import numpy as np
import torch
from torch.utils.data import Dataset
from tqdm import tqdm


class PretrainDataset(Dataset):
    """
    预训练数据集

    从 JSONL 格式文件中读取文本数据，每行为一条 JSON 记录，需包含 "text" 字段。
    采用字节偏移索引实现高效的随机访问（避免将整个文件加载入内存）。

    Args:
        data_path (str): JSONL 数据文件路径
        tokenizer: HuggingFace 分词器
        max_length (int): 最大序列长度，超出部分截断
    """

    def __init__(self, data_path: str, tokenizer, max_length: int = 512):
        self.data_path = data_path
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.padding = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else 0

        # 尝试加载缓存的偏移量文件，避免每次都扫描大文件
        cache_file = data_path + ".offsets.npy"
        if os.path.exists(cache_file) and os.path.getmtime(cache_file) >= os.path.getmtime(data_path):
            self._offsets = np.load(cache_file).tolist()
        else:
            print(f"正在索引数据文件（首次运行需要几分钟）：{data_path}")
            file_size = os.path.getsize(data_path)
            self._offsets = []
            with open(data_path, "rb") as f:
                self._offsets.append(0)
                with tqdm(total=file_size, unit="B", unit_scale=True, unit_divisor=1024, desc="索引中") as pbar:
                    while True:
                        line = f.readline()
                        if not line:
                            break
                        pbar.update(len(line))
                        self._offsets.append(f.tell())
            np.save(cache_file, np.array(self._offsets, dtype=np.int64))
            print(f"偏移量缓存已保存至 {cache_file}")
        self._total_lines = len(self._offsets) - 1

    def __len__(self) -> int:
        return self._total_lines

    def __getitem__(self, index: int):
        # 按字节偏移定位到指定行
        with open(self.data_path, "rb") as f:
            f.seek(self._offsets[index])
            line = f.readline().decode("utf-8")

        sample = json.loads(line)
        # 在文本开头添加 BOS token
        text = f"{self.tokenizer.bos_token}{sample['text']}"
        input_id = self.tokenizer(text).data["input_ids"][: self.max_length]

        text_len = len(input_id)
        padding_len = self.max_length - text_len
        input_id = input_id + [self.padding] * padding_len
        # loss_mask: 实际文本位置为 1，padding 位置为 0
        loss_mask = [1] * text_len + [0] * padding_len

        input_id = np.array(input_id)
        X = np.array(input_id[:-1]).astype(np.int64)   # 输入序列（去掉最后一个 token）
        Y = np.array(input_id[1:]).astype(np.int64)    # 目标序列（去掉第一个 token）
        loss_mask = np.array(loss_mask[1:]).astype(np.int64)

        return torch.from_numpy(X), torch.from_numpy(Y), torch.from_numpy(loss_mask)


class SFTDataset(Dataset):
    """
    监督微调（SFT）数据集

    从 JSONL 格式文件中读取多轮对话数据，格式为 HuggingFace chat template 兼容的消息列表。
    仅对助手（assistant）的回复部分计算损失，通过 loss_mask 实现。

    Args:
        data_path (str): JSONL 数据文件路径
        tokenizer: HuggingFace 分词器
        max_length (int): 最大序列长度，超出部分截断
    """

    def __init__(self, data_path: str, tokenizer, max_length: int = 512):
        self.data_path = data_path
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.padding = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else 0

        # 尝试加载缓存的偏移量文件，避免每次都扫描大文件
        cache_file = data_path + ".offsets.npy"
        if os.path.exists(cache_file) and os.path.getmtime(cache_file) >= os.path.getmtime(data_path):
            self._offsets = np.load(cache_file).tolist()
        else:
            print(f"正在索引数据文件（首次运行需要几分钟）：{data_path}")
            file_size = os.path.getsize(data_path)
            self._offsets = []
            with open(data_path, "rb") as f:
                self._offsets.append(0)
                with tqdm(total=file_size, unit="B", unit_scale=True, unit_divisor=1024, desc="索引中") as pbar:
                    while True:
                        line = f.readline()
                        if not line:
                            break
                        pbar.update(len(line))
                        self._offsets.append(f.tell())
            np.save(cache_file, np.array(self._offsets, dtype=np.int64))
            print(f"偏移量缓存已保存至 {cache_file}")
        self._total_lines = len(self._offsets) - 1

    def __len__(self) -> int:
        return self._total_lines

    def generate_loss_mask(self, input_id: list) -> list:
        """
        生成 loss mask，只对 assistant 回复部分计算损失。

        策略：找到 "<|im_start|>assistant\\n" 序列，
        将其后直到 EOS token（含）的范围标记为 1。
        """
        mask = [0] * len(input_id)
        a_sequence = self.tokenizer("<|im_start|>assistant\n").input_ids
        a_length = len(a_sequence)
        n = len(input_id)
        i = 0

        while i < n - a_length:
            if input_id[i : i + a_length] == a_sequence:
                # 找到 assistant 回复的起始位置，向后查找 EOS token
                j = None
                for idx in range(i + a_length, n):
                    if input_id[idx] == self.tokenizer.eos_token_id:
                        j = idx
                        break
                if j is not None and (i + a_length) <= j:
                    for pos in range(i + a_length, j + 1):
                        if pos < len(mask):
                            mask[pos] = 1
                i += a_length
            else:
                i += 1
        return mask

    def __getitem__(self, index: int):
        # 按字节偏移定位到指定行
        with open(self.data_path, "rb") as f:
            f.seek(self._offsets[index])
            line = f.readline().decode("utf-8")

        sample = json.loads(line)
        # 将对话列表应用 chat template 转为字符串
        text = self.tokenizer.apply_chat_template(
            sample, tokenize=False, add_generation_prompt=False
        )
        input_id = self.tokenizer(text).data["input_ids"][: self.max_length]

        text_len = len(input_id)
        padding_len = self.max_length - text_len
        input_id = input_id + [self.padding] * padding_len
        loss_mask = self.generate_loss_mask(input_id)

        input_id = np.array(input_id)
        X = np.array(input_id[:-1]).astype(np.int64)
        Y = np.array(input_id[1:]).astype(np.int64)
        loss_mask = np.array(loss_mask[1:]).astype(np.int64)

        return torch.from_numpy(X), torch.from_numpy(Y), torch.from_numpy(loss_mask)
