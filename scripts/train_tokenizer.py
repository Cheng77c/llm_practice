"""
分词器训练脚本

使用 BPE 算法在自定义语料上训练分词器并保存，支持评估已训练的分词器。

用法：
    # 训练分词器
    python scripts/train_tokenizer.py --train

    # 评估分词器
    python scripts/train_tokenizer.py --eval

    # 训练并评估
    python scripts/train_tokenizer.py --train --eval
"""

import argparse
import json
import os
import random
from typing import Generator

from tokenizers import Tokenizer, decoders, models, pre_tokenizers, trainers
from tokenizers.normalizers import NFKC
from transformers import AutoTokenizer

random.seed(42)


def read_texts_from_jsonl(file_path: str) -> Generator[str, None, None]:
    """从 JSONL 文件逐行读取文本数据（生成器，避免大文件内存占用）"""
    with open(file_path, "r", encoding="utf-8") as f:
        for line_num, line in enumerate(f, 1):
            try:
                data = json.loads(line)
                if "text" not in data:
                    raise KeyError(f"第 {line_num} 行缺少 'text' 字段")
                yield data["text"]
            except json.JSONDecodeError:
                print(f"JSON 解析错误，跳过第 {line_num} 行")
            except KeyError as e:
                print(e)


def create_tokenizer_config(save_dir: str) -> None:
    """创建 HuggingFace 兼容的 tokenizer 配置文件"""
    config = {
        "add_bos_token": False,
        "add_eos_token": False,
        "add_prefix_space": True,
        "bos_token": "<|im_start|>",
        "eos_token": "<|im_end|>",
        "pad_token": "<|im_end|>",
        "unk_token": "<unk>",
        "model_max_length": 1000000000000000019884624838656,
        "clean_up_tokenization_spaces": False,
        "tokenizer_class": "PreTrainedTokenizerFast",
        "chat_template": (
            "{% for message in messages %}"
            "{% if message['role'] == 'system' %}"
            "<|im_start|>system\n{{ message['content'] }}<|im_end|>\n"
            "{% elif message['role'] == 'user' %}"
            "<|im_start|>user\n{{ message['content'] }}<|im_end|>\n"
            "{% elif message['role'] == 'assistant' %}"
            "<|im_start|>assistant\n{{ message['content'] }}<|im_end|>\n"
            "{% endif %}"
            "{% endfor %}"
            "{% if add_generation_prompt %}"
            "{{ '<|im_start|>assistant\n' }}"
            "{% endif %}"
        ),
    }
    with open(os.path.join(save_dir, "tokenizer_config.json"), "w", encoding="utf-8") as f:
        json.dump(config, f, ensure_ascii=False, indent=4)

    special_tokens_map = {
        "bos_token": "<|im_start|>",
        "eos_token": "<|im_end|>",
        "unk_token": "<unk>",
        "pad_token": "<|im_end|>",
        "additional_special_tokens": ["<s>", "</s>"],
    }
    with open(os.path.join(save_dir, "special_tokens_map.json"), "w", encoding="utf-8") as f:
        json.dump(special_tokens_map, f, ensure_ascii=False, indent=4)


def train_tokenizer(data_path: str, save_dir: str, vocab_size: int = 6144) -> None:
    """
    训练并保存 BPE 分词器

    Args:
        data_path: 训练语料 JSONL 文件路径
        save_dir: 分词器保存目录
        vocab_size: 词表大小
    """
    os.makedirs(save_dir, exist_ok=True)

    tokenizer = Tokenizer(models.BPE(unk_token="<unk>"))
    tokenizer.normalizer = NFKC()
    tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    tokenizer.decoder = decoders.ByteLevel()

    special_tokens = ["<unk>", "<s>", "</s>", "<|im_start|>", "<|im_end|>"]
    trainer = trainers.BpeTrainer(
        vocab_size=vocab_size,
        special_tokens=special_tokens,
        min_frequency=2,
        show_progress=True,
        initial_alphabet=pre_tokenizers.ByteLevel.alphabet(),
    )

    print(f">>> 开始训练分词器，数据路径：{data_path}，词表大小：{vocab_size}")
    texts = read_texts_from_jsonl(data_path)
    tokenizer.train_from_iterator(texts, trainer=trainer, length=os.path.getsize(data_path))

    # 验证特殊 token 映射
    expected = {
        "<unk>": 0, "<s>": 1, "</s>": 2, "<|im_start|>": 3, "<|im_end|>": 4
    }
    for token, expected_id in expected.items():
        actual_id = tokenizer.token_to_id(token)
        assert actual_id == expected_id, (
            f"特殊 token '{token}' 的 ID 不符：期望 {expected_id}，实际 {actual_id}"
        )

    tokenizer.save(os.path.join(save_dir, "tokenizer.json"))
    create_tokenizer_config(save_dir)
    print(f"分词器已保存至：{save_dir}")


def eval_tokenizer(tokenizer_path: str) -> None:
    """
    评估分词器的基本功能

    Args:
        tokenizer_path: 分词器目录路径
    """
    try:
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
    except Exception as e:
        print(f"加载分词器失败：{e}")
        return

    print("\n=== 基本信息 ===")
    print(f"词表大小：{len(tokenizer)}")
    print(f"特殊 token：{tokenizer.all_special_tokens}")
    print(f"特殊 token ID：{tokenizer.all_special_ids}")

    messages = [
        {"role": "system", "content": "你是一个AI助手。"},
        {"role": "user", "content": "How are you?"},
        {"role": "assistant", "content": "I'm fine, thank you. And you?"},
        {"role": "user", "content": "I'm good too."},
        {"role": "assistant", "content": "That's great to hear!"},
    ]
    print("\n=== Chat Template 测试 ===")
    prompt = tokenizer.apply_chat_template(messages, tokenize=False)
    print(prompt)

    print("\n=== 编码/解码测试 ===")
    encoded = tokenizer(prompt, truncation=True, max_length=256)
    decoded = tokenizer.decode(encoded["input_ids"], skip_special_tokens=False)
    print(f"解码结果与原始文本一致：{decoded == prompt}")

    print("\n=== 特殊 token 保留测试 ===")
    test_text = "<|im_start|>user\nHello<|im_end|>"
    encoded = tokenizer(test_text).input_ids
    decoded = tokenizer.decode(encoded)
    print(f"原始：{test_text}")
    print(f"解码：{decoded}")
    print(f"特殊 token 保留：{decoded == test_text}")


def main():
    parser = argparse.ArgumentParser(description="BPE 分词器训练与评估")
    parser.add_argument("--train", action="store_true", help="训练分词器")
    parser.add_argument("--eval", action="store_true", help="评估分词器")
    parser.add_argument(
        "--data_path",
        type=str,
        default="./dataset/pretrain/mobvoi_seq_monkey_general_open_corpus_small.jsonl",
        help="训练数据路径",
    )
    parser.add_argument("--save_dir", type=str, default="./tokenizer_k", help="分词器保存目录")
    parser.add_argument("--vocab_size", type=int, default=6144, help="词表大小")
    args = parser.parse_args()

    if not args.train and not args.eval:
        parser.print_help()
        return

    if args.train:
        train_tokenizer(args.data_path, args.save_dir, args.vocab_size)

    if args.eval:
        eval_tokenizer(args.save_dir)


if __name__ == "__main__":
    main()
