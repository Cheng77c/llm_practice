"""
数据预处理脚本

处理原始数据集，生成适合训练的 JSONL 格式文件：
1. 预训练数据：将长文本切分为固定长度的块
2. SFT 数据：将对话数据转换为标准消息格式

用法：
    python scripts/process_data.py
    python scripts/process_data.py --output_dir /share/cheng/llama_data
"""

import argparse
import json
import os

from tqdm import tqdm


def split_text(text: str, chunk_size: int = 512) -> list[str]:
    """将长文本切分为固定长度的块"""
    return [text[i : i + chunk_size] for i in range(0, len(text), chunk_size)]


def process_pretrain_data(input_file: str, output_file: str, chunk_size: int = 512) -> None:
    """
    处理预训练数据

    将原始 JSONL 中的长文本切分为 chunk_size 长度的片段，
    每个片段作为一条独立的训练样本。

    Args:
        input_file: 原始 JSONL 文件路径
        output_file: 输出 JSONL 文件路径
        chunk_size: 文本切分长度（字符数）
    """
    if not os.path.exists(input_file):
        print(f"提示：找不到预训练输入文件 {input_file}，跳过。")
        return

    print(f">>> 处理预训练数据：{input_file} -> {output_file}")
    file_size = os.path.getsize(input_file)

    with open(output_file, "w", encoding="utf-8") as out_f:
        with tqdm(total=file_size, unit="B", unit_scale=True, unit_divisor=1024, desc="Pretrain") as pbar:
            with open(input_file, "r", encoding="utf-8") as in_f:
                for line in in_f:
                    try:
                        data = json.loads(line)
                        text = data.get("text", "")
                        for chunk in split_text(text, chunk_size):
                            out_f.write(json.dumps({"text": chunk}, ensure_ascii=False) + "\n")
                    except json.JSONDecodeError:
                        pass
                    pbar.update(len(line.encode("utf-8")))

    print(f"预训练数据处理完成：{output_file}")


def convert_sft_message(data: list) -> list:
    """
    将 BelleGroup 格式的对话数据转换为标准消息格式

    Args:
        data: 原始对话列表，每条含 "from" 和 "value" 字段

    Returns:
        标准格式消息列表（含 system/user/assistant 角色）
    """
    messages = [{"role": "system", "content": "你是一个AI助手"}]
    for item in data:
        if item["from"] == "human":
            messages.append({"role": "user", "content": item["value"]})
        elif item["from"] == "gpt" or item["from"] == "assistant":
            messages.append({"role": "assistant", "content": item["value"]})
    return messages


def process_sft_data(input_file: str, output_file: str) -> None:
    """
    处理 SFT 数据

    将 BelleGroup 格式的对话 JSON 转换为标准多轮对话 JSONL。

    Args:
        input_file: 原始 JSON 文件路径
        output_file: 输出 JSONL 文件路径
    """
    if not os.path.exists(input_file):
        print(f"提示：找不到 SFT 输入文件 {input_file}，跳过。")
        return

    print(f">>> 处理 SFT 数据：{input_file} -> {output_file}")
    file_size = os.path.getsize(input_file)

    with open(output_file, "w", encoding="utf-8") as out_f:
        with tqdm(total=file_size, unit="B", unit_scale=True, unit_divisor=1024, desc="SFT") as pbar:
            with open(input_file, "r", encoding="utf-8") as in_f:
                for line in in_f:
                    line = line.strip()
                    if not line or line in ["[", "]", "],"]:
                        pbar.update(len(line.encode("utf-8")) + 1)
                        continue
                    if line.endswith(","):
                        line = line[:-1]
                    try:
                        item = json.loads(line)
                        convs = item.get("conversations", [])
                        if convs:
                            messages = convert_sft_message(convs)
                            out_f.write(json.dumps(messages, ensure_ascii=False) + "\n")
                    except json.JSONDecodeError:
                        pass
                    pbar.update(len(line.encode("utf-8")) + 1)

    print(f"SFT 数据处理完成：{output_file}")


def main():
    parser = argparse.ArgumentParser(description="数据预处理脚本")
    parser.add_argument("--output_dir", type=str, default="/share/cheng/llama_data", help="输出目录")
    parser.add_argument("--pretrain_input", type=str,
                        default="./dataset/pretrain/mobvoi_seq_monkey_general_open_corpus.jsonl",
                        help="预训练原始数据路径")
    parser.add_argument("--sft_input", type=str,
                        default="./dataset/sft/train_3.5M_CN.json",
                        help="SFT 原始数据路径")
    parser.add_argument("--chunk_size", type=int, default=512, help="预训练文本切分长度（字符）")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    process_pretrain_data(
        input_file=args.pretrain_input,
        output_file=os.path.join(args.output_dir, "seq_monkey_datawhale.jsonl"),
        chunk_size=args.chunk_size,
    )

    process_sft_data(
        input_file=args.sft_input,
        output_file=os.path.join(args.output_dir, "BelleGroup_sft.jsonl"),
    )

    print("\n所有数据处理完成！")


if __name__ == "__main__":
    main()
