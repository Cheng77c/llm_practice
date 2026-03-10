"""
数据集下载脚本

下载预训练数据集（seq-monkey）和 SFT 数据集（BelleGroup）。

用法：
    python scripts/download_dataset.py
"""

import os


def download_pretrain_dataset(local_dir: str = "./dataset/pretrain") -> None:
    """从 ModelScope 下载预训练数据集并解压"""
    os.makedirs(local_dir, exist_ok=True)
    archive = os.path.join(local_dir, "mobvoi_seq_monkey_general_open_corpus.jsonl.tar.bz2")

    print(">>> 下载预训练数据集（seq-monkey）...")
    ret = os.system(
        f"modelscope download --dataset ddzhu123/seq-monkey "
        f"mobvoi_seq_monkey_general_open_corpus.jsonl.tar.bz2 --local_dir {local_dir}"
    )
    if ret != 0:
        print("ERROR: 下载失败，请确认已安装 modelscope：pip install modelscope")
        return

    print(">>> 解压预训练数据集...")
    ret = os.system(f"tar -xvf {archive} -C {local_dir}")
    if ret != 0:
        print(f"ERROR: 解压失败，请手动解压 {archive}")
        return

    print(f"预训练数据集已准备完毕，路径：{local_dir}")


def download_sft_dataset(local_dir: str = "./dataset/sft") -> None:
    """从 HuggingFace 下载 SFT 数据集"""
    os.makedirs(local_dir, exist_ok=True)

    # 使用镜像站加速
    os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

    print(">>> 下载 SFT 数据集（BelleGroup/train_3.5M_CN）...")
    ret = os.system(
        f"huggingface-cli download --repo-type dataset BelleGroup/train_3.5M_CN --local-dir {local_dir}"
    )
    if ret != 0:
        print("ERROR: 下载失败，请确认已安装 huggingface_hub：pip install huggingface_hub")
        return

    print(f"SFT 数据集已准备完毕，路径：{local_dir}")


if __name__ == "__main__":
    download_pretrain_dataset()
    download_sft_dataset()
