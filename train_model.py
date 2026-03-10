"""
Tiny-LLM 预训练入口脚本

用法示例：
    # 单卡训练
    python train_model.py --device cuda:0

    # 多卡训练（DataParallel）
    python train_model.py --gpus 0,1,2,3

    # 使用 SwanLab 记录实验
    python train_model.py --use_swanlab
"""

import argparse
import math
import os
import time
from contextlib import nullcontext

import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from transformers import AutoTokenizer

from src.model import ModelConfig, Transformer
from src.dataset import PretrainDataset

# 可选依赖：SwanLab 实验跟踪
try:
    import swanlab
    SWANLAB_AVAILABLE = True
except ImportError:
    SWANLAB_AVAILABLE = False


# ---------------------------------------------------------------------------
# 辅助函数
# ---------------------------------------------------------------------------

def Logger(content: str) -> None:
    """带时间戳的日志打印"""
    print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {content}", flush=True)


def get_lr(it: int, total: int, args: argparse.Namespace) -> float:
    """
    余弦退火学习率调度（含 Warmup）

    三个阶段：
    1. Warmup：学习率从 0 线性增长到 args.learning_rate
    2. 余弦退火：学习率从 args.learning_rate 衰减到 min_lr
    3. 超出 total 后：保持 min_lr

    Args:
        it: 当前全局迭代步数
        total: 总迭代步数
        args: 训练参数命名空间

    Returns:
        当前步骤对应的学习率
    """
    min_lr = args.learning_rate / 10

    # 阶段 1：Warmup
    if it < args.warmup_iters:
        return args.learning_rate * it / args.warmup_iters

    # 阶段 3：超出训练步数
    if it > total:
        return min_lr

    # 阶段 2：余弦退火
    decay_ratio = (it - args.warmup_iters) / (total - args.warmup_iters)
    assert 0 <= decay_ratio <= 1
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
    return min_lr + coeff * (args.learning_rate - min_lr)


def init_model(args: argparse.Namespace, lm_config: ModelConfig):
    """
    初始化模型和分词器

    Args:
        args: 命令行参数
        lm_config: 模型配置

    Returns:
        tuple: (model, tokenizer)
    """
    def count_parameters(model) -> int:
        return sum(p.numel() for p in model.parameters() if p.requires_grad)

    tokenizer = AutoTokenizer.from_pretrained("./tokenizer_k/")
    model = Transformer(lm_config)

    # 多卡：DataParallel
    num_gpus = torch.cuda.device_count()
    if num_gpus > 1:
        Logger(f"检测到 {num_gpus} 块 GPU，使用 DataParallel 并行训练")
        model = torch.nn.DataParallel(model)

    model = model.to(args.device)
    Logger(f"LLM 总参数量：{count_parameters(model) / 1e6:.3f} 百万")
    return model, tokenizer


def train_epoch(
    epoch: int,
    model,
    optimizer,
    scaler,
    train_loader: DataLoader,
    args: argparse.Namespace,
    lm_config: ModelConfig,
    iter_per_epoch: int,
    ctx,
) -> None:
    """
    训练一个 epoch

    实现了完整的训练循环：
    - 动态学习率（余弦退火 + Warmup）
    - 混合精度训练（AMP）
    - 梯度累积
    - 梯度裁剪
    - 按间隔记录日志并保存检查点

    Args:
        epoch: 当前 epoch 编号（从 0 开始）
        model: 待训练模型
        optimizer: 优化器
        scaler: 混合精度梯度缩放器
        train_loader: 训练数据加载器
        args: 命令行参数
        lm_config: 模型配置
        iter_per_epoch: 每个 epoch 的迭代次数
        ctx: 混合精度上下文管理器
    """
    model.train()
    start_time = time.time()

    for step, (X, Y, loss_mask) in enumerate(train_loader):
        X = X.to(args.device)
        Y = Y.to(args.device)
        loss_mask = loss_mask.to(args.device)

        # 计算并设置当前步骤的学习率
        global_step = epoch * iter_per_epoch + step
        lr = get_lr(global_step, args.epochs * iter_per_epoch, args)
        for param_group in optimizer.param_groups:
            param_group["lr"] = lr

        # 前向传播（混合精度）
        with ctx:
            out = model(X, Y)
            loss = out.loss / args.accumulation_steps
            loss_mask = loss_mask.view(-1)
            # 只对非 padding 位置计算损失
            loss = torch.sum(loss * loss_mask) / loss_mask.sum()

        # 反向传播
        scaler.scale(loss).backward()

        # 每 accumulation_steps 步执行一次优化器更新
        if (step + 1) % args.accumulation_steps == 0:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)

        # 日志记录
        if step % args.log_interval == 0:
            elapsed = time.time() - start_time
            eta_min = elapsed / (step + 1) * iter_per_epoch // 60 - elapsed // 60
            Logger(
                "Epoch:[{}/{}]({}/{}) loss:{:.3f} lr:{:.7f} ETA:{}min".format(
                    epoch + 1,
                    args.epochs,
                    step,
                    iter_per_epoch,
                    loss.item() * args.accumulation_steps,
                    optimizer.param_groups[-1]["lr"],
                    int(eta_min),
                )
            )
            if args.use_swanlab and SWANLAB_AVAILABLE:
                swanlab.log({
                    "loss": loss.item() * args.accumulation_steps,
                    "lr": optimizer.param_groups[-1]["lr"],
                })

        # 定期保存检查点
        if (step + 1) % args.save_interval == 0:
            _save_checkpoint(model, lm_config, args.save_dir)

        # 每 20000 步额外保存一个带步数标记的检查点
        if (step + 1) % 20000 == 0:
            _save_checkpoint(model, lm_config, args.save_dir, step=step + 1)


def _save_checkpoint(model, lm_config: ModelConfig, save_dir: str, step: int = None) -> None:
    """保存模型检查点"""
    model.eval()
    suffix = f"_step{step}" if step is not None else ""
    ckp = os.path.join(
        save_dir,
        f"pretrain_{lm_config.dim}_{lm_config.n_layers}_{lm_config.vocab_size}{suffix}.pth",
    )
    state_dict = model.module.state_dict() if isinstance(model, torch.nn.DataParallel) else model.state_dict()
    torch.save(state_dict, ckp)
    Logger(f"检查点已保存：{ckp}")
    model.train()


# ---------------------------------------------------------------------------
# 主程序入口
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    # ==================== 命令行参数解析 ====================
    parser = argparse.ArgumentParser(description="Tiny-LLM 预训练")

    # 基础训练参数
    parser.add_argument("--out_dir", type=str, default="base_model_215M", help="模型输出目录")
    parser.add_argument("--epochs", type=int, default=1, help="训练轮数")
    parser.add_argument("--batch_size", type=int, default=64, help="批次大小")
    parser.add_argument("--learning_rate", type=float, default=2e-4, help="学习率")
    parser.add_argument(
        "--device",
        type=str,
        default="cuda:0" if torch.cuda.is_available() else "cpu",
        help="训练设备",
    )
    parser.add_argument("--dtype", type=str, default="bfloat16", help="混合精度数据类型 (float16/bfloat16/float32)")

    # 实验跟踪与数据加载
    parser.add_argument("--use_swanlab", action="store_true", help="使用 SwanLab 记录实验")
    parser.add_argument("--num_workers", type=int, default=8, help="DataLoader 工作进程数")
    parser.add_argument(
        "--data_path",
        type=str,
        default="./seq_monkey_datawhale.jsonl",
        help="训练数据路径（JSONL 格式）",
    )

    # 训练优化参数
    parser.add_argument("--accumulation_steps", type=int, default=8, help="梯度累积步数")
    parser.add_argument("--grad_clip", type=float, default=1.0, help="梯度裁剪阈值")
    parser.add_argument("--warmup_iters", type=int, default=0, help="Warmup 迭代次数")

    # 日志与保存
    parser.add_argument("--log_interval", type=int, default=100, help="日志记录间隔（步）")
    parser.add_argument("--save_interval", type=int, default=1000, help="检查点保存间隔（步）")

    # 多 GPU
    parser.add_argument(
        "--gpus",
        type=str,
        default="0,1,2,3,4,5,6,7",
        help="使用的 GPU ID，逗号分隔（例如：'0,1,2'）",
    )

    args = parser.parse_args()

    # ==================== GPU 环境设置 ====================
    if args.gpus is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = args.gpus
        args.device = "cuda:0" if torch.cuda.is_available() else "cpu"

    # ==================== SwanLab 初始化 ====================
    if args.use_swanlab:
        if not SWANLAB_AVAILABLE:
            raise ImportError("请先安装 swanlab：pip install swanlab")
        # 注意：使用前请先调用 swanlab.login(api_key='your_key')
        swanlab.init(
            project="Happy-LLM",
            experiment_name="Pretrain-215M",
            config=vars(args),
        )

    # ==================== 模型配置 ====================
    lm_config = ModelConfig(
        dim=1024,      # 隐藏维度
        n_layers=18,   # Transformer 层数
    )

    # ==================== 训练环境设置 ====================
    max_seq_len = lm_config.max_seq_len
    args.save_dir = args.out_dir
    os.makedirs(args.out_dir, exist_ok=True)

    torch.manual_seed(42)

    device_type = "cuda" if "cuda" in args.device else "cpu"
    # CPU 使用 nullcontext，GPU 使用 autocast
    ctx = nullcontext() if device_type == "cpu" else torch.amp.autocast("cuda")

    # ==================== 模型和数据初始化 ====================
    model, tokenizer = init_model(args, lm_config)

    train_ds = PretrainDataset(args.data_path, tokenizer, max_length=max_seq_len)
    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        pin_memory=True,
        drop_last=False,
        shuffle=True,
        num_workers=args.num_workers,
    )

    # ==================== 优化器和梯度缩放器 ====================
    scaler = torch.amp.GradScaler("cuda", enabled=(args.dtype in ["float16", "bfloat16"]))
    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)

    # ==================== 训练循环 ====================
    iter_per_epoch = len(train_loader)
    Logger(f"开始训练，共 {args.epochs} 个 epoch，每 epoch {iter_per_epoch} 步")

    for epoch in range(args.epochs):
        train_epoch(epoch, model, optimizer, scaler, train_loader, args, lm_config, iter_per_epoch, ctx)
