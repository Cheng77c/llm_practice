"""
模型单元测试

验证 Attention、MLP、DecoderLayer 和 Transformer 各模块的输出形状、
数值稳定性（无 NaN/Inf）以及 eval 模式的确定性。

运行：
    python tests/test_model.py
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import torch
from src.model import (
    Attention,
    DecoderLayer,
    MLP,
    ModelConfig,
    Transformer,
    precompute_freqs_cis,
)


def test_attention():
    print("=== 测试 Attention ===")
    args = ModelConfig()
    model = Attention(args)
    model.eval()

    batch_size, seq_len = 2, 50
    head_dim = args.dim // args.n_heads
    x = torch.rand(batch_size, seq_len, args.dim)
    freqs_cos, freqs_sin = precompute_freqs_cis(head_dim, seq_len)

    with torch.no_grad():
        output = model(x, freqs_cos, freqs_sin)
        output2 = model(x, freqs_cos, freqs_sin)

    assert output.shape == (batch_size, seq_len, args.dim), \
        f"形状不符：{output.shape}"
    assert not torch.isnan(output).any(), "输出含 NaN"
    assert not torch.isinf(output).any(), "输出含 Inf"
    assert torch.allclose(output, output2), "eval 模式输出不一致"
    print(f"  输出形状：{output.shape} ✓")
    print("  无 NaN/Inf ✓")
    print("  eval 模式确定性 ✓")


def test_mlp():
    print("=== 测试 MLP ===")
    args = ModelConfig()
    mlp = MLP(args.dim, args.hidden_dim, args.multiple_of, args.dropout)

    x = torch.randn(1, 50, args.dim)
    output = mlp(x)
    assert output.shape == x.shape, f"形状不符：{output.shape}"
    print(f"  输出形状：{output.shape} ✓")


def test_decoder_layer():
    print("=== 测试 DecoderLayer ===")
    args = ModelConfig()
    layer = DecoderLayer(0, args)

    seq_len = 50
    x = torch.randn(1, seq_len, args.dim)
    freqs_cos, freqs_sin = precompute_freqs_cis(args.dim // args.n_heads, seq_len)

    output = layer(x, freqs_cos, freqs_sin)
    assert output.shape == x.shape, f"形状不符：{output.shape}"
    print(f"  输出形状：{output.shape} ✓")


def test_transformer():
    print("=== 测试 Transformer ===")
    args = ModelConfig()
    model = Transformer(args)

    num_params = sum(p.numel() for p in model.parameters())
    print(f"  参数量：{num_params:,}")

    # 推理模式（无 targets）
    x = torch.randint(0, args.vocab_size, (1, 50))
    out = model(x)
    assert out.logits.shape == (1, 1, args.vocab_size), \
        f"推理 logits 形状不符：{out.logits.shape}"
    print(f"  推理 logits 形状：{out.logits.shape} ✓")

    # 训练模式（有 targets）
    targets = torch.randint(0, args.vocab_size, (1, 50))
    out = model(x, targets)
    assert out.logits.shape == (1, 50, args.vocab_size), \
        f"训练 logits 形状不符：{out.logits.shape}"
    assert out.loss is not None, "训练模式下 loss 不应为 None"
    print(f"  训练 logits 形状：{out.logits.shape} ✓")
    print(f"  loss shape：{out.loss.shape} ✓")


if __name__ == "__main__":
    test_attention()
    test_mlp()
    test_decoder_layer()
    test_transformer()
    print("\n所有测试通过 ✓")
