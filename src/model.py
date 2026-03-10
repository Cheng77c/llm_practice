"""
模型定义模块

包含：
- ModelConfig: 模型配置类
- RMSNorm: 均方根归一化层
- Attention: 多头注意力（支持 GQA 和 Flash Attention）
- MLP: 前馈网络（SwiGLU 激活）
- DecoderLayer: Transformer 解码器层
- Transformer: 完整的 Transformer 模型
"""

import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import PreTrainedConfig, PreTrainedModel
from transformers.modeling_outputs import CausalLMOutputWithPast


class ModelConfig(PreTrainedConfig):
    """Transformer 模型配置"""

    model_type = "Tiny-K"

    def __init__(
        self,
        dim: int = 768,           # 模型隐藏维度
        n_layers: int = 12,       # Transformer 层数
        n_heads: int = 16,        # 注意力头数
        n_kv_heads: int = 8,      # KV 头数（用于 GQA）
        vocab_size: int = 6144,   # 词表大小
        hidden_dim: int = None,   # 前馈网络隐藏维度，None 时自动计算
        multiple_of: int = 64,    # 隐藏维度对齐到该值的倍数
        norm_eps: float = 1e-5,   # 归一化层的 epsilon
        max_seq_len: int = 512,   # 最大序列长度
        dropout: float = 0.0,     # Dropout 概率
        flash_attn: bool = True,  # 是否优先使用 Flash Attention
        **kwargs,
    ):
        self.dim = dim
        self.n_layers = n_layers
        self.n_heads = n_heads
        self.n_kv_heads = n_kv_heads
        self.vocab_size = vocab_size
        self.hidden_dim = hidden_dim
        self.multiple_of = multiple_of
        self.norm_eps = norm_eps
        self.max_seq_len = max_seq_len
        self.dropout = dropout
        self.flash_attn = flash_attn
        super().__init__(**kwargs)


# ---------------------------------------------------------------------------
# 工具函数
# ---------------------------------------------------------------------------

def precompute_freqs_cis(dim: int, end: int, theta: float = 10000.0):
    """预计算 RoPE 的余弦/正弦分量"""
    freqs = 1.0 / (theta ** (torch.arange(0, dim, 2).float() / dim))
    t = torch.arange(end, device=freqs.device)
    freqs = torch.outer(t, freqs)
    freqs_cos = torch.cos(freqs)
    freqs_sin = torch.sin(freqs)
    return freqs_cos, freqs_sin


def reshape_for_broadcast(freqs_cis: torch.Tensor, x: torch.Tensor):
    """将频率张量广播到与 x 相同的维度"""
    ndim = x.ndim
    assert 0 <= 1 < ndim
    assert freqs_cis.shape == (x.shape[1], x.shape[-1])
    shape = [d if i == 1 or i == ndim - 1 else 1 for i, d in enumerate(x.shape)]
    return freqs_cis.view(*shape)


def apply_rotary_emb(
    xq: torch.Tensor,
    xk: torch.Tensor,
    freqs_cos: torch.Tensor,
    freqs_sin: torch.Tensor,
):
    """应用旋转位置编码（RoPE）"""
    xq_r, xq_i = xq.float().reshape(xq.shape[:-1] + (-1, 2)).unbind(-1)
    xk_r, xk_i = xk.float().reshape(xk.shape[:-1] + (-1, 2)).unbind(-1)

    freqs_cos = reshape_for_broadcast(freqs_cos, xq_r)
    freqs_sin = reshape_for_broadcast(freqs_sin, xq_r)

    xq_out_r = xq_r * freqs_cos - xq_i * freqs_sin
    xq_out_i = xq_r * freqs_sin + xq_i * freqs_cos
    xk_out_r = xk_r * freqs_cos - xk_i * freqs_sin
    xk_out_i = xk_r * freqs_sin + xk_i * freqs_cos

    xq_out = torch.stack([xq_out_r, xq_out_i], dim=-1).flatten(3)
    xk_out = torch.stack([xk_out_r, xk_out_i], dim=-1).flatten(3)
    return xq_out.type_as(xq), xk_out.type_as(xk)


def repeat_kv(x: torch.Tensor, n_rep: int) -> torch.Tensor:
    """将 KV 头重复扩展，用于 GQA（分组查询注意力）"""
    bs, slen, n_kv_heads, head_dim = x.shape
    if n_rep == 1:
        return x
    return (
        x[:, :, :, None, :]
        .expand(bs, slen, n_kv_heads, n_rep, head_dim)
        .reshape(bs, slen, n_kv_heads * n_rep, head_dim)
    )


# ---------------------------------------------------------------------------
# 网络模块
# ---------------------------------------------------------------------------

class RMSNorm(nn.Module):
    """均方根归一化（Root Mean Square Normalization）"""

    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def _norm(self, x):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)

    def forward(self, x):
        output = self._norm(x).to(self.weight.dtype)
        return output * self.weight


class Attention(nn.Module):
    """多头注意力模块，支持 GQA 和 Flash Attention"""

    def __init__(self, args: ModelConfig):
        super().__init__()
        self.n_kv_heads = args.n_heads if args.n_kv_heads is None else args.n_kv_heads
        assert args.n_heads % self.n_kv_heads == 0, \
            f"n_heads ({args.n_heads}) must be divisible by n_kv_heads ({self.n_kv_heads})"

        self.n_local_heads = args.n_heads
        self.n_local_kv_heads = self.n_kv_heads
        self.n_rep = self.n_local_heads // self.n_local_kv_heads
        self.head_dim = args.dim // args.n_heads

        self.wq = nn.Linear(args.dim, args.n_heads * self.head_dim, bias=False)
        self.wk = nn.Linear(args.dim, self.n_kv_heads * self.head_dim, bias=False)
        self.wv = nn.Linear(args.dim, self.n_kv_heads * self.head_dim, bias=False)
        self.wo = nn.Linear(args.n_heads * self.head_dim, args.dim, bias=False)

        self.attn_dropout = nn.Dropout(args.dropout)
        self.resid_dropout = nn.Dropout(args.dropout)
        self.dropout = args.dropout

        # 检测是否支持 Flash Attention
        self.flash = hasattr(torch.nn.functional, "scaled_dot_product_attention")
        if not self.flash:
            print("WARNING: Flash Attention not available, using manual attention with causal mask.")
            mask = torch.full((1, 1, args.max_seq_len, args.max_seq_len), float("-inf"))
            mask = torch.triu(mask, diagonal=1)
            self.register_buffer("mask", mask)

    def forward(
        self,
        x: torch.Tensor,
        freqs_cos: torch.Tensor,
        freqs_sin: torch.Tensor,
    ):
        bsz, slen, _ = x.shape

        xq, xk, xv = self.wq(x), self.wk(x), self.wv(x)
        xq = xq.view(bsz, slen, self.n_local_heads, self.head_dim)
        xk = xk.view(bsz, slen, self.n_local_kv_heads, self.head_dim)
        xv = xv.view(bsz, slen, self.n_local_kv_heads, self.head_dim)

        xq, xk = apply_rotary_emb(xq, xk, freqs_cos, freqs_sin)

        xk = repeat_kv(xk, self.n_rep)
        xv = repeat_kv(xv, self.n_rep)

        xq = xq.transpose(1, 2)
        xk = xk.transpose(1, 2)
        xv = xv.transpose(1, 2)

        if self.flash:
            output = torch.nn.functional.scaled_dot_product_attention(
                xq, xk, xv,
                attn_mask=None,
                dropout_p=self.dropout if self.training else 0.0,
                is_causal=True,
            )
        else:
            scores = torch.matmul(xq, xk.transpose(2, 3)) / math.sqrt(self.head_dim)
            assert hasattr(self, "mask")
            scores = scores + self.mask[:, :, :slen, :slen]
            scores = torch.softmax(scores.float(), dim=-1).type_as(xq)
            scores = self.attn_dropout(scores)
            output = torch.matmul(scores, xv)

        output = output.transpose(1, 2).contiguous().view(bsz, slen, -1)
        output = self.resid_dropout(self.wo(output))
        return output


class MLP(nn.Module):
    """前馈网络，使用 SwiGLU 激活函数"""

    def __init__(self, dim: int, hidden_dim: int, multiple_of: int, dropout: float):
        super().__init__()
        if hidden_dim is None:
            hidden_dim = 4 * dim
            hidden_dim = int(2 * hidden_dim / 3)
            hidden_dim = multiple_of * ((hidden_dim + multiple_of - 1) // multiple_of)

        self.w1 = nn.Linear(dim, hidden_dim, bias=False)
        self.w2 = nn.Linear(hidden_dim, dim, bias=False)
        self.w3 = nn.Linear(dim, hidden_dim, bias=False)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor):
        # SwiGLU: w2(silu(w1(x)) * w3(x))
        return self.dropout(self.w2(F.silu(self.w1(x)) * self.w3(x)))


class DecoderLayer(nn.Module):
    """单个 Transformer 解码器层（Pre-Norm 结构）"""

    def __init__(self, layer_id: int, args: ModelConfig):
        super().__init__()
        self.layer_id = layer_id
        self.attention = Attention(args)
        self.mlp = MLP(args.dim, args.hidden_dim, args.multiple_of, args.dropout)
        self.attention_norm = RMSNorm(args.dim, eps=args.norm_eps)
        self.ffn_norm = RMSNorm(args.dim, eps=args.norm_eps)

    def forward(self, x: torch.Tensor, freqs_cos: torch.Tensor, freqs_sin: torch.Tensor):
        # Pre-Norm: 先归一化，再做 attention/MLP，再残差连接
        h = x + self.attention(self.attention_norm(x), freqs_cos, freqs_sin)
        out = h + self.mlp(self.ffn_norm(h))
        return out


class Transformer(PreTrainedModel):
    """
    基于 LLaMA 架构的 Transformer 语言模型

    特性：
    - 旋转位置编码（RoPE）
    - 分组查询注意力（GQA）
    - SwiGLU 前馈激活
    - Pre-Norm（RMSNorm）
    - 词嵌入与输出层权重共享
    """

    config_class = ModelConfig

    def __init__(self, args: ModelConfig):
        super().__init__(args)
        self.args = args
        self.vocab_size = args.vocab_size
        self.n_layers = args.n_layers

        self.tok_embeddings = nn.Embedding(args.vocab_size, args.dim)
        self.dropout = nn.Dropout(args.dropout)

        self.layers = nn.ModuleList(
            [DecoderLayer(layer_id, args) for layer_id in range(args.n_layers)]
        )

        self.norm = RMSNorm(args.dim, eps=args.norm_eps)
        self.output = nn.Linear(args.dim, args.vocab_size, bias=False)

        # 词嵌入与 LM head 权重共享
        self.tok_embeddings.weight = self.output.weight

        # 预计算 RoPE 频率
        freqs_cos, freqs_sin = precompute_freqs_cis(
            args.dim // args.n_heads, args.max_seq_len
        )
        self.register_buffer("freqs_cos", freqs_cos)
        self.register_buffer("freqs_sin", freqs_sin)

        # 权重初始化
        self.apply(self._init_weights)
        # 对残差连接的输出投影使用缩放初始化
        for pn, p in self.named_parameters():
            if pn.endswith("w2.weight") or pn.endswith("wo.weight"):
                torch.nn.init.normal_(p, mean=0.0, std=0.02 / math.sqrt(2 * self.n_layers))

        # 输出缓存（复用对象避免频繁分配）
        self.last_loss: Optional[torch.Tensor] = None
        self.OUT = CausalLMOutputWithPast()
        self._no_split_modules = [name for name, _ in self.named_modules()]

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)   # FIX: 原代码缺少第二个参数
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(
        self,
        tokens: torch.Tensor,
        targets: Optional[torch.Tensor] = None,
        **kwargs,
    ):
        """
        前向传播

        Args:
            tokens: 输入 token 索引张量，shape (batch, seq_len)
            targets: 目标 token 张量，shape (batch, seq_len)，训练时提供

        Returns:
            CausalLMOutputWithPast，包含 logits 和 loss（训练时）
        """
        # 兼容 HuggingFace 接口参数名
        if "input_ids" in kwargs:
            tokens = kwargs["input_ids"]
        if "labels" in kwargs:
            targets = kwargs["labels"]

        _bsz, seqlen = tokens.shape
        h = self.tok_embeddings(tokens)
        h = self.dropout(h)

        freqs_cos = self.freqs_cos[:seqlen]
        freqs_sin = self.freqs_sin[:seqlen]

        for layer in self.layers:
            h = layer(h, freqs_cos, freqs_sin)

        h = self.norm(h)

        if targets is not None:
            # 训练模式：对所有位置计算 logits 和 loss
            logits = self.output(h)
            self.last_loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)),
                targets.view(-1),
                ignore_index=0,
                reduction="none",
            )
        else:
            # 推理模式：只计算最后一个 token 的 logits
            logits = self.output(h[:, [-1], :])
            self.last_loss = None

        self.OUT.__setitem__("logits", logits)
        self.OUT.__setitem__("loss", self.last_loss)
        return self.OUT

    @torch.inference_mode()
    def generate(
        self,
        idx: torch.Tensor,
        stop_id: Optional[int] = None,
        max_new_tokens: int = 256,
        temperature: float = 1.0,
        top_k: Optional[int] = None,
    ) -> torch.Tensor:
        """
        自回归文本生成（无 KV Cache，适用于短序列推理）

        Args:
            idx: 输入 token 索引，shape (batch, seq_len)
            stop_id: 停止生成的 token ID
            max_new_tokens: 最多生成的 token 数
            temperature: 采样温度，0 表示贪心解码
            top_k: Top-K 采样的 K 值，None 表示不限制

        Returns:
            新生成的 token 序列，shape (batch, new_tokens)
        """
        index = idx.shape[1]
        for _ in range(max_new_tokens):
            # 超出最大长度时截断
            idx_cond = idx if idx.size(1) <= self.args.max_seq_len else idx[:, -self.args.max_seq_len:]

            logits = self(idx_cond).logits
            logits = logits[:, -1, :]  # 取最后一个位置

            if temperature == 0.0:
                # 贪心解码
                _, idx_next = torch.topk(logits, k=1, dim=-1)
            else:
                logits = logits / temperature
                if top_k is not None:
                    v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                    logits[logits < v[:, [-1]]] = float("-inf")
                probs = F.softmax(logits, dim=-1)
                idx_next = torch.multinomial(probs, num_samples=1)

            if stop_id is not None and idx_next.item() == stop_id:
                break
            idx = torch.cat((idx, idx_next), dim=1)

        return idx[:, index:]
