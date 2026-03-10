import math
import torch
import torch.nn as nn
import torch.nn.functional as F

# 导入当前项目中的 LLaMA 源码 (ModelConfig 和 Transformer)
from src.model import ModelConfig, Transformer

# =========================================================================
# 1. 纯手工实现的最底层 LoRA 结构
# =========================================================================

class LoraLayer:
    """LoRA 超参数基类，负责保存秩、缩放系数及 Dropout"""
    def __init__(self, r: int, lora_alpha: int, lora_dropout: float):
        self.r = r
        self.lora_alpha = lora_alpha
        self.lora_dropout = nn.Dropout(p=lora_dropout) if lora_dropout > 0. else nn.Identity()
        # 归一化缩放系数 scaling
        self.scaling = self.lora_alpha / self.r

class LinearWithLoRA(nn.Linear, LoraLayer):
    """
    带有 LoRA 旁路的线性层。
    继承了系统的 nn.Linear 可以保留原本的 weight 和 bias，
    同时结合自己定义的 lora_A 和 lora_B 矩阵来进行运算。
    """
    def __init__(
        self, 
        in_features: int, 
        out_features: int, 
        r: int = 8, 
        lora_alpha: int = 16, 
        lora_dropout: float = 0.05,
        **kwargs
    ):
        # 1. 初始化标准线性层 (原参数 W0)
        nn.Linear.__init__(self, in_features, out_features, **kwargs)
        # 2. 初始化 LoRA 基类超参数
        LoraLayer.__init__(self, r=r, lora_alpha=lora_alpha, lora_dropout=lora_dropout)
        
        # 3. 定义 LoRA 特有的 A 和 B 低秩矩阵
        if r > 0:
            # A 矩阵负责降维：in_features -> r (r 很小，比如 8)
            self.lora_A = nn.Linear(in_features, r, bias=False)
            # B 矩阵负责升维：r -> out_features
            self.lora_B = nn.Linear(r, out_features, bias=False)
            
            # 根据 LoRA 原理初始化 A 和 B
            self.reset_lora_parameters()
            
            # 冻结原始全连接层权重 W0
            self.weight.requires_grad = False
            if self.bias is not None:
                self.bias.requires_grad = False

    def reset_lora_parameters(self):
        """
        初始化机制严格依据论文原理：
        A 进行高斯分布/Kaiming 随机初始化
        B 进行全零初始化。
        这样可以保证模型在训练刚开始时，旁路 BAx 结果为0，使得模型输出完全等价于预训练基座模型，不破坏原有能力。
        """
        # A 使用 Kaiming 均匀分布
        nn.init.kaiming_uniform_(self.lora_A.weight, a=math.sqrt(5))
        # B 使用 全零初始化
        nn.init.zeros_(self.lora_B.weight)

    def forward(self, x: torch.Tensor):
        """核心前向传播，模拟 h = W0x + BAx"""
        # (1) 主干计算: 获取原始冻结模型的计算结果 -> W0 * x
        result = F.linear(x, self.weight, bias=self.bias)
        
        if self.r > 0:
            # (2) 旁路计算: 计算低秩矩阵结果 -> B * A * dropout(x)
            lora_out = self.lora_B(self.lora_A(self.lora_dropout(x)))
            # (3) 按照 scaling 进行缩放，加到最终结果中
            result += lora_out * self.scaling
            
        return result

# =========================================================================
# 2. 模型入侵与层替换注入 (手写实现，代替 PEFT 库)
# =========================================================================

def inject_lora(model: nn.Module, target_module_names=["wq", "wv"], r=8, lora_alpha=16, lora_dropout=0.05):
    """
    遍历复杂的大模型，寻找名字匹配 target_module_names 的密集层 (一般是注意力中的 Q 和 V)
    并用我们自定义的 LinearWithLoRA 层替换掉它们。
    """
    for name, module in model.named_children():
        # 如果当前 module 正好是个原生的线性层，并且名字在我们制定的替换列表里
        if isinstance(module, nn.Linear) and name in target_module_names:
            
            # ============ 偷梁换柱 ============
            # A. 提取原有层的属性
            in_features = module.in_features
            out_features = module.out_features
            has_bias = module.bias is not None
            
            # B. 实例化出我们带有 LoRA 的混合层
            new_module = LinearWithLoRA(
                in_features, out_features, 
                r=r, lora_alpha=lora_alpha, lora_dropout=lora_dropout,
                bias=has_bias
            )
            
            # C. 把原来的权重和偏置硬复制给新的 LoRA 层里的骨干权重参数，保证预训练知识不丢失
            new_module.weight.data.copy_(module.weight.data)
            if has_bias:
                new_module.bias.data.copy_(module.bias.data)
                
            # D. 利用反射机制将新层注入回去，替换老层
            setattr(model, name, new_module)
            
        else:
            # 如果这是一个容器层 (比如 DecoderLayer 甚至整个 Attention 对象)，就往里面递归寻找
            inject_lora(module, target_module_names, r, lora_alpha, lora_dropout)
            
    return model

def mark_only_lora_as_trainable(model: nn.Module, bias: str = 'none'):
    """
    强制遍历整个模型，将不是 LoRA 参数的梯度全部关掉，仅开放 lora_a 和 lora_b 参与链式法则计算更新。
    """
    for n, p in model.named_parameters():
        if 'lora_' not in n:
            p.requires_grad = False

# =========================================================================
# 3. 演示与学习用的 Main 函数
# =========================================================================

def print_trainable_parameters(model):
    """直观打印可训练参数占的比例"""
    trainable_params, all_param = 0, 0
    for _, param in model.named_parameters():
        num_params = param.numel()
        all_param += num_params
        if param.requires_grad:
            trainable_params += num_params
    print(f"[{model.__class__.__name__}] trainable params: {trainable_params} || all params: {all_param} || trainable%: {100 * trainable_params / all_param:.4f}")

if __name__ == "__main__":
    print("========== 1. 加载原生 LLaMA 架构 ==========")
    # 这里引用项目中真实的 LLaMA Transformer 配置
    config = ModelConfig(
        dim=768,           
        n_layers=6,        
        n_heads=12,        
        vocab_size=32000,
        max_seq_len=512
    )
    # 初始化你们代码库里面的 Transformer
    model = Transformer(config)
    
    print("注入前的网络结构：")
    print(model.layers[0].attention)  # 查看第 0 层的注意力机制
    print_trainable_parameters(model) # 此时全是可训练的 (100%)

    print("\n========== 2. 底层侵入式注入 LoRA ==========")
    # 为模型中所有的 wq 和 wv 密集层插入自定义旁路
    model = inject_lora(
        model, 
        target_module_names=["wq", "wv"], 
        r=8, 
        lora_alpha=16
    )
    
    # 将骨干网络全部冻结，仅留下 LoRA 旁路的梯度
    mark_only_lora_as_trainable(model)

    print("注入后的网络结构（可以发现 wq 和 wv 层已经变成了我们自制的 LinearWithLoRA）：")
    print(model.layers[0].attention)  
    print_trainable_parameters(model) # 此时你会看到可训练参数立刻降到可能不到 1%

    print("\n========== 3. 模拟一次完整的前向 Forward 流水线 ==========")
    dummy_input = torch.randint(0, 32000, (1, 128))  # Batch = 1, Seq_len = 128
    
    # 模拟真实前向传播计算 loss 过程
    # 前向过程走的就是 LinearWithLoRA 里面的 forward(x) = W0x + B(A(x)) 公式
    out = model(dummy_input)
    
    loss = out.loss if out.loss is not None else out.logits.sum()  # 随便构造个损失
    print(f"模拟前向传播结束，获得输出 logits shape: {out.logits.shape}")
    
    print("进行反向传播...")
    loss.backward()
    print("顺利完成了 LoRA 的梯度求导！")
