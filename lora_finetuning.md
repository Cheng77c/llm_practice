# LoRA 微调 (Low-Rank Adaptation) 详解

## 1. LoRA 微调背景
如果一个大模型是将数据映射到高维空间进行处理，在处理一个细分的小任务时，可能不需要复杂的全量大模型参数，仅在某个子空间范围内就可以解决问题。此时也就不需要对全量参数进行优化。

假设当对某个子空间参数进行优化时，能够达到全量参数优化的性能的一定水平（如 90% 精度），那么这个子空间参数矩阵的秩就可以称为对应当前待解决问题的**本征秩（intrinsic rank）**。

预训练模型本身隐式地降低了本征秩，针对特定任务微调后，权重矩阵具有更低的本征秩。同时，越简单的下游任务，对应的本征秩越低。因此，权重更新的部分参数矩阵即使被随机投影到较小的子空间中，仍然可以有效学习（即针对特定下游任务不要求满秩）。我们可以通过优化密集层在适应过程中变化的**秩分解矩阵**来间接训练神经网络中的密集层，实现仅优化秩分解矩阵来达到微调效果。

假设预训练参数为 $\theta_{0}^{D}$，在特定下游任务上对应的本征秩为 $\theta_{d}$，特定下游任务微调参数为 $\theta_{D}$，那么有：
$$ \theta_{D} = \theta_{0}^{D} + \theta_{d}M $$
这里的 $M$ 即为 LoRA 优化的秩分解矩阵。

### LoRA 的优势
相比于其他高效微调方法，LoRA 存在以下优势：
1. **灵活切换**：可以针对不同的下游任务构建小型 LoRA 模块，在共享预训练模型参数基础上有效地切换下游任务。
2. **硬件门槛低**：使用自适应优化器（Adaptive Optimizer），不需要计算梯度或维护大多数参数的优化器状态，训练更高效。
3. **无推理延迟**：使用简单的线性设计，在部署时将可训练矩阵与冻结权重合并，不存在额外的推理延迟。
4. **正交兼容**：LoRA 与其他微调方法正交，可以随意组合使用。

由于其在资源和数据受限情况下的卓越表现，LoRA 已成为目前高效微调 LLM 的主流和首选方法。

---

## 2. LoRA 微调的原理

### 2.1 低秩参数化更新矩阵
LoRA 假设权重更新的过程中也有一个较低的本征秩。对于预训练的权重参数矩阵 $W_0 \in \mathbb{R}^{d \times k}$ ($d$ 为上一层输出维度，$k$ 为下一层输入维度)，使用低秩分解来表示其更新：
$$ W_0 + \Delta W = W_0 + BA $$
其中 $B \in \mathbb{R}^{d \times r}, A \in \mathbb{R}^{r \times k}$，并且秩 $r \ll \min(d, k)$。

在训练过程中：
- $W_0$ 被冻结，不更新。
- $A, B$ 包含可训练参数，被更新。

LoRA 的前向传递函数为：
$$ h = W_0x + \Delta Wx = W_0x + BAx $$

- **初始化策略**：在开始训练时，对 $A$ 使用随机高斯初始化，对 $B$ 使用零初始化。因此训练开始时 $\Delta W = BA = 0$。随后使用 Adam 等优化器进行优化。

### 2.2 应用于 Transformer
在 Transformer 结构中，LoRA 技术主要应用在注意力模块的四个权重矩阵：$W_q$、$W_k$、$W_v$、$W_o$，通常会冻结 MLP 的权重矩阵。
经过消融实验发现，同时调整 $W_q$ 和 $W_v$ 会产生最佳结果。

在此条件下，可训练参数个数为：
$$ \Theta = 2 \times L_{LoRA} \times d_{model} \times r $$
- $L_{LoRA}$：应用 LoRA 的权重矩阵个数
- $d_{model}$：Transformer 的输入输出维度
- $r$：设定的 LoRA 秩（一般取 4、8、16）

---

## 3. LoRA 的代码实现 (基于 PEFT)
目前一般通过 Hugging Face 开发的 `peft` (Parameter-Efficient Fine-Tuning) 库来实现模型的 LoRA 微调。

### 3.1 实现流程
LoRA 微调的内部实现流程主要包括：
1. **确定目标层**：确定要使用 LoRA 的层（`peft` 支持 `nn.Linear`、`nn.Embedding`、`nn.Conv2d` 等）。
2. **替换为 LoRA 层**：对每一个目标层替换为 LoRA 层。在原结果基础上增加一个旁路，通过低秩分解矩阵 A 和 B 模拟参数更新。
3. **微调训练**：冻结原参数，仅更新 LoRA 层参数。

### 3.2 确定 LoRA 层
通过参数 `target_modules` 确定需要微调的层名（字符串列表）：
```python
target_modules = ["q_proj", "v_proj"]
```
通过正则匹配在原模型中寻找对应的层（以组件名正则匹配为例）：
```python
# 找到模型的各个组件中，名字里带"q_proj"，"v_proj"的
target_module_found = re.fullmatch(self.peft_config.target_modules, key)
```

### 3.3 替换 LoRA 层
找到目标层后，创建一个继承自 `nn.Linear` 和 `LoraLayer` 的新对象进行替换。
`LoraLayer` 基类构建了 LoRA 的各种超参：
```python
class LoraLayer:
    def __init__(self, r: int, lora_alpha: int, lora_dropout: float, merge_weights: bool):
        self.r = r
        self.lora_alpha = lora_alpha
        self.lora_dropout = nn.Dropout(p=lora_dropout) if lora_dropout > 0.0 else lambda x: x
        self.merged = False
        self.merge_weights = merge_weights
        self.disable_adapters = False
```

具体的 `Linear` 层实现（旁路矩阵注入）：
```python
class Linear(nn.Linear, LoraLayer):
    def __init__(self, in_features: int, out_features: int, r: int = 0, lora_alpha: int = 1, ...):
        nn.Linear.__init__(self, in_features, out_features, **kwargs)
        LoraLayer.__init__(self, r=r, lora_alpha=lora_alpha, ...)

        if r > 0:
            self.lora_A = nn.Linear(in_features, r, bias=False)  # 降维矩阵 A
            self.lora_B = nn.Linear(r, out_features, bias=False) # 升维矩阵 B
            self.scaling = self.lora_alpha / self.r              # 归一化系数
            self.weight.requires_grad = False                    # 冻结原参数
        self.reset_parameters()                                  # 初始化 A 和 B
```
替换时，直接将原层的 `weight` 和 `bias` 复制给新的 LoRA 层。

### 3.4 微调训练过程中的前向传播 (Forward)
原参数冻结，仅更新 $A$ 和 $B$ 参数。前向计算过程反映了 $h = W_0x + BAx$ 的公式逻辑：
```python
def forward(self, x: torch.Tensor):
    # ... 省略其他分支 ...
    elif self.r > 0 and not self.merged:
        # 计算主干: W_0 * x
        result = F.linear(x, transpose(self.weight, self.fan_in_fan_out), bias=self.bias)
        if self.r > 0:
            # 计算旁路: B * A * dropout(x) * scaling，并加到 result 上
            result += self.lora_B(self.lora_A(self.lora_dropout(x))) * self.scaling
        return result
    # ...
```
这部分代码完美契合了前文公式。在不改变原始大模型权重的情况下，通过线性相加的方式，实现了极其高效的参数更新。
