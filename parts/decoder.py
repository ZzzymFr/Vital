import torch
import torch.nn as nn
import torch.nn.functional as F
import math

from parts.config import attention as num_heads, dimension_word


class FNN(nn.Module):
    """逐位置计算的前馈网络，输入和输出均为 dimension_word 维。"""

    def __init__(self):
        super().__init__()
        self.norm = nn.LayerNorm(dimension_word)
        self.fc1 = nn.Linear(dimension_word, 2048)
        self.hid = nn.Linear(2048, 2048)
        self.fc2 = nn.Linear(2048, dimension_word)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        normalised = self.norm(x)
        hidden = F.silu(self.fc1(normalised))
        # SiLU 的函数接口与上一层相同；保留新增隐藏层及其参数。
        hidd = F.silu(self.hid(hidden))
        return self.fc2(hidd) + x


class attention(nn.Module):
    def __init__(self):
        super(attention,self).__init__()
        self.norm = nn.LayerNorm(dimension_word)
        self.query_fc = nn.Linear(dimension_word, dimension_word)
        self.key_fc = nn.Linear(dimension_word, dimension_word)
        self.value_fc = nn.Linear(dimension_word, dimension_word)
        self.attention_fc = nn.Linear(dimension_word, dimension_word)
    def forward(
        self, x: torch.Tensor, pad_mask: torch.Tensor | None = None
    ) -> torch.Tensor:
        # pad_mask：[B, T] 的布尔张量，True 表示该位置是 PAD；省略则只做因果遮罩。
        #先做归一
        residual = x
        x = self.norm(x)
        #获得q,k,v的值
        query = self.query_fc(x)
        key = self.key_fc(x)
        value = self.value_fc(x)
        # [序列编号, 词元位置, 词维] -> [序列编号, 注意力头编号, 词元位置, 头内特征]
        head_dim = dimension_word // num_heads
        query = query.view(query.size(0), -1, num_heads, head_dim).transpose(1, 2)
        key = key.view(key.size(0), -1, num_heads, head_dim).transpose(1, 2)
        value = value.view(value.size(0), -1, num_heads, head_dim).transpose(1, 2)

        # RoPE：B 是序列数，T 是当前输入长度，head_dim 是每个头的特征维度。
        # query、key 为 [B, 头数, T, head_dim]；旋转它们，value 保持不变。
        if head_dim % 2 != 0:
            raise ValueError("RoPE 要求每个注意力头的维度为偶数")
        # positions：[T, 1]，当前整段前缀的位置编号为 0 到 T-1。
        # 将来使用 KV 缓存时，需要加上已缓存词元数，不能每次都从 0 开始。
        positions = torch.arange(
            query.size(2), device=query.device, dtype=torch.float32
        ).unsqueeze(1)
        # inv_freq：[head_dim/2]，每对相邻特征使用一个固定角频率。
        # 第 r 对对应维度 (2r, 2r+1)，频率为 10000^(-2r/head_dim)，无需训练。
        inv_freq = 10000.0 ** (
            -torch.arange(0, head_dim, 2, device=query.device, dtype=torch.float32)
            / head_dim
        )
        # angles：[T, head_dim/2]，位置 t 的第 r 对特征旋转 t * inv_freq[r] 弧度。
        angles = positions * inv_freq
        # 先用 float32 计算角度及三角函数，再转回 query 的数据类型。
        # cos、sin：[1, 1, T, head_dim/2]，前两维广播到所有序列和注意力头。
        cos = angles.cos()[None, None, :, :].to(dtype=query.dtype)
        sin = angles.sin()[None, None, :, :].to(dtype=query.dtype)

        # 按下标拆出偶数维和奇数维，每个切片为 [B, 头数, T, head_dim/2]。
        query_even, query_odd = query[..., 0::2], query[..., 1::2]
        key_even, key_odd = key[..., 0::2], key[..., 1::2]
        # 每对相邻特征 (a, b) -> (a*cos - b*sin, a*sin + b*cos)。
        # stack 先得到 [B, 头数, T, head_dim/2, 2]；flatten 合并最后两维，
        # 将每对旋转结果交错排回原位置，恢复 [B, 头数, T, head_dim]。
        query = torch.stack(
            (
                query_even * cos - query_odd * sin,
                query_even * sin + query_odd * cos,
            ),
            dim=-1,
        ).flatten(-2)
        key = torch.stack(
            (
                key_even * cos - key_odd * sin,
                key_even * sin + key_odd * cos,
            ),
            dim=-1,
        ).flatten(-2)

        # 原手动注意力实现保留为注释，便于对照 SDPA 的数学含义。
        # 每个查询向量与所有键向量分别做点积，并除以 sqrt(head_dim)。
        # [B, H, T, D] @ [B, H, D, T] -> [B, H, T, T]
        # scores = (query @ key.transpose(-2, -1)) / math.sqrt(head_dim)
        # T = query.size(-2)
        # [T, T]：严格上三角为 True，表示未来位置。
        # diagonal=1 保留当前位置自身
        # future_mask = torch.ones(T,T,device=query.device, dtype=torch.bool).triu(diagonal=1)
        # if pad_mask is not None:
        #     # PAD 遮罩扩展为 [B, 1, 1, T]，屏蔽被读取的键位置（最后一维）。
        #     # 与因果遮罩合并为 [B, 1, T, T]，广播到所有注意力头。
        #     future_mask = future_mask[None, None, :, :] | pad_mask[:, None, None, :]
        #
        # 未来位置或 PAD 位置的分数设为负无穷。
        # scores = scores.masked_fill(future_mask, float("-inf"))
        #
        # 左侧 PAD 的查询可能没有任何可读取的键；整行 -inf 做 softmax 会产生 NaN。
        # 先将这种行的分数暂设为 0，softmax 后再把整行权重清零。
        # fully_masked = future_mask.all(dim=-1, keepdim=True)
        # scores = scores.masked_fill(fully_masked, 0.0)
        # 沿最后一维，即“被读取的位置”，计算注意力权重。
        # weights = torch.softmax(scores, dim=-1)
        # weights = weights.masked_fill(fully_masked, 0.0)
        # 加权输出
        # output = weights @ value

        # SDPA 输入/输出均为 [B, H, T, D]：B 是批次，H 是头数，T 是长度，D 是头维度。
        # 内部完成缩放、softmax 和加权求和；由 PyTorch 按设备/精度选择可用内核。
        T = query.size(-2)
        if pad_mask is None:
            output = F.scaled_dot_product_attention(
                query, key, value, dropout_p=0.0, is_causal=True,
            )
        else:
            # SDPA 的布尔遮罩 True 表示“允许读取”，与上方旧代码的 True 含义相反。
            # allowed：[B, 1, T, T]，同时允许当前位置及过去位置，且禁止读取 PAD 键。
            causal = torch.ones(T, T, device=query.device, dtype=torch.bool).tril()
            allowed = causal[None, None, :, :] & ~pad_mask[:, None, None, :]
            # 左侧 PAD/全 PAD 行可能没有合法键；临时允许第 0 列，计算后将该行输出清零。
            # 这样各 SDPA 后端都无需处理全被遮住的 softmax，反向梯度也保持有限。
            fully_masked = ~allowed.any(dim=-1, keepdim=True)
            allowed[..., :1] |= fully_masked
            output = F.scaled_dot_product_attention(
                query, key, value, attn_mask=allowed, dropout_p=0.0,
            )
            output = output.masked_fill(fully_masked, 0.0)
        # 原实现没有注意力 dropout，因此训练和推理都明确传入 dropout_p=0.0。
        # 将所有注意力头合并回 [B, T, dimension_word]，保留原投影和残差连接。
        B = query.size(0)
        merged_output = output.transpose(1,2).contiguous().reshape(B,T,dimension_word)
        attention_output = self.attention_fc(merged_output)
        return residual + attention_output



class Decoder(nn.Module):
    attention_module : attention
    fnn: FNN

    def __init__(self):
        super(Decoder,self).__init__()
        self.attention_module = attention()
        self.fnn = FNN()
    def forward(
        self, x: torch.Tensor, pad_mask: torch.Tensor | None = None
    ) -> torch.Tensor:
        # 每个 Decoder 都传递同一份 PAD 位置，防止后续层重新读取 PAD。
        hidden = self.attention_module(x, pad_mask=pad_mask)
        result = self.fnn.forward(hidden)
        return result
