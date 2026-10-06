import torch
import torch.nn as nn
import torch.nn.functional as F

import vital


class FNN(nn.Module):
    """逐位置计算的前馈网络，输入和输出均为 dimension_word 维。"""

    def __init__(self):
        super().__init__()
        self.norm = nn.LayerNorm(vital.dimension_word)
        self.fc1 = nn.Linear(vital.dimension_word, 2048)
        self.fc2 = nn.Linear(2048, vital.dimension_word)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        normalised = self.norm(x)
        hidden = F.silu(self.fc1(normalised))
        return F.silu(self.fc2(hidden))


class attention(nn.Module):
    def __init__(self):
        super(attention,self).__init__()
        self.norm = nn.LayerNorm(vital.dimension_word)
        self.query_fc = nn.Linear(vital.dimension_word, vital.dimension_word)
        self.key_fc = nn.Linear(vital.dimension_word, vital.dimension_word)
        self.value_fc = nn.Linear(vital.dimension_word, vital.dimension_word)
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        normalised = self.norm(x)
        query = self.query_fc(x)
        key = self.key_fc(x)
        value = self.value_fc(x)
        return normalised


class Decoder(nn.Module):
    attention_module : attention
    fnn: FNN

    def __init__(self):
        super(Decoder,self).__init__()
        self.attention_module = attention()
        self.fnn = FNN()
        self.norm = nn.LayerNorm(vital.dimension_word)
    def forward(self,x : torch.Tensor) -> torch.Tensor:
        hidden = self.attention_module.forward(x)
        result = self.norm(self.fnn(hidden))
        return result
