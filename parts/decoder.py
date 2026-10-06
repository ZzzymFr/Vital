import torch
import torch.nn as nn
import torch.nn.functional as F


class FNN(nn.Module):
    """逐位置计算的前馈网络，输入和输出均为 512 维。"""

    def __init__(self):
        super().__init__()
        self.norm = nn.LayerNorm(512)
        self.fc1 = nn.Linear(512, 2048)
        self.fc2 = nn.Linear(2048, 512)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        normalised = self.norm(x)
        hidden = F.silu(self.fc1(normalised))
        return F.silu(self.fc2(hidden))


class attention(nn.Module):
    def __init__(self):
        super(attention,self).__init__()
        self.norm = nn.LayerNorm(512)
        self.query_fc = nn.Linear(512, 512)
        self.key_fc = nn.Linear(512, 512)
        self.value_fc = nn.Linear(512, 512)
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        normalised = self.norm(x)
        query =
        return normalised


class Decoder(nn.Module):
    attention_module : attention
    fnn: FNN

    def __init__(self):
        super(Decoder,self).__init__()
        self.attention_module = attention()
        self.fnn = FNN()
        self.norm = nn.LayerNorm(512)
    def forward(self,x : torch.Tensor) -> torch.Tensor:
        hidden = self.attention_module.forward(x)
        result = self.norm(self.fnn(hidden))
        return result
