"""Vital 模型与多语言 BPE 的兼容入口。

分词器实现在 parts/tokenizer.py，原有导入和调用方式保持可用：
    train_bpe(texts, vocab_size=16384, min_frequency=2)
    load_bpe(path)
    bpe_tokenise(text, tokenizer=None, add_bos=False, add_eos=False)
    BPETokenizer().train(texts).encode(text)

BPETokenizer 提供 encode_batch、decode、save、load，以及 vocab_size、vocab、merges。
encode/encode_batch 的 allow_special_tokens=False 将角色标记字符串视为普通文字。
默认保留旧行为：识别 <pad>、<bos>、<eos>、<assistant>，编号固定为 256～259。
所有有效 Unicode 文本都有字节兜底；空格、换行、大小写和组合字符原样保留。
词表训练后应保存并复用，模型权重、词元缓存与对应分词器版本必须配套。
"""

from __future__ import annotations

import torch
import torch.nn as nn

from parts.config import attention, dimension_word
from parts.decoder import Decoder
from parts.tokenizer import BPETokenizer, RuleTokenizer, bpe_tokenise, load_bpe, train_bpe


class Vital(nn.Module):

    vocab_chart : nn.Embedding
    decoder_1 : Decoder
    decoder_2 : Decoder
    decoder_3 : Decoder
    decoder_4 : Decoder
    decoder_5 : Decoder
    decoder_6 : Decoder

    def __init__(self,vocab_size : int):
        super(Vital, self).__init__()
        #创建嵌入矩阵
        self.vocab_chart = nn.Embedding(num_embeddings=vocab_size, embedding_dim=dimension_word)
        self.decoder_1 = Decoder()
        self.decoder_2 = Decoder()
        self.decoder_3 = Decoder()
        self.decoder_4 = Decoder()
        self.decoder_5 = Decoder()
        self.decoder_6 = Decoder()
        self.norm = nn.LayerNorm(dimension_word)
        self.lm_head = nn.Linear(
            dimension_word,  # 输入：dimension_word 维隐藏向量
            vocab_size,  # 输出：每个词元各一个分数
            bias=False,
        )

    def forward(
        self, x: torch.Tensor, *, last_token_only: bool = True
    ) -> torch.Tensor:
        """输入 [批次数, 序列长度]，返回用于预测下一词元的分数。

        last_token_only=False 返回 [批次数, 序列长度, 词表大小]，用于逐位置训练。
        last_token_only=True 时只返回最后一个非 PAD 位置的分数，
        形状为 [批次数, 词表大小]；在调用处用 argmax 或采样选择编号。
        """
        if last_token_only and (x.ndim != 2 or x.size(1) == 0):
            raise ValueError("只预测下一词元时，输入必须是序列长度非零的二维编号张量")
        # 从原始词元编号识别 PAD：[B, T]，True 表示需要屏蔽的键位置。
        pad_mask = x == BPETokenizer.pad_token_id
        # 无 PAD 时直接使用 SDPA 的因果模式，避免显式分配 [B, 1, T, T] 掩码。
        # 每次前向只检查一次；整篇单样本训练通常走这条路径。
        if not pad_mask.any():
            pad_mask = None
        #将线性词表嵌入矩阵
        token = self.vocab_chart(x)
        #应该调用6次Decoder，暂时只调用2次
        decoded = self.decoder_1(
            self.decoder_6(
            self.decoder_5(
            self.decoder_4(
            self.decoder_3(
                self.decoder_2(token, pad_mask=pad_mask), pad_mask=pad_mask)
            ,pad_mask = pad_mask
        ), pad_mask=pad_mask),pad_mask=pad_mask),pad_mask=pad_mask)
        normalised = self.norm(decoded)

        #之预测下一个token
        if last_token_only:
            positions = torch.arange(x.size(1), device=x.device).expand_as(x)
            last_positions = positions.masked_fill(
                x == BPETokenizer.pad_token_id, -1
            ).max(dim=1).values
            if (last_positions < 0).any():
                raise ValueError("每条输入至少需要一个非 PAD 词元")
            batch_positions = torch.arange(x.size(0), device=x.device)
            normalised = normalised[batch_positions, last_positions]

        #把向量token转化成线性词表
        result = self.lm_head(normalised)
        return result
