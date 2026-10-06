"""BPE 分词器：先训练或加载词表，再把文字转换成整数编号。

分词器部分只依赖 Python 标准库，已有的神经网络代码保留在文件末尾。

BPETokenizer 的公开接口：
    BPETokenizer()
        创建分词器，初始词表包含 256 个字节词元和 4 个特殊词元。
        此时尚无 BPE 合并规则，但已能按原始字节编码和解码文字。

    tokenizer.train(texts, vocab_size=512, min_frequency=2) -> BPETokenizer
        从零训练词表和合并规则，成功后替换当前状态，并返回自身。
        texts：一段字符串，或可迭代的多段字符串；不跨文本边界合并。
        vocab_size：包含特殊词元的词表大小上限，必须为至少 260 的整数。
        min_frequency：允许合并的相邻词元对的最低出现次数，必须为正整数。
        语料不足时，实际词表可能小于指定上限。

    tokenizer.encode(text, *, add_bos=False, add_eos=False) -> list[int]
        将字符串转换为词元编号列表；文本中的特殊标记会编码为独立编号。
        add_bos：是否在开头加入 BOS；add_eos：是否在末尾加入 EOS。
        这两个选项只能通过关键字传入；此方法不会自动补 PAD。
        例如 encode("<assistant>你好") 的第一个编号固定为 259。

    tokenizer.decode(token_ids, *, skip_special_tokens=True, errors="replace") -> str
        将可迭代的整数编号先还原并拼接为字节，再解码为文字。
        skip_special_tokens：是否跳过 PAD、BOS、EOS、助手标记。
        设置为 False 时，助手编号还原为字符串 "<assistant>"。
        errors：UTF-8 解码错误的处理方式；默认用替代字符处理非法字节，
        设置为 "strict" 时会对非法或不完整的 UTF-8 字节报错。

    tokenizer.save(path) -> None
        将词表、特殊词元和合并规则保存为 JSON；自动创建父目录。
        path 可以是字符串或 pathlib.Path。

    BPETokenizer.load(path) -> BPETokenizer
        类方法：加载并校验 JSON，返回一个新的分词器实例。
        path 可以是字符串或 pathlib.Path。

公开属性和常量：
    tokenizer.vocab_size：实际词表大小，包含特殊词元；只读属性。
    tokenizer.vocab：词元编号到 bytes 内容的字典副本；只读属性。
    tokenizer.merges：按学习顺序排列的合并规则列表副本；只读属性。
        每条规则为 (左词元编号, 右词元编号, 新词元编号)。
    tokenizer.pad_token_id = 256：补齐长度的占位词元 PAD。
    tokenizer.bos_token_id = 257：序列开始词元 BOS。
    tokenizer.eos_token_id = 258：序列结束词元 EOS。
    tokenizer.assistant_token_id = 259：回答开始的助手标记。
    BPETokenizer.ASSISTANT_TOKEN = "<assistant>"：助手标记的文本写法。
    BPETokenizer.SPECIAL_TOKENS：特殊词元名称到编号的映射。
    BPETokenizer.MIN_VOCAB_SIZE = 260：包含基础字节和特殊词元的最小词表大小。
    以上属性和常量直接访问，不需要加括号；_initial_vocab、_split_text、_merge 为内部辅助方法。

模块级便捷函数（不属于 BPETokenizer 的实例方法）：
    train_bpe(texts, vocab_size=512, min_frequency=2) -> BPETokenizer
        训练并设置默认分词器，同时返回该实例。
    load_bpe(path) -> BPETokenizer
        加载并设置默认分词器，同时返回该实例。
    bpe_tokenise(text, tokenizer=None, *, add_bos=False, add_eos=False) -> list[int]
        使用显式传入的分词器，或 train_bpe/load_bpe 设置的默认分词器编码。

使用示例：
    tokenizer = train_bpe(["我喜欢猫。" * 10], vocab_size=320)
    ids = bpe_tokenise("我喜欢猫，也喜欢狗。", add_eos=True)
    text = tokenizer.decode(ids)
    tokenizer.save("tokenizer.json")
    loaded = load_bpe("tokenizer.json")
    assert bpe_tokenise("猫") == loaded.encode("猫")

输出与存储：
    encode/bpe_tokenise 返回 list[int]，结果保存在调用处的 ids 等变量中。
    分词器不会自动保存各条文本的编号列表，也不会生成神经网络嵌入向量。
    tokenizer.vocab、tokenizer.merges 可查看内存中的词表与合并规则。
    save(path) 只将词表、特殊词元和合并规则写入指定 JSON 文件。
    将编号映射为浮点嵌入向量由 Vital.vocab_chart 嵌入层完成。
    新增助手标记后，旧词表需要重新训练并保存，不能直接复用旧合并编号。

这是基础字节级 BPE，适合学习和小语料，没有 GPT 分词器的预分词规则。
"""

from __future__ import annotations

import json
import re
from collections import Counter
from collections.abc import Iterable, Sequence
from pathlib import Path

import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

from parts.decoder import Decoder

dimension_word = 1024
attention = 16


class BPETokenizer:
    """可训练、可编码解码、可保存加载的字节级 BPE。

    编号 0～255 对应原始字节，256～259 为特殊词元，260 起为合并词元。
    特殊标记不会参与 BPE 合并，也不会被拆成普通字节词元。
    一个词元可能只包含汉字的部分字节，解码时必须先拼接字节再解码。

    示例：
        tokenizer = BPETokenizer().train(["我喜欢猫。" * 10], vocab_size=320)
        ids = tokenizer.encode("我喜欢猫，也喜欢狗。", add_eos=True)
        assert tokenizer.decode(ids) == "我喜欢猫，也喜欢狗。"
        tokenizer.save("tokenizer.json")
        restored = BPETokenizer.load("tokenizer.json")
        assert restored.encode("猫") == tokenizer.encode("猫")
    """

    pad_token_id = 256
    bos_token_id = 257
    eos_token_id = 258
    assistant_token_id = 259
    ASSISTANT_TOKEN = "<assistant>"
    SPECIAL_TOKENS = {
        "<pad>": pad_token_id,
        "<bos>": bos_token_id,
        "<eos>": eos_token_id,
        ASSISTANT_TOKEN: assistant_token_id,
    }
    MIN_VOCAB_SIZE = 260

    @classmethod
    def _split_text(cls, text: str) -> list[str]:
        """分隔普通文本和特殊标记，禁止 BPE 合并跨越角色等边界。"""
        pattern = "(" + "|".join(re.escape(token) for token in cls.SPECIAL_TOKENS) + ")"
        return [piece for piece in re.split(pattern, text) if piece]

    def __init__(self) -> None:
        self._vocab = self._initial_vocab()
        # 每条规则保存：左词元编号、右词元编号、合并后的新编号。
        self._merges: list[tuple[int, int, int]] = []
        # 相邻编号对 -> (学习顺序, 新编号)，用于选择最早学到的规则。
        self._rules: dict[tuple[int, int], tuple[int, int]] = {}

    @classmethod
    def _initial_vocab(cls) -> dict[int, bytes]:
        vocab = {token_id: bytes([token_id]) for token_id in range(256)}
        vocab.update(
            {token_id: token.encode("utf-8") for token, token_id in cls.SPECIAL_TOKENS.items()}
        )
        return vocab

    @property
    def vocab_size(self) -> int:
        """实际词表大小，包含特殊词元；可作为嵌入层的词表大小。"""
        return len(self._vocab)

    @property
    def vocab(self) -> dict[int, bytes]:
        """返回词元编号到字节内容的副本，防止外部修改内部词表。"""
        return self._vocab.copy()

    @property
    def merges(self) -> list[tuple[int, int, int]]:
        """返回按学习顺序排列的合并规则副本。"""
        return self._merges.copy()

    @staticmethod
    def _merge(tokens: Sequence[int], pair: tuple[int, int], new_id: int) -> list[int]:
        """从左到右合并所有不重叠的指定编号对。"""
        merged: list[int] = []
        position = 0
        while position < len(tokens):
            if (
                position + 1 < len(tokens)
                and tokens[position] == pair[0]
                and tokens[position + 1] == pair[1]
            ):
                merged.append(new_id)
                position += 2
            else:
                merged.append(tokens[position])
                position += 1
        return merged

    def train(
        self,
        texts: str | Iterable[str],
        vocab_size: int = 512,
        min_frequency: int = 2,
    ) -> BPETokenizer:
        """从零训练并返回自身；vocab_size 是包含特殊词元的词表上限。

        texts 可以是一段文字或多条文字；不会跨两条文字的边界合并。
        若已没有足够频繁的编号对，实际词表可能小于指定上限。
        """
        if type(vocab_size) is not int or vocab_size < self.MIN_VOCAB_SIZE:
            raise ValueError(f"vocab_size 必须是至少 {self.MIN_VOCAB_SIZE} 的整数")
        if type(min_frequency) is not int or min_frequency < 1:
            raise ValueError("min_frequency 必须是正整数")

        documents = [texts] if isinstance(texts, str) else texts
        sequences: list[list[int]] = []
        has_text = False
        for text in documents:
            if not isinstance(text, str):
                raise TypeError("训练语料中的每条内容都必须是字符串")
            if text:
                has_text = True
                sequences.extend(
                    list(piece.encode("utf-8"))
                    for piece in self._split_text(text)
                    if piece not in self.SPECIAL_TOKENS
                )
        if not has_text:
            raise ValueError("训练语料至少需要一条非空文字")

        # 先使用局部变量训练，成功后再整体替换当前分词器的状态。
        vocab = self._initial_vocab()
        merges: list[tuple[int, int, int]] = []
        while len(vocab) < vocab_size:
            pair_counts: Counter[tuple[int, int]] = Counter()
            for sequence in sequences:
                pair_counts.update(zip(sequence, sequence[1:]))
            if not pair_counts:
                break

            # 出现次数最多的对优先；次数相同则按编号排序，保证可复现。
            pair = min(pair_counts, key=lambda item: (-pair_counts[item], item))
            if pair_counts[pair] < min_frequency:
                break

            new_id = len(vocab)
            vocab[new_id] = vocab[pair[0]] + vocab[pair[1]]
            merges.append((pair[0], pair[1], new_id))
            sequences = [self._merge(sequence, pair, new_id) for sequence in sequences]

        self._vocab = vocab
        self._merges = merges
        self._rules = {
            (left, right): (rank, new_id)
            for rank, (left, right, new_id) in enumerate(merges)
        }
        return self

    def encode(
        self, text: str, *, add_bos: bool = False, add_eos: bool = False
    ) -> list[int]:
        """编码普通文字与特殊标记；add_bos/add_eos 可额外添加首尾标记。"""
        if not isinstance(text, str):
            raise TypeError("text 必须是字符串")
        tokens: list[int] = []
        for piece in self._split_text(text):
            if piece in self.SPECIAL_TOKENS:
                tokens.append(self.SPECIAL_TOKENS[piece])
                continue
            piece_tokens = list(piece.encode("utf-8"))
            while len(piece_tokens) > 1:
                candidates = (
                    pair for pair in zip(piece_tokens, piece_tokens[1:]) if pair in self._rules
                )
                pair = min(candidates, key=lambda item: self._rules[item][0], default=None)
                if pair is None:
                    break
                piece_tokens = self._merge(piece_tokens, pair, self._rules[pair][1])
            tokens.extend(piece_tokens)
        if add_bos:
            tokens.insert(0, self.bos_token_id)
        if add_eos:
            tokens.append(self.eos_token_id)
        return tokens

    def decode(
        self,
        token_ids: Iterable[int],
        *,
        skip_special_tokens: bool = True,
        errors: str = "replace",
    ) -> str:
        """将编号还原成文字；errors='strict' 可检查不完整的 UTF-8 输出。"""
        pieces: list[bytes] = []
        special_ids = set(self.SPECIAL_TOKENS.values())
        for token_id in token_ids:
            if type(token_id) is not int or token_id not in self._vocab:
                raise ValueError(f"未知或非法的词元编号：{token_id!r}")
            if skip_special_tokens and token_id in special_ids:
                continue
            pieces.append(self._vocab[token_id])
        return b"".join(pieces).decode("utf-8", errors=errors)

    def save(self, path: str | Path) -> None:
        """保存词表、特殊标记和规则；字节用十六进制存储以避免编码损失。"""
        data = {
            "format": "byte-level-bpe",
            "version": 1,
            "special_tokens": self.SPECIAL_TOKENS,
            "vocab": {str(token_id): value.hex() for token_id, value in self._vocab.items()},
            "merges": self._merges,
        }
        destination = Path(path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(
            json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )

    @classmethod
    def load(cls, path: str | Path) -> BPETokenizer:
        """加载并校验分词器，确保规则和编号对应关系没有改变。"""
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        if (
            not isinstance(data, dict)
            or data.get("format") != "byte-level-bpe"
            or type(data.get("version")) is not int
            or data.get("version") != 1
        ):
            raise ValueError("不是支持的 BPE 分词器文件")
        if data.get("special_tokens") != cls.SPECIAL_TOKENS:
            raise ValueError("特殊词元的编号与当前格式不一致，请重新训练并保存分词器")
        records = data.get("merges")
        if not isinstance(records, list):
            raise ValueError("分词器文件缺少合并规则列表")

        tokenizer = cls()
        special_ids = set(cls.SPECIAL_TOKENS.values())
        for rank, record in enumerate(records):
            if (
                not isinstance(record, list)
                or len(record) != 3
                or any(type(value) is not int for value in record)
            ):
                raise ValueError(f"第 {rank} 条合并规则格式非法")
            left, right, new_id = record
            if (
                left not in tokenizer._vocab
                or right not in tokenizer._vocab
                or left in special_ids
                or right in special_ids
                or new_id != len(tokenizer._vocab)
                or (left, right) in tokenizer._rules
            ):
                raise ValueError(f"第 {rank} 条合并规则引用了非法编号")
            tokenizer._vocab[new_id] = tokenizer._vocab[left] + tokenizer._vocab[right]
            tokenizer._merges.append((left, right, new_id))
            tokenizer._rules[(left, right)] = (rank, new_id)

        expected_vocab = {
            str(token_id): value.hex() for token_id, value in tokenizer._vocab.items()
        }
        if data.get("vocab") != expected_vocab:
            raise ValueError("保存的词表与合并规则不一致")
        return tokenizer

_default_bpe_tokenizer: BPETokenizer | None = None


def train_bpe(
    texts: str | Iterable[str], vocab_size: int = 512, min_frequency: int = 2
) -> BPETokenizer:
    """训练默认分词器，并返回实例供查看词表、解码和保存。"""
    global _default_bpe_tokenizer
    tokenizer = BPETokenizer().train(texts, vocab_size, min_frequency)
    _default_bpe_tokenizer = tokenizer
    return tokenizer


def load_bpe(path: str | Path) -> BPETokenizer:
    """加载保存的分词器，作为 bpe_tokenise 的默认分词器。"""
    global _default_bpe_tokenizer
    tokenizer = BPETokenizer.load(path)
    _default_bpe_tokenizer = tokenizer
    return tokenizer


def bpe_tokenise(
    text: str,
    tokenizer: BPETokenizer | None = None,
    *,
    add_bos: bool = False,
    add_eos: bool = False,
) -> list[int]:
    """把文字转换为词元编号；先调用 train_bpe/load_bpe，或传入分词器。"""
    selected = tokenizer if tokenizer is not None else _default_bpe_tokenizer
    if selected is None:
        raise RuntimeError("请先调用 train_bpe 或 load_bpe，或传入训练过的 tokenizer")
    return selected.encode(text, add_bos=add_bos, add_eos=add_eos)


class Vital(nn.Module):

    vocab_chart : nn.Embedding
    decoder_1 : Decoder
    decoder_2 : Decoder

    def __init__(self,vocab_size : int):
        super(Vital, self).__init__()
        #创建嵌入矩阵
        self.vocab_chart = nn.Embedding(num_embeddings=vocab_size, embedding_dim=dimension_word)
        self.decoder_1 = Decoder()
        self.decoder_2 = Decoder()
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

        默认返回 [批次数, 序列长度, 词表大小]，用于逐位置训练。
        last_token_only=True 时只返回最后一个非 PAD 位置的分数，
        形状为 [批次数, 词表大小]；在调用处用 argmax 或采样选择编号。
        """
        if last_token_only and (x.ndim != 2 or x.size(1) == 0):
            raise ValueError("只预测下一词元时，输入必须是序列长度非零的二维编号张量")
        #将线性词表嵌入矩阵
        token = self.vocab_chart(x)
        #应该调用6次Decoder，暂时只调用2次
        decoded = self.decoder_1(self.decoder_2(token))
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
