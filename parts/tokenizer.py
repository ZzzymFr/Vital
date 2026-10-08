"""规则分词与多语言字节级 BPE，兼容旧版词表。

训练或加载后使用 encode/decode；普通文本可指定 allow_special_tokens=False。
0～255 始终是原始字节，PAD/BOS/EOS/ASSISTANT 始终为 256～259。
新版词表保存为版本 2；版本 1 的词表按原合并规则编码，保持旧编号不变。
RuleTokenizer 另用 han-word-rules 格式：汉字逐字、单词分隔，不设词表上限。
本模块不导入 PyTorch，可单独用于大规模数据预处理。
"""

from __future__ import annotations

import json
import re
from collections.abc import Iterable, Sequence
from pathlib import Path

from tokenizers import Regex, Tokenizer, models, pre_tokenizers, trainers


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
    # Unicode 字母和组合标记保持在一起，兼容中文、阿拉伯文、印度文字等。
    # 一个前导空格可并入单词，减少英文词元数；换行和缩进仍原样保留。
    # 数字每组最多三位，不做大小写或 Unicode 归一化。
    PRETOKEN_PATTERN = r" ?[\p{L}\p{M}]+|\p{N}{1,3}| ?[^\s\p{L}\p{M}\p{N}]+|\s+"

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
        self._backend: Tokenizer | None = None
        self._literal_backend: Tokenizer | None = None
        self._backend_to_id: list[int] = []

    @staticmethod
    def _byte_alphabet() -> dict[int, str]:
        """ByteLevel 使用的可逆字节映射；外部编号仍为原始字节值。"""
        visible = list(range(33, 127)) + list(range(161, 173)) + list(range(174, 256))
        remaining = [byte for byte in range(256) if byte not in visible]
        characters = visible + list(range(256, 256 + len(remaining)))
        return dict(zip(visible + remaining, map(chr, characters)))

    @classmethod
    def _pre_tokenizer(cls):
        return pre_tokenizers.Sequence([
            pre_tokenizers.Split(Regex(cls.PRETOKEN_PATTERN), behavior="isolated"),
            pre_tokenizers.ByteLevel(add_prefix_space=False, use_regex=False),
        ])

    def _install_backend(self, backend: Tokenizer) -> None:
        """将后端编号映射到稳定的字节/特殊编号，不改变后端合并规则。"""
        state = json.loads(backend.to_str())
        model = state["model"]
        if model["type"] != "BPE" or model.get("dropout") is not None:
            raise ValueError("词表必须使用确定性的 BPE")
        if any(state.get(key) is not None for key in (
            "normalizer", "post_processor", "padding", "truncation"
        )):
            raise ValueError("分词器不能隐式归一化、插入词元、补齐或截断")
        probe = Tokenizer(models.BPE())
        probe.pre_tokenizer = self._pre_tokenizer()
        expected_pre = json.loads(probe.to_str())["pre_tokenizer"]
        if state.get("pre_tokenizer") != expected_pre:
            raise ValueError("分词器的预分词规则与当前版本不一致")
        added = state["added_tokens"]
        if {token["content"] for token in added} != set(self.SPECIAL_TOKENS):
            raise ValueError("分词器的特殊词元不完整")
        if any(not token["special"] or token["lstrip"] or token["rstrip"]
               or token["normalized"] or token["single_word"] for token in added):
            raise ValueError("特殊词元配置不兼容")
        backend_vocab = model["vocab"]
        if sorted(backend_vocab.values()) != list(range(len(backend_vocab))):
            raise ValueError("后端词元编号必须连续且唯一")
        for token in added:
            if backend_vocab.get(token["content"]) != token["id"]:
                raise ValueError("特殊词元编号与词表不一致")

        alphabet = self._byte_alphabet()
        if not set(alphabet.values()).issubset(backend_vocab):
            raise ValueError("分词器必须包含全部 256 个基础字节")
        character_to_byte = {character: byte for byte, character in alphabet.items()}
        external_ids = {character: byte for byte, character in alphabet.items()}
        external_ids.update(self.SPECIAL_TOKENS)
        vocab = self._initial_vocab()
        backend_to_id = [0] * len(backend_vocab)
        for token, backend_id in sorted(backend_vocab.items(), key=lambda item: item[1]):
            if token not in external_ids:
                external_ids[token] = len(vocab)
                try:
                    vocab[len(vocab)] = bytes(character_to_byte[c] for c in token)
                except KeyError as error:
                    raise ValueError("词表包含非字节级词元") from error
            backend_to_id[backend_id] = external_ids[token]
        merges = [
            (external_ids[left], external_ids[right], external_ids[left + right])
            for left, right in model["merges"]
        ]
        for left, right, new in merges:
            if left in self.SPECIAL_TOKENS.values() or right in self.SPECIAL_TOKENS.values():
                raise ValueError("特殊词元不能参与合并")
            if vocab[left] + vocab[right] != vocab[new]:
                raise ValueError("合并规则与词表不一致")

        # 单独创建普通文字编码器，避免逐次修改共享后端的特殊词元设置。
        state["added_tokens"] = []
        literal_backend = Tokenizer.from_str(json.dumps(state, ensure_ascii=False))
        self._backend = backend
        self._literal_backend = literal_backend
        self._backend_to_id = backend_to_id
        self._vocab = vocab
        self._merges = merges
        self._rules = {
            (left, right): (rank, new) for rank, (left, right, new) in enumerate(merges)
        }

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
        *,
        max_token_length: int = 64,
        show_progress: bool = False,
    ) -> BPETokenizer:
        """从零训练并返回自身；vocab_size 是包含特殊词元的词表上限。

        texts 可以是一段文字或多条文字；不会跨两条文字的边界合并。
        若已没有足够频繁的编号对，实际词表可能小于指定上限。
        """
        if type(vocab_size) is not int or vocab_size < self.MIN_VOCAB_SIZE:
            raise ValueError(f"vocab_size 必须是至少 {self.MIN_VOCAB_SIZE} 的整数")
        if type(min_frequency) is not int or min_frequency < 1:
            raise ValueError("min_frequency 必须是正整数")

        if type(max_token_length) is not int or max_token_length < 2:
            raise ValueError("max_token_length 必须为至少 2 的整数")
        documents = [texts] if isinstance(texts, str) else texts
        has_text = False

        def batches():
            nonlocal has_text
            batch: list[str] = []
            for text in documents:
                if not isinstance(text, str):
                    raise TypeError("训练语料中的每条内容都必须是字符串")
                has_text = has_text or bool(text)
                for piece in self._split_text(text):
                    if piece not in self.SPECIAL_TOKENS:
                        batch.append(piece)
                        if len(batch) >= 1000:
                            yield batch
                            batch = []
            if batch:
                yield batch

        backend = Tokenizer(models.BPE())
        backend.pre_tokenizer = self._pre_tokenizer()
        trainer = trainers.BpeTrainer(
            vocab_size=vocab_size,
            min_frequency=min_frequency,
            special_tokens=list(self.SPECIAL_TOKENS),
            initial_alphabet=pre_tokenizers.ByteLevel.alphabet(),
            max_token_length=max_token_length,
            show_progress=show_progress,
        )
        # Rust 增量维护词频与合并候选；Python 不再保存每篇文章的字节列表。
        # 后端仍需保存不同片段的统计，内存并非与语料规模完全无关。
        backend.train_from_iterator(batches(), trainer=trainer)
        if not has_text:
            raise ValueError("训练语料至少需要一条非空文字")
        self._install_backend(backend)
        return self

    def encode(
        self, text: str, *, add_bos: bool = False, add_eos: bool = False,
        allow_special_tokens: bool = True,
    ) -> list[int]:
        """编码普通文字与特殊标记；add_bos/add_eos 可额外添加首尾标记。"""
        if not isinstance(text, str):
            raise TypeError("text 必须是字符串")
        if self._backend is not None:
            backend = self._backend if allow_special_tokens else self._literal_backend
            token_ids = backend.encode(text, add_special_tokens=False).ids
            tokens = [self._backend_to_id[index] for index in token_ids]
            return self._with_boundaries(tokens, add_bos, add_eos)
        tokens: list[int] = []
        for piece in self._split_text(text) if allow_special_tokens else [text]:
            if allow_special_tokens and piece in self.SPECIAL_TOKENS:
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
        return self._with_boundaries(tokens, add_bos, add_eos)

    def _with_boundaries(self, tokens: list[int], add_bos: bool, add_eos: bool) -> list[int]:
        if add_bos:
            tokens.insert(0, self.bos_token_id)
        if add_eos:
            tokens.append(self.eos_token_id)
        return tokens

    def encode_batch(
        self, texts: Iterable[str], *, add_bos: bool = False, add_eos: bool = False,
        allow_special_tokens: bool = True,
    ) -> list[list[int]]:
        """并行编码一个批次；不补齐、不截断。大语料请分批调用。"""
        if isinstance(texts, str):
            raise TypeError("encode_batch 需要多条字符串，请勿直接传入单个字符串")
        batch = list(texts)
        if any(not isinstance(text, str) for text in batch):
            raise TypeError("每条内容都必须是字符串")
        if self._backend is None:
            return [self.encode(text, add_bos=add_bos, add_eos=add_eos,
                                allow_special_tokens=allow_special_tokens) for text in batch]
        backend = self._backend if allow_special_tokens else self._literal_backend
        return [
            self._with_boundaries([self._backend_to_id[index] for index in encoding.ids],
                                  add_bos, add_eos)
            for encoding in backend.encode_batch(batch, add_special_tokens=False)
        ]

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
        """新词表保存完整后端；旧词表保持十六进制字节及原合并规则。"""
        if self._backend is not None:
            data = {
                "format": "byte-level-bpe",
                "version": 2,
                "special_tokens": self.SPECIAL_TOKENS,
                "backend": json.loads(self._backend.to_str()),
            }
        else:
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
        """自动识别 BPE/规则格式并校验，确保规则和编号对应关系没有改变。"""
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        if isinstance(data, dict) and data.get("format") == "han-word-rules":
            return RuleTokenizer._from_data(data)
        if (
            not isinstance(data, dict)
            or data.get("format") != "byte-level-bpe"
            or type(data.get("version")) is not int
            or data.get("version") not in (1, 2)
        ):
            raise ValueError("不是支持的 BPE 分词器文件")
        if data.get("special_tokens") != cls.SPECIAL_TOKENS:
            raise ValueError("特殊词元的编号与当前格式不一致，请重新训练并保存分词器")
        if data["version"] == 2:
            tokenizer = cls()
            try:
                backend = Tokenizer.from_str(json.dumps(data["backend"], ensure_ascii=False))
                tokenizer._install_backend(backend)
            except (KeyError, TypeError, IndexError) as error:
                raise ValueError("分词器后端数据不完整") from error
            return tokenizer
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


class RuleTokenizer(BPETokenizer):
    """汉字逐字、其他文字按空白/标点分隔；词表不设数量或词长上限。

    建表阶段收集全部实际出现的单元，不进行 BPE 合并或低频过滤。
    编码阶段词表固定：未见单词退回字符，未见字符退回 UTF-8 字节。
    空格、换行、大小写和 Unicode 组合字符全部保留。
    """

    # Han 覆盖 Unicode 汉字（含扩展区）；标点、符号、空白分别保留。
    # 非汉字的连续文字/数字构成单元，例如 café、hello、abc123。
    UNIT_PATTERN = r"\p{Han}|[^\p{Han}\s\p{P}\p{S}\p{C}]+|[\s\S]"

    def __init__(self) -> None:
        super().__init__()
        self._splitter = pre_tokenizers.Split(Regex(self.UNIT_PATTERN), behavior="isolated")
        self._unit_to_id = {bytes([index]): index for index in range(256)}

    def split_units(self, text: str) -> list[str]:
        """返回规则边界，便于检查；词表未见单元的实际编码可能更细。"""
        if not isinstance(text, str):
            raise TypeError("text 必须是字符串")
        return [piece for piece, _ in self._splitter.pre_tokenize_str(text)]

    def train(self, texts: str | Iterable[str], *, show_progress: bool = False) -> RuleTokenizer:
        """从头收集词表；无 vocab_size、min_frequency 或 max_token_length 参数。"""
        candidate = RuleTokenizer()
        documents = [texts] if isinstance(texts, str) else texts
        has_text = False

        def add(unit: str) -> None:
            value = unit.encode("utf-8")
            if value not in candidate._unit_to_id:
                index = len(candidate._vocab)
                candidate._unit_to_id[value] = index
                candidate._vocab[index] = value

        for number, text in enumerate(documents, start=1):
            if not isinstance(text, str):
                raise TypeError("训练语料中的每条内容都必须是字符串")
            has_text = has_text or bool(text)
            for piece in self._split_text(text):
                if piece in self.SPECIAL_TOKENS:
                    continue
                for unit in self.split_units(piece):
                    # 保存字符兜底，新单词不必整体退回字节，也不新增编号。
                    for character in unit:
                        add(character)
                    add(unit)
            if show_progress and number % 10000 == 0:
                print(f"规则词表：已读取 {number} 段文字，当前 {candidate.vocab_size} 个词元")
        if not has_text:
            raise ValueError("训练语料至少需要一条非空文字")
        # 仅在完整读取成功后替换；坏记录不破坏之前的可用词表。
        self.__dict__.update(candidate.__dict__)
        return self

    def encode(
        self, text: str, *, add_bos: bool = False, add_eos: bool = False,
        allow_special_tokens: bool = True,
    ) -> list[int]:
        if not isinstance(text, str):
            raise TypeError("text 必须是字符串")
        tokens: list[int] = []
        for piece in self._split_text(text) if allow_special_tokens else [text]:
            if allow_special_tokens and piece in self.SPECIAL_TOKENS:
                tokens.append(self.SPECIAL_TOKENS[piece])
                continue
            for unit in self.split_units(piece):
                index = self._unit_to_id.get(unit.encode("utf-8"))
                if index is not None:
                    tokens.append(index)
                    continue
                for character in unit:
                    value = character.encode("utf-8")
                    index = self._unit_to_id.get(value)
                    tokens.extend([index] if index is not None else value)
        return self._with_boundaries(tokens, add_bos, add_eos)

    def save(self, path: str | Path) -> None:
        data = {
            "format": "han-word-rules", "version": 1,
            "special_tokens": self.SPECIAL_TOKENS, "pattern": self.UNIT_PATTERN,
            "units": [self._vocab[index].decode("utf-8")
                      for index in range(self.MIN_VOCAB_SIZE, self.vocab_size)],
        }
        destination = Path(path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n",
                               encoding="utf-8")

    @classmethod
    def _from_data(cls, data: dict) -> RuleTokenizer:
        if (type(data.get("version")) is not int or data["version"] != 1
                or data.get("special_tokens") != cls.SPECIAL_TOKENS
                or data.get("pattern") != cls.UNIT_PATTERN
                or not isinstance(data.get("units"), list)):
            raise ValueError("规则分词器的版本、特殊编号、边界或词表不兼容")
        tokenizer = cls()
        for unit in data["units"]:
            if (not isinstance(unit, str) or not unit
                    or tokenizer.split_units(unit) != [unit]):
                raise ValueError("规则词表包含非法或跨边界单元")
            value = unit.encode("utf-8")
            if value in tokenizer._unit_to_id:
                raise ValueError("规则词表包含重复单元")
            index = len(tokenizer._vocab)
            tokenizer._unit_to_id[value] = index
            tokenizer._vocab[index] = value
        return tokenizer

_default_bpe_tokenizer: BPETokenizer | None = None


def train_bpe(
    texts: str | Iterable[str], vocab_size: int = 512, min_frequency: int = 2,
    *, max_token_length: int = 64, show_progress: bool = False,
) -> BPETokenizer:
    """训练默认分词器，并返回实例供查看词表、解码和保存。"""
    global _default_bpe_tokenizer
    tokenizer = BPETokenizer().train(texts, vocab_size, min_frequency,
                                    max_token_length=max_token_length,
                                    show_progress=show_progress)
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
    allow_special_tokens: bool = True,
) -> list[int]:
    """把文字转换为词元编号；先调用 train_bpe/load_bpe，或传入分词器。"""
    selected = tokenizer if tokenizer is not None else _default_bpe_tokenizer
    if selected is None:
        raise RuntimeError("请先调用 train_bpe 或 load_bpe，或传入训练过的 tokenizer")
    return selected.encode(text, add_bos=add_bos, add_eos=add_eos,
                           allow_special_tokens=allow_special_tokens)


