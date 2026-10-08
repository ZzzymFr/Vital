"""正文与指令语料的流式读取、下一词元标签和小批次补齐。"""

from __future__ import annotations

from collections.abc import Iterable, Iterator
from itertools import chain, islice
import json
from pathlib import Path
import random
import tempfile

import torch
from torch.utils.data import DataLoader, IterableDataset

from parts.tokenizer import BPETokenizer
from datasets import Dataset, Features, Sequence, Value, interleave_datasets, load_dataset




IGNORE_INDEX = -100
STAGE_FILES = {"pretrain": "pretrain.jsonl", "instruction": "instruction.jsonl"}




def read_records(path: Path, stage: str) -> Iterator[dict[str, str]]:
    """每次读取一条 JSONL；不把整份语料转换成列表或张量。"""
    if stage not in STAGE_FILES:
        raise ValueError(f"未知训练阶段：{stage}")
    with Path(path).open(encoding="utf-8-sig") as source:
        for line_number, line in enumerate(source, start=1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
                if not isinstance(row, dict):
                    raise ValueError("每行必须是 JSON 对象")
                if stage == "pretrain":
                    record = {"text": row["text"]}
                elif "prompt" in row and "answer" in row:
                    record = {"prompt": row["prompt"], "answer": row["answer"]}
                else:
                    instruction, extra = row["instruction"], row.get("input", "")
                    if not isinstance(instruction, str) or not isinstance(extra, str):
                        raise ValueError("instruction 和 input 必须是字符串")
                    record = {
                        "prompt": instruction + ("\n" + extra if extra else ""),
                        "answer": row["output"],
                    }
                if any(not isinstance(value, str) or not value.strip() for value in record.values()):
                    raise ValueError("训练文本、请求和回答必须是非空字符串")
            except (KeyError, TypeError, ValueError) as error:
                raise ValueError(f"{path}:{line_number}：{error}") from error
            yield record


def tokenizer_texts(data_dir: Path) -> Iterator[str]:
    """词表只学习两份训练语料，正文和指令共用一个固定词表。"""
    for row in read_records(data_dir / STAGE_FILES["pretrain"], "pretrain"):
        yield row["text"]
    for row in read_records(data_dir / STAGE_FILES["instruction"], "instruction"):
        yield "用户：" + row["prompt"] + "\n"
        yield row["answer"]


def read_pretrain_records(records) -> Iterator[dict[str, str]]:
    """远程正文只取 text；格式错误时报告记录位置，不静默丢弃语料。"""
    for row_number, row in enumerate(records, start=1):
        try:
            text = row["text"]
            if not isinstance(text, str) or not text.strip():
                raise ValueError("text 必须是非空字符串")
        except (KeyError, TypeError, ValueError) as error:
            raise ValueError(f"远程正文第 {row_number} 条：{error}") from error
        yield {"text": text}


def read_instruction_records(records, excluded_prompts=()) -> Iterator[dict[str, str]]:
    """远程指令只读取 instruction/output，input 即使非空也忽略。"""
    excluded = frozenset(excluded_prompts)
    for row_number, row in enumerate(records, start=1):
        try:
            prompt, answer = row["instruction"], row["output"]
            if any(not isinstance(value, str) or not value.strip() for value in (prompt, answer)):
                raise ValueError("instruction 和 output 必须是非空字符串")
        except (KeyError, TypeError, ValueError) as error:
            raise ValueError(f"远程指令第 {row_number} 条：{error}") from error
        if prompt not in excluded:
            yield {"prompt": prompt, "answer": answer}


def encode_prompt(tokenizer: BPETokenizer, prompt: str) -> list[int]:
    """普通内容按字面编码，由程序插入真正的 BOS 和助手编号。"""
    return [tokenizer.bos_token_id] + tokenizer.encode(
        "用户：" + prompt + "\n", allow_special_tokens=False
    ) + [tokenizer.assistant_token_id]


def _windows(sequence: list[int], targets: list[int], max_seq_len: int | None):
    """默认保留完整序列；显式设置长度时才使用相邻重叠一个词元的窗口。"""
    if max_seq_len is None:
        labels = targets[1:]
        if any(label != IGNORE_INDEX for label in labels):
            yield sequence[:-1], labels
        return
    for start in range(0, len(sequence) - 1, max_seq_len):
        inputs = sequence[start:start + max_seq_len]
        labels = targets[start + 1:start + max_seq_len + 1]
        inputs = inputs[:len(labels)]
        if any(label != IGNORE_INDEX for label in labels):
            yield inputs, labels


def encoded_samples(
    records: Iterable[dict[str, str]], stage: str, tokenizer: BPETokenizer,
    max_seq_len: int | None = None, encode_batch_size: int = 32,
) -> Iterator[tuple[list[int], list[int]]]:
    """每次编码有限条记录；默认完整训练，只有指定长度时才分块。"""
    if (stage not in STAGE_FILES or encode_batch_size < 1
            or (max_seq_len is not None and max_seq_len < 1)):
        raise ValueError("训练阶段无效，或序列长度/编码批次小于 1")
    iterator = iter(records)
    while rows := list(islice(iterator, encode_batch_size)):
        if stage == "pretrain":
            texts = [row["text"] for row in rows]
        else:
            texts = ["用户：" + row["prompt"] + "\n" for row in rows]
            texts += [row["answer"] for row in rows]
        encoded = tokenizer.encode_batch(texts, allow_special_tokens=False)
        for index in range(len(rows)):
            if stage == "pretrain":
                sequence = [tokenizer.bos_token_id] + encoded[index] + [tokenizer.eos_token_id]
                targets = sequence
            else:
                prefix = [tokenizer.bos_token_id] + encoded[index] + [tokenizer.assistant_token_id]
                answer = encoded[len(rows) + index] + [tokenizer.eos_token_id]
                sequence = prefix + answer
                # 请求、BOS 和助手标记只作上下文；损失只覆盖回答及 EOS。
                targets = [IGNORE_INDEX] * len(prefix) + answer
            yield from _windows(sequence, targets, max_seq_len)


def buffered_shuffle(samples, buffer_size: int, seed: int):
    """有限缓冲区洗牌；buffer_size=1 时保持文件顺序。"""
    if buffer_size < 1:
        raise ValueError("洗牌缓冲区大小必须至少为 1")
    rng = random.Random(seed)
    iterator = iter(samples)
    buffer = list(islice(iterator, buffer_size))
    for sample in iterator:
        index = rng.randrange(len(buffer))
        yield buffer[index]
        buffer[index] = sample
    rng.shuffle(buffer)
    yield from buffer


def length_grouped_samples(samples, batch_size: int, bucket_batches: int, seed: int):
    """在有限缓冲区内按输入 token 数组批；只重排样本，不拼接或删改标签。"""
    if batch_size < 1 or bucket_batches < 0:
        raise ValueError("批次大小必须为正数，长度分组缓冲批次数不能为负数")
    if bucket_batches == 0:
        yield from samples
        return
    rng = random.Random(seed)
    iterator = iter(samples)
    # 缓冲容量是 batch_size 的整数倍，展开后 DataLoader 恰好沿分组边界切批。
    # 长度在分词、长文切窗之后计算，中文/英文字符数不会被误当成 token 数。
    while pool := list(islice(iterator, batch_size * bucket_batches)):
        pool.sort(key=lambda sample: len(sample[0]))
        batches = [pool[start:start + batch_size]
                   for start in range(0, len(pool), batch_size)]
        # 尾批不足 batch_size 时留到最后，防止与另一组拼成长度相差很大的批次。
        tail = batches.pop() if len(batches[-1]) < batch_size else []
        # 组内长度相近，组间随机排列，避免整轮始终从短句训练到长句。
        rng.shuffle(batches)
        for batch in batches:
            yield from batch
        yield from tail


class TrainingDataset(IterableDataset):
    """可重复迭代的语料流；内存取决于批次、缓冲区及单条记录长度。"""

    def __init__(self, path, stage, tokenizer, max_seq_len=None, shuffle_buffer=256, seed=42,
                 instruction_records=None, excluded_prompts=(), *,
                 batch_size=1, length_bucket_batches=0, pretrain_records=None):
        super().__init__()
        self.path = Path(path)
        self.stage = stage
        self.tokenizer = tokenizer
        self.max_seq_len = max_seq_len
        self.shuffle_buffer = shuffle_buffer
        self.seed = seed
        self.instruction_records = instruction_records
        self.pretrain_records = pretrain_records
        self.excluded_prompts = frozenset(excluded_prompts)
        self.batch_size = batch_size
        self.length_bucket_batches = length_bucket_batches
        self.token_cache_path = None

    def records(self):
        """按训练实际范围返回原始记录，供建词表和编码共用。"""
        records = read_records(self.path, self.stage)
        if self.pretrain_records is not None:
            records = chain(records, read_pretrain_records(self.pretrain_records))
        if self.instruction_records is not None:
            records = chain(records, read_instruction_records(self.instruction_records,
                                                              self.excluded_prompts))
        yield from records

    def tokenizer_texts(self):
        """使用相同过滤及文档范围；建词表时不提前切断文字。"""
        for row in self.records():
            if self.stage == "pretrain":
                yield row["text"]
            else:
                yield "用户：" + row["prompt"] + "\n"
                yield row["answer"]

    def uncached_samples(self):
        """原始语料只在预处理时编码；范围及目标遮罩沿用现有逻辑。"""
        yield from encoded_samples(self.records(), self.stage, self.tokenizer, self.max_seq_len)

    def prepare_token_cache(self, path):
        """启动时重建缓存，避免混用旧词表或旧范围；完成后原子替换。"""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        count = 0
        temporary = None
        try:
            with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", dir=path.parent,
                                             suffix=".tmp", delete=False) as output:
                temporary = Path(output.name)
                for inputs, targets in self.uncached_samples():
                    output.write(json.dumps([inputs, targets], separators=(",", ":")) + "\n")
                    count += 1
                    if count % 10000 == 0:
                        print(f"{self.stage} 分词：已缓存 {count} 条训练序列")
            if count == 0:
                raise ValueError("没有可缓存的有效训练序列")
            temporary.replace(path)
        finally:
            if temporary is not None and temporary.exists():
                temporary.unlink()
        self.token_cache_path = path
        return count

    def cached_samples(self):
        with self.token_cache_path.open(encoding="utf-8") as source:
            for line in source:
                inputs, targets = json.loads(line)
                yield inputs, targets

    def __iter__(self):
        if torch.utils.data.get_worker_info() is not None:
            raise RuntimeError("当前流式数据集请使用 num_workers=0，避免重复读取训练样本")
        samples = self.cached_samples() if self.token_cache_path is not None else self.uncached_samples()
        shuffled = buffered_shuffle(samples, self.shuffle_buffer, self.seed)
        yield from length_grouped_samples(shuffled, self.batch_size,
                                          self.length_bucket_batches, self.seed)


class PadBatch:
    def __init__(self, pad_token_id: int):
        self.pad_token_id = pad_token_id

    def __call__(self, samples):
        # B 是本批样本数，T 是本批最长输入；两个张量均为 [B, T]。
        width = max(len(inputs) for inputs, _ in samples)
        input_ids = torch.full((len(samples), width), self.pad_token_id, dtype=torch.long)
        labels = torch.full((len(samples), width), IGNORE_INDEX, dtype=torch.long)
        for row, (inputs, targets) in enumerate(samples):
            input_ids[row, :len(inputs)] = torch.tensor(inputs, dtype=torch.long)
            labels[row, :len(targets)] = torch.tensor(targets, dtype=torch.long)
        return {"input_ids": input_ids, "labels": labels}


def make_mixed_document_dataset(
    arxiv_records, tokenizer: BPETokenizer, pretrain_remaining: dict,
    instruction_remaining: dict, *, seed: int = 42, shuffle_buffer: int = 64,
    excluded_prompts=(), max_seq_len: int | None = None,
):
    """三路按文档混合洗牌，然后按可选的长度上限分块，不丢弃后续块。

    每条记录包含 input_ids、labels、stage、source。一篇正文始终是一条
    完整序列（max_seq_len=None）；指定长度时按块输出，指令仍只监督回答及 EOS。
    返回可重复迭代的 HF IterableDataset；每轮可调用 set_epoch(epoch)。
    """
    if shuffle_buffer < 1:
        raise ValueError("洗牌缓冲区必须为正整数")
    if max_seq_len is not None and max_seq_len < 1:
        raise ValueError("序列长度上限必须为正整数或 None")
    for remaining in (pretrain_remaining, instruction_remaining):
        if remaining["start_document"] < 0:
            raise ValueError("远程起始文档位置不能为负数")
        limit = remaining.get("document_limit")
        if limit is not None and limit < 1:
            raise ValueError("远程文档数量上限必须为正整数或 None")

    def remaining_stream(config):
        records = load_dataset(config["dataset"], name=config.get("config"),
                               split="train", streaming=True).skip(config["start_document"])
        # 数量按原始文档计，必须在编码、分块和混合之前限制。
        limit = config.get("document_limit")
        return records.take(limit) if limit is not None else records

    # 支持调用者已有的普通 Dataset；其余两路始终流式读取，避免全量下载。
    if isinstance(arxiv_records, Dataset):
        arxiv_records = arxiv_records.to_iterable_dataset()
    body_records = remaining_stream(pretrain_remaining)
    instruction_records = remaining_stream(instruction_remaining)
    excluded = frozenset(excluded_prompts)
    instruction_records = instruction_records.filter(lambda row: row["instruction"] not in excluded)
    features = Features({
        "input_ids": Sequence(Value("int64")), "labels": Sequence(Value("int64")),
        "stage": Value("string"), "source": Value("string"),
    })

    def encode_document(row, stage, source):
        records = (read_pretrain_records([row]) if stage == "pretrain"
                   else read_instruction_records([row]))
        # max_seq_len=None：一篇正文/一组问答只产生一条完整训练序列。
        inputs, labels = next(encoded_samples(records, stage, tokenizer,
                                              max_seq_len=None, encode_batch_size=1))
        return {"input_ids": inputs, "labels": labels, "stage": stage, "source": source}

    streams = []
    for records, stage, source in (
        (arxiv_records, "pretrain", "arxiv"),
        (body_records, "pretrain", "fineweb"),
        (instruction_records, "instruction", "belle"),
    ):
        columns = ["text"] if stage == "pretrain" else ["instruction", "output"]
        streams.append(records.select_columns(columns).map(
            encode_document, fn_kwargs={"stage": stage, "source": source},
            remove_columns=columns, features=features,
        ))
    mixed = interleave_datasets(
        streams, probabilities=[1 / 3] * 3, seed=seed,
        stopping_strategy="all_exhausted_without_replacement",
    )
    # 先按整篇洗牌，再依次输出同一文档的块，避免长文改变数据源的抽样概率。
    mixed = mixed.shuffle(seed=seed, buffer_size=shuffle_buffer)
    if max_seq_len is None:
        return mixed

    def split_document(batch):
        result = {name: [] for name in features}
        for inputs, labels, stage, source in zip(
            batch["input_ids"], batch["labels"], batch["stage"], batch["source"],
        ):
            # 标签已经右移，输入与标签按相同位置切片，不能再次移位。
            for start in range(0, len(inputs), max_seq_len):
                targets = labels[start:start + max_seq_len]
                if not any(target != IGNORE_INDEX for target in targets):
                    continue  # 纯问题块无监督目标；回答块及 EOS 全部保留。
                result["input_ids"].append(inputs[start:start + max_seq_len])
                result["labels"].append(targets)
                result["stage"].append(stage)
                result["source"].append(source)
        return result

    return mixed.map(split_document, batched=True, batch_size=1, features=features)


def make_loader(
    path: Path, stage: str, tokenizer: BPETokenizer, *, batch_size: int,
    max_seq_len: int | None = None, shuffle_buffer: int = 256, seed: int = 42,
    instruction_dataset: str | None = None, excluded_prompts=(),
    length_bucket_batches: int = 64,
    pretrain_dataset: str | None = None, pretrain_config: str | None = None,
    remote_start_document: int = 0, remote_document_limit: int | None = None,
) -> DataLoader:
    if (batch_size < 1 or shuffle_buffer < 1
            or (max_seq_len is not None and max_seq_len < 1)):
        raise ValueError("批次大小、序列长度和洗牌缓冲区必须是正整数")
    if length_bucket_batches < 0:
        raise ValueError("长度分组缓冲批次数不能为负数")
    if remote_start_document < 0 or (remote_document_limit is not None and remote_document_limit < 1):
        raise ValueError("远程起始文档位置不能为负数，文档上限必须为正整数")
    if pretrain_dataset is not None and stage != "pretrain":
        raise ValueError("远程正文数据只能加入 pretrain 阶段")
    if instruction_dataset is not None and stage != "instruction":
        raise ValueError("远程指令数据只能加入 instruction 阶段")
    pretrain_records = None
    if pretrain_dataset is not None:
        pretrain_records = load_dataset(
            pretrain_dataset, name=pretrain_config, split="train", streaming=True,
        ).select_columns(["text"])
    instruction_records = None
    if instruction_dataset is not None:
        instruction_records = load_dataset(
            instruction_dataset, split="train", streaming=True,
        ).select_columns(["instruction", "output"])
    # 原始文档范围必须在过滤验证请求、切分 token 窗口和洗牌之前确定。
    # 同一文档的所有窗口会留在同一部分；本地语料不参与远程范围计数。
    def select_range(records):
        if records is None:
            return None
        if remote_start_document:
            records = records.skip(remote_start_document)
        if remote_document_limit is not None:
            records = records.take(remote_document_limit)
        return records

    pretrain_records = select_range(pretrain_records)
    instruction_records = select_range(instruction_records)
    dataset = TrainingDataset(path, stage, tokenizer, max_seq_len, shuffle_buffer, seed,
                              instruction_records, excluded_prompts, batch_size=batch_size,
                              length_bucket_batches=length_bucket_batches,
                              pretrain_records=pretrain_records)
    return DataLoader(dataset, batch_size=batch_size, num_workers=0,
                      collate_fn=PadBatch(tokenizer.pad_token_id), drop_last=False)
