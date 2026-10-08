"""仅训练多语言词表，不加载模型；支持 UTF-8 TXT 和 JSONL 流式读取。"""

import argparse
import json
from pathlib import Path

from parts.tokenizer import BPETokenizer, RuleTokenizer


def read_texts(paths):
    """JSONL 接受 text、prompt/answer 或 instruction/input/output 字段。"""
    for path in paths:
        with path.open(encoding="utf-8-sig", newline="") as source:
            for line_number, line in enumerate(source, start=1):
                if path.suffix.lower() != ".jsonl":
                    yield line
                    continue
                if not line.strip():
                    continue
                try:
                    row = json.loads(line)
                    if not isinstance(row, dict):
                        raise TypeError("JSONL 每行必须是对象")
                    if "text" in row:
                        fields = [row["text"]]
                    elif "prompt" in row and "answer" in row:
                        fields = [row["prompt"], row["answer"]]
                    else:
                        fields = [row["instruction"], row.get("input", ""), row["output"]]
                    if any(not isinstance(field, str) for field in fields):
                        raise TypeError("文本字段必须为字符串")
                except (KeyError, TypeError, json.JSONDecodeError) as error:
                    raise ValueError(f"{path}:{line_number} 不是支持的文本记录") from error
                yield from fields


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("inputs", nargs="+", type=Path, help="一种或多种语言的 TXT/JSONL 训练语料")
    parser.add_argument("--output", type=Path, default=Path(__file__).with_name("tokenizer.json"))
    parser.add_argument("--mode", choices=("rules", "bpe"), default="rules",
                        help="默认 rules：汉字逐字、单词分隔，不设词表数量或词长上限")
    parser.add_argument("--vocab-size", type=int, help="仅 bpe：默认 16384")
    parser.add_argument("--min-frequency", type=int, help="仅 bpe：默认 2")
    parser.add_argument("--max-token-length", type=int, help="仅 bpe：默认 64 字节")
    parser.add_argument("--overwrite", action="store_true", help="明确覆盖旧词表；对应模型需重新训练")
    args = parser.parse_args()
    if args.mode == "rules" and any(value is not None for value in
                                     (args.vocab_size, args.min_frequency, args.max_token_length)):
        parser.error("rules 模式不使用词表/频率/词长上限；这些参数仅适用于 --mode bpe")
    if args.output.exists() and not args.overwrite:
        parser.error("输出词表已经存在；要重新训练请指定 --overwrite")
    if any(path.resolve() == args.output.resolve() for path in args.inputs):
        parser.error("输出词表不能覆盖输入语料")
    if args.mode == "rules":
        tokenizer = RuleTokenizer().train(read_texts(args.inputs), show_progress=True)
    else:
        tokenizer = BPETokenizer().train(
            read_texts(args.inputs), vocab_size=16384 if args.vocab_size is None else args.vocab_size,
            min_frequency=2 if args.min_frequency is None else args.min_frequency,
            max_token_length=64 if args.max_token_length is None else args.max_token_length,
            show_progress=True,
        )
    tokenizer.save(args.output)
    print(f"词表已保存：{args.output.resolve()}；实际词元数：{tokenizer.vocab_size}")


if __name__ == "__main__":
    main()
