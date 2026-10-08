"""先完整分词并缓存选定语料，再进行正文预训练和指令训练。

training_set/pretrain.jsonl：每行一个 {"text": "正文"}。
training_set/instruction.jsonl：每行一个 {"prompt": "请求", "answer": "回答"}。
正文阶段还会流式读取 FineWeb-Edu 的 text 字段，与本地正文共用下一词元训练流程。
指令阶段还会流式读取 BELLE 的 instruction/output，忽略远程数据的 input 字段。
两阶段共用 tokenizer.json。正文对所有后续词元计算损失，指令只对回答及 EOS 计算损失。
测试请求不参与词表或模型训练；生成时仅传入请求和模型自己的预测。

示例：.venv/Scripts/python.exe train.py --batch-size 4 --max-seq-len 256
轮数由下方文件参数或命令行指定；--max-batches 可限制每阶段每轮的批次数。
--mixed-precision 选择混合精度；--length-bucket-batches 控制按 token 长度组批的缓冲大小。
--usejsononly 只使用本地 JSONL，词表构建和两个训练阶段均不加载远程数据。
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime
from itertools import islice
from pathlib import Path
import torch
from torch import nn

import vital
from vital import Vital
from parts.tokenizer import RuleTokenizer
from parts.training_data import (
    IGNORE_INDEX, STAGE_FILES, encode_prompt, make_loader, read_records, tokenizer_texts,
)


DATA_DIR = Path(__file__).resolve().parent / "training_set"
TOKENIZER_PATH = Path(__file__).resolve().with_name("tokenizer.json")
OUTPUT_DIR = Path(__file__).resolve().parent / "checkpoints" / "latest"
INSTRUCTION_DATASET = "BelleGroup/train_1M_CN"
PRETRAIN_DATASET = "HuggingFaceFW/fineweb-edu"
# default 使用全量正文；也可选 sample-10BT 等官方子集。数据始终按需流式读取。
PRETRAIN_DATASET_CONFIG = "default"
# 首次训练的外接语料范围：按原始文档数划分，先截取、再分词和洗牌。
# 每轮重复训练这一首段；本地 JSONL 不受这些上限影响。
PRETRAIN_FIRST_PART_DOCUMENTS = 400_000
INSTRUCTION_FIRST_PART_DOCUMENTS = 300_000
# 供其他脚本导入；仅描述计划范围，不是自动记录的实际训练进度。
# dataset/config 用于 load_dataset，start_document 用于 dataset.skip(...)。
# 后续训练须复用同一数据集版本、划分值，以及已保存的权重和分词器。
PRETRAIN_REMAINING_DATA = {
    "dataset": PRETRAIN_DATASET,
    "config": PRETRAIN_DATASET_CONFIG,
    "start_document": PRETRAIN_FIRST_PART_DOCUMENTS,
}
INSTRUCTION_REMAINING_DATA = {
    "dataset": INSTRUCTION_DATASET,
    "config": None,
    "start_document": INSTRUCTION_FIRST_PART_DOCUMENTS,
}
BATCH_SIZE = 4
MAX_SEQ_LEN = 512
# 混合精度：auto 在 CUDA 上优先用 BF16，否则用 FP16；CPU 自动保持 FP32。
# 也可明确设为 "off"、"bf16" 或 "fp16"；模型参数本身仍保留 FP32。
MIXED_PRECISION = "auto"
# 每次缓存多少个批次的样本来按 token 长度分组；0 关闭，64 表示最多缓存 64 * BATCH_SIZE 条。
LENGTH_BUCKET_BATCHES = 64
PRETRAIN_EPOCHS = 1000
INSTRUCTION_EPOCHS = 1000
lr = 3e-3
minimum_lr = 2e-5
weight_decay = 1e-3

# 独立验证集，只用于分词器还原检查和训练后的生成评估。
TEST_TEXTS: list[tuple[str, str]] = [
    (
        "请用一句话解释嵌入层的作用。",
        "嵌入层把词元编号映射为模型能够学习和处理的向量。",
    ),
    ("请计算 46 加 27，只输出结果。", "73"),
    (
        "请解方程 3x - 9 = 12，并说明步骤。",
        "两边加上 9，得到 3x = 21；再将两边除以 3，得到 x = 7。",
    ),
    ("请把 The library closes at six. 翻译成中文。", "图书馆六点关门。"),
    (
        "请用一句话总结：小周提前整理复习材料，每天完成一组练习，考前集中复习错题。",
        "小周通过提前准备、每日练习和复习错题来备考。",
    ),
    (
        "请写一个 Python 函数，判断一个整数是否为偶数。",
        "```python\ndef is_even(n):\n    return n % 2 == 0\n```",
    ),
    (
        "请从“课程：线性代数；教室：204；时间：上午十点”中提取信息，只输出 JSON。",
        '{"课程": "线性代数", "教室": 204, "时间": "上午十点"}',
    ),
    (
        "请写一句同时包含 🦊 和 ☕ 的温暖问候。",
        "愿你今天像 🦊 一样充满活力，也能享受一杯 ☕ 带来的温暖。",
    ),
]


def format_dialogue(prompt: str, answer: str) -> str:
    """用户前缀是普通文本；助手标记编码为独立的特殊词元。"""
    return f"用户：{prompt}\n{vital.BPETokenizer.ASSISTANT_TOKEN}{answer}"


def test_tokenizer(tokenizer: vital.BPETokenizer) -> None:
    """验证每条测试文本经过编码、解码后仍与原文完全一致。"""
    # prompt：用户请求；answer：参考回答；text：用于分词器检查的完整对话。
    for sample_number, (prompt, answer) in enumerate(TEST_TEXTS, start=1):
        text = format_dialogue(prompt, answer)
        # token_ids：整数编号列表，包括开头的 <bos> 和结尾的 <eos>。
        token_ids = vital.bpe_tokenise(
            text, tokenizer, add_bos=True, add_eos=True
        )
        # 去掉额外添加的 BOS/EOS，再保留文本中的助手标记以检查完整还原。
        restored_text = tokenizer.decode(
            token_ids[1:-1], skip_special_tokens=False, errors="strict"
        )
        if restored_text != text:
            raise AssertionError(f"第 {sample_number} 条测试文本未能完整还原")

        print(f"\n测试 {sample_number}：{text}")
        print(f"词元编号：{token_ids}")
        print(f"还原文字：{restored_text}")

    print(f"\n编码与解码检查：{len(TEST_TEXTS)}/{len(TEST_TEXTS)} 条通过")


def texts_to_tensor(
    texts: list[str],
    tokenizer: vital.BPETokenizer,
    device: torch.device,
    *,
    add_bos: bool,
    add_eos: bool,
) -> torch.Tensor:
    """把一批文字编成补齐后的词元编号张量，形状为 [样本数, 序列长度]。"""
    sequences = tokenizer.encode_batch(texts, add_bos=add_bos, add_eos=add_eos)
    width = max(len(sequence) for sequence in sequences)
    padded = [
        sequence + [tokenizer.pad_token_id] * (width - len(sequence))
        for sequence in sequences
    ]
    return torch.tensor(padded, dtype=torch.long, device=device)


def save_model(model: nn.Module, tokenizer: vital.BPETokenizer, output_dir: Path) -> None:
    """保存所有模型参数、持久缓冲区，以及与当前权重配套的完整分词器。"""
    output_dir.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), output_dir / "model.pt")
    tokenizer.save(output_dir / "tokenizer.json")
    print(f"模型权重已保存到：{output_dir / 'model.pt'}")
    print(f"配套分词器已保存到：{output_dir / 'tokenizer.json'}")


def save_train_result(
    records: list[dict[str, str]],
    trained_at: str,
    parameter_count: int,
) -> None:
    """把本次验证集结果追加到 audit/validation.json，保留已有训练记录。"""
    output_dir = Path(__file__).resolve().parent / "audit"
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "validation.json"
    history: list[dict[str, object]] = []
    if output_path.exists():
        loaded = json.loads(output_path.read_text(encoding="utf-8"))
        if isinstance(loaded, list):
            history = loaded
    # 旧文件是扁平的验证样本列表，先收成一条历史记录再追加。
    if history and isinstance(history[0], dict) and "请求" in history[0]:
        previous_time = datetime.fromtimestamp(output_path.stat().st_mtime).strftime(
            "%Y-%m-%d %H:%M:%S"
        )
        history = [{"训练时间": previous_time, "验证集": history}]
    history.append(
        {"训练时间": trained_at, "总参数量": parameter_count, "验证集": records}
    )
    output_path.write_text(
        json.dumps(history, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(f"验证集结果已追加到：{output_path}")


def train_tokenizer(*, retrain: bool = False, data_dir: Path | None = None,
                    mode: str = "auto", texts=None) -> vital.BPETokenizer:
    """复用固定词表；新词表默认按规则收集，auto 仍兼容已有 BPE。"""
    if mode not in ("auto", "rules", "bpe"):
        raise ValueError("分词模式必须是 auto、rules 或 bpe")
    if TOKENIZER_PATH.exists() and not retrain:
        tokenizer = vital.load_bpe(TOKENIZER_PATH)
        actual_mode = "rules" if isinstance(tokenizer, RuleTokenizer) else "bpe"
        if mode != "auto" and mode != actual_mode:
            raise ValueError("已有词表的模式不匹配；请用 --tokenizer 指定新路径，"
                             "或明确传入 --retrain-tokenizer 重建并从头训练模型")
        print(f"复用分词器：{TOKENIZER_PATH}；实际词表：{tokenizer.vocab_size}")
        return tokenizer
    source = texts if texts is not None else tokenizer_texts(data_dir or DATA_DIR)
    if mode == "bpe":
        tokenizer = vital.train_bpe(source, vocab_size=16384)
    else:
        tokenizer = RuleTokenizer().train(source, show_progress=True)
    tokenizer.save(TOKENIZER_PATH)
    print(f"词表已保存：{TOKENIZER_PATH}；实际词表：{tokenizer.vocab_size}")
    return tokenizer


def validate_training_data(data_dir: Path) -> dict[str, int]:
    """流式检查记录格式、非空语料和验证请求泄漏，只保留计数。"""
    forbidden = {prompt for prompt, _ in TEST_TEXTS}
    counts = {}
    for stage, filename in STAGE_FILES.items():
        count = 0
        for row in read_records(data_dir / filename, stage):
            if stage == "instruction" and row["prompt"] in forbidden:
                raise ValueError(f"指令训练集包含验证请求：{row['prompt']}")
            count += 1
        if count == 0:
            raise ValueError(f"训练文件没有有效记录：{data_dir / filename}")
        counts[stage] = count
    return counts


def resolve_mixed_precision(device: torch.device, mode: str) -> str:
    """解析实际精度；显式请求不支持的模式时立即报错，避免静默切换。"""
    if mode not in ("auto", "off", "bf16", "fp16"):
        raise ValueError("混合精度必须是 auto、off、bf16 或 fp16")
    if mode == "off":
        return mode
    if device.type == "cuda":
        if not torch.cuda.is_available():
            raise ValueError("当前环境没有可用的 CUDA 设备")
        with torch.cuda.device(device):
            supports_bf16 = torch.cuda.is_bf16_supported()
        if mode == "auto":
            return "bf16" if supports_bf16 else "fp16"
        if mode == "bf16" and not supports_bf16:
            raise ValueError("当前 CUDA 设备不支持 BF16，请使用 fp16 或 off")
        return mode
    if device.type == "cpu":
        if mode == "fp16":
            raise ValueError("本训练脚本的 FP16 模式需要 CUDA；CPU 请使用 off 或 bf16")
        return "off" if mode == "auto" else mode
    raise ValueError(f"暂不支持在 {device.type} 设备上使用混合精度")


def train_epoch(
    model: nn.Module, loader, optimizer, device: torch.device, *,
    max_batches: int | None = None, log_interval: int = 10,
    mixed_precision: str = "off", scaler: torch.amp.GradScaler | None = None,
) -> dict[str, float | int]:
    """一轮只搬运当前批次到设备，一次计算批次内所有有效位置的损失。"""
    if max_batches is not None and max_batches < 1:
        raise ValueError("max_batches 必须至少为 1")
    model.train()
    precision = resolve_mixed_precision(device, mixed_precision)
    amp_dtype = torch.bfloat16 if precision == "bf16" else torch.float16
    # FP16 的梯度范围较小，使用动态损失缩放；BF16/FP32 不需要缩放。
    # main 会跨 epoch 复用 scaler，避免每轮重置已学习的缩放比例。
    if scaler is None:
        scaler = torch.amp.GradScaler(device.type, enabled=precision == "fp16")
    criterion = nn.CrossEntropyLoss(ignore_index=IGNORE_INDEX)
    batches = islice(loader, max_batches) if max_batches is not None else loader
    total_tokens, steps = 0, 0
    # 累计统计留在训练设备上；FP64 对应原先 Python float 的累计精度。
    # 只在打印或返回时读取，避免每批 loss.item() 强制 CPU 等待 GPU。
    total_loss = torch.zeros((), device=device, dtype=torch.float64)
    for batch in batches:
        valid_tokens = int((batch["labels"] != IGNORE_INDEX).sum().item())
        if valid_tokens == 0:
            raise ValueError("当前批次没有可训练的目标词元")
        # input_ids、labels：[B, T]，B <= batch_size，T <= max_seq_len。
        input_ids = batch["input_ids"].to(device)
        labels = batch["labels"].to(device)
        optimizer.zero_grad(set_to_none=True)
        # autocast 只包住前向和损失：矩阵运算使用选定精度，交叉熵自动保留 FP32。
        # input_ids/labels 仍为整数；logits：[B, T, V]，V 是实际词表大小。
        with torch.autocast(device_type=device.type, dtype=amp_dtype,
                            enabled=precision != "off"):
            logits = model(input_ids, last_token_only=False)
            loss = criterion(logits.reshape(-1, logits.size(-1)), labels.reshape(-1))
        if not torch.isfinite(loss):
            raise RuntimeError("训练损失不是有限数值，请检查数据和学习率")
        # 反向传播在 autocast 外执行；关闭缩放时以下调用等价于原来的 backward/step。
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        steps += 1
        total_tokens += valid_tokens
        total_loss.add_(loss.detach().to(torch.float64), alpha=valid_tokens)
        if log_interval > 0 and steps % log_interval == 0:
            average_loss = total_loss.item() / total_tokens
            print(f"批次：{steps}；有效目标：{total_tokens}；平均损失：{average_loss:.4f}")
    if steps == 0:
        raise ValueError("本轮没有产生训练批次")
    return {"steps": steps, "tokens": total_tokens, "loss": total_loss.item() / total_tokens}


def evaluate_replies(model, tokenizer, device, max_seq_len, max_new_tokens):
    """参考答案只用于记录；生成输入只包含请求和模型自己的预测。"""
    model.eval()
    records = []
    with torch.inference_mode():
        for number, (prompt, answer) in enumerate(TEST_TEXTS, start=1):
            input_ids = torch.tensor([encode_prompt(tokenizer, prompt)],
                                     dtype=torch.long, device=device)
            generated_ids = []
            stopped_at_eos = False
            for _ in range(max_new_tokens):
                result = model(input_ids[:, -max_seq_len:], last_token_only=True)
                # 这些控制词元不会作为正文/回答目标，生成时也不应输出。
                result[:, [tokenizer.pad_token_id, tokenizer.bos_token_id,
                           tokenizer.assistant_token_id]] = float("-inf")
                next_id = result.argmax(dim=-1).item()
                if next_id == tokenizer.eos_token_id:
                    stopped_at_eos = True
                    break
                generated_ids.append(next_id)
                next_tensor = torch.tensor([[next_id]], dtype=torch.long, device=device)
                input_ids = torch.cat((input_ids[:, -max_seq_len:], next_tensor), dim=1)
            reply = tokenizer.decode(generated_ids)
            reason = "EOS" if stopped_at_eos else f"达到 {max_new_tokens} 个词元上限"
            print(f"\n测试 {number}：{prompt}\n模型回答：{reply}\n参考回答：{answer}\n结束原因：{reason}")
            records.append({"请求": prompt, "模型回答": reply, "参考回答": answer, "结束原因": reason})
    return records


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description="正文预训练 → 指令训练，流式小批次加载")
    parser.add_argument("--data-dir", type=Path, default=DATA_DIR)
    parser.add_argument("--usejsononly", action="store_true",
                        help="只使用 --data-dir 下的本地 JSONL；不加载远程正文或指令数据")
    parser.add_argument("--tokenizer", type=Path, default=TOKENIZER_PATH)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR,
                        help="训练完成后保存 model.pt 和 tokenizer.json 的目录；同名文件会覆盖")
    parser.add_argument("--retrain-tokenizer", action="store_true", help="明确重训词表；本脚本从头训练模型")
    parser.add_argument("--tokenizer-mode", choices=("auto", "rules", "bpe"), default="auto",
                        help="auto 复用现有格式，新建时使用 rules；rules 为汉字逐字/单词分隔，无词表数量上限")
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    parser.add_argument("--max-seq-len", type=int, default=MAX_SEQ_LEN)
    parser.add_argument("--mixed-precision", choices=("auto", "off", "bf16", "fp16"),
                        default=MIXED_PRECISION, help="自动混合精度；off 保持 FP32")
    parser.add_argument("--length-bucket-batches", type=int, default=LENGTH_BUCKET_BATCHES,
                        help="每次按长度分组的批次数；0 关闭，批次之间仍随机打乱")
    parser.add_argument("--pretrain-epochs", type=int, default=PRETRAIN_EPOCHS)
    parser.add_argument("--instruction-epochs", type=int, default=INSTRUCTION_EPOCHS)
    parser.add_argument("--shuffle-buffer", type=int, default=256)
    parser.add_argument("--max-batches", type=int, help="每阶段每轮最多训练几个批次，用于快速检查")
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-new-tokens", type=int, default=128)
    parser.add_argument("--skip-validation", action="store_true", help="跳过最后的回复生成检查")
    args = parser.parse_args(argv)
    if min(args.batch_size, args.max_seq_len, args.shuffle_buffer, args.max_new_tokens) < 1:
        parser.error("批次、序列、缓冲区及生成长度必须为正整数")
    if min(args.pretrain_epochs, args.instruction_epochs) < 0:
        parser.error("训练轮数不能为负数")
    if args.length_bucket_batches < 0:
        parser.error("长度分组缓冲批次数不能为负数")
    if args.pretrain_epochs + args.instruction_epochs == 0:
        parser.error("至少启用一个训练阶段")
    if args.max_batches is not None and args.max_batches < 1:
        parser.error("--max-batches 必须至少为 1")
    if any(args.tokenizer.resolve() == (args.data_dir / filename).resolve()
           for filename in STAGE_FILES.values()):
        parser.error("分词器输出不能覆盖训练语料")
    return args


def main(argv=None) -> None:
    global TOKENIZER_PATH
    args = parse_args(argv)
    TOKENIZER_PATH = args.tokenizer
    counts = validate_training_data(args.data_dir)
    # 使用本次运行的局部配置，不修改全局数据源，避免影响后续调用。
    pretrain_dataset = None if args.usejsononly else PRETRAIN_DATASET
    instruction_dataset = None if args.usejsononly else INSTRUCTION_DATASET
    device = torch.device("cuda" if args.device == "auto" and torch.cuda.is_available()
                          else "cpu" if args.device == "auto" else args.device)
    precision = resolve_mixed_precision(device, args.mixed_precision)
    scaler = torch.amp.GradScaler(device.type, enabled=precision == "fp16")
    torch.manual_seed(args.seed)
    print(f"设备：{device}；本地正文：{counts['pretrain']} 条；本地指令：{counts['instruction']} 条")
    if args.usejsononly:
        print("数据来源：仅本地 JSONL（不加载远程数据，建词表也只使用本地数据）")
    if args.pretrain_epochs > 0 and pretrain_dataset is not None:
        print(f"额外正文数据：{pretrain_dataset}（{PRETRAIN_DATASET_CONFIG}），只读取 text")
        print(f"本次范围：前 {PRETRAIN_FIRST_PART_DOCUMENTS} 篇原始文档；其余范围见 PRETRAIN_REMAINING_DATA")
    if args.instruction_epochs > 0 and instruction_dataset is not None:
        print(f"额外指令数据：{instruction_dataset}，只读取 instruction 和 output")
        print(f"本次范围：前 {INSTRUCTION_FIRST_PART_DOCUMENTS} 篇原始文档；其余范围见 INSTRUCTION_REMAINING_DATA")
    if args.max_batches is not None and (pretrain_dataset is not None or instruction_dataset is not None):
        print("注意：--max-batches 可能提前结束首段；剩余范围变量不会随实际进度自动更新。")
    print(f"每批最多 {args.batch_size} 条序列；每条最多 {args.max_seq_len} 个输入词元")
    print(f"训练精度：{'FP32' if precision == 'off' else precision.upper()}；"
          f"长度分组缓冲：{args.length_bucket_batches} 个批次（0 表示关闭）")
    stages = (("pretrain", "正文预训练", args.pretrain_epochs),
              ("instruction", "指令训练", args.instruction_epochs))
    loaders = {}
    for stage, name, epochs in stages:
        if epochs == 0:
            continue
        # 每个阶段只创建一次数据流；后续轮次重新迭代，不重复调用 load_dataset。
        loader = make_loader(args.data_dir / STAGE_FILES[stage], stage, vital.BPETokenizer(),
                             batch_size=args.batch_size, max_seq_len=args.max_seq_len,
                             shuffle_buffer=args.shuffle_buffer,
                             length_bucket_batches=args.length_bucket_batches,
                             pretrain_dataset=pretrain_dataset if stage == "pretrain" else None,
                             pretrain_config=PRETRAIN_DATASET_CONFIG if stage == "pretrain" else None,
                             instruction_dataset=instruction_dataset if stage == "instruction" else None,
                             remote_document_limit=(PRETRAIN_FIRST_PART_DOCUMENTS if stage == "pretrain"
                                                    else INSTRUCTION_FIRST_PART_DOCUMENTS),
                             excluded_prompts={prompt for prompt, _ in TEST_TEXTS})
        loaders[stage] = loader

    # 首次建表扫描启用阶段的完整选定范围（含远程），然后固定编号再缓存。
    # --max-batches 只限制模型更新，既不限制词表扫描，也不限制分词缓存。
    texts = (text for loader in loaders.values() for text in loader.dataset.tokenizer_texts())
    tokenizer = train_tokenizer(retrain=args.retrain_tokenizer, data_dir=args.data_dir,
                                mode=args.tokenizer_mode, texts=texts)
    test_tokenizer(tokenizer)
    loaded = vital.load_bpe(TOKENIZER_PATH)
    if loaded.encode_batch([prompt for prompt, _ in TEST_TEXTS]) != tokenizer.encode_batch(
        [prompt for prompt, _ in TEST_TEXTS]
    ):
        raise AssertionError("保存加载后词元编号发生变化")
    if isinstance(loaded, RuleTokenizer):
        # 当前模型的嵌入层和输出层不共享参数，每个词元各占两行。
        vocab_parameters = 2 * loaded.vocab_size * vital.dimension_word
        print(f"规则词表已固定：{loaded.vocab_size} 个词元；"
              f"嵌入层与输出层共 {vocab_parameters:,} 个参数。上下文长度仍由 --max-seq-len 控制。")
    for stage, name, epochs in stages:
        if epochs == 0:
            continue
        loader = loaders[stage]
        loader.dataset.tokenizer = loaded
        print(f"开始预先分词：{name}（--max-batches 只限制训练，不限制本次分词范围）")
        cache_path = args.output_dir / "token_cache" / f"{stage}.jsonl"
        count = loader.dataset.prepare_token_cache(cache_path)
        print(f"{name}分词完成：{count} 条训练序列；缓存：{cache_path}")

    # 两个启用阶段的分词全部结束后，才创建模型、占用训练显存。
    model = Vital(loaded.vocab_size).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay, fused= True)
    total_epochs = args.pretrain_epochs + args.instruction_epochs
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=total_epochs, eta_min=minimum_lr)
    trained_at = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    completed_epochs = 0
    for stage, name, epochs in stages:
        if epochs == 0:
            continue
        loader = loaders[stage]
        for epoch in range(epochs):
            print(f"\n{name}：第 {epoch + 1}/{epochs} 轮")
            loader.dataset.seed = args.seed + completed_epochs
            metrics = train_epoch(model, loader, optimizer, device, max_batches=args.max_batches,
                                  mixed_precision=precision, scaler=scaler)
            scheduler.step()
            completed_epochs += 1
            print(f"{name}完成：{metrics['steps']} 批；{metrics['tokens']} 个有效目标；损失 {metrics['loss']:.4f}")
    parameter_count = sum(parameter.numel() for parameter in model.parameters())
    print(f"总参数量：{parameter_count}")
    # 先保存，再生成验证回答；跳过验证也不会跳过保存。
    save_model(model, loaded, args.output_dir)
    if not args.skip_validation:
        records = evaluate_replies(model, loaded, device, args.max_seq_len, args.max_new_tokens)
        save_train_result(records, trained_at, parameter_count)


if __name__ == "__main__":
    main()
