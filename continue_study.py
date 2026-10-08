import torch
import vital
from pathlib import Path
from parts.training_data import make_mixed_document_dataset
from datasets import load_dataset
from train import (
    INSTRUCTION_REMAINING_DATA, PRETRAIN_REMAINING_DATA, TEST_TEXTS,
    resolve_mixed_precision, save_model, train_epoch,
)
from torch.utils.data import DataLoader
from parts.training_data import PadBatch

TOKENISER_DIR = Path(__file__).resolve().parent / "checkpoints" / "latest"/ "tokenizer.json"
OUTPUT_DIR = Path(__file__).resolve().parent / "checkpoints" / "continued"
EPOCHS = 10
LEARNING_RATE = 3e-3
MINIMUM_LEARNING_RATE = 2e-5
WEIGHT_DECAY = 0.003
BATCH_SIZE = 32
MAX_BATCH = None  # 每轮最多更新的批次数；设为 None 则遍历完整数据流。
LOG_INTERVAL = 100
MIXED_PRECISION = "auto"  # CUDA 优先 BF16，不支持时用 FP16；CPU 用 FP32。可选 off/bf16/fp16。
MAX_SEQ_LEN = 1024  # 每块最多的输入 token 数；None 表示保留整篇、不分块。
ARXIV_DATASET = "secemp9/arxiv-complete"
ARXIV_CONFIG = "paper_text"  # 全部可用的逐论文 TeX 文本，不是 sample 小样本。
FINEWEB_DATA = {**PRETRAIN_REMAINING_DATA, "document_limit": 2_000_000}
BELLE_DATA = {**INSTRUCTION_REMAINING_DATA, "start_document": 0, "document_limit": None}


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    precision = resolve_mixed_precision(device, MIXED_PRECISION)
    # FP16 使用梯度缩放；跨轮复用同一个 scaler，保留动态缩放状态。
    scaler = torch.amp.GradScaler(device.type, enabled=precision == "fp16")
    print(f"训练设备：{device}；计算精度：{'fp32' if precision == 'off' else precision}")
    print(f"每批最多 {BATCH_SIZE} 条；每块长度上限：{MAX_SEQ_LEN}")
    parameters = torch.load(
        Path(__file__).resolve().parent / "checkpoints" / "latest" / "model.pt",
        map_location=device, weights_only=True,
    )
    if not parameters:
        raise ValueError("模型参数为空，请先完成训练并保存有效权重。")

    tokeniser = vital.load_bpe(TOKENISER_DIR)
    vocab_size = parameters["vocab_chart.weight"].shape[0]
    if tokeniser.vocab_size != vocab_size:
        raise ValueError("分词器词表大小与模型权重不匹配，请使用配套的分词器。")
    pad_batch = PadBatch(tokeniser.pad_token_id)
    ds = load_dataset(ARXIV_DATASET, ARXIV_CONFIG, split="train", streaming=True)
    # 先按文档混合，再按长度分块；后续正文块保留，不只截取文章开头。
    training_set = make_mixed_document_dataset(
        ds, tokeniser, FINEWEB_DATA, BELLE_DATA,
        excluded_prompts={prompt for prompt, _ in TEST_TEXTS},
        max_seq_len=MAX_SEQ_LEN,
    )

    def collate_batch(records):
        return pad_batch([
            (record["input_ids"], record["labels"])
            for record in records
        ])

    model = vital.Vital(vocab_size).to(device)
    model.load_state_dict(parameters)
    # 从已有权重开始新的续训阶段；优化器状态重新初始化。
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=EPOCHS, eta_min=MINIMUM_LEARNING_RATE,
    )
    optimizer_steps = 0

    def record_optimizer_step(optimizer, args, kwargs):
        nonlocal optimizer_steps
        optimizer_steps += 1

    # FP16 溢出时 scaler 会跳过 optimizer.step，此钩子只统计真实更新。
    optimizer.register_step_post_hook(record_optimizer_step)
    loader = DataLoader(
        training_set,
        batch_size=BATCH_SIZE,
        collate_fn=collate_batch,
        num_workers=0,
        drop_last=False,
    )
    for epoch in range(EPOCHS):
        training_set.set_epoch(epoch)  # 每轮重新打乱文档；同一篇的块按原顺序输出。
        learning_rate = optimizer.param_groups[0]["lr"]
        steps_before = optimizer_steps
        print(f"续训轮次：{epoch + 1}/{EPOCHS}；学习率：{learning_rate:.6g}")
        # input_ids/labels 为 [B, T]，设置上限时 T <= MAX_SEQ_LEN。
        # train_epoch 每批清梯度，计算 [B, T, V] 的预测并反向更新；
        # labels 已经移位，不再移位；-100（问题部分及 PAD）不计入损失。
        metrics = train_epoch(
            model, loader, optimizer, device,
            max_batches=MAX_BATCH, log_interval=LOG_INTERVAL,
            mixed_precision=precision, scaler=scaler,
        )
        if optimizer_steps > steps_before:
            scheduler.step()
        else:
            print("本轮未完成参数更新，保持当前学习率；请检查梯度溢出情况。")
        print(
            f"本轮完成：{metrics['steps']} 批；有效目标：{metrics['tokens']}；"
            f"按有效目标 token 加权的平均损失：{metrics['loss']:.4f}"
        )
        save_model(model, tokeniser, OUTPUT_DIR)


if __name__ == "__main__":
    main()
