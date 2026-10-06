"""用户请求与助手回复语料：训练 BPE，并检查对话文本的编码、解码和保存加载。

在 Vital 项目的 Python 环境中运行本文件即可。
每条语料保存为 (prompt, answer) 元组：第一个元素为用户请求，第二个为参考回答。
训练 BPE 时，通过 format_dialogue 合成“用户：请求内容\n<assistant>回答内容”。
其中 <assistant> 是固定编号 259 的特殊词元，不参与 BPE 合并。
目标行为是根据用户请求给出相应结果，而不只是复述请求或承诺执行。
本脚本训练对象仍是分词器；测试结果表示对话文字能否完整还原。
训练语言模型时，将用户请求作为上下文，以助手回复作为预测目标；
可屏蔽用户部分的损失，对助手回复及末尾 EOS 计算下一词元预测损失。

分词器用法：
    tokenizer = train_tokenizer()
    prompt, answer = TRAIN_TEXTS[0]
    text = format_dialogue(prompt, answer)
    token_ids = tokenizer.encode(text, add_bos=True, add_eos=True)
    restored_text = tokenizer.decode(token_ids[1:-1], skip_special_tokens=False, errors="strict")

输出保存位置：
    encode 返回的是整数编号列表，保存在调用处的 token_ids 等变量中。
    本脚本没有将每条语料的词元编号写入文件，也没有生成嵌入向量。
    运行 main 后，词表和合并规则保存到本脚本同目录的 demo_tokenizer.json。
    该 JSON 是分词器配置，不包含每条语料的编号列表或神经网络嵌入向量。
"""

from __future__ import annotations

from pathlib import Path

import torch
from torch import nn

import vital
from vital import Vital


# TRAIN_TEXTS：训练集，每条样例为 (prompt, answer)，两个元素都是字符串。
# prompt 说明任务，answer 给出实际结果，存储时不包含角色前缀。
# BPE 只接收字符串，因此训练前通过 format_dialogue 格式化每个元组。
TRAIN_TEXTS: list[tuple[str, str]] = [
    (
        "用一句话向我打招呼，并询问我需要什么帮助。",
        "你好！你希望我帮你完成什么事情？",
    ),
    (
        "请用一句话解释分词器的作用。",
        "分词器把文字转换成词元编号，让模型能够处理这些文字。",
    ),
    (
        "请解释 PAD 词元的作用。",
        "PAD 是补齐长度的占位词元，让同一批次中的不同长度序列能够组成矩形张量。",
    ),
    ("请把 I enjoy learning mathematics. 翻译成中文。", "我喜欢学习数学。"),
    ("请把“今天的天气很好。”翻译成英文。", "The weather is nice today."),
    (
        "请把“由于时间方面的原因，我们决定将会议安排推迟到明天。”改写得简洁一些。",
        "时间有限，会议推迟到明天。",
    ),
    (
        "请用一句话总结：小林每天练习 Python，遇到错误就查阅文档，一个月后能够独立编写小程序。",
        "小林通过持续练习和查阅文档，逐渐掌握了独立编写 Python 小程序的能力。",
    ),
    ("请计算 37 加 58，只输出结果。", "95"),
    (
        "请解方程 2x + 6 = 18，并说明步骤。",
        "两边减去 6，得到 2x = 12；再将两边除以 2，得到 x = 6。",
    ),
    (
        "请写一个 Python 函数，返回两个数中较大的那个。",
        "```python\ndef larger(a, b):\n    return max(a, b)\n```",
    ),
    (
        "请用 Python 列表推导式生成 1 到 5 的平方，只输出代码。",
        "```python\nsquares = [n * n for n in range(1, 6)]\n```",
    ),
    (
        "请把“阅读教材、完成练习、复习错题”整理成编号列表。",
        "1. 阅读教材\n2. 完成练习\n3. 复习错题",
    ),
    (
        "请从“姓名：李明；年龄：20；专业：数学”中提取信息，只输出 JSON。",
        '{"姓名": "李明", "年龄": 20, "专业": "数学"}',
    ),
    ("请修正这句话的语病：我昨天去了图书馆，并阅读书籍了。", "我昨天去图书馆看书了。"),
    (
        "请写一封简短的中文邮件，向老师询问作业截止时间。",
        "老师您好：\n请问本次作业的截止时间是什么时候？感谢您的帮助。\n祝好！",
    ),
    (
        "请制定一个总时长为 30 分钟的数学复习计划。",
        "前 10 分钟复习定义和公式，中间 15 分钟完成练习，最后 5 分钟整理错题。",
    ),
    (
        "请说明 Python 列表和元组在修改元素方面的区别。",
        "列表可以增删或替换元素；元组创建后不能增删或替换其中的元素。",
    ),
    (
        "请解释什么是平均数，并用 2、4、6 举例。",
        "平均数是所有数的总和除以数的个数。2、4、6 的平均数是 (2 + 4 + 6) / 3 = 4。",
    ),
    ("请把“赶快把文件发给我。”改写得礼貌一些。", "方便的话，请把文件发给我，谢谢。"),
    ("请写一句包含猫和狗的简短故事。", "小猫迷路时，一只热心的小狗陪它找到了回家的路。"),
    ("你是什么模型？", "我是Vital模型。"),
]

# TEST_TEXTS：独立测试集，不参与学习词表或合并规则。
# 同样使用 (prompt, answer) 元组，但请求与训练集不同。
# 这些文本当前用于分词器检查；评估模型回复时，只输入“用户：...\n<assistant>”。
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


def train_tokenizer() -> vital.BPETokenizer:
    """只使用 TRAIN_TEXTS 训练分词器，并设置默认分词器。"""
    # tokenizer：训练后的分词器对象。
    # vocab_size 是包含特殊词元的词表上限；min_frequency 是最低出现次数。
    texts = [format_dialogue(prompt, answer) for prompt, answer in TRAIN_TEXTS]
    tokenizer = vital.train_bpe(texts, vocab_size=51200, min_frequency=2)
    print(f"训练集：{len(TRAIN_TEXTS)} 条；测试集：{len(TEST_TEXTS)} 条")
    print(f"实际词表：{tokenizer.vocab_size} 个词元；合并规则：{len(tokenizer.merges)} 条")
    return tokenizer


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
    sequences = [
        tokenizer.encode(text, add_bos=add_bos, add_eos=add_eos) for text in texts
    ]
    width = max(len(sequence) for sequence in sequences)
    padded = [
        sequence + [tokenizer.pad_token_id] * (width - len(sequence))
        for sequence in sequences
    ]
    return torch.tensor(padded, dtype=torch.long, device=device)


EPOCH = 500
lr = 1e-3
minimum_lr = 1e-5
weight_decay = 1e-3

def main() -> None:
    # 检查请求没有同时出现在训练集和测试集中，即使参考回答不同也不允许重复。
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_prompts = {prompt for prompt, _ in TRAIN_TEXTS}
    train_labels = {label for _, label in TRAIN_TEXTS}
    test_labels = {label for _, label in TEST_TEXTS}
    test_prompts = {prompt for prompt, _ in TEST_TEXTS}
    if train_prompts & test_prompts:
        raise ValueError("训练集和测试集包含重复请求")

    tokenizer = train_tokenizer()
    test_tokenizer(tokenizer)

    # output_path：输出文件，固定保存在本脚本所在目录。
    output_path = Path(__file__).resolve().with_name("demo_tokenizer.json")
    tokenizer.save(output_path)

    # loaded_tokenizer：从文件加载的分词器，同时设置为默认分词器。
    loaded_tokenizer = vital.load_bpe(output_path)
    for prompt, answer in TEST_TEXTS:
        text = format_dialogue(prompt, answer)
        if loaded_tokenizer.encode(text) != tokenizer.encode(text):
            raise AssertionError("保存加载后，词元编号发生了变化")

    print("保存与加载检查：通过")
    print(f"分词器已保存到：{output_path}")

    model = Vital(loaded_tokenizer.vocab_size).to(device)
    criterion = nn.CrossEntropyLoss(ignore_index=loaded_tokenizer.pad_token_id)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=EPOCH, eta_min=minimum_lr
    )

    # 按 TRAIN_TEXTS / TEST_TEXTS 的原顺序编码，保证每一行的请求和回答仍然成对。
    # 请求以 BOS 开头、助手标记结尾；回答以 EOS 结尾。较短序列用 PAD 补齐。
    train_prompts = texts_to_tensor(
        [format_dialogue(prompt, "") for prompt, _ in TRAIN_TEXTS],
        loaded_tokenizer,
        device,
        add_bos=True,
        add_eos=False,
    )
    train_labels = texts_to_tensor(
        [label for _, label in TRAIN_TEXTS],
        loaded_tokenizer,
        device,
        add_bos=False,
        add_eos=True,
    )
    test_prompts = texts_to_tensor(
        [format_dialogue(prompt, "") for prompt, _ in TEST_TEXTS],
        loaded_tokenizer,
        device,
        add_bos=True,
        add_eos=False,
    )
    test_labels = texts_to_tensor(
        [label for _, label in TEST_TEXTS],
        loaded_tokenizer,
        device,
        add_bos=False,
        add_eos=True,
    )

    # 保留初始编号张量；逐词元训练只修改当前请求和剩余回答。
    initial_train_prompts = train_prompts.clone()
    initial_train_labels = train_labels.clone()

    for epoch in range(EPOCH):
        if train_labels.size(1) == 0:
            train_prompts = initial_train_prompts.clone()
            train_labels = initial_train_labels.clone()

        # 本轮只预测剩余回答的第一个词元，标签形状为 [样本数]。
        next_labels = train_labels[:, 0].clone()
        optimizer.zero_grad()
        loss = criterion(model(train_prompts, last_token_only=True), next_labels)
        loss.backward()
        optimizer.step()
        scheduler.step()

        # 训练完成后，将正确词元放到各请求的有效末尾，再移除回答首列。teacher forcing
        # 已结束样本的下一标签是 PAD，不追加，也不参与损失。
        active = next_labels != loaded_tokenizer.pad_token_id
        prompt_lengths = (train_prompts != loaded_tokenizer.pad_token_id).sum(dim=1)
        if (prompt_lengths[active] == train_prompts.size(1)).any():
            padding_column = torch.full(
                (train_prompts.size(0), 1),
                loaded_tokenizer.pad_token_id,
                dtype=train_prompts.dtype,
                device=device,
            )
            train_prompts = torch.cat((train_prompts, padding_column), dim=1)
        rows = torch.arange(train_prompts.size(0), device=device)[active]
        train_prompts[rows, prompt_lengths[active]] = next_labels[active]
        train_labels = train_labels[:, 1:]

        if (epoch + 1) % 10 == 0:
            print(f'epoch:{epoch + 1}, loss:{loss.item():.4f}')

    model.eval()
    with torch.no_grad():
        max_new_tokens = 128
        print("\n模型回复测试：")
        for sample_number, (prompt, answer) in enumerate(TEST_TEXTS, start=1):
            prompt_ids = test_prompts[sample_number - 1]
            input_ids = prompt_ids[
                prompt_ids != loaded_tokenizer.pad_token_id
            ].unsqueeze(0)
            generated_ids: list[int] = []
            stopped_at_eos = False

            # 只接入模型预测的编号；参考回答仅用于最后显示。
            for _ in range(max_new_tokens):
                result = model(input_ids, last_token_only=True)
                next_ids = result.argmax(dim=-1)
                next_id = next_ids.item()
                if next_id == loaded_tokenizer.eos_token_id:
                    stopped_at_eos = True
                    break
                generated_ids.append(next_id)
                input_ids = torch.cat((input_ids, next_ids.unsqueeze(1)), dim=1)

            reply = loaded_tokenizer.decode(generated_ids)
            stop_reason = "EOS" if stopped_at_eos else f"达到 {max_new_tokens} 个词元上限"
            print(f"\n测试 {sample_number}：{prompt}")
            print(f"模型回答：{reply}")
            print(f"参考回答：{answer}")
            print(f"结束原因：{stop_reason}")

if __name__ == "__main__":
    main()
