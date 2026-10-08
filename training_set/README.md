# 正文和指令训练数据

`pretrain.jsonl` 是正文语料，每行一篇文档：

```json
{"text": "一段完整的正文。可以包含多种语言，也可以包含换行。"}
```

`instruction.jsonl` 是指令语料，每行一个请求和回答：

```json
{"prompt": "请计算 1+1。", "answer": "2"}
```

指令文件也接受 `instruction`、`input`、`output` 字段；非空 `input` 会追加到请求后。
请使用 UTF-8 编码，把正文中的换行写成 JSON 字符串中的 `\n`。
空行会跳过，缺失字段、非字符串或空白文本会报告文件名和行号。

正文文件目前包含 14 条自编的多语言演示文本，指令文件保留原先的 24 条问答，
并追加 100 条简短问答，合计 124 条。新增样本涵盖中英文问候和身份、双向翻译、
基础算术、分类提取、文本格式转换、阅读理解和简单推理。
这些样例用于验证流程，不是大规模预训练语料。可按相同格式继续追加实际训练数据。
独立验证集仍位于 `train.py` 的 `TEST_TEXTS`；它不参与词表或模型训练。

## 只使用本地 JSONL

`--usejsononly` 是布尔开关：添加即启用，不添加则保留本地加远程的行为。
启用后，正文阶段只读取 `--data-dir/pretrain.jsonl`，指令阶段只读取
`--data-dir/instruction.jsonl`；不会调用远程数据集加载器。
首次建词表或显式重建词表也只扫描启用阶段的本地 JSONL。
已有词表仍按原有规则复用；这个开关不改变分词器编号，也不改变训练轮数。

只训练当前 124 条指令、跳过正文预训练，例如：

```powershell
.venv\Scripts\python.exe train.py --usejsononly --pretrain-epochs 0 --instruction-epochs 100 --output-dir checkpoints/json-only
```

若还希望从本地指令重新建立规则词表，追加
`--tokenizer-mode rules --tokenizer checkpoints/json-only/local_rules.json`，首次运行时使用新路径。
这条命令从头训练模型，100 轮是实验设置，不保证收敛。检查时需分别比较训练原题和
未参与训练的改写问题；现有内置验证集仅用于独立生成检查。
若保留正文训练，将 `--pretrain-epochs` 设为正数即可。两份本地 JSONL 仍需存在以通过入口校验。

## FineWeb-Edu 正文数据接入

正文阶段同时使用本地 `pretrain.jsonl` 和
[`HuggingFaceFW/fineweb-edu`](https://huggingface.co/datasets/HuggingFaceFW/fineweb-edu)
的 `train` 划分。这是英文教育类网页正文语料，远程只取 `text` 字段；
网址、评分等元数据不参与训练。

`train.py` 中的 `PRETRAIN_DATASET` 控制远程正文来源，设为 `None` 可只训练本地正文。
`PRETRAIN_DATASET_CONFIG` 默认为 `default`（全量），可改为官方子集 `sample-10BT`。
使用 `streaming=True` 按需读取，不预先下载整份数据集；本地正文先进入数据流，随后是远程正文。
两者共用分词、连续分块、有限缓冲洗牌、长度分组和补齐，预测全部正文词元及文档末尾 EOS。
每个阶段只创建一次远程数据流，多轮训练重新迭代；格式错误会报告远程记录位置。
现有词表继续复用，首次建词表使用启用阶段的本地与远程选定范围。

当前远程正文和指令各选前 160,000 条原始记录；不设置 `--max-batches` 时，
一轮会遍历本地数据及这段选定范围。范围在洗牌、过滤和分词之前确定。
当前两阶段默认轮数分别由 `PRETRAIN_EPOCHS` 和 `INSTRUCTION_EPOCHS` 控制，均为 1000；
短运行应同时指定轮数和批次数上限，见下方快速检查命令。

## BELLE 指令数据接入

指令阶段的 `loader` 同时包含本地 `instruction.jsonl` 和
`BelleGroup/train_1M_CN` 的训练划分。远程数据使用 `streaming=True` 读取，
只选择 `instruction`、`output` 两列，分别映射为请求和回答；即使 `input` 非空也忽略。
远程记录中与 `TEST_TEXTS` 请求完全相同的样本会跳过。

每个阶段只创建一次 loader；多轮训练重新迭代数据流，不在每一轮重新调用 `load_dataset`。
本地指令先进入数据流，远程指令随后进入，共用有限的洗牌缓冲区、分词器、补齐和回答损失掩码。
远程数据读取需要网络，不会预先把百万条记录变成一个张量。

数据集名称位于 `train.py` 的 `INSTRUCTION_DATASET`，设为 `None` 可只使用本地指令。
现有词表继续复用；首次建词表覆盖启用阶段的本地和远程选定范围，不会自动重训旧词表。

2026-10-08 核查：Hugging Face 的非抽样全量列统计显示，`train` 的 917,424 条记录
中 `input` 全部为空字符串，非空为 0 条。因此当前保留只读 `instruction/output` 的逻辑。
证据和复查脚本为 `audit/belle_input_audit.json`、`audit/check_belle_input.py`；
此结论仅针对这个数据集，不代表其他指令数据集也没有有效 `input`。

## 默认训练流程

在项目根目录运行：

```powershell
.venv\Scripts\python.exe train.py --batch-size 4
```

1. 流式检查两份文件及验证请求是否混入指令集。
2. 加载固定的 `tokenizer.json`；文件不存在时，扫描启用阶段的本地和远程选定数据，
   默认建立汉字逐字、单词按空白/标点分隔的规则词表，不设词表数量上限。
3. 按指定轮数执行正文预训练：本地正文与 FineWeb-Edu 都对正文后续词元及 EOS 计算损失。
4. 在同一个模型上按指定轮数进行指令训练：请求仅提供上下文，只对回答和 EOS 计算损失。
5. 用独立请求生成回答，默认保留最后 256 个词元作为上下文，最多生成 128 个新词元，遇 EOS 提前停止；参考回答只用于显示，
   结果追加到原来的 `audit/validation.json`。

这里一轮是遍历该阶段全部训练样本，不再把一次参数更新称为一轮。
可以通过 `--pretrain-epochs` 和 `--instruction-epochs` 调整轮数；某阶段设为 0 可跳过，
但两份数据文件仍需存在，以供数据校验和首次词表训练。

## 分批与长文本

本地记录从磁盘逐行读取，远程记录流式读取，每次最多批量编码 32 条记录。
`train.py` 默认按最多 256 个输入词元连续分块，再通过有限的洗牌和长度分组缓冲区送入 DataLoader。
每次只有当前小批次的张量移动到 GPU/CPU 训练设备。
内存仍受单条记录长度、编码批次和洗牌缓冲区影响，但不会把全体语料转成一个设备张量。

每个批次只补齐到本批最长序列。补齐位置和指令请求位置的目标为 `-100`，不计算损失。
最后一个不足 4 条的批次仍参与训练。默认 `num_workers=0`，避免流式文件被多个工作进程重复读取。

`--max-seq-len` 控制训练分块长度和生成上下文长度。分块时下一块使用上一块最后预测的
词元作为首个上下文词元，所有有效目标仍各预测一次；纯请求块跳过，块之间不共享隐藏状态。
底层 `make_loader(max_seq_len=None)` 仍支持完整序列，训练命令行则要求正整数长度。

训练对每批所有有效位置同时预测下一词元，输入是完整序列去掉末尾，目标右移一位。
因果注意力遮罩阻止当前位置看到后面的正确答案；它仍然是下一词元预测训练。
指令中的 `<pad>` 等文字按普通字符编码，真正的控制编号由程序插入。

## 参数与快速检查

```powershell
.venv\Scripts\python.exe train.py --batch-size 2 --pretrain-epochs 1 --instruction-epochs 1 --max-batches 1 --skip-validation --output-dir checkpoints/smoke
```

该命令只训练正文和指令各一个批次；不代表训练完成。
`--max-batches` 限制每个阶段每一轮的批次数；不设置时遍历全部数据。
`--shuffle-buffer 1` 保持文件顺序；默认缓冲区为 256 个训练样本。
`--data-dir` 可指定包含这两份文件的其他目录，`--device cpu` 可指定 CPU。

`--max-seq-len` 默认 256，`--max-new-tokens` 默认 128，均要求正整数，不接受 `0`。
例如 `--max-seq-len 512 --max-new-tokens 128` 会把上下文长度改为 512。

添加语料后，默认继续复用原词表。需要重新学习词表时显式传入 `--retrain-tokenizer`。
普通词元编号可能改变，必须让模型权重和词元缓存与对应词表配套。
当前入口每次创建新的模型，尚未提供模型权重载入或断点续训功能。

切换规则分词器时建议指定新路径，保留原模型和原词表配套使用：

```powershell
.venv\Scripts\python.exe train.py --tokenizer-mode rules --tokenizer tokenizer_rules.json --output-dir checkpoints/rules --pretrain-epochs 1 --instruction-epochs 1
```

`auto` 复用旧格式、新建时采用规则格式；`rules` 明确要求规则格式；`bpe` 保留原有 BPE。
已有词表与明确指定的模式不一致时会报错，不会悄悄覆盖。新词表编码未知单词会退回字符，
未知字符退回字节，不在推理时新增编号。词表无限额不等于上下文无限长。
详细边界及独立建表命令见 `docs/tokenizer.md`。

## 训练结束后保存

所有启用的训练阶段完成后、生成验证回答之前，自动保存到项目下的 `checkpoints/latest/`：

- `model.pt`：完整的模型 `state_dict`，包含所有层的参数（含冻结参数）和持久缓冲区。
- `tokenizer.json`：本次训练使用的完整分词器，包含词表、合并规则与特殊词元配置。

`--skip-validation` 不影响保存。可用 `--output-dir checkpoints/run-1` 指定其他目录；
目录不存在时会创建，同名文件会覆盖。权重与该目录的分词器应一起保留。
这里只保存模型与分词器，不保存优化器状态，也不自动载入之前的权重。
