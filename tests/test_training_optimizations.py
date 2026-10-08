"""验证长度分组的覆盖率、SDPA 数值/遮罩，以及混合精度下的真实权重更新。"""

from collections import Counter
from contextlib import redirect_stderr, redirect_stdout
from io import StringIO
from itertools import islice
import math
import unittest
from unittest.mock import patch

import torch
from torch import nn

import train
import vital
import parts.decoder as decoder
from parts.tokenizer import BPETokenizer
from parts.training_data import (
    IGNORE_INDEX, PadBatch, encoded_samples, length_grouped_samples,
)


class LengthGroupingTests(unittest.TestCase):
    def test_grouping_reduces_padding_without_losing_targets_or_tail(self):
        # 长短交错的多语言指令，确保比较的是分词长度并保留回答/EOS 掩码。
        tokenizer = BPETokenizer()
        rows = [{"prompt": "问题 " + str(i), "answer": "日本語 abc " * repeats}
                for i, repeats in enumerate((1, 40, 2, 39, 3, 38, 4, 37, 5))]
        samples = list(encoded_samples(rows, "instruction", tokenizer))
        grouped = list(length_grouped_samples(iter(samples), 2, 8, 42))
        fingerprint = lambda items: Counter((tuple(x), tuple(y)) for x, y in items)
        self.assertEqual(fingerprint(grouped), fingerprint(samples))
        padding = lambda items: sum(
            len(batch) * max(len(x) for x, _ in batch) - sum(len(x) for x, _ in batch)
            for start in range(0, len(items), 2) if (batch := items[start:start + 2])
        )
        self.assertLess(padding(grouped), padding(samples))
        batches = [PadBatch(tokenizer.pad_token_id)(grouped[i:i + 2])
                   for i in range(0, len(grouped), 2)]
        self.assertEqual([batch["input_ids"].size(0) for batch in batches], [2, 2, 2, 2, 1])
        for batch in batches:
            self.assertTrue(torch.all(batch["labels"][batch["input_ids"] == 256] == IGNORE_INDEX))
        self.assertEqual(sum((batch["labels"] != IGNORE_INDEX).sum().item() for batch in batches),
                         sum(y != IGNORE_INDEX for _, labels in samples for y in labels))

    def test_buffer_is_bounded_and_reproducible_across_multiple_pools(self):
        consumed = []
        def source():
            for i in range(25):
                consumed.append(i)
                yield [i] * (1 + i % 9), [i] * (1 + i % 9)
        grouped = length_grouped_samples(source(), 3, 2, 42)
        first_batch = list(islice(grouped, 3))
        self.assertEqual(len(consumed), 6)
        result = first_batch + list(grouped)
        self.assertEqual(sorted(x[0] for x, _ in result), list(range(25)))
        self.assertEqual(result, list(length_grouped_samples(source(), 3, 2, 42)))
        self.assertNotEqual(result, list(length_grouped_samples(source(), 3, 2, 7)))
        original = list(source())
        self.assertEqual(list(length_grouped_samples(iter(original), 3, 0, 42)), original)
        self.assertEqual(list(length_grouped_samples([], 3, 2, 42)), [])


class SdpaTests(unittest.TestCase):
    def test_vital_omits_empty_pad_mask_and_preserves_causal_outputs_and_gradients(self):
        with patch.object(vital, "dimension_word", 32), \
             patch.object(decoder, "dimension_word", 32), \
             patch.object(decoder, "num_heads", 4), \
             patch.object(vital, "Decoder", side_effect=decoder.attention):
            model = vital.Vital(260)
            ids = torch.tensor([[257, 97, 98]])
            sdpa = decoder.F.scaled_dot_product_attention
            with patch.object(decoder.F, "scaled_dot_product_attention", wraps=sdpa) as call:
                actual = model(ids, last_token_only=False)
            self.assertEqual(call.call_count, 6)
            for invocation in call.call_args_list:
                self.assertTrue(invocation.kwargs["is_causal"])
                self.assertNotIn("attn_mask", invocation.kwargs)
            gradients = torch.autograd.grad(actual.square().sum(), tuple(model.parameters()))

            # 右侧加 PAD 强制走原显式掩码路径，原位置的结果和梯度应相同。
            padded_ids = torch.tensor([[257, 97, 98, 256]])
            with patch.object(decoder.F, "scaled_dot_product_attention", wraps=sdpa) as call:
                expected = model(padded_ids, last_token_only=False)[:, :3]
            self.assertEqual(call.call_count, 6)
            self.assertTrue(all("attn_mask" in c.kwargs for c in call.call_args_list))
            expected_gradients = torch.autograd.grad(expected.square().sum(), tuple(model.parameters()))
            torch.testing.assert_close(actual, expected, rtol=1e-4, atol=1e-5)
            for actual_g, expected_g in zip(gradients, expected_gradients):
                torch.testing.assert_close(actual_g, expected_g, rtol=1e-4, atol=1e-5)

    def test_matches_original_attention_outputs_and_gradients_for_all_pad_layouts(self):
        torch.manual_seed(10)
        masks = [None, torch.tensor([
            [False, False, True, True],   # 右侧 PAD
            [True, True, False, False],   # 左侧 PAD，有完全被遮住的查询行
            [False, True, False, True],   # 中间 PAD
            [True, True, True, True],     # 全 PAD
        ])]
        with patch.object(decoder, "dimension_word", 32), patch.object(decoder, "num_heads", 4):
            layer = decoder.attention().double()
            for pad_mask in masks:
                with self.subTest(pad_mask=pad_mask):
                    x = torch.randn(4, 4, 32, dtype=torch.float64, requires_grad=True)
                    parameters = (x, *layer.parameters())
                    actual = layer(x, pad_mask)
                    actual_grad = torch.autograd.grad(actual.square().sum(), parameters)

                    def original_attention(q, k, v, **kwargs):
                        # 独立使用旧公式及外部 PAD 位置，不复用新实现传入的 allowed 遮罩。
                        blocked = torch.ones(4, 4, dtype=torch.bool).triu(1)
                        if pad_mask is not None:
                            blocked = blocked[None, None] | pad_mask[:, None, None, :]
                        scores = (q @ k.transpose(-2, -1)) / math.sqrt(q.size(-1))
                        scores = scores.masked_fill(blocked, -torch.inf)
                        empty = blocked.all(-1, keepdim=True)
                        weights = scores.masked_fill(empty, 0).softmax(-1).masked_fill(empty, 0)
                        return weights @ v

                    with patch.object(decoder.F, "scaled_dot_product_attention", original_attention):
                        expected = layer(x, pad_mask)
                        expected_grad = torch.autograd.grad(expected.square().sum(), parameters)
                    torch.testing.assert_close(actual, expected, rtol=1e-8, atol=1e-9)
                    for actual_g, expected_g in zip(actual_grad, expected_grad):
                        self.assertTrue(torch.isfinite(actual_g).all())
                        torch.testing.assert_close(actual_g, expected_g, rtol=1e-7, atol=1e-8)

    def test_future_and_pad_values_cannot_change_valid_outputs(self):
        with patch.object(decoder, "dimension_word", 32), patch.object(decoder, "num_heads", 4):
            layer = decoder.attention().eval()
            x = torch.randn(1, 6, 32)
            mask = torch.tensor([[False, True, False, False, False, True]])
            expected = layer(x, mask)
            changed = x.clone()
            changed[:, 3:] += torch.randn_like(changed[:, 3:]) * 10
            changed[:, 1] += 100
            actual = layer(changed, mask)
            torch.testing.assert_close(actual[:, [0, 2]], expected[:, [0, 2]])


class PrecisionTests(unittest.TestCase):
    def test_accumulated_loss_preserves_token_weighting_and_logging(self):
        class Probe(nn.Module):
            def __init__(self):
                super().__init__()
                self.scores = nn.Parameter(torch.tensor([0.2, -0.3, 1.1]))

            def forward(self, ids, *, last_token_only=False):
                return self.scores.expand(*ids.shape, 3)

        batches = [
            {"input_ids": torch.zeros(1, 3, dtype=torch.long),
             "labels": torch.tensor([[0, IGNORE_INDEX, IGNORE_INDEX]])},
            {"input_ids": torch.zeros(1, 3, dtype=torch.long),
             "labels": torch.tensor([[1, 2, 1]])},
        ]
        for interval in (0, 1, 3):
            model = Probe()
            optimizer = torch.optim.SGD(model.parameters(), lr=0.0)
            expected = sum(nn.functional.cross_entropy(
                model(batch["input_ids"]).reshape(-1, 3), batch["labels"].flatten()
            ).item() * int((batch["labels"] != IGNORE_INDEX).sum()) for batch in batches) / 4
            output = StringIO()
            with redirect_stdout(output):
                result = train.train_epoch(model, batches, optimizer, torch.device("cpu"),
                                           log_interval=interval)
            self.assertEqual((result["steps"], result["tokens"]), (2, 4))
            self.assertAlmostEqual(result["loss"], expected, places=12)
            self.assertEqual(len(output.getvalue().splitlines()), 2 if interval == 1 else 0)
        with torch.no_grad():
            model.scores.fill_(float("nan"))
        with patch.object(optimizer, "step") as step:
            with self.assertRaisesRegex(RuntimeError, "不是有限数值"):
                train.train_epoch(model, batches, optimizer, torch.device("cpu"))
            step.assert_not_called()

    def test_options_and_cpu_precision_resolution(self):
        options = train.parse_args(["--mixed-precision", "bf16", "--length-bucket-batches", "8"])
        self.assertEqual((options.mixed_precision, options.length_bucket_batches), ("bf16", 8))
        with redirect_stderr(StringIO()), self.assertRaises(SystemExit):
            train.parse_args(["--length-bucket-batches", "-1"])
        cpu = torch.device("cpu")
        self.assertEqual(train.resolve_mixed_precision(cpu, "auto"), "off")
        self.assertEqual(train.resolve_mixed_precision(cpu, "bf16"), "bf16")
        with self.assertRaisesRegex(ValueError, "CUDA"):
            train.resolve_mixed_precision(cpu, "fp16")

    def run_training_check(self, device, mode):
        # 使用实际注意力和前馈层，检查 autocast、SDPA、反向与优化器协同工作。
        # 只缩小注意力维度；不读取或覆盖用户的模型权重。
        with patch.object(decoder, "dimension_word", 32), patch.object(decoder, "num_heads", 4):
            class ProbeModel(nn.Module):
                def __init__(self):
                    super().__init__()
                    self.embedding = nn.Embedding(260, 32)
                    self.block = decoder.Decoder()
                    self.head = nn.Linear(32, 260)
                    self.output_dtypes = []

                def forward(self, ids, *, last_token_only=False):
                    hidden = self.block(self.embedding(ids), pad_mask=ids == 256)
                    logits = self.head(hidden)
                    self.output_dtypes.append(logits.dtype)
                    return logits

            torch.manual_seed(123)
            model = ProbeModel().to(device)
            optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
            before = model.head.weight.detach().clone()
            batch = {"input_ids": torch.tensor([[257, 97, 98, 256], [257, 99, 256, 256]]),
                     "labels": torch.tensor([[IGNORE_INDEX, 98, 258, IGNORE_INDEX],
                                             [99, 258, IGNORE_INDEX, IGNORE_INDEX]])}
            scaler = torch.amp.GradScaler(device.type, enabled=mode == "fp16", init_scale=128)
            # 同一个 scaler 连续用于两轮，覆盖 main 的复用方式。
            for _ in range(2):
                metrics = train.train_epoch(model, [batch], optimizer, device, log_interval=0,
                                            mixed_precision=mode, scaler=scaler)
                self.assertEqual((metrics["steps"], metrics["tokens"]), (1, 4))
                self.assertTrue(math.isfinite(metrics["loss"]))
            expected_dtype = {"off": torch.float32, "bf16": torch.bfloat16, "fp16": torch.float16}[mode]
            self.assertEqual(model.output_dtypes, [expected_dtype, expected_dtype])
            self.assertFalse(torch.equal(before, model.head.weight))
            for parameter in model.parameters():
                self.assertEqual(parameter.dtype, torch.float32)
                self.assertTrue(torch.isfinite(parameter).all())
                self.assertTrue(torch.isfinite(parameter.grad).all())
            # PAD 不参与监督，且注意力不读取它，其嵌入梯度必须为零。
            self.assertEqual(model.embedding.weight.grad[256].abs().sum().item(), 0)

    def test_cpu_fp32_and_bf16_updates(self):
        for mode in ("off", "bf16"):
            with self.subTest(mode=mode):
                self.run_training_check(torch.device("cpu"), mode)

    @unittest.skipUnless(torch.cuda.is_available(), "需要 CUDA")
    def test_cuda_mixed_precision_updates(self):
        device = torch.device("cuda")
        modes = ["fp16"]
        if torch.cuda.is_bf16_supported():
            modes.append("bf16")
        for mode in modes:
            with self.subTest(mode=mode):
                self.run_training_check(device, mode)


if __name__ == "__main__":
    unittest.main()
