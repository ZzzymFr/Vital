"""检查正文/指令目标对齐、流式分批、真实反向传播和两阶段衔接。"""

from contextlib import redirect_stdout, redirect_stderr
from io import StringIO
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import torch
from torch import nn

import train
import vital
import parts.decoder as decoder
from parts.tokenizer import BPETokenizer, RuleTokenizer
from parts.training_data import (
    IGNORE_INDEX, buffered_shuffle, encode_prompt, encoded_samples, make_loader,
    read_records, tokenizer_texts,
)


def write_rows(path, rows):
    path.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows), encoding="utf-8")


class RecordingModel(nn.Module):
    def __init__(self, vocab_size):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, 8)
        self.head = nn.Linear(8, vocab_size)
        self.calls = []

    def forward(self, ids, *, last_token_only=True):
        self.calls.append((ids.detach().clone(), last_token_only))
        scores = self.head(self.embedding(ids))
        return scores[:, -1] if last_token_only else scores


class TrainingDataTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # 使用无合并的字节兜底，让标签编号容易逐位置核对。
        cls.tokenizer = BPETokenizer().train(["abc 中文 Hello"], vocab_size=260)

    def test_pretrain_shift_and_long_document_coverage(self):
        text = "中文正文 abc def。" * 9
        samples = list(encoded_samples([{"text": text}], "pretrain", self.tokenizer, 7))
        expected = self.tokenizer.encode(text, allow_special_tokens=False) + [258]
        self.assertEqual([value for _, labels in samples for value in labels], expected)
        full = [257] + expected
        start = 0
        for inputs, labels in samples:
            self.assertGreater(len(inputs), 0)
            self.assertLessEqual(len(inputs), 7)
            self.assertEqual(inputs, full[start:start + len(inputs)])
            self.assertEqual(labels, full[start + 1:start + 1 + len(inputs)])
            start += len(inputs)

    def test_default_preserves_complete_long_pretrain_and_instruction(self):
        text = "abc " * 200
        samples = list(encoded_samples([{"text": text}], "pretrain", self.tokenizer))
        ids = self.tokenizer.encode(text, allow_special_tokens=False)
        self.assertEqual(samples, [([257] + ids, ids + [258])])
        self.assertGreater(len(samples[0][0]), 256)

        prompt, answer = "p" * 400, "a" * 200
        prefix = encode_prompt(self.tokenizer, prompt)
        answer_ids = self.tokenizer.encode(answer, allow_special_tokens=False)
        samples = list(encoded_samples([{"prompt": prompt, "answer": answer}],
                                       "instruction", self.tokenizer))
        self.assertEqual(samples, [(prefix + answer_ids,
                                   [IGNORE_INDEX] * (len(prefix) - 1) + answer_ids + [258])])

    def test_loader_without_limit_keeps_long_sequences_and_batching(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "pretrain.jsonl"
            texts = ["a" * 600, "短", "b" * 300]
            write_rows(path, [{"text": text} for text in texts])
            batches = list(make_loader(path, "pretrain", self.tokenizer,
                                       batch_size=2, shuffle_buffer=1, length_bucket_batches=0))
            self.assertEqual([batch["input_ids"].shape for batch in batches],
                             [torch.Size([2, 601]), torch.Size([1, 301])])
            actual_targets = sum((batch["labels"] != IGNORE_INDEX).sum().item() for batch in batches)
            self.assertEqual(actual_targets, sum(len(text.encode("utf-8")) + 1 for text in texts))

    def test_cli_defaults_and_positive_length_limits(self):
        defaults = train.parse_args([])
        self.assertEqual(defaults.max_seq_len, train.MAX_SEQ_LEN)
        self.assertEqual(defaults.max_new_tokens, 128)
        bounded = train.parse_args(["--max-seq-len", "64", "--max-new-tokens", "10"])
        self.assertEqual((bounded.max_seq_len, bounded.max_new_tokens), (64, 10))
        for flag in ("--max-seq-len", "--max-new-tokens"):
            for invalid in ("-1", "0"):
                with redirect_stderr(StringIO()), self.assertRaises(SystemExit):
                    train.parse_args([flag, invalid])

    def test_instruction_masks_prompt_and_includes_eos(self):
        prompt, answer = "计算 1+1", "2"
        prefix = encode_prompt(self.tokenizer, prompt)
        target = self.tokenizer.encode(answer, allow_special_tokens=False) + [258]
        samples = list(encoded_samples([{"prompt": prompt, "answer": answer}],
                                       "instruction", self.tokenizer, 128))
        self.assertEqual(len(samples), 1)
        inputs, labels = samples[0]
        self.assertEqual(inputs, (prefix + target)[:-1])
        self.assertEqual(labels, [IGNORE_INDEX] * (len(prefix) - 1) + target)
        self.assertEqual(inputs[len(prefix) - 1], 259)

    def test_long_instruction_never_loses_answer_targets(self):
        prompt, answer = "一个很长的请求" * 20, "分步说明 abc " * 20
        samples = list(encoded_samples([{"prompt": prompt, "answer": answer}],
                                       "instruction", self.tokenizer, 11))
        actual = [value for _, labels in samples for value in labels if value != IGNORE_INDEX]
        self.assertEqual(actual, self.tokenizer.encode(answer, allow_special_tokens=False) + [258])
        self.assertTrue(all(any(value != IGNORE_INDEX for value in labels) for _, labels in samples))
        self.assertTrue(all(len(inputs) <= 11 for inputs, _ in samples))

    def test_literal_control_markers_are_not_pad_or_role_tokens(self):
        row = {"prompt": "解释 <pad><assistant>", "answer": "<eos> 是标记"}
        samples = list(encoded_samples([row], "instruction", self.tokenizer, 256))
        inputs, labels = samples[0]
        self.assertNotIn(256, inputs)
        self.assertEqual(inputs.count(259), 1)
        self.assertEqual(inputs.count(257), 1)
        expected = self.tokenizer.encode(row["answer"], allow_special_tokens=False) + [258]
        self.assertEqual([label for label in labels if label != IGNORE_INDEX], expected)

    def test_encoding_only_reads_one_bounded_group(self):
        consumed = []
        def records():
            for index in range(10000):
                consumed.append(index)
                yield {"text": "abc " * 100}
        iterator = encoded_samples(records(), "pretrain", self.tokenizer, 16, encode_batch_size=3)
        next(iterator)
        self.assertEqual(consumed, [0, 1, 2])

    def test_minibatch_padding_tail_and_repeat_iteration(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "pretrain.jsonl"
            write_rows(path, [{"text": text} for text in ("a", "abc", "你好", "x", "tail")])
            loader = make_loader(path, "pretrain", self.tokenizer, batch_size=2,
                                 max_seq_len=128, shuffle_buffer=1, length_bucket_batches=0)
            batches = list(loader)
            self.assertEqual([batch["input_ids"].size(0) for batch in batches], [2, 2, 1])
            self.assertEqual(len(list(loader)), 3)
            for batch in batches:
                x, y = batch["input_ids"], batch["labels"]
                self.assertEqual(x.shape, y.shape)
                self.assertEqual(x.dtype, torch.long)
                self.assertEqual(x.device.type, "cpu")
                self.assertTrue(torch.all(y[x == 256] == IGNORE_INDEX))
            self.assertEqual(batches[0]["input_ids"].size(1), 4)

    def test_shuffle_preserves_samples_and_is_reproducible(self):
        samples = list(range(100))
        result = list(buffered_shuffle(iter(samples), 5, 42))
        self.assertEqual(sorted(result), samples)
        self.assertEqual(result, list(buffered_shuffle(iter(samples), 5, 42)))
        self.assertEqual(list(buffered_shuffle(iter(samples), 1, 42)), samples)

    def test_file_errors_and_validation_leakage(self):
        with tempfile.TemporaryDirectory() as directory:
            data_dir = Path(directory)
            pretrain = data_dir / "pretrain.jsonl"
            instruction = data_dir / "instruction.jsonl"
            write_rows(pretrain, [{"text": "正文"}])
            write_rows(instruction, [{"instruction": "问", "input": "上下文", "output": "答"}])
            self.assertEqual(list(read_records(instruction, "instruction")),
                             [{"prompt": "问\n上下文", "answer": "答"}])
            self.assertEqual(list(tokenizer_texts(data_dir)), ["正文", "用户：问\n上下文\n", "答"])
            self.assertEqual(train.validate_training_data(data_dir), {"pretrain": 1, "instruction": 1})
            write_rows(instruction, [{"prompt": train.TEST_TEXTS[0][0], "answer": "答"}])
            with self.assertRaisesRegex(ValueError, "验证请求"):
                train.validate_training_data(data_dir)
            write_rows(pretrain, [{"text": " "}])
            with self.assertRaisesRegex(ValueError, r"pretrain.jsonl:1"):
                list(read_records(pretrain, "pretrain"))

    def test_train_epoch_updates_weights_and_honors_batch_limit(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "pretrain.jsonl"
            write_rows(path, [{"text": "abc"}] * 5)
            loader = make_loader(path, "pretrain", self.tokenizer, batch_size=2,
                                 max_seq_len=16, shuffle_buffer=1)
            model = RecordingModel(260)
            optimizer = torch.optim.AdamW(model.parameters(), lr=0.01)
            before = model.head.weight.detach().clone()
            metrics = train.train_epoch(model, loader, optimizer, torch.device("cpu"),
                                        max_batches=1, log_interval=0)
            self.assertEqual(metrics["steps"], 1)
            self.assertEqual(metrics["tokens"], 8)
            self.assertEqual(len(model.calls), 1)
            self.assertFalse(model.calls[0][1])
            self.assertEqual(model.calls[0][0].shape, (2, 4))
            self.assertFalse(torch.equal(before, model.head.weight))

    def test_main_runs_pretrain_then_instruction_on_same_model(self):
        with tempfile.TemporaryDirectory() as directory:
            data_dir = Path(directory)
            write_rows(data_dir / "pretrain.jsonl", [{"text": "abc 正文"}] * 3)
            write_rows(data_dir / "instruction.jsonl", [{"prompt": "请求", "answer": "回答"}] * 3)
            created = []
            def create_model(vocab_size):
                model = RecordingModel(vocab_size)
                created.append(model)
                return model
            stream = StringIO()
            with patch.object(train, "Vital", side_effect=create_model), \
                 patch.object(train, "TOKENIZER_PATH", data_dir / "tokenizer.json"), \
                 patch.object(train, "PRETRAIN_DATASET", None), \
                 patch("parts.training_data.load_dataset") as remote, \
                 redirect_stdout(stream):
                from datasets import IterableDataset
                remote.return_value = IterableDataset.from_generator(
                    lambda: iter([{"instruction": "远程请求", "output": "远程回答", "input": "忽略"}])
                )
                train.main(["--data-dir", str(data_dir), "--device", "cpu", "--batch-size", "2",
                            "--max-seq-len", "64", "--max-batches", "1", "--shuffle-buffer", "1",
                            "--pretrain-epochs", "1", "--instruction-epochs", "1",
                            "--output-dir", str(data_dir / "checkpoint"),
                            "--skip-validation"])
                remote.assert_called_once_with("BelleGroup/train_1M_CN", split="train", streaming=True)
            self.assertEqual(len(created), 1)
            self.assertEqual(len(created[0].calls), 2)
            self.assertLess(stream.getvalue().index("正文预训练："), stream.getvalue().index("指令训练："))
            saved = torch.load(data_dir / "checkpoint" / "model.pt", map_location="cpu", weights_only=True)
            self.assertEqual(saved.keys(), created[0].state_dict().keys())
            for name, parameter in created[0].state_dict().items():
                torch.testing.assert_close(saved[name], parameter, rtol=0, atol=0)
            original = BPETokenizer.load(data_dir / "tokenizer.json")
            self.assertIsInstance(original, RuleTokenizer)
            # 新建词表确实扫描了远程样本，而非仅使用本地的“请求/回答”。
            self.assertEqual(len(original.encode("远")), 1)
            self.assertEqual(len(original.encode("程")), 1)
            restored = BPETokenizer.load(data_dir / "checkpoint" / "tokenizer.json")
            texts = ["中文 English 日本語 🦊", "<bos><assistant>回答<eos>"]
            self.assertEqual(restored.encode_batch(texts), original.encode_batch(texts))

    def test_save_model_includes_frozen_parameters_buffers_and_tokenizer(self):
        model = RecordingModel(self.tokenizer.vocab_size)
        model.embedding.weight.requires_grad_(False)
        model.register_buffer("saved_counter", torch.tensor(7))
        with tempfile.TemporaryDirectory() as directory, redirect_stdout(StringIO()):
            output_dir = Path(directory) / "nested" / "checkpoint"
            train.save_model(model, self.tokenizer, output_dir)
            saved = torch.load(output_dir / "model.pt", map_location="cpu", weights_only=True)
            restored_model = RecordingModel(self.tokenizer.vocab_size)
            restored_model.register_buffer("saved_counter", torch.tensor(0))
            restored_model.load_state_dict(saved, strict=True)
            self.assertEqual(restored_model.saved_counter.item(), 7)
            tokens = torch.tensor([[257, 97, 98, 259]])
            torch.testing.assert_close(restored_model(tokens), model(tokens), rtol=0, atol=0)
            restored = BPETokenizer.load(output_dir / "tokenizer.json")
            self.assertEqual(restored.vocab, self.tokenizer.vocab)
            self.assertEqual(restored.merges, self.tokenizer.merges)
            self.assertEqual(restored.encode("中文 English 🦊"), self.tokenizer.encode("中文 English 🦊"))

    def test_real_vital_all_position_loss_is_causal_and_backpropagates(self):
        # 缩小隐藏维度以快速验证真实模型路径，不改仓库的模型超参。
        with patch.object(vital, "dimension_word", 32), \
             patch.object(decoder, "dimension_word", 32), patch.object(decoder, "num_heads", 4):
            model = vital.Vital(260)
            first = torch.tensor([[257, 97, 98, 99]])
            second = torch.tensor([[257, 97, 98, 100]])
            model.eval()
            logits = model(first, last_token_only=False)
            other = model(second, last_token_only=False)
            torch.testing.assert_close(logits[:, :3], other[:, :3])
            self.assertEqual(logits.shape, (1, 4, 260))
            loss = nn.CrossEntropyLoss()(logits.reshape(-1, 260), torch.tensor([97, 98, 99, 258]))
            loss.backward()
            self.assertTrue(torch.isfinite(loss))
            self.assertTrue(torch.isfinite(model.vocab_chart.weight.grad).all())
            self.assertGreater(model.vocab_chart.weight.grad.abs().sum().item(), 0)

    def test_generation_uses_prompt_only_and_stops_at_eos(self):
        seen = []
        class EosModel(nn.Module):
            def forward(self, ids, *, last_token_only=True):
                seen.append(ids.tolist()[0])
                result = torch.zeros((1, 260))
                result[0, 258] = 10
                return result
        with patch.object(train, "TEST_TEXTS", [("请求", "不可泄露的参考回答")]), redirect_stdout(StringIO()):
            records = train.evaluate_replies(EosModel(), self.tokenizer, torch.device("cpu"), 128, 5)
        self.assertEqual(seen, [encode_prompt(self.tokenizer, "请求")])
        self.assertEqual(records[0]["结束原因"], "EOS")
        self.assertEqual(records[0]["模型回答"], "")

    def test_explicit_large_generation_limits_keep_full_context(self):
        seen = []
        class DelayedEosModel(nn.Module):
            def forward(self, ids, *, last_token_only=True):
                seen.append(ids.tolist()[0])
                scores = torch.zeros((1, 260))
                scores[0, 97 if len(seen) <= 160 else 258] = 10
                return scores
        prompt = "prompt " * 50
        prefix = encode_prompt(self.tokenizer, prompt)
        self.assertGreater(len(prefix), 256)
        with patch.object(train, "TEST_TEXTS", [(prompt, "参考回答")]), redirect_stdout(StringIO()):
            records = train.evaluate_replies(DelayedEosModel(), self.tokenizer, torch.device("cpu"),
                                             max_seq_len=len(prefix) + 160, max_new_tokens=200)
        self.assertEqual(len(seen), 161)
        self.assertEqual(seen[0], prefix)
        self.assertEqual(seen[-1], prefix + [97] * 160)
        self.assertEqual(records[0]["模型回答"], "a" * 160)
        self.assertEqual(records[0]["结束原因"], "EOS")

    def test_explicit_generation_limits_remain_optional(self):
        lengths = []
        class NeverEosModel(nn.Module):
            def forward(self, ids, *, last_token_only=True):
                lengths.append(ids.size(1))
                scores = torch.zeros((1, 260))
                scores[0, 97] = 10
                return scores
        with patch.object(train, "TEST_TEXTS", [("长请求" * 20, "参考回答")]), redirect_stdout(StringIO()):
            records = train.evaluate_replies(NeverEosModel(), self.tokenizer, torch.device("cpu"),
                                             max_seq_len=8, max_new_tokens=3)
        self.assertEqual(lengths, [8, 8, 8])
        self.assertEqual(records[0]["模型回答"], "aaa")
        self.assertEqual(records[0]["结束原因"], "达到 3 个词元上限")


if __name__ == "__main__":
    unittest.main()
