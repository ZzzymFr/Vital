"""用小模型和本地整篇样本验证续训入口，不访问远程数据或真实检查点。"""

from contextlib import ExitStack, redirect_stdout
import io
import math
from pathlib import Path
import tempfile
import unittest
import warnings
from unittest.mock import patch

import torch
from torch import nn
from torch.utils.data import IterableDataset

import continue_study
from parts.tokenizer import BPETokenizer
from parts.training_data import encoded_samples


class TinyModel(nn.Module):
    def __init__(self, vocab_size):
        super().__init__()
        self.vocab_chart = nn.Embedding(vocab_size, 4)
        self.head = nn.Linear(4, vocab_size)
        self.batch_shapes = []
        self.output_dtypes = []

    def forward(self, inputs, *, last_token_only=True):
        if last_token_only:
            raise AssertionError("训练必须预测所有位置")
        self.batch_shapes.append(tuple(inputs.shape))
        logits = self.head(self.vocab_chart(inputs))
        self.output_dtypes.append(logits.dtype)
        return logits


class LocalDocuments(IterableDataset):
    def __init__(self, rows):
        self.rows = rows
        self.epochs = []

    def set_epoch(self, epoch):
        self.epochs.append(epoch)

    def __iter__(self):
        yield from self.rows


class ContinueStudyTests(unittest.TestCase):
    def test_training_updates_weights_keeps_whole_documents_and_saves_pair(self):
        tokenizer = BPETokenizer()
        samples = list(encoded_samples(
            [{"text": "A" * 600}], "pretrain", tokenizer,
        ))
        samples += list(encoded_samples(
            [{"prompt": "问题", "answer": "好的"},
             {"prompt": "另一问题", "answer": "回答"}],
            "instruction", tokenizer,
        ))
        rows = [{"input_ids": inputs, "labels": labels} for inputs, labels in samples]
        cases = [
            ("auto", False, None, "off"),
            ("bf16", False, 1, "bf16"),
        ]
        if torch.cuda.is_available():
            cases += [
                ("auto", True, 1, "bf16" if torch.cuda.is_bf16_supported() else "fp16"),
                ("fp16", True, 1, "fp16"),
            ]
        for mode, use_cuda, max_batches, expected_precision in cases:
            expected_shapes = ([(2, 601), (1, len(samples[2][0]))] * 2
                               if max_batches is None else [(2, 601)] * 2)
            with self.subTest(mode=mode, cuda=use_cuda), tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                tokenizer_path = root / "source-tokenizer.json"
                tokenizer.save(tokenizer_path)
                model = TinyModel(tokenizer.vocab_size)
                original = {name: value.detach().clone() for name, value in model.state_dict().items()}
                dataset = LocalDocuments(rows)
                output = root / "continued"
                actual_train_epoch = continue_study.train_epoch
                with ExitStack() as stack:
                    for name, value in {
                        "TOKENISER_DIR": tokenizer_path, "OUTPUT_DIR": output,
                        "EPOCHS": 2, "BATCH_SIZE": 2, "MAX_BATCH": max_batches,
                        "LOG_INTERVAL": 0, "LEARNING_RATE": 0.001,
                        "MINIMUM_LEARNING_RATE": 0.0001,
                        "MIXED_PRECISION": mode,
                    }.items():
                        stack.enter_context(patch.object(continue_study, name, value))
                    stack.enter_context(patch("continue_study.torch.cuda.is_available", return_value=use_cuda))
                    stack.enter_context(patch("continue_study.torch.load", return_value=original))
                    stack.enter_context(patch("continue_study.vital.Vital", return_value=model))
                    remote = stack.enter_context(patch("continue_study.load_dataset", return_value=object()))
                    mixed = stack.enter_context(patch("continue_study.make_mixed_document_dataset", return_value=dataset))
                    training = stack.enter_context(patch("continue_study.train_epoch", wraps=actual_train_epoch))
                    stack.enter_context(redirect_stdout(io.StringIO()))
                    caught_warnings = stack.enter_context(warnings.catch_warnings(record=True))
                    warnings.simplefilter("always")
                    continue_study.main()
                self.assertFalse(any("lr_scheduler.step()" in str(w.message) for w in caught_warnings))
                self.assertEqual(dataset.epochs, [0, 1])
                self.assertEqual(mixed.call_args.kwargs["max_seq_len"], continue_study.MAX_SEQ_LEN)
                self.assertEqual(mixed.call_args.args[2]["document_limit"], 2_000_000)
                self.assertEqual(mixed.call_args.args[2]["start_document"],
                                 continue_study.PRETRAIN_REMAINING_DATA["start_document"])
                self.assertEqual(mixed.call_args.args[3]["start_document"], 0)
                self.assertIsNone(mixed.call_args.args[3]["document_limit"])
                self.assertEqual(remote.call_args.args, ("secemp9/arxiv-complete", "paper_text"))
                self.assertEqual(model.batch_shapes, expected_shapes)
                self.assertEqual(training.call_count, 2)
                expected_dtype = {"off": torch.float32, "bf16": torch.bfloat16,
                                  "fp16": torch.float16}[expected_precision]
                self.assertEqual(model.output_dtypes, [expected_dtype] * len(expected_shapes))
                first_scaler = training.call_args_list[0].kwargs["scaler"]
                self.assertEqual(first_scaler.is_enabled(), expected_precision == "fp16")
                for call in training.call_args_list:
                    self.assertEqual(call.kwargs["mixed_precision"], expected_precision)
                    self.assertIs(call.kwargs["scaler"], first_scaler)
                optimizer = training.call_args.args[2]
                actual_steps = max(int(state["step"].item()) for state in optimizer.state.values())
                # 每轮一批时，FP16 的溢出跳步也应暂停该轮学习率调度。
                scheduled_epochs = min(2, actual_steps)
                expected_lr = 0.0001 + (0.001 - 0.0001) * (1 + math.cos(math.pi * scheduled_epochs / 2)) / 2
                self.assertAlmostEqual(optimizer.param_groups[0]["lr"], expected_lr)
                self.assertEqual(optimizer.param_groups[0]["weight_decay"], continue_study.WEIGHT_DECAY)
                self.assertTrue(remote.call_args.kwargs["streaming"])
                saved = torch.load(output / "model.pt", weights_only=True)
                self.assertTrue(any(not torch.equal(saved[name].cpu(), value) for name, value in original.items()))
                for name, value in model.state_dict().items():
                    self.assertEqual(value.dtype, torch.float32)
                    self.assertTrue(torch.isfinite(value).all())
                    torch.testing.assert_close(saved[name], value)
                self.assertEqual((output / "tokenizer.json").read_bytes(), tokenizer_path.read_bytes())


if __name__ == "__main__":
    unittest.main()
