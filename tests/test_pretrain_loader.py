"""检查远程正文进入训练、目标完整覆盖和两阶段衔接；离线运行。"""

from contextlib import redirect_stdout
from io import StringIO
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from datasets import IterableDataset
import torch
from torch import nn

import train
from parts.tokenizer import BPETokenizer
from parts.training_data import IGNORE_INDEX, make_loader, read_pretrain_records


class PretrainLoaderTests(unittest.TestCase):
    def test_text_is_preserved_and_metadata_is_ignored(self):
        text = "正文\nEnglish <bos> <pad> 🦊"
        self.assertEqual(list(read_pretrain_records([
            {"text": text, "url": "不参与训练", "score": 4.0},
        ])), [{"text": text}])
        for row in ({}, {"text": None}, {"text": " \n"}, {"text": 1}, None):
            with self.subTest(row=row), self.assertRaisesRegex(ValueError, "远程正文第 2 条"):
                list(read_pretrain_records([{"text": "合法"}, row]))

    def test_local_and_remote_targets_survive_chunking_padding_and_repeated_epochs(self):
        tokenizer = BPETokenizer()
        texts = ["本地", "Remote text. " * 8, "<bos><pad> 是正文"]
        stream = IterableDataset.from_generator(lambda: iter([
            {"text": text, "url": "忽略元数据"} for text in texts[1:]
        ]))
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "pretrain.jsonl"
            path.write_text(json.dumps({"text": texts[0]}) + "\n", encoding="utf-8")
            with patch("parts.training_data.load_dataset", return_value=stream) as load:
                loader = make_loader(path, "pretrain", tokenizer, batch_size=2,
                                     max_seq_len=7, shuffle_buffer=1, length_bucket_batches=0,
                                     pretrain_dataset="HuggingFaceFW/fineweb-edu",
                                     pretrain_config="sample-10BT")
                self.assertEqual(set(next(iter(loader.dataset.pretrain_records))), {"text"})
                expected = []
                for text in texts:
                    expected += tokenizer.encode(text, allow_special_tokens=False)
                    expected += [tokenizer.eos_token_id]
                for _ in range(2):
                    actual = []
                    for batch in loader:
                        ids, labels = batch["input_ids"], batch["labels"]
                        self.assertEqual(ids.shape, labels.shape)
                        self.assertLessEqual(ids.size(1), 7)
                        self.assertTrue(torch.all(labels[ids == tokenizer.pad_token_id] == IGNORE_INDEX))
                        actual.extend(labels[labels != IGNORE_INDEX].tolist())
                    self.assertEqual(actual, expected)
                load.assert_called_once_with("HuggingFaceFW/fineweb-edu", name="sample-10BT",
                                             split="train", streaming=True)

    def test_first_batch_does_not_materialize_remote_corpus(self):
        consumed = []
        def rows():
            for index in range(10000):
                consumed.append(index)
                yield {"text": "remote"}
        stream = IterableDataset.from_generator(rows)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "pretrain.jsonl"
            path.write_text('{"text":"local"}\n', encoding="utf-8")
            with patch("parts.training_data.load_dataset", return_value=stream):
                loader = make_loader(path, "pretrain", BPETokenizer(), batch_size=2,
                                     shuffle_buffer=1, length_bucket_batches=0,
                                     pretrain_dataset="HuggingFaceFW/fineweb-edu")
                self.assertEqual(consumed, [])
                batch = next(iter(loader))
                self.assertEqual(batch["input_ids"].size(0), 2)
                self.assertGreater(len(consumed), 0)
                self.assertLessEqual(len(consumed), 32)

    def test_instruction_stage_rejects_remote_pretrain_before_network_access(self):
        with patch("parts.training_data.load_dataset") as load:
            with self.assertRaisesRegex(ValueError, "pretrain 阶段"):
                make_loader(Path("unused"), "instruction", BPETokenizer(), batch_size=1,
                            pretrain_dataset="HuggingFaceFW/fineweb-edu",
                            instruction_dataset="BelleGroup/train_1M_CN")
            load.assert_not_called()

    def test_main_trains_both_sources_and_reuses_stream_between_epochs(self):
        class SmallModel(nn.Module):
            def __init__(self, vocab_size):
                super().__init__()
                self.embedding = nn.Embedding(vocab_size, 8)
                self.head = nn.Linear(8, vocab_size)
                self.seen = []

            def forward(self, ids, *, last_token_only=False):
                self.seen.extend(ids.detach().cpu().tolist())
                return self.head(self.embedding(ids))

        def remote_stream(dataset, **kwargs):
            rows = ([{"text": "Remote document", "url": "metadata"}]
                    if dataset == train.PRETRAIN_DATASET else
                    [{"instruction": "Remote question", "output": "Remote answer"}])
            return IterableDataset.from_generator(lambda: iter(rows))

        with tempfile.TemporaryDirectory() as directory, redirect_stdout(StringIO()):
            root = Path(directory)
            (root / "pretrain.jsonl").write_text('{"text":"Local document"}\n', encoding="utf-8")
            (root / "instruction.jsonl").write_text(
                '{"prompt":"Local question","answer":"Local answer"}\n', encoding="utf-8")
            tokenizer = BPETokenizer().train(["abc"], vocab_size=260)
            tokenizer.save(root / "tokenizer.json")
            model = SmallModel(tokenizer.vocab_size)
            before = model.head.weight.detach().clone()
            with patch.object(train, "TOKENIZER_PATH", root / "tokenizer.json"), \
                 patch.object(train, "Vital", return_value=model), \
                 patch.object(train, "TEST_TEXTS", []), \
                 patch("parts.training_data.load_dataset", side_effect=remote_stream) as load:
                train.main(["--data-dir", str(root), "--tokenizer", str(root / "tokenizer.json"),
                            "--output-dir", str(root / "checkpoint"), "--device", "cpu",
                            "--pretrain-epochs", "2", "--instruction-epochs", "1",
                            "--batch-size", "2", "--shuffle-buffer", "1",
                            "--length-bucket-batches", "0", "--skip-validation"])
                self.assertEqual(load.call_count, 2)
                self.assertEqual(load.call_args_list[0].args, (train.PRETRAIN_DATASET,))
                self.assertEqual(load.call_args_list[0].kwargs,
                                 {"name": train.PRETRAIN_DATASET_CONFIG, "split": "train", "streaming": True})
                self.assertEqual(load.call_args_list[1].args, (train.INSTRUCTION_DATASET,))
            contexts = [tokenizer.decode(ids) for ids in model.seen]
            self.assertEqual(contexts.count("Local document"), 2)
            self.assertEqual(contexts.count("Remote document"), 2)
            self.assertTrue(any("Remote question" in text and "Remote answer" in text for text in contexts))
            self.assertFalse(torch.equal(before, model.head.weight))
            self.assertTrue(torch.isfinite(model.head.weight).all())
            self.assertTrue((root / "checkpoint" / "model.pt").exists())


if __name__ == "__main__":
    unittest.main()
