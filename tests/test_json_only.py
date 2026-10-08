"""本地开关覆盖建词表、缓存和训练两阶段，远程入口一旦调用即失败。"""

from contextlib import redirect_stdout
from io import StringIO
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import torch
from torch import nn

import train
from parts.tokenizer import BPETokenizer, RuleTokenizer


class SmallModel(nn.Module):
    def __init__(self, vocab_size):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, 8)
        self.head = nn.Linear(8, vocab_size)
        self.calls = 0

    def forward(self, ids, *, last_token_only=False):
        self.calls += 1
        return self.head(self.embedding(ids))


class JsonOnlyTests(unittest.TestCase):
    def test_switch_is_opt_in(self):
        self.assertFalse(train.parse_args([]).usejsononly)
        self.assertTrue(train.parse_args(["--usejsononly"]).usejsononly)

    def test_local_only_builds_vocabulary_and_trains_without_remote_loading(self):
        for pretrain_epochs, reuse_tokenizer in ((1, False), (0, False), (1, True)):
            with self.subTest(pretrain=pretrain_epochs, reuse=reuse_tokenizer), \
                    tempfile.TemporaryDirectory() as directory, redirect_stdout(StringIO()) as output:
                root = Path(directory)
                (root / "pretrain.jsonl").write_text('{"text":"pretrainonlyword"}\n', encoding="utf-8")
                (root / "instruction.jsonl").write_text(
                    '{"prompt":"localquestion","answer":"localanswer"}\n', encoding="utf-8")
                tokenizer_path = root / "tokenizer.json"
                before = None
                if reuse_tokenizer:
                    BPETokenizer().train("abc", vocab_size=260).save(tokenizer_path)
                    before = tokenizer_path.read_bytes()
                checkpoint = root / "output"
                cache = checkpoint / "token_cache"
                cache.mkdir(parents=True)
                # 旧缓存不应成为本地模式的训练数据，启用阶段会重新构建。
                (cache / "instruction.jsonl").write_text("stale remote cache\n", encoding="utf-8")
                models = []

                def create_model(size):
                    model = SmallModel(size)
                    models.append(model)
                    return model

                with patch.object(train, "TOKENIZER_PATH", tokenizer_path), \
                     patch.object(train, "Vital", side_effect=create_model), \
                     patch.object(train, "TEST_TEXTS", []), \
                     patch.object(train, "PRETRAIN_DATASET", "remote-body"), \
                     patch.object(train, "INSTRUCTION_DATASET", "remote-instruction"), \
                     patch("parts.training_data.load_dataset", side_effect=AssertionError("访问了远程")) as remote:
                    train.main(["--usejsononly", "--data-dir", str(root),
                                "--tokenizer", str(tokenizer_path), "--output-dir", str(checkpoint),
                                "--pretrain-epochs", str(pretrain_epochs), "--instruction-epochs", "1",
                                "--device", "cpu", "--mixed-precision", "off", "--skip-validation",
                                "--max-seq-len", "64", "--batch-size", "2"])
                    remote.assert_not_called()
                    self.assertEqual(train.PRETRAIN_DATASET, "remote-body")
                    self.assertEqual(train.INSTRUCTION_DATASET, "remote-instruction")
                self.assertEqual(models[0].calls, pretrain_epochs + 1)
                self.assertTrue((checkpoint / "model.pt").exists())
                self.assertIn("仅本地 JSONL", output.getvalue())
                tokenizer = BPETokenizer.load(checkpoint / "tokenizer.json")
                if reuse_tokenizer:
                    self.assertEqual(tokenizer_path.read_bytes(), before)
                else:
                    self.assertIsInstance(tokenizer, RuleTokenizer)
                    self.assertEqual(len(tokenizer.encode("localquestion")), 1)
                    self.assertEqual(len(tokenizer.encode("localanswer")), 1)
                    self.assertEqual(len(tokenizer.encode("pretrainonlyword")) == 1, pretrain_epochs > 0)
                samples = [json.loads(line) for line in (cache / "instruction.jsonl").read_text().splitlines()]
                self.assertEqual(len(samples), 1)
                self.assertEqual(tokenizer.decode(samples[0][0]), "用户：localquestion\nlocalanswer")


if __name__ == "__main__":
    unittest.main()
