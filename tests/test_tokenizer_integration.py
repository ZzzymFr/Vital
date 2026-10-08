"""外部语料入口和模型训练脚本的分词器复用验证。"""

from contextlib import redirect_stdout, redirect_stderr
from io import StringIO
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

from parts.tokenizer import BPETokenizer
import train_tokenizer as cli


class TokenizerIntegrationTests(unittest.TestCase):
    def test_jsonl_formats_and_invalid_records(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "mixed.jsonl"
            path.write_text(
                '\ufeff{"text":"你好"}\n{"prompt":"Hello","answer":"世界"}\n'
                '{"instruction":"Bonjour","input":"","output":"مرحبا"}\n',
                encoding="utf-8",
            )
            self.assertEqual(list(cli.read_texts([path])),
                             ["你好", "Hello", "世界", "Bonjour", "", "مرحبا"])
            text_path = Path(directory) / "code.txt"
            text_path.write_bytes(b"  def f():\r\n\treturn 1\n")
            self.assertEqual(list(cli.read_texts([text_path])),
                             ["  def f():\r\n", "\treturn 1\n"])
            for invalid in ('[]', '{"text":12}', '{', '{"prompt":"缺失回答"}'):
                path.write_text(invalid, encoding="utf-8")
                with self.assertRaisesRegex(ValueError, r"mixed.jsonl:1"):
                    list(cli.read_texts([path]))

    def test_cli_saves_and_refuses_accidental_overwrite(self):
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "train.txt"
            output = Path(directory) / "tokenizer.json"
            source.write_text("你好 Hello مرحبا\n" * 3, encoding="utf-8")
            arguments = ["train_tokenizer.py", str(source), "--output", str(output),
                         "--mode", "bpe", "--vocab-size", "512"]
            with patch.object(sys, "argv", arguments), redirect_stdout(StringIO()):
                cli.main()
            tokenizer = BPETokenizer.load(output)
            self.assertEqual(tokenizer.decode(tokenizer.encode("你好 Hello مرحبا")),
                             "你好 Hello مرحبا")
            before = output.read_bytes()
            with patch.object(sys, "argv", arguments), redirect_stderr(StringIO()):
                with self.assertRaises(SystemExit) as error:
                    cli.main()
            self.assertEqual(error.exception.code, 2)
            self.assertEqual(output.read_bytes(), before)
            overwrite_source = ["train_tokenizer.py", str(source), "--output", str(source),
                                "--overwrite"]
            with patch.object(sys, "argv", overwrite_source), redirect_stderr(StringIO()):
                with self.assertRaises(SystemExit):
                    cli.main()

    def test_model_training_reuses_cached_vocabulary(self):
        import train
        import torch

        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "tokenizer.json"
            with patch.object(train, "TOKENIZER_PATH", path), redirect_stdout(StringIO()):
                tokenizer = train.train_tokenizer()
                with patch.object(BPETokenizer, "train", side_effect=AssertionError("不应重新训练")):
                    loaded = train.train_tokenizer()
                self.assertEqual(loaded.vocab, tokenizer.vocab)
                rows = train.read_records(train.DATA_DIR / "instruction.jsonl", "instruction")
                prompts = [train.format_dialogue(row["prompt"], "") for row in rows]
                tensor = train.texts_to_tensor(prompts, loaded, torch.device("cpu"),
                                               add_bos=True, add_eos=False)
                self.assertEqual(tensor.dtype, torch.long)
                for row, text in zip(tensor, prompts):
                    actual = row[row != loaded.pad_token_id].tolist()
                    self.assertEqual(actual, loaded.encode(text, add_bos=True))


if __name__ == "__main__":
    unittest.main()
