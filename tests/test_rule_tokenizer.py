"""规则边界、未知词兜底、编号固定和训练入口的回归验证。"""

import json
from contextlib import redirect_stdout, redirect_stderr
from io import StringIO
from pathlib import Path
import random
import sys
import tempfile
import unittest
from unittest.mock import patch

from parts.tokenizer import BPETokenizer, RuleTokenizer, load_bpe
import train_tokenizer as cli


class RuleTokenizerTests(unittest.TestCase):
    def test_units_are_characters_words_and_lossless_separators(self):
        text = "我爱吃水果 hello world, café\r\n\t  abc123。𠀀 e\u0301"
        tokenizer = RuleTokenizer().train([text])
        units = tokenizer.split_units(text)
        self.assertEqual(units[:14],
                         ["我", "爱", "吃", "水", "果", " ", "hello", " ",
                          "world", ",", " ", "café", "\r", "\n"])
        self.assertIn("𠀀", units)
        self.assertIn("e\u0301", units)
        self.assertEqual([tokenizer.decode([i], errors="strict") for i in tokenizer.encode(text)], units)
        self.assertEqual(tokenizer.decode(tokenizer.encode(text), errors="strict"), text)

    def test_no_frequency_vocabulary_or_token_length_cap(self):
        words = [f"word{number}" for number in range(17000)]
        words += ["superlong" * 100]
        tokenizer = RuleTokenizer().train(iter(words))
        self.assertGreater(tokenizer.vocab_size, 17000)
        for word in (words[0], words[-2], words[-1]):
            self.assertEqual(len(tokenizer.encode(word)), 1)
            self.assertEqual(tokenizer.decode(tokenizer.encode(word)), word)

    def test_unknown_text_never_changes_vocabulary(self):
        tokenizer = RuleTokenizer().train(["hello café 中文"])
        before = tokenizer.vocab
        # 已见字符组成的新单词退回字符；完全未见的字符退回字节。
        self.assertEqual(tokenizer.encode("hellocafé"),
                         [i for character in "hellocafé" for i in tokenizer.encode(character)])
        rng = random.Random(5)
        text = "新词\x00\r\n🦊 " + "".join(chr(rng.randrange(0x10000, 0x110000)) for _ in range(500))
        self.assertEqual(tokenizer.decode(tokenizer.encode(text), errors="strict"), text)
        self.assertEqual(tokenizer.vocab, before)
        with self.assertRaises(TypeError):
            tokenizer.train(["替换文字", None])
        self.assertEqual(tokenizer.vocab, before)

    def test_special_markers_literal_text_and_batch(self):
        tokenizer = RuleTokenizer().train(["解释 <assistant> 标记，hello"])
        for text, index in tokenizer.SPECIAL_TOKENS.items():
            self.assertEqual(tokenizer.encode(text), [index])
            ids = tokenizer.encode(text, allow_special_tokens=False)
            self.assertTrue(set(ids).isdisjoint(tokenizer.SPECIAL_TOKENS.values()))
            self.assertEqual(tokenizer.decode(ids), text)
        texts = ["", "中文 hello", "<assistant>解释", "  \n"]
        options = dict(add_bos=True, add_eos=True, allow_special_tokens=False)
        self.assertEqual(tokenizer.encode_batch(texts, **options),
                         [tokenizer.encode(text, **options) for text in texts])

    def test_save_load_preserves_every_id_and_rejects_invalid_units(self):
        tokenizer = RuleTokenizer().train(["中文 hello café\n"])
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "rules.json"
            tokenizer.save(path)
            loaded = load_bpe(path)
            self.assertIsInstance(loaded, RuleTokenizer)
            self.assertEqual(loaded.vocab, tokenizer.vocab)
            self.assertEqual(loaded.encode("中文 hello 新"), tokenizer.encode("中文 hello 新"))
            good = json.loads(path.read_text(encoding="utf-8"))
            for extra in ("中文", "two words", "中", "a", ""):
                bad = dict(good, units=good["units"] + [extra])
                path.write_text(json.dumps(bad), encoding="utf-8")
                with self.assertRaises(ValueError):
                    BPETokenizer.load(path)

    def test_cli_defaults_to_rules_and_rejects_ignored_limits(self):
        with tempfile.TemporaryDirectory() as directory:
            source, output = Path(directory) / "input.txt", Path(directory) / "rules.json"
            source.write_text("hello 中文", encoding="utf-8")
            args = ["train_tokenizer.py", str(source), "--output", str(output)]
            with patch.object(sys, "argv", args), redirect_stdout(StringIO()):
                cli.main()
            self.assertIsInstance(load_bpe(output), RuleTokenizer)
            with patch.object(sys, "argv", args + ["--vocab-size", "512"]), redirect_stderr(StringIO()):
                with self.assertRaises(SystemExit):
                    cli.main()

    def test_training_reuses_old_bpe_and_requires_explicit_mode_change(self):
        import train
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "tokenizer.json"
            BPETokenizer().train("旧词表", vocab_size=270).save(path)
            before = path.read_bytes()
            with patch.object(train, "TOKENIZER_PATH", path), redirect_stdout(StringIO()):
                self.assertNotIsInstance(train.train_tokenizer(), RuleTokenizer)
                with self.assertRaisesRegex(ValueError, "模式不匹配"):
                    train.train_tokenizer(mode="rules")
                self.assertEqual(path.read_bytes(), before)
                new = train.train_tokenizer(mode="rules", retrain=True, texts=["新词表 wholeword"])
                self.assertIsInstance(new, RuleTokenizer)
                self.assertEqual(len(new.encode("wholeword")), 1)


if __name__ == "__main__":
    unittest.main()
