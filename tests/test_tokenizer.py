"""分词器的多语言无损性、编号兼容性与保存加载回归测试。"""

import json
from pathlib import Path
import random
import tempfile
import unittest

from parts.tokenizer import BPETokenizer, bpe_tokenise, load_bpe, train_bpe


SAMPLES = [
    "你好，世界！繁體中文與简体中文。",
    "Hello, world! I’m learning Python and mathematics.",
    "Bonjour ! élève, déjà vu. Straße und Grüße.",
    "Привет, мир! Ελληνικά γράμματα.",
    "مرحبا بالعالم! עברית עם ניקוד: שָׁלוֹם",
    "नमस्ते दुनिया। বাংলা ভাষা। தமிழ் மொழி.",
    "日本語の文章です。한국어 문장입니다.",
    "สวัสดีชาวโลก ภาษาไทยไม่มีช่องว่างทุกคำ",
    "👩🏽‍💻 🦊 ☕ 🏳️‍🌈 e\u0301 é \u200d\u200b\ufeff",
    "  def f(x):\r\n\treturn x ** 2  # 保留缩进\n\n",
    "x² + y² = 25；∫₀¹ x dx = ½；1234567890",
    "解释 <pad>、<assistant>、<bos> 和 <eos> 的含义。",
]


class TokenizerTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tokenizer = BPETokenizer().train(iter(SAMPLES * 3), vocab_size=1024)

    def test_multilingual_exact_round_trip(self):
        for text in SAMPLES:
            with self.subTest(text=text):
                ids = self.tokenizer.encode(text, allow_special_tokens=False)
                self.assertEqual(self.tokenizer.decode(ids, errors="strict"), text)

    def test_unseen_unicode_and_untrained_byte_fallback(self):
        rng = random.Random(42)
        scalars = [rng.randrange(0x110000) for _ in range(3000)]
        text = "".join(chr(n) for n in scalars if not 0xD800 <= n <= 0xDFFF)
        text += "\x00\r\n\t" + "".join(chr(n) for n in range(256))
        for tokenizer in (BPETokenizer(), self.tokenizer):
            ids = tokenizer.encode(text, allow_special_tokens=False)
            self.assertEqual(tokenizer.decode(ids, errors="strict"), text)

    def test_fixed_ids_and_literal_special_markers(self):
        tokenizer = self.tokenizer
        for byte in range(256):
            self.assertEqual(tokenizer.vocab[byte], bytes([byte]))
        for marker, token_id in tokenizer.SPECIAL_TOKENS.items():
            self.assertEqual(tokenizer.encode(marker), [token_id])
            literal = tokenizer.encode(marker, allow_special_tokens=False)
            self.assertTrue(set(literal).isdisjoint(tokenizer.SPECIAL_TOKENS.values()))
            self.assertEqual(tokenizer.decode(literal, errors="strict"), marker)
        self.assertEqual(tokenizer.encode("", add_bos=True, add_eos=True), [257, 258])

    def test_batch_matches_individual_encoding(self):
        for allow_special in (True, False):
            for bos, eos in ((False, False), (True, True)):
                options = dict(add_bos=bos, add_eos=eos, allow_special_tokens=allow_special)
                texts = SAMPLES + ["", "<assistant>你好", " " * 30]
                self.assertEqual(self.tokenizer.encode_batch(iter(texts), **options),
                                 [self.tokenizer.encode(text, **options) for text in texts])
        self.assertEqual(self.tokenizer.encode_batch([]), [])
        with self.assertRaises(TypeError):
            self.tokenizer.encode_batch("abc")

    def test_version_two_save_load_and_helpers(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "tokenizer.json"
            self.tokenizer.save(path)
            self.assertEqual(json.loads(path.read_text(encoding="utf-8"))["version"], 2)
            loaded = load_bpe(path)
            self.assertEqual(loaded.vocab, self.tokenizer.vocab)
            self.assertEqual(loaded.merges, self.tokenizer.merges)
            self.assertEqual(loaded.encode_batch(SAMPLES), self.tokenizer.encode_batch(SAMPLES))
            for text in SAMPLES:
                self.assertEqual(bpe_tokenise(text), self.tokenizer.encode(text))
                self.assertEqual(loaded.decode(loaded.encode(text, allow_special_tokens=False)), text)

    def test_legacy_format_keeps_original_ids(self):
        vocab = BPETokenizer().vocab
        vocab.update({260: b"ab", 261: b"abc"})
        legacy = {
            "format": "byte-level-bpe", "version": 1,
            "special_tokens": BPETokenizer.SPECIAL_TOKENS,
            "vocab": {str(index): value.hex() for index, value in vocab.items()},
            "merges": [[97, 98, 260], [260, 99, 261]],
        }
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "legacy.json"
            path.write_text(json.dumps(legacy), encoding="utf-8")
            loaded = BPETokenizer.load(path)
            self.assertEqual(loaded.encode("abc<assistant>abc"), [261, 259, 261])
            self.assertEqual(loaded.encode_batch(["abc", ""]), [[261], []])
            for text in SAMPLES:
                self.assertEqual(loaded.decode(loaded.encode(text, allow_special_tokens=False)), text)
            loaded.save(path)
            self.assertEqual(BPETokenizer.load(path).vocab, vocab)

    def test_training_is_deterministic_and_accepts_one_pass_input(self):
        def once():
            yield from SAMPLES * 3
        other = train_bpe(once(), vocab_size=1024)
        self.assertEqual(other.vocab, self.tokenizer.vocab)
        self.assertEqual(other.encode_batch(SAMPLES), self.tokenizer.encode_batch(SAMPLES))

    def test_minimum_vocabulary_retains_every_byte(self):
        tokenizer = BPETokenizer().train(["a"], vocab_size=260)
        self.assertEqual(tokenizer.vocab_size, 260)
        for text in SAMPLES:
            self.assertEqual(tokenizer.decode(tokenizer.encode(text, allow_special_tokens=False)), text)

    def test_failed_training_does_not_replace_previous_vocabulary(self):
        tokenizer = BPETokenizer().train(["abc abc"])
        before = tokenizer.vocab
        for samples, options in (([], {}), ([""], {}), (["abc", 5], {}),
                                 (["abc"], {"vocab_size": 259}),
                                 (["abc"], {"min_frequency": 0}),
                                 (["abc"], {"max_token_length": 1})):
            with self.assertRaises((TypeError, ValueError)):
                tokenizer.train(samples, **options)
            self.assertEqual(tokenizer.vocab, before)

    def test_pre_tokenization_preserves_boundaries(self):
        tokenizer = BPETokenizer().train(["你好 hello\n123456\t世界!"] * 10,
                                         vocab_size=1024)
        self.assertEqual(tokenizer.encode("hello\n世界"),
                         tokenizer.encode("hello") + tokenizer.encode("\n") + tokenizer.encode("世界"))
        self.assertEqual(tokenizer.encode("123456"), tokenizer.encode("123") + tokenizer.encode("456"))
        self.assertEqual(tokenizer.decode(tokenizer.encode("\t\r\n  ")), "\t\r\n  ")

    def test_decode_rejects_invalid_ids_and_incomplete_utf8(self):
        for bad in ([True], [-1], [self.tokenizer.vocab_size], [1.5]):
            with self.assertRaises(ValueError):
                self.tokenizer.decode(bad)
        with self.assertRaises(UnicodeDecodeError):
            self.tokenizer.decode([0xE4], errors="strict")

    def test_corrupted_file_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "broken.json"
            self.tokenizer.save(path)
            data = json.loads(path.read_text(encoding="utf-8"))
            data["special_tokens"]["<pad>"] = 99
            path.write_text(json.dumps(data), encoding="utf-8")
            with self.assertRaises(ValueError):
                BPETokenizer.load(path)


if __name__ == "__main__":
    unittest.main()
