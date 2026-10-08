"""离线验证两份远程资料的文档切分、重复迭代和目标覆盖。"""
from collections import Counter
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from datasets import IterableDataset

from parts.tokenizer import BPETokenizer
from parts.training_data import IGNORE_INDEX, make_loader


class RemoteRangeTests(unittest.TestCase):
    def test_both_stages_split_before_filtering_and_windowing(self):
        tokenizer = BPETokenizer()
        for stage in ("pretrain", "instruction"):
            with self.subTest(stage=stage), tempfile.TemporaryDirectory() as directory:
                path = Path(directory) / "empty.jsonl"
                path.write_text("", encoding="utf-8")
                rows = ([{"text": str(i) + " long document" * 8} for i in range(5)]
                        if stage == "pretrain" else
                        [{"instruction": str(i), "output": str(i) + " long answer" * 8}
                         for i in range(5)])
                stream = IterableDataset.from_generator(lambda: iter(rows))
                remote_args = ({"pretrain_dataset": "test"} if stage == "pretrain"
                               else {"instruction_dataset": "test", "excluded_prompts": {"1"}})
                with patch("parts.training_data.load_dataset", return_value=stream):
                    def loader(start=0, limit=None):
                        return make_loader(path, stage, tokenizer, batch_size=3,
                                           max_seq_len=7, shuffle_buffer=5,
                                           length_bucket_batches=2,
                                           remote_start_document=start,
                                           remote_document_limit=limit, **remote_args)

                    def targets(batches):
                        return Counter(int(label) for batch in batches
                                       for label in batch["labels"].flatten()
                                       if label != IGNORE_INDEX)

                    first, rest, full = loader(limit=2), loader(start=2), loader()
                    attr = "pretrain_records" if stage == "pretrain" else "instruction_records"
                    self.assertEqual(list(getattr(first.dataset, attr)), rows[:2])
                    self.assertEqual(list(getattr(rest.dataset, attr)), rows[2:])
                    self.assertEqual(targets(first) + targets(rest), targets(full))
                    self.assertEqual(targets(first), targets(first))
                    self.assertEqual(list(loader(start=5)), [])
                    expected = targets(first)
                    cache = Path(directory) / "tokens.jsonl"
                    count = first.dataset.prepare_token_cache(cache)
                    self.assertGreater(count, 0)
                    # 缓存后的多轮迭代不允许再次分词或读取原始资料。
                    with patch.object(tokenizer, "encode_batch", side_effect=AssertionError("重复分词")), \
                         patch.object(first.dataset, "uncached_samples", side_effect=AssertionError("重复读取")):
                        self.assertEqual(targets(first), expected)
                        first.dataset.seed += 1
                        self.assertEqual(targets(first), expected)

    def test_bad_ranges_fail_before_loading(self):
        with patch("parts.training_data.load_dataset") as load:
            for kwargs in ({"remote_start_document": -1}, {"remote_document_limit": 0}):
                with self.assertRaises(ValueError):
                    make_loader(Path("unused"), "pretrain", BPETokenizer(), batch_size=1,
                                pretrain_dataset="test", **kwargs)
            load.assert_not_called()


if __name__ == "__main__":
    unittest.main()
