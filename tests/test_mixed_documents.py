"""续训三路整篇混合：原文顺序、完整上下文、范围和目标掩码。"""

from collections import Counter
import importlib
import unittest
from unittest.mock import patch

from datasets import Dataset, Features, IterableDataset, Value

from parts.tokenizer import BPETokenizer
from parts.training_data import IGNORE_INDEX, encode_prompt, make_mixed_document_dataset


def stream(rows, columns):
    return IterableDataset.from_generator(
        lambda: iter(rows), features=Features({name: Value("string") for name in columns}),
    )


class MixedDocumentTests(unittest.TestCase):
    def test_document_limit_applies_after_skip_before_chunking(self):
        tokenizer = BPETokenizer()
        bodies = [{"text": str(i) + "X" * 5000} for i in range(7)]
        sources = {"body": stream(bodies, ["text"]),
                   "instruction": stream([{"instruction": "问", "output": "答"}],
                                         ["instruction", "output"])}
        with patch("parts.training_data.load_dataset", side_effect=lambda path, **kw: sources[path]):
            args = (stream([{"text": "论文"}], ["text"]), tokenizer,
                    {"dataset": "body", "start_document": 2, "document_limit": 3},
                    {"dataset": "instruction", "start_document": 0, "document_limit": None})
            mixed = make_mixed_document_dataset(*args, max_seq_len=2048, shuffle_buffer=2)
            for epoch in (0, 1):
                mixed.set_epoch(epoch)
                rows = list(mixed)
                body_chunks = [row for row in rows if row["source"] == "fineweb"]
                self.assertEqual(len(body_chunks), 9)  # 三篇文档，每篇三块，不是只取三块。
                expected = Counter(token for row in bodies[2:5]
                                   for token in tokenizer.encode(row["text"]) + [tokenizer.eos_token_id])
                self.assertEqual(Counter(token for row in body_chunks for token in row["labels"]), expected)
                self.assertEqual(Counter(row["source"] for row in rows),
                                 {"arxiv": 1, "fineweb": 9, "belle": 1})
            with self.assertRaises(ValueError):
                make_mixed_document_dataset(args[0], tokenizer,
                    {"dataset": "body", "start_document": 0, "document_limit": 0}, args[3])

    def test_limit_splits_all_sources_without_losing_targets(self):
        tokenizer = BPETokenizer()
        sources = {"body": stream([{"text": "B" * 4200}], ["text"]),
                   "instruction": stream([{"instruction": "Q" * 2400, "output": "A" * 4500}],
                                         ["instruction", "output"])}
        with patch("parts.training_data.load_dataset", side_effect=lambda path, **kw: sources[path]):
            args = (stream([{"text": "C" * 6000}], ["text"]), tokenizer,
                    {"dataset": "body", "start_document": 0},
                    {"dataset": "instruction", "start_document": 0})
            full = list(make_mixed_document_dataset(*args, shuffle_buffer=2))
            mixed = make_mixed_document_dataset(*args, max_seq_len=2048, shuffle_buffer=2)
            chunks = list(mixed)
            self.assertGreater(len(chunks), len(full))
            expected = []
            for row in full:
                for start in range(0, len(row["input_ids"]), 2048):
                    labels = row["labels"][start:start + 2048]
                    if any(label != IGNORE_INDEX for label in labels):
                        expected.append(dict(row, input_ids=row["input_ids"][start:start + 2048], labels=labels))
            self.assertEqual(chunks, expected)
            for row in chunks:
                self.assertLessEqual(len(row["input_ids"]), 2048)
                self.assertEqual(len(row["input_ids"]), len(row["labels"]))
            for source in ("arxiv", "fineweb", "belle"):
                targets = lambda rows: [v for row in rows if row["source"] == source
                                        for v in row["labels"] if v != IGNORE_INDEX]
                self.assertEqual(targets(chunks), targets(full))
                self.assertEqual(targets(chunks)[-1], tokenizer.eos_token_id)
            mixed.set_epoch(1)
            self.assertEqual(Counter((row["source"], tuple(row["labels"])) for row in chunks),
                             Counter((row["source"], tuple(row["labels"])) for row in mixed))
            with self.assertRaises(ValueError):
                make_mixed_document_dataset(*args, max_seq_len=0)

    def test_all_documents_remain_whole_and_each_record_occurs_once(self):
        tokenizer = BPETokenizer()
        arxiv_rows = [{"text": "论文开头" + "A" * 1200 + "论文结尾"}]
        body_rows = [{"text": f"正文{i} " + "B" * 600} for i in range(10)]
        instruction_rows = [{"instruction": "旧请求", "output": "旧回答"},
                            {"instruction": "验证问题", "output": "不参与"}]
        instruction_rows += [{"instruction": f"问题{i}" + "Q" * 600,
                              "output": f"回答{i}" + "A" * 600} for i in range(4)]
        sources = {"body": stream(body_rows, ["text"]),
                   "instruction": stream(instruction_rows, ["instruction", "output"])}
        with patch("parts.training_data.load_dataset", side_effect=lambda path, **kw: sources[path]) as load:
            mixed = make_mixed_document_dataset(
                Dataset.from_list(arxiv_rows), tokenizer,
                {"dataset": "body", "config": "default", "start_document": 2},
                {"dataset": "instruction", "config": None, "start_document": 1},
                excluded_prompts={"验证问题"}, seed=42, shuffle_buffer=16,
            )
            self.assertEqual(load.call_count, 2)
            for call in load.call_args_list:
                self.assertTrue(call.kwargs["streaming"])
            first = list(mixed)
            self.assertEqual(first, list(mixed))
            self.assertEqual(Counter(row["source"] for row in first),
                             {"arxiv": 1, "fineweb": 8, "belle": 4})
            expected_bodies = Counter(row["text"] for row in arxiv_rows + body_rows[2:])
            actual_bodies = Counter()
            expected_answers = {row["instruction"]: row["output"] for row in instruction_rows[2:]}
            seen_prompts = set()
            for row in first:
                inputs, labels = row["input_ids"], row["labels"]
                self.assertEqual(len(inputs), len(labels))
                self.assertGreater(len(inputs), 512)
                self.assertEqual(labels[-1], tokenizer.eos_token_id)
                if row["stage"] == "pretrain":
                    body = tokenizer.decode(inputs)
                    actual_bodies[body] += 1
                    self.assertEqual(inputs, [tokenizer.bos_token_id] + tokenizer.encode(body))
                    self.assertEqual(labels, tokenizer.encode(body) + [tokenizer.eos_token_id])
                else:
                    for prompt, answer in expected_answers.items():
                        prefix = encode_prompt(tokenizer, prompt)
                        if inputs[:len(prefix)] == prefix:
                            seen_prompts.add(prompt)
                            self.assertEqual(inputs, prefix + tokenizer.encode(answer))
                            self.assertEqual(labels, [IGNORE_INDEX] * (len(prefix) - 1)
                                             + tokenizer.encode(answer) + [tokenizer.eos_token_id])
                            break
                    else:
                        self.fail("混入了未知或被截断的指令")
            self.assertEqual(actual_bodies, expected_bodies)
            self.assertEqual(seen_prompts, set(expected_answers))
            mixed.set_epoch(1)
            second = list(mixed)
            self.assertNotEqual(first, second)
            self.assertEqual(Counter(tuple(row["input_ids"]) for row in first),
                             Counter(tuple(row["input_ids"]) for row in second))

    def test_empty_remaining_source_does_not_discard_other_sources(self):
        sources = {"body": stream([], ["text"]),
                   "instruction": stream([{"instruction": "问", "output": "答"}],
                                         ["instruction", "output"])}
        with patch("parts.training_data.load_dataset", side_effect=lambda path, **kw: sources[path]):
            mixed = make_mixed_document_dataset(
                stream([{"text": "整篇正文"}], ["text"]), BPETokenizer(),
                {"dataset": "body", "start_document": 0},
                {"dataset": "instruction", "start_document": 0}, shuffle_buffer=2,
            )
            self.assertEqual(Counter(row["source"] for row in mixed), {"arxiv": 1, "belle": 1})

    def test_importing_continue_script_does_not_load_remote_data(self):
        with patch("datasets.load_dataset", side_effect=AssertionError("导入触发远程读取")) as load:
            import continue_study
            importlib.reload(continue_study)
            load.assert_not_called()


if __name__ == "__main__":
    unittest.main()
