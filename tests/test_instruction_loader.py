"""远程指令流进入 DataLoader 的端到端验证，不依赖网络。"""

import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from datasets import IterableDataset

from parts.tokenizer import BPETokenizer
from parts.training_data import IGNORE_INDEX, make_loader, read_instruction_records


class InstructionLoaderTests(unittest.TestCase):
    def test_only_instruction_and_output_are_used(self):
        rows = [
            {"instruction": "问题一", "output": "回答一", "input": "不得拼接这段内容"},
            {"instruction": "问题二", "output": "回答二", "input": None},
            {"instruction": "问题三", "output": "回答三"},
        ]
        self.assertEqual(list(read_instruction_records(rows)), [
            {"prompt": "问题一", "answer": "回答一"},
            {"prompt": "问题二", "answer": "回答二"},
            {"prompt": "问题三", "answer": "回答三"},
        ])

    def test_validation_prompts_are_excluded_and_bad_fields_are_reported(self):
        rows = [{"instruction": "验证问题", "output": "不可训练"},
                {"instruction": "训练问题", "output": "训练回答"}]
        self.assertEqual(list(read_instruction_records(rows, {"验证问题"})),
                         [{"prompt": "训练问题", "answer": "训练回答"}])
        for row in ({"instruction": "缺回答"}, {"instruction": "问", "output": None},
                    {"instruction": "", "output": "答"}):
            with self.assertRaisesRegex(ValueError, "远程指令第 1 条"):
                list(read_instruction_records([row]))

    def test_remote_and_local_records_form_repeatable_minibatches(self):
        tokenizer = BPETokenizer().train(["abc"], vocab_size=260)
        remote_rows = [
            {"instruction": "远程一", "output": "答一", "input": "此字段必须忽略"},
            {"instruction": "验证问题", "output": "不参与训练", "input": ""},
            {"instruction": "远程二", "output": "答二", "input": ""},
        ]
        stream = IterableDataset.from_generator(lambda: iter(remote_rows))
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "instruction.jsonl"
            path.write_text(json.dumps({"prompt": "本地", "answer": "本地答"}, ensure_ascii=False) + "\n",
                            encoding="utf-8")
            with patch("parts.training_data.load_dataset", return_value=stream) as load:
                loader = make_loader(path, "instruction", tokenizer, batch_size=2,
                                     shuffle_buffer=1, instruction_dataset="BelleGroup/train_1M_CN",
                                     excluded_prompts={"验证问题"}, length_bucket_batches=0)
                self.assertEqual(next(iter(loader.dataset.instruction_records)).keys(),
                                 {"instruction", "output"})
                for _ in range(2):
                    batches = list(loader)
                    self.assertEqual([batch["input_ids"].size(0) for batch in batches], [2, 1])
                    answers, contexts = [], []
                    for batch in batches:
                        for inputs, labels in zip(batch["input_ids"], batch["labels"]):
                            answers.append(tokenizer.decode(labels[labels != IGNORE_INDEX].tolist()))
                            contexts.append(tokenizer.decode(inputs.tolist()))
                    self.assertEqual(answers, ["本地答", "答一", "答二"])
                    self.assertTrue(any("远程一" in text for text in contexts))
                    self.assertFalse(any("此字段必须忽略" in text for text in contexts))
                load.assert_called_once_with("BelleGroup/train_1M_CN", split="train", streaming=True)

    def test_pretrain_never_loads_remote_instructions(self):
        tokenizer = BPETokenizer()
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "pretrain.jsonl"
            path.write_text('{"text":"正文"}\n', encoding="utf-8")
            with patch("parts.training_data.load_dataset") as load:
                loader = make_loader(path, "pretrain", tokenizer, batch_size=2)
                self.assertEqual(len(list(loader)), 1)
                load.assert_not_called()
                with self.assertRaisesRegex(ValueError, "instruction 阶段"):
                    make_loader(path, "pretrain", tokenizer, batch_size=2,
                                instruction_dataset="BelleGroup/train_1M_CN")


if __name__ == "__main__":
    unittest.main()
