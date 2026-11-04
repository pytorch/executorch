# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import tempfile
import unittest
from inspect import signature
from unittest.mock import Mock, patch

import torch
from datasets import Dataset, DatasetDict
from executorch.backends.samsung.test.models import (
    test_mobilebert_finetuning,
    test_mobilebert_qat,
)
from executorch.examples.samsung.scripts import (
    mobilebert_finetune,
    mobilebert_finetune_QAT,
)


class TinyClassifier(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.ones(()))
        self.steps = 0
        self.save_pretrained = Mock()

    def forward(self, texts, attention_mask, labels):
        self.steps += 1
        return ((self.weight * texts.float().mean()).square(),)


class TestMobileBertFinetune(unittest.TestCase):
    def test_qat_pretraining_accepts_custom_learning_rate(self):
        finetune = mobilebert_finetune_QAT.MobileBertFinetune.__new__(
            mobilebert_finetune_QAT.MobileBertFinetune
        )
        datasets = DatasetDict(
            train=Dataset.from_dict({"label": [0, 1]}),
            validation=Dataset.from_dict({"label": [0, 1]}),
        )
        with patch.object(
            mobilebert_finetune_QAT, "TrainingArguments"
        ) as training_args, patch.object(mobilebert_finetune_QAT, "Trainer"):
            finetune.training(
                TinyClassifier(), datasets, None, None, learning_rate=5e-4
            )
        self.assertEqual(training_args.call_args.kwargs["learning_rate"], 5e-4)
        self.assertEqual(
            signature(
                mobilebert_finetune_QAT.MobileBertFinetune.get_finetune_mobilebert
            )
            .parameters["learning_rate"]
            .default,
            2e-5,
        )

    def test_qat_initialization_uses_trainer_seed_and_restores_rng(self):
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(43)
            state = torch.get_rng_state()
            test = test_mobilebert_qat.TestMilestoneMobileBertQAT()
            test.setUp()
            try:
                self.assertEqual(torch.initial_seed(), 42)
            finally:
                test.tearDown()
            self.assertTrue(torch.equal(state, torch.get_rng_state()))

    def test_custom_learning_rate_controls_training_updates(self):
        self.assertIn(
            "learning_rate",
            signature(
                mobilebert_finetune.MobileBertFinetune.get_finetune_mobilebert
            ).parameters,
        )
        finetune = mobilebert_finetune.MobileBertFinetune.__new__(
            mobilebert_finetune.MobileBertFinetune
        )
        finetune.tokenizer = lambda texts, **kwargs: {
            "input_ids": torch.ones(len(texts), 2, dtype=torch.long),
            "attention_mask": torch.ones(len(texts), 2, dtype=torch.long),
        }
        with tempfile.TemporaryDirectory() as directory, patch(
            "requests.get", return_value=Mock(content=b"text\t0\nother\t1\n" * 10)
        ), patch.object(
            mobilebert_finetune.MobileBertForSequenceClassification,
            "from_pretrained",
            return_value=TinyClassifier(),
        ):
            model, _ = finetune.get_finetune_mobilebert(
                directory, batch_size=4, max_train_samples=8, learning_rate=0.002
            )
        self.assertEqual(model.steps, 10)
        self.assertLess(model.weight.item(), 0.99)

    def test_ci_finetuning_does_not_change_global_rng(self):
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(43)
            state = torch.get_rng_state()
            with patch.object(
                test_mobilebert_finetuning, "MobileBertFinetune"
            ) as finetune, patch.object(
                test_mobilebert_finetuning, "SamsungTester"
            ), patch.object(
                test_mobilebert_finetuning, "gen_samsung_backend_compile_spec"
            ):
                finetune.return_value.get_finetune_mobilebert.side_effect = (
                    lambda *args, **kwargs: (torch.rand(1), None)
                )
                test_mobilebert_finetuning.Test_Milestone_MobileBertFinetune().test_mobilebert_finetuning_fp16()
            self.assertTrue(torch.equal(state, torch.get_rng_state()))

    def test_limits_training_samples_without_changing_epochs_or_validation(self):
        finetune = mobilebert_finetune.MobileBertFinetune.__new__(
            mobilebert_finetune.MobileBertFinetune
        )
        finetune.tokenizer = lambda texts, **kwargs: {
            "input_ids": torch.ones(len(texts), 2, dtype=torch.long),
            "attention_mask": torch.ones(len(texts), 2, dtype=torch.long),
        }
        for limit, expected_steps in ((None, 25), (8, 10), (100, 25)):
            with self.subTest(limit=limit), tempfile.TemporaryDirectory() as directory:
                model = TinyClassifier()
                with patch(
                    "requests.get",
                    return_value=Mock(content=b"text\t0\nother\t1\n" * 10),
                ), patch.object(
                    mobilebert_finetune.MobileBertForSequenceClassification,
                    "from_pretrained",
                    return_value=model,
                ):
                    _, validation = finetune.get_finetune_mobilebert(
                        directory, batch_size=4, max_train_samples=limit
                    )
                self.assertEqual(model.steps, expected_steps)
                self.assertEqual(len(validation.dataset), 20)
                model.save_pretrained.assert_called_once_with(directory)

    def test_qat_limits_only_pretraining_not_calibration_or_validation(self):
        finetune = mobilebert_finetune_QAT.MobileBertFinetune.__new__(
            mobilebert_finetune_QAT.MobileBertFinetune
        )
        datasets = DatasetDict(
            train=Dataset.from_dict({"label": list(range(20))}),
            validation=Dataset.from_dict({"label": list(range(10))}),
        )
        finetune.load_CSV_dataset = Mock(return_value=(datasets, {"a": 0, "b": 1}))
        finetune.tokenizer = None
        finetune.metric = None
        finetune.batch_size_training = 2
        finetune.num_epochs = 1
        finetune.training = Mock()
        with tempfile.TemporaryDirectory() as directory, patch.object(
            mobilebert_finetune_QAT.MobileBertForSequenceClassification,
            "from_pretrained",
            return_value=TinyClassifier(),
        ):
            _, returned = finetune.get_finetune_mobilebert(
                directory, max_train_samples=8, learning_rate=5e-4
            )
        training_data = finetune.training.call_args.args[1]
        self.assertEqual(len(training_data["train"]), 8)
        self.assertEqual(len(training_data["validation"]), 10)
        self.assertEqual(len(returned["train"]), 20)
        self.assertEqual(finetune.training.call_args.kwargs["learning_rate"], 5e-4)
        finetune.training.return_value.train.assert_called_once()


if __name__ == "__main__":
    unittest.main()
