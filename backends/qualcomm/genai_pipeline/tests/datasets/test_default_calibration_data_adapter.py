# Copyright (c) Qualcomm Innovation Center, Inc.
# All rights reserved
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import unittest
from unittest.mock import MagicMock

import torch

from executorch.backends.qualcomm.genai_pipeline.datasets.default_calibration_data_adapter import (
    DefaultCalibrationDataAdapter,
)

TEST_VOCAB_SIZE = 32000
TEST_NUM_SAMPLES = 4
TEST_SEQ_LENGTH = 16


def _make_tokenizer(vocab_size=TEST_VOCAB_SIZE):
    tokenizer = MagicMock()
    tokenizer.vocab_size = vocab_size
    return tokenizer


class TestGenerateCalibrationData(unittest.TestCase):

    def setUp(self):
        self.adapter = DefaultCalibrationDataAdapter()

    def test_generates_requested_number_of_samples(self):
        data = self.adapter.generate_calibration_data(
            _make_tokenizer(),
            num_samples=TEST_NUM_SAMPLES,
            seq_length=TEST_SEQ_LENGTH,
        )
        self.assertEqual(len(data), TEST_NUM_SAMPLES)

    def test_sample_shape_and_dtype(self):
        data = self.adapter.generate_calibration_data(
            _make_tokenizer(), num_samples=1, seq_length=TEST_SEQ_LENGTH
        )
        input_ids, attention_mask = data[0]
        self.assertEqual(input_ids.shape, (1, TEST_SEQ_LENGTH))
        self.assertEqual(attention_mask.shape, (1, TEST_SEQ_LENGTH))
        self.assertEqual(attention_mask.dtype, torch.long)

    def test_tokens_within_vocab_range(self):
        vocab_size = 128
        data = self.adapter.generate_calibration_data(
            _make_tokenizer(vocab_size),
            num_samples=TEST_NUM_SAMPLES,
            seq_length=TEST_SEQ_LENGTH,
        )
        for input_ids, _ in data:
            self.assertGreaterEqual(int(input_ids.min()), 0)
            self.assertLess(int(input_ids.max()), vocab_size)

    def test_seed_makes_generation_reproducible(self):
        kwargs = {
            "num_samples": TEST_NUM_SAMPLES,
            "seq_length": TEST_SEQ_LENGTH,
            "extra_options": {"seed": 7},
        }
        first = self.adapter.generate_calibration_data(_make_tokenizer(), **kwargs)
        second = self.adapter.generate_calibration_data(_make_tokenizer(), **kwargs)
        for (ids_a, _), (ids_b, _) in zip(first, second):
            self.assertTrue(torch.equal(ids_a, ids_b))

    def test_provided_list_dataset_is_returned_unchanged(self):
        dataset = [(torch.zeros(1, 4, dtype=torch.long),)]
        result = self.adapter.generate_calibration_data(
            _make_tokenizer(), extra_options={"dataset": dataset}
        )
        self.assertIs(result, dataset)

    def test_provided_dataloader_is_returned_unchanged(self):
        # A DataLoader is an Iterable, so it is accepted without materialization.
        dataloader = torch.utils.data.DataLoader(
            [(torch.zeros(4, dtype=torch.long),)], batch_size=1
        )
        result = self.adapter.generate_calibration_data(
            _make_tokenizer(), extra_options={"dataset": dataloader}
        )
        self.assertIs(result, dataloader)

    def test_unusable_vocab_size_raises(self):
        for vocab_size in (None, 0):
            with self.subTest(vocab_size=vocab_size):
                with self.assertRaises(ValueError):
                    self.adapter.generate_calibration_data(_make_tokenizer(vocab_size))


if __name__ == "__main__":
    unittest.main()
