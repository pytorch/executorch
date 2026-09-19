# Copyright (c) Qualcomm Innovation Center, Inc.
# All rights reserved
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import unittest
from unittest.mock import MagicMock

import torch

from executorch.backends.qualcomm.genai_pipeline.datasets.default_training_data_adapter import (
    DefaultTrainingDataAdapter,
)


class TestGenerateTrainingData(unittest.TestCase):

    def setUp(self):
        self.adapter = DefaultTrainingDataAdapter()
        self.tokenizer = MagicMock()

    def test_returns_provided_feature_label_pairs(self):
        training_data = [((torch.zeros(1, 4),), torch.ones(1))]
        result = self.adapter.generate_training_data(
            self.tokenizer, extra_options={"training_data": training_data}
        )
        self.assertIs(result, training_data)

    def test_accepts_a_dataloader(self):
        dataloader = torch.utils.data.DataLoader(
            [(torch.zeros(4), torch.ones(1))], batch_size=1
        )
        result = self.adapter.generate_training_data(
            self.tokenizer, extra_options={"training_data": dataloader}
        )
        self.assertIs(result, dataloader)

    def test_raises_when_no_training_data_supplied(self):
        # Labelled data cannot be synthesized, so QAT must fail loudly rather
        # than silently degrading to PTQ.
        for extra_options in (None, {}, {"training_data": None}):
            with self.subTest(extra_options=extra_options):
                with self.assertRaises(ValueError):
                    self.adapter.generate_training_data(
                        self.tokenizer, extra_options=extra_options
                    )


if __name__ == "__main__":
    unittest.main()
