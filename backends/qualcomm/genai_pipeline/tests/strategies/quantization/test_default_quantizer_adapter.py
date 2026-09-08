# Copyright (c) Qualcomm Innovation Center, Inc.
# All rights reserved
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import unittest
from unittest.mock import MagicMock

import torch

from executorch.backends.qualcomm.genai_pipeline.strategies.quantization.default_quantizer_adapter import (
    DefaultQuantizerAdapter,
)


class TestCalibrate(unittest.TestCase):

    def setUp(self):
        self.adapter = DefaultQuantizerAdapter()

    def test_runs_one_forward_pass_per_sample(self):
        model = MagicMock()
        data = [(torch.zeros(1, 2),), (torch.ones(1, 2),)]
        self.adapter.calibrate(model, data)
        self.assertEqual(model.call_count, len(data))

    def test_unpacks_tuple_into_positional_args(self):
        model = MagicMock()
        input_ids, attention_mask = torch.zeros(1, 2), torch.ones(1, 2)
        self.adapter.calibrate(model, [(input_ids, attention_mask)])
        model.assert_called_once_with(input_ids, attention_mask)

    def test_returns_the_model(self):
        model = MagicMock()
        self.assertIs(self.adapter.calibrate(model, []), model)

    def test_accepts_a_dataloader(self):
        # A DataLoader with a collate_fn that yields the sample tuple unchanged
        # is consumed directly, with no re-wrapping by the caller.
        model = MagicMock()
        input_ids, attention_mask = torch.zeros(1, 2), torch.ones(1, 2)
        dataloader = torch.utils.data.DataLoader(
            [(input_ids, attention_mask)],
            batch_size=1,
            collate_fn=lambda batch: batch[0],
        )
        self.adapter.calibrate(model, dataloader)
        model.assert_called_once_with(input_ids, attention_mask)

    def test_empty_dataset_is_a_no_op(self):
        model = MagicMock()
        self.adapter.calibrate(model, [])
        model.assert_not_called()


if __name__ == "__main__":
    unittest.main()
