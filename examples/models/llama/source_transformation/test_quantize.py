# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import logging
import unittest
from unittest.mock import patch

import torch
import torch.nn as nn

from executorch.examples.models.llama.source_transformation.quantize import quantize


class TestQuantize(unittest.TestCase):
    def test_8da8w_quantizes_and_exports(self):
        model = nn.Sequential(nn.Linear(128, 32), nn.ReLU(), nn.Linear(32, 16))
        inputs = (torch.randn(2, 128),)

        quantize(model, qmode="8da8w", group_size=0)
        exported = torch.export.export(model, inputs)
        output = exported.module()(*inputs)

        self.assertEqual(output.shape, (2, 16))
        self.assertTrue(torch.isfinite(output).all())

    @patch("torchao.quantization.quantize_")
    def test_reports_linears_skipped_by_group_size(self, mock_quantize):
        model = nn.Sequential(nn.Linear(128, 16), nn.Linear(96, 16))

        with self.assertLogs(level=logging.WARNING) as logs:
            quantize(model, qmode="8da4w", group_size=128)

        self.assertIn(
            "8da4w quantization: quantized 1 linear layer(s), skipped 1 linear "
            "layer(s) because in_features is not divisible by group_size=128.",
            logs.output[0],
        )
        mock_quantize.assert_called_once()


if __name__ == "__main__":
    unittest.main()
