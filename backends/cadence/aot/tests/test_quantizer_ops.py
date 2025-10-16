# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

import unittest

import torch
from executorch.backends.cadence.aot.quantizer.patterns import AddmmPattern

from executorch.backends.cadence.aot.quantizer.quantizer import (
    CadenceAtenQuantizer,
    CadenceDefaultQuantizer,
    CadenceW8A32MixedQuantizer,
    qconfig_A8W8,
)


class QuantizerOpsPreserveTest(unittest.TestCase):
    def test_mixed_w8a32_ops_to_preserve(self) -> None:
        q = CadenceW8A32MixedQuantizer()
        actual = q.get_ops_to_preserve_from_decomposition()
        expected = [
            torch.ops.aten.linear.default,
            torch.ops.aten.conv1d.default,
            torch.ops.aten.gru.input,
        ]
        self.assertCountEqual(actual, expected)

    def test_default_quantizer_ops_to_preserve(self) -> None:
        q = CadenceDefaultQuantizer()
        actual = q.get_ops_to_preserve_from_decomposition()
        expected = [
            torch.ops.aten.addmm.default,
            torch.ops.aten.bmm.default,
            torch.ops.aten.conv1d.default,
            torch.ops.aten.conv2d.default,
            torch.ops.aten.linear.default,
            torch.ops.aten.matmul.default,
            torch.ops.aten.relu.default,
            torch.ops.aten.relu_.default,
        ]
        self.assertCountEqual(actual, expected)

    def test_nested_quantizer_ops_to_preserve(self) -> None:
        # Setup: Create a nested CadenceQuantizer-like structure by composing
        # - CadenceW8A32MixedQuantizer (which preserves linear, conv1d, gru.input)
        # - A CadenceAtenQuantizer with AddmmPattern (which preserves addmm)
        nested = CadenceDefaultQuantizer(
            quantizers=[
                CadenceW8A32MixedQuantizer(),
                CadenceAtenQuantizer(AddmmPattern(), qconfig_A8W8),
            ]
        )

        # Execute
        actual = nested.get_ops_to_preserve_from_decomposition()

        # Assert: union of both sets without duplicates
        expected = [
            torch.ops.aten.linear.default,
            torch.ops.aten.conv1d.default,
            torch.ops.aten.gru.input,
            torch.ops.aten.addmm.default,
        ]
        self.assertCountEqual(actual, expected)


if __name__ == "__main__":
    unittest.main()
