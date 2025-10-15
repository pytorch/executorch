# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

import unittest

import torch

from executorch.backends.cadence.aot.quantizer.quantizer import (
    CadenceW8A32MixedQuantizer,
    CadenceDefaultQuantizer,
    get_cadence_default_ops,
)


class DerivedMixedQuantizer(CadenceW8A32MixedQuantizer):
    """
    Test-only subclass to validate MRO aggregation:
    contributes one additional op beyond CadenceW8A32MixedQuantizer.
    """

    ADDITIONAL_OPS_TO_PRESERVE: tuple[torch._ops.OpOverload, ...] = (
        torch.ops.aten.batch_norm.default,
    )


class QuantizerOpsPreserveTest(unittest.TestCase):
    def test_mixed_w8a32_ops_to_preserve(self) -> None:
        q = CadenceW8A32MixedQuantizer()
        actual = q.get_ops_to_preserve_from_decomposition()
        expected = get_cadence_default_ops()
        expected += [
            torch.ops.aten.gru.input,
            torch.ops.aten.gru.data,
        ]
        self.assertCountEqual(actual, expected)

    def test_default_quantizer_ops_to_preserve(self) -> None:
        q = CadenceDefaultQuantizer()
        actual = q.get_ops_to_preserve_from_decomposition()
        expected = get_cadence_default_ops()
        self.assertCountEqual(actual, expected)

    def test_mro_aggregation_includes_subclass_ops(self) -> None:
        """
        Validate MRO aggregation: DerivedMixedQuantizer should include
        base Cadence ops, GRU ops from CadenceW8A32MixedQuantizer, and
        the subclass-contributed batch_norm op.
        """
        q = DerivedMixedQuantizer()
        actual = q.get_ops_to_preserve_from_decomposition()
        expected = get_cadence_default_ops()
        expected += [
            torch.ops.aten.gru.input,
            torch.ops.aten.gru.data,
            torch.ops.aten.batch_norm.default,
        ]
        self.assertCountEqual(actual, expected)


if __name__ == "__main__":
    unittest.main()
