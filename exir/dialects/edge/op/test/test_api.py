#!/usr/bin/env fbpython
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import unittest

import torch

from executorch.exir.dialects.edge.op.api import to_variant
from torchgen.model import SchemaKind

aten = torch.ops.aten

OPS_TO_FUNCTIONAL = {
    aten.add.out: aten.add.Tensor,
    aten._native_batch_norm_legit_no_training.out: aten._native_batch_norm_legit_no_training.default,
    aten.addmm.out: aten.addmm.default,
    aten.view_copy.out: aten.view_copy.default,
}


class TestApi(unittest.TestCase):
    """Test api.py"""

    def test_to_out_variant_returns_self_when_given_out_variant(self) -> None:
        op = aten.add.out
        variant = to_variant(op, SchemaKind.out)
        self.assertEqual(variant, op)

    def test_to_functional_variant_returns_self_when_given_functional(self) -> None:
        op = aten.leaky_relu.default
        variant = to_variant(op, SchemaKind.functional)
        self.assertEqual(variant, op)

    def test_to_functional_variant_returns_correct_op(
        self,
    ) -> None:
        for op in OPS_TO_FUNCTIONAL:
            variant = to_variant(op, SchemaKind.functional)
            self.assertEqual(variant, OPS_TO_FUNCTIONAL[op])

    def test_to_out_variant_returns_correct_op(
        self,
    ) -> None:
        inv_map = {v: k for k, v in OPS_TO_FUNCTIONAL.items()}
        for op in inv_map:
            variant = to_variant(op, SchemaKind.out)
            self.assertEqual(variant, inv_map[op])
