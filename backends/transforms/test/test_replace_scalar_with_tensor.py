# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import unittest
from typing import cast

import torch
from executorch.backends.test.graph_builder import single_op_builder
from executorch.backends.transforms.replace_scalar_with_tensor import (
    ReplaceScalarWithTensorArgPass,
)
from executorch.exir.dialects._ops import ops as exir_ops
from executorch.exir.pass_base import PassResult
from torch.testing._internal.common_utils import TestCase

_BIG = 2**53 + 1


class TestReplaceScalarWithTensorArgPass(TestCase):
    def _scalar_constant(self, scalar, dtype):
        """The value of the ``full`` tensor the pass creates for ``scalar``."""
        x = torch.zeros(3, dtype=dtype)
        original_gm = single_op_builder(
            placeholders=(x,),
            op=exir_ops.edge.aten.add.Scalar,
            args=(x, scalar),
        )
        p = ReplaceScalarWithTensorArgPass()
        graph_after_passes = cast(PassResult, p(original_gm)).graph_module
        (full,) = [
            node
            for node in graph_after_passes.graph.nodes
            if node.op == "call_function"
            and node.target == exir_ops.edge.aten.full.default
        ]
        return full.args[1]

    def test_integer_scalar_is_carried_exactly(self):
        value = self._scalar_constant(_BIG, torch.int64)
        self.assertIsInstance(value, int)
        self.assertTrue(value == _BIG, f"{value} != {_BIG}")

    def test_float_scalar_is_unchanged(self):
        value = self._scalar_constant(1.5, torch.float32)
        self.assertIsInstance(value, float)
        self.assertTrue(value == 1.5, f"{value} != 1.5")


if __name__ == "__main__":
    unittest.main()
