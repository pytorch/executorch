# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import unittest

import torch
from executorch.backends.transforms.addmm_mm_to_linear import (
    apply_addmm_mm_to_linear_transform,
)
from executorch.exir import to_edge
from executorch.exir.dialects._ops import ops as exir_ops
from torch.export import Dim


def _dequantize(xq: torch.Tensor) -> torch.Tensor:
    return torch.ops.quantized_decomposed.dequantize_per_tensor.tensor(
        xq, torch.tensor(0.1), torch.tensor(0), -128, 127, torch.int8
    )


def _int8(*shape: int) -> torch.Tensor:
    return torch.randint(-128, 127, shape, dtype=torch.int8)


class _DequantizeLinearView(torch.nn.Module):
    def __init__(self, swap: bool) -> None:
        super().__init__()
        self.linear = torch.nn.Linear(16, 8)
        self.swap = swap

    def forward(self, xq: torch.Tensor) -> torch.Tensor:
        a, b = xq.shape[0], xq.shape[1]
        y = self.linear(_dequantize(xq).view(-1, 16))
        return y.view(b, a, 8) if self.swap else y.view(a, b, 8)


_BOUNDED = ({0: Dim("a", min=2, max=8), 1: Dim("b", min=2, max=8)},)
_UNBOUNDED = ({0: Dim("a"), 1: Dim("b")},)


class TestAddmmMmToLinear(unittest.TestCase):
    def _transform(self, model, example, dynamic_shapes=None):
        ep = torch.export.export(model, (example,), dynamic_shapes=dynamic_shapes)
        program = to_edge(ep).exported_program()
        apply_addmm_mm_to_linear_transform(program.graph_module.graph)
        return program

    def _output_target(self, program):
        return program.graph_module.graph.output_node().args[0][0].target

    def test_view_to_the_linear_output_shape_is_removed(self):
        model = _DequantizeLinearView(swap=False)
        for name, dynamic_shapes in (
            ("static", None),
            ("bounded", _BOUNDED),
            ("unbounded", _UNBOUNDED),
        ):
            with self.subTest(name):
                program = self._transform(model, _int8(2, 3, 16), dynamic_shapes)
                self.assertEqual(
                    self._output_target(program), exir_ops.edge.aten.linear.default
                )
                x = _int8(2, 3, 16)
                torch.testing.assert_close(program.module()(x), model(x))

    def test_view_to_another_shape_with_equal_bounds_is_kept(self):
        model = _DequantizeLinearView(swap=True)
        # [b, a, 8] and the linear output [a, b, 8] have the same upper bounds,
        # and without upper bounds the same example sizes.
        for name, example, dynamic_shapes in (
            ("bounded", _int8(2, 3, 16), _BOUNDED),
            ("unbounded", _int8(3, 3, 16), _UNBOUNDED),
        ):
            with self.subTest(name):
                program = self._transform(model, example, dynamic_shapes)
                self.assertEqual(
                    self._output_target(program), exir_ops.edge.aten.view_copy.default
                )
                x = _int8(2, 3, 16)
                torch.testing.assert_close(program.module()(x), model(x))


if __name__ == "__main__":
    unittest.main()
