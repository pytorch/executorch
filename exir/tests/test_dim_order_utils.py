# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict
import unittest

from typing import List

import torch
from executorch.exir import to_edge, to_edge_transform_and_lower
from executorch.exir.dim_order_utils import get_dim_order, get_memory_format
from executorch.exir.schema import KernelCall, Tensor


class TestDimOrderUtils(unittest.TestCase):
    def test_get_memory_format(self) -> None:
        mem_format = torch.contiguous_format
        for ndim in range(1, 7):
            dim_order = list(range(ndim))
            self.assertEqual(mem_format, get_memory_format(dim_order))

        mem_format = torch.channels_last
        self.assertEqual(mem_format, get_memory_format([0, 2, 3, 1]))

    def test_get_dim_order(self) -> None:
        for ndim in range(1, 7):
            self.assertEqual(
                list(range(ndim)), get_dim_order(torch.contiguous_format, ndim)
            )
        self.assertEqual([0, 2, 3, 1], get_dim_order(torch.channels_last, 4))

    def test_dim_order_from_stride(self):
        class Test(torch.nn.Module):
            def __init__(self):
                super().__init__()

            def forward(self, t1, t2):
                idx = torch.nonzero(t1).reshape(-1)
                y = torch.index_select(t2, 0, idx)
                return y

        M = Test()
        x = torch.tensor([0, 1, 1, 0, 1], dtype=torch.bool)
        y = torch.randn(5, 6)
        M(x, y)

        expo_prog = torch.export.export(M, (x, y))
        edge_prog = to_edge_transform_and_lower(expo_prog)
        edge_prog.to_executorch()

    def test_channels_last_single_channel_conv_dim_order(self) -> None:
        class Conv(torch.nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.conv = torch.nn.Conv2d(1, 8, kernel_size=3, padding=1)

            def forward(self, x: torch.Tensor) -> torch.Tensor:
                return self.conv(x)

        model = Conv().to(memory_format=torch.channels_last).eval()
        inputs = (torch.randn(1, 1, 12, 12).to(memory_format=torch.channels_last),)
        edge_prog = to_edge(torch.export.export(model, inputs, strict=True))
        et_program = edge_prog.to_executorch().executorch_program

        conv_found = False
        for inst in et_program.execution_plan[0].chains[0].instructions:
            kernel = inst.instr_args
            if not isinstance(kernel, KernelCall):
                continue
            op = et_program.execution_plan[0].operators[kernel.op_index]
            if op.name != "aten::convolution":
                continue
            conv_found = True
            tensors: List[Tensor] = []
            for arg in dict.fromkeys(kernel.args):
                val = et_program.execution_plan[0].values[arg].val
                if isinstance(val, Tensor) and len(val.sizes) == 4:
                    tensors.append(val)
            # Input, weight, and output.
            self.assertEqual(len(tensors), 3)
            for tensor in tensors:
                self.assertIn(
                    list(tensor.dim_order),
                    ([0, 1, 2, 3], [0, 2, 3, 1]),
                )
            self.assertEqual(list(tensors[0].dim_order), list(tensors[-1].dim_order))
        self.assertTrue(conv_found)

    def test_singleton_dims_preserve_default_order(self) -> None:
        class Add(torch.nn.Module):
            def forward(self, x, y):
                return x + y

        cases = (
            (
                Add(),
                (
                    torch.randn(2, 1, 3, 1).to(memory_format=torch.channels_last),
                    torch.randn(2, 1, 3, 1),
                ),
            ),
            (
                torch.nn.ReLU(),
                (torch.randn(2, 1, 3, 1, 1).to(memory_format=torch.channels_last_3d),),
            ),
        )
        for model, inputs in cases:
            with self.subTest(model=type(model).__name__):
                program = (
                    to_edge_transform_and_lower(torch.export.export(model, inputs))
                    .to_executorch()
                    .executorch_program
                )
                tensors = [
                    value.val
                    for value in program.execution_plan[0].values
                    if isinstance(value.val, Tensor)
                ]
                self.assertEqual(len(tensors), len(inputs) + 1)
                for tensor in tensors:
                    self.assertEqual(
                        list(tensor.dim_order), list(range(len(tensor.sizes)))
                    )
