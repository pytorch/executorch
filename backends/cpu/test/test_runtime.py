# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

import unittest
from contextlib import contextmanager

import torch
from executorch.backends.cpu.partitioner import CPUPartitioner
from executorch.exir import to_edge_transform_and_lower
from executorch.extension.pybindings.portable_lib import (
    _load_for_executorch_from_buffer,
)


class CPURuntimeTest(unittest.TestCase):
    @contextmanager
    def _lower(self, model, inputs):
        edge = to_edge_transform_and_lower(
            torch.export.export(model, inputs), partitioner=[CPUPartitioner()]
        )
        program = edge.to_executorch()
        plan = program.executorch_program.execution_plan[0]
        self.assertEqual([delegate.id for delegate in plan.delegates], ["CpuBackend"])
        self.assertEqual(plan.operators, [])
        # The runtime borrows the serialized program buffer.
        yield _load_for_executorch_from_buffer(program.buffer)

    def test_integer_inputs_and_outputs_preserve_precision(self):
        class Add(torch.nn.Module):
            def forward(self, x, y):
                return x + y

        for dtype, values in (
            (torch.uint8, [0, 128, 250]),
            (torch.int8, [-100, 0, 100]),
            (torch.int16, [-30000, 0, 30000]),
            (torch.int32, [-(2**24 + 1), 0, 2**24 + 1]),
            (torch.int64, [-(2**40 + 3), 0, 2**40 + 3]),
        ):
            with self.subTest(dtype=dtype):
                model = Add()
                x = torch.tensor(values, dtype=dtype)
                y = torch.ones_like(x)
                with self._lower(model, (x, y)) as runtime:
                    for inputs in ((x, y), (y, x)):
                        actual = runtime.forward(inputs)[0]
                        torch.testing.assert_close(
                            actual, model(*inputs), rtol=0, atol=0
                        )

    def test_integer_indices_and_mixed_outputs(self):
        class Select(torch.nn.Module):
            def forward(self, x, indices):
                selected = torch.index_select(x, 1, indices)
                positive = selected > 0
                return (
                    torch.where(positive, selected, -selected),
                    selected.argmax(dim=1),
                    positive,
                )

        model = Select()
        x = torch.tensor([[-3.0, 5.0, 2.0], [7.0, -2.0, -4.0]])
        for dtype in (torch.int32, torch.int64):
            with self.subTest(dtype=dtype):
                indices = torch.tensor([2, 0], dtype=dtype)
                with self._lower(model, (x, indices)) as runtime:
                    for current in (indices, indices.flip(0)):
                        actual = runtime.forward((x, current))
                        for output, expected in zip(
                            actual, model(x, current), strict=True
                        ):
                            torch.testing.assert_close(output, expected, rtol=0, atol=0)

    def test_boolean_inputs_and_integer_constants(self):
        class MaskedAdd(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.register_buffer("offset", torch.tensor([2**40 + 3, -7]))

            def forward(self, x, mask):
                return torch.where(mask, x + self.offset, x), torch.logical_not(mask)

        model = MaskedAdd()
        x = torch.tensor([2**40 + 1, -(2**40 + 5)])
        mask = torch.tensor([True, False])
        with self._lower(model, (x, mask)) as runtime:
            for current in (mask, ~mask):
                actual = runtime.forward((x, current))
                for output, expected in zip(actual, model(x, current), strict=True):
                    torch.testing.assert_close(output, expected, rtol=0, atol=0)

    def test_empty_tensors_execute(self):
        class Add(torch.nn.Module):
            def forward(self, x):
                return x + 1

        model = Add()
        x = torch.empty(2, 0, 3)
        with self._lower(model, (x,)) as runtime:
            actual = runtime.forward((x,))[0]
            torch.testing.assert_close(actual, model(x))

    def test_neg_view_copy_routes_to_neg(self):
        class NegView(torch.nn.Module):
            def forward(self, x):
                return torch.ops.aten._neg_view_copy.default(x)

        model = NegView()
        x = torch.randn(2, 3)
        with self._lower(model, (x,)) as runtime:
            actual = runtime.forward((x,))[0]
            torch.testing.assert_close(actual, model(x))

    def test_neg_view_copy_empty_bool(self):
        class NegView(torch.nn.Module):
            def forward(self, x):
                return torch.ops.aten._neg_view_copy.default(x)

        model = NegView()
        x = torch.empty(0, dtype=torch.bool)
        expected = model(x)
        with self._lower(model, (x,)) as runtime:
            actual = runtime.forward((x,))[0]
            torch.testing.assert_close(actual, expected)

    def test_optional_integer_index_lists(self):
        class Index(torch.nn.Module):
            def __init__(self, accumulate):
                super().__init__()
                self.accumulate = accumulate

            def forward(self, x, indices, values):
                updated = torch.index_put(x, (indices,), values, self.accumulate)
                return updated, updated[indices], updated[:, indices]

        x = torch.arange(9, dtype=torch.float32).reshape(3, 3)
        indices = torch.tensor([0, 2])
        values = torch.tensor([[10.0, 20.0, 30.0], [-1.0, -2.0, -3.0]])
        inputs = (x, indices, values)
        for accumulate in (False, True):
            with self.subTest(accumulate=accumulate):
                model = Index(accumulate)
                with self._lower(model, inputs) as runtime:
                    actual = runtime.forward(inputs)
                    for output, expected in zip(actual, model(*inputs), strict=True):
                        torch.testing.assert_close(output, expected, rtol=0, atol=0)

    def test_batch_norm_fused_into_conv_and_linear(self):
        class ConvBnLinearBn(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.conv = torch.nn.Conv2d(4, 4, 3, padding=1)
                self.bn1 = torch.nn.BatchNorm2d(4)
                self.linear = torch.nn.Linear(4, 8)
                self.bn2 = torch.nn.BatchNorm1d(8)

            def forward(self, x):
                y = self.bn1(self.conv(x))
                return self.bn2(self.linear(y.mean(dim=(2, 3))))

        torch.manual_seed(0)
        model = ConvBnLinearBn().eval()
        for bn in (model.bn1, model.bn2):
            bn.running_mean.uniform_(-1, 1)
            bn.running_var.uniform_(0.5, 2)
            bn.weight.data.uniform_(0.5, 2)
            bn.bias.data.uniform_(-1, 1)
        inputs = (torch.randn(2, 4, 8, 8),)
        with self._lower(model, inputs) as runtime:
            torch.testing.assert_close(
                runtime.forward(inputs)[0], model(*inputs), rtol=1e-5, atol=1e-5
            )
