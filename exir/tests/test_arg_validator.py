#!/usr/bin/env fbpython
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import itertools
import unittest

import torch
from executorch.exir import EdgeCompileConfig, to_edge
from executorch.exir.dialects._ops import ops
from executorch.exir.dialects.edge._ops import EdgeOpOverload
from executorch.exir.verification.arg_validator import EdgeOpArgValidator
from torch.export import Dim, export


class TestArgValidator(unittest.TestCase):
    """Test for EdgeOpArgValidator"""

    def setUp(self) -> None:
        super().setUp()

    def test_tensor_list_return_preserves_dynamic_batch(self) -> None:
        class Split(torch.nn.Module):
            def __init__(self, operation):
                super().__init__()
                self.operation = operation

            def forward(self, x):
                if self.operation == "chunk":
                    return torch.chunk(x, 3, dim=-1)
                sizes = [2, 2, 2] if self.operation == "split_with_sizes" else 2
                return torch.split(x, sizes, dim=-1)

        dtypes = (
            torch.bool,
            torch.uint8,
            torch.int8,
            torch.int16,
            torch.int32,
            torch.int64,
            torch.uint16,
            torch.float16,
            torch.bfloat16,
            torch.float32,
            torch.float64,
        )
        for operation, dtype in itertools.product(
            ("split", "split_with_sizes", "chunk"), dtypes
        ):
            with self.subTest(operation=operation, dtype=dtype):
                model = Split(operation)
                ep = export(
                    model,
                    (torch.randint(0, 5, (2, 3, 6)).to(dtype),),
                    dynamic_shapes=({0: Dim("batch", min=1, max=8)},),
                )
                ranges = dict(ep.range_constraints)
                edge = to_edge(ep).exported_program()
                self.assertEqual(edge.range_constraints, ranges)
                placeholder = next(
                    node for node in edge.graph.nodes if node.op == "placeholder"
                )
                self.assertIsInstance(placeholder.meta["val"].shape[0], torch.SymInt)
                for batch in (1, 2, 5, 8):
                    x = torch.randint(0, 5, (batch, 3, 6)).to(dtype)
                    torch.testing.assert_close(edge.module()(x), model(x))

    def test_edge_dialect_passes(self) -> None:
        class TestModel(torch.nn.Module):
            def __init__(self):
                super().__init__()

            def forward(self, x):
                return x + x

        m = TestModel()
        inputs = (torch.randn(1, 3, 100, 100).to(dtype=torch.int),)
        egm = to_edge(export(m, inputs, strict=True)).exported_program().graph_module
        validator = EdgeOpArgValidator(egm)
        validator.run(*inputs)
        self.assertEqual(len(validator.violating_ops), 0)

    def test_edge_dialect_fails(self) -> None:
        # torch.complex64 is not supported by edge::aten::add
        class M(torch.nn.Module):
            def __init__(self):
                super().__init__()

            def forward(self, x):
                return x + x

        inputs = (torch.randn(1, 3, 100, 100).to(dtype=torch.complex64),)
        egm = (
            to_edge(
                export(M(), inputs, strict=True),
                compile_config=EdgeCompileConfig(_check_ir_validity=False),
            )
            .exported_program()
            .graph_module
        )
        validator = EdgeOpArgValidator(egm)
        validator.run(*inputs)
        self.assertEqual(len(validator.violating_ops), 1)
        key: EdgeOpOverload = next(iter(validator.violating_ops))
        self.assertEqual(
            key.name(),
            ops.edge.aten.add.Tensor.name(),
        )
        self.assertDictEqual(
            validator.violating_ops[key][0],
            {
                "self": torch.complex64,
                "other": torch.complex64,
                "__ret_0": torch.complex64,
            },
        )

    def test_edge_dialect_fails_on_later_tensor_list_dtype(self) -> None:
        # Only the second tensor has another dtype, so checking the first misses it.
        x = torch.randn(2, 3)
        y = torch.randn(2, 3, dtype=torch.bfloat16)
        graph = torch.fx.Graph()
        tensors = [graph.placeholder("x"), graph.placeholder("y")]
        cat = graph.call_function(ops.edge.aten.cat.default, (tensors,))
        cat.meta["val"] = torch.cat([x, y])
        graph.output(cat)

        validator = EdgeOpArgValidator(torch.fx.GraphModule(torch.nn.Module(), graph))
        validator.run(x, y)
        self.assertEqual(len(validator.violating_ops), 1)
        self.assertDictEqual(
            validator.violating_ops[ops.edge.aten.cat.default][0],
            {"tensors": torch.bfloat16, "__ret_0": torch.float32},
        )
