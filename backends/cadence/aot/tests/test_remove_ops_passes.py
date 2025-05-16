# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-unsafe


import unittest
from typing import cast, Tuple

import executorch.backends.cadence.aot.ops_registrations  # noqa
import torch
import torch.nn as nn
import torch.nn.functional as F
from executorch.backends.cadence.aot import compiler
from executorch.backends.cadence.aot.compiler import export_to_edge
from executorch.backends.cadence.aot.fuse_ops import FuseQuantDequantToRequantizePass
from executorch.backends.cadence.aot.graph_builder import GraphBuilder

from executorch.backends.cadence.aot.pass_utils import count_node, op_counts_match
from executorch.backends.cadence.aot.remove_ops import (
    RemoveAliasCopyOpPass,
    RemoveBranchedQuantDequant,
    RemoveCatFromSliceCopyPass,
    RemoveCloneOpPass,
    RemoveContiguousOpPass,
    RemoveDetachCopyPass,
    RemoveNopAddOpPass,
    RemoveNopExpandOpPass,
    RemoveNopLinalgVectorNormOpPass,
    RemoveNopMulOpPass,
    RemoveNopSelectOpPass,
    RemoveNopSliceOrViewOpPass,
    RemovePermutesAroundElementwiseOps,
    RemoveToOpsPass,
    RemoveZeroSizedCatArgsPass,
    RemoveZeroSizedConstantPadNd,
)
from executorch.exir.dialects._ops import ops as exir_ops
from parameterized.parameterized import parameterized
from pyre_extensions import none_throws

from torch.fx.passes.infra.pass_base import PassResult


class TestRemoveOpsPasses(unittest.TestCase):
    @parameterized.expand(
        [
            [(1, 2, 3)],
        ]
    )
    @torch.no_grad()
    def test_remove_to_ops(self, shape: Tuple[int]):
        class M(torch.nn.Module):
            def forward(self, x: torch.Tensor):
                return exir_ops.edge.aten.to(x, dtype=torch.float32)

        model = M()
        x = torch.randn(shape)
        graph_module = export_to_edge(model, (x,)).exported_program().graph_module
        p = RemoveToOpsPass()

        graph_after_passes = cast(PassResult, p(graph_module)).graph_module

        self.assertEqual(
            count_node(graph_after_passes, exir_ops.edge.aten.to.dtype),
            0,
        )

        self.assertEqual(
            count_node(graph_after_passes, exir_ops.edge.aten.to.dtype_layout),
            0,
        )

    @parameterized.expand(
        [
            [(7, 6, 5)],
            [(7, 6)],
            [(7,)],
        ]
    )
    @torch.no_grad()
    def test_remove_nop_add_op_pass(self, shape: Tuple[int]):
        class FullX(torch.nn.Module):
            def forward(self, t: torch.Tensor):
                return torch.add(torch.full(shape, 0), t)

        class FullY(torch.nn.Module):
            def forward(self, t: torch.Tensor):
                return torch.add(t, torch.full(shape, 0))

        model = FullX()
        t = torch.full(shape, 3)
        graph_module = export_to_edge(model, (t,)).exported_program().graph_module

        p = RemoveNopAddOpPass()

        graph_after_passes = cast(PassResult, p(graph_module)).graph_module
        self.assertEqual(
            count_node(graph_after_passes, exir_ops.edge.aten.add.Tensor),
            0,
        )

        model = FullY()
        graph_module = export_to_edge(model, (t,)).exported_program().graph_module

        graph_after_passes = cast(PassResult, p(graph_module)).graph_module

        self.assertEqual(
            count_node(graph_after_passes, exir_ops.edge.aten.add.Tensor),
            0,
        )

    @parameterized.expand(
        [
            [(7, 6, 5)],
            [(7, 6)],
            [(7,)],
        ]
    )
    @torch.no_grad()
    def test_remove_nop_mul_op_pass(self, shape: Tuple[int]):
        class FullX(torch.nn.Module):
            def forward(self, t: torch.Tensor):
                return torch.mul(torch.full(shape, 0), t)

        class FullY(torch.nn.Module):
            def forward(self, t: torch.Tensor):
                return torch.mul(t, torch.full(shape, 0))

        model = FullX()
        t = torch.full(shape, 3)
        graph_module = export_to_edge(model, (t,)).exported_program().graph_module

        p = RemoveNopMulOpPass()

        graph_after_passes = cast(PassResult, p(graph_module)).graph_module
        self.assertEqual(
            count_node(graph_after_passes, exir_ops.edge.aten.mul.Tensor),
            0,
        )

        model = FullY()
        graph_module = export_to_edge(model, (t,)).exported_program().graph_module

        graph_after_passes = cast(PassResult, p(graph_module)).graph_module

        self.assertEqual(
            count_node(graph_after_passes, exir_ops.edge.aten.mul.Tensor),
            0,
        )

    @parameterized.expand(
        [
            [(1, 2, 3)],
        ]
    )
    @torch.no_grad()
    def test_remove_alias_copy(self, shape: Tuple[int]):
        class M(torch.nn.Module):
            def forward(self, x: torch.Tensor):
                return exir_ops.edge.aten.alias_copy(x)

        model = M()
        x = torch.randn(shape)
        graph_module = export_to_edge(model, (x,)).exported_program().graph_module

        p = RemoveAliasCopyOpPass()

        graph_after_passes = cast(PassResult, p(graph_module)).graph_module

        self.assertEqual(
            count_node(graph_after_passes, exir_ops.edge.aten.alias_copy.default),
            0,
        )

    @parameterized.expand(
        [
            [(1, 2, 3)],
        ]
    )
    @torch.no_grad()
    def test_remove_detach_copy(self, shape: Tuple[int]):
        # aten::detach is converted to aten::alias_copy after functionalization & decomposition.
        class M(torch.nn.Module):
            def forward(self, x: torch.Tensor):
                return exir_ops.edge.aten.detach_copy(x)

        model = M()
        x = torch.randn(shape)
        graph_module = export_to_edge(model, (x,)).exported_program().graph_module

        p = RemoveDetachCopyPass()

        graph_after_passes = cast(PassResult, p(graph_module)).graph_module

        self.assertEqual(
            count_node(graph_after_passes, exir_ops.edge.aten.detach_copy.default),
            0,
        )

    @parameterized.expand(
        [
            [(1, 2, 3), (0, 0)],
        ]
    )
    @torch.no_grad()
    def test_remove_zero_sized_constant_pad_nd(
        self, shape: Tuple[int], padding: Tuple[int]
    ):
        # F.pad is converted to aten::constant_pad_nd after functionalization & decomposition.
        class Padding(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.padding = padding

            def forward(self, x: torch.Tensor):
                return F.pad(x, self.padding)

        model = Padding()
        x = torch.randn(shape)
        graph_module = export_to_edge(model, (x,)).exported_program().graph_module

        p = RemoveZeroSizedConstantPadNd()

        graph_after_passes = cast(PassResult, p(graph_module)).graph_module

        self.assertEqual(
            count_node(graph_after_passes, exir_ops.edge.aten.constant_pad_nd.default),
            0,
        )

    def test_remove_expand(self):
        class Expand(torch.nn.Module):
            def forward(self, x):
                return torch.ops.aten.expand_copy(x, [2, 3, 5])

        x = torch.ones(2, 3, 5)
        p = RemoveNopExpandOpPass()
        graph_module = export_to_edge(Expand(), (x,)).exported_program().graph_module
        graph_module = p(graph_module).graph_module
        # Assert that expand op is optimized away, since it is a nop
        self.assertEqual(
            count_node(graph_module, exir_ops.edge.aten.expand_copy.default), 0
        )

    def test_remove_zero_arg_cat(self):
        class Cat(torch.nn.Module):
            def forward(self, x, y):
                return torch.ops.aten.cat((x, y), 0)

        x = torch.ones(1, 0, 3, 5)
        y = torch.ones(2, 0, 3, 5)
        graph_module = (
            compiler.export_to_cadence(Cat(), (x, y)).exported_program().graph_module
        )
        # Assert that cat op is optimized away, since it concatenates
        # two zero-sized tensors
        self.assertEqual(count_node(graph_module, exir_ops.edge.aten.cat.default), 0)

    def test_remove_single_arg_cat(self):
        class Cat(torch.nn.Module):
            def forward(self, x, y):
                z = torch.ones(0, 5)
                # z is an empty tensor, and concatenation of x with z will
                # be x. So we can safely eliminate the following cat op.
                x1 = torch.ops.aten.cat((x, z))
                x2 = torch.add(x1, 2.4, 3.1)
                y1 = torch.add(y, 1, 2)
                return torch.add(x2, y1)

        x = torch.ones(3, 5)
        y = torch.ones(3, 5)
        graph_module = export_to_edge(Cat(), (x, y)).exported_program().graph_module
        new_graph_module = RemoveZeroSizedCatArgsPass()(graph_module).graph_module
        new_graph_module.graph.eliminate_dead_code()
        # Assert that x1 is optimized away
        self.assertEqual(count_node(new_graph_module, torch.ops.aten.cat.out), 0)

    def test_remove_zero_sized_cat(self):
        class Cat(torch.nn.Module):
            def __init__(self, dim: int):
                super().__init__()
                self.dim = dim

            def forward(self, tensors):
                return torch.cat(tensors, self.dim)

        shapes, dim, dtype, _max = [(1, 0, 3), (2, 0, 3)], 0, torch.float32, 127

        in_tensors = [(torch.rand(shape) * _max).to(dtype=dtype) for shape in shapes]

        model = Cat(dim)
        graph_module = (
            export_to_edge(model, (in_tensors,)).exported_program().graph_module
        )
        new_graph_module = RemoveZeroSizedCatArgsPass()(graph_module).graph_module
        new_graph_module.graph.eliminate_dead_code()
        self.assertEqual(count_node(new_graph_module, torch.ops.aten.cat.out), 0)

    def test_remove_clone(self):
        class Clone(torch.nn.Module):
            def forward(self, x, y):
                t1 = x.clone()
                t2 = y.clone()
                return t1 + t2

        x = torch.ones(3, 5)
        y = torch.ones(3, 5)
        graph_module = export_to_edge(Clone(), (x, y)).exported_program().graph_module
        new_graph_module = RemoveCloneOpPass()(graph_module).graph_module
        new_graph_module.graph.eliminate_dead_code()
        # Assert that t1 and t2 are optimized away
        self.assertEqual(count_node(new_graph_module, torch.ops.aten.clone.out), 0)

    def test_remove_contiguous(self):
        class Contiguous(torch.nn.Module):
            def forward(self, x, y):
                t1 = x.contiguous()
                t2 = y.contiguous()
                return t1 + t2

        x = torch.ones(3, 5)
        y = torch.ones(3, 5)
        graph_module = (
            export_to_edge(Contiguous(), (x, y)).exported_program().graph_module
        )
        new_graph_module = RemoveContiguousOpPass()(graph_module).graph_module
        new_graph_module.graph.eliminate_dead_code()
        # Assert that t1 and t2 are optimized away
        self.assertEqual(count_node(new_graph_module, torch.ops.aten.contiguous.out), 0)

    @parameterized.expand(
        [
            [(3, 5), [3, 5]],
            [(1,), [-1]],
        ]
    )
    @torch.no_grad()
    def test_remove_nop_view(self, shape, new_shape):
        class View(torch.nn.Module):
            def __init__(self, new_shape):
                super().__init__()
                self.new_shape = new_shape

            def forward(self, x: torch.Tensor):
                return x.view(self.new_shape)

        model = View(new_shape)
        x = torch.randn(shape)
        graph_module = export_to_edge(model, (x,)).exported_program().graph_module
        p = RemoveNopSliceOrViewOpPass()
        graph_after_passes = cast(PassResult, p(graph_module)).graph_module
        graph_after_passes.graph.eliminate_dead_code()
        # Assert that view op was removed
        self.assertEqual(
            count_node(graph_after_passes, exir_ops.edge.aten.view_copy.default), 0
        )

    def test_remove_nop_slice(self):
        class Slice(torch.nn.Module):
            def forward(self, x):
                return torch.slice_copy(x, dim=0, start=0, step=1)

        x = torch.ones(3, 5)
        model = Slice()
        graph_module = export_to_edge(model, (x,)).exported_program().graph_module
        p = RemoveNopSliceOrViewOpPass()
        graph_after_passes = cast(PassResult, p(graph_module)).graph_module
        graph_after_passes.graph.eliminate_dead_code()
        # Assert that slice op was removed
        self.assertEqual(
            count_node(graph_after_passes, exir_ops.edge.aten.slice_copy.Tensor), 0
        )

    def test_remove_nop_select(self):
        class SelectFeasible1(torch.nn.Module):
            def forward(self, x):
                y = x.select(0, 0)
                z = y.view([1, 5, 6])
                return z

        x = torch.ones(1, 5, 6)
        graph_module = (
            export_to_edge(SelectFeasible1(), (x,)).exported_program().graph_module
        )
        self.assertEqual(
            count_node(graph_module, exir_ops.edge.aten.select_copy.int), 1
        )
        graph_module = RemoveNopSelectOpPass()(graph_module).graph_module
        # Assert that select op was removed
        self.assertEqual(
            count_node(graph_module, exir_ops.edge.aten.select_copy.int), 0
        )

        class SelectFeasible2(torch.nn.Module):
            def forward(self, x, y):
                x = x.select(0, 0)
                z = x + y
                return z

        x = torch.ones(1, 5, 6)
        y = torch.ones(1, 5, 6)
        graph_module = (
            export_to_edge(SelectFeasible2(), (x, y)).exported_program().graph_module
        )
        self.assertEqual(
            count_node(graph_module, exir_ops.edge.aten.select_copy.int), 1
        )
        graph_module = RemoveNopSelectOpPass()(graph_module).graph_module
        # Assert that select op was removed
        self.assertEqual(
            count_node(graph_module, exir_ops.edge.aten.select_copy.int), 0
        )

        class SelectFeasible3(torch.nn.Module):
            def forward(self, x, y):
                x = x.select(0, 0)
                z = x * y
                return z

        x = torch.ones(1, 5, 6)
        y = torch.ones(1, 5, 6)
        graph_module = (
            export_to_edge(SelectFeasible3(), (x, y)).exported_program().graph_module
        )
        self.assertEqual(
            count_node(graph_module, exir_ops.edge.aten.select_copy.int), 1
        )
        graph_module = RemoveNopSelectOpPass()(graph_module).graph_module
        # Assert that select op was removed
        self.assertEqual(
            count_node(graph_module, exir_ops.edge.aten.select_copy.int), 0
        )

        class SelectFeasible4(torch.nn.Module):
            def forward(self, x, y):
                x = x.select(0, 0)
                z = x / y
                return z

        x = torch.ones(1, 5, 6)
        y = torch.ones(1, 5, 6)
        graph_module = (
            export_to_edge(SelectFeasible4(), (x, y)).exported_program().graph_module
        )
        self.assertEqual(
            count_node(graph_module, exir_ops.edge.aten.select_copy.int), 1
        )
        graph_module = RemoveNopSelectOpPass()(graph_module).graph_module
        # Assert that select op was removed
        self.assertEqual(
            count_node(graph_module, exir_ops.edge.aten.select_copy.int), 0
        )

    def test_remove_nop_quant_dequant(self):
        builder = GraphBuilder()
        x = builder.placeholder("x", torch.randn(8, 8))
        q0 = builder.call_operator(
            op=exir_ops.edge.cadence.quantize_per_tensor.default,
            args=(x, 0.01662161760032177, -4, -128, 127, torch.int8),
        )
        dq0 = builder.call_operator(
            op=exir_ops.edge.cadence.dequantize_per_tensor.default,
            args=(q0, 0.01662161760032177, -4, -128, 127, torch.int8),
        )
        q1 = builder.call_operator(
            op=exir_ops.edge.cadence.quantize_per_tensor.default,
            args=(x, 0.012577153742313385, -9, -128, 127, torch.int8),
        )
        builder.output([dq0, q1])
        graph_module = builder.get_graph_module()

        # Expect the dq op to be removed by the pass
        self.assertEqual(
            count_node(
                graph_module, exir_ops.edge.cadence.dequantize_per_tensor.default
            ),
            1,
        )

        # Expect 1 quantize op left since it has no matching dequant
        self.assertEqual(
            count_node(graph_module, exir_ops.edge.cadence.quantize_per_tensor.default),
            2,
        )

        p = FuseQuantDequantToRequantizePass()

        graph_after_passes = cast(PassResult, p(graph_module)).graph_module

        # Expect the dq op to be removed by the pass
        self.assertEqual(
            count_node(
                graph_after_passes, exir_ops.edge.cadence.dequantize_per_tensor.default
            ),
            0,
        )

        # Expect 1 quantize op left since it has no matching dequant
        self.assertEqual(
            count_node(
                graph_after_passes, exir_ops.edge.cadence.quantize_per_tensor.default
            ),
            1,
        )

    def test_remove_nop_aten_linalg_vector_norm(self):
        class LinalgVectorNorm(torch.nn.Module):
            def forward(self, x: torch.Tensor):
                return torch.linalg.vector_norm(x, 2, [0, 1], True)

        model = LinalgVectorNorm()
        x = torch.randn([1, 1, 128])
        inputs = (x,)

        graph_module = (
            export_to_edge(
                model,
                inputs,
            )
            .exported_program()
            .graph_module
        )

        graph_module = none_throws(
            RemoveNopLinalgVectorNormOpPass()(graph_module)
        ).graph_module

        # Expect the linalg_vector_norm op to be removed by the pass
        self.assertEqual(
            count_node(graph_module, exir_ops.edge.aten.linalg_vector_norm.default)
            + count_node(
                graph_module, exir_ops.edge.cadence.linalg_vector_norm.default
            ),
            0,
        )

    def test_remove_permutes_around_elemwise_ops_add(self) -> None:
        class M(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.conv = nn.Conv2d(8, 8, 1, bias=False)

            def forward(self, x):
                x = self.conv(x)
                x = torch.permute(x, [0, 3, 1, 2])
                x = torch.add(x, x)
                x = torch.permute(x, [0, 2, 3, 1])
                x = self.conv(x)
                return x

        inputs = (torch.randn(1, 8, 4, 4),)
        graph_module = export_to_edge(M(), inputs).exported_program().graph_module
        p = RemovePermutesAroundElementwiseOps()
        graph_module = cast(PassResult, p(graph_module)).graph_module

        self.assertEqual(
            count_node(graph_module, exir_ops.edge.aten.permute_copy.default), 0
        )

    def test_remove_permutes_around_elemwise_ops_add_mean(self) -> None:
        class M(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.conv2d = nn.Conv2d(8, 8, 1)

            def forward(self, x, y):
                x = self.conv2d(x)
                y = self.conv2d(y)
                x = torch.permute(x, [0, 3, 1, 2])
                y = torch.permute(y, [0, 3, 1, 2])
                z = torch.add(x, y)
                z = torch.mean(z, dim=[-1, -3], keepdim=True)
                z = torch.permute(z, [0, 2, 3, 1])
                z = self.conv2d(z)
                return z

        inputs = (torch.randn(1, 8, 4, 4), torch.randn(1, 8, 4, 4))
        graph_module = export_to_edge(M(), inputs).exported_program().graph_module
        p = RemovePermutesAroundElementwiseOps()
        graph_module = cast(PassResult, p(graph_module)).graph_module

        self.assertEqual(
            count_node(graph_module, exir_ops.edge.aten.permute_copy.default), 0
        )

        # verify that mean was updated correctly
        mean = [
            n
            for n in graph_module.graph.nodes
            if n.target == exir_ops.edge.aten.mean.dim
        ][0]
        self.assertEqual(mean.args[1], [2, 3])

    def test_remove_permutes_around_elemwise_ops_slice(self) -> None:
        builder = GraphBuilder()
        x = builder.placeholder("x", torch.randn(1, 8, 4, 4))
        permute_node = builder.call_operator(
            op=exir_ops.edge.aten.permute_copy.default,
            args=(x, [0, 2, 3, 1]),
        )
        slice_copy = builder.call_operator(
            op=exir_ops.edge.aten.slice_copy.Tensor,
            args=(permute_node, 1, 2, 4, 1),
        )
        output = builder.call_operator(
            op=exir_ops.edge.aten.permute_copy.default,
            args=(slice_copy, [0, 3, 1, 2]),
        )
        builder.output([output])
        graph_module = builder.get_graph_module()

        p = RemovePermutesAroundElementwiseOps()
        graph_module = cast(PassResult, p(graph_module)).graph_module

        # No permutes should remain.
        self.assertEqual(
            count_node(graph_module, exir_ops.edge.aten.permute_copy.default), 0
        )

        # Verify that slice dimension was updated correctly.
        slices = graph_module.graph.find_nodes(
            op="call_function", target=exir_ops.edge.aten.slice_copy.Tensor
        )
        self.assertEqual(len(slices), 1)
        self.assertEqual(slices[0].args[1], 2)

    def test_remove_permutes_around_elemwise_ops_mul(self) -> None:
        class M(torch.nn.Module):
            def forward(self, x, y):
                x = torch.slice_copy(x, 0, 0, 1)
                x = torch.permute(x, [0, 3, 1, 2])
                y = torch.permute(y, [0, 3, 1, 2])
                x = torch.ops.quantized_decomposed.dequantize_per_tensor(
                    x, 1.5, 0, 0, 255, torch.uint8
                )
                z = x * y
                z = torch.ops.quantized_decomposed.quantize_per_tensor(
                    z, 2.5, 0, 0, 255, torch.uint8
                )
                z = torch.permute(z, [0, 2, 3, 1])
                z = torch.unsqueeze_copy(z, 0)
                return z

        inputs = (torch.randn(2, 4, 4, 8), torch.randn(2, 4, 4, 8))
        graph_module = export_to_edge(M(), inputs).exported_program().graph_module

        p = RemovePermutesAroundElementwiseOps()
        graph_module = cast(PassResult, p(graph_module)).graph_module

        self.assertEqual(
            count_node(graph_module, exir_ops.edge.aten.permute_copy.default), 0
        )

    def test_remove_permutes_around_elemwise_ops_double_permutes(self) -> None:
        class M(torch.nn.Module):
            def forward(self, x, y):
                x = torch.slice_copy(x, 0, 0, 1)
                x = torch.permute(x, [0, 3, 1, 2])
                x = torch.permute(x, [0, 3, 1, 2])
                x = torch.ops.quantized_decomposed.dequantize_per_tensor(
                    x, 1.5, 0, 0, 255, torch.uint8
                )
                y = torch.permute(y, [0, 3, 1, 2])
                y = torch.ops.quantized_decomposed.dequantize_per_tensor(
                    y, 1.5, 0, 0, 255, torch.uint8
                )
                z = torch.cat((x, y), 1)
                z = torch.ops.quantized_decomposed.quantize_per_tensor(
                    z, 2.5, 0, 0, 255, torch.uint8
                )
                z = torch.permute(z, [0, 2, 3, 1])
                z = torch.permute(z, [0, 2, 3, 1])
                z = torch.unsqueeze_copy(z, 0)
                return z

        inputs = (torch.randn(2, 4, 4, 8), torch.randn(1, 8, 4, 4))
        graph_module = export_to_edge(M(), inputs).exported_program().graph_module
        p = RemovePermutesAroundElementwiseOps()
        graph_module = cast(PassResult, p(graph_module)).graph_module

        # Expect 2 permutes to remain, one on input x and one on output z
        self.assertEqual(
            count_node(graph_module, exir_ops.edge.aten.permute_copy.default), 2
        )

        # verify that cat was updated correctly
        cat = [
            n
            for n in graph_module.graph.nodes
            if n.target == exir_ops.edge.aten.cat.default
        ][0]
        self.assertEqual(cat.args[1], 3)

    def test_remove_permutes_around_elemwise_ops_complicated_case(self) -> None:
        """
        A complicated case touching many edge cases.
        """
        builder = GraphBuilder()
        a = builder.placeholder("a", torch.randn(1, 4, 4, 8))
        a = builder.call_operator(
            op=exir_ops.edge.aten.permute_copy.default,
            args=(a, [0, 3, 1, 2]),
        )
        # Multiple inputs to same subgraph:
        b = builder.placeholder("b", torch.randn(1, 4, 4, 8))
        b = builder.call_operator(
            op=exir_ops.edge.aten.permute_copy.default,
            args=(b, [0, 3, 1, 2]),
        )
        c = builder.call_operator(op=exir_ops.edge.aten.hardtanh.default, args=(b,))
        # Traverse upstream:
        d = builder.call_operator(op=exir_ops.edge.aten.add.Tensor, args=(a, c))
        # Will see an already visited node:
        e = builder.call_operator(op=exir_ops.edge.aten.add.Tensor, args=(d, c))
        f = builder.call_operator(
            op=exir_ops.edge.aten.permute_copy.default,
            args=(e, [0, 2, 3, 1]),
        )
        # Multiple outputs from the same subgraph:
        g = builder.call_operator(
            op=exir_ops.edge.aten.permute_copy.default,
            args=(c, [0, 2, 3, 1]),
        )
        # Different subgraphs from the same input permutation:
        h = builder.call_operator(op=exir_ops.edge.aten.hardtanh.default, args=(b,))
        h = builder.call_operator(
            op=exir_ops.edge.aten.permute_copy.default,
            args=(h, [0, 2, 3, 1]),
        )
        # Bad output permutation:
        i = builder.call_operator(op=exir_ops.edge.aten.hardtanh.default, args=(b,))
        i = builder.call_operator(
            op=exir_ops.edge.aten.permute_copy.default,
            args=(i, [0, 3, 1, 2]),
        )
        # Bad input permutation:
        j = builder.placeholder("j", torch.randn(1, 4, 4, 8))
        j = builder.call_operator(
            op=exir_ops.edge.aten.permute_copy.default,
            args=(j, [0, 3, 2, 1]),
        )
        k = builder.call_operator(op=exir_ops.edge.aten.add.Tensor, args=(b, j))
        k = builder.call_operator(
            op=exir_ops.edge.aten.permute_copy.default,
            args=(k, [0, 3, 2, 1]),
        )
        builder.output([f, g, h, i, k])
        graph_module = builder.get_graph_module()

        p = RemovePermutesAroundElementwiseOps()
        graph_module = cast(PassResult, p(graph_module)).graph_module

        # Permutations (a, f, g, h) will be eliminated but (b, i, j, k) will remain.
        self.assertEqual(
            count_node(graph_module, exir_ops.edge.aten.permute_copy.default), 4
        )

    def test_remove_permutes_around_elemwise_ops_noop(self) -> None:
        class M(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.conv = nn.Conv2d(8, 8, 1, bias=False)

            def forward(self, x):
                x = self.conv(x)
                x = torch.permute(x, [2, 1, 0, 3])
                x = torch.add(x, x)
                x = torch.permute(x, [0, 1, 3, 2])
                x = self.conv(x)
                return x

        inputs = (torch.randn(1, 8, 4, 4),)
        graph_module = export_to_edge(M(), inputs).exported_program().graph_module
        p = RemovePermutesAroundElementwiseOps()
        graph_module = cast(PassResult, p(graph_module)).graph_module

        # Ensure no permutes were removed, since the dimensions don't fit the expected pattern
        self.assertEqual(
            count_node(graph_module, exir_ops.edge.aten.permute_copy.default), 2
        )

    def test_remove_dequant_on_branch(self):
        class M(torch.nn.Module):
            def forward(self, x):
                x = torch.abs(x)
                x0 = torch.ops.quantized_decomposed.quantize_per_tensor(
                    x, 1.2, 3, 0, 127, torch.int8
                )
                x1 = torch.abs(x0)
                y0 = torch.ops.quantized_decomposed.dequantize_per_tensor(
                    x0, 1.2, 3, 0, 127, torch.int8
                )
                y1 = y0.view(-1)
                return x1, y1

        inputs = torch.rand(1, 8, 4, 6)
        model = M()
        graph_module = export_to_edge(model, (inputs,)).exported_program().graph_module

        graph_module = RemoveBranchedQuantDequant()(graph_module).graph_module
        self.assertTrue(
            op_counts_match(
                graph_module,
                expected_op_counts={
                    exir_ops.edge.quantized_decomposed.quantize_per_tensor.default: 1,
                    # we expect the pass to remove the dequantize node
                    exir_ops.edge.quantized_decomposed.dequantize_per_tensor.default: 0,
                    exir_ops.edge.aten.abs.default: 2,
                },
            )
        )

    def test_remove_cat_from_slice_copy_all_removal(self) -> None:
        class M(torch.nn.Module):
            def __init__(self):
                super().__init__()

            def forward(self, x, y):
                x1 = torch.cat((x, y), 0)  # (2, 4)
                return torch.slice_copy(x1, dim=0, start=0, end=1)

        inputs = tuple(torch.randn(2, 4) for _ in range(2))
        graph_module = export_to_edge(M(), inputs).exported_program().graph_module
        p = RemoveCatFromSliceCopyPass()
        graph_module = cast(PassResult, p(graph_module)).graph_module

        # Ensure both cat nodes were removed
        self.assertEqual(count_node(graph_module, exir_ops.edge.aten.cat.default), 0)

    def test_remove_cat_from_slice_copy_no_removal(self) -> None:
        class M(torch.nn.Module):
            def __init__(self):
                super().__init__()

            def forward(self, x, y):
                x1 = torch.cat((x, y), 0)  # (2, 4)
                return torch.slice_copy(x1, dim=0, start=0, end=3)

        inputs = tuple(torch.randn(2, 4) for _ in range(2))
        graph_module = export_to_edge(M(), inputs).exported_program().graph_module
        p = RemoveCatFromSliceCopyPass()
        graph_module = cast(PassResult, p(graph_module)).graph_module

        # Ensure both cat nodes were removed
        self.assertEqual(count_node(graph_module, exir_ops.edge.aten.cat.default), 1)

    def test_remove_cat_from_slice_copy_zero_range(self) -> None:
        class M(torch.nn.Module):
            def __init__(self):
                super().__init__()

            def forward(self, x, y):
                x1 = torch.cat((x, y), 0)  # (2, 4)
                return torch.slice_copy(x1, dim=0, start=0, end=0)

        inputs = tuple(torch.randn(2, 4) for _ in range(2))
        graph_module = export_to_edge(M(), inputs).exported_program().graph_module
        p = RemoveCatFromSliceCopyPass()
        graph_module = cast(PassResult, p(graph_module)).graph_module

        # Ensure both cat nodes were removed
        self.assertEqual(count_node(graph_module, exir_ops.edge.aten.cat.default), 0)

    def test_remove_cat_from_slice_copy_second_input(self) -> None:
        builder = GraphBuilder()
        x = builder.placeholder("x", torch.randn(2, 4))
        y = builder.placeholder("y", torch.randn(2, 4))
        cat = builder.call_operator(
            op=exir_ops.edge.aten.cat.default,
            args=((x, y), 1),
        )
        slice_copy = builder.call_operator(
            op=exir_ops.edge.aten.slice_copy.Tensor,
            args=(cat, 1, 5, 7, 1),
        )
        builder.output([slice_copy])
        graph_module = builder.get_graph_module()

        inputs = (torch.randn(2, 4), torch.randn(2, 4))
        expected_outputs = graph_module(*inputs)[0]

        p = RemoveCatFromSliceCopyPass()
        graph_module = cast(PassResult, p(graph_module)).graph_module

        # Cat should be removed.
        self.assertEqual(count_node(graph_module, exir_ops.edge.aten.cat.default), 0)

        # Output should remain the same.
        self.assertTrue(torch.equal(graph_module(*inputs)[0], expected_outputs))
