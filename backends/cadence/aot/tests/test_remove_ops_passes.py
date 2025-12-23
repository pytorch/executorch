# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict


import unittest
from copy import deepcopy
from typing import cast, List, Tuple

import executorch.backends.cadence.aot.ops_registrations  # noqa
import torch
from executorch.backends.cadence.aot.fuse_ops import FuseQuantDequantToRequantizePass
from executorch.backends.cadence.aot.graph_builder import GraphBuilder

from executorch.backends.cadence.aot.pass_utils import count_node
from executorch.backends.cadence.aot.remove_ops import (
    RemoveAliasCopyOpPass,
    RemoveBranchedQuantDequant,
    RemoveCatFromSliceCopyPass,
    RemoveCloneOpsTransformImported,
    RemoveContiguousOpPass,
    RemoveDetachCopyPass,
    RemoveNopAddOpPass,
    RemoveNopExpandOpPass,
    RemoveNopLinalgVectorNormOpPass,
    RemoveNopMulOpPass,
    RemoveNopSelectOpPass,
    RemoveNopSliceOrViewOpPass,
    RemovePermutesAroundElementwiseOps,
    RemoveSqueezeViewBeforeElementwiseOps,
    RemoveToOpsPass,
    RemoveZeroSizedCatArgsPass,
    RemoveZeroSizedConstantPadNd,
)
from executorch.backends.cadence.aot.typing_stubs import expand
from executorch.exir.dialects._ops import ops as exir_ops
from pyre_extensions import none_throws

from torch.fx.passes.infra.pass_base import PassResult


class TestRemoveOpsPasses(unittest.TestCase):

    @expand(
        [
            [(1, 2, 3)],
        ]
    )
    @torch.no_grad()
    def test_remove_to_ops(self, shape: Tuple[int]) -> None:
        builder = GraphBuilder()
        x = builder.placeholder("x", torch.randn(*shape, dtype=torch.float32))
        x = builder.call_operator(
            op=exir_ops.edge.aten.to.dtype,
            args=(x, torch.float32),
        )
        builder.output([x])
        original = builder.get_graph_module()
        graph_after_passes = cast(PassResult, RemoveToOpsPass()(original)).graph_module

        self.assertEqual(
            count_node(graph_after_passes, exir_ops.edge.aten.to.dtype),
            0,
        )

        self.assertEqual(
            count_node(graph_after_passes, exir_ops.edge.aten.to.dtype_layout),
            0,
        )

    @expand(
        [
            [(7, 6, 5)],
            [(7, 6)],
            [(7,)],
        ]
    )
    @torch.no_grad()
    def test_remove_nop_add_op_pass(self, shape: Tuple[int]) -> None:
        builder = GraphBuilder()
        x = builder.placeholder("x", torch.randn(*shape, dtype=torch.float32))
        zeros = builder.call_operator(
            op=exir_ops.edge.aten.full.default, args=(shape, 0)
        )
        left_add = builder.call_operator(
            op=exir_ops.edge.aten.add.Tensor,
            args=(zeros, x),
        )
        right_add = builder.call_operator(
            op=exir_ops.edge.aten.add.Tensor,
            args=(left_add, zeros),
        )
        builder.output([right_add])
        original = builder.get_graph_module()
        graph_after_passes = cast(
            PassResult, RemoveNopAddOpPass()(original)
        ).graph_module
        self.assertEqual(
            count_node(graph_after_passes, exir_ops.edge.aten.add.Tensor),
            0,
        )

    @expand(
        [
            [(7, 6, 5)],
            [(7, 6)],
            [(7,)],
        ]
    )
    @torch.no_grad()
    def test_remove_nop_mul_op_pass(self, shape: Tuple[int]) -> None:
        builder = GraphBuilder()
        x = builder.placeholder("x", torch.randn(*shape, dtype=torch.float32))
        zeros = builder.call_operator(
            op=exir_ops.edge.aten.full.default, args=(shape, 0)
        )
        left_mul = builder.call_operator(
            op=exir_ops.edge.aten.mul.Tensor,
            args=(zeros, x),
        )
        right_mul = builder.call_operator(
            op=exir_ops.edge.aten.mul.Tensor,
            args=(left_mul, zeros),
        )
        builder.output([right_mul])
        original = builder.get_graph_module()
        graph_after_passes = cast(
            PassResult, RemoveNopMulOpPass()(original)
        ).graph_module
        self.assertEqual(
            count_node(graph_after_passes, exir_ops.edge.aten.mul.Tensor),
            0,
        )

    @expand(
        [
            [(1, 2, 3)],
        ]
    )
    @torch.no_grad()
    def test_remove_alias_copy(self, shape: Tuple[int]) -> None:
        builder = GraphBuilder()
        x = builder.placeholder("x", torch.randn(*shape, dtype=torch.float32))
        alias = builder.call_operator(
            op=exir_ops.edge.aten.alias_copy.default, args=(x,)
        )
        builder.output([alias])
        original = builder.get_graph_module()
        graph_after_passes = cast(
            PassResult, RemoveAliasCopyOpPass()(original)
        ).graph_module
        self.assertEqual(
            count_node(graph_after_passes, exir_ops.edge.aten.alias_copy.default),
            0,
        )

    @expand(
        [
            [(1, 2, 3)],
        ]
    )
    @torch.no_grad()
    def test_remove_detach_copy(self, shape: Tuple[int]) -> None:
        builder = GraphBuilder()
        x = builder.placeholder("x", torch.randn(*shape, dtype=torch.float32))
        detach = builder.call_operator(
            op=exir_ops.edge.aten.detach_copy.default, args=(x,)
        )
        builder.output([detach])
        original = builder.get_graph_module()
        graph_after_passes = cast(
            PassResult, RemoveDetachCopyPass()(original)
        ).graph_module
        self.assertEqual(
            count_node(graph_after_passes, exir_ops.edge.aten.detach_copy.default),
            0,
        )

    @expand(
        [
            [(1, 2, 3), (0, 0)],
        ]
    )
    @torch.no_grad()
    def test_remove_zero_sized_constant_pad_nd(
        self, shape: Tuple[int], padding: Tuple[int]
    ) -> None:
        builder = GraphBuilder()
        x = builder.placeholder("x", torch.randn(*shape, dtype=torch.float32))
        pad = builder.call_operator(
            op=exir_ops.edge.aten.constant_pad_nd.default, args=(x, padding)
        )
        builder.output([pad])
        original = builder.get_graph_module()
        pass_result = cast(PassResult, RemoveZeroSizedConstantPadNd()(original))
        graph_after_passes = pass_result.graph_module
        self.assertEqual(
            count_node(graph_after_passes, exir_ops.edge.aten.constant_pad_nd.default),
            0,
        )
        self.assertTrue(pass_result.modified)

    def test_remove_expand(self) -> None:
        builder = GraphBuilder()
        x = builder.placeholder("x", torch.randn([2, 3, 5], dtype=torch.float32))
        expand = builder.call_operator(
            op=exir_ops.edge.aten.expand_copy.default, args=(x, [2, 3, 5])
        )
        builder.output([expand])
        original = builder.get_graph_module()
        graph_after_passes = cast(
            PassResult, RemoveNopExpandOpPass()(original)
        ).graph_module
        self.assertEqual(
            count_node(graph_after_passes, exir_ops.edge.aten.expand_copy.default), 0
        )

    def test_remove_zero_arg_cat(self) -> None:
        builder = GraphBuilder()
        x = builder.placeholder("x", torch.randn([1, 0, 3, 5], dtype=torch.float32))
        y = builder.placeholder("y", torch.randn([2, 0, 3, 5], dtype=torch.float32))
        concat = builder.call_operator(
            op=exir_ops.edge.aten.cat.default, args=([x, y], 0)
        )
        builder.output([concat])
        original = builder.get_graph_module()
        pass_result = cast(PassResult, RemoveZeroSizedCatArgsPass()(original))
        graph_after_passes = pass_result.graph_module
        self.assertEqual(
            count_node(graph_after_passes, exir_ops.edge.aten.cat.default), 0
        )
        self.assertTrue(pass_result.modified)

    def test_remove_clone(self) -> None:
        builder = GraphBuilder()
        x = builder.placeholder("x", torch.randn([3, 5], dtype=torch.float32))
        clone = builder.call_operator(op=exir_ops.edge.aten.clone.default, args=(x,))
        builder.output([clone])
        original = builder.get_graph_module()
        p = RemoveCloneOpsTransformImported()
        graph_after_passes = cast(PassResult, p(original)).graph_module
        self.assertEqual(
            count_node(graph_after_passes, torch.ops.aten.clone.default), 0
        )

    def test_remove_contiguous(self) -> None:
        builder = GraphBuilder()
        x = builder.placeholder("x", torch.randn([3, 5], dtype=torch.float32))
        contiguous = builder.call_operator(
            op=exir_ops.edge.aten.contiguous.default, args=(x,)
        )
        builder.output([contiguous])
        original = builder.get_graph_module()
        p = RemoveContiguousOpPass()
        graph_after_passes = cast(PassResult, p(original)).graph_module
        self.assertEqual(
            count_node(graph_after_passes, torch.ops.aten.contiguous.default), 0
        )

    @expand(
        [
            [(3, 5), [3, 5]],
            [(1,), [-1]],
        ]
    )
    @torch.no_grad()
    def test_remove_nop_view(self, shape: Tuple[int], new_shape: List[int]) -> None:
        builder = GraphBuilder()
        x = builder.placeholder("x", torch.randn(*shape, dtype=torch.float32))
        view = builder.call_operator(
            op=exir_ops.edge.aten.view_copy.default, args=(x, new_shape)
        )
        builder.output([view])
        original = builder.get_graph_module()
        graph_after_passes = cast(
            PassResult, RemoveNopSliceOrViewOpPass()(original)
        ).graph_module
        self.assertEqual(
            count_node(graph_after_passes, exir_ops.edge.aten.view_copy.default), 0
        )

    def test_remove_nop_slice(self) -> None:
        builder = GraphBuilder()
        x = builder.placeholder("x", torch.randn(3, 5, dtype=torch.float32))
        slice_ = builder.call_operator(
            op=exir_ops.edge.aten.slice_copy.Tensor,
            args=(
                x,
                0,  # dim
                0,  # start
                3,  # end
            ),
        )
        builder.output([slice_])
        original = builder.get_graph_module()
        graph_after_passes = cast(
            PassResult, RemoveNopSliceOrViewOpPass()(original)
        ).graph_module
        self.assertEqual(
            count_node(graph_after_passes, exir_ops.edge.aten.slice_copy.Tensor), 0
        )

    def test_remove_nop_slice_or_view_not_modified(self) -> None:
        builder = GraphBuilder()
        x = builder.placeholder("x", torch.randn(3, 5, dtype=torch.float32))
        abs_x = builder.call_operator(
            op=exir_ops.edge.aten.abs.default,
            args=(x,),
        )
        builder.output([abs_x])
        original = builder.get_graph_module()
        pass_result = cast(PassResult, RemoveNopSliceOrViewOpPass()(original))
        self.assertFalse(pass_result.modified)
        graph_after_passes = pass_result.graph_module
        self.assertEqual(
            count_node(graph_after_passes, exir_ops.edge.aten.abs.default), 1
        )

    def test_remove_nop_select_before_view(self) -> None:
        builder = GraphBuilder()
        x = builder.placeholder("x", torch.randn(1, 5, 6, dtype=torch.float32))
        select = builder.call_operator(
            op=exir_ops.edge.aten.select_copy.int,
            args=(
                x,
                0,  # dim
                0,  # index
            ),
        )
        view = builder.call_operator(
            op=exir_ops.edge.aten.view_copy.default,
            args=(select, [1, 5, 6]),  # new shape
        )
        builder.output([view])
        original = builder.get_graph_module()
        graph_after_passes = cast(
            PassResult, RemoveNopSelectOpPass()(original)
        ).graph_module
        self.assertEqual(
            count_node(graph_after_passes, exir_ops.edge.aten.select_copy.int), 0
        )

    def test_remove_nop_select_before_add(self) -> None:
        builder = GraphBuilder()
        x = builder.placeholder("x", torch.randn(1, 5, 6, dtype=torch.float32))
        y = builder.placeholder("y", torch.randn(1, 5, 6, dtype=torch.float32))
        select = builder.call_operator(
            op=exir_ops.edge.aten.select_copy.int,
            args=(
                x,
                0,  # dim
                0,  # index
            ),
        )
        add = builder.call_operator(op=exir_ops.edge.aten.add.Tensor, args=(select, y))
        builder.output([add])
        original = builder.get_graph_module()
        graph_after_passes = cast(
            PassResult, RemoveNopSelectOpPass()(original)
        ).graph_module
        self.assertEqual(
            count_node(graph_after_passes, exir_ops.edge.aten.select_copy.int), 0
        )

    def test_remove_nop_select_before_mul(self) -> None:
        builder = GraphBuilder()
        x = builder.placeholder("x", torch.randn(1, 5, 6, dtype=torch.float32))
        y = builder.placeholder("y", torch.randn(1, 5, 6, dtype=torch.float32))
        select = builder.call_operator(
            op=exir_ops.edge.aten.select_copy.int,
            args=(
                x,
                0,  # dim
                0,  # index
            ),
        )
        mul = builder.call_operator(op=exir_ops.edge.aten.mul.Tensor, args=(select, y))
        builder.output([mul])
        original = builder.get_graph_module()
        graph_after_passes = cast(
            PassResult, RemoveNopSelectOpPass()(original)
        ).graph_module
        self.assertEqual(
            count_node(graph_after_passes, exir_ops.edge.aten.select_copy.int), 0
        )

    def test_remove_nop_select_before_div(self) -> None:
        builder = GraphBuilder()
        x = builder.placeholder("x", torch.randn(1, 5, 6, dtype=torch.float32))
        y = builder.placeholder("y", torch.randn(1, 5, 6, dtype=torch.float32))
        select = builder.call_operator(
            op=exir_ops.edge.aten.select_copy.int,
            args=(
                x,
                0,  # dim
                0,  # index
            ),
        )
        div = builder.call_operator(op=exir_ops.edge.aten.div.Tensor, args=(select, y))
        builder.output([div])
        original = builder.get_graph_module()
        graph_after_passes = cast(
            PassResult, RemoveNopSelectOpPass()(original)
        ).graph_module
        self.assertEqual(
            count_node(graph_after_passes, exir_ops.edge.aten.select_copy.int), 0
        )

    def test_remove_nop_quant_dequant(self) -> None:
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

    def test_remove_nop_aten_linalg_vector_norm(self) -> None:
        builder = GraphBuilder()
        x = builder.placeholder("x", torch.randn(1, 1, 128, dtype=torch.float32))
        linalg_vector_norm = builder.call_operator(
            op=exir_ops.edge.aten.linalg_vector_norm.default, args=(x, 2, [0, 1], True)
        )
        builder.output([linalg_vector_norm])
        original = builder.get_graph_module()
        graph_after_passes = none_throws(
            RemoveNopLinalgVectorNormOpPass()(original)
        ).graph_module
        self.assertEqual(
            count_node(
                graph_after_passes, exir_ops.edge.aten.linalg_vector_norm.default
            ),
            0,
        )

    def test_remove_permutes_around_elemwise_ops_add(self) -> None:
        builder = GraphBuilder()
        x = builder.placeholder("x", torch.randn(1, 8, 4, 4, dtype=torch.float32))
        permute = builder.call_operator(
            op=exir_ops.edge.aten.permute_copy.default, args=(x, [0, 3, 1, 2])
        )
        add = builder.call_operator(
            op=exir_ops.edge.aten.add.Tensor, args=(permute, permute)
        )
        permute = builder.call_operator(
            op=exir_ops.edge.aten.permute_copy.default, args=(add, [0, 2, 3, 1])
        )
        builder.output([permute])
        original = builder.get_graph_module()
        p = RemovePermutesAroundElementwiseOps()
        graph_after_passes = cast(PassResult, p(original)).graph_module
        self.assertEqual(
            count_node(graph_after_passes, exir_ops.edge.aten.permute_copy.default), 0
        )

    def test_keep_permutes_around_elemwise_ops_add(self) -> None:
        builder = GraphBuilder()
        x = builder.placeholder("x", torch.randn(1, 8, 4, 4, dtype=torch.float32))
        permute = builder.call_operator(
            op=exir_ops.edge.aten.permute_copy.default, args=(x, [2, 1, 0, 3])
        )
        add = builder.call_operator(
            op=exir_ops.edge.aten.add.Tensor, args=(permute, permute)
        )
        permute = builder.call_operator(
            op=exir_ops.edge.aten.permute_copy.default, args=(add, [0, 1, 3, 2])
        )
        builder.output([permute])
        original = builder.get_graph_module()
        p = RemovePermutesAroundElementwiseOps()
        graph_after_passes = cast(PassResult, p(original)).graph_module
        # Ensure no permutes were removed, since the dimensions don't fit the expected pattern
        self.assertEqual(
            count_node(graph_after_passes, exir_ops.edge.aten.permute_copy.default), 2
        )

    def test_remove_permutes_around_elemwise_ops_add_mean(self) -> None:
        builder = GraphBuilder()
        x = builder.placeholder("x", torch.randn(1, 8, 4, 4, dtype=torch.float32))
        y = builder.placeholder("y", torch.randn(1, 8, 4, 4, dtype=torch.float32))
        permute_x = builder.call_operator(
            op=exir_ops.edge.aten.permute_copy.default, args=(x, [0, 3, 1, 2])
        )
        permute_y = builder.call_operator(
            op=exir_ops.edge.aten.permute_copy.default, args=(y, [0, 3, 1, 2])
        )
        add = builder.call_operator(
            op=exir_ops.edge.aten.add.Tensor, args=(permute_x, permute_y)
        )
        mean = builder.call_operator(
            op=exir_ops.edge.aten.mean.dim, args=(add, [3, 1], True)
        )
        permute = builder.call_operator(
            op=exir_ops.edge.aten.permute_copy.default, args=(mean, [0, 2, 3, 1])
        )
        builder.output([permute])
        original = builder.get_graph_module()
        p = RemovePermutesAroundElementwiseOps()
        graph_after_passes = cast(PassResult, p(original)).graph_module
        self.assertEqual(
            count_node(graph_after_passes, exir_ops.edge.aten.permute_copy.default), 0
        )
        mean_op = [
            n
            for n in graph_after_passes.graph.nodes
            if n.target == exir_ops.edge.aten.mean.dim
        ][0]
        self.assertEqual(mean_op.args[1], [2, 3])

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
        original = builder.get_graph_module()

        p = RemovePermutesAroundElementwiseOps()
        graph_after_passes = cast(PassResult, p(original)).graph_module

        # No permutes should remain.
        self.assertEqual(
            count_node(graph_after_passes, exir_ops.edge.aten.permute_copy.default), 0
        )

        # Verify that slice dimension was updated correctly.
        slices = graph_after_passes.graph.find_nodes(
            op="call_function", target=exir_ops.edge.aten.slice_copy.Tensor
        )
        self.assertEqual(len(slices), 1)
        self.assertEqual(slices[0].args[1], 2)

    def test_remove_squeeze_view_before_elemwise_ops(self) -> None:
        builder = GraphBuilder()
        x = builder.placeholder("x", torch.randn(8, 1, 4, 4))
        squeeze = builder.call_operator(
            op=exir_ops.edge.aten.view_copy.default,
            args=(x, [8, 4, 4]),
        )
        quantize = builder.call_operator(
            op=exir_ops.edge.quantized_decomposed.quantize_per_tensor.default,
            args=(squeeze, 0.12, -4, -128, 127, torch.int8),
        )
        slice_copy = builder.call_operator(
            op=exir_ops.edge.aten.slice_copy.Tensor,
            args=(quantize, 1, 0, 2, 1),
        )
        unsqueeze = builder.call_operator(
            op=exir_ops.edge.aten.view_copy.default,
            args=(slice_copy, [8, 1, 2, 4]),
        )
        builder.output([unsqueeze])
        model = builder.get_graph_module()
        original = deepcopy(model)

        p = RemoveSqueezeViewBeforeElementwiseOps()
        pass_result = cast(PassResult, p(model))
        self.assertTrue(pass_result.modified)
        transformed = pass_result.graph_module

        # First view should be eliminated and second view should be trivial.
        views = transformed.graph.find_nodes(
            op="call_function", target=exir_ops.edge.aten.view_copy.default
        )
        self.assertEqual(len(views), 1)
        self.assertEqual(views[0].args[0].meta["val"].shape, views[0].meta["val"].shape)

        # Verify that slice dimension was updated correctly.
        slices = transformed.graph.find_nodes(
            op="call_function", target=exir_ops.edge.aten.slice_copy.Tensor
        )
        self.assertEqual(len(slices), 1)
        self.assertEqual(slices[0].args[1], 2)

        # Verify the output of the model is the same as the original.
        sample_input = torch.randn(8, 1, 4, 4)
        self.assertTrue(
            torch.allclose(
                original(sample_input)[0],
                transformed(sample_input)[0],
            )
        )

    def test_remove_squeeze_view_before_elemwise_ops_multiple_squeeze(self) -> None:
        builder = GraphBuilder()
        x = builder.placeholder("x", torch.randn(8, 1, 1, 4, 1, 4))
        squeeze = builder.call_operator(
            op=exir_ops.edge.aten.view_copy.default,
            args=(x, [8, 4, 4]),
        )
        quantize = builder.call_operator(
            op=exir_ops.edge.quantized_decomposed.quantize_per_tensor.default,
            args=(squeeze, 0.12, -4, -128, 127, torch.int8),
        )
        slice_copy = builder.call_operator(
            op=exir_ops.edge.aten.slice_copy.Tensor,
            args=(quantize, 1, 0, 2, 1),
        )
        view_copy = builder.call_operator(
            op=exir_ops.edge.aten.view_copy.default,
            args=(slice_copy, [16, 4]),
        )
        builder.output([view_copy])
        model = builder.get_graph_module()
        original = deepcopy(model)

        p = RemoveSqueezeViewBeforeElementwiseOps()
        transformed = cast(PassResult, p(model)).graph_module

        # First view should be eliminated.
        self.assertEqual(
            count_node(transformed, exir_ops.edge.aten.view_copy.default), 1
        )

        # Verify that slice dimension was updated correctly.
        slices = transformed.graph.find_nodes(
            op="call_function", target=exir_ops.edge.aten.slice_copy.Tensor
        )
        self.assertEqual(len(slices), 1)
        self.assertEqual(slices[0].args[1], 3)

        # Verify the output of the model is the same as the original.
        sample_input = torch.randn(8, 1, 1, 4, 1, 4)
        self.assertTrue(
            torch.allclose(
                original(sample_input)[0],
                transformed(sample_input)[0],
            )
        )

    def test_remove_permutes_around_elemwise_ops_mul(self) -> None:
        builder = GraphBuilder()
        x = builder.placeholder("x", torch.randn(2, 4, 4, 8))
        y = builder.placeholder("y", torch.randn(2, 4, 4, 8))
        sliced_x = builder.call_operator(
            op=exir_ops.edge.aten.slice_copy.Tensor,
            args=(x, 0, 0, 1),
        )
        permuted_x = builder.call_operator(
            op=exir_ops.edge.aten.permute_copy.default,
            args=(sliced_x, [0, 3, 1, 2]),
        )
        permuted_y = builder.call_operator(
            op=exir_ops.edge.aten.permute_copy.default,
            args=(y, [0, 3, 1, 2]),
        )
        dequantized_x = builder.call_operator(
            op=exir_ops.edge.cadence.dequantize_per_tensor.default,
            args=(permuted_x, 1.5, 0, 0, 255, torch.uint8),
        )
        z = builder.call_operator(
            op=exir_ops.edge.aten.mul.Tensor, args=(dequantized_x, permuted_y)
        )
        quantized_z = builder.call_operator(
            op=exir_ops.edge.cadence.quantize_per_tensor.default,
            args=(z, 2.5, 0, 0, 255, torch.uint8),
        )
        permuted_z = builder.call_operator(
            op=exir_ops.edge.aten.permute_copy.default,
            args=(quantized_z, [0, 2, 3, 1]),
        )
        output = builder.call_operator(
            op=exir_ops.edge.aten.unsqueeze_copy.default,
            args=(permuted_z, 0),
        )
        builder.output([output])
        original = builder.get_graph_module()
        p = RemovePermutesAroundElementwiseOps()
        graph_after_passes = cast(PassResult, p(original)).graph_module
        self.assertEqual(
            count_node(graph_after_passes, exir_ops.edge.aten.permute_copy.default), 0
        )

    def test_remove_permutes_around_elemwise_ops_double_permutes(self) -> None:
        builder = GraphBuilder()
        x = builder.placeholder("x", torch.randn(2, 4, 4, 8))
        y = builder.placeholder("y", torch.randn(1, 8, 4, 4))
        sliced_x = builder.call_operator(
            op=exir_ops.edge.aten.slice_copy.Tensor,
            args=(x, 0, 0, 1),
        )
        permuted_x = builder.call_operator(
            op=exir_ops.edge.aten.permute_copy.default,
            args=(sliced_x, [0, 3, 1, 2]),
        )
        permuted_x = builder.call_operator(
            op=exir_ops.edge.aten.permute_copy.default,
            args=(permuted_x, [0, 3, 1, 2]),
        )
        dequantized_x = builder.call_operator(
            op=exir_ops.edge.cadence.dequantize_per_tensor.default,
            args=(permuted_x, 1.5, 0, 0, 255, torch.uint8),
        )
        permuted_y = builder.call_operator(
            op=exir_ops.edge.aten.permute_copy.default,
            args=(y, [0, 3, 1, 2]),
        )
        dequantized_y = builder.call_operator(
            op=exir_ops.edge.cadence.dequantize_per_tensor.default,
            args=(permuted_y, 1.5, 0, 0, 255, torch.uint8),
        )
        z = builder.call_operator(
            op=exir_ops.edge.aten.cat.default, args=((dequantized_x, dequantized_y), 1)
        )
        quantized_z = builder.call_operator(
            op=exir_ops.edge.cadence.quantize_per_tensor.default,
            args=(z, 2.5, 0, 0, 255, torch.uint8),
        )
        permuted_z = builder.call_operator(
            op=exir_ops.edge.aten.permute_copy.default,
            args=(quantized_z, [0, 2, 3, 1]),
        )
        permuted_z = builder.call_operator(
            op=exir_ops.edge.aten.permute_copy.default,
            args=(permuted_z, [0, 2, 3, 1]),
        )
        output = builder.call_operator(
            op=exir_ops.edge.aten.unsqueeze_copy.default,
            args=(permuted_z, 0),
        )
        builder.output([output])
        original = builder.get_graph_module()
        p = RemovePermutesAroundElementwiseOps()
        graph_after_passes = cast(PassResult, p(original)).graph_module
        # Expect 2 permutes to remain, one on input x and one on output z
        self.assertEqual(
            count_node(graph_after_passes, exir_ops.edge.aten.permute_copy.default), 2
        )
        # verify that cat was updated correctly
        cat = [
            n
            for n in graph_after_passes.graph.nodes
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

    def test_remove_dequant_on_branch(self) -> None:
        builder = GraphBuilder()
        x = builder.placeholder("x", torch.randn(1, 8, 4, 6))
        x = builder.call_operator(op=exir_ops.edge.aten.abs.default, args=(x,))
        x0 = builder.call_operator(
            op=exir_ops.edge.quantized_decomposed.quantize_per_tensor.default,
            args=(x, 1.2, 3, 0, 127, torch.int8),
        )
        x1_output = builder.call_operator(op=exir_ops.edge.aten.abs.default, args=(x0,))
        y0 = builder.call_operator(
            op=exir_ops.edge.quantized_decomposed.dequantize_per_tensor.default,
            args=(x0, 1.2, 3, 0, 127, torch.int8),
        )
        y1_output = builder.call_operator(
            op=exir_ops.edge.aten.view.default,
            args=(y0, [-1]),
        )
        builder.output([x1_output, y1_output])
        original = builder.get_graph_module()
        pass_result = cast(PassResult, RemoveBranchedQuantDequant()(original))
        self.assertTrue(pass_result.modified)
        graph_after_passes = pass_result.graph_module
        self.assertEqual(
            count_node(
                graph_after_passes,
                exir_ops.edge.quantized_decomposed.quantize_per_tensor.default,
            ),
            1,
        )
        self.assertEqual(
            count_node(
                graph_after_passes,
                exir_ops.edge.quantized_decomposed.dequantize_per_tensor.default,
            ),
            0,
        )
        self.assertEqual(
            count_node(graph_after_passes, exir_ops.edge.aten.abs.default), 2
        )

    def test_remove_cat_from_slice_copy(self) -> None:
        builder = GraphBuilder()
        x = builder.placeholder("x", torch.randn(2, 4))
        y = builder.placeholder("y", torch.randn(2, 4))
        z = builder.call_operator(op=exir_ops.edge.aten.cat.default, args=((x, y), 0))
        output = builder.call_operator(
            op=exir_ops.edge.aten.slice_copy.Tensor,
            args=(z, 0, 0, 1),
        )
        builder.output([output])
        original = builder.get_graph_module()
        pass_result = cast(PassResult, RemoveCatFromSliceCopyPass()(original))
        self.assertTrue(pass_result.modified)
        graph_after_passes = pass_result.graph_module
        self.assertEqual(
            count_node(graph_after_passes, exir_ops.edge.aten.cat.default), 0
        )

    def test_keep_cat_from_slice_copy(self) -> None:
        builder = GraphBuilder()
        x = builder.placeholder("x", torch.randn(2, 4))
        y = builder.placeholder("y", torch.randn(2, 4))
        z = builder.call_operator(op=exir_ops.edge.aten.cat.default, args=((x, y), 0))
        output = builder.call_operator(
            op=exir_ops.edge.aten.slice_copy.Tensor,
            args=(z, 0, 0, 3),
        )
        builder.output([output])
        original = builder.get_graph_module()
        pass_result = cast(PassResult, RemoveCatFromSliceCopyPass()(original))
        self.assertFalse(pass_result.modified)
        graph_after_passes = pass_result.graph_module
        self.assertEqual(
            count_node(graph_after_passes, exir_ops.edge.aten.cat.default), 1
        )

    def test_remove_cat_from_slice_copy_zero_range(self) -> None:
        builder = GraphBuilder()
        x = builder.placeholder("x", torch.randn(2, 4))
        y = builder.placeholder("y", torch.randn(2, 4))
        z = builder.call_operator(op=exir_ops.edge.aten.cat.default, args=((x, y), 0))
        output = builder.call_operator(
            op=exir_ops.edge.aten.slice_copy.Tensor,
            args=(z, 0, 0, 0),
        )
        builder.output([output])
        original = builder.get_graph_module()
        graph_after_passes = cast(
            PassResult, RemoveCatFromSliceCopyPass()(original)
        ).graph_module
        self.assertEqual(
            count_node(graph_after_passes, exir_ops.edge.aten.cat.default), 0
        )

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
            args=(cat, 1, 5, 7, 1),  # dim start end step
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
