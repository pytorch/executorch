# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.
# pyre-strict

import unittest
from typing import cast

import executorch.backends.cadence.aot.ops_registrations  # noqa
import torch
from executorch.backends.cadence.aot.graph_builder import single_op_builder
from executorch.backends.cadence.aot.pass_utils import count_node
from executorch.backends.cadence.aot.type_dispatch import CompileTimeTypeDispatchPass
from executorch.backends.cadence.aot.typing_stubs import expand
from executorch.exir.dialects._ops import ops as exir_ops
from torch.fx.passes.infra.pass_base import PassResult


class TestTypeDispatchPasses(unittest.TestCase):
    @expand(
        [
            (
                "int8",
                torch.int8,
                exir_ops.edge.cadence.quantized_fully_connected_asym8sxasym8s_asym8s.per_tensor,
            ),
            (
                "uint8",
                torch.uint8,
                exir_ops.edge.cadence.quantized_fully_connected_asym8uxasym8u_asym8u.per_tensor,
            ),
        ]
    )
    def test_dispatch_quantized_fully_connected(
        self,
        _: str,
        dtype: torch.dtype,
        expected_op: torch._ops.OpOverload,
    ) -> None:
        """Test quantized_fully_connected dispatches to correct dtype-specific variant"""
        min_val, max_val = torch.iinfo(dtype).min, torch.iinfo(dtype).max
        x = torch.randint(min_val, max_val, (1, 3), dtype=dtype)
        w = torch.randint(min_val, max_val, (4, 3), dtype=dtype)
        b = torch.randint(-2147483648, 2147483647, (4,), dtype=torch.int32)
        gm = single_op_builder(
            placeholders=(x, w, b),
            op=exir_ops.edge.cadence.quantized_fully_connected.per_tensor,
            args=(x, w, b, 0, 0, 1, 0, 0, None),
        )
        p = CompileTimeTypeDispatchPass()
        gm = cast(PassResult, p(gm)).graph_module
        # Original op should be replaced
        self.assertEqual(
            count_node(gm, exir_ops.edge.cadence.quantized_fully_connected.per_tensor),
            0,
        )
        # Should be replaced with dtype-specific variant
        self.assertEqual(count_node(gm, expected_op), 1)

    @expand(
        [
            (
                "int8",
                torch.int8,
                exir_ops.edge.cadence.quantized_linear_asym8sxasym8s_asym8s.per_tensor,
            ),
            (
                "uint8",
                torch.uint8,
                exir_ops.edge.cadence.quantized_linear_asym8uxasym8u_asym8u.per_tensor,
            ),
        ]
    )
    def test_dispatch_quantized_linear(
        self,
        _: str,
        dtype: torch.dtype,
        expected_op: torch._ops.OpOverload,
    ) -> None:
        """Test quantized_linear dispatches to correct dtype-specific variant"""
        min_val, max_val = torch.iinfo(dtype).min, torch.iinfo(dtype).max
        x = torch.randint(min_val, max_val, (2, 3), dtype=dtype)
        w = torch.randint(min_val, max_val, (4, 3), dtype=dtype)
        b = torch.randint(-2147483648, 2147483647, (4,), dtype=torch.int32)
        gm = single_op_builder(
            placeholders=(x, w, b),
            op=exir_ops.edge.cadence.quantized_linear.per_tensor,
            args=(x, w, b, 0, 0, 1, 0, 0, None),
        )
        p = CompileTimeTypeDispatchPass()
        gm = cast(PassResult, p(gm)).graph_module
        # Original op should be replaced
        self.assertEqual(
            count_node(gm, exir_ops.edge.cadence.quantized_linear.per_tensor),
            0,
        )
        # Should be replaced with dtype-specific variant
        self.assertEqual(count_node(gm, expected_op), 1)

    def test_mixed_types_error(self) -> None:
        """Test mixed int8/uint8 inputs should raise RuntimeError"""
        x = torch.randint(-128, 127, (1, 3), dtype=torch.int8)
        w = torch.randint(0, 255, (4, 3), dtype=torch.uint8)
        b = torch.randint(-2147483648, 2147483647, (4,), dtype=torch.int32)
        gm = single_op_builder(
            placeholders=(x, w, b),
            op=exir_ops.edge.cadence.quantized_fully_connected.per_tensor,
            args=(x, w, b, 0, 0, 1, 0, 0, None),
        )
        p = CompileTimeTypeDispatchPass()
        # Mixed types should raise RuntimeError
        with self.assertRaises(RuntimeError) as context:
            cast(PassResult, p(gm)).graph_module
        self.assertIn("Unsupported input types", str(context.exception))

    @expand(
        [
            (
                "int8",
                torch.int8,
                exir_ops.edge.cadence.quantized_relu_asym8s_asym8s.per_tensor,
            ),
            (
                "uint8",
                torch.uint8,
                exir_ops.edge.cadence.quantized_relu_asym8u_asym8u.per_tensor,
            ),
        ]
    )
    def test_dispatch_quantized_relu(
        self,
        _: str,
        dtype: torch.dtype,
        expected_op: torch._ops.OpOverload,
    ) -> None:
        """Test quantized_relu dispatches to correct dtype-specific variant"""
        min_val, max_val = torch.iinfo(dtype).min, torch.iinfo(dtype).max
        x = torch.randint(min_val, max_val, (2, 3), dtype=dtype)
        gm = single_op_builder(
            placeholders=(x,),
            op=exir_ops.edge.cadence.quantized_relu.per_tensor,
            args=(x, 0, 0, 1, 0),
        )
        p = CompileTimeTypeDispatchPass()
        gm = cast(PassResult, p(gm)).graph_module
        # Original op should be replaced
        self.assertEqual(
            count_node(gm, exir_ops.edge.cadence.quantized_relu.per_tensor),
            0,
        )
        # Should be replaced with dtype-specific variant
        self.assertEqual(count_node(gm, expected_op), 1)

    @expand(
        [
            (
                "int8",
                torch.int8,
                exir_ops.edge.cadence.quantized_matmul_asym8sxasym8s_asym8s.default,
            ),
            (
                "uint8",
                torch.uint8,
                exir_ops.edge.cadence.quantized_matmul_asym8uxasym8u_asym8u.default,
            ),
        ]
    )
    def test_dispatch_quantized_matmul(
        self,
        _: str,
        dtype: torch.dtype,
        expected_op: torch._ops.OpOverload,
    ) -> None:
        """Test quantized_matmul dispatches to correct dtype-specific variant"""
        min_val, max_val = torch.iinfo(dtype).min, torch.iinfo(dtype).max
        x = torch.randint(min_val, max_val, (2, 3), dtype=dtype)
        y = torch.randint(min_val, max_val, (3, 4), dtype=dtype)
        bias = torch.randint(-2147483648, 2147483647, (4,), dtype=torch.int32)
        gm = single_op_builder(
            placeholders=(x, y, bias),
            op=exir_ops.edge.cadence.quantized_matmul.default,
            args=(x, 0, y, 0, bias, 1, 0, 0, False),
        )
        p = CompileTimeTypeDispatchPass()
        gm = cast(PassResult, p(gm)).graph_module
        # Original op should be replaced
        self.assertEqual(
            count_node(gm, exir_ops.edge.cadence.quantized_matmul.default),
            0,
        )
        # Should be replaced with dtype-specific variant
        self.assertEqual(count_node(gm, expected_op), 1)

    @expand(
        [
            (
                "int8_nchw",
                torch.int8,
                (1, 3, 8, 8),  # x_shape
                exir_ops.edge.cadence.quantized_conv2d_nchw.per_tensor,
                exir_ops.edge.cadence.quantized_conv2d_nchw_asym8sxsym8s_asym8s.per_tensor,
            ),
            (
                "uint8_nchw",
                torch.uint8,
                (1, 3, 8, 8),  # x_shape
                exir_ops.edge.cadence.quantized_conv2d_nchw.per_tensor,
                exir_ops.edge.cadence.quantized_conv2d_nchw_asym8uxsym8u_asym8u.per_tensor,
            ),
            (
                "int8_nhwc",
                torch.int8,
                (1, 8, 8, 3),  # x_shape
                exir_ops.edge.cadence.quantized_conv2d_nhwc.per_tensor,
                exir_ops.edge.cadence.quantized_conv2d_nhwc_asym8sxsym8s_asym8s.per_tensor,
            ),
            (
                "uint8_nhwc",
                torch.uint8,
                (1, 8, 8, 3),  # x_shape
                exir_ops.edge.cadence.quantized_conv2d_nhwc.per_tensor,
                exir_ops.edge.cadence.quantized_conv2d_nhwc_asym8uxsym8u_asym8u.per_tensor,
            ),
        ]
    )
    def test_dispatch_quantized_conv_2d(
        self,
        _: str,
        dtype: torch.dtype,
        x_shape: tuple[int, ...],
        original_op: torch._ops.OpOverload,
        expected_op: torch._ops.OpOverload,
    ) -> None:
        """Test quantized_conv_2d (nchw/nhwc) dispatches to correct dtype-specific variant"""
        min_val, max_val = torch.iinfo(dtype).min, torch.iinfo(dtype).max
        x = torch.randint(min_val, max_val, x_shape, dtype=dtype)
        w = torch.randint(min_val, max_val, (16, 3, 3, 3), dtype=dtype)
        b = torch.randint(-2147483648, 2147483647, (16,), dtype=torch.int32)
        gm = single_op_builder(
            placeholders=(x, w, b),
            op=original_op,
            args=(x, w, b, [1, 1], [0, 0], [1, 1], 1, 0, 0, 1.0, 1.0, 0, 1, 1),
        )
        p = CompileTimeTypeDispatchPass()
        gm = cast(PassResult, p(gm)).graph_module
        # Original op should be replaced
        self.assertEqual(count_node(gm, original_op), 0)
        # Should be replaced with dtype-specific variant
        self.assertEqual(count_node(gm, expected_op), 1)

    @expand(
        [
            (
                "int8_nchw_dilated",
                torch.int8,
                (1, 3, 8, 8),  # x_shape
                exir_ops.edge.cadence.quantized_conv2d_nchw.per_tensor,
                exir_ops.edge.cadence.quantized_conv2d_nchw_dilated_asym8sxsym8s_asym8s.per_tensor,
            ),
            (
                "uint8_nchw_dilated",
                torch.uint8,
                (1, 3, 8, 8),  # x_shape
                exir_ops.edge.cadence.quantized_conv2d_nchw.per_tensor,
                exir_ops.edge.cadence.quantized_conv2d_nchw_dilated_asym8uxsym8u_asym8u.per_tensor,
            ),
            (
                "int8_nhwc_dilated",
                torch.int8,
                (1, 8, 8, 3),  # x_shape
                exir_ops.edge.cadence.quantized_conv2d_nhwc.per_tensor,
                exir_ops.edge.cadence.quantized_conv2d_nhwc_dilated_asym8sxsym8s_asym8s.per_tensor,
            ),
            (
                "uint8_nhwc_dilated",
                torch.uint8,
                (1, 8, 8, 3),  # x_shape
                exir_ops.edge.cadence.quantized_conv2d_nhwc.per_tensor,
                exir_ops.edge.cadence.quantized_conv2d_nhwc_dilated_asym8uxsym8u_asym8u.per_tensor,
            ),
        ]
    )
    def test_dispatch_quantized_conv_2d_dilated(
        self,
        _: str,
        dtype: torch.dtype,
        x_shape: tuple[int, ...],
        original_op: torch._ops.OpOverload,
        expected_op: torch._ops.OpOverload,
    ) -> None:
        """Test quantized_conv_2d with dilation dispatches to correct dtype-specific variant"""
        min_val, max_val = torch.iinfo(dtype).min, torch.iinfo(dtype).max
        x = torch.randint(min_val, max_val, x_shape, dtype=dtype)
        w = torch.randint(min_val, max_val, (16, 3, 3, 3), dtype=dtype)
        b = torch.randint(-2147483648, 2147483647, (16,), dtype=torch.int32)
        gm = single_op_builder(
            placeholders=(x, w, b),
            op=original_op,
            args=(x, w, b, [1, 1], [0, 0], [2, 2], 1, 0, 0, 1.0, 1.0, 0, 1, 1),
        )
        p = CompileTimeTypeDispatchPass()
        gm = cast(PassResult, p(gm)).graph_module
        # Original op should be replaced
        self.assertEqual(count_node(gm, original_op), 0)
        # Should be replaced with dtype-specific variant
        self.assertEqual(count_node(gm, expected_op), 1)

    @expand(
        [
            (
                "int8_nchw_1d",
                torch.int8,
                (1, 3, 8),  # x_shape
                exir_ops.edge.cadence.quantized_conv2d_nchw.per_tensor,
                exir_ops.edge.cadence.quantized_conv1d_ncl_asym8sxsym8s_asym8s.per_tensor,
            ),
            (
                "uint8_nchw_1d",
                torch.uint8,
                (1, 3, 8),  # x_shape
                exir_ops.edge.cadence.quantized_conv2d_nchw.per_tensor,
                exir_ops.edge.cadence.quantized_conv1d_ncl_asym8uxsym8u_asym8u.per_tensor,
            ),
            (
                "int8_nhwc_1d",
                torch.int8,
                (1, 8, 3),  # x_shape
                exir_ops.edge.cadence.quantized_conv2d_nhwc.per_tensor,
                exir_ops.edge.cadence.quantized_conv1d_nlc_asym8sxsym8s_asym8s.per_tensor,
            ),
            (
                "uint8_nhwc_1d",
                torch.uint8,
                (1, 8, 3),  # x_shape
                exir_ops.edge.cadence.quantized_conv2d_nhwc.per_tensor,
                exir_ops.edge.cadence.quantized_conv1d_nlc_asym8uxsym8u_asym8u.per_tensor,
            ),
        ]
    )
    def test_dispatch_quantized_conv_1d(
        self,
        _: str,
        dtype: torch.dtype,
        x_shape: tuple[int, ...],
        original_op: torch._ops.OpOverload,
        expected_op: torch._ops.OpOverload,
    ) -> None:
        """Test quantized_conv_1d (nchw/nhwc) dispatches to correct dtype-specific variant"""
        min_val, max_val = torch.iinfo(dtype).min, torch.iinfo(dtype).max
        x = torch.randint(min_val, max_val, x_shape, dtype=dtype)
        w = torch.randint(min_val, max_val, (16, 3, 3), dtype=dtype)
        b = torch.randint(-2147483648, 2147483647, (16,), dtype=torch.int32)
        gm = single_op_builder(
            placeholders=(x, w, b),
            op=original_op,
            args=(x, w, b, [1, 1], [0, 0], [1, 1], 1, 0, 0, 1.0, 1.0, 0, 1, 1),
        )
        p = CompileTimeTypeDispatchPass()
        gm = cast(PassResult, p(gm)).graph_module
        # Original op should be replaced
        self.assertEqual(count_node(gm, original_op), 0)
        # Should be replaced with dtype-specific variant
        self.assertEqual(count_node(gm, expected_op), 1)

    @expand(
        [
            (
                "int8",
                torch.int8,
                exir_ops.edge.cadence.quantized_add_asym8sxasym8s_asym8s.per_tensor,
            ),
            (
                "uint8",
                torch.uint8,
                exir_ops.edge.cadence.quantized_add_asym8uxasym8u_asym8u.per_tensor,
            ),
        ]
    )
    def test_dispatch_quantized_add(
        self,
        _: str,
        dtype: torch.dtype,
        expected_op: torch._ops.OpOverload,
    ) -> None:
        """Test quantized_add dispatches to correct dtype-specific variant"""
        min_val, max_val = torch.iinfo(dtype).min, torch.iinfo(dtype).max
        x = torch.randint(min_val, max_val, (2, 3), dtype=dtype)
        y = torch.randint(min_val, max_val, (2, 3), dtype=dtype)
        gm = single_op_builder(
            placeholders=(x, y),
            op=exir_ops.edge.cadence.quantized_add.per_tensor,
            args=(x, 1.0, 0, y, 1.0, 0, 1.0, 0),
        )
        p = CompileTimeTypeDispatchPass()
        gm = cast(PassResult, p(gm)).graph_module
        # Original op should be replaced
        self.assertEqual(
            count_node(gm, exir_ops.edge.cadence.quantized_add.per_tensor),
            0,
        )
        # Should be replaced with dtype-specific variant
        self.assertEqual(count_node(gm, expected_op), 1)

    @expand(
        [
            (
                "int8_nchw_depthwise",
                torch.int8,
                (1, 3, 8, 8),  # x_shape
                (3, 1, 3, 3),  # w_shape (groups=3, input_channels=3)
                exir_ops.edge.cadence.quantized_conv2d_nchw.per_tensor,
                exir_ops.edge.cadence.quantized_conv2d_nchw_depthwise_asym8sxsym8s_asym8s.per_tensor,
            ),
            (
                "uint8_nchw_depthwise",
                torch.uint8,
                (1, 3, 8, 8),  # x_shape
                (3, 1, 3, 3),  # w_shape (groups=3, input_channels=3)
                exir_ops.edge.cadence.quantized_conv2d_nchw.per_tensor,
                exir_ops.edge.cadence.quantized_conv2d_nchw_depthwise_asym8uxsym8u_asym8u.per_tensor,
            ),
            (
                "int8_nhwc_depthwise",
                torch.int8,
                (1, 8, 8, 3),  # x_shape
                (3, 3, 3, 1),  # w_shape (groups=3, input_channels=3)
                exir_ops.edge.cadence.quantized_conv2d_nhwc.per_tensor,
                exir_ops.edge.cadence.quantized_conv2d_nhwc_depthwise_asym8sxsym8s_asym8s.per_tensor,
            ),
            (
                "uint8_nhwc_depthwise",
                torch.uint8,
                (1, 8, 8, 3),  # x_shape
                (3, 3, 3, 1),  # w_shape (groups=3, input_channels=3)
                exir_ops.edge.cadence.quantized_conv2d_nhwc.per_tensor,
                exir_ops.edge.cadence.quantized_conv2d_nhwc_depthwise_asym8uxsym8u_asym8u.per_tensor,
            ),
        ]
    )
    def test_dispatch_quantized_conv_depthwise(
        self,
        _: str,
        dtype: torch.dtype,
        x_shape: tuple[int, ...],
        w_shape: tuple[int, ...],
        original_op: torch._ops.OpOverload,
        expected_op: torch._ops.OpOverload,
    ) -> None:
        """Test quantized_conv depthwise (groups == input_channels) dispatches to correct dtype-specific variant"""
        min_val, max_val = torch.iinfo(dtype).min, torch.iinfo(dtype).max
        x = torch.randint(min_val, max_val, x_shape, dtype=dtype)
        w = torch.randint(min_val, max_val, w_shape, dtype=dtype)
        b = torch.randint(-2147483648, 2147483647, (3,), dtype=torch.int32)
        gm = single_op_builder(
            placeholders=(x, w, b),
            op=original_op,
            args=(
                x,
                w,
                b,
                [1, 1],
                [0, 0],
                [1, 1],
                3,
                0,
                0,
                1.0,
                1.0,
                0,
                1,
                1,
            ),
        )
        p = CompileTimeTypeDispatchPass()
        gm = cast(PassResult, p(gm)).graph_module
        # Original op should be replaced
        self.assertEqual(count_node(gm, original_op), 0)
        # Should be replaced with dtype-specific variant
        self.assertEqual(count_node(gm, expected_op), 1)
