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
from executorch.exir.dialects._ops import ops as exir_ops
from torch.fx.passes.infra.pass_base import PassResult


class TestTypeDispatchPasses(unittest.TestCase):
    def test_int8_dispatch_quantized_fully_connected(self) -> None:
        """Test int8 x int8 inputs should dispatch to asym8sxasym8s_asym8s variant"""
        x = torch.randint(-128, 127, (1, 3), dtype=torch.int8)
        w = torch.randint(-128, 127, (4, 3), dtype=torch.int8)
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
        # Should be replaced with int8 specific variant
        self.assertEqual(
            count_node(
                gm,
                exir_ops.edge.cadence.quantized_fully_connected_asym8sxasym8s_asym8s.per_tensor,
            ),
            1,
        )

    def test_uint8_dispatch_quantized_fully_connected(self) -> None:
        """Test uint8 x uint8 inputs should dispatch to asym8uxasym8u_asym8u variant"""
        x = torch.randint(0, 255, (1, 3), dtype=torch.uint8)
        w = torch.randint(0, 255, (4, 3), dtype=torch.uint8)
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
        # Should be replaced with uint8 specific variant
        self.assertEqual(
            count_node(
                gm,
                exir_ops.edge.cadence.quantized_fully_connected_asym8uxasym8u_asym8u.per_tensor,
            ),
            1,
        )

    def test_int8_dispatch_quantized_linear(self) -> None:
        """Test int8 x int8 inputs should dispatch to asym8sxasym8s_asym8s variant for quantized_linear"""
        x = torch.randint(-128, 127, (2, 3), dtype=torch.int8)
        w = torch.randint(-128, 127, (4, 3), dtype=torch.int8)
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
        # Should be replaced with int8 specific variant
        self.assertEqual(
            count_node(
                gm,
                exir_ops.edge.cadence.quantized_linear_asym8sxasym8s_asym8s.per_tensor,
            ),
            1,
        )

    def test_uint8_quantized_linear_dispatch(self) -> None:
        """Test uint8 x uint8 inputs should dispatch to asym8uxasym8u_asym8u variant for quantized_linear"""
        x = torch.randint(0, 255, (2, 3), dtype=torch.uint8)
        w = torch.randint(0, 255, (4, 3), dtype=torch.uint8)
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
        # Should be replaced with uint8 specific variant
        self.assertEqual(
            count_node(
                gm,
                exir_ops.edge.cadence.quantized_linear_asym8uxasym8u_asym8u.per_tensor,
            ),
            1,
        )

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

    def test_int8_dispatch_quantized_relu(self) -> None:
        """Test int8 input should dispatch to asym8s_asym8s variant for quantized_relu"""
        x = torch.randint(-128, 127, (2, 3), dtype=torch.int8)
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
        # Should be replaced with int8 specific variant
        self.assertEqual(
            count_node(
                gm,
                exir_ops.edge.cadence.quantized_relu_asym8s_asym8s.per_tensor,
            ),
            1,
        )

    def test_uint8_dispatch_quantized_relu(self) -> None:
        """Test uint8 input should dispatch to asym8u_asym8u variant for quantized_relu"""
        x = torch.randint(0, 255, (2, 3), dtype=torch.uint8)
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
        # Should be replaced with uint8 specific variant
        self.assertEqual(
            count_node(
                gm,
                exir_ops.edge.cadence.quantized_relu_asym8u_asym8u.per_tensor,
            ),
            1,
        )

    def test_int8_dispatch_quantized_matmul(self) -> None:
        """Test int8 x int8 inputs should dispatch to asym8sxasym8s_asym8s variant for quantized_matmul"""
        x = torch.randint(-128, 127, (2, 3), dtype=torch.int8)
        y = torch.randint(-128, 127, (3, 4), dtype=torch.int8)
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
        # Should be replaced with int8 specific variant
        self.assertEqual(
            count_node(
                gm,
                exir_ops.edge.cadence.quantized_matmul_asym8sxasym8s_asym8s.default,
            ),
            1,
        )

    def test_uint8_dispatch_quantized_matmul(self) -> None:
        """Test uint8 x uint8 inputs should dispatch to asym8uxasym8u_asym8u variant for quantized_matmul"""
        x = torch.randint(0, 255, (2, 3), dtype=torch.uint8)
        y = torch.randint(0, 255, (3, 4), dtype=torch.uint8)
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
        # Should be replaced with uint8 specific variant
        self.assertEqual(
            count_node(
                gm,
                exir_ops.edge.cadence.quantized_matmul_asym8uxasym8u_asym8u.default,
            ),
            1,
        )

    def test_int8_dispatch_quantized_conv_nchw(self) -> None:
        """Test int8 x int8 inputs should dispatch to asym8sxasym8s_asym8s variant for quantized_conv_nchw"""
        x = torch.randint(-128, 127, (1, 3, 8, 8), dtype=torch.int8)
        w = torch.randint(-128, 127, (16, 3, 3, 3), dtype=torch.int8)
        b = torch.randint(-2147483648, 2147483647, (16,), dtype=torch.int32)
        gm = single_op_builder(
            placeholders=(x, w, b),
            op=exir_ops.edge.cadence.quantized_conv_nchw.per_tensor,
            args=(x, w, b, [1, 1], [0, 0], [1, 1], 1, 0, 0, 1.0, 1.0, 0, 1, 1),
        )
        p = CompileTimeTypeDispatchPass()
        gm = cast(PassResult, p(gm)).graph_module
        # Original op should be replaced
        self.assertEqual(
            count_node(gm, exir_ops.edge.cadence.quantized_conv_nchw.per_tensor),
            0,
        )
        # Should be replaced with int8 specific variant
        self.assertEqual(
            count_node(
                gm,
                exir_ops.edge.cadence.quantized_conv_nchw_asym8sxsym8s_asym8s.per_tensor,
            ),
            1,
        )

    def test_uint8_dispatch_quantized_conv_nchw(self) -> None:
        """Test uint8 x uint8 inputs should dispatch to asym8uxasym8u_asym8u variant for quantized_conv_nchw"""
        x = torch.randint(0, 255, (1, 3, 8, 8), dtype=torch.uint8)
        w = torch.randint(0, 255, (16, 3, 3, 3), dtype=torch.uint8)
        b = torch.randint(-2147483648, 2147483647, (16,), dtype=torch.int32)
        gm = single_op_builder(
            placeholders=(x, w, b),
            op=exir_ops.edge.cadence.quantized_conv_nchw.per_tensor,
            args=(x, w, b, [1, 1], [0, 0], [1, 1], 1, 0, 0, 1.0, 1.0, 0, 1, 1),
        )
        p = CompileTimeTypeDispatchPass()
        gm = cast(PassResult, p(gm)).graph_module
        # Original op should be replaced
        self.assertEqual(
            count_node(gm, exir_ops.edge.cadence.quantized_conv_nchw.per_tensor),
            0,
        )
        # Should be replaced with uint8 specific variant
        self.assertEqual(
            count_node(
                gm,
                exir_ops.edge.cadence.quantized_conv_nchw_asym8uxsym8u_asym8u.per_tensor,
            ),
            1,
        )

    def test_int8_dispatch_quantized_conv_nhwc(self) -> None:
        """Test int8 x int8 inputs should dispatch to asym8sxasym8s_asym8s variant for quantized_conv_nhwc"""
        x = torch.randint(-128, 127, (1, 8, 8, 3), dtype=torch.int8)
        w = torch.randint(-128, 127, (16, 3, 3, 3), dtype=torch.int8)
        b = torch.randint(-2147483648, 2147483647, (16,), dtype=torch.int32)
        gm = single_op_builder(
            placeholders=(x, w, b),
            op=exir_ops.edge.cadence.quantized_conv_nhwc.per_tensor,
            args=(x, w, b, [1, 1], [0, 0], [1, 1], 1, 0, 0, 1.0, 1.0, 0, 1, 1),
        )
        p = CompileTimeTypeDispatchPass()
        gm = cast(PassResult, p(gm)).graph_module
        # Original op should be replaced
        self.assertEqual(
            count_node(gm, exir_ops.edge.cadence.quantized_conv_nhwc.per_tensor),
            0,
        )
        # Should be replaced with int8 specific variant
        self.assertEqual(
            count_node(
                gm,
                exir_ops.edge.cadence.quantized_conv_nhwc_asym8sxsym8s_asym8s.per_tensor,
            ),
            1,
        )

    def test_uint8_dispatch_quantized_conv_nhwc(self) -> None:
        """Test uint8 x uint8 inputs should dispatch to asym8uxasym8u_asym8u variant for quantized_conv_nhwc"""
        x = torch.randint(0, 255, (1, 8, 8, 3), dtype=torch.uint8)
        w = torch.randint(0, 255, (16, 3, 3, 3), dtype=torch.uint8)
        b = torch.randint(-2147483648, 2147483647, (16,), dtype=torch.int32)
        gm = single_op_builder(
            placeholders=(x, w, b),
            op=exir_ops.edge.cadence.quantized_conv_nhwc.per_tensor,
            args=(x, w, b, [1, 1], [0, 0], [1, 1], 1, 0, 0, 1.0, 1.0, 0, 1, 1),
        )
        p = CompileTimeTypeDispatchPass()
        gm = cast(PassResult, p(gm)).graph_module
        # Original op should be replaced
        self.assertEqual(
            count_node(gm, exir_ops.edge.cadence.quantized_conv_nhwc.per_tensor),
            0,
        )
        # Should be replaced with uint8 specific variant
        self.assertEqual(
            count_node(
                gm,
                exir_ops.edge.cadence.quantized_conv_nhwc_asym8uxsym8u_asym8u.per_tensor,
            ),
            1,
        )

    def test_int8_dispatch_quantized_conv_nchw_dilated(self) -> None:
        """Test int8 x int8 inputs with dilation should dispatch to dilated_asym8sxasym8s_asym8s variant for quantized_conv_nchw_dilated"""
        x = torch.randint(-128, 127, (1, 3, 8, 8), dtype=torch.int8)
        w = torch.randint(-128, 127, (16, 3, 3, 3), dtype=torch.int8)
        b = torch.randint(-2147483648, 2147483647, (16,), dtype=torch.int32)
        gm = single_op_builder(
            placeholders=(x, w, b),
            op=exir_ops.edge.cadence.quantized_conv_nchw.per_tensor,
            args=(x, w, b, [1, 1], [0, 0], [2, 2], 1, 0, 0, 1.0, 1.0, 0, 1, 1),
        )
        p = CompileTimeTypeDispatchPass()
        gm = cast(PassResult, p(gm)).graph_module
        # Original op should be replaced
        self.assertEqual(
            count_node(gm, exir_ops.edge.cadence.quantized_conv_nchw.per_tensor),
            0,
        )
        # Should be replaced with int8 specific variant
        self.assertEqual(
            count_node(
                gm,
                exir_ops.edge.cadence.quantized_conv_nchw_dilated_asym8sxsym8s_asym8s.per_tensor,
            ),
            1,
        )

    def test_uint8_dispatch_quantized_conv_nchw_dilated(self) -> None:
        """Test uint8 x uint8 inputs with dilation should dispatch to dilated_asym8uxasym8u_asym8u variant for quantized_conv_nchw"""
        x = torch.randint(0, 255, (1, 3, 8, 8), dtype=torch.uint8)
        w = torch.randint(0, 255, (16, 3, 3, 3), dtype=torch.uint8)
        b = torch.randint(-2147483648, 2147483647, (16,), dtype=torch.int32)
        gm = single_op_builder(
            placeholders=(x, w, b),
            op=exir_ops.edge.cadence.quantized_conv_nchw.per_tensor,
            args=(x, w, b, [1, 1], [0, 0], [2, 2], 1, 0, 0, 1.0, 1.0, 0, 1, 1),
        )
        p = CompileTimeTypeDispatchPass()
        gm = cast(PassResult, p(gm)).graph_module
        # Original op should be replaced
        self.assertEqual(
            count_node(gm, exir_ops.edge.cadence.quantized_conv_nchw.per_tensor),
            0,
        )
        # Should be replaced with uint8 specific variant
        self.assertEqual(
            count_node(
                gm,
                exir_ops.edge.cadence.quantized_conv_nchw_dilated_asym8uxsym8u_asym8u.per_tensor,
            ),
            1,
        )

    def test_int8_dispatch_quantized_conv_nhwc_dilated(self) -> None:
        """Test int8 x int8 inputs with dilation should dispatch to dilated_asym8sxasym8s_asym8s variant for quantized_conv_nhwc"""
        x = torch.randint(-128, 127, (1, 8, 8, 3), dtype=torch.int8)
        w = torch.randint(-128, 127, (16, 3, 3, 3), dtype=torch.int8)
        b = torch.randint(-2147483648, 2147483647, (16,), dtype=torch.int32)
        gm = single_op_builder(
            placeholders=(x, w, b),
            op=exir_ops.edge.cadence.quantized_conv_nhwc.per_tensor,
            args=(x, w, b, [1, 1], [0, 0], [2, 2], 1, 0, 0, 1.0, 1.0, 0, 1, 1),
        )
        p = CompileTimeTypeDispatchPass()
        gm = cast(PassResult, p(gm)).graph_module
        # Original op should be replaced
        self.assertEqual(
            count_node(gm, exir_ops.edge.cadence.quantized_conv_nhwc.per_tensor),
            0,
        )
        # Should be replaced with int8 specific variant
        self.assertEqual(
            count_node(
                gm,
                exir_ops.edge.cadence.quantized_conv_nhwc_dilated_asym8sxsym8s_asym8s.per_tensor,
            ),
            1,
        )

    def test_uint8_dispatch_quantized_conv_nhwc_dilated(self) -> None:
        """Test uint8 x uint8 inputs with dilation should dispatch to dilated_asym8uxasym8u_asym8u variant for quantized_conv_nhwc"""
        x = torch.randint(0, 255, (1, 8, 8, 3), dtype=torch.uint8)
        w = torch.randint(0, 255, (16, 3, 3, 3), dtype=torch.uint8)
        b = torch.randint(-2147483648, 2147483647, (16,), dtype=torch.int32)
        gm = single_op_builder(
            placeholders=(x, w, b),
            op=exir_ops.edge.cadence.quantized_conv_nhwc.per_tensor,
            args=(x, w, b, [1, 1], [0, 0], [2, 2], 1, 0, 0, 1.0, 1.0, 0, 1, 1),
        )
        p = CompileTimeTypeDispatchPass()
        gm = cast(PassResult, p(gm)).graph_module
        # Original op should be replaced
        self.assertEqual(
            count_node(gm, exir_ops.edge.cadence.quantized_conv_nhwc.per_tensor),
            0,
        )
        # Should be replaced with uint8 specific variant
        self.assertEqual(
            count_node(
                gm,
                exir_ops.edge.cadence.quantized_conv_nhwc_dilated_asym8uxsym8u_asym8u.per_tensor,
            ),
            1,
        )

    def test_int8_dispatch_quantized_conv_nchw_1d(self) -> None:
        """Test int8 x int8 inputs for 1D conv should dispatch to 1d_asym8sxasym8s_asym8s variant for quantized_conv_nchw"""
        x = torch.randint(-128, 127, (1, 3, 8), dtype=torch.int8)
        w = torch.randint(-128, 127, (16, 3, 3), dtype=torch.int8)
        b = torch.randint(-2147483648, 2147483647, (16,), dtype=torch.int32)
        gm = single_op_builder(
            placeholders=(x, w, b),
            op=exir_ops.edge.cadence.quantized_conv_nchw.per_tensor,
            args=(x, w, b, [1, 1], [0, 0], [1, 1], 1, 0, 0, 1.0, 1.0, 0, 1, 1),
        )
        p = CompileTimeTypeDispatchPass()
        gm = cast(PassResult, p(gm)).graph_module
        # Original op should be replaced
        self.assertEqual(
            count_node(gm, exir_ops.edge.cadence.quantized_conv_nchw.per_tensor),
            0,
        )
        # Should be replaced with 1D int8 specific variant
        self.assertEqual(
            count_node(
                gm,
                exir_ops.edge.cadence.quantized_conv1d_nchw_asym8sxsym8s_asym8s.per_tensor,
            ),
            1,
        )

    def test_uint8_dispatch_quantized_conv_nchw_1d(self) -> None:
        """Test uint8 x uint8 inputs for 1D conv should dispatch to 1d_asym8uxasym8u_asym8u variant for quantized_conv_nchw"""
        x = torch.randint(0, 255, (1, 3, 8), dtype=torch.uint8)
        w = torch.randint(0, 255, (16, 3, 3), dtype=torch.uint8)
        b = torch.randint(-2147483648, 2147483647, (16,), dtype=torch.int32)
        gm = single_op_builder(
            placeholders=(x, w, b),
            op=exir_ops.edge.cadence.quantized_conv_nchw.per_tensor,
            args=(x, w, b, [1, 1], [0, 0], [1, 1], 1, 0, 0, 1.0, 1.0, 0, 1, 1),
        )
        p = CompileTimeTypeDispatchPass()
        gm = cast(PassResult, p(gm)).graph_module
        # Original op should be replaced
        self.assertEqual(
            count_node(gm, exir_ops.edge.cadence.quantized_conv_nchw.per_tensor),
            0,
        )
        # Should be replaced with 1D uint8 specific variant
        self.assertEqual(
            count_node(
                gm,
                exir_ops.edge.cadence.quantized_conv1d_nchw_asym8uxsym8u_asym8u.per_tensor,
            ),
            1,
        )

    def test_int8_dispatch_quantized_conv_nhwc_1d(self) -> None:
        """Test int8 x int8 inputs for 1D conv should dispatch to 1d_asym8sxasym8s_asym8s variant for quantized_conv_nhwc"""
        x = torch.randint(-128, 127, (1, 8, 3), dtype=torch.int8)
        w = torch.randint(-128, 127, (16, 3, 3), dtype=torch.int8)
        b = torch.randint(-2147483648, 2147483647, (16,), dtype=torch.int32)
        gm = single_op_builder(
            placeholders=(x, w, b),
            op=exir_ops.edge.cadence.quantized_conv_nhwc.per_tensor,
            args=(x, w, b, [1, 1], [0, 0], [1, 1], 1, 0, 0, 1.0, 1.0, 0, 1, 1),
        )
        p = CompileTimeTypeDispatchPass()
        gm = cast(PassResult, p(gm)).graph_module
        # Original op should be replaced
        self.assertEqual(
            count_node(gm, exir_ops.edge.cadence.quantized_conv_nhwc.per_tensor),
            0,
        )
        # Should be replaced with 1D int8 specific variant
        self.assertEqual(
            count_node(
                gm,
                exir_ops.edge.cadence.quantized_conv1d_nhwc_asym8sxsym8s_asym8s.per_tensor,
            ),
            1,
        )

    def test_uint8_dispatch_quantized_conv_nhwc_1d(self) -> None:
        """Test uint8 x uint8 inputs for 1D conv should dispatch to 1d_asym8uxasym8u_asym8u variant for quantized_conv_nhwc"""
        x = torch.randint(0, 255, (1, 8, 3), dtype=torch.uint8)
        w = torch.randint(0, 255, (16, 3, 3), dtype=torch.uint8)
        b = torch.randint(-2147483648, 2147483647, (16,), dtype=torch.int32)
        gm = single_op_builder(
            placeholders=(x, w, b),
            op=exir_ops.edge.cadence.quantized_conv_nhwc.per_tensor,
            args=(x, w, b, [1, 1], [0, 0], [1, 1], 1, 0, 0, 1.0, 1.0, 0, 1, 1),
        )
        p = CompileTimeTypeDispatchPass()
        gm = cast(PassResult, p(gm)).graph_module
        # Original op should be replaced
        self.assertEqual(
            count_node(gm, exir_ops.edge.cadence.quantized_conv_nhwc.per_tensor),
            0,
        )
        # Should be replaced with 1D uint8 specific variant
        self.assertEqual(
            count_node(
                gm,
                exir_ops.edge.cadence.quantized_conv1d_nhwc_asym8uxsym8u_asym8u.per_tensor,
            ),
            1,
        )

    def test_int8_dispatch_quantized_add(self) -> None:
        """Test int8 x int8 inputs should dispatch to asym8sxasym8s_asym8s variant for quantized_add"""
        x = torch.randint(-128, 127, (2, 3), dtype=torch.int8)
        y = torch.randint(-128, 127, (2, 3), dtype=torch.int8)
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
        # Should be replaced with int8 specific variant
        self.assertEqual(
            count_node(
                gm,
                exir_ops.edge.cadence.quantized_add_asym8sxasym8s_asym8s.per_tensor,
            ),
            1,
        )

    def test_uint8_dispatch_quantized_add(self) -> None:
        """Test uint8 x uint8 inputs should dispatch to asym8uxasym8u_asym8u variant for quantized_add"""
        x = torch.randint(0, 255, (2, 3), dtype=torch.uint8)
        y = torch.randint(0, 255, (2, 3), dtype=torch.uint8)
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
        # Should be replaced with uint8 specific variant
        self.assertEqual(
            count_node(
                gm,
                exir_ops.edge.cadence.quantized_add_asym8uxasym8u_asym8u.per_tensor,
            ),
            1,
        )

    def test_int8_dispatch_quantized_conv_nchw_depthwise(self) -> None:
        """Test int8 x int8 inputs with depthwise should dispatch to depthwise_asym8sxsym8s_asym8s variant for quantized_conv_nchw"""
        # Depthwise convolution: groups == input_channels
        x = torch.randint(-128, 127, (1, 3, 8, 8), dtype=torch.int8)
        w = torch.randint(
            -128, 127, (3, 1, 3, 3), dtype=torch.int8
        )  # groups=3, input_channels=3
        b = torch.randint(-2147483648, 2147483647, (3,), dtype=torch.int32)
        gm = single_op_builder(
            placeholders=(x, w, b),
            op=exir_ops.edge.cadence.quantized_conv_nchw.per_tensor,
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
            ),  # groups=3
        )
        p = CompileTimeTypeDispatchPass()
        gm = cast(PassResult, p(gm)).graph_module
        # Original op should be replaced
        self.assertEqual(
            count_node(gm, exir_ops.edge.cadence.quantized_conv_nchw.per_tensor),
            0,
        )
        # Should be replaced with int8 depthwise specific variant
        self.assertEqual(
            count_node(
                gm,
                exir_ops.edge.cadence.quantized_conv_nchw_depthwise_asym8sxsym8s_asym8s.per_tensor,
            ),
            1,
        )

    def test_uint8_dispatch_quantized_conv_nchw_depthwise(self) -> None:
        """Test uint8 x uint8 inputs with depthwise should dispatch to depthwise_asym8uxasym8u_asym8u variant for quantized_conv_nchw"""
        # Depthwise convolution: groups == input_channels
        x = torch.randint(0, 255, (1, 3, 8, 8), dtype=torch.uint8)
        w = torch.randint(
            0, 255, (3, 1, 3, 3), dtype=torch.uint8
        )  # groups=3, input_channels=3
        b = torch.randint(-2147483648, 2147483647, (3,), dtype=torch.int32)
        gm = single_op_builder(
            placeholders=(x, w, b),
            op=exir_ops.edge.cadence.quantized_conv_nchw.per_tensor,
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
            ),  # groups=3
        )
        p = CompileTimeTypeDispatchPass()
        gm = cast(PassResult, p(gm)).graph_module
        # Original op should be replaced
        self.assertEqual(
            count_node(gm, exir_ops.edge.cadence.quantized_conv_nchw.per_tensor),
            0,
        )
        # Should be replaced with uint8 depthwise specific variant
        self.assertEqual(
            count_node(
                gm,
                exir_ops.edge.cadence.quantized_conv_nchw_depthwise_asym8uxsym8u_asym8u.per_tensor,
            ),
            1,
        )

    def test_int8_dispatch_quantized_conv_nhwc_depthwise(self) -> None:
        """Test int8 x int8 inputs with depthwise should dispatch to depthwise_asym8sxsym8s_asym8s variant for quantized_conv_nhwc"""
        # Depthwise convolution: groups == input_channels
        x = torch.randint(-128, 127, (1, 8, 8, 3), dtype=torch.int8)
        w = torch.randint(
            -128, 127, (3, 3, 3, 1), dtype=torch.int8
        )  # groups=3, input_channels=3
        b = torch.randint(-2147483648, 2147483647, (3,), dtype=torch.int32)
        gm = single_op_builder(
            placeholders=(x, w, b),
            op=exir_ops.edge.cadence.quantized_conv_nhwc.per_tensor,
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
            ),  # groups=3
        )
        p = CompileTimeTypeDispatchPass()
        gm = cast(PassResult, p(gm)).graph_module
        # Original op should be replaced
        self.assertEqual(
            count_node(gm, exir_ops.edge.cadence.quantized_conv_nhwc.per_tensor),
            0,
        )
        # Should be replaced with int8 depthwise specific variant
        self.assertEqual(
            count_node(
                gm,
                exir_ops.edge.cadence.quantized_conv_nhwc_depthwise_asym8sxsym8s_asym8s.per_tensor,
            ),
            1,
        )

    def test_uint8_dispatch_quantized_conv_nhwc_depthwise(self) -> None:
        """Test uint8 x uint8 inputs with depthwise should dispatch to depthwise_asym8uxasym8u_asym8u variant for quantized_conv_nhwc"""
        # Depthwise convolution: groups == input_channels
        x = torch.randint(0, 255, (1, 8, 8, 3), dtype=torch.uint8)
        w = torch.randint(
            0, 255, (3, 3, 3, 1), dtype=torch.uint8
        )  # groups=3, input_channels=3
        b = torch.randint(-2147483648, 2147483647, (3,), dtype=torch.int32)
        gm = single_op_builder(
            placeholders=(x, w, b),
            op=exir_ops.edge.cadence.quantized_conv_nhwc.per_tensor,
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
            ),  # groups=3
        )
        p = CompileTimeTypeDispatchPass()
        gm = cast(PassResult, p(gm)).graph_module
        # Original op should be replaced
        self.assertEqual(
            count_node(gm, exir_ops.edge.cadence.quantized_conv_nhwc.per_tensor),
            0,
        )
        # Should be replaced with uint8 depthwise specific variant
        self.assertEqual(
            count_node(
                gm,
                exir_ops.edge.cadence.quantized_conv_nhwc_depthwise_asym8uxsym8u_asym8u.per_tensor,
            ),
            1,
        )
