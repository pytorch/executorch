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
        x = torch.randint(-128, 127, (2, 3), dtype=torch.int8)
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
        x = torch.randint(0, 255, (2, 3), dtype=torch.uint8)
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
        x = torch.randint(-128, 127, (2, 3), dtype=torch.int8)
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
