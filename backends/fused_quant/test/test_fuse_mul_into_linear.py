# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.


import unittest
from typing import Any, Optional

import executorch.backends.cadence.aot.ops_registrations  # noqa
import executorch.backends.fused_quant.ops  # noqa
import torch
from executorch.backends.fused_quant.graph_utils import get_constant
from executorch.backends.fused_quant.optimization_passes.fuse_mul_into_linear import (
    FuseMulIntoLinear,
)
from executorch.backends.fused_quant.test.helpers import (
    create_per_channel_qparams,
    create_per_tensor_qparams,
)
from executorch.backends.test.program_builder import ProgramBuilder
from executorch.backends.transforms.permute_pass_utils import get_arg
from executorch.exir.dialects._ops import ops as exir_ops
from parameterized import parameterized
from torch.export import ExportedProgram
from torch.export.graph_signature import InputKind


class FuseMulIntoLinearTest(unittest.TestCase):
    """Tests for the FuseMulIntoLinear optimization pass."""

    IN_FEATURES = 8
    OUT_FEATURES = 16

    def _build_linear_mul(
        self,
        x: torch.Tensor,
        weight: torch.Tensor,
        bias: Optional[torch.Tensor],
        mul_scale: torch.Tensor | float,
        x_scale: float = 1.0,
        w_scale_per_ch: Optional[torch.Tensor] = None,
        w_zero_point_per_ch: Optional[torch.Tensor] = None,
        per_tensor_weight_scale: bool = False,
        linear_out_scale: float = 0.5,
        mul_out_scale: float = 0.125,
        mul_other_scale: float | None = None,
    ) -> ExportedProgram:
        out_f = weight.shape[0]
        builder = ProgramBuilder()

        x_ph = builder.placeholder("x", x)
        w_ph = builder.placeholder("weight", weight, input_kind=InputKind.BUFFER)
        b_ph = (
            builder.placeholder("bias", bias, input_kind=InputKind.BUFFER)
            if bias is not None
            else None
        )
        scale_ph = (
            builder.placeholder("scale", mul_scale, input_kind=InputKind.BUFFER)
            if isinstance(mul_scale, torch.Tensor)
            else None
        )

        inp_qparams = create_per_tensor_qparams(
            builder, scale=x_scale, dtype=torch.float32
        )
        weight_qparams = (
            create_per_tensor_qparams(builder, dtype=torch.float32)
            if per_tensor_weight_scale
            else create_per_channel_qparams(builder, out_f, dtype=torch.float32)
        )
        if w_scale_per_ch is not None:
            builder.state_dict["wt_scale_0"] = w_scale_per_ch.clone().reshape(-1, 1)
        if w_zero_point_per_ch is not None:
            builder.state_dict["wt_zp_0"] = w_zero_point_per_ch.clone().reshape(-1, 1)

        out_qparams = create_per_tensor_qparams(
            builder, scale=linear_out_scale, dtype=torch.int8
        )

        linear = builder.call_operator(
            op=exir_ops.edge.fused_quant.linear.default,
            args=(
                x_ph,
                w_ph,
                b_ph,
                *inp_qparams,
                *weight_qparams,
                None,
                None,
                torch.uint8,
                0,
                0,
                *out_qparams,
            ),
        )

        mul_inp_qp = create_per_tensor_qparams(
            builder, scale=linear_out_scale, dtype=torch.float32
        )
        mul_out_qp = create_per_tensor_qparams(
            builder, scale=mul_out_scale, dtype=torch.int8
        )

        if scale_ph is not None:
            if mul_other_scale is None:
                mul_other_qp: tuple[Any, ...] = (
                    None,
                    None,
                    torch.float32,
                    0,
                    0,
                )
            else:
                other_scale, other_zero_point, other_dtype, _, _ = (
                    create_per_tensor_qparams(
                        builder,
                        scale=mul_other_scale,
                        dtype=torch.float32,
                    )
                )
                mul_other_qp = (
                    other_scale,
                    other_zero_point,
                    other_dtype,
                    -32768,
                    32767,
                )
            mul = builder.call_operator(
                op=exir_ops.edge.fused_quant.mul.default,
                args=(linear, scale_ph, *mul_inp_qp, *mul_other_qp, *mul_out_qp),
            )
        else:
            mul = builder.call_operator(
                op=exir_ops.edge.fused_quant.mul.Scalar,
                args=(linear, *mul_inp_qp, *mul_out_qp, mul_scale),
            )

        dequantize = builder.call_operator(
            op=exir_ops.edge.quantized_decomposed.dequantize_per_tensor.default,
            args=(mul, mul_out_scale, 0, -128, 127, torch.int8),
        )
        builder.output([dequantize])

        return builder.get_program()

    def _default_linear_mul(self, with_bias: bool = True) -> ExportedProgram:
        x = torch.randn(1, self.IN_FEATURES)
        weight = torch.randint(
            -16, 16, (self.OUT_FEATURES, self.IN_FEATURES), dtype=torch.int8
        )
        bias = torch.randn(self.OUT_FEATURES) if with_bias else None
        mul_scale = torch.randn(self.OUT_FEATURES).abs().clamp(min=0.5, max=2.0)
        return self._build_linear_mul(x, weight, bias, mul_scale)

    def test_mul_folded_into_linear(self) -> None:
        ep = self._default_linear_mul(with_bias=True)

        result = FuseMulIntoLinear().call(ep)
        self.assertTrue(result.modified)

        graph = result.exported_program.graph_module.graph
        mul_nodes = graph.find_nodes(
            op="call_function",
            target=exir_ops.edge.fused_quant.mul.default,
        )
        self.assertEqual(len(mul_nodes), 0, "Mul node should be removed")

        linear_nodes = graph.find_nodes(
            op="call_function",
            target=exir_ops.edge.fused_quant.linear.default,
        )
        self.assertEqual(len(linear_nodes), 1, "Linear node should remain")

    def test_mul_folded_no_bias(self) -> None:
        ep = self._default_linear_mul(with_bias=False)

        result = FuseMulIntoLinear().call(ep)
        self.assertTrue(result.modified)

        mul_nodes = result.exported_program.graph_module.graph.find_nodes(
            op="call_function",
            target=exir_ops.edge.fused_quant.mul.default,
        )
        self.assertEqual(len(mul_nodes), 0, "Mul node should be removed")

    def test_scalar_mul_folded_with_per_tensor_weight_scale(self) -> None:
        x = torch.randn(1, self.IN_FEATURES)
        weight = torch.randint(
            -16,
            16,
            (self.OUT_FEATURES, self.IN_FEATURES),
            dtype=torch.int8,
        )
        ep = self._build_linear_mul(
            x,
            weight,
            torch.randn(self.OUT_FEATURES),
            torch.tensor(2.0),
            per_tensor_weight_scale=True,
        )

        result = FuseMulIntoLinear().call(ep)

        self.assertTrue(result.modified)
        linear = result.exported_program.graph_module.graph.find_nodes(
            op="call_function",
            target=exir_ops.edge.fused_quant.linear.default,
        )[0]
        weight_scale_node = get_arg(linear, "weight_scale", torch.fx.Node)
        weight_scale = get_constant(result.exported_program, weight_scale_node)
        self.assertIsNotNone(weight_scale)
        self.assertEqual(weight_scale.shape, torch.Size([]))

    def test_scalar_mul_folded_into_linear(self) -> None:
        ep = self._build_linear_mul(
            torch.randn(1, self.IN_FEATURES),
            torch.randint(
                -16,
                16,
                (self.OUT_FEATURES, self.IN_FEATURES),
                dtype=torch.int8,
            ),
            torch.randn(self.OUT_FEATURES),
            1.5,
        )

        result = FuseMulIntoLinear().call(ep)

        self.assertTrue(result.modified)
        scalar_mul_nodes = result.exported_program.graph_module.graph.find_nodes(
            op="call_function",
            target=exir_ops.edge.fused_quant.mul.Scalar,
        )
        self.assertEqual(len(scalar_mul_nodes), 0, "Scalar mul should be removed")

    def test_negative_scalar_mul_preserves_numerics(self) -> None:
        x = torch.tensor([[1, -1, 2, -2, 1, 0, -1, 2]], dtype=torch.int8)
        weight = (
            torch.arange(self.OUT_FEATURES * self.IN_FEATURES)
            .reshape(self.OUT_FEATURES, self.IN_FEATURES)
            .remainder(5)
            .sub(2)
            .to(torch.int8)
        )
        bias = torch.arange(self.OUT_FEATURES).remainder(3).sub(1).to(torch.float32)
        weight_zero_point = (
            torch.arange(self.OUT_FEATURES).remainder(5).sub(2).to(torch.int64)
        )
        multiplier = -2.0
        ep = self._build_linear_mul(
            x,
            weight,
            bias,
            multiplier,
            w_scale_per_ch=torch.ones(self.OUT_FEATURES),
            w_zero_point_per_ch=weight_zero_point,
            linear_out_scale=1.0,
            mul_out_scale=1.0,
        )
        (unfused_output,) = ep.module()(x)

        result = FuseMulIntoLinear()(ep)

        self.assertTrue(result.modified)
        fused_ep = result.exported_program
        (fused_output,) = fused_ep.module()(x)
        torch.testing.assert_close(fused_output, unfused_output, rtol=0, atol=0)

        linear = fused_ep.graph_module.graph.find_nodes(
            op="call_function",
            target=exir_ops.edge.fused_quant.linear.default,
        )[0]
        weight_node = get_arg(linear, "weight", torch.fx.Node)
        weight_scale_node = get_arg(linear, "weight_scale", torch.fx.Node)
        weight_zero_point_node = get_arg(linear, "weight_zero_point", torch.fx.Node)
        bias_node = get_arg(linear, "bias", torch.fx.Node)
        actual_weight = get_constant(fused_ep, weight_node)
        actual_weight_scale = get_constant(fused_ep, weight_scale_node)
        actual_weight_zero_point = get_constant(fused_ep, weight_zero_point_node)
        actual_bias = get_constant(fused_ep, bias_node)
        self.assertIsNotNone(actual_weight)
        self.assertIsNotNone(actual_weight_scale)
        self.assertIsNotNone(actual_weight_zero_point)
        self.assertIsNotNone(actual_bias)
        torch.testing.assert_close(
            actual_weight,
            (-1 - weight.to(torch.int16)).to(torch.int8),
        )
        torch.testing.assert_close(
            actual_weight_scale,
            torch.full((self.OUT_FEATURES, 1), abs(multiplier)),
        )
        torch.testing.assert_close(
            actual_weight_zero_point,
            -1 - weight_zero_point.reshape(-1, 1),
        )
        torch.testing.assert_close(actual_bias, bias * multiplier)

    def test_vector_mul_preserves_numerics(self) -> None:
        x = torch.tensor([[1, -1, 2, -2, 1, 0, -1, 2]], dtype=torch.int8)
        weight = (
            torch.arange(self.OUT_FEATURES * self.IN_FEATURES)
            .reshape(self.OUT_FEATURES, self.IN_FEATURES)
            .remainder(5)
            .sub(2)
            .to(torch.int8)
        )
        bias = torch.arange(self.OUT_FEATURES).remainder(3).sub(1).to(torch.float32)
        multiplier = torch.tensor([1.0, 2.0] * (self.OUT_FEATURES // 2))
        ep = self._build_linear_mul(
            x,
            weight,
            bias,
            multiplier,
            w_scale_per_ch=torch.ones(self.OUT_FEATURES),
            linear_out_scale=1.0,
            mul_out_scale=1.0,
        )
        (unfused_output,) = ep.module()(x)

        result = FuseMulIntoLinear()(ep)

        self.assertTrue(result.modified)
        fused_ep = result.exported_program
        self._assert_mul_removed(fused_ep)
        (fused_output,) = fused_ep.module()(x)
        torch.testing.assert_close(fused_output, unfused_output, rtol=0, atol=0)

        linear = fused_ep.graph_module.graph.find_nodes(
            op="call_function",
            target=exir_ops.edge.fused_quant.linear.default,
        )[0]
        weight_scale_node = get_arg(linear, "weight_scale", torch.fx.Node)
        bias_node = get_arg(linear, "bias", torch.fx.Node)
        actual_weight_scale = get_constant(fused_ep, weight_scale_node)
        actual_bias = get_constant(fused_ep, bias_node)
        self.assertIsNotNone(actual_weight_scale)
        self.assertIsNotNone(actual_bias)
        torch.testing.assert_close(actual_weight_scale, multiplier.reshape(-1, 1))
        torch.testing.assert_close(actual_bias, bias * multiplier)

    def test_quantized_constant_mul_is_dequantized_before_fold(self) -> None:
        represented_value = 0.5
        quantized_value = torch.tensor(32767, dtype=torch.int16)
        ep = self._build_linear_mul(
            torch.randn(1, self.IN_FEATURES),
            torch.randint(
                -16,
                16,
                (self.OUT_FEATURES, self.IN_FEATURES),
                dtype=torch.int8,
            ),
            torch.randn(self.OUT_FEATURES),
            quantized_value,
            mul_other_scale=represented_value / int(quantized_value),
        )

        result = FuseMulIntoLinear().call(ep)

        self.assertTrue(result.modified)
        linear = result.exported_program.graph_module.graph.find_nodes(
            op="call_function",
            target=exir_ops.edge.fused_quant.linear.default,
        )[0]
        weight_scale_node = get_arg(linear, "weight_scale", torch.fx.Node)
        weight_scale = get_constant(result.exported_program, weight_scale_node)
        self.assertIsNotNone(weight_scale)
        torch.testing.assert_close(
            weight_scale,
            torch.full((self.OUT_FEATURES, 1), represented_value),
        )

    def test_graph_signature_updated(self) -> None:
        """After folding, the scale placeholder used by mul becomes dead and
        should be removed from the graph signature by constant_prop_pass."""
        ep = self._default_linear_mul(with_bias=True)

        input_names_before = {s.arg.name for s in ep.graph_signature.input_specs}
        self.assertIn("scale", input_names_before)

        result = FuseMulIntoLinear().call(ep)
        self.assertTrue(result.modified)

        fused_ep = result.exported_program
        input_names_after = {s.arg.name for s in fused_ep.graph_signature.input_specs}
        placeholder_names = {
            n.name for n in fused_ep.graph_module.graph.find_nodes(op="placeholder")
        }
        self.assertEqual(
            input_names_after,
            placeholder_names,
            "Graph signature input_specs should match actual placeholder nodes",
        )
        self.assertNotIn(
            "scale",
            input_names_after,
            "Dead scale placeholder should be removed from signature",
        )

    def test_numerics_improve(self) -> None:
        """Fusing mul into linear eliminates an intermediate quantization
        boundary, so the fused result should be closer to the true fp32
        result than the unfused quantized graph.

        We build an executable fused_quant graph (linear->mul->dequant),
        run it before and after the optimization, and compare both outputs
        against a pure fp32 reference.
        """
        torch.manual_seed(42)
        in_f, out_f = self.IN_FEATURES, self.OUT_FEATURES
        batch = 4

        x_float = torch.randn(batch, in_f)
        weight_float = torch.randn(out_f, in_f)
        bias_float = torch.randn(out_f)
        mul_scale_val = torch.rand(out_f).clamp(min=0.5, max=2.0)

        # fp32 reference: linear(x, w, b) * scale
        fp32_ref = (
            torch.nn.functional.linear(x_float, weight_float, bias_float)
            * mul_scale_val
        )

        # Compute scales from data ranges: (max - min) / (qmax - qmin)
        x_scale = (x_float.max() - x_float.min()).item() / 255.0
        w_scale_per_ch = (
            weight_float.max(dim=1).values - weight_float.min(dim=1).values
        ) / 255.0

        # Quantize tensors
        x_int8 = torch.clamp(torch.round(x_float / x_scale), -128, 127).to(torch.int8)
        weight_int8 = torch.clamp(
            torch.round(weight_float / w_scale_per_ch.unsqueeze(1)), -128, 127
        ).to(torch.int8)

        # Compute output scales from approximate output ranges
        linear_approx = torch.nn.functional.linear(x_float, weight_float, bias_float)
        linear_out_scale = (linear_approx.max() - linear_approx.min()).item() / 255.0
        mul_approx = linear_approx * mul_scale_val
        mul_out_scale = (mul_approx.max() - mul_approx.min()).item() / 255.0

        ep = self._build_linear_mul(
            x_int8,
            weight_int8,
            bias_float,
            mul_scale_val,
            x_scale=x_scale,
            w_scale_per_ch=w_scale_per_ch,
            linear_out_scale=linear_out_scale,
            mul_out_scale=mul_out_scale,
        )

        (unfused_out,) = ep.module()(x_int8)

        result = FuseMulIntoLinear().call(ep)
        self.assertTrue(result.modified)
        (fused_out,) = result.exported_program.module()(x_int8)

        unfused_err = (unfused_out.float() - fp32_ref).abs().mean()
        fused_err = (fused_out.float() - fp32_ref).abs().mean()

        self.assertLess(
            fused_err,
            unfused_err,
            f"Fused error ({fused_err:.6f}) should be < unfused error ({unfused_err:.6f})",
        )

    def test_no_fold_when_linear_has_multiple_users(self) -> None:
        """Mul should not be folded if the linear output is used elsewhere."""
        builder = ProgramBuilder()

        x = builder.placeholder("x", torch.randn(1, 8))
        w = builder.placeholder(
            "weight",
            torch.randint(-16, 16, (16, 8), dtype=torch.int8),
            input_kind=InputKind.BUFFER,
        )
        scale = builder.placeholder(
            "scale",
            torch.randn(16).abs() + 0.1,
            input_kind=InputKind.BUFFER,
        )

        inp_qparams = create_per_tensor_qparams(builder)
        weight_qparams = create_per_channel_qparams(builder, 16)
        out_qparams = create_per_tensor_qparams(builder)

        linear = builder.call_operator(
            op=exir_ops.edge.fused_quant.linear.default,
            args=(
                x,
                w,
                None,
                *inp_qparams,
                *weight_qparams,
                None,
                None,
                torch.uint8,
                0,
                0,
                *out_qparams,
            ),
        )

        mul_inp_qp = create_per_tensor_qparams(builder)
        mul_other_qp = create_per_tensor_qparams(builder)
        mul_out_qp = create_per_tensor_qparams(builder)

        mul = builder.call_operator(
            op=exir_ops.edge.fused_quant.mul.default,
            args=(linear, scale, *mul_inp_qp, *mul_other_qp, *mul_out_qp),
        )

        dequant_linear = builder.call_operator(
            op=exir_ops.edge.quantized_decomposed.dequantize_per_tensor.default,
            args=(linear, 0.5, 0, -128, 127, torch.int8),
        )
        dequant_mul = builder.call_operator(
            op=exir_ops.edge.quantized_decomposed.dequantize_per_tensor.default,
            args=(mul, 0.125, 0, -128, 127, torch.int8),
        )
        builder.output([dequant_linear, dequant_mul])

        ep = builder.get_program()
        result = FuseMulIntoLinear().call(ep)

        self.assertFalse(
            result.modified, "Should not fold when linear has multiple users"
        )

    def test_no_fold_when_weight_scale_is_shared(self) -> None:
        """Mul should not be folded if weight_scale is used by another op."""
        builder = ProgramBuilder()

        x = builder.placeholder("x", torch.randn(1, 8))
        w = builder.placeholder(
            "weight",
            torch.randint(-16, 16, (16, 8), dtype=torch.int8),
            input_kind=InputKind.BUFFER,
        )
        scale = builder.placeholder(
            "scale",
            torch.randn(16).abs() + 0.1,
            input_kind=InputKind.BUFFER,
        )

        inp_qparams = create_per_tensor_qparams(builder)
        weight_qparams = create_per_channel_qparams(builder, 16)
        out_qparams = create_per_tensor_qparams(builder)

        linear1 = builder.call_operator(
            op=exir_ops.edge.fused_quant.linear.default,
            args=(
                x,
                w,
                None,
                *inp_qparams,
                *weight_qparams,
                None,
                None,
                torch.uint8,
                0,
                0,
                *out_qparams,
            ),
        )

        inp_qparams2 = create_per_tensor_qparams(builder)
        out_qparams2 = create_per_tensor_qparams(builder)

        linear2 = builder.call_operator(
            op=exir_ops.edge.fused_quant.linear.default,
            args=(
                x,
                w,
                None,
                *inp_qparams2,
                *weight_qparams,
                None,
                None,
                torch.uint8,
                0,
                0,
                *out_qparams2,
            ),
        )

        mul_inp_qp = create_per_tensor_qparams(builder)
        mul_other_qp = create_per_tensor_qparams(builder)
        mul_out_qp = create_per_tensor_qparams(builder)

        mul = builder.call_operator(
            op=exir_ops.edge.fused_quant.mul.default,
            args=(linear1, scale, *mul_inp_qp, *mul_other_qp, *mul_out_qp),
        )

        dequant_mul = builder.call_operator(
            op=exir_ops.edge.quantized_decomposed.dequantize_per_tensor.default,
            args=(mul, 0.125, 0, -128, 127, torch.int8),
        )
        dequant_linear2 = builder.call_operator(
            op=exir_ops.edge.quantized_decomposed.dequantize_per_tensor.default,
            args=(linear2, 0.5, 0, -128, 127, torch.int8),
        )
        builder.output([dequant_mul, dequant_linear2])

        ep = builder.get_program()
        result = FuseMulIntoLinear().call(ep)

        self.assertFalse(
            result.modified,
            "Should not fold when weight_scale is shared with another linear",
        )

    def _build_linear_passthrough_mul(
        self,
        passthrough: str = "permute",
        constant_shape: tuple[int, ...] = (1,),
        linear_out_dtype: torch.dtype = torch.int8,
        mul_out_dtype: torch.dtype = torch.int8,
        constant_dtype: torch.dtype = torch.float32,
        scalar: Optional[float] = None,
    ) -> tuple[ExportedProgram, torch.Tensor]:
        """Build a linear → passthrough → mul(constant) graph.

        Args:
            passthrough: "permute", "view", or "permute_view".
            constant_shape: Shape of the mul constant.
            linear_out_dtype: Output dtype of the linear's out qparams.
            mul_out_dtype: Output dtype of the mul's out qparams.
            constant_dtype: Dtype of the mul constant tensor.

        Returns:
            (program, input_tensor)
        """

        builder = ProgramBuilder()
        in_features, out_features = 8, 16
        inp_tensor = torch.randn(1, in_features)

        x = builder.placeholder("x", inp_tensor)
        w = builder.placeholder(
            "weight",
            torch.randint(-16, 16, (out_features, in_features), dtype=torch.int8),
            input_kind=InputKind.BUFFER,
        )
        b = builder.placeholder(
            "bias", torch.randn(out_features), input_kind=InputKind.BUFFER
        )
        scale = (
            builder.placeholder(
                "scale",
                torch.randn(*constant_shape)
                .abs()
                .clamp(min=0.5, max=2.0)
                .to(constant_dtype),
                input_kind=InputKind.BUFFER,
            )
            if scalar is None
            else None
        )

        weight_qparams = create_per_channel_qparams(
            builder, out_features, scale=0.05, dtype=torch.float32
        )
        out_qparams = create_per_tensor_qparams(
            builder, scale=0.5, dtype=linear_out_dtype
        )

        linear = builder.call_operator(
            op=exir_ops.edge.fused_quant.linear.default,
            args=(
                x,
                w,
                b,
                None,
                None,
                torch.float32,
                0,
                0,
                *weight_qparams,
                None,
                None,
                torch.uint8,
                0,
                0,
                *out_qparams,
            ),
        )

        node = linear
        if "permute" in passthrough:
            node = builder.call_operator(
                op=exir_ops.edge.aten.permute_copy.default,
                args=(node, [0, 1]),
            )
        if "view" in passthrough:
            node = builder.call_operator(
                op=exir_ops.edge.aten.view_copy.default,
                args=(node, [1, out_features]),
            )

        mul_inp_qp = create_per_tensor_qparams(builder, scale=0.5, dtype=torch.float32)
        mul_out_qp = create_per_tensor_qparams(
            builder, scale=0.125, dtype=mul_out_dtype
        )

        if scale is not None:
            mul_other_qp = create_per_tensor_qparams(
                builder, scale=0.25, dtype=torch.float32
            )
            mul = builder.call_operator(
                op=exir_ops.edge.fused_quant.mul.default,
                args=(node, scale, *mul_inp_qp, *mul_other_qp, *mul_out_qp),
            )
        else:
            mul = builder.call_operator(
                op=exir_ops.edge.fused_quant.mul.Scalar,
                args=(node, *mul_inp_qp, *mul_out_qp, scalar),
            )

        dequantize = builder.call_operator(
            op=exir_ops.edge.quantized_decomposed.dequantize_per_tensor.default,
            args=(
                mul,
                0.125,
                0,
                torch.iinfo(mul_out_dtype).min,
                torch.iinfo(mul_out_dtype).max,
                mul_out_dtype,
            ),
        )
        builder.output([dequantize])
        return builder.get_program(), inp_tensor

    def _run_ep(self, ep: ExportedProgram, inp: torch.Tensor) -> torch.Tensor:
        full_inputs = []
        sig = ep.graph_signature
        for spec in sig.input_specs:
            if spec.kind == InputKind.USER_INPUT:
                full_inputs.append(inp)
            else:
                assert spec.target is not None
                full_inputs.append(ep.state_dict[spec.target])
        return ep.graph_module(*full_inputs)

    def _assert_mul_removed(self, ep: ExportedProgram) -> None:
        mul_nodes = ep.graph_module.graph.find_nodes(
            op="call_function", target=exir_ops.edge.fused_quant.mul.default
        )
        self.assertEqual(len(mul_nodes), 0, "Mul should be removed")

    @parameterized.expand(
        [
            ("permute_scalar", "permute", (1,)),
            ("view_scalar", "view", (1,)),
            ("permute_view_scalar", "permute_view", (1,)),
        ]
    )
    def test_mul_folded_through_passthrough(
        self, _name: str, passthrough: str, constant_shape: tuple[int, ...]
    ) -> None:
        ep, _ = self._build_linear_passthrough_mul(
            passthrough=passthrough, constant_shape=constant_shape
        )

        result = FuseMulIntoLinear().call(ep)
        self.assertTrue(result.modified)
        self._assert_mul_removed(result.exported_program)

    def test_scalar_mul_folded_through_passthrough(self) -> None:
        ep, _ = self._build_linear_passthrough_mul(
            passthrough="permute_view", scalar=1.5
        )

        result = FuseMulIntoLinear().call(ep)

        self.assertTrue(result.modified)
        scalar_mul_nodes = result.exported_program.graph_module.graph.find_nodes(
            op="call_function",
            target=exir_ops.edge.fused_quant.mul.Scalar,
        )
        self.assertEqual(len(scalar_mul_nodes), 0, "Scalar mul should be removed")

    def test_mul_not_folded_multi_dim_constant(self) -> None:
        """Constant that varies in non-output dimensions should not fold."""

        builder = ProgramBuilder()
        in_features, out_features, batch = 8, 16, 4

        x = builder.placeholder("x", torch.randn(batch, in_features))
        w = builder.placeholder(
            "weight",
            torch.randint(-16, 16, (out_features, in_features), dtype=torch.int8),
            input_kind=InputKind.BUFFER,
        )
        scale = builder.placeholder(
            "scale",
            torch.randn(batch, 1).abs().clamp(min=0.5, max=2.0),
            input_kind=InputKind.BUFFER,
        )

        weight_qparams = create_per_channel_qparams(
            builder, out_features, scale=0.05, dtype=torch.float32
        )
        out_qparams = create_per_tensor_qparams(builder, scale=0.5, dtype=torch.int8)

        linear = builder.call_operator(
            op=exir_ops.edge.fused_quant.linear.default,
            args=(
                x,
                w,
                None,
                None,
                None,
                torch.float32,
                0,
                0,
                *weight_qparams,
                None,
                None,
                torch.uint8,
                0,
                0,
                *out_qparams,
            ),
        )

        mul_inp_qp = create_per_tensor_qparams(builder, scale=0.5, dtype=torch.float32)
        mul_other_qp = create_per_tensor_qparams(
            builder, scale=0.25, dtype=torch.float32
        )
        mul_out_qp = create_per_tensor_qparams(builder, scale=0.125, dtype=torch.int8)

        mul = builder.call_operator(
            op=exir_ops.edge.fused_quant.mul.default,
            args=(linear, scale, *mul_inp_qp, *mul_other_qp, *mul_out_qp),
        )
        dequantize = builder.call_operator(
            op=exir_ops.edge.quantized_decomposed.dequantize_per_tensor.default,
            args=(mul, 0.125, 0, -128, 127, torch.int8),
        )
        builder.output([dequantize])

        ep = builder.get_program()
        result = FuseMulIntoLinear().call(ep)
        self.assertFalse(
            result.modified,
            "Should not fold constant that varies in non-output dimensions",
        )

    @parameterized.expand(
        [
            ("permute", "permute", exir_ops.edge.aten.permute_copy.default),
            ("view", "view", exir_ops.edge.aten.view_copy.default),
        ]
    )
    def test_mul_folded_through_passthrough_retypes_meta(
        self, _name: str, passthrough: str, pt_target: Any
    ) -> None:
        """When the mul's output dtype differs from the linear's original
        (int8 → uint8), folding a scalar through a passthrough must retype the
        linear and every passthrough node's meta['val'] to the new dtype."""
        ep, _ = self._build_linear_passthrough_mul(
            passthrough=passthrough,
            constant_shape=(1,),
            linear_out_dtype=torch.int8,
            mul_out_dtype=torch.uint8,
        )

        result = FuseMulIntoLinear().call(ep)
        self.assertTrue(result.modified)
        self._assert_mul_removed(result.exported_program)

        graph = result.exported_program.graph_module.graph
        linear_nodes = graph.find_nodes(
            op="call_function", target=exir_ops.edge.fused_quant.linear.default
        )
        self.assertEqual(len(linear_nodes), 1)
        self.assertEqual(linear_nodes[0].meta["val"].dtype, torch.uint8)

        pt_nodes = graph.find_nodes(op="call_function", target=pt_target)
        self.assertEqual(len(pt_nodes), 1)
        self.assertEqual(
            pt_nodes[0].meta["val"].dtype,
            torch.uint8,
            "passthrough meta['val'] should take the mul's output dtype",
        )

    def test_mul_folded_fp16_constant(self) -> None:
        """A non-float32 (fp16) constant still folds; multiplying it into the
        float32 weight_scale promotes the result back to float32."""
        ep, _ = self._build_linear_passthrough_mul(
            passthrough="permute",
            constant_shape=(1,),
            constant_dtype=torch.float16,
        )
        result = FuseMulIntoLinear().call(ep)
        self.assertTrue(result.modified)
        self._assert_mul_removed(result.exported_program)

        fused_ep = result.exported_program
        linear_nodes = fused_ep.graph_module.graph.find_nodes(
            op="call_function", target=exir_ops.edge.fused_quant.linear.default
        )
        self.assertEqual(len(linear_nodes), 1)
        weight_scale_node = get_arg(
            linear_nodes[0], "weight_scale", Optional[torch.fx.Node]
        )
        assert isinstance(weight_scale_node, torch.fx.Node)
        weight_scale = get_constant(fused_ep, weight_scale_node)
        assert weight_scale is not None
        self.assertEqual(
            weight_scale.dtype,
            torch.float32,
            "weight_scale should stay float32 after folding an fp16 constant",
        )

    def test_mul_folded_quantized_constant(self) -> None:
        """The constant may reach the mul through a quantize_per_tensor node;
        _trace_to_constant recovers the float tensor behind it and folds."""

        builder = ProgramBuilder()
        in_features, out_features = self.IN_FEATURES, self.OUT_FEATURES

        x = builder.placeholder("x", torch.randn(1, in_features))
        w = builder.placeholder(
            "weight",
            torch.randint(-16, 16, (out_features, in_features), dtype=torch.int8),
            input_kind=InputKind.BUFFER,
        )
        b = builder.placeholder(
            "bias", torch.randn(out_features), input_kind=InputKind.BUFFER
        )
        scale_fp = builder.placeholder(
            "scale_fp",
            torch.randn(1).abs().clamp(min=0.5, max=2.0),
            input_kind=InputKind.BUFFER,
        )

        weight_qparams = create_per_channel_qparams(
            builder, out_features, scale=0.05, dtype=torch.float32
        )
        out_qparams = create_per_tensor_qparams(builder, scale=0.5, dtype=torch.int8)

        linear = builder.call_operator(
            op=exir_ops.edge.fused_quant.linear.default,
            args=(
                x,
                w,
                b,
                None,
                None,
                torch.float32,
                0,
                0,
                *weight_qparams,
                None,
                None,
                torch.uint8,
                0,
                0,
                *out_qparams,
            ),
        )

        # The constant operand reaches the mul through a quantize_per_tensor.
        quant_const = builder.call_operator(
            op=exir_ops.edge.quantized_decomposed.quantize_per_tensor.default,
            args=(scale_fp, 0.01, 0, -128, 127, torch.int8),
        )

        mul_inp_qp = create_per_tensor_qparams(builder, scale=0.5, dtype=torch.float32)
        mul_other_qp = create_per_tensor_qparams(
            builder, scale=0.01, dtype=torch.float32
        )
        mul_out_qp = create_per_tensor_qparams(builder, scale=0.125, dtype=torch.int8)

        mul = builder.call_operator(
            op=exir_ops.edge.fused_quant.mul.default,
            args=(linear, quant_const, *mul_inp_qp, *mul_other_qp, *mul_out_qp),
        )
        dequantize = builder.call_operator(
            op=exir_ops.edge.quantized_decomposed.dequantize_per_tensor.default,
            args=(mul, 0.125, 0, -128, 127, torch.int8),
        )
        builder.output([dequantize])

        ep = builder.get_program()
        result = FuseMulIntoLinear().call(ep)
        self.assertTrue(result.modified)
        self._assert_mul_removed(result.exported_program)

    def test_per_channel_mul_folded_through_passthrough(self) -> None:
        ep, _ = self._build_linear_passthrough_mul(
            passthrough="permute",
            constant_shape=(1, 16),
        )

        result = FuseMulIntoLinear().call(ep)

        self.assertTrue(result.modified)
        self._assert_mul_removed(result.exported_program)

    def test_mul_not_folded_when_constant_expands_rank(self) -> None:
        """A scalar (numel 1) constant with more dims than the data broadcasts
        the data up to a higher rank, so it must not fold (folding would drop
        the added unary dimension)."""
        x = torch.randn(1, self.IN_FEATURES)
        weight = torch.randint(
            -16, 16, (self.OUT_FEATURES, self.IN_FEATURES), dtype=torch.int8
        )
        bias = torch.randn(self.OUT_FEATURES)
        # numel == 1 (scalar), but rank 3 vs the rank-2 (1, OUT_FEATURES) linear
        # output: linear[1,16] * const[1,1,1] broadcasts to [1,1,16].
        mul_scale = torch.randn(1, 1, 1).abs().clamp(min=0.5, max=2.0)
        ep = self._build_linear_mul(x, weight, bias, mul_scale)

        result = FuseMulIntoLinear().call(ep)
        self.assertFalse(
            result.modified,
            "Should not fold a constant that broadcasts the data to a higher rank",
        )

    def test_mul_not_folded_wrong_axis_constant(self) -> None:
        """A constant that is shape-preserving but varies along a non-output
        dimension (here dim 0, with batch == out_features so numel matches)
        must not fold."""

        builder = ProgramBuilder()
        # batch == out_features so a [out_features, 1] constant is both
        # shape-preserving and numel-matching, yet varies on the wrong axis.
        in_features = out_features = batch = 16

        x = builder.placeholder("x", torch.randn(batch, in_features))
        w = builder.placeholder(
            "weight",
            torch.randint(-16, 16, (out_features, in_features), dtype=torch.int8),
            input_kind=InputKind.BUFFER,
        )
        scale = builder.placeholder(
            "scale",
            torch.randn(out_features, 1).abs().clamp(min=0.5, max=2.0),
            input_kind=InputKind.BUFFER,
        )

        weight_qparams = create_per_channel_qparams(
            builder, out_features, scale=0.05, dtype=torch.float32
        )
        out_qparams = create_per_tensor_qparams(builder, scale=0.5, dtype=torch.int8)

        linear = builder.call_operator(
            op=exir_ops.edge.fused_quant.linear.default,
            args=(
                x,
                w,
                None,
                None,
                None,
                torch.float32,
                0,
                0,
                *weight_qparams,
                None,
                None,
                torch.uint8,
                0,
                0,
                *out_qparams,
            ),
        )

        mul_inp_qp = create_per_tensor_qparams(builder, scale=0.5, dtype=torch.float32)
        mul_other_qp = create_per_tensor_qparams(
            builder, scale=0.25, dtype=torch.float32
        )
        mul_out_qp = create_per_tensor_qparams(builder, scale=0.125, dtype=torch.int8)

        mul = builder.call_operator(
            op=exir_ops.edge.fused_quant.mul.default,
            args=(linear, scale, *mul_inp_qp, *mul_other_qp, *mul_out_qp),
        )
        dequantize = builder.call_operator(
            op=exir_ops.edge.quantized_decomposed.dequantize_per_tensor.default,
            args=(mul, 0.125, 0, -128, 127, torch.int8),
        )
        builder.output([dequantize])

        ep = builder.get_program()
        result = FuseMulIntoLinear().call(ep)
        self.assertFalse(
            result.modified,
            "Should not fold a constant that varies along a non-output axis",
        )
