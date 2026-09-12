# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

import unittest
from typing import cast

import executorch.backends.cadence.aot.ops_registrations  # noqa: F401

import torch
from executorch.backends.cadence.aot.compiler_funcs import QuantFusionPass
from executorch.backends.cadence.aot.quantizer.pattern_utils import (
    add_constant_placeholder,
    DQ_PER_CHANNEL,
    DQ_PER_TENSOR,
    EXPORTED_PROGRAM_META_KEY,
    get_exported_program,
    get_weight_scale,
    get_weight_zero_point,
    is_per_channel_dq,
    is_weight_dq,
    resolve_constant,
    tensor_qparam_overload,
)
from executorch.backends.cadence.aot.quantizer.patterns import (
    AddmmPattern,
    Conv1dPattern,
    Conv1dReluPattern0,
    LinearPattern,
)
from executorch.backends.cadence.aot.quantizer.utils import quantize_tensor_multiplier
from executorch.backends.test.graph_builder import GraphBuilder
from executorch.backends.test.program_builder import ProgramBuilder
from torch.export.exported_program import ExportedProgram
from torch.export.graph_signature import InputKind

Q_PER_TENSOR: torch._ops.OpOverload = (
    torch.ops.quantized_decomposed.quantize_per_tensor.default
)


def _placeholder_names(ep: ExportedProgram) -> list[str]:
    return [n.name for n in ep.graph.nodes if n.op == "placeholder"]


def _spec_names(ep: ExportedProgram) -> list[str]:
    return [cast(str, s.arg.name) for s in ep.graph_signature.input_specs]


class TestPatternUtils(unittest.TestCase):
    def _build_per_channel_dq_program(
        self,
        scales: torch.Tensor,
        zero_points: torch.Tensor | None,
        axis: int = 0,
    ) -> tuple[ExportedProgram, torch.fx.Node]:
        """A program holding a single per-channel dequantize over lifted constants."""
        out_channels = scales.numel()
        builder = ProgramBuilder()
        w_q = builder.placeholder(
            "w_q",
            torch.randint(-8, 7, (out_channels, 3), dtype=torch.int8),
            input_kind=InputKind.CONSTANT_TENSOR,
        )
        scales_proxy = builder.placeholder(
            "scales", scales, input_kind=InputKind.CONSTANT_TENSOR
        )
        zp_proxy = (
            None
            if zero_points is None
            else builder.placeholder(
                "zero_points", zero_points, input_kind=InputKind.CONSTANT_TENSOR
            )
        )
        dq = builder.call_operator(
            op=DQ_PER_CHANNEL,
            args=(w_q, scales_proxy, zp_proxy, axis, -128, 127, torch.int8),
        )
        builder.output([dq])
        ep = builder.get_program()
        ep.graph_module.meta[EXPORTED_PROGRAM_META_KEY] = ep
        dq_node = ep.graph.find_nodes(op="call_function", target=DQ_PER_CHANNEL)[0]
        return ep, dq_node

    def test_is_weight_dq_accepts_both_granularities(self) -> None:
        builder = GraphBuilder()
        x_q = builder.placeholder("x_q", torch.zeros(4, 3, dtype=torch.int8))
        scales = builder.placeholder("scales", torch.ones(4))
        zps = builder.placeholder("zps", torch.zeros(4, dtype=torch.int64))
        per_tensor = builder.call_operator(
            op=DQ_PER_TENSOR, args=(x_q, 0.1, 0, -128, 127, torch.int8)
        )
        per_channel = builder.call_operator(
            op=DQ_PER_CHANNEL, args=(x_q, scales, zps, 0, -128, 127, torch.int8)
        )
        builder.output([per_tensor, per_channel])
        gm = builder.get_graph_module()

        pt_node = gm.graph.find_nodes(op="call_function", target=DQ_PER_TENSOR)[0]
        pc_node = gm.graph.find_nodes(op="call_function", target=DQ_PER_CHANNEL)[0]

        self.assertTrue(is_weight_dq(pt_node))
        self.assertTrue(is_weight_dq(pc_node))
        self.assertFalse(is_per_channel_dq(pt_node))
        self.assertTrue(is_per_channel_dq(pc_node))

        # Non-node arguments (a plain weight tensor, or a missing bias) are common
        # in the pattern guards, so they must be rejected rather than raise.
        self.assertFalse(is_weight_dq(None))
        self.assertFalse(is_weight_dq(torch.zeros(3)))
        self.assertFalse(is_per_channel_dq(None))

    def test_tensor_qparam_overload_maps_conv_ops(self) -> None:
        for packet in (
            torch.ops.cadence.quantized_conv1d_ncl,
            torch.ops.cadence.quantized_conv1d_nlc,
            torch.ops.cadence.quantized_conv2d_nchw,
            torch.ops.cadence.quantized_conv2d_nhwc,
        ):
            with self.subTest(op=packet):
                self.assertIs(
                    tensor_qparam_overload(packet.per_tensor), packet.default
                )

    def test_tensor_qparam_overload_keeps_operands_aligned(self) -> None:
        """The two overloads must differ only in how the qparams are carried."""
        per_tensor = torch.ops.cadence.quantized_conv1d_ncl.per_tensor
        default = tensor_qparam_overload(per_tensor)
        self.assertEqual(
            [a.name for a in per_tensor._schema.arguments],
            [a.name for a in default._schema.arguments],
        )

    def test_get_weight_scale_per_channel(self) -> None:
        scales = torch.tensor([0.25, 0.5, 0.125, 1.0])
        _, dq_node = self._build_per_channel_dq_program(
            scales, torch.zeros(4, dtype=torch.int64)
        )
        gm = dq_node.graph.owning_module

        resolved = get_weight_scale(gm, dq_node)
        self.assertIsInstance(resolved, torch.Tensor)
        self.assertEqual(cast(torch.Tensor, resolved).dtype, torch.float32)
        torch.testing.assert_close(cast(torch.Tensor, resolved), scales)

    def test_get_weight_zero_point_per_channel(self) -> None:
        zps = torch.tensor([1, -2, 3, 0], dtype=torch.int64)
        _, dq_node = self._build_per_channel_dq_program(torch.ones(4), zps)
        gm = dq_node.graph.owning_module

        resolved = get_weight_zero_point(gm, dq_node)
        self.assertIsInstance(resolved, torch.Tensor)
        self.assertEqual(cast(torch.Tensor, resolved).dtype, torch.int32)
        torch.testing.assert_close(
            cast(torch.Tensor, resolved), zps.to(torch.int32)
        )

    def test_get_weight_zero_point_defaults_to_zeros_when_symmetric(self) -> None:
        """Symmetric per-channel leaves zero_points unset; it must read as zeros."""
        _, dq_node = self._build_per_channel_dq_program(torch.ones(4), None)
        gm = dq_node.graph.owning_module

        resolved = get_weight_zero_point(gm, dq_node)
        self.assertIsInstance(resolved, torch.Tensor)
        torch.testing.assert_close(
            cast(torch.Tensor, resolved), torch.zeros(4, dtype=torch.int32)
        )

    def test_get_weight_scale_rejects_non_output_channel_axis(self) -> None:
        """Cadence kernels index qparams by output channel, so axis must be 0."""
        _, dq_node = self._build_per_channel_dq_program(
            torch.ones(3), torch.zeros(3, dtype=torch.int64), axis=1
        )
        gm = dq_node.graph.owning_module

        with self.assertRaisesRegex(AssertionError, "output-channel axis"):
            get_weight_scale(gm, dq_node)

    def _build_program_with_user_input(self) -> tuple[ExportedProgram, torch.fx.Node]:
        builder = ProgramBuilder()
        w = builder.placeholder(
            "w", torch.randn(4, 3), input_kind=InputKind.CONSTANT_TENSOR
        )
        x = builder.placeholder("x", torch.randn(2, 3))
        out = builder.call_operator(op=torch.ops.aten.linear.default, args=(x, w))
        builder.output([out])
        ep = builder.get_program()
        ep.graph_module.meta[EXPORTED_PROGRAM_META_KEY] = ep
        node = ep.graph.find_nodes(
            op="call_function", target=torch.ops.aten.linear.default
        )[0]
        return ep, node

    def test_add_constant_placeholder_keeps_graph_and_signature_aligned(self) -> None:
        """Placeholder order and input_spec order have to stay in lockstep.

        The runtime feeds inputs positionally, so a constant appended to the specs
        but inserted elsewhere in the graph silently shifts every later input.
        """
        ep, anchor = self._build_program_with_user_input()
        tensor = torch.tensor([1, 2, 3, 4], dtype=torch.int32)

        node = add_constant_placeholder(ep.graph_module, tensor, anchor, "out_shift")

        self.assertEqual(_placeholder_names(ep), _spec_names(ep))
        self.assertIn(node.name, _placeholder_names(ep))

        spec = next(
            s for s in ep.graph_signature.input_specs if s.arg.name == node.name
        )
        self.assertEqual(spec.kind, InputKind.CONSTANT_TENSOR)
        self.assertIn(spec.target, ep.constants)
        torch.testing.assert_close(ep.constants[spec.target], tensor)

    def test_add_constant_placeholder_precedes_user_inputs(self) -> None:
        """Constants must be lifted ahead of user inputs, as export emits them."""
        ep, anchor = self._build_program_with_user_input()

        node = add_constant_placeholder(
            ep.graph_module,
            torch.tensor([7], dtype=torch.int32),
            anchor,
            "out_multiplier",
        )

        names = _placeholder_names(ep)
        user_inputs = set(ep.graph_signature.user_inputs)
        first_user_idx = min(i for i, n in enumerate(names) if n in user_inputs)
        self.assertLess(names.index(node.name), first_user_idx)

    def test_add_constant_placeholder_names_are_unique(self) -> None:
        ep, anchor = self._build_program_with_user_input()

        first = add_constant_placeholder(
            ep.graph_module, torch.tensor([1], dtype=torch.int32), anchor, "out_shift"
        )
        second = add_constant_placeholder(
            ep.graph_module, torch.tensor([2], dtype=torch.int32), anchor, "out_shift"
        )

        self.assertNotEqual(first.name, second.name)
        self.assertEqual(_placeholder_names(ep), _spec_names(ep))
        self.assertEqual(len(set(_placeholder_names(ep))), len(_placeholder_names(ep)))

    def test_linear_pattern_fuses_per_channel_weights(self) -> None:
        """Per-channel weights on linear reach the tensor-qparam overload.

        Linear is the shape that matters for fully-connected models, so it has to
        accept a per-channel weight dequantize rather than decline and leave an
        unfused dq in the graph.
        """
        out_features, in_features = 4, 3
        scales = torch.tensor([0.1, 0.2, 0.05, 0.4])
        builder = ProgramBuilder()
        x_q = builder.placeholder(
            "x_q", torch.randint(-8, 8, (2, in_features), dtype=torch.int8)
        )
        w_q = builder.placeholder(
            "w_q",
            torch.randint(-8, 8, (out_features, in_features), dtype=torch.int8),
            input_kind=InputKind.CONSTANT_TENSOR,
        )
        scales_proxy = builder.placeholder(
            "scales", scales, input_kind=InputKind.CONSTANT_TENSOR
        )
        dq_input = builder.call_operator(
            op=DQ_PER_TENSOR, args=(x_q, 0.1, 0, -128, 127, torch.int8)
        )
        dq_weight = builder.call_operator(
            op=DQ_PER_CHANNEL,
            args=(w_q, scales_proxy, None, 0, -128, 127, torch.int8),
        )
        linear = builder.call_operator(
            op=torch.ops.aten.linear.default, args=(dq_input, dq_weight)
        )
        q = builder.call_operator(
            op=Q_PER_TENSOR, args=(linear, 0.2, 0, -128, 127, torch.int8)
        )
        builder.output([q])
        ep = builder.get_program()
        ep.graph_module.meta[EXPORTED_PROGRAM_META_KEY] = ep

        linear_node = ep.graph.find_nodes(
            op="call_function", target=torch.ops.aten.linear.default
        )[0]
        fused = LinearPattern().fuse(ep.graph_module, linear_node)

        self.assertIsNotNone(fused)
        fused = cast(torch.fx.Node, fused)
        self.assertIs(fused.target, torch.ops.cadence.quantized_linear.default)
        for name in ("weight_zero_point", "out_multiplier", "out_shift"):
            arg = fused.kwargs[name]
            self.assertIsInstance(arg, torch.fx.Node, f"{name} should be lifted")
            tensor = resolve_constant(ep.graph_module, arg)
            self.assertIsNotNone(tensor)
            self.assertEqual(
                cast(torch.Tensor, tensor).numel(),
                out_features,
                f"{name} should have one entry per output channel",
            )

    def test_linear_pattern_keeps_scalar_overload_for_per_tensor(self) -> None:
        builder = GraphBuilder()
        x_q = builder.placeholder("x_q", torch.zeros(2, 3, dtype=torch.int8))
        w_q = builder.placeholder("w_q", torch.zeros(4, 3, dtype=torch.int8))
        dq_input = builder.call_operator(
            op=DQ_PER_TENSOR, args=(x_q, 0.1, 0, -128, 127, torch.int8)
        )
        dq_weight = builder.call_operator(
            op=DQ_PER_TENSOR, args=(w_q, 0.05, 0, -128, 127, torch.int8)
        )
        linear = builder.call_operator(
            op=torch.ops.aten.linear.default, args=(dq_input, dq_weight)
        )
        q = builder.call_operator(
            op=Q_PER_TENSOR, args=(linear, 0.2, 0, -128, 127, torch.int8)
        )
        builder.output([q])
        gm = builder.get_graph_module()

        linear_node = gm.graph.find_nodes(
            op="call_function", target=torch.ops.aten.linear.default
        )[0]
        fused = LinearPattern().fuse(gm, linear_node)

        self.assertIsNotNone(fused)
        self.assertIs(
            cast(torch.fx.Node, fused).target,
            torch.ops.cadence.quantized_linear.per_tensor,
        )


class _RecordingPattern:
    """Minimal stand-in for a QuantizationPattern that never fuses."""

    def __init__(self, boom: bool = False) -> None:
        self.seen: list[object] = []
        self.boom = boom

    def anchor_ops(self) -> list[torch._ops.OpOverload]:
        return [torch.ops.aten.linear.default]

    def fuse(self, graph_module: torch.fx.GraphModule, node: torch.fx.Node) -> None:
        self.seen.append(get_exported_program(graph_module))
        if self.boom:
            raise RuntimeError("pattern blew up")
        return None


class TestQuantFusionPassProgramStash(unittest.TestCase):
    """QuantFusionPass hands the ExportedProgram to patterns via graph meta.

    Per-channel fusion needs the program to read weight scales and register new
    constants, but PassBase.call only gets a GraphModule. The stash makes that
    reachable without changing the fuse() signature of every pattern, so it has to
    be scoped strictly to the pass.
    """

    def _build_program(self) -> ExportedProgram:
        builder = ProgramBuilder()
        w = builder.placeholder(
            "w", torch.randn(4, 3), input_kind=InputKind.CONSTANT_TENSOR
        )
        x = builder.placeholder("x", torch.randn(2, 3))
        out = builder.call_operator(op=torch.ops.aten.linear.default, args=(x, w))
        builder.output([out])
        return builder.get_program()

    def test_program_is_visible_to_patterns(self) -> None:
        ep = self._build_program()
        pattern = _RecordingPattern()

        QuantFusionPass([pattern], ep).call(ep.graph_module)

        self.assertEqual(pattern.seen, [ep])

    def test_program_is_not_left_behind_in_meta(self) -> None:
        """Leaving the stash set would keep the whole program alive in graph meta."""
        ep = self._build_program()

        QuantFusionPass([_RecordingPattern()], ep).call(ep.graph_module)

        self.assertNotIn(EXPORTED_PROGRAM_META_KEY, ep.graph_module.meta)
        self.assertIsNone(get_exported_program(ep.graph_module))

    def test_stash_is_cleared_when_a_pattern_raises(self) -> None:
        ep = self._build_program()

        with self.assertRaisesRegex(RuntimeError, "pattern blew up"):
            QuantFusionPass([_RecordingPattern(boom=True)], ep).call(ep.graph_module)

        self.assertNotIn(EXPORTED_PROGRAM_META_KEY, ep.graph_module.meta)

    def test_pass_without_a_program_leaves_meta_untouched(self) -> None:
        """Per-tensor fusion is unchanged: no program, no stash."""
        ep = self._build_program()
        pattern = _RecordingPattern()

        QuantFusionPass([pattern]).call(ep.graph_module)

        self.assertEqual(pattern.seen, [None])
        self.assertNotIn(EXPORTED_PROGRAM_META_KEY, ep.graph_module.meta)


class TestFuseConvPerChannel(unittest.TestCase):
    """`fuse_conv` is where per-channel weights become quantized conv qparams.

    The arithmetic here (bias_scale, and its Q31 decomposition into
    out_multiplier/out_shift) is the mathematical core of per-channel support,
    so it is checked directly rather than only through a lowered model.
    """

    INPUT_SCALE = 0.02
    INPUT_ZERO_POINT = -3
    OUT_SCALE = 0.5
    OUT_ZERO_POINT = 7

    def _build_conv1d_program(
        self,
        weight_scales: torch.Tensor,
        weight_zero_points: torch.Tensor | None = None,
        in_channels: int = 4,
        groups: int = 1,
        with_bias: bool = False,
        length: int = 6,
        kernel_size: int = 3,
    ) -> tuple[ExportedProgram, torch.fx.Node]:
        """A dq(per-channel weight) -> conv1d -> q program ready for fusion."""
        out_channels = weight_scales.numel()
        builder = ProgramBuilder()
        x_q = builder.placeholder(
            "x_q",
            torch.randint(-128, 127, (1, in_channels, length), dtype=torch.int8),
        )
        w_q = builder.placeholder(
            "w_q",
            torch.randint(
                -127, 127, (out_channels, in_channels // groups, kernel_size),
                dtype=torch.int8,
            ),
            input_kind=InputKind.CONSTANT_TENSOR,
        )
        scales = builder.placeholder(
            "scales", weight_scales, input_kind=InputKind.CONSTANT_TENSOR
        )
        zps = (
            None
            if weight_zero_points is None
            else builder.placeholder(
                "zero_points",
                weight_zero_points,
                input_kind=InputKind.CONSTANT_TENSOR,
            )
        )

        dq_input = builder.call_operator(
            op=DQ_PER_TENSOR,
            args=(x_q, self.INPUT_SCALE, self.INPUT_ZERO_POINT, -128, 127, torch.int8),
        )
        dq_weight = builder.call_operator(
            op=DQ_PER_CHANNEL,
            args=(w_q, scales, zps, 0, -128, 127, torch.int8),
        )

        conv_args: tuple[object, ...] = (dq_input, dq_weight)
        if with_bias:
            b_q = builder.placeholder(
                "b_q",
                torch.randint(-64, 64, (out_channels,), dtype=torch.int32),
                input_kind=InputKind.CONSTANT_TENSOR,
            )
            bias_scales = builder.placeholder(
                "bias_scales",
                weight_scales * self.INPUT_SCALE,
                input_kind=InputKind.CONSTANT_TENSOR,
            )
            dq_bias = builder.call_operator(
                op=DQ_PER_CHANNEL,
                args=(b_q, bias_scales, None, 0, -(2**31), 2**31 - 1, torch.int32),
            )
            conv_args = conv_args + (dq_bias,)
        else:
            conv_args = conv_args + (None,)
        if groups != 1:
            conv_args = conv_args + ((1,), (0,), (1,), groups)

        conv = builder.call_operator(
            op=torch.ops.aten.conv1d.default,
            args=conv_args,
        )
        q = builder.call_operator(
            op=Q_PER_TENSOR,
            args=(conv, self.OUT_SCALE, self.OUT_ZERO_POINT, -128, 127, torch.int8),
        )
        builder.output([q])
        ep = builder.get_program()
        ep.graph_module.meta[EXPORTED_PROGRAM_META_KEY] = ep
        conv_node = ep.graph.find_nodes(
            op="call_function", target=torch.ops.aten.conv1d.default
        )[0]
        return ep, conv_node

    def _fuse(self, ep: ExportedProgram, conv_node: torch.fx.Node) -> torch.fx.Node:
        fused = Conv1dPattern().fuse(ep.graph_module, conv_node)
        self.assertIsNotNone(fused, "per-channel conv should fuse")
        return cast(torch.fx.Node, fused)

    def _kwarg_tensor(
        self, ep: ExportedProgram, node: torch.fx.Node, name: str
    ) -> torch.Tensor:
        arg = node.kwargs[name]
        self.assertIsInstance(
            arg, torch.fx.Node, f"{name} must be lifted as a constant node"
        )
        tensor = resolve_constant(ep.graph_module, arg)
        self.assertIsNotNone(tensor, f"{name} constant should resolve")
        return cast(torch.Tensor, tensor)

    def test_emits_tensor_qparam_overload(self) -> None:
        scales = torch.tensor([0.1, 0.2, 0.05])
        ep, conv_node = self._build_conv1d_program(scales)

        fused = self._fuse(ep, conv_node)

        self.assertIs(fused.target, torch.ops.cadence.quantized_conv1d_ncl.default)

    def test_per_tensor_weights_keep_the_scalar_overload(self) -> None:
        """The per-channel path must not capture ordinary per-tensor convs."""
        builder = ProgramBuilder()
        x_q = builder.placeholder(
            "x_q", torch.randint(-128, 127, (1, 4, 6), dtype=torch.int8)
        )
        w_q = builder.placeholder(
            "w_q",
            torch.randint(-127, 127, (3, 4, 3), dtype=torch.int8),
            input_kind=InputKind.CONSTANT_TENSOR,
        )
        dq_input = builder.call_operator(
            op=DQ_PER_TENSOR, args=(x_q, 0.02, -3, -128, 127, torch.int8)
        )
        dq_weight = builder.call_operator(
            op=DQ_PER_TENSOR, args=(w_q, 0.1, 0, -128, 127, torch.int8)
        )
        conv = builder.call_operator(
            op=torch.ops.aten.conv1d.default, args=(dq_input, dq_weight)
        )
        q = builder.call_operator(
            op=Q_PER_TENSOR, args=(conv, 0.5, 7, -128, 127, torch.int8)
        )
        builder.output([q])
        ep = builder.get_program()
        ep.graph_module.meta[EXPORTED_PROGRAM_META_KEY] = ep
        conv_node = ep.graph.find_nodes(
            op="call_function", target=torch.ops.aten.conv1d.default
        )[0]

        fused = self._fuse(ep, conv_node)

        self.assertIs(fused.target, torch.ops.cadence.quantized_conv1d_ncl.per_tensor)
        self.assertIsInstance(fused.kwargs["out_multiplier"], int)

    def test_bias_scale_is_input_scale_times_weight_scale(self) -> None:
        scales = torch.tensor([0.1, 0.2, 0.05, 0.4])
        ep, conv_node = self._build_conv1d_program(scales)

        fused = self._fuse(ep, conv_node)

        bias_scale = self._kwarg_tensor(ep, fused, "bias_scale")
        torch.testing.assert_close(
            bias_scale, (scales * self.INPUT_SCALE).to(torch.float32)
        )

    def test_multiplier_and_shift_reconstruct_the_requantize_scale(self) -> None:
        """out_multiplier/out_shift are a Q31 encoding of bias_scale / out_scale.

        Checked per channel, so a scalar that happened to be right for channel 0
        would not pass.
        """
        scales = torch.tensor([0.1, 0.2, 0.05, 0.4])
        ep, conv_node = self._build_conv1d_program(scales)

        fused = self._fuse(ep, conv_node)

        out_multiplier = self._kwarg_tensor(ep, fused, "out_multiplier")
        out_shift = self._kwarg_tensor(ep, fused, "out_shift")
        self.assertEqual(out_multiplier.dtype, torch.int32)
        self.assertEqual(out_shift.dtype, torch.int32)
        self.assertEqual(out_multiplier.shape, scales.shape)
        self.assertEqual(out_shift.shape, scales.shape)

        expected = (scales * self.INPUT_SCALE / self.OUT_SCALE).to(torch.float64)
        reconstructed = (
            out_multiplier.to(torch.float64)
            / (2**31)
            * torch.pow(2.0, out_shift.to(torch.float64))
        )
        torch.testing.assert_close(reconstructed, expected, rtol=1e-4, atol=1e-4)

    def test_symmetric_weights_give_zero_zero_points(self) -> None:
        """Symmetric per-channel quantization leaves zero_points unset."""
        scales = torch.tensor([0.1, 0.2, 0.05])
        ep, conv_node = self._build_conv1d_program(scales, weight_zero_points=None)

        fused = self._fuse(ep, conv_node)

        wzp = self._kwarg_tensor(ep, fused, "weight_zero_point")
        self.assertEqual(wzp.shape, scales.shape)
        self.assertTrue(torch.all(wzp == 0), f"expected zeros, got {wzp}")

    def test_affine_weight_zero_points_are_carried_through(self) -> None:
        scales = torch.tensor([0.1, 0.2, 0.05])
        zps = torch.tensor([1, -2, 3], dtype=torch.int32)
        ep, conv_node = self._build_conv1d_program(scales, weight_zero_points=zps)

        fused = self._fuse(ep, conv_node)

        wzp = self._kwarg_tensor(ep, fused, "weight_zero_point")
        self.assertTrue(torch.equal(wzp.to(torch.int32), zps))

    def test_missing_bias_becomes_a_zero_int32_bias(self) -> None:
        scales = torch.tensor([0.1, 0.2, 0.05])
        ep, conv_node = self._build_conv1d_program(scales, with_bias=False)

        fused = self._fuse(ep, conv_node)

        bias = fused.args[2]
        self.assertIsInstance(bias, torch.fx.Node)
        bias_val = cast(torch.fx.Node, bias).meta["val"]
        self.assertEqual(bias_val.dtype, torch.int32)
        self.assertEqual(tuple(bias_val.shape), (scales.numel(),))

    def test_depthwise_per_channel_routes_to_the_depthwise_op(self) -> None:
        """Depthwise selection has to survive the per-channel overload swap.

        The granularity swap replaces the op, so if depthwise is decided after
        it, a per-channel depthwise conv silently stays a dense conv.
        """
        channels = 4
        scales = torch.rand(channels) * 0.1 + 0.01
        ep, conv_node = self._build_conv1d_program(
            scales, in_channels=channels, groups=channels
        )

        fused = self._fuse(ep, conv_node)

        self.assertIs(
            fused.target,
            torch.ops.cadence.quantized_depthwise_conv1d_ncl.default,
        )

    def test_depthwise_per_tensor_still_routes_to_the_depthwise_op(self) -> None:
        channels = 4
        builder = ProgramBuilder()
        x_q = builder.placeholder(
            "x_q", torch.randint(-128, 127, (1, channels, 6), dtype=torch.int8)
        )
        w_q = builder.placeholder(
            "w_q",
            torch.randint(-127, 127, (channels, 1, 3), dtype=torch.int8),
            input_kind=InputKind.CONSTANT_TENSOR,
        )
        dq_input = builder.call_operator(
            op=DQ_PER_TENSOR, args=(x_q, 0.02, -3, -128, 127, torch.int8)
        )
        dq_weight = builder.call_operator(
            op=DQ_PER_TENSOR, args=(w_q, 0.1, 0, -128, 127, torch.int8)
        )
        conv = builder.call_operator(
            op=torch.ops.aten.conv1d.default,
            args=(dq_input, dq_weight, None, (1,), (0,), (1,), channels),
        )
        q = builder.call_operator(
            op=Q_PER_TENSOR, args=(conv, 0.5, 7, -128, 127, torch.int8)
        )
        builder.output([q])
        ep = builder.get_program()
        ep.graph_module.meta[EXPORTED_PROGRAM_META_KEY] = ep
        conv_node = ep.graph.find_nodes(
            op="call_function", target=torch.ops.aten.conv1d.default
        )[0]

        fused = self._fuse(ep, conv_node)

        self.assertIs(
            fused.target,
            torch.ops.cadence.quantized_depthwise_conv1d_ncl.per_tensor,
        )


class TestPerChannelPatternRouting(unittest.TestCase):
    """Which patterns are wired to accept per-channel weights and which aren't.

    ``fuse_conv`` and ``fuse_linear`` can absorb a per-channel weight dequant.
    ``fuse_matmul`` and the mixed w8a32 paths cannot; their patterns must still
    decline so the model falls back to float rather than raising a ``KeyError``
    when a scalar ``scale`` arg is missing. AddmmPattern is intentionally in the
    declining set today: its fuse guards on ``DQ_PER_TENSOR`` directly, and
    per-channel addmm currently falls back to float.
    """

    def _build_conv1d_relu_per_channel(
        self, out_channels: int = 3, in_channels: int = 2, kernel_size: int = 3
    ) -> tuple[ExportedProgram, torch.fx.Node]:
        # Distinct per-channel scales so a broadcast-of-channel-0 bug is visible.
        weight_scales = torch.tensor([0.1, 0.25, 0.05])[:out_channels]

        builder = ProgramBuilder()
        x_q = builder.placeholder(
            "x_q",
            torch.randint(-16, 16, (1, in_channels, 6), dtype=torch.int8),
        )
        w_q = builder.placeholder(
            "w_q",
            torch.randint(
                -16, 16, (out_channels, in_channels, kernel_size), dtype=torch.int8
            ),
            input_kind=InputKind.CONSTANT_TENSOR,
        )
        scales = builder.placeholder(
            "scales", weight_scales, input_kind=InputKind.CONSTANT_TENSOR
        )
        dq_input = builder.call_operator(
            op=DQ_PER_TENSOR, args=(x_q, 0.02, -3, -128, 127, torch.int8)
        )
        dq_weight = builder.call_operator(
            op=DQ_PER_CHANNEL,
            args=(w_q, scales, None, 0, -128, 127, torch.int8),
        )
        conv = builder.call_operator(
            op=torch.ops.aten.conv1d.default, args=(dq_input, dq_weight)
        )
        relu = builder.call_operator(op=torch.ops.aten.relu.default, args=(conv,))
        q = builder.call_operator(
            op=Q_PER_TENSOR, args=(relu, 0.5, 7, -128, 127, torch.int8)
        )
        builder.output([q])
        ep = builder.get_program()
        ep.graph_module.meta[EXPORTED_PROGRAM_META_KEY] = ep
        conv_node = ep.graph.find_nodes(
            op="call_function", target=torch.ops.aten.conv1d.default
        )[0]
        return ep, conv_node

    def test_conv_relu_pattern_fuses_per_channel_weights(self) -> None:
        """ConvReluBase.fuse takes the per-channel path via is_weight_dq."""
        ep, conv_node = self._build_conv1d_relu_per_channel()

        fused = Conv1dReluPattern0().fuse(ep.graph_module, conv_node)

        self.assertIsNotNone(fused, "conv_relu should fuse a per-channel weight")
        fused = cast(torch.fx.Node, fused)
        self.assertIs(fused.target, torch.ops.cadence.quantized_conv1d_ncl.default)
        for name in ("weight_zero_point", "out_multiplier", "out_shift", "bias_scale"):
            arg = fused.kwargs[name]
            self.assertIsInstance(
                arg, torch.fx.Node, f"{name} should be a lifted constant"
            )
            resolved = resolve_constant(ep.graph_module, arg)
            self.assertIsNotNone(resolved)
            # 3 output channels declared above.
            self.assertEqual(cast(torch.Tensor, resolved).numel(), 3)

    def test_addmm_pattern_declines_per_channel_weight(self) -> None:
        """AddmmPattern intentionally does not carry per-channel; it falls back.

        If this ever changes silently the pattern will start reading the missing
        scalar ``scale`` arg through ``fuse_linear`` and blow up.
        """
        builder = ProgramBuilder()
        bias_q = builder.placeholder(
            "bias_q", torch.zeros(3, dtype=torch.int32),
            input_kind=InputKind.CONSTANT_TENSOR,
        )
        x_q = builder.placeholder(
            "x_q", torch.randint(-8, 8, (2, 4), dtype=torch.int8)
        )
        w_q = builder.placeholder(
            "w_q",
            torch.randint(-8, 8, (4, 3), dtype=torch.int8),
            input_kind=InputKind.CONSTANT_TENSOR,
        )
        scales = builder.placeholder(
            "scales",
            torch.tensor([0.1, 0.25, 0.05]),
            input_kind=InputKind.CONSTANT_TENSOR,
        )
        dq_bias = builder.call_operator(
            op=DQ_PER_TENSOR, args=(bias_q, 0.02, 0, -(2**31), 2**31 - 1, torch.int32)
        )
        dq_input = builder.call_operator(
            op=DQ_PER_TENSOR, args=(x_q, 0.1, 0, -128, 127, torch.int8)
        )
        dq_weight = builder.call_operator(
            op=DQ_PER_CHANNEL,
            args=(w_q, scales, None, 0, -128, 127, torch.int8),
        )
        addmm = builder.call_operator(
            op=torch.ops.aten.addmm.default, args=(dq_bias, dq_input, dq_weight)
        )
        q = builder.call_operator(
            op=Q_PER_TENSOR, args=(addmm, 0.2, 0, -128, 127, torch.int8)
        )
        builder.output([q])
        ep = builder.get_program()
        ep.graph_module.meta[EXPORTED_PROGRAM_META_KEY] = ep
        addmm_node = ep.graph.find_nodes(
            op="call_function", target=torch.ops.aten.addmm.default
        )[0]

        result = AddmmPattern().fuse(ep.graph_module, addmm_node)

        self.assertIsNone(
            result, "AddmmPattern must decline per-channel to keep the float fallback"
        )
        # And the graph should still contain the dq nodes untouched.
        self.assertEqual(
            len(ep.graph.find_nodes(op="call_function", target=DQ_PER_CHANNEL)),
            1,
        )


class TestFuseConvPerChannelRequantEncoding(unittest.TestCase):
    """The Q31 encoding of the per-channel requant scale must be exact per channel.

    The existing TestFuseConvPerChannel checks the scale reconstruction round
    trip; this class checks the encoding directly against
    ``quantize_tensor_multiplier`` (which is what the runtime uses to interpret
    the multiplier/shift pair), so a per-channel multiplier that happened to
    reconstruct close to the right float value but was actually built from the
    wrong index would be caught.
    """

    def test_encoded_multiplier_shift_matches_reference(self) -> None:
        weight_scales = torch.tensor([0.11, 0.007, 0.34, 0.05])
        input_scale = 0.02
        out_scale = 0.31

        builder = ProgramBuilder()
        x_q = builder.placeholder(
            "x_q", torch.randint(-8, 8, (1, 3, 6), dtype=torch.int8)
        )
        w_q = builder.placeholder(
            "w_q",
            torch.randint(-8, 8, (4, 3, 3), dtype=torch.int8),
            input_kind=InputKind.CONSTANT_TENSOR,
        )
        scales = builder.placeholder(
            "scales", weight_scales, input_kind=InputKind.CONSTANT_TENSOR
        )
        dq_input = builder.call_operator(
            op=DQ_PER_TENSOR,
            args=(x_q, input_scale, 0, -128, 127, torch.int8),
        )
        dq_weight = builder.call_operator(
            op=DQ_PER_CHANNEL, args=(w_q, scales, None, 0, -128, 127, torch.int8)
        )
        conv = builder.call_operator(
            op=torch.ops.aten.conv1d.default, args=(dq_input, dq_weight)
        )
        q = builder.call_operator(
            op=Q_PER_TENSOR, args=(conv, out_scale, 0, -128, 127, torch.int8)
        )
        builder.output([q])
        ep = builder.get_program()
        ep.graph_module.meta[EXPORTED_PROGRAM_META_KEY] = ep
        conv_node = ep.graph.find_nodes(
            op="call_function", target=torch.ops.aten.conv1d.default
        )[0]

        fused = Conv1dPattern().fuse(ep.graph_module, conv_node)
        self.assertIsNotNone(fused)
        fused = cast(torch.fx.Node, fused)

        out_multiplier = cast(
            torch.Tensor,
            resolve_constant(ep.graph_module, fused.kwargs["out_multiplier"]),
        )
        out_shift = cast(
            torch.Tensor,
            resolve_constant(ep.graph_module, fused.kwargs["out_shift"]),
        )
        expected_mult, expected_shift = quantize_tensor_multiplier(
            weight_scales * input_scale / out_scale
        )
        self.assertTrue(
            torch.equal(out_multiplier, expected_mult.to(torch.int32)),
            f"out_multiplier per channel: got {out_multiplier}, "
            f"expected {expected_mult.to(torch.int32)}",
        )
        self.assertTrue(
            torch.equal(out_shift, expected_shift.to(torch.int32)),
            f"out_shift per channel: got {out_shift}, "
            f"expected {expected_shift.to(torch.int32)}",
        )
