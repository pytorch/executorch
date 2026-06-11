# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
# Copyright 2025-2026 Arm Limited and/or its affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import math
from typing import cast, Optional

import cmsis_nn  # type: ignore[import-not-found, import-untyped]
import executorch.backends.cortex_m.ops.operators  # noqa
import executorch.exir as exir
import torch
import torch.fx
from executorch.backends.arm._passes.arm_pass_utils import get_first_fake_tensor

from executorch.backends.cortex_m.passes.passes_utils import (
    build_activation_lut,
    quantize_multiplier_aot,
    quantize_val,
    SHIFT_INT8,
    to_physical_order,
)
from executorch.backends.cortex_m.passes.scratch_buffer_sizes import (
    required_cmsis_nn_buffer_sizes,
)
from executorch.backends.cortex_m.quantizer.quantization_configs import (
    CMSIS_SOFTMAX_SCALE,
    CMSIS_SOFTMAX_ZERO_POINT,
)
from executorch.backends.cortex_m.target_config import CortexMTargetConfig
from executorch.backends.transforms.aten_to_dialect_pass import (
    AtenToDialectPass,
    DialectNodeSpec,
)
from executorch.backends.transforms.utils import (
    create_constant_placeholder,
    get_param_tensor,
    is_param_node,
)
from executorch.exir.dialects._ops import ops as exir_ops
from executorch.exir.passes import make_alloc_node
from torch._subclasses.fake_tensor import FakeTensorMode
from torch.export import ExportedProgram
from torch.export.graph_signature import InputKind
from torch.fx import Node
from torch.fx.node import Argument
from torch.fx.passes.infra.pass_manager import PassResult


class AtenToCortexMPass(AtenToDialectPass):
    """
    Cortex-M backend pass for replacing supported quantized kernels with Cortex-M
    accelerated kernels.
    """

    def __init__(
        self,
        exported_program: ExportedProgram,
        target_config: CortexMTargetConfig,
    ) -> None:
        super().__init__(exported_program=exported_program)
        self.target_config = target_config

    def call(self, graph_module: torch.fx.GraphModule) -> PassResult:
        result = super().call(graph_module)

        for node in result.graph_module.graph.nodes:
            self._initialize_alloc_node_size(node)

        return result

    def _initialize_alloc_node_size(self, node: torch.fx.Node) -> None:
        """Initialize trailing scratch alloc nodes for CMSIS-NN kernels."""
        scratch_buffer_sizes = required_cmsis_nn_buffer_sizes(
            node, self.target_config.backend
        )
        if scratch_buffer_sizes is None:
            return

        for i, scratch_buffer_size in enumerate(reversed(scratch_buffer_sizes)):
            scratch_arg = node.args[-(i + 1)]
            if (
                not isinstance(scratch_arg, torch.fx.Node)
                or scratch_arg.target != exir.memory.alloc
            ):
                raise RuntimeError(
                    f"Expected scratch alloc node as final argument(s) for {node.target}, got {scratch_arg}."
                )

            scratch_arg.args = (((scratch_buffer_size,), torch.uint8),)
            scratch_arg.meta["val"] = torch.empty(
                (scratch_buffer_size,), dtype=torch.uint8, device="meta"
            )


def _create_uninitialized_alloc_node(
    node: Node, exported_program: ExportedProgram
) -> Node:
    with FakeTensorMode() as mode:
        with node.graph.inserting_before(node):
            return make_alloc_node(
                exported_program.graph_module,
                mode.from_tensor(torch.empty(0)),
                None,
            )


_SOFTMAX_INPUT_INTEGER_BITS = 5

def _to_int_pair(
    value: Argument, default: Optional[tuple[int, int]]
) -> tuple[int, int]:
    if value is None:
        assert default is not None, "Expected default sequence for normalization"
        return (default[0], default[1])

    try:
        int_pair = cast(tuple[int, int], value)
        return int_pair
    except Exception as exc:
        raise ValueError(f"Expected a tuple of two integers, got {value}") from exc


def _to_bool(value: Argument, default: bool) -> bool:
    if value is None:
        return default
    try:
        bool_value = cast(bool, value)
        return bool_value
    except Exception as exc:
        raise ValueError(f"Expected a boolean value, got {value}") from exc


def _is_quant_per_tensor_qualified(node: Node) -> bool:
    """Match int8 OR int16 (de)quantize_per_tensor nodes."""
    dtype = node.args[5]
    if dtype == torch.int8:
        return (
            cast(int, node.args[3]) >= torch.iinfo(torch.int8).min
            and cast(int, node.args[4]) <= torch.iinfo(torch.int8).max
        )
    if dtype == torch.int16:
        return (
            cast(int, node.args[3]) >= torch.iinfo(torch.int16).min
            and cast(int, node.args[4]) <= torch.iinfo(torch.int16).max
        )
    return False


def _compute_softmax_params(input_scale: float) -> tuple[int, int, int]:
    """
    Convert per-tensor input scale into fixed-point params for arm_softmax_s8.
    """
    real_multiplier = min(
        input_scale * (1 << (31 - _SOFTMAX_INPUT_INTEGER_BITS)),
        float((1 << 31) - 1),
    )
    input_multiplier, input_shift = quantize_multiplier_aot(real_multiplier)
    diff_min_term = (
        ((1 << _SOFTMAX_INPUT_INTEGER_BITS) - 1)
        * math.ldexp(1.0, 31 - _SOFTMAX_INPUT_INTEGER_BITS)
        / math.ldexp(1.0, input_shift)
    )
    diff_min = -int(math.floor(diff_min_term))
    return int(input_multiplier), int(input_shift), diff_min


def _get_input_tensor_data(node: Node, arg_index: int = 0):
    arg = node.args[arg_index]
    if isinstance(arg, Node) and "val" in arg.meta:
        return get_first_fake_tensor(arg)
    if "val" in node.meta:
        return get_first_fake_tensor(node)
    raise KeyError(
        f"Expected fake tensor metadata on input arg {arg_index} or node {node.name}."
    )


def _compute_kernel_sum(weights, bias, input_offset, weight_offset):
    """
    Computes the precomputed kernel sum term (bias optional)
        a * sum_j(wij + b) + ci

    for i = (1, ..., n), where j indexes the input activations.
    """
    weights_transposed = weights.T
    weights_int32 = weights_transposed.to(torch.int32)
    offset_weights = weights_int32 + weight_offset
    kernel_sum = torch.sum(offset_weights, dim=0, keepdim=True, dtype=torch.int32)
    kernel_sum_offset = kernel_sum * input_offset

    if bias is not None:
        kernel_sum_offset += bias

    return kernel_sum_offset


def _get_batch_size_from_conv(conv_node: torch.fx.Node):
    """
    Extract batch size from convolution node's output shape.

    Returns None if shape metadata is unavailable, which can occur when
    processing nodes created earlier in the same pass iteration.

    For Conv2d operations, output_batch_size always equals input_batch_size.
    Conv2d outputs are always 4D (N, C, H, W) in the edge dialect.
    """
    try:
        if "val" in conv_node.meta:
            output_shape = conv_node.meta["val"].shape
            return output_shape[0]
    except (AttributeError, TypeError):
        pass
    return None


def _has_qparams(node: Node) -> bool:
    return (
        node.meta.get("input_qparams", {}) != {}
        and node.meta.get("output_qparams", {}) != {}
    )


@AtenToCortexMPass.register_dialect_substitution(exir_ops.edge.aten.sigmoid.default)
@AtenToCortexMPass.register_dialect_substitution(exir_ops.edge.aten.tanh.default)
@AtenToCortexMPass.register_dialect_substitution(exir_ops.edge.aten.silu.default)
def _get_activation_replacement(
    node: Node, dialect_pass: AtenToDialectPass
) -> DialectNodeSpec | None:
    """Lower a standalone quantized sigmoid / tanh / silu to a single
    cortex_m.quantized_activation call backed by an AoT-built 256-entry
    int8 LUT. The kernel is shape-agnostic; the LUT encodes both the
    activation function and the input/output qparams.
    """
    if not _has_qparams(node):
        return None

    exported_program = dialect_pass.exported_program
    input_qparams = node.meta["input_qparams"][0]
    output_qparams = node.meta["output_qparams"][0]
    lut_tensor = build_activation_lut(
        node.target,
        float(input_qparams.scale),
        int(input_qparams.zp),
        float(output_qparams.scale),
        int(output_qparams.zp),
    )

    # Constant placeholders must appear before user-input placeholders;
    # anchor on the first existing placeholder so the new LUT lands in the
    # constant-placeholder block at the top of the graph.
    first_placeholder = next(n for n in node.graph.nodes if n.op == "placeholder")
    with node.graph.inserting_before(first_placeholder):
        lut_node = create_constant_placeholder(
            exported_program,
            node.graph,
            node.name + "_lut",
            InputKind.PARAMETER,
            lut_tensor,
        )

    new_args = (node.args[0], lut_node)
    return DialectNodeSpec(
        exir_ops.edge.cortex_m.quantized_activation.default, new_args
    )


@AtenToCortexMPass.register_dialect_substitution(exir_ops.edge.aten.linear.default)
def _get_linear_replacement(
    node: Node, dialect_pass: AtenToDialectPass
) -> DialectNodeSpec | None:
    """
    Let
    - yi be the output activations (y1, ... yn)
    - xj be the input activations (x1, ... xm)
    - wij be the weights (w11, ... wnm)
    - a be the input offset
    - b be the weight offset
    - ci be the bias

    Then the linear operation can be written as:
    yi = sum_j((xj + a) * (wij + b)) + ci
    = sum_j(xj*wij + xj*b + a*wij + a*b) + ci
    = sum_j(xj*wij) + sum_j(xj)*b + (a * sum_j(wij + b) + ci)
    = sum_j(xj*wij) + sum_j(xj)*b + kernel_sum

    where kernel_sum is precomputed aot.
    """
    if not _has_qparams(node):
        return None

    assert isinstance(dialect_pass, AtenToCortexMPass)
    exported_program = dialect_pass.exported_program
    target_config = dialect_pass.target_config

    input_scale = node.meta["input_qparams"][0].scale
    input_zp = node.meta["input_qparams"][0].zp
    weight_scale = node.meta["input_qparams"][1].scale
    weight_zp = node.meta["input_qparams"][1].zp
    output_scale = node.meta["output_qparams"][0].scale
    output_zp = node.meta["output_qparams"][0].zp
    output_min = node.meta["output_qparams"][0].qmin
    output_max = node.meta["output_qparams"][0].qmax

    if weight_zp != 0:
        raise NotImplementedError(
            f"cortex_m::quantized_linear assumes symmetric weight "
            f"quantization (weight_zp == 0); got weight_zp={weight_zp}"
        )

    quantized_multiplier, quantized_shift = quantize_multiplier_aot(
        (input_scale * weight_scale) / output_scale
    )

    # CMSIS-NN's MVE `arm_fully_connected_s8` path reads a precomputed
    # kernel_sum (input_offset×sum(weight) + bias) from ctx.buf and
    # ignores the bias argument. The DSP and scalar paths do the opposite
    # — they read the bias argument at runtime and ignore ctx.buf
    # (see arm_nn_vec_mat_mult_t_s8.c). Pick the right input format here
    # based on the target ISA so the runtime gets exactly what it expects.
    linear_args = node.args
    weights = cast(Node, linear_args[1])
    weights_tensor = get_param_tensor(exported_program, weights)
    bias_node = cast(Node | None, linear_args[2]) if len(linear_args) > 2 else None
    bias_tensor = (
        get_param_tensor(exported_program, bias_node) if bias_node is not None else None
    )

    if target_config.backend == cmsis_nn.Backend.MVE:
        kernel_sum_tensor = _compute_kernel_sum(
            weights_tensor, bias_tensor, -input_zp, -weight_zp
        )
        with node.graph.inserting_after(weights):
            kernel_sum_arg = create_constant_placeholder(
                exported_program,
                node.graph,
                node.name + "_kernel_sum",
                InputKind.PARAMETER,
                kernel_sum_tensor,
            )
        bias_arg = None
    else:
        kernel_sum_arg = None
        bias_arg = bias_node

    args = (
        linear_args[0],
        weights,
        bias_arg,
        kernel_sum_arg,
        -input_zp,
        -weight_zp,
        output_zp,
        [quantized_multiplier],
        [quantized_shift],
        output_max,
        output_min,
    )

    return DialectNodeSpec(exir_ops.edge.cortex_m.quantized_linear.default, args)


@AtenToCortexMPass.register_dialect_substitution(exir_ops.edge.aten.convolution.default)
def _get_convolution_replacement(
    node: Node, dialect_pass: AtenToDialectPass
) -> DialectNodeSpec | None:
    if not _has_qparams(node):
        return None

    exported_program = dialect_pass.exported_program
    conv_args = node.args
    (
        x,
        weight,
        bias,
        stride,
        padding,
        dilation,
        transposed,
        _,
        groups,
    ) = (
        conv_args[0],
        cast(Node, conv_args[1]),
        conv_args[2],
        conv_args[3],
        conv_args[4],
        conv_args[5],
        cast(bool, conv_args[6]),
        conv_args[7],
        cast(int, conv_args[8]),
    )

    if transposed:
        return _get_transpose_conv2d_replacement(node, dialect_pass)

    input_scale = node.meta["input_qparams"][0].scale
    input_zero_point = node.meta["input_qparams"][0].zp
    weight_scales = node.meta["input_qparams"][1].scale
    if not isinstance(weight_scales, list):
        fake_weight_tensor = get_first_fake_tensor(weight)
        weight_scales = [weight_scales] * fake_weight_tensor.shape[0]

    output_qparams = node.meta["output_qparams"][0]
    output_scale = output_qparams.scale
    output_zero_point = output_qparams.zp
    output_qmin = output_qparams.qmin
    output_qmax = output_qparams.qmax

    quantized_multipliers = []
    quantized_shifts = []
    for weight_scale in weight_scales:
        quantized_multiplier, quantized_shift = quantize_multiplier_aot(
            input_scale * weight_scale / output_scale
        )
        quantized_multipliers.append(quantized_multiplier)
        quantized_shifts.append(quantized_shift)

    param_weight_tensor = get_param_tensor(exported_program, weight)
    if param_weight_tensor is None:
        raise RuntimeError(
            f"Expected convolution weight parameter tensor for node {node.name}."
        )

    # Detect depthwise convolution:
    # Depthwise means groups == in_channels, out_channels == K * in_channels
    # Weight shape is [out_ch, in_ch_per_group, H, W]
    in_channels = param_weight_tensor.shape[1] * groups
    out_channels = param_weight_tensor.shape[0]
    is_depthwise = (in_channels == groups) and (out_channels % in_channels == 0)

    # Only use DW path if batch_size==1, as CMSIS-NN DW falls back to
    # unoptimized implementation otherwise.
    batch_size = _get_batch_size_from_conv(node)

    # TODO(#16347): It is likely but not certain that the un-optimized
    # CMSIS-NN DW conv or the one without any SIMD is less efficient that
    # the corresponding CMSIS-NN conv. We should benchmark and update the
    # constraints.
    # optimal_dw_conv_constraints = (batch_size == 1) and (
    #    (in_channels == out_channels and dilation == [1, 1]) or (in_channels == 1)
    # )
    use_depthwise_conv = is_depthwise and (batch_size == 1)

    if use_depthwise_conv:
        # For depthwise: OIHW -> IHWO which gives [1, H, W, C_OUT] for CMSIS-NN
        # PyTorch depthwise weight is [out_ch, 1, H, W], permute to [1, H, W, out_ch]
        # The permute achieves the desired logical layout (IHWO). CMSIS-NN expects
        # weights in physically contiguous memory after the permute (not in channels-last)
        # so we use contiguous() here.
        weight_permuted = param_weight_tensor.permute(1, 2, 3, 0).contiguous()
    else:
        # For regular conv: OIHW -> OHWI
        # The permute achieves the desired logical layout (OHWI). CMSIS-NN expects
        # weights in physically contiguous memory after the permute (not in channels-last)
        # so we use contiguous() here.
        weight_permuted = param_weight_tensor.permute(0, 2, 3, 1).contiguous()

    with node.graph.inserting_after(weight):
        weight_nhwc = create_constant_placeholder(
            exported_program,
            node.graph,
            node.name + "_weight_nhwc",
            InputKind.PARAMETER,
            weight_permuted,
        )

        quantized_multiplier_tensor = create_constant_placeholder(
            exported_program,
            node.graph,
            node.name + "_quantized_multiplier",
            InputKind.PARAMETER,
            torch.tensor(quantized_multipliers, dtype=torch.int32),
        )

        quantized_shift_tensor = create_constant_placeholder(
            exported_program,
            node.graph,
            node.name + "_quantized_shift",
            InputKind.PARAMETER,
            torch.tensor(quantized_shifts, dtype=torch.int32),
        )

    if use_depthwise_conv:
        # Compute depth_multiplier for depthwise convolution
        # For depthwise: output_channels = input_channels * depth_multiplier

        if out_channels % in_channels != 0:
            raise ValueError(
                f"Depthwise conv: output_channels ({out_channels}) must be "
                f"divisible by input_channels ({in_channels})"
            )
        depth_multiplier = out_channels // in_channels

        scratch = _create_uninitialized_alloc_node(node, exported_program)

        depthwise_args = (
            x,
            weight_nhwc,
            bias,
            stride,
            padding,
            dilation,
            depth_multiplier,
            -input_zero_point,
            output_zero_point,
            quantized_multiplier_tensor,
            quantized_shift_tensor,
            output_qmin,
            output_qmax,
            scratch,
        )
        return DialectNodeSpec(
            exir_ops.edge.cortex_m.quantized_depthwise_conv2d.default,
            depthwise_args,
        )

    # Use regular convolution operator
    scratch = _create_uninitialized_alloc_node(node, exported_program)

    conv2d_args = (
        x,
        weight_nhwc,
        bias,
        stride,
        padding,
        dilation,
        -input_zero_point,
        output_zero_point,
        quantized_multiplier_tensor,
        quantized_shift_tensor,
        output_qmin,
        output_qmax,
        scratch,
    )
    return DialectNodeSpec(exir_ops.edge.cortex_m.quantized_conv2d.default, conv2d_args)


def _get_transpose_conv2d_replacement(
    node: Node, dialect_pass: AtenToDialectPass
) -> DialectNodeSpec | None:
    """
    Transform aten.convolution with transposed=True to cortex_m.quantized_transpose_conv2d.
    """
    if not _has_qparams(node):
        return None

    exported_program = dialect_pass.exported_program
    conv_t_args = node.args
    (
        x,
        weight,
        bias,
        stride,
        padding,
        dilation,
        transposed,
        output_padding,
        _,
    ) = (
        conv_t_args[0],
        cast(Node, conv_t_args[1]),
        conv_t_args[2],
        conv_t_args[3],
        conv_t_args[4],
        conv_t_args[5],
        cast(bool, conv_t_args[6]),
        conv_t_args[7],
        cast(int, conv_t_args[8]),
    )

    if not transposed:
        return None

    input_scale = node.meta["input_qparams"][0].scale
    input_zero_point = node.meta["input_qparams"][0].zp
    weight_scales = node.meta["input_qparams"][1].scale

    # For transposed conv: weight shape is (in_channels, out_channels/groups, H, W)
    # We need requantization params for each output channel.
    weight_tensor = get_first_fake_tensor(weight)
    if not isinstance(weight_scales, list):
        # weight_tensor.shape[1] is out_channels for transposed conv.
        num_output_channels = weight_tensor.shape[1]
        weight_scales = [weight_scales] * num_output_channels

    output_qparams = node.meta["output_qparams"][0]
    output_scale = output_qparams.scale
    output_zero_point = output_qparams.zp
    output_qmin = output_qparams.qmin
    output_qmax = output_qparams.qmax

    # Compute per-channel requantization parameters.
    quantized_multipliers = []
    quantized_shifts = []
    for weight_scale in weight_scales:
        quantized_multiplier, quantized_shift = quantize_multiplier_aot(
            input_scale * weight_scale / output_scale
        )
        quantized_multipliers.append(quantized_multiplier)
        quantized_shifts.append(quantized_shift)

    # CRITICAL: Weight layout transformation for transposed conv
    # PyTorch ConvTranspose2d: (in_channels, out_channels/groups, H, W)
    # CMSIS-NN expects: (out_channels, H, W, in_channels) = OHWI
    # Permutation: (1, 2, 3, 0)
    weight_tensor_param = get_param_tensor(exported_program, weight)
    if weight_tensor_param is None:
        raise RuntimeError(
            f"Expected transpose conv weight parameter tensor for node {node.name}."
        )
    weight_permuted = weight_tensor_param.permute(1, 2, 3, 0).contiguous()

    with node.graph.inserting_after(weight):
        weight_nhwc = create_constant_placeholder(
            exported_program,
            node.graph,
            node.name + "_weight_nhwc",
            InputKind.PARAMETER,
            weight_permuted,
        )

        quantized_multiplier_tensor = create_constant_placeholder(
            exported_program,
            node.graph,
            node.name + "_quantized_multiplier",
            InputKind.PARAMETER,
            torch.tensor(quantized_multipliers, dtype=torch.int32),
        )

        quantized_shift_tensor = create_constant_placeholder(
            exported_program,
            node.graph,
            node.name + "_quantized_shift",
            InputKind.PARAMETER,
            torch.tensor(quantized_shifts, dtype=torch.int32),
        )

    scratch = _create_uninitialized_alloc_node(node, exported_program)
    output_scratch = _create_uninitialized_alloc_node(node, exported_program)

    new_args = (
        x,
        weight_nhwc,
        bias,
        stride,
        padding,
        output_padding,  # output_padding is NEW for transposed conv
        dilation,
        -input_zero_point,
        output_zero_point,
        quantized_multiplier_tensor,
        quantized_shift_tensor,
        output_qmin,
        output_qmax,
        scratch,
        output_scratch,
    )
    return DialectNodeSpec(
        exir_ops.edge.cortex_m.quantized_transpose_conv2d.default, new_args
    )


@AtenToCortexMPass.register_dialect_substitution(exir_ops.edge.aten.bmm.default)
def _get_bmm_replacement(
    node: Node, dialect_pass: AtenToDialectPass
) -> DialectNodeSpec | None:
    if not _has_qparams(node):
        return None

    exported_program = dialect_pass.exported_program
    lhs_scale = node.meta["input_qparams"][0].scale
    lhs_zp = node.meta["input_qparams"][0].zp
    rhs_scale = node.meta["input_qparams"][1].scale
    rhs_zp = node.meta["input_qparams"][1].zp
    output_scale = node.meta["output_qparams"][0].scale
    output_zp = node.meta["output_qparams"][0].zp

    output_mult, output_shift = quantize_multiplier_aot(
        (lhs_scale * rhs_scale) / output_scale
    )

    bmm_args = node.args
    lhs_node = cast(Node, bmm_args[0])
    rhs_node = cast(Node, bmm_args[1])

    is_constant_rhs = is_param_node(exported_program, rhs_node)
    if is_constant_rhs:
        rhs_tensor = get_param_tensor(exported_program, rhs_node)
        if rhs_tensor is None:
            raise RuntimeError(
                f"Expected constant RHS parameter tensor for node {node.name}."
            )
        rhs_transposed_tensor = rhs_tensor.permute(0, 2, 1).contiguous()
        with node.graph.inserting_after(rhs_node):
            rhs_transposed = create_constant_placeholder(
                exported_program,
                node.graph,
                node.name + "_rhs_transposed",
                InputKind.PARAMETER,
                rhs_transposed_tensor,
            )
    else:
        with node.graph.inserting_before(node):
            rhs_transposed = node.graph.create_node(
                "call_function",
                target=exir_ops.edge.cortex_m.transpose.default,
                args=(rhs_node, [0, 2, 1]),
            )

    scratch = _create_uninitialized_alloc_node(node, exported_program)

    args = (
        lhs_node,
        -lhs_zp,
        rhs_transposed,
        -rhs_zp,
        output_zp,
        output_mult,
        output_shift,
        scratch,
    )
    return DialectNodeSpec(exir_ops.edge.cortex_m.quantized_batch_matmul.default, args)


@AtenToCortexMPass.register_dialect_substitution(exir_ops.edge.aten.avg_pool2d.default)
def _get_avg_pool2d_replacement(
    node: Node, dialect_pass: AtenToDialectPass
) -> DialectNodeSpec | None:
    if not _has_qparams(node):
        return None

    exported_program = dialect_pass.exported_program
    pool_args = node.args
    kernel_size = cast(list[int], pool_args[1])
    stride = cast(list[int], pool_args[2]) if len(pool_args) > 2 else list(kernel_size)
    padding = cast(list[int], pool_args[3]) if len(pool_args) > 3 else [0, 0]
    ceil_mode = cast(bool, pool_args[4]) if len(pool_args) > 4 else False
    count_include_pad = cast(bool, pool_args[5]) if len(pool_args) > 5 else True
    divisor_override = pool_args[6] if len(pool_args) > 6 else None

    if ceil_mode or divisor_override is not None:
        return None

    input_node = cast(Node, pool_args[0])
    input_zp = node.meta["input_qparams"][0].zp
    input_scale = node.meta["input_qparams"][0].scale
    output_mult, output_shift = quantize_multiplier_aot(input_scale)

    avg_padding = padding
    if count_include_pad:
        pad_h, pad_w = padding
        input_tensor = get_first_fake_tensor(input_node)
        pre_pad = post_pad = to_physical_order([0, 0, pad_h, pad_w], input_tensor)
        with node.graph.inserting_before(node):
            input_node = node.graph.create_node(
                "call_function",
                target=exir_ops.edge.cortex_m.pad.default,
                args=(input_node, pre_pad, post_pad, int(input_zp)),
            )
        avg_padding = [0, 0]

    scratch = _create_uninitialized_alloc_node(node, exported_program)

    new_args = (
        input_node,
        kernel_size,
        stride,
        avg_padding,
        int(input_zp),
        int(output_mult),
        int(output_shift),
        scratch,
    )
    return DialectNodeSpec(
        exir_ops.edge.cortex_m.quantized_avg_pool2d.default, new_args
    )


@AtenToCortexMPass.register_dialect_substitution(
    exir_ops.edge.quantized_decomposed.quantize_per_tensor.default
)
def _get_quantize_per_tensor_replacement(
    node: Node, dialect_pass: AtenToDialectPass
) -> DialectNodeSpec | None:
    del dialect_pass
    if not _is_quant_per_tensor_qualified(node):
        return None
    return DialectNodeSpec(
        exir_ops.edge.cortex_m.quantize_per_tensor.default, node.args
    )


@AtenToCortexMPass.register_dialect_substitution(
    exir_ops.edge.quantized_decomposed.dequantize_per_tensor.default
)
def _get_dequantize_per_tensor_replacement(
    node: Node, dialect_pass: AtenToDialectPass
) -> DialectNodeSpec | None:
    del dialect_pass
    if not _is_quant_per_tensor_qualified(node):
        return None
    return DialectNodeSpec(
        exir_ops.edge.cortex_m.dequantize_per_tensor.default, node.args
    )


@AtenToCortexMPass.register_dialect_substitution(exir_ops.edge.aten.add.Tensor)
def _get_add_replacement(
    node: Node, dialect_pass: AtenToDialectPass
) -> DialectNodeSpec | None:
    del dialect_pass
    if not _has_qparams(node):
        return None

    scale1 = node.meta["input_qparams"][0].scale
    zero_point1 = node.meta["input_qparams"][0].zp
    scale2 = node.meta["input_qparams"][1].scale
    zero_point2 = node.meta["input_qparams"][1].zp
    output_scale = node.meta["output_qparams"][0].scale
    output_zero_point = node.meta["output_qparams"][0].zp

    max_scale_2x = 2 * max(scale1, scale2)
    input1_mult, input1_shift = quantize_multiplier_aot(scale1 / max_scale_2x)
    input2_mult, input2_shift = quantize_multiplier_aot(scale2 / max_scale_2x)
    output_mult, output_shift = quantize_multiplier_aot(
        max_scale_2x / (output_scale * (1 << SHIFT_INT8))
    )

    activation_min = node.meta["output_qparams"][0].qmin
    activation_max = node.meta["output_qparams"][0].qmax

    args = (
        node.args[0],
        zero_point1,
        input1_mult,
        input1_shift,
        node.args[1],
        zero_point2,
        input2_mult,
        input2_shift,
        output_zero_point,
        output_mult,
        output_shift,
        activation_min,
        activation_max,
    )
    return DialectNodeSpec(exir_ops.edge.cortex_m.quantized_add.default, args)


@AtenToCortexMPass.register_dialect_substitution(exir_ops.edge.aten.mul.Tensor)
def _get_mul_replacement(
    node: Node, dialect_pass: AtenToDialectPass
) -> DialectNodeSpec | None:
    del dialect_pass
    if not _has_qparams(node):
        return None

    scale1 = node.meta["input_qparams"][0].scale
    zero_point1 = node.meta["input_qparams"][0].zp
    scale2 = node.meta["input_qparams"][1].scale
    zero_point2 = node.meta["input_qparams"][1].zp
    output_scale = node.meta["output_qparams"][0].scale
    output_zero_point = node.meta["output_qparams"][0].zp

    output_mult, output_shift = quantize_multiplier_aot(
        (scale1 * scale2) / output_scale
    )
    args = (
        node.args[0],
        zero_point1,
        node.args[1],
        zero_point2,
        output_zero_point,
        output_mult,
        output_shift,
    )
    return DialectNodeSpec(exir_ops.edge.cortex_m.quantized_mul.default, args)


@AtenToCortexMPass.register_dialect_substitution(exir_ops.edge.aten._softmax.default)
def _get_softmax_replacement(
    node: Node, dialect_pass: AtenToDialectPass
) -> DialectNodeSpec | None:
    del dialect_pass
    if not _has_qparams(node):
        return None

    half_to_float = node.args[2] if len(node.args) > 2 else False
    if cast(bool, half_to_float):
        return None

    input_qparams = node.meta["input_qparams"][0]
    output_qparams = node.meta["output_qparams"][0]

    input_multiplier, input_shift, diff_min = _compute_softmax_params(
        float(input_qparams.scale)
    )

    output_scale_attr = getattr(output_qparams, "scale", None)
    output_zp_attr = getattr(output_qparams, "zp", None)
    if output_scale_attr is None or output_zp_attr is None:
        raise AssertionError("Softmax requires output quantization parameters.")

    output_scale_val = float(output_scale_attr)
    output_zp_val = int(output_zp_attr)
    if not math.isclose(
        output_scale_val, CMSIS_SOFTMAX_SCALE, rel_tol=0.0, abs_tol=1e-12
    ):
        raise AssertionError(
            "Softmax output scale must match CMSIS (1/256). " f"Got {output_scale_val}."
        )
    if output_zp_val != CMSIS_SOFTMAX_ZERO_POINT:
        raise AssertionError(
            "Softmax output zero-point must match CMSIS (-128). "
            f"Got {output_zp_val}."
        )

    args = (
        node.args[0],
        node.args[1],
        int(input_qparams.zp),
        output_zp_val,
        input_multiplier,
        input_shift,
        diff_min,
    )
    return DialectNodeSpec(exir_ops.edge.cortex_m.softmax.default, args)


@AtenToCortexMPass.register_dialect_substitution(exir_ops.edge.aten.max_pool2d.default)
def _get_max_pool2d_replacement(
    node: Node, dialect_pass: AtenToDialectPass
) -> DialectNodeSpec | None:
    del dialect_pass
    input_qparams = node.meta.get("input_qparams", {}).get(0)
    cortex_m_meta = node.meta.get("custom", {}).get("cortex_m", {})
    if input_qparams is None or cortex_m_meta.get("skip_quantized_max_pool2d", False):
        return None

    input_scale = float(input_qparams.scale)
    input_zero_point = int(input_qparams.zp)

    output_qparams = None
    if node.meta.get("output_qparams"):
        output_qparams = node.meta["output_qparams"].get(0)

    if output_qparams is not None:
        if getattr(output_qparams, "per_channel", False):
            return None
        output_scale = float(output_qparams.scale)
        output_zero_point = int(output_qparams.zp)
        activation_min = int(output_qparams.qmin)
        activation_max = int(output_qparams.qmax)
        if abs(input_scale - output_scale) > 1e-6:
            return None
        if input_zero_point != output_zero_point:
            return None
    else:
        output_zero_point = input_zero_point
        activation_min = torch.iinfo(torch.int8).min
        activation_max = torch.iinfo(torch.int8).max

    kernel_size = _to_int_pair(node.args[1], None)
    stride_arg = node.args[2] if len(node.args) > 2 else None
    stride = _to_int_pair(stride_arg, kernel_size)
    padding_arg = node.args[3] if len(node.args) > 3 else None
    padding = _to_int_pair(padding_arg, (0, 0))
    dilation_arg = node.args[4] if len(node.args) > 4 else None
    dilation = _to_int_pair(dilation_arg, (1, 1))
    ceil_mode_arg = node.args[5] if len(node.args) > 5 else False
    ceil_mode = _to_bool(ceil_mode_arg, False)

    if dilation != (1, 1) or ceil_mode:
        return None

    quantized_op = getattr(exir_ops.edge.cortex_m, "quantized_max_pool2d", None)
    if quantized_op is None:
        return None

    args = (
        node.args[0],
        kernel_size,
        stride,
        padding,
        dilation,
        ceil_mode,
        input_zero_point,
        output_zero_point,
        activation_min,
        activation_max,
    )
    return DialectNodeSpec(quantized_op.default, args)


@AtenToCortexMPass.register_dialect_substitution(exir_ops.edge.aten.minimum.default)
def _get_minimum_replacement(
    node: Node, dialect_pass: AtenToDialectPass
) -> DialectNodeSpec | None:
    del dialect_pass
    input_tensor = _get_input_tensor_data(node)
    if input_tensor.dtype not in (torch.int8, torch.int32):
        return None
    return DialectNodeSpec(exir_ops.edge.cortex_m.minimum.default, node.args)


@AtenToCortexMPass.register_dialect_substitution(exir_ops.edge.aten.maximum.default)
def _get_maximum_replacement(
    node: Node, dialect_pass: AtenToDialectPass
) -> DialectNodeSpec | None:
    del dialect_pass
    input_tensor = _get_input_tensor_data(node)
    if input_tensor.dtype != torch.int8:
        return None
    return DialectNodeSpec(exir_ops.edge.cortex_m.maximum.default, node.args)


@AtenToCortexMPass.register_dialect_substitution(
    exir_ops.edge.aten.permute_copy.default
)
def _get_permute_replacement(
    node: Node, dialect_pass: AtenToDialectPass
) -> DialectNodeSpec | None:
    del dialect_pass
    input_tensor = _get_input_tensor_data(node)
    if input_tensor.dtype != torch.int8:
        return None

    rank = len(input_tensor.shape)
    perms = [p % rank for p in cast(tuple[int, ...], node.args[1])]
    return DialectNodeSpec(
        exir_ops.edge.cortex_m.transpose.default, (node.args[0], perms)
    )


@AtenToCortexMPass.register_dialect_substitution(
    exir_ops.edge.aten.constant_pad_nd.default
)
def _get_pad_replacement(
    node: Node, dialect_pass: AtenToDialectPass
) -> DialectNodeSpec | None:
    del dialect_pass
    input_qparams = node.meta.get("input_qparams", {})
    if not input_qparams:
        return None

    scale = float(input_qparams[0].scale)
    zero_point = int(input_qparams[0].zp)
    padding = cast(tuple[int, ...], node.args[1])
    pad_value_raw = node.args[2] if len(node.args) > 2 else 0
    pad_value_float = float(cast(float, pad_value_raw))

    quantized_pad_value = int(
        quantize_val(pad_value_float, scale, zero_point, -128, 127)
    )

    input_tensor = _get_input_tensor_data(node)
    rank = len(input_tensor.shape)
    assert 1 <= rank <= 4, f"cortex_m pad: expected rank in [1, 4], got {rank}"
    n_pairs = len(padding) // 2
    assert (
        len(padding) % 2 == 0 and n_pairs <= rank
    ), f"cortex_m pad: invalid padding length {len(padding)} for rank {rank}"

    pre_pad = [0, 0, 0, 0]
    post_pad = [0, 0, 0, 0]
    for i in range(n_pairs):
        dim_4d = 3 - i
        pre_pad[dim_4d] = int(padding[2 * i])
        post_pad[dim_4d] = int(padding[2 * i + 1])

    pre_pad = to_physical_order(pre_pad, input_tensor)
    post_pad = to_physical_order(post_pad, input_tensor)

    args = (node.args[0], pre_pad, post_pad, int(quantized_pad_value))
    return DialectNodeSpec(exir_ops.edge.cortex_m.pad.default, args)
