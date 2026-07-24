# Copyright (c) 2026 iote.ai
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.
"""TOSA to AXON per-operation converters.

Each ``_convert_*`` function handles one TOSA operation type and
produces the corresponding AXON layer descriptor(s). Called by
``tosa_to_axon_layers()`` in ``axon_compiler.py``.
"""
from __future__ import annotations

import logging
import struct

import numpy as np

from .axon.compile_spec import (
    AXON_MAX_CONV2D_FILTER,
    AXON_MAX_CONV_STRIDE,
    AXON_MAX_FC_INPUT,
    AXON_MAX_FC_OUTPUT,
    AXON_MAX_POOL_FILTER,
    AXON_MAX_TENSOR_DIM,
)
from .axon_types import (
    ActivationQuantInfo,
    AxonActivation,
    AxonByteWidth,
    AxonDimensions,
    AxonLayer,
    AxonOp,
)
from .tosa_reader import TosaGraph, TosaOperator, TosaTensor

logger = logging.getLogger(__name__)

def _handle_table_op(
    op: TosaOperator,
    ops: list[TosaOperator],
    op_idx: int,
    layers: list[AxonLayer],
    tensor_to_layer: dict[str, int],
    tensor_to_zp: dict[str, int],
    activation_info: list["ActivationQuantInfo"],
    activation_idx: int,
    axon_layer_idx: int,
) -> tuple[int, int, int] | None:
    """Convert a TOSA TABLE op into an AXON op extension layer.

    Modifies the preceding AXON layer to output INT16 q3.12 (sigmoid/tanh)
    and adds an op extension layer (operation 101 or 102) that runs
    sigmoid/tanh on the q3.12 input.

    Returns:
        (new_axon_layer_idx, new_activation_idx, ops_to_advance) on success,
        or None if this TABLE could not be matched / handled.
    """
    if activation_idx >= len(activation_info):
        return None

    info = activation_info[activation_idx]
    if info.op_type not in ("sigmoid", "tanh"):
        # Softmax TABLE is part of a multi-op decomposition; handled elsewhere.
        return None

    if not op.input_tensors or not op.output_tensors:
        return None

    in_name = op.input_tensors[0].name
    out_name = op.output_tensors[0].name
    if in_name not in tensor_to_layer:
        logger.warning(f"  TABLE input tensor not produced by an AXON layer; skipping")
        return None

    prev_idx = tensor_to_layer[in_name]
    prev_layer = layers[prev_idx]

    # ────────────────────────────────────────────────────────────
    # 1. Modify the preceding layer to output INT16 q3.12.
    #
    # The preceding layer's RESCALE currently encodes:
    #     int8_out = (acc * mult >> shift) + zp_int8
    #     mult/2^shift  ≈  S_in_act / 1                 (per Nordic)
    # but in our pipeline mult/shift currently encodes
    #     S_in_acc * S_w / S_int8_out
    # i.e. acc → INT8.
    #
    # For q3.12 we need: q3_12 = float_value * 2^12.
    # Where float_value = acc * S_in_acc * S_w = (acc * mult/2^shift) * S_int8_out.
    # So new_factor = old_factor * S_int8_out * 2^12.
    # ────────────────────────────────────────────────────────────
    if not prev_layer.multiplier_data or not prev_layer.shift_data:
        logger.warning(f"  Preceding layer has no rescale; cannot retarget to q3.12")
        return None

    prev_mult = np.frombuffer(prev_layer.multiplier_data, dtype=np.int32).copy()
    prev_shift = np.frombuffer(prev_layer.shift_data, dtype=np.int8).copy()

    s_int8_out = info.input_scale  # output scale of the preceding INT8 RESCALE
    new_mult = np.zeros_like(prev_mult)
    new_shift = np.zeros_like(prev_shift)
    for ch in range(len(prev_mult)):
        old_factor = float(prev_mult[ch]) / (2.0 ** int(prev_shift[ch]))
        # ×4096 because q3.12 stores float_value * 2^12
        q312_scale = old_factor * s_int8_out * 4096.0
        # Nordic limits the q3.12 preceding-layer rescale to max_shift=28.
        m, s = _optimized_scaling_shift(q312_scale, output_zp=0,
                                         min_shift=8, max_shift=28, bit_limit=31)
        new_mult[ch] = m
        new_shift[ch] = s

    prev_layer.multiplier_data = new_mult.tobytes()
    prev_layer.shift_data = new_shift.tobytes()
    # The preceding layer's INT16 q3.12 output has zp=0 by definition.
    prev_layer.output_zero_point = 0
    prev_layer.output_dimensions.byte_width = AxonByteWidth.INT16

    logger.debug(
        f"  Retargeted layer {prev_idx} to INT16 q3.12 for {info.op_type}: "
        f"new mult[0]={int(new_mult[0])} shift[0]={int(new_shift[0])}"
    )

    # ────────────────────────────────────────────────────────────
    # 2. Build the op extension layer (sigmoid=101, tanh=102).
    #
    # The op extension takes a q3.12 INT16 input, applies the function,
    # and produces an INT8 output. The layer's mult/shift encode 1/S_out
    # so that: int8_out = (float_result * mult >> shift) + zp_out.
    # ────────────────────────────────────────────────────────────
    op_enum = AxonOp.SIGMOID if info.op_type == "sigmoid" else AxonOp.TANH
    out_scale = info.output_scale
    out_zp = info.output_zp

    inv_scale = 1.0 / out_scale if out_scale > 0 else 0.0
    ext_mult, ext_shift = _optimized_scaling_shift(
        inv_scale, output_zp=out_zp,
        min_shift=8, max_shift=31, bit_limit=31,
    )

    # Dimensions: same as preceding layer's output (which is now INT16 q3.12).
    # AXON op extensions are element-wise — no spatial change.
    out_h = prev_layer.output_dimensions.height
    out_w = prev_layer.output_dimensions.width
    out_c = prev_layer.output_dimensions.channel_cnt

    ext_layer = AxonLayer(
        input_ids=[prev_idx],
        operation=op_enum,
        input_dimensions=[AxonDimensions(
            height=out_h, width=out_w, channel_cnt=out_c,
            byte_width=AxonByteWidth.INT16,
        )],
        output_dimensions=AxonDimensions(
            height=out_h, width=out_w, channel_cnt=out_c,
            byte_width=AxonByteWidth.INT8,
        ),
        input_zero_point=0,
        output_zero_point=out_zp,
        activation=AxonActivation.DISABLED,
        multiplier_data=np.array([ext_mult], dtype=np.int32).tobytes(),
        shift_data=np.array([ext_shift], dtype=np.int8).tobytes(),
        scale_shift_cnt=1,
    )
    layers.append(ext_layer)
    tensor_to_layer[out_name] = axon_layer_idx
    tensor_to_zp[out_name] = out_zp

    logger.debug(
        f"  Layer {axon_layer_idx} ({info.op_type.upper()} ext): "
        f"mult={ext_mult} shift={ext_shift} out_zp={out_zp}"
    )

    return (axon_layer_idx + 1, activation_idx + 1, 1)


# TOSA ops that appear in the quantized softmax decomposition.
# A softmax (with stable variant) lowers to roughly:
#   RESCALE → REDUCE_MAX → RESCALE → SUB → RESCALE → TABLE(exp)
#   → RESCALE → RESCALE → REDUCE_SUM → RESCALE → TABLE(reciprocal)
#   → RESCALE → MUL → RESCALE
_SOFTMAX_DECOMP_OPS = frozenset({
    "RESCALE", "REDUCE_MAX", "REDUCE_SUM", "SUB", "TABLE",
    "MUL", "EXP", "RECIPROCAL", "RESHAPE",
})


def _handle_softmax_pattern(
    start_op: TosaOperator,
    ops: list[TosaOperator],
    start_idx: int,
    layers: list[AxonLayer],
    tensor_to_layer: dict[str, int],
    tensor_to_zp: dict[str, int],
    activation_info: list["ActivationQuantInfo"],
    activation_idx: int,
    axon_layer_idx: int,
) -> tuple[int, int, int] | None:
    """Replace a TOSA softmax decomposition with a single AXON op extension.

    Modifies the preceding layer to output INT32 q11.12 with PREPARE_SOFTMAX
    activation, then adds an op extension layer (operation=100) that runs
    softmax on-device via nrf_axon_nn_op_extension_softmax().

    Returns:
        (new_axon_layer_idx, new_activation_idx, ops_to_advance) on success.
    """
    info = activation_info[activation_idx]

    # The REDUCE_MAX's input tensor (possibly through standalone RESCALEs that
    # we already skipped/passed through) should map to the preceding AXON layer.
    in_name = start_op.input_tensors[0].name
    if in_name not in tensor_to_layer:
        logger.warning("  Softmax: REDUCE_MAX input not produced by an AXON layer")
        return None

    prev_idx = tensor_to_layer[in_name]
    prev_layer = layers[prev_idx]
    if not prev_layer.multiplier_data or not prev_layer.shift_data:
        logger.warning("  Softmax: preceding layer has no rescale; cannot retarget")
        return None

    # Walk forward to find the end of the softmax pattern. The chain should
    # consist entirely of softmax-decomposition ops; the end is the last
    # MUL plus any trailing RESCALE.
    last_mul_idx = None
    end_idx = start_idx
    for k in range(start_idx, len(ops)):
        if ops[k].op_name not in _SOFTMAX_DECOMP_OPS:
            break
        end_idx = k
        if ops[k].op_name == "MUL":
            last_mul_idx = k

    if last_mul_idx is None:
        logger.warning("  Softmax: no MUL found in expected decomposition pattern")
        return None

    # Include the trailing RESCALE if present.
    if last_mul_idx + 1 < len(ops) and ops[last_mul_idx + 1].op_name == "RESCALE":
        end_idx = last_mul_idx + 1
    else:
        end_idx = last_mul_idx

    final_op = ops[end_idx]
    if not final_op.output_tensors:
        return None
    final_tensor_name = final_op.output_tensors[0].name

    # ────────────────────────────────────────────────────────────
    # 1. Retarget preceding layer to INT32 q11.12 with PREPARE_SOFTMAX.
    #
    # Same q-format math as sigmoid/tanh (×4096 = 2^12), but with INT32
    # output to give 11 integer bits of headroom for the unnormalised
    # softmax inputs. Nordic uses a data-dependent scaleshift_max_range
    # (31 - bits_needed(input_range)) for softmax; we use 28 like the
    # sigmoid/tanh case as a safe upper bound.
    # ────────────────────────────────────────────────────────────
    prev_mult = np.frombuffer(prev_layer.multiplier_data, dtype=np.int32).copy()
    prev_shift = np.frombuffer(prev_layer.shift_data, dtype=np.int8).copy()

    s_int8_out = info.input_scale
    new_mult = np.zeros_like(prev_mult)
    new_shift = np.zeros_like(prev_shift)
    for ch in range(len(prev_mult)):
        old_factor = float(prev_mult[ch]) / (2.0 ** int(prev_shift[ch]))
        q11_12_scale = old_factor * s_int8_out * 4096.0
        m, s = _optimized_scaling_shift(q11_12_scale, output_zp=0,
                                         min_shift=8, max_shift=28, bit_limit=31)
        new_mult[ch] = m
        new_shift[ch] = s

    prev_layer.multiplier_data = new_mult.tobytes()
    prev_layer.shift_data = new_shift.tobytes()
    prev_layer.output_zero_point = 0
    prev_layer.output_dimensions.byte_width = AxonByteWidth.INT32
    prev_layer.activation = AxonActivation.PREPARE_SOFTMAX

    logger.debug(
        f"  Retargeted layer {prev_idx} to INT32 q11.12 PREPARE_SOFTMAX: "
        f"new mult[0]={int(new_mult[0])} shift[0]={int(new_shift[0])}"
    )

    # ────────────────────────────────────────────────────────────
    # 2. Build the SOFTMAX op extension layer (operation 100).
    # ────────────────────────────────────────────────────────────
    out_scale = info.output_scale
    out_zp = info.output_zp
    inv_scale = 1.0 / out_scale if out_scale > 0 else 0.0
    ext_mult, ext_shift = _optimized_scaling_shift(
        inv_scale, output_zp=out_zp,
        min_shift=8, max_shift=31, bit_limit=31,
    )

    out_h = prev_layer.output_dimensions.height
    out_w = prev_layer.output_dimensions.width
    out_c = prev_layer.output_dimensions.channel_cnt

    ext_layer = AxonLayer(
        input_ids=[prev_idx],
        operation=AxonOp.SOFTMAX,
        input_dimensions=[AxonDimensions(
            height=out_h, width=out_w, channel_cnt=out_c,
            byte_width=AxonByteWidth.INT32,
        )],
        output_dimensions=AxonDimensions(
            height=out_h, width=out_w, channel_cnt=out_c,
            byte_width=AxonByteWidth.INT8,
        ),
        input_zero_point=0,
        output_zero_point=out_zp,
        activation=AxonActivation.DISABLED,
        multiplier_data=np.array([ext_mult], dtype=np.int32).tobytes(),
        shift_data=np.array([ext_shift], dtype=np.int8).tobytes(),
        scale_shift_cnt=1,
    )
    layers.append(ext_layer)
    tensor_to_layer[final_tensor_name] = axon_layer_idx
    tensor_to_zp[final_tensor_name] = out_zp

    logger.debug(
        f"  Layer {axon_layer_idx} (SOFTMAX ext): "
        f"mult={ext_mult} shift={ext_shift} out_zp={out_zp}, "
        f"replaced TOSA[{start_idx}..{end_idx}]"
    )

    # Skip ahead past the entire softmax decomposition.
    advance = (end_idx - start_idx) + 1
    return (axon_layer_idx + 1, activation_idx + 1, advance)


def _optimized_scaling_shift(scale: float, output_zp: int,
                              min_shift: int = 8, max_shift: int = 30,
                              bit_limit: int = 31) -> tuple[int, int]:
    """Find optimal multiplier and shift for a given scale.

    Matches Nordic's optimized_ip_scaling_shift algorithm:
    - Searches shift range [min_shift, max_shift)
    - Ensures abs(mult) < 2^bit_limit
    - Ensures abs(output_zp * 2^shift) < 2^bit_limit
    - Picks highest shift that satisfies constraints (best precision)

    Returns:
        (multiplier, shift)
    """
    best_shift = min_shift
    for s in range(min_shift, max_shift):
        m = abs(int(np.round(scale * (2 ** s))))
        zp_scaled = abs(int(np.round(output_zp * (2 ** s))))
        if m < (1 << bit_limit) and zp_scaled < (1 << bit_limit):
            best_shift = s
        else:
            break
    mult = abs(int(np.round(scale * (2 ** best_shift))))
    return mult, best_shift


def _extract_rescale_params(
    rescale_op: TosaOperator,
) -> tuple[int, bytes, bytes, int]:
    """Extract quantization parameters from a TOSA RESCALE op.

    Recovers the floating-point scale from TOSA's mult/shift, then
    recomputes optimal mult/shift using Nordic's algorithm (range [8, 30),
    bit_limit=31). This is critical — TOSA's raw shift values (32-34)
    are out of AXON's effective range and cause output_zp to be ignored.

    Returns:
        (output_zp, multiplier_data, shift_data, scale_shift_cnt)
    """
    mult_tensor = rescale_op.input_tensors[1]
    shift_tensor = rescale_op.input_tensors[2]
    rescale_out_zp_tensor = rescale_op.input_tensors[4]

    output_zp = 0
    if rescale_out_zp_tensor.data is not None:
        output_zp = int(rescale_out_zp_tensor.data.flat[0])

    multiplier_data = b""
    shift_data = b""
    scale_shift_cnt = 0

    if mult_tensor.raw_bytes and shift_tensor.data is not None:
        # TOSA stores multiplier as INT16 dtype but raw bytes are INT32
        tosa_mult = np.frombuffer(mult_tensor.raw_bytes, dtype=np.int32)
        tosa_shift = shift_tensor.data.flatten().astype(np.int32)

        num_channels = len(tosa_mult)

        if num_channels > 1:
            logger.debug(f"  Per-channel RESCALE: {num_channels} channels")

        # Recover floating-point scale from TOSA's mult/shift:
        #   scale = tosa_mult / 2^tosa_shift
        # Then recompute with Nordic's algorithm
        new_mult = np.zeros(num_channels, dtype=np.int32)
        new_shift = np.zeros(num_channels, dtype=np.int8)

        for ch in range(num_channels):
            # Recover the original floating-point scale
            tm = int(tosa_mult[ch])
            ts = int(tosa_shift[ch])
            if ts > 0 and tm != 0:
                scale = tm / (2.0 ** ts)
            else:
                scale = 0.0

            # Recompute with Nordic's optimal algorithm
            m, s = _optimized_scaling_shift(scale, output_zp)
            new_mult[ch] = m
            new_shift[ch] = s

        if num_channels == 1:
            logger.debug(f"  RESCALE recomputed: tosa_mult={int(tosa_mult[0])}/tosa_shift={int(tosa_shift[0])} "
                        f"→ mult={int(new_mult[0])}/shift={int(new_shift[0])} "
                        f"(scale={tosa_mult[0]/(2.0**tosa_shift[0]):.8f})")

        multiplier_data = new_mult.tobytes()
        shift_data = new_shift.tobytes()
        scale_shift_cnt = num_channels

    elif mult_tensor.raw_bytes:
        mult_int32 = np.frombuffer(mult_tensor.raw_bytes, dtype=np.int32)
        multiplier_data = mult_int32.tobytes()
        scale_shift_cnt = len(mult_int32)
    elif shift_tensor.data is not None:
        shift_data = shift_tensor.data.astype(np.int8).tobytes()
        scale_shift_cnt = len(shift_data)

    return output_zp, multiplier_data, shift_data, scale_shift_cnt


def _resolve_input_id(tensor: TosaTensor, tensor_to_layer: dict[str, int]) -> int:
    """Look up which AXON layer produced a tensor, or -1 for graph input."""
    return tensor_to_layer.get(tensor.name, -1)


def _validate_axon_layer(layer: AxonLayer, layer_idx: int) -> list[str]:
    """Validate an AXON layer against hardware constraints.

    Returns list of warning strings. Empty = all good.
    """
    warnings = []
    op = layer.operation

    # Check tensor dimension limits
    for i, dim in enumerate(layer.input_dimensions):
        if dim.height > AXON_MAX_TENSOR_DIM:
            warnings.append(f"Layer {layer_idx}: input[{i}] height {dim.height} > max {AXON_MAX_TENSOR_DIM}")
        if dim.width > AXON_MAX_TENSOR_DIM:
            warnings.append(f"Layer {layer_idx}: input[{i}] width {dim.width} > max {AXON_MAX_TENSOR_DIM}")
        if dim.channel_cnt > AXON_MAX_TENSOR_DIM:
            warnings.append(f"Layer {layer_idx}: input[{i}] channels {dim.channel_cnt} > max {AXON_MAX_TENSOR_DIM}")

    od = layer.output_dimensions
    if od.height > AXON_MAX_TENSOR_DIM:
        warnings.append(f"Layer {layer_idx}: output height {od.height} > max {AXON_MAX_TENSOR_DIM}")
    if od.width > AXON_MAX_TENSOR_DIM:
        warnings.append(f"Layer {layer_idx}: output width {od.width} > max {AXON_MAX_TENSOR_DIM}")
    if od.channel_cnt > AXON_MAX_TENSOR_DIM:
        warnings.append(f"Layer {layer_idx}: output channels {od.channel_cnt} > max {AXON_MAX_TENSOR_DIM}")

    # Op-specific constraints
    if op == AxonOp.FULLY_CONNECTED:
        fd = layer.filter_dimensions
        in_size = fd.width   # width = in_features for FC
        out_size = fd.height  # height = out_features for FC
        if in_size > AXON_MAX_FC_INPUT:
            warnings.append(f"Layer {layer_idx}: FC input size {in_size} > max {AXON_MAX_FC_INPUT}")
        if out_size > AXON_MAX_FC_OUTPUT:
            warnings.append(f"Layer {layer_idx}: FC output size {out_size} > max {AXON_MAX_FC_OUTPUT}")

    elif op in (AxonOp.CONV2D, AxonOp.DEPTHWISE_CONV2D, AxonOp.POINTWISE_CONV2D):
        fd = layer.filter_dimensions
        if fd.height > AXON_MAX_CONV2D_FILTER:
            warnings.append(f"Layer {layer_idx}: conv filter height {fd.height} > max {AXON_MAX_CONV2D_FILTER}")
        if fd.width > AXON_MAX_CONV2D_FILTER:
            warnings.append(f"Layer {layer_idx}: conv filter width {fd.width} > max {AXON_MAX_CONV2D_FILTER}")
        if layer.stride_x > AXON_MAX_CONV_STRIDE:
            warnings.append(f"Layer {layer_idx}: conv stride_x {layer.stride_x} > max {AXON_MAX_CONV_STRIDE}")
        if layer.stride_y > AXON_MAX_CONV_STRIDE:
            warnings.append(f"Layer {layer_idx}: conv stride_y {layer.stride_y} > max {AXON_MAX_CONV_STRIDE}")

    elif op in (AxonOp.AVERAGE_POOLING, AxonOp.MAX_POOLING):
        fd = layer.filter_dimensions
        if fd.height > AXON_MAX_POOL_FILTER:
            warnings.append(f"Layer {layer_idx}: pool filter height {fd.height} > max {AXON_MAX_POOL_FILTER}")
        if fd.width > AXON_MAX_POOL_FILTER:
            warnings.append(f"Layer {layer_idx}: pool filter width {fd.width} > max {AXON_MAX_POOL_FILTER}")

    return warnings


def _create_rescale_layer(
    rescale_op: TosaOperator,
    tensor_to_layer: dict[str, int],
    layer_idx: int,
) -> AxonLayer:
    """Create an AXON layer for a standalone RESCALE operation.

    Standalone RESCALEs requantize tensors between different quantization
    domains (e.g., before ADD inputs need matching scales). We implement
    this as a POINTWISE_CONV2D with identity weights — each output channel
    copies the corresponding input channel, with the RESCALE mult/shift
    applied as the output requantization.

    RESCALE inputs: [data, multiplier, shift, input_zp, output_zp]
    """
    input_tensor = rescale_op.input_tensors[0]
    in_shape = input_tensor.shape  # [N, H, W, C] or [N, C]

    # Determine spatial dims and channels
    if len(in_shape) == 4:
        h, w, c = in_shape[1], in_shape[2], in_shape[3]
    elif len(in_shape) == 3:
        h, w, c = in_shape[0], in_shape[1], in_shape[2]
    elif len(in_shape) == 2:
        h, w, c = 1, 1, in_shape[1]
    else:
        h, w, c = 1, 1, in_shape[0] if in_shape else 1

    # Identity weights: C→C pointwise conv (1x1 kernel, identity per channel)
    identity = np.eye(c, dtype=np.int8)  # [C, C]
    # AXON expects OIHW: [out_channels, in_channels, 1, 1]
    identity = identity.reshape(c, c, 1, 1)
    filter_data = identity.tobytes()

    # Input zero point from RESCALE
    input_zp = 0
    if len(rescale_op.input_tensors) > 3 and rescale_op.input_tensors[3].data is not None:
        input_zp = int(rescale_op.input_tensors[3].data.flat[0])

    # Bias prime: identity conv weights have sum=1 per channel
    # b_prime[ch] = 0 + (-1 * input_zp) = -input_zp
    if input_zp != 0:
        bias_int32 = np.full(c, -input_zp, dtype=np.int32)
    else:
        bias_int32 = np.zeros(c, dtype=np.int32)
    bias_data = bias_int32.tobytes()

    # Rescale params (recomputed from TOSA's mult/shift using Nordic's algorithm)
    output_zp, multiplier_data, shift_data, scale_shift_cnt = _extract_rescale_params(rescale_op)

    input_id = _resolve_input_id(input_tensor, tensor_to_layer)

    # Use FC for 1x1 spatial, POINTWISE_CONV2D for spatial
    if h == 1 and w == 1:
        axon_op = AxonOp.FULLY_CONNECTED
        input_dims = AxonDimensions(height=1, width=c, channel_cnt=1)
        filter_dims = AxonDimensions(height=c, width=c, channel_cnt=1)
        output_dims = AxonDimensions(height=1, width=c, channel_cnt=1)
    else:
        axon_op = AxonOp.POINTWISE_CONV2D
        input_dims = AxonDimensions(height=h, width=w, channel_cnt=c)
        filter_dims = AxonDimensions(height=1, width=1, channel_cnt=c)
        output_dims = AxonDimensions(height=h, width=w, channel_cnt=c)

    logger.debug(f"  Creating rescale layer: {axon_op} {h}x{w}x{c} (identity conv)")

    return AxonLayer(
        input_ids=[input_id],
        operation=axon_op,
        input_dimensions=[input_dims],
        filter_dimensions=filter_dims,
        output_dimensions=output_dims,
        input_zero_point=0,  # b_prime handles input zp correction
        output_zero_point=output_zp,
        filter_data=filter_data,
        bias_data=bias_data,
        multiplier_data=multiplier_data,
        shift_data=shift_data,
        scale_shift_cnt=scale_shift_cnt,
    )


def _convert_conv2d(
    conv_op: TosaOperator,
    rescale_op: TosaOperator | None,
    tensor_to_layer: dict[str, int],
    layer_idx: int,
    pre_flatten_shape: tuple[int, int, int] | None = None,
) -> AxonLayer:
    """Convert a TOSA CONV2D (+ optional RESCALE) to an AXON layer."""

    # CONV2D inputs: [input, weights, bias, input_zp, weight_zp]
    input_tensor = conv_op.input_tensors[0]
    weight_tensor = conv_op.input_tensors[1]
    bias_tensor = conv_op.input_tensors[2]
    input_zp_tensor = conv_op.input_tensors[3]
    weight_zp_tensor = conv_op.input_tensors[4]
    output_tensor = conv_op.output_tensors[0]

    # Determine AXON operation type
    weight_shape = weight_tensor.shape  # [O, H, W, I] for CONV2D
    in_shape = input_tensor.shape  # [N, H, W, C]
    in_h = in_shape[1] if len(in_shape) > 1 else 1
    in_w = in_shape[2] if len(in_shape) > 2 else 1

    if len(weight_shape) == 4 and weight_shape[1] == 1 and weight_shape[2] == 1:
        if in_h == 1 and in_w == 1:
            # True FC: 1×1 conv on 1×1 spatial input
            axon_op = AxonOp.FULLY_CONNECTED
        else:
            # Pointwise conv on spatial input
            axon_op = AxonOp.POINTWISE_CONV2D
    else:
        axon_op = AxonOp.CONV2D

    # Dimension mapping for AXON:
    # AXON uses (height, width, channel_cnt) where for FC:
    #   input:  height=1, width=input_features, channel_cnt=1
    #   filter: height=1, width=output_features, channel_cnt=1
    #   output: height=1, width=output_features, channel_cnt=1
    # This matches TFLite's 2D tensor convention where shape=[batch, features]
    # maps to TensorShape(height=batch, width=features, depth=1).

    in_shape = input_tensor.shape  # TOSA: [N, H, W, C] or [N, features]

    if axon_op == AxonOp.FULLY_CONNECTED:
        # FC dimension mapping (verified from Nordic's TensorShape class):
        # TFLite FC input [batch, features] has shape.size==2 →
        #   height=batch(1), width=features, depth=1
        # Filter [outputs, inputs] →
        #   height=outputs, width=inputs, depth=1
        in_features = in_shape[-1]
        out_features = weight_shape[0]

        input_dims = AxonDimensions(
            height=1,
            width=in_features,
            channel_cnt=1,
            byte_width=AxonByteWidth.INT8,
        )
        filter_dims = AxonDimensions(
            height=out_features,
            width=in_features,
            channel_cnt=1,
            byte_width=AxonByteWidth.INT8,
        )
        output_dims = AxonDimensions(
            height=1,
            width=out_features,
            channel_cnt=1,
            byte_width=AxonByteWidth.INT8,
        )
    else:
        # Conv2D: standard NHWC → AXON HWC
        input_dims = AxonDimensions(
            height=in_shape[1] if len(in_shape) > 1 else 1,
            width=in_shape[2] if len(in_shape) > 2 else 1,
            channel_cnt=in_shape[3] if len(in_shape) > 3 else in_shape[-1],
            byte_width=AxonByteWidth.INT8,
        )
        filter_dims = AxonDimensions(
            height=weight_shape[1],
            width=weight_shape[2],
            channel_cnt=weight_shape[0],
            byte_width=AxonByteWidth.INT8,
        )
        out_shape = output_tensor.shape
        output_dims = AxonDimensions(
            height=out_shape[1] if len(out_shape) > 1 else 1,
            width=out_shape[2] if len(out_shape) > 2 else 1,
            channel_cnt=out_shape[3] if len(out_shape) > 3 else out_shape[-1],
            byte_width=AxonByteWidth.INT8,
        )

    # Zero points
    input_zp = int(input_zp_tensor.data.flat[0]) if input_zp_tensor.data is not None else 0
    weight_zp = int(weight_zp_tensor.data.flat[0]) if weight_zp_tensor.data is not None else 0

    # Padding, stride, dilation from TOSA attributes
    attrs = conv_op.attributes
    pad = attrs.get("pad", [0, 0, 0, 0])  # [top, bottom, left, right]
    stride = attrs.get("stride", [1, 1])   # [height, width]
    dilation = attrs.get("dilation", [1, 1])  # [height, width]

    # FC layers don't use stride/dilation — Nordic sets them to 0
    if axon_op == AxonOp.FULLY_CONNECTED:
        stride = [0, 0]
        dilation = [0, 0]

    # Weights: TOSA stores as [O, H, W, I], AXON expects [O, I, H, W] (NCHW/OIHW)
    if weight_tensor.data is not None:
        weights = weight_tensor.data.astype(np.int8)
        if weights.ndim == 4:
            weights = weights.transpose(0, 3, 1, 2)  # OHWI → OIHW
        elif weights.ndim == 3:
            weights = weights.transpose(2, 0, 1)

        filter_data = weights.tobytes()
    else:
        filter_data = b""

    # Bias prime for ALL layer types: b_prime = bias + (-sum(weight[ch]) * input_zp)
    # Nordic uses BOTH b_prime AND input_zp in the struct simultaneously.
    # Verified from Nordic's intermediate binary for FC, Conv, and Pool layers.
    if bias_tensor.raw_bytes:
        bias_int32 = np.frombuffer(bias_tensor.raw_bytes, dtype=np.int32).copy()
        if weight_tensor.data is not None and input_zp != 0:
            weights_orig = weight_tensor.data.astype(np.int32)
            for ch in range(min(weights_orig.shape[0], len(bias_int32))):
                kernel_sum = int(np.sum(weights_orig[ch]))
                bias_int32[ch] += -kernel_sum * input_zp
        bias_data = bias_int32.tobytes()
    else:
        bias_data = b""

    # Quantization from RESCALE op
    output_zp = 0
    multiplier_data = b""
    shift_data = b""
    scale_shift_cnt = 0
    activation = AxonActivation.DISABLED

    if rescale_op:
        output_zp, multiplier_data, shift_data, scale_shift_cnt = _extract_rescale_params(rescale_op)

    input_id = _resolve_input_id(input_tensor, tensor_to_layer)

    return AxonLayer(
        input_ids=[input_id],
        operation=axon_op,
        input_dimensions=[input_dims],
        filter_dimensions=filter_dims,
        output_dimensions=output_dims,
        stride_x=stride[1] if len(stride) > 1 else 1,
        stride_y=stride[0] if len(stride) > 0 else 1,
        dilation_x=dilation[1] if len(dilation) > 1 else 1,
        dilation_y=dilation[0] if len(dilation) > 0 else 1,
        input_zero_point=input_zp,  # Nordic uses both b_prime AND input_zp
        output_zero_point=output_zp,
        pad_top=pad[0] if len(pad) > 0 else 0,
        pad_bottom=pad[1] if len(pad) > 1 else 0,
        pad_left=pad[2] if len(pad) > 2 else 0,
        pad_right=pad[3] if len(pad) > 3 else 0,
        filter_data=filter_data,
        bias_data=bias_data,
        multiplier_data=multiplier_data,
        shift_data=shift_data,
        scale_shift_cnt=scale_shift_cnt,
        activation=activation,
    )


def _convert_depthwise_conv2d(
    conv_op: TosaOperator,
    rescale_op: TosaOperator | None,
    tensor_to_layer: dict[str, int],
    layer_idx: int,
    as_conv2d: bool = False,
) -> AxonLayer:
    """Convert TOSA DEPTHWISE_CONV2D (+ optional RESCALE) to AXON layer.

    TOSA DEPTHWISE_CONV2D inputs: [input, weights, bias, input_zp, weight_zp]
    Weights shape: [KH, KW, C_in, M] where C_out = C_in * M.
    """
    input_tensor = conv_op.input_tensors[0]
    weight_tensor = conv_op.input_tensors[1]
    bias_tensor = conv_op.input_tensors[2]
    input_zp_tensor = conv_op.input_tensors[3]
    output_tensor = conv_op.output_tensors[0]

    in_shape = input_tensor.shape   # [N, H, W, C_in]
    out_shape = output_tensor.shape  # [N, OH, OW, C_out]
    weight_shape = weight_tensor.shape  # [KH, KW, C_in, M]

    kh, kw = weight_shape[0], weight_shape[1]
    c_in = weight_shape[2]
    depth_mult = weight_shape[3]
    out_channels = c_in * depth_mult

    attrs = conv_op.attributes
    pad = attrs.get("pad", [0, 0, 0, 0])
    stride = attrs.get("stride", [1, 1])
    dilation = attrs.get("dilation", [1, 1])

    input_zp = int(input_zp_tensor.data.flat[0]) if input_zp_tensor.data is not None else 0

    # Weights transpose
    if weight_tensor.data is not None:
        weights = weight_tensor.data.astype(np.int8)
        if as_conv2d:
            # Convert DW format [KH, KW, C_in=1, M] to CONV2D format [O, I, KH, KW]
            # [KH, KW, 1, M] → [M, KH, KW, 1] → [O, I, H, W] with I=1
            weights = weights.reshape(kh, kw, out_channels)  # squeeze c_in=1
            weights = weights.transpose(2, 0, 1)  # [O, KH, KW]
            weights = weights.reshape(out_channels, 1, kh, kw)  # [O, I=1, KH, KW]
            # Then OHWI→OIHW: already in OIHW format
        else:
            # DW: TOSA [KH, KW, C_in, M] → AXON [C_out, 1, KH, KW]
            weights = weights.reshape(kh, kw, out_channels)
            weights = weights.transpose(2, 0, 1)  # [C_out, KH, KW]
            weights = weights.reshape(out_channels, 1, kh, kw)
        filter_data = weights.tobytes()
    else:
        filter_data = b""

    # Bias prime for DW conv (same as Conv2D — Nordic uses b_prime for all types).
    if bias_tensor.raw_bytes:
        bias_int32 = np.frombuffer(bias_tensor.raw_bytes, dtype=np.int32).copy()
        if weight_tensor.data is not None and input_zp != 0:
            weights_orig = weight_tensor.data.astype(np.int32)
            for ch in range(min(out_channels, len(bias_int32))):
                kernel_sum = int(np.sum(weights_orig[:, :, ch % c_in, ch // c_in]))
                bias_int32[ch] += -kernel_sum * input_zp
        bias_data = bias_int32.tobytes()
    else:
        bias_data = b""

    # Rescale
    output_zp = 0
    multiplier_data = b""
    shift_data = b""
    scale_shift_cnt = 0
    if rescale_op:
        output_zp, multiplier_data, shift_data, scale_shift_cnt = _extract_rescale_params(rescale_op)

    input_dims = AxonDimensions(
        height=in_shape[1], width=in_shape[2],
        channel_cnt=in_shape[3] if len(in_shape) > 3 else in_shape[-1],
        byte_width=AxonByteWidth.INT8,
    )
    # For as_conv2d: filter_dims.c = input_channels (1), not output_channels
    # Nordic Conv2D: filter c = input channels per filter
    filter_dims = AxonDimensions(
        height=kh, width=kw,
        channel_cnt=c_in if as_conv2d else out_channels,
        byte_width=AxonByteWidth.INT8,
    )
    output_dims = AxonDimensions(
        height=out_shape[1], width=out_shape[2],
        channel_cnt=out_shape[3] if len(out_shape) > 3 else out_shape[-1],
        byte_width=AxonByteWidth.INT8,
    )

    return AxonLayer(
        input_ids=[_resolve_input_id(input_tensor, tensor_to_layer)],
        operation=AxonOp.CONV2D if as_conv2d else AxonOp.DEPTHWISE_CONV2D,
        input_dimensions=[input_dims],
        filter_dimensions=filter_dims,
        output_dimensions=output_dims,
        stride_x=stride[1] if len(stride) > 1 else 1,
        stride_y=stride[0] if len(stride) > 0 else 1,
        dilation_x=dilation[1] if len(dilation) > 1 else 1,
        dilation_y=dilation[0] if len(dilation) > 0 else 1,
        input_zero_point=input_zp,  # Nordic uses both b_prime AND input_zp
        output_zero_point=output_zp,
        pad_top=pad[0] if len(pad) > 0 else 0,
        pad_bottom=pad[1] if len(pad) > 1 else 0,
        pad_left=pad[2] if len(pad) > 2 else 0,
        pad_right=pad[3] if len(pad) > 3 else 0,
        filter_data=filter_data,
        bias_data=bias_data,
        multiplier_data=multiplier_data,
        shift_data=shift_data,
        scale_shift_cnt=scale_shift_cnt,
    )


def _convert_elementwise(
    op: TosaOperator,
    rescale_op: TosaOperator | None,
    tensor_to_layer: dict[str, int],
    layer_idx: int,
    axon_op: int,
    graph: TosaGraph | None = None,
    tensor_to_zp: dict[str, int] | None = None,
    tensor_rescale_info: dict | None = None,
) -> AxonLayer:
    """Convert TOSA ADD or MUL (+ optional RESCALE) to AXON ADD2 or MULTIPLY.

    TOSA ADD inputs: [input1, input2]
    TOSA MUL inputs: [input1, input2, shift]
    Both support broadcasting on H/W dimensions.

    For MULTIPLY with a constant second input (scalar multiply), the constant
    is stored as filter_data. Nordic's AXON MUL reads the multiplier constant
    from the filter data, not from a layer input.
    """
    input1 = op.input_tensors[0]
    input2 = op.input_tensors[1]
    output_tensor = op.output_tensors[0]

    def _dims_from_shape(shape: list[int]) -> AxonDimensions:
        if len(shape) == 4:
            return AxonDimensions(height=shape[1], width=shape[2], channel_cnt=shape[3])
        elif len(shape) == 3:
            return AxonDimensions(height=shape[0], width=shape[1], channel_cnt=shape[2])
        elif len(shape) == 2:
            # 2D tensor [batch, features] → h=1, w=features, c=1 (matches FC convention)
            return AxonDimensions(height=1, width=shape[1], channel_cnt=1)
        else:
            return AxonDimensions(height=1, width=shape[0] if shape else 1, channel_cnt=1)

    input1_dims = _dims_from_shape(input1.shape)
    input2_dims = _dims_from_shape(input2.shape)
    output_dims = _dims_from_shape(output_tensor.shape)

    output_zp = 0
    multiplier_data = b""
    shift_data = b""
    scale_shift_cnt = 0
    bias_data = b""
    if rescale_op:
        output_zp, multiplier_data, shift_data, scale_shift_cnt = _extract_rescale_params(rescale_op)

    input1_id = _resolve_input_id(input1, tensor_to_layer)
    input2_id = _resolve_input_id(input2, tensor_to_layer)

    # For MUL with a constant second input, store the constant as filter_data.
    # Nordic's AXON MUL expects the multiplier constant in the filter data.
    # The filter_dimensions should be 0x0x0 (Nordic convention for MUL).
    #
    # The TOSA graph may have RESCALE(const) → MUL, where the constant comes
    # through a RESCALE op. In that case, input2.data is None but we can
    # compute the rescaled value from the RESCALE's input constant.
    filter_data = b""
    filter_dims = AxonDimensions()  # default 1x1x1
    if axon_op == AxonOp.MULTIPLY:
        mul_const = None
        if input2.data is not None:
            mul_const = input2.data.astype(np.int8)
        elif input2_id == -1:
            # Input2 might come from a RESCALE of a constant (not tracked as a layer).
            # Search the graph for a RESCALE that outputs this tensor with a constant input.
            for prev_op in graph.get_non_const_operators():
                if (prev_op.op_name == "RESCALE" and prev_op.output_tensors
                        and prev_op.output_tensors[0].name == input2.name):
                    # Found the RESCALE that produces input2
                    const_input = prev_op.input_tensors[0]
                    if const_input.data is not None:
                        # Compute the rescaled constant value
                        in_val = int(const_input.data.flat[0])
                        r_mult = prev_op.input_tensors[1]
                        r_shift = prev_op.input_tensors[2]
                        r_in_zp = prev_op.input_tensors[3]
                        r_out_zp = prev_op.input_tensors[4]
                        t_mult = int(np.frombuffer(r_mult.raw_bytes, dtype=np.int32)[0]) if r_mult.raw_bytes else 0
                        t_shift = int(r_shift.data.flat[0]) if r_shift.data is not None else 0
                        t_in_zp = int(r_in_zp.data.flat[0]) if r_in_zp.data is not None else 0
                        t_out_zp = int(r_out_zp.data.flat[0]) if r_out_zp.data is not None else 0
                        if t_shift > 0 and t_mult != 0:
                            rescaled = round((in_val - t_in_zp) * t_mult / (2 ** t_shift)) + t_out_zp
                        else:
                            rescaled = in_val
                        rescaled = max(-128, min(127, rescaled))
                        mul_const = np.array([rescaled], dtype=np.int8)
                        mul_const_zp = t_in_zp  # constant's quantization zero point
                        logger.debug(f"  MUL: computed constant from RESCALE: "
                                    f"input={in_val} → rescaled={rescaled} (const_zp={t_in_zp})")
                    break

        if mul_const is not None:
            filter_data = mul_const.tobytes()
            # For 1D input (h=1), Nordic uses flt=0x0x0. For spatial input, use 1x1x1 (broadcast).
            if input1_dims.height <= 1:
                filter_dims = AxonDimensions(height=0, width=0, channel_cnt=0)
            else:
                filter_dims = AxonDimensions(height=1, width=1, channel_cnt=1)
            # Nordic MUL with constant: input_id_cnt=1 (only activation input).
            # The constant is stored as filter_data, NOT as a second layer input.
            input2_id = None  # will be excluded from input_ids
            input_zp_for_mul = mul_const_zp if 'mul_const_zp' in dir() else 0
            logger.debug(f"  MUL constant: {mul_const.flat[:4]} zp={input_zp_for_mul} stored as filter_data")
        else:
            logger.warning(f"  MUL: no constant found for input2 — "
                          f"AXON MUL requires a constant multiplier")
    elif axon_op == AxonOp.ADD2:
        # ADD: filter_dims = input_dims (Nordic convention)
        filter_dims = input1_dims

        # Nordic ADD uses TWO multipliers (one per input) + bias:
        #   acc = in1 * mult_a + in2 * mult_b + bias
        #   out = (acc >> shift) + out_zp
        #
        # mult_a = round(s_in1/s_out * 2^shift)
        # mult_b = round(s_in2/s_out * 2^shift)
        # bias = round(-(s_in1*zp_in1 + s_in2*zp_in2) / s_out * 2^shift)
        #
        # The scale ratios come from the skipped standalone RESCALEs.
        # tensor_rescale_info[name] = (scale_ratio, input_zp)
        _zp_map = tensor_to_zp or {}
        _ri_map = tensor_rescale_info or {}

        zp1 = _zp_map.get(input1.name, 0)
        zp2 = _zp_map.get(input2.name, 0)
        ri1 = _ri_map.get(input1.name)  # (scale_ratio, in_zp) or None
        ri2 = _ri_map.get(input2.name)

        if ri1 and ri2 and multiplier_data:
            # Both inputs have skipped rescale info — compute proper two multipliers.
            #
            # The skipped RESCALEs have scale_ratio = s_in / s_add_internal.
            # The ADD's fused RESCALE has scale = s_add_internal / s_out.
            # Net scale per input: s_in / s_out = scale_ratio * add_output_scale.
            add_mult = np.frombuffer(multiplier_data, dtype=np.int32)
            add_shift = np.frombuffer(shift_data, dtype=np.int8)
            add_output_scale = float(add_mult[0]) / (2.0 ** int(add_shift[0]))

            net_s1 = ri1[0] * add_output_scale  # s_in1 / s_out
            net_s2 = ri2[0] * add_output_scale  # s_in2 / s_out

            # Use Nordic's formula with bit_limit=15
            best_shift = 8
            for s in range(8, 31):
                m1 = abs(int(round(net_s1 * (2 ** s))))
                m2 = abs(int(round(net_s2 * (2 ** s))))
                zp_check = abs(int(round(output_zp * (2 ** s))))
                if m1 < (1 << 15) and m2 < (1 << 15) and zp_check < (1 << 31):
                    best_shift = s
                else:
                    break

            mult_a = abs(int(round(net_s1 * (2 ** best_shift))))
            mult_b = abs(int(round(net_s2 * (2 ** best_shift))))

            # Bias: use exact float scales (Nordic formula, line 1995):
            #   bias = round(-(s1*zp1 + s2*zp2) / s_out * 2^shift)
            # Using net_s1/net_s2 directly (which ARE s_in/s_out already):
            add_bias = int(round(-(net_s1 * zp1 + net_s2 * zp2) * (2 ** best_shift)))

            multiplier_data = np.array([mult_a, mult_b], dtype=np.int32).tobytes()
            shift_data = np.array([best_shift], dtype=np.int8).tobytes()
            scale_shift_cnt = 1  # cnt=1 in Nordic (the shift count, not mult count)
            bias_data = np.array([add_bias], dtype=np.int32).tobytes()

            logger.info(f"  ADD: two multipliers: mult_a={mult_a} mult_b={mult_b} "
                        f"shift={best_shift} bias={add_bias} (zp1={zp1} zp2={zp2})")
        else:
            # No rescale info — use simple bias and scale=1.0
            add_bias = -(zp1 + zp2)
            bias_data = np.array([add_bias], dtype=np.int32).tobytes()

            # Override tiny TOSA scale with 1.0
            if multiplier_data:
                mult_arr = np.frombuffer(multiplier_data, dtype=np.int32)
                shift_arr = np.frombuffer(shift_data, dtype=np.int8)
                scale = float(mult_arr[0]) / (2.0 ** int(shift_arr[0]))
                if scale < 0.01:
                    multiplier_data = np.array([16384], dtype=np.int32).tobytes()
                    shift_data = np.array([14], dtype=np.int8).tobytes()

            logger.debug(f"  ADD: simple bias={add_bias} (zp1={zp1} zp2={zp2})")

    # For MUL with constant: input_zero_point = constant's quantization zero point
    # For ADD: input_zero_point = 0 (Nordic convention)
    input_zp = input_zp_for_mul if (axon_op == AxonOp.MULTIPLY and filter_data) else 0

    # Build input_ids and dims: MUL constant is in filter_data, not a layer input
    if input2_id is not None:
        input_ids = [input1_id, input2_id]
        input_dims_list = [input1_dims, input2_dims]
    else:
        input_ids = [input1_id]
        input_dims_list = [input1_dims]

    return AxonLayer(
        input_ids=input_ids,
        operation=axon_op,
        input_dimensions=input_dims_list,
        filter_dimensions=filter_dims,
        output_dimensions=output_dims,
        stride_x=0,      # Nordic uses 0 for elementwise ops
        stride_y=0,
        dilation_x=0,
        dilation_y=0,
        input_zero_point=input_zp,
        output_zero_point=output_zp,
        filter_data=filter_data,
        bias_data=bias_data,
        multiplier_data=multiplier_data,
        shift_data=shift_data,
        scale_shift_cnt=scale_shift_cnt,
    )


def _convert_pool2d(
    op: TosaOperator,
    rescale_op: TosaOperator | None,
    tensor_to_layer: dict[str, int],
    layer_idx: int,
    axon_op: int,
    input_zp_from_graph: int = 0,
) -> AxonLayer:
    """Convert TOSA AVG_POOL2D or MAX_POOL2D to AXON pooling layer.

    AVG_POOL2D inputs: [input, input_zp, output_zp]
    MAX_POOL2D inputs: [input]
    Both have kernel, pad, stride attributes.
    """
    input_tensor = op.input_tensors[0]
    output_tensor = op.output_tensors[0]
    in_shape = input_tensor.shape   # [N, H, W, C]
    out_shape = output_tensor.shape

    attrs = op.attributes
    kernel = attrs.get("kernel", [1, 1])  # [KH, KW]
    pad = attrs.get("pad", [0, 0, 0, 0])
    stride = attrs.get("stride", [1, 1])

    # Zero points (AVG_POOL2D has input_zp/output_zp as tensor inputs)
    # MAX_POOL2D has no zp tensors — use the propagated input_zp from the graph
    input_zp = input_zp_from_graph if axon_op == AxonOp.MAX_POOLING else 0
    output_zp = 0
    if axon_op == AxonOp.AVERAGE_POOLING and len(op.input_tensors) >= 3:
        zp_in = op.input_tensors[1]
        zp_out = op.input_tensors[2]
        if zp_in.data is not None:
            input_zp = int(zp_in.data.flat[0])
        if zp_out.data is not None:
            output_zp = int(zp_out.data.flat[0])

    multiplier_data = b""
    shift_data = b""
    scale_shift_cnt = 0
    if rescale_op:
        output_zp, multiplier_data, shift_data, scale_shift_cnt = _extract_rescale_params(rescale_op)

    # Pooling layers need valid rescale params even without explicit RESCALE.
    # Nordic's approach: compute b_prime (bias) and mult/shift for the pool.
    kh = kernel[0] if len(kernel) > 0 else 1
    kw = kernel[1] if len(kernel) > 1 else 1
    kernel_area = kh * kw
    bias_data = b""

    if scale_shift_cnt == 0:
        if axon_op == AxonOp.MAX_POOLING:
            # MAX_POOL: identity rescale. Nordic uses mult=1, shift=0 but
            # shift=0 may be out of AXON's valid range [1,32].
            # Use mult=2, shift=1 as safe identity: (val * 2 + 1) >> 1 = val
            multiplier_data = np.array([2], dtype=np.int32).tobytes()
            shift_data = np.array([1], dtype=np.int8).tobytes()
            scale_shift_cnt = 1
            # MAX_POOL preserves quantization: Nordic sets output_zp=0
            # and input_zp=actual (the preceding layer's output_zp)
            output_zp = 0

        elif axon_op == AxonOp.AVERAGE_POOLING and kernel_area > 1:
            # AVG_POOL: AXON sums values, rescale divides by kernel area.
            # Following Nordic's CalculateMultiplierandScaleshift:
            #   scale = 1/kernel_area (when input/output scales match)
            #   b_prime = -input_zp * kernel_area (zero-point correction for sum)
            #   mult/shift chosen so mult * max_accumulator < 2^31
            #   Also: output_zp * 2^shift must fit in 2^bit_limit
            scale = 1.0 / kernel_area
            bit_limit = 25  # Nordic uses 25 for same-scale pool

            # Search for optimal shift (Nordic's optimized_ip_scaling_shift)
            best_shift = 8
            for s in range(8, 31):
                m = abs(round(scale * (1 << s)))
                zp_scaled = abs(round(output_zp * (1 << s)))
                if m < (1 << bit_limit) and zp_scaled < (1 << bit_limit):
                    best_shift = s
                else:
                    break
            mult_val = abs(round(scale * (1 << best_shift)))
            shift_val = best_shift

            # b_prime: zero-point correction (Nordic: -input_zp * kernel_area)
            b_prime = round(-input_zp * kernel_area)
            bias_data = np.array([b_prime], dtype=np.int32).tobytes()

            multiplier_data = np.array([mult_val], dtype=np.int32).tobytes()
            shift_data = np.array([shift_val], dtype=np.int8).tobytes()
            scale_shift_cnt = 1

            logger.debug(f"  AVG_POOL: area={kernel_area} b_prime={b_prime} "
                        f"mult={mult_val} shift={shift_val}")
        else:
            # Fallback: identity rescale
            multiplier_data = np.array([2], dtype=np.int32).tobytes()
            shift_data = np.array([1], dtype=np.int8).tobytes()
            scale_shift_cnt = 1

    in_channels = in_shape[3] if len(in_shape) > 3 else in_shape[-1]

    input_dims = AxonDimensions(
        height=in_shape[1] if len(in_shape) > 1 else 1,
        width=in_shape[2] if len(in_shape) > 2 else 1,
        channel_cnt=in_channels,
        byte_width=AxonByteWidth.INT8,
    )
    # Filter dimensions encode the kernel size for pooling
    # Nordic uses channel_cnt=0 for MAX_POOL, channel_cnt for AVG_POOL/MEAN
    filter_ch = 0 if axon_op == AxonOp.MAX_POOLING else in_channels
    filter_dims = AxonDimensions(
        height=kernel[0] if len(kernel) > 0 else 1,
        width=kernel[1] if len(kernel) > 1 else 1,
        channel_cnt=filter_ch,
        byte_width=AxonByteWidth.INT8,
    )
    output_dims = AxonDimensions(
        height=out_shape[1] if len(out_shape) > 1 else 1,
        width=out_shape[2] if len(out_shape) > 2 else 1,
        channel_cnt=out_shape[3] if len(out_shape) > 3 else out_shape[-1],
        byte_width=AxonByteWidth.INT8,
    )

    return AxonLayer(
        input_ids=[_resolve_input_id(input_tensor, tensor_to_layer)],
        operation=axon_op,
        input_dimensions=[input_dims],
        filter_dimensions=filter_dims,
        output_dimensions=output_dims,
        stride_x=stride[1] if len(stride) > 1 else 1,
        stride_y=stride[0] if len(stride) > 0 else 1,
        dilation_x=0,  # Nordic uses 0 for pool dilation (not 1)
        dilation_y=0,
        input_zero_point=input_zp,
        output_zero_point=output_zp,
        pad_top=pad[0] if len(pad) > 0 else 0,
        pad_bottom=pad[1] if len(pad) > 1 else 0,
        pad_left=pad[2] if len(pad) > 2 else 0,
        pad_right=pad[3] if len(pad) > 3 else 0,
        bias_data=bias_data,
        multiplier_data=multiplier_data,
        shift_data=shift_data,
        scale_shift_cnt=scale_shift_cnt,
    )


def _convert_reduce_sum(
    op: TosaOperator,
    rescale_op: TosaOperator | None,
    tensor_to_layer: dict[str, int],
    layer_idx: int,
) -> AxonLayer:
    """Convert TOSA REDUCE_SUM to AXON MEAN (global average pooling).

    TOSA REDUCE_SUM inputs: [input]
    Attribute: axis (which dimension to reduce).
    AXON MEAN uses concatenate_axis field for the reduction axis.
    """
    input_tensor = op.input_tensors[0]
    output_tensor = op.output_tensors[0]
    in_shape = input_tensor.shape
    out_shape = output_tensor.shape

    attrs = op.attributes
    tosa_axis = attrs.get("axis", 1)

    # Map TOSA NHWC axis to AXON axis enum: 0=CHANNEL, 1=HEIGHT, 2=WIDTH
    # TOSA axis: 0=N, 1=H, 2=W, 3=C
    axon_axis_map = {1: 1, 2: 2, 3: 0}  # H→HEIGHT, W→WIDTH, C→CHANNEL
    axon_axis = axon_axis_map.get(tosa_axis, 1)

    output_zp = 0
    multiplier_data = b""
    shift_data = b""
    scale_shift_cnt = 0
    if rescale_op:
        output_zp, multiplier_data, shift_data, scale_shift_cnt = _extract_rescale_params(rescale_op)

    in_channels = in_shape[3] if len(in_shape) > 3 else in_shape[-1]
    input_dims = AxonDimensions(
        height=in_shape[1] if len(in_shape) > 1 else 1,
        width=in_shape[2] if len(in_shape) > 2 else 1,
        channel_cnt=in_channels,
        byte_width=AxonByteWidth.INT8,
    )
    output_dims = AxonDimensions(
        height=out_shape[1] if len(out_shape) > 1 else 1,
        width=out_shape[2] if len(out_shape) > 2 else 1,
        channel_cnt=out_shape[3] if len(out_shape) > 3 else out_shape[-1],
        byte_width=AxonByteWidth.INT8,
    )

    return AxonLayer(
        input_ids=[_resolve_input_id(input_tensor, tensor_to_layer)],
        operation=AxonOp.MEAN,
        concatenate_axis=axon_axis,
        input_dimensions=[input_dims],
        output_dimensions=output_dims,
        output_zero_point=output_zp,
        multiplier_data=multiplier_data,
        shift_data=shift_data,
        scale_shift_cnt=scale_shift_cnt,
    )


def _convert_concat(
    op: TosaOperator,
    tensor_to_layer: dict[str, int],
    layer_idx: int,
) -> AxonLayer:
    """Convert TOSA CONCAT to AXON CONCATENATE layer.

    TOSA CONCAT inputs: [tensor1, tensor2, ...] (2+ tensors to concatenate)
    Attribute: axis (NHWC dimension to concatenate along).
    AXON supports up to 4 inputs via input_ids.
    """
    output_tensor = op.output_tensors[0]
    out_shape = output_tensor.shape

    attrs = op.attributes
    tosa_axis = attrs.get("axis", 3)  # Default: channel axis (last dim in NHWC)

    # Map TOSA NHWC axis to AXON axis enum: 0=CHANNEL, 1=HEIGHT, 2=WIDTH
    # TOSA axis: 0=N, 1=H, 2=W, 3=C
    axon_axis_map = {1: 1, 2: 2, 3: 0}
    axon_axis = axon_axis_map.get(tosa_axis, 0)

    # Collect input IDs and dimensions
    input_ids = []
    input_dims = []
    for inp in op.input_tensors:
        input_ids.append(_resolve_input_id(inp, tensor_to_layer))
        shape = inp.shape
        if len(shape) == 4:
            input_dims.append(AxonDimensions(height=shape[1], width=shape[2], channel_cnt=shape[3]))
        elif len(shape) == 3:
            input_dims.append(AxonDimensions(height=shape[0], width=shape[1], channel_cnt=shape[2]))
        else:
            input_dims.append(AxonDimensions(height=1, width=shape[0] if shape else 1, channel_cnt=1))

    # Output dimensions
    if len(out_shape) == 4:
        output_dims = AxonDimensions(height=out_shape[1], width=out_shape[2], channel_cnt=out_shape[3])
    elif len(out_shape) == 3:
        output_dims = AxonDimensions(height=out_shape[0], width=out_shape[1], channel_cnt=out_shape[2])
    else:
        output_dims = AxonDimensions(height=1, width=out_shape[0] if out_shape else 1, channel_cnt=1)

    return AxonLayer(
        input_ids=input_ids,
        operation=AxonOp.CONCATENATE,
        concatenate_axis=axon_axis,
        input_dimensions=input_dims,
        output_dimensions=output_dims,
    )


def _convert_slice(
    op: TosaOperator,
    tensor_to_layer: dict[str, int],
    layer_idx: int,
) -> AxonLayer:
    """Convert TOSA SLICE to AXON STRIDED_SLICE layer.

    TOSA SLICE inputs: [input, start_tensor, size_tensor]
    start and size are constant tensors with indices per dimension.
    AXON STRIDED_SLICE uses begin/end/strides arrays in CHW order.
    TOSA SLICE has no strides (all = 1), so we set strides to 1.

    The strided slice parameters are packed as 9 × int32 in the filter_data
    field: [begin_c, begin_h, begin_w, end_c, end_h, end_w, stride_c, stride_h, stride_w].
    """
    input_tensor = op.input_tensors[0]
    output_tensor = op.output_tensors[0]
    in_shape = input_tensor.shape  # [N, H, W, C]
    out_shape = output_tensor.shape

    # Extract start and size from constant input tensors
    start_tensor = op.input_tensors[1] if len(op.input_tensors) > 1 else None
    size_tensor = op.input_tensors[2] if len(op.input_tensors) > 2 else None

    if start_tensor is not None and start_tensor.data is not None:
        start = start_tensor.data.flatten().tolist()
    else:
        start = [0] * len(in_shape)

    if size_tensor is not None and size_tensor.data is not None:
        size = size_tensor.data.flatten().tolist()
    else:
        size = list(out_shape)

    # Convert NHWC start/size to AXON CHW begin/end/strides
    # TOSA: [N, H, W, C], AXON: [C, H, W]
    if len(start) == 4:
        begin = [int(start[3]), int(start[1]), int(start[2])]  # C, H, W
        end = [int(start[3] + size[3]), int(start[1] + size[1]), int(start[2] + size[2])]
    elif len(start) == 3:
        begin = [int(start[2]), int(start[0]), int(start[1])]
        end = [int(start[2] + size[2]), int(start[0] + size[0]), int(start[1] + size[1])]
    else:
        begin = [0, 0, 0]
        end = [int(size[0]) if size else 1, 1, 1]

    strides = [1, 1, 1]  # TOSA SLICE has no strides

    # Pack as 9 × int32: begin[3] + end[3] + strides[3]
    params = np.array(begin + end + strides, dtype=np.int32)
    filter_data = params.tobytes()

    # Dimensions
    if len(in_shape) == 4:
        input_dims = AxonDimensions(height=in_shape[1], width=in_shape[2], channel_cnt=in_shape[3])
        output_dims = AxonDimensions(height=out_shape[1], width=out_shape[2], channel_cnt=out_shape[3])
    else:
        input_dims = AxonDimensions(height=1, width=in_shape[0] if in_shape else 1, channel_cnt=1)
        output_dims = AxonDimensions(height=1, width=out_shape[0] if out_shape else 1, channel_cnt=1)

    return AxonLayer(
        input_ids=[_resolve_input_id(input_tensor, tensor_to_layer)],
        operation=AxonOp.STRIDED_SLICE,
        input_dimensions=[input_dims],
        output_dimensions=output_dims,
        filter_data=filter_data,
    )


def _convert_pad(
    op: TosaOperator,
    tensor_to_layer: dict[str, int],
    layer_idx: int,
) -> AxonLayer:
    """Convert TOSA PAD to AXON CHANNEL_PADDING layer.

    TOSA PAD inputs: [input, padding_tensor, pad_const_tensor]
    padding_tensor shape: [N_dims, 2] with [before, after] per dimension.
    AXON CHANNEL_PADDING uses the standard padding fields and only supports
    channel-dimension padding (top/bottom must be 0).
    """
    input_tensor = op.input_tensors[0]
    output_tensor = op.output_tensors[0]
    in_shape = input_tensor.shape
    out_shape = output_tensor.shape

    # Extract padding values from constant tensor
    pad_tensor = op.input_tensors[1] if len(op.input_tensors) > 1 else None
    padding = [[0, 0]] * len(in_shape)
    if pad_tensor is not None and pad_tensor.data is not None:
        pad_data = pad_tensor.data.reshape(-1, 2).tolist()
        for i, (before, after) in enumerate(pad_data):
            if i < len(padding):
                padding[i] = [int(before), int(after)]

    # NHWC: padding[0]=N, [1]=H, [2]=W, [3]=C
    if len(padding) >= 4:
        pad_top, pad_bottom = padding[1]
        pad_left, pad_right = padding[2]
        # Channel padding stored as pad_top/pad_bottom of the channel axis
        # AXON expects front/back channel padding
    elif len(padding) >= 3:
        pad_top, pad_bottom = padding[0]
        pad_left, pad_right = padding[1]
    else:
        pad_top = pad_bottom = pad_left = pad_right = 0

    if len(in_shape) == 4:
        input_dims = AxonDimensions(height=in_shape[1], width=in_shape[2], channel_cnt=in_shape[3])
        output_dims = AxonDimensions(height=out_shape[1], width=out_shape[2], channel_cnt=out_shape[3])
    else:
        input_dims = AxonDimensions(height=1, width=in_shape[0] if in_shape else 1, channel_cnt=1)
        output_dims = AxonDimensions(height=1, width=out_shape[0] if out_shape else 1, channel_cnt=1)

    return AxonLayer(
        input_ids=[_resolve_input_id(input_tensor, tensor_to_layer)],
        operation=AxonOp.CHANNEL_PADDING,
        input_dimensions=[input_dims],
        output_dimensions=output_dims,
        pad_top=pad_top,
        pad_bottom=pad_bottom,
        pad_left=pad_left,
        pad_right=pad_right,
    )


def _convert_persistent_var(
    op: TosaOperator,
    tensor_to_layer: dict[str, int],
    layer_idx: int,
) -> AxonLayer:
    """Convert TOSA VARIABLE/VARIABLE_READ/VARIABLE_WRITE to AXON PERSISTENT_VAR.

    Used in streaming/stateful models to persist intermediate results between
    inference calls. The persistent buffer is allocated separately from the
    interlayer buffer.

    TOSA VARIABLE declares the var, VARIABLE_READ reads, VARIABLE_WRITE writes.
    """
    output_tensor = op.output_tensors[0] if op.output_tensors else op.input_tensors[0]
    shape = output_tensor.shape

    if len(shape) == 4:
        dims = AxonDimensions(height=shape[1], width=shape[2], channel_cnt=shape[3])
    elif len(shape) == 3:
        dims = AxonDimensions(height=shape[0], width=shape[1], channel_cnt=shape[2])
    else:
        total = 1
        for s in shape:
            total *= s
        dims = AxonDimensions(height=1, width=total, channel_cnt=1)

    input_id = -1
    if op.input_tensors:
        input_id = _resolve_input_id(op.input_tensors[0], tensor_to_layer)

    return AxonLayer(
        input_ids=[input_id],
        operation=AxonOp.PERSISTENT_VAR,
        input_dimensions=[dims],
        output_dimensions=dims,
    )

