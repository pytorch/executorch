# Copyright (c) 2026 iote.ai
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""TOSA → AXON compiler bridge.

Reads a TOSA flatbuffer, converts operations to AXON layer descriptors,
packs them into the AXON intermediate binary format, and calls Nordic's
compiler library to produce command buffers.

Supported AXON operators:
    - FULLY_CONNECTED (CONV2D 1x1 on 1x1 spatial)
    - CONV2D (standard 2D convolution)
    - DEPTHWISE_CONV2D (depthwise separable convolution)
    - POINTWISE_CONV2D (1x1 conv on spatial input)
    - ADD2 (element-wise add with broadcast)
    - MULTIPLY (element-wise multiply with broadcast)
    - AVERAGE_POOLING (average pool 2D)
    - MAX_POOLING (max pool 2D)
    - MEAN (global average pooling / reduce)

Architecture:
    TOSA flatbuffer → parse (tosa_reader.py)
        → fuse ops + RESCALE into AXON layers
        → pack into AXON intermediate binary (nrf_axon_nn_model_desc_hdr_s format)
        → call nrf_axon_compile_model() via ctypes
        → produces C header with command buffers
"""

from __future__ import annotations

import ctypes
import logging
import os
import struct
import tempfile
from dataclasses import dataclass, field
from pathlib import Path

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
from .tosa_reader import TosaGraph, TosaOperator, TosaTensor, parse_tosa_flatbuffer
from .axon_converters import (
    _convert_concat,
    _convert_conv2d,
    _convert_depthwise_conv2d,
    _convert_elementwise,
    _convert_pad,
    _convert_persistent_var,
    _convert_pool2d,
    _convert_reduce_sum,
    _convert_slice,
    _create_rescale_layer,
    _extract_rescale_params,
    _handle_softmax_pattern,
    _handle_table_op,
    _optimized_scaling_shift,
    _resolve_input_id,
    _validate_axon_layer,
)

logger = logging.getLogger(__name__)


def tosa_to_axon_layers(
    graph: TosaGraph,
    activation_info: list["ActivationQuantInfo"] | None = None,
) -> list[AxonLayer]:
    """Convert a TOSA graph to AXON layer descriptors.

    Fuses CONV2D + RESCALE pairs into single AXON layers with
    quantization parameters. Skips RESHAPE/CONST/CONST_SHAPE ops.

    The TOSA graph for a quantized Linear layer looks like:
        CONV2D(input, weights[O,1,1,I], bias[O], input_zp, weight_zp) → INT16
        RESCALE(result, multiplier, shift, input_zp, output_zp) → INT8

    This fuses into one AXON FULLY_CONNECTED layer with:
        - weights from CONV2D input[1]
        - bias from CONV2D input[2]
        - zero points from CONV2D input[3,4]
        - multiplier/shift from RESCALE input[1,2]
        - output zero point from RESCALE input[4]

    Args:
        activation_info: Quantization info for sigmoid/tanh/softmax ops,
            extracted from the PyTorch FX graph before TOSA lowering. Used
            to convert TOSA TABLE ops into AXON op extensions. The list
            order matches the order of TABLE ops in the TOSA graph.
    """
    ops = graph.get_non_const_operators()
    layers = []
    axon_layer_idx = 0
    # Index into activation_info — incremented as we consume TABLE ops
    activation_info = activation_info or []
    activation_idx = 0
    # Map TOSA output tensor name → AXON layer index
    tensor_to_layer = {}
    # Track spatial shapes before flatten (for CHW→HWC weight permutation)
    # Maps tensor name → (H, W, C) shape before the flatten RESHAPE
    tensor_pre_flatten_shape = {}
    # Track output zero points through the graph (for ops like MAX_POOL that
    # don't have explicit zp tensors but need to inherit from their input)
    tensor_to_zp = {}  # tensor_name → int zero_point
    # Track skipped rescale info for ADD (scale ratios from standalone RESCALEs
    # that were too large to implement as identity layers)
    tensor_rescale_info = {}  # tensor_name → (scale_ratio, input_zp)

    if logger.isEnabledFor(logging.DEBUG):
        for idx, op in enumerate(ops):
            ins = [f"{t.shape}" for t in op.input_tensors if not t.has_data]
            outs = [f"{t.shape}" for t in op.output_tensors]
            logger.debug(f"  TOSA[{idx}] {op.op_name}: in={ins} -> out={outs}")

    i = 0
    while i < len(ops):
        op = ops[i]

        if op.op_name in ("RESHAPE", "TRANSPOSE"):
            # Transparent — just pass through tensor mapping
            if op.input_tensors and op.output_tensors:
                in_name = op.input_tensors[0].name
                out_name = op.output_tensors[0].name
                if in_name in tensor_to_layer:
                    tensor_to_layer[out_name] = tensor_to_layer[in_name]

                # Track flatten: RESHAPE from [N,H,W,C] to [N,1,1,H*W*C] or [N,H*W*C]
                # The pre-flatten spatial shape is needed to permute FC weights
                # from HWC (TOSA) to CHW (AXON) order.
                if op.op_name == "RESHAPE":
                    in_shape = op.input_tensors[0].shape
                    out_shape = op.output_tensors[0].shape
                    in_numel = 1
                    for s in in_shape:
                        in_numel *= s
                    out_numel = 1
                    for s in out_shape:
                        out_numel *= s
                    # Detect flatten: same element count, input has spatial dims, output is flat
                    if (in_numel == out_numel and len(in_shape) >= 3
                            and in_shape[-1] > 1 and in_shape[-2] > 1):
                        # Input is [N, H, W, C] — record spatial shape
                        if len(in_shape) == 4:
                            tensor_pre_flatten_shape[out_name] = (in_shape[1], in_shape[2], in_shape[3])
                        elif len(in_shape) == 3:
                            tensor_pre_flatten_shape[out_name] = (in_shape[0], in_shape[1], in_shape[2])
                        logger.debug(f"  Flatten detected: {in_shape} → {out_shape}, "
                                    f"pre-flatten HWC={tensor_pre_flatten_shape.get(out_name)}")

                # Propagate pre-flatten shape through pass-through ops
                if in_name in tensor_pre_flatten_shape:
                    tensor_pre_flatten_shape[out_name] = tensor_pre_flatten_shape[in_name]

                # Propagate zero points through pass-through ops
                if in_name in tensor_to_zp:
                    tensor_to_zp[out_name] = tensor_to_zp[in_name]

            i += 1
            continue

        if op.op_name == "CLAMP":
            # Fuse CLAMP into the preceding AXON layer's activation field.
            # CLAMP(min=0, max=127) = ReLU, CLAMP(min=-128, max=X) where X < 127 = custom clamp.
            if op.input_tensors and op.output_tensors:
                in_name = op.input_tensors[0].name
                out_name = op.output_tensors[0].name
                attrs = op.attributes
                min_int = attrs.get("min_int", -128)
                max_int = attrs.get("max_int", 127)

                if in_name in tensor_to_layer:
                    prev_layer_idx = tensor_to_layer[in_name]
                    if min_int >= 0 and max_int > 0:
                        # ReLU or ReLU6: clamp to [0, X].
                        # Nordic: "ReLU6 is mapped to ReLU because quantization
                        # causes saturation at 6" — the clip at 6 is already
                        # handled by INT8 quantization range clipping.
                        layers[prev_layer_idx].activation = AxonActivation.RELU
                        if max_int < 127:
                            logger.debug(f"  Fused ReLU6/CLAMP(0,{max_int}) as ReLU into layer {prev_layer_idx}")
                        else:
                            logger.debug(f"  Fused ReLU into layer {prev_layer_idx}")
                    elif min_int == -128 and max_int == 127:
                        # No-op clamp (full INT8 range)
                        pass
                    else:
                        # Negative min with non-full range — could be LeakyReLU territory
                        logger.warning(f"  CLAMP({min_int},{max_int}) not fused — "
                                     f"AXON only supports ReLU/LeakyReLU activation fusion")
                    tensor_to_layer[out_name] = tensor_to_layer[in_name]
            i += 1
            continue

        if op.op_name == "RESCALE":
            # Standalone RESCALE: chain into the preceding AXON layer by
            # combining the rescale parameters. This avoids creating extra
            # identity conv layers that add quantization error.
            # Nordic's TFLite pipeline doesn't produce standalone RESCALEs.
            if op.input_tensors and op.output_tensors:
                in_name = op.input_tensors[0].name
                out_name = op.output_tensors[0].name
                if in_name in tensor_to_layer:
                    prev_idx = tensor_to_layer[in_name]
                    prev_layer = layers[prev_idx]

                    # Get the standalone RESCALE's parameters
                    new_zp, new_mult_data, new_shift_data, new_cnt = _extract_rescale_params(op)

                    can_chain = False
                    if prev_layer.multiplier_data and new_mult_data:
                        # Check if chaining is safe: combined scale must be < 1.0
                        # to keep multiplier within AXON's INT32 range.
                        # Standalone RESCALEs with large scale (>1) are for
                        # requantizing between different domains (e.g., for ADD)
                        # and produce multipliers near INT32 max when chained.
                        prev_mult = np.frombuffer(prev_layer.multiplier_data, dtype=np.int32)
                        prev_shift = np.frombuffer(prev_layer.shift_data, dtype=np.int8)
                        new_mult = np.frombuffer(new_mult_data, dtype=np.int32)
                        new_shift = np.frombuffer(new_shift_data, dtype=np.int8)

                        # Check combined scale magnitude
                        ps0 = float(prev_mult[0]) / (2.0 ** int(prev_shift[0]))
                        ns0 = float(new_mult[0]) / (2.0 ** int(new_shift[0]))
                        combined_scale = ps0 * ns0

                        if combined_scale < 1.0:
                            # Safe to chain — combined scale fits in AXON range
                            combined_mult = np.zeros_like(prev_mult)
                            combined_shift = np.zeros_like(prev_shift)
                            for ch in range(len(prev_mult)):
                                ps = float(prev_mult[ch]) / (2.0 ** int(prev_shift[ch]))
                                ns_idx = min(ch, len(new_mult) - 1)
                                ns = float(new_mult[ns_idx]) / (2.0 ** int(new_shift[ns_idx]))
                                cs = ps * ns
                                m, s = _optimized_scaling_shift(cs, new_zp)
                                combined_mult[ch] = m
                                combined_shift[ch] = s

                            prev_layer.multiplier_data = combined_mult.tobytes()
                            prev_layer.shift_data = combined_shift.tobytes()
                            prev_layer.output_zero_point = new_zp
                            can_chain = True
                            logger.debug(f"  Chained RESCALE into layer {prev_idx}: "
                                        f"new out_zp={new_zp}")
                        else:
                            logger.debug(f"  Cannot chain RESCALE: combined_scale={combined_scale:.2f} > 1.0, "
                                        f"creating identity conv instead")

                    if not can_chain:
                        # Combined scale > 1.0 means the RESCALE is amplifying.
                        # Identity conv layers with large multipliers destroy
                        # precision (int8 clips all values). Skip and keep the
                        # original tensor — the ADD will use the rescale info
                        # to compute proper per-input multipliers and bias.
                        #
                        # Save the skipped rescale's scale ratio for ADD:
                        # The standalone RESCALE converts from prev domain to ADD domain.
                        # scale_ratio = prev_scale / add_scale (the ns0 value)
                        rescale_in_zp = 0
                        if len(op.input_tensors) > 3 and op.input_tensors[3].data is not None:
                            rescale_in_zp = int(op.input_tensors[3].data.flat[0])
                        tensor_rescale_info[out_name] = (ns0, rescale_in_zp)
                        logger.debug(f"  Skipping amplifying RESCALE (scale={ns0:.0f}, "
                                    f"in_zp={rescale_in_zp}), keeping zp={prev_layer.output_zero_point}")
                        tensor_to_layer[out_name] = prev_idx
                        tensor_to_zp[out_name] = prev_layer.output_zero_point
                        i += 1
                        continue

                    tensor_to_layer[out_name] = prev_idx
                    # Track the output zero point for downstream ops (e.g., MAX_POOL)
                    tensor_to_zp[out_name] = new_zp
                else:
                    # Input is a graph-level input — pass through
                    pass
            i += 1
            continue

        # Check if next op is RESCALE (many ops fuse with it)
        rescale_op = None
        if i + 1 < len(ops) and ops[i + 1].op_name == "RESCALE":
            rescale_op = ops[i + 1]

        layer = None

        if op.op_name == "CONV2D":
            # Check if this FC's input was spatially flattened (needs weight permutation)
            input_name = op.input_tensors[0].name
            pfs = tensor_pre_flatten_shape.get(input_name)
            layer = _convert_conv2d(op, rescale_op, tensor_to_layer, axon_layer_idx,
                                    pre_flatten_shape=pfs)

        elif op.op_name == "DEPTHWISE_CONV2D":
            # TOSA uses DEPTHWISE_CONV2D for Conv2d with 1 input channel.
            # But AXON should use regular CONV2D for this (Nordic's convention).
            # True depthwise: c_in>1, depth_mult=1 (groups=channels).
            # Fake depthwise: c_in=1, depth_mult>1 (really a standard conv).
            weight_shape = op.input_tensors[1].shape  # [KH, KW, C_in, M]
            c_in = weight_shape[2] if len(weight_shape) > 2 else 1
            depth_mult = weight_shape[3] if len(weight_shape) > 3 else 1
            if c_in == 1 and depth_mult > 1:
                # Standard conv disguised as depthwise — use CONV2D
                layer = _convert_depthwise_conv2d(op, rescale_op, tensor_to_layer, axon_layer_idx,
                                                  as_conv2d=True)
            else:
                layer = _convert_depthwise_conv2d(op, rescale_op, tensor_to_layer, axon_layer_idx)

        elif op.op_name == "ADD":
            layer = _convert_elementwise(op, rescale_op, tensor_to_layer, axon_layer_idx, AxonOp.ADD2, graph, tensor_to_zp, tensor_rescale_info)

        elif op.op_name == "MUL":
            layer = _convert_elementwise(op, rescale_op, tensor_to_layer, axon_layer_idx, AxonOp.MULTIPLY, graph, tensor_to_zp, tensor_rescale_info)

        elif op.op_name == "AVG_POOL2D":
            layer = _convert_pool2d(op, rescale_op, tensor_to_layer, axon_layer_idx, AxonOp.AVERAGE_POOLING)

        elif op.op_name == "MAX_POOL2D":
            # MAX_POOL has no zp tensors in TOSA — propagate from preceding layer
            pool_input_zp = tensor_to_zp.get(op.input_tensors[0].name, 0)
            layer = _convert_pool2d(op, rescale_op, tensor_to_layer, axon_layer_idx, AxonOp.MAX_POOLING,
                                    input_zp_from_graph=pool_input_zp)

        elif op.op_name == "REDUCE_SUM":
            layer = _convert_reduce_sum(op, rescale_op, tensor_to_layer, axon_layer_idx)

        elif op.op_name == "CONCAT":
            # CONCAT doesn't fuse with RESCALE — no rescale_op consumed
            layer = _convert_concat(op, tensor_to_layer, axon_layer_idx)
            if layer is not None:
                layers.append(layer)
                out_name = op.output_tensors[0].name
                tensor_to_layer[out_name] = axon_layer_idx
                axon_layer_idx += 1
            i += 1
            continue

        # Softmax decomposition: TOSA decomposes quantized softmax into
        # REDUCE_MAX → SUB → TABLE(exp) → REDUCE_SUM → TABLE(reciprocal) → MUL.
        # We detect this at the REDUCE_MAX op and replace the entire chain
        # with a single AXON SOFTMAX op extension (operation 100).
        if (op.op_name == "REDUCE_MAX"
                and activation_idx < len(activation_info)
                and activation_info[activation_idx].op_type == "softmax"):
            handled = _handle_softmax_pattern(
                op, ops, i, layers, tensor_to_layer, tensor_to_zp,
                activation_info, activation_idx, axon_layer_idx,
            )
            if handled is not None:
                axon_layer_idx, activation_idx, advance = handled
                i += advance
                continue

        # TABLE op: sigmoid/tanh activation lookup table.
        # ExecuTorch's quantized sigmoid/tanh becomes a TOSA TABLE.
        # We convert this into an AXON op extension (operation=101 sigmoid,
        # 102 tanh) and modify the preceding layer to output INT16 q3.12.
        #
        # NOTE: TABLE ops also appear in softmax decompositions (one for exp,
        # one for reciprocal). The softmax decomposition has a characteristic
        # surrounding pattern (REDUCE_MAX/SUB before, REDUCE_SUM/MUL after);
        # for now we treat any TABLE that has matching activation_info as a
        # standalone activation. Softmax handling is added separately.
        if op.op_name == "TABLE":
            handled = _handle_table_op(
                op, ops, i, layers, tensor_to_layer, tensor_to_zp,
                activation_info, activation_idx, axon_layer_idx,
            )
            if handled is not None:
                axon_layer_idx, activation_idx, advance = handled
                i += advance
                continue

        # Ops that don't fuse with RESCALE — handle and continue
        _no_rescale_layer = None
        if op.op_name == "SLICE":
            _no_rescale_layer = _convert_slice(op, tensor_to_layer, axon_layer_idx)
        elif op.op_name == "PAD":
            _no_rescale_layer = _convert_pad(op, tensor_to_layer, axon_layer_idx)
        elif op.op_name in ("VARIABLE", "VARIABLE_READ", "VARIABLE_WRITE"):
            _no_rescale_layer = _convert_persistent_var(op, tensor_to_layer, axon_layer_idx)

        if _no_rescale_layer is not None:
            layers.append(_no_rescale_layer)
            if op.output_tensors:
                tensor_to_layer[op.output_tensors[0].name] = axon_layer_idx
            axon_layer_idx += 1
            i += 1
            continue

        if layer is not None:
            # Detect ReLU from output_zp == -128 (fused into quantization)
            if layer.output_zero_point == -128 and layer.activation == AxonActivation.DISABLED:
                layer.activation = AxonActivation.RELU
                logger.debug(f"  Detected ReLU from output_zp=-128 on layer {axon_layer_idx}")

            # Validate against AXON hardware constraints
            constraint_warnings = _validate_axon_layer(layer, axon_layer_idx)
            for w in constraint_warnings:
                logger.warning(f"AXON CONSTRAINT: {w} — may fail on hardware or fall back to CPU")

            if logger.isEnabledFor(logging.DEBUG):
                shift_vals = np.frombuffer(layer.shift_data, dtype=np.int8).tolist() if layer.shift_data else []
                mult_vals = np.frombuffer(layer.multiplier_data, dtype=np.int32).tolist() if layer.multiplier_data else []
                act_names = {0: "", 1: " ReLU", 2: " Softmax", 3: " LeakyReLU"}
                logger.debug(f"  Layer {axon_layer_idx} ({op.op_name}): shift={shift_vals}, mult={mult_vals}, "
                            f"cnt={layer.scale_shift_cnt}, in_zp={layer.input_zero_point}, "
                            f"out_zp={layer.output_zero_point}{act_names.get(layer.activation, '')}")
            layers.append(layer)
            if rescale_op:
                out_name = rescale_op.output_tensors[0].name
                i += 2
            else:
                out_name = op.output_tensors[0].name
                i += 1
            tensor_to_layer[out_name] = axon_layer_idx
            # Track output zero point for downstream ops (MAX_POOL, etc.)
            tensor_to_zp[out_name] = layer.output_zero_point
            axon_layer_idx += 1
            continue

        logger.warning(f"Skipping unsupported TOSA op: {op.op_name}")
        i += 1

    # Nordic keeps output_zp on ALL layers (verified from binary).
    # No clearing needed.

    return layers



def pack_intermediate_binary(
    layers: list[AxonLayer],
    model_name: str = "model",
    interlayer_buffer_size: int = 125000,
    psum_buffer_size: int = 4096,
) -> bytes:
    """Pack AXON layers into the intermediate binary format.

    The binary format (nrf_axon_nn_model_desc_hdr_s) is:
        Header: 6 × bin_item_s (offset, length pairs)
        Title string: "AXON_INTERMEDIATE_REPRESENTATION_FILE"
        Version string
        Meta info: nrf_axon_nn_model_meta_info_s
        Layer descriptors: nrf_axon_nn_model_layer_desc_s[]
        Constants: weights, biases, multipliers, shifts (concatenated)
        Compilation options: nrf_axon_nn_model_compilation_options_s

    Each pointer field in layer_desc_s uses the offset union member,
    pointing into the constants section.
    """
    # Build constants section, tracking offsets
    consts = bytearray()
    layer_const_offsets = []  # Per layer: (filter_off, bias_off, mult_off, shift_off)

    for layer in layers:
        offsets = {}
        if layer.filter_data:
            offsets["filter"] = len(consts)
            consts.extend(layer.filter_data)
            # Align to 4 bytes
            while len(consts) % 4 != 0:
                consts.append(0)

        if layer.bias_data:
            offsets["bias"] = len(consts)
            consts.extend(layer.bias_data)
            while len(consts) % 4 != 0:
                consts.append(0)

        if layer.multiplier_data:
            offsets["multiplier"] = len(consts)
            consts.extend(layer.multiplier_data)
            while len(consts) % 4 != 0:
                consts.append(0)

        if layer.shift_data:
            offsets["shift"] = len(consts)
            consts.extend(layer.shift_data)
            while len(consts) % 4 != 0:
                consts.append(0)

        layer_const_offsets.append(offsets)

    # Build layer descriptors section
    # sizeof(nrf_axon_nn_model_layer_desc_s) — we need to match the C struct exactly
    # This is the tricky part — struct layout depends on alignment and platform.
    # For now, build a simplified version and validate against Nordic's executor.
    layers_bin = bytearray()
    for i, layer in enumerate(layers):
        offsets = layer_const_offsets[i]
        layer_bin = _pack_layer_desc(layer, offsets)
        layers_bin.extend(layer_bin)

    # Build meta info
    model_name_bytes = model_name.encode("utf-8") + b"\x00"
    meta_bin = _pack_meta_info(len(layers), model_name_bytes)

    # Build compilation options
    options_bin = _pack_compilation_options(interlayer_buffer_size, psum_buffer_size)

    # Build version string
    version_str = b"1.0.0\x00"

    # Now build the header and assemble the file
    # Header is 6 × bin_item_s = 6 × 8 = 48 bytes
    header_size = 48
    title_str = b"AXON_INTERMEDIATE_REPRESENTATION_FILE\x00"

    # Calculate offsets for each section
    title_offset = header_size
    version_offset = title_offset + len(title_str)
    meta_offset = version_offset + len(version_str)
    # Align meta to 4 bytes
    while (meta_offset) % 4 != 0:
        meta_offset += 1
    model_name_offset = meta_offset + len(meta_bin)
    layers_offset = model_name_offset + len(model_name_bytes)
    # Align layers to 4 bytes
    while (layers_offset) % 4 != 0:
        layers_offset += 1
    consts_offset = layers_offset + len(layers_bin)
    # Align consts to 4 bytes
    while (consts_offset) % 4 != 0:
        consts_offset += 1
    options_offset = consts_offset + len(consts)
    while (options_offset) % 4 != 0:
        options_offset += 1

    # Pack header (6 × nrf_axon_nn_model_bin_item_s)
    header = struct.pack("<II", title_offset, len(title_str))       # title
    header += struct.pack("<II", version_offset, len(version_str))  # version
    header += struct.pack("<II", meta_offset, len(meta_bin))        # meta
    header += struct.pack("<II", layers_offset, len(layers_bin))    # layers
    header += struct.pack("<II", consts_offset, len(consts))        # consts
    header += struct.pack("<II", options_offset, len(options_bin))  # compilation_option

    # Assemble
    binary = bytearray(header)
    # Pad to title offset
    while len(binary) < title_offset:
        binary.append(0)
    binary.extend(title_str)
    # Pad to version offset
    while len(binary) < version_offset:
        binary.append(0)
    binary.extend(version_str)
    # Pad to meta offset
    while len(binary) < meta_offset:
        binary.append(0)
    binary.extend(meta_bin)
    binary.extend(model_name_bytes)
    # Pad to layers offset
    while len(binary) < layers_offset:
        binary.append(0)
    binary.extend(layers_bin)
    # Pad to consts offset
    while len(binary) < consts_offset:
        binary.append(0)
    binary.extend(consts)
    # Pad to options offset
    while len(binary) < options_offset:
        binary.append(0)
    binary.extend(options_bin)

    return bytes(binary)


def _pack_layer_desc(layer: AxonLayer, const_offsets: dict[str, int]) -> bytes:
    """Pack a single nrf_axon_nn_model_layer_desc_s.

    WARNING: This struct packing must exactly match the C struct layout.
    The struct is complex with unions and alignment. This is a best-effort
    implementation — will need validation against Nordic's executor output.
    """
    buf = bytearray()

    # input_id_cnt (uint8)
    buf.append(len(layer.input_ids))

    # padding for alignment (3 bytes to align input_ids to 2-byte boundary)
    buf.append(0)

    # input_ids[4] (int16 × 4)
    for j in range(4):
        if j < len(layer.input_ids):
            buf.extend(struct.pack("<h", layer.input_ids[j]))
        else:
            buf.extend(struct.pack("<h", 0))

    # nn_operation (int32 — enum)
    buf.extend(struct.pack("<i", layer.operation))

    # input_dimensions[4] (each: uint16 height, uint16 width, uint16 channel_cnt, [pad16], int32 byte_width)
    # C struct has 2 bytes padding before int32 byte_width for 4-byte alignment
    for j in range(4):
        if j < len(layer.input_dimensions):
            d = layer.input_dimensions[j]
        else:
            d = AxonDimensions()
        buf.extend(struct.pack("<HHH", d.height, d.width, d.channel_cnt))
        buf.extend(struct.pack("<xx"))  # 2 bytes padding for int32 alignment
        buf.extend(struct.pack("<i", d.byte_width))

    # filter_dimensions
    d = layer.filter_dimensions
    buf.extend(struct.pack("<HHH", d.height, d.width, d.channel_cnt))
    buf.extend(struct.pack("<xx"))
    buf.extend(struct.pack("<i", d.byte_width))

    # output_dimensions
    d = layer.output_dimensions
    buf.extend(struct.pack("<HHH", d.height, d.width, d.channel_cnt))
    buf.extend(struct.pack("<xx"))
    buf.extend(struct.pack("<i", d.byte_width))

    # concatenate_axis (uint8), stride_x (uint8), stride_y (uint8),
    # dilation_x (uint8), dilation_y (uint8)
    buf.append(layer.concatenate_axis)
    buf.append(layer.stride_x)
    buf.append(layer.stride_y)
    buf.append(layer.dilation_x)
    buf.append(layer.dilation_y)

    # input_zero_point (int8), output_zero_point (int8)
    buf.extend(struct.pack("<b", layer.input_zero_point))
    buf.extend(struct.pack("<b", layer.output_zero_point))

    # Pad to 8-byte alignment for the union
    while len(buf) % 8 != 0:
        buf.append(0)

    # bias_prime (uint64 offset)
    bias_off = const_offsets.get("bias", 0)
    buf.extend(struct.pack("<Q", bias_off))

    # output_multipliers (uint64 offset)
    mult_off = const_offsets.get("multiplier", 0)
    buf.extend(struct.pack("<Q", mult_off))

    # scale_shifts (uint64 offset)
    shift_off = const_offsets.get("shift", 0)
    buf.extend(struct.pack("<Q", shift_off))

    # scale_shift_cnt (uint16)
    buf.extend(struct.pack("<H", layer.scale_shift_cnt))

    # Pad for alignment
    while len(buf) % 4 != 0:
        buf.append(0)

    # activation_function (int32 enum)
    buf.extend(struct.pack("<i", layer.activation))

    # padding: pad_left, pad_right, pad_top, pad_bottom (uint8 × 4)
    buf.append(layer.pad_left)
    buf.append(layer.pad_right)
    buf.append(layer.pad_top)
    buf.append(layer.pad_bottom)

    # Pad for alignment before filter union
    while len(buf) % 8 != 0:
        buf.append(0)

    # filter (uint64 offset)
    filter_off = const_offsets.get("filter", 0)
    buf.extend(struct.pack("<Q", filter_off))

    # cpu_op_additional_attributes_count (uint32)
    buf.extend(struct.pack("<I", 0))

    # Pad for alignment
    while len(buf) % 8 != 0:
        buf.append(0)

    # cpu_op_additional_attributes (uint64 offset)
    buf.extend(struct.pack("<Q", 0))

    return bytes(buf)


def _pack_meta_info(layer_count: int, model_name_bytes: bytes) -> bytes:
    """Pack nrf_axon_nn_model_meta_info_s."""
    buf = bytearray()
    # model_name: bin_item_s (offset relative to meta section start, length)
    # The name follows immediately after the meta struct
    meta_struct_size = 8 + 8 + 4 + 6 + 6  # approximate
    buf.extend(struct.pack("<II", 0, len(model_name_bytes)))  # placeholder
    # model_labels: bin_item_s
    buf.extend(struct.pack("<II", 0, 0))  # no labels
    # model_layer_cnt (uint32)
    buf.extend(struct.pack("<I", layer_count))
    # input_quant: nrf_axon_nn_model_quant_paramters_s (mult:u32, round:u8, zp:i8)
    buf.extend(struct.pack("<IBb", 1, 0, 0))
    # output_dequant
    buf.extend(struct.pack("<IBb", 1, 0, 0))
    return bytes(buf)


def _pack_compilation_options(
    interlayer_buffer_size: int,
    psum_buffer_size: int,
    psum_buffer_placement: int = 1,
) -> bytes:
    """Pack nrf_axon_nn_model_compilation_options_s.

    psum_buffer_placement: 0=shared, 1=DEDICATED_MEM (required for Conv/Pool
    to avoid PSUM/FILTER overlap).
    """
    buf = struct.pack("<IIIiIi",
        interlayer_buffer_size,   # interlayer_buffer_size
        psum_buffer_size,         # psum_buffer_size
        0,                        # header_file_test_vector_cnt
        0,                        # convolution_2d_setting (unused)
        0,                        # log_level (unused)
        psum_buffer_placement,    # psum_buffer_placement
    )
    return buf


def compile_from_tosa(
    tosa_flatbuffer: bytes,
    sdk_edge_ai_path: str,
    model_name: str = "model",
    output_dir: str | None = None,
) -> dict:
    """Full compilation pipeline: TOSA → AXON command buffers.

    Args:
        tosa_flatbuffer: Serialized TOSA graph bytes.
        sdk_edge_ai_path: Path to sdk-edge-ai repo.
        model_name: Name for the compiled model.
        output_dir: Directory for compiler output. Uses tempdir if None.

    Returns:
        Dict with compilation results and paths to output files.
    """
    # 1. Parse TOSA
    graph = parse_tosa_flatbuffer(tosa_flatbuffer)
    logger.info(f"Parsed TOSA graph: {len(graph.tensors)} tensors, {len(graph.operators)} operators")

    # 2. Convert to AXON layers
    layers = tosa_to_axon_layers(graph)
    logger.info(f"Converted to {len(layers)} AXON layers")

    op_names = {
        0: "FC", 1: "CONV2D", 2: "DW_CONV2D", 3: "PW_CONV2D",
        4: "AVG_POOL", 5: "MAX_POOL", 6: "ADD", 7: "CH_PAD",
        8: "PERSIST_VAR", 9: "CONCAT", 10: "SLICE", 11: "MUL", 12: "MEAN",
    }
    for i, layer in enumerate(layers):
        in_desc = f"in={layer.input_dimensions[0].channel_cnt}" if layer.input_dimensions else "in=?"
        logger.info(f"  Layer {i}: {op_names.get(layer.operation, '?')} "
                    f"{in_desc} "
                    f"out={layer.output_dimensions.channel_cnt} "
                    f"weights={len(layer.filter_data)}B")

    # 3. Pack intermediate binary
    binary = pack_intermediate_binary(layers, model_name)
    logger.info(f"Packed intermediate binary: {len(binary)} bytes")

    # 4. Write binary file and call compiler
    if output_dir is None:
        output_dir = tempfile.mkdtemp(prefix="axon_compile_")

    bin_path = os.path.join(output_dir, f"{model_name}.bin")
    with open(bin_path, "wb") as f:
        f.write(binary)

    logger.info(f"Wrote intermediate binary to {bin_path}")

    # 5. Call Nordic compiler lib
    import platform
    system = platform.system()
    if system == "Linux":
        lib_name = "libnrf-axon-nn-compiler-lib-amd64.so"
    elif system == "Darwin":
        lib_name = "libnrf-axon-nn-compiler-lib-arm64.dylib"
    else:
        lib_name = "nrf-axon-nn-compiler-lib-amd64.dll"

    compiler_lib_path = os.path.join(
        sdk_edge_ai_path, "tools", "axon", "compiler", "bin", system, lib_name
    )

    if not os.path.exists(compiler_lib_path):
        logger.warning(f"AXON compiler lib not found at {compiler_lib_path}")
        logger.warning("Skipping compilation — intermediate binary saved for manual compilation")
        return {
            "binary_path": bin_path,
            "layers": len(layers),
            "compiled": False,
        }

    output_prefix = os.path.join(output_dir, f"nrf_axon_model_{model_name}")
    result = _call_compiler_lib(compiler_lib_path, bin_path, output_prefix)

    return {
        "binary_path": bin_path,
        "output_prefix": output_prefix,
        "layers": len(layers),
        "compiled": result == 0,
        "return_code": result,
    }


def _call_compiler_lib(compiler_lib_path: str, bin_path: str, output_prefix: str) -> int:
    """Call Nordic's nrf_axon_compile_model() via ctypes.

    Args:
        compiler_lib_path: Path to the shared library.
        bin_path: Path to the intermediate binary file.
        output_prefix: Output file prefix for compiled header.

    Returns:
        Compiler return code (0 = success).

    Raises:
        OSError: If the compiler library cannot be loaded.
    """
    logger.info("Loading compiler lib: %s", compiler_lib_path)
    try:
        lib = ctypes.CDLL(compiler_lib_path)
    except OSError as e:
        logger.error(
            "Failed to load AXON compiler library: %s\n"
            "  Path: %s\n"
            "  Is SDK_EDGE_AI_PATH set correctly?",
            e, compiler_lib_path,
        )
        raise

    # Build command-line arguments (matches Nordic's compiler CLI)
    args = [
        f"-c{compiler_lib_path}",
        f"-b{bin_path}",
        f"-f{output_prefix}",
    ]

    # Convert to ctypes
    argc = len(args)
    argv_type = ctypes.c_char_p * argc
    argv = argv_type(*[a.encode("utf-8") for a in args])

    # Return struct: 5 × uint32 (model_const_size, interlayer_buf,
    # psum_buf, cmd_buf_size, profiling_ticks)
    return_type = ctypes.c_uint32 * 5
    return_buf = return_type()

    lib.nrf_axon_compile_model.argtypes = [
        ctypes.c_int,
        ctypes.POINTER(argv_type),
        ctypes.POINTER(return_type),
    ]
    lib.nrf_axon_compile_model.restype = ctypes.c_int

    logger.info("Calling nrf_axon_compile_model with args: %s", args)
    result = lib.nrf_axon_compile_model(
        argc, ctypes.pointer(argv), ctypes.pointer(return_buf)
    )

    if result == 0:
        logger.info("Compilation successful")
        logger.info("  Model const size: %d bytes", return_buf[0])
        logger.info("  Interlayer buffer: %d bytes", return_buf[1])
        logger.info("  PSUM buffer: %d bytes", return_buf[2])
        logger.info("  Command buffer: %d bytes", return_buf[3])
    else:
        logger.error("Compilation failed with code %d", result)

    return result
