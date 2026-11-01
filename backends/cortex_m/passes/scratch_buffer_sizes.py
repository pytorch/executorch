# Copyright 2026 Arm Limited and/or its affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import math
from collections.abc import Callable
from functools import partial
from typing import Any, cast

import executorch.backends.cortex_m.ops.operators  # noqa

import torch
import torch.fx
from executorch.backends.cortex_m.library import cmsis_nn

from executorch.exir.dialects._ops import ops as exir_ops

BufferSizeFunction = Callable[[cmsis_nn.Backend, torch.fx.Node], list[int]]


def _tensor_from_node(node: torch.fx.Node) -> torch.Tensor:
    if "val" in node.meta:
        return node.meta["val"]
    elif node.op == "call_function":
        args = (
            _tensor_from_node(arg) if isinstance(arg, torch.fx.Node) else arg
            for arg in node.args
        )
        return node.target(*args, **node.kwargs)  # type: ignore[operator]
    else:
        raise RuntimeError("Encountered non-call_function without 'val' meta.")


def _shape_from_node(node: torch.fx.Node) -> torch.Size:
    return _tensor_from_node(node).shape


def _get_common_conv_buffer_size_inputs(
    conv_node: torch.fx.Node,
    *,
    nhwc_logical: bool = False,
    stride_arg_idx: int = 3,
    padding_arg_idx: int = 4,
    dilation_arg_idx: int = 5,
) -> tuple[
    list[int],
    list[int],
    list[int],
    list[int],
    list[int],
    list[int],
]:
    x = cast(torch.fx.Node, conv_node.args[0])
    weight = cast(torch.fx.Node, conv_node.args[1])
    stride = cast(list[int], conv_node.args[stride_arg_idx])
    padding = cast(list[int], conv_node.args[padding_arg_idx])
    dilation = cast(list[int], conv_node.args[dilation_arg_idx])

    weight_shape = _shape_from_node(weight)

    if nhwc_logical:
        n, height, width, c_in = _shape_from_node(x)
        out_n, out_h, out_w, out_c = _shape_from_node(conv_node)
    else:
        n, c_in, height, width = _shape_from_node(x)
        out_n, out_c, out_h, out_w = _shape_from_node(conv_node)

    input_nhwc = [n, height, width, c_in]
    output_nhwc = [out_n, out_h, out_w, out_c]
    stride_hw = [int(stride[0]), int(stride[1])]
    padding_hw = [int(padding[0]), int(padding[1])]
    dilation_hw = [int(dilation[0]), int(dilation[1])]

    return (
        input_nhwc,
        list(weight_shape),
        output_nhwc,
        stride_hw,
        padding_hw,
        dilation_hw,
    )


def _use_generic_s4_kernel(
    input_nhwc: list[int],
    filter_nhwc: list[int],
    output_nhwc: list[int],
    padding_hw: list[int],
    stride_hw: list[int],
    dilation_hw: list[int],
) -> bool:
    """Mirrors use_generic_s4_kernel in ops/op_quantized_conv2d.cpp: the 1xN
    shapes that the s4 op runs with arm_convolve_s4."""
    _, input_h, input_w, c_in = input_nhwc
    _, kernel_h, kernel_w, _ = filter_nhwc
    output_w = output_nhwc[2]
    stride, pad = stride_hw[1], padding_hw[1]
    is_1x1 = (
        padding_hw == [0, 0] and kernel_h == kernel_w == 1 and dilation_hw == [1, 1]
    )
    is_1_x_n = (
        input_h == 1
        and kernel_h == 1
        and dilation_hw[1] == 1
        and stride * c_in % 4 == 0
    )
    if is_1x1 or not is_1_x_n:
        return False
    if c_in % 4:
        return True
    total_pad = (output_w - 1) * stride + kernel_w - input_w
    # C remainder, as in the C++ and CMSIS-NN: a strided convolution without
    # padding can have total_pad == -1.
    asym_pad = int(math.fmod(total_pad, 2))
    if pad * 2 + asym_pad != total_pad:
        return True
    right_pad_outputs = (
        max(1, (pad + asym_pad + stride - 1) // stride) if pad + asym_pad else 0
    )
    left_pad_outputs = max(1, (pad + stride - 1) // stride) if pad else 0
    return left_pad_outputs + right_pad_outputs > output_w


def cmsis_nn_conv_buffer_size(
    backend: cmsis_nn.Backend,
    conv_node: torch.fx.Node,
    nhwc_logical: bool = False,
    packed_int4: bool = False,
) -> list[int]:
    (
        input_nhwc,
        weight_shape,
        output_nhwc,
        stride_hw,
        padding_hw,
        dilation_hw,
    ) = _get_common_conv_buffer_size_inputs(
        conv_node=conv_node, nhwc_logical=nhwc_logical
    )
    input_offset = cast(int, conv_node.args[6])
    output_offset = cast(int, conv_node.args[7])
    output_qmin = cast(int, conv_node.args[10])
    output_qmax = cast(int, conv_node.args[11])

    # Weight is in OHWI layout after conversion. Packed int4 weights are 1-D,
    # with the OHWI shape in the weight_sizes arg.
    if packed_int4:
        weight_shape = list(cast(list[int], conv_node.args[12]))
    c_out, kernel_h, kernel_w, c_in = weight_shape
    filter_nhwc = [c_out, kernel_h, kernel_w, c_in]
    if packed_int4 and _use_generic_s4_kernel(
        input_nhwc, filter_nhwc, output_nhwc, padding_hw, stride_hw, dilation_hw
    ):
        # Dilation keeps the wrapper off its 1x1 and 1xN paths, and the
        # arm_convolve_s4 buffer size does not depend on it.
        dilation_hw = [dilation_hw[0], 2]

    return [
        int(
            cmsis_nn.convolve_wrapper_buffer_size(
                backend,
                cmsis_nn.DataType.A8W4 if packed_int4 else cmsis_nn.DataType.A8W8,
                input_nhwc=input_nhwc,
                filter_nhwc=filter_nhwc,
                output_nhwc=output_nhwc,
                padding_hw=padding_hw,
                stride_hw=stride_hw,
                dilation_hw=dilation_hw,
                input_offset=input_offset,
                output_offset=output_offset,
                activation_min=output_qmin,
                activation_max=output_qmax,
            )
        )
    ]


def cmsis_nn_depthwise_conv_buffer_size(
    backend: cmsis_nn.Backend,
    conv_node: torch.fx.Node,
    nhwc_logical: bool = False,
) -> list[int]:
    (
        input_nhwc,
        weight_shape,
        output_nhwc,
        stride_hw,
        padding_hw,
        dilation_hw,
    ) = _get_common_conv_buffer_size_inputs(
        conv_node=conv_node, nhwc_logical=nhwc_logical
    )
    depth_multiplier = cast(int, conv_node.args[6])
    input_offset = cast(int, conv_node.args[7])
    output_offset = cast(int, conv_node.args[8])
    output_qmin = cast(int, conv_node.args[11])
    output_qmax = cast(int, conv_node.args[12])

    # Weight is in IHWO layout after conversion.
    _, kernel_h, kernel_w, c_out = weight_shape
    filter_nhwc = [c_out, kernel_h, kernel_w, 1]

    return [
        int(
            cmsis_nn.depthwise_conv_wrapper_buffer_size(
                backend,
                cmsis_nn.DataType.A8W8,
                input_nhwc=input_nhwc,
                filter_nhwc=filter_nhwc,
                output_nhwc=output_nhwc,
                padding_hw=padding_hw,
                stride_hw=stride_hw,
                dilation_hw=dilation_hw,
                ch_mult=depth_multiplier,
                input_offset=input_offset,
                output_offset=output_offset,
                activation_min=output_qmin,
                activation_max=output_qmax,
            )
        )
    ]


def cmsis_nn_batch_matmul_buffer_size(
    backend: cmsis_nn.Backend,
    matmul_node: torch.fx.Node,
) -> list[int]:
    rhs_transposed = cast(torch.fx.Node, matmul_node.args[2])
    rhs_shape = _shape_from_node(rhs_transposed)

    _, rhs_cols, inner = rhs_shape

    return [
        int(
            cmsis_nn.fully_connected_buffer_size(
                backend,
                cmsis_nn.DataType.A8W8,
                filter_nhwc=[inner, -1, -1, rhs_cols],  # H and W values are unused.
            )
        )
    ]


def cmsis_nn_transpose_conv_buffer_size(
    backend: cmsis_nn.Backend,
    conv_node: torch.fx.Node,
    nhwc_logical: bool = False,
) -> list[int]:
    (
        input_nhwc,
        weight_shape,
        output_nhwc,
        stride_hw,
        padding_hw,
        dilation_hw,
    ) = _get_common_conv_buffer_size_inputs(
        conv_node=conv_node,
        nhwc_logical=nhwc_logical,
        stride_arg_idx=3,
        padding_arg_idx=4,
        dilation_arg_idx=6,
    )
    output_padding = cast(list[int], conv_node.args[5])
    input_offset = cast(int, conv_node.args[7])
    output_offset = cast(int, conv_node.args[8])
    output_qmin = cast(int, conv_node.args[11])
    output_qmax = cast(int, conv_node.args[12])
    c_out, kernel_h, kernel_w, kernel_c_in = weight_shape
    filter_nhwc = [c_out, kernel_h, kernel_w, kernel_c_in]
    padding_offsets_hw = [int(output_padding[0]), int(output_padding[1])]

    buffer_bytes, reverse_buffer_bytes = (
        int(
            cmsis_nn.transpose_conv_buffer_size(
                backend,
                cmsis_nn.DataType.A8W8,
                input_nhwc=input_nhwc,
                filter_nhwc=filter_nhwc,
                output_nhwc=output_nhwc,
                padding_hw=padding_hw,
                stride_hw=stride_hw,
                dilation_hw=dilation_hw,
                padding_offsets_hw=padding_offsets_hw,
                input_offset=input_offset,
                output_offset=output_offset,
                activation_min=output_qmin,
                activation_max=output_qmax,
            )
        ),
        int(
            cmsis_nn.transpose_conv_reverse_conv_buffer_size(
                backend,
                cmsis_nn.DataType.A8W8,
                input_nhwc=input_nhwc,
                filter_nhwc=filter_nhwc,
                padding_hw=padding_hw,
                stride_hw=stride_hw,
                dilation_hw=dilation_hw,
                padding_offsets_hw=padding_offsets_hw,
                input_offset=input_offset,
                output_offset=output_offset,
                activation_min=output_qmin,
                activation_max=output_qmax,
            )
        ),
    )

    # A zero reverse buffer means CMSIS-NN selected its rolling-buffer path.
    if reverse_buffer_bytes == 0:
        # The sizing API uses stride.h where the kernel uses stride.w. Preserve
        # the API's requirement while ensuring enough space for the kernel.
        # TODO: Remove this correction once our CMSIS-NN pin includes
        # https://github.com/ARM-software/CMSIS-NN/pull/243.
        buffer_w = (input_nhwc[2] - 1) * stride_hw[1] + max(kernel_w, stride_hw[1])
        buffer_h = max(kernel_h, stride_hw[0])
        buffer_bytes = max(buffer_bytes, buffer_w * buffer_h * c_out * 4)

    return [buffer_bytes, reverse_buffer_bytes]


def cmsis_nn_avgpool_buffer_size(
    backend: cmsis_nn.Backend,
    pool_node: torch.fx.Node,
    nhwc_logical: bool = False,
) -> list[int]:
    x = cast(torch.fx.Node, pool_node.args[0])

    if nhwc_logical:
        _, _, _, c_in = _shape_from_node(x)
        _, _, out_w, _ = _shape_from_node(pool_node)
    else:
        _, c_in, _, _ = _shape_from_node(x)
        _, _, _, out_w = _shape_from_node(pool_node)

    return [
        int(
            cmsis_nn.avgpool_buffer_size(
                backend,
                cmsis_nn.DataType.A8W8,
                dim_dst_width=int(out_w),
                ch_src=int(c_in),
            )
        )
    ]


_target_to_buffer_sizes_registry: dict[Any, BufferSizeFunction] = {
    exir_ops.edge.cortex_m.quantized_conv2d.default: cmsis_nn_conv_buffer_size,
    exir_ops.edge.cortex_m.quantized_conv2d_nhwc.default: partial(
        cmsis_nn_conv_buffer_size, nhwc_logical=True
    ),
    exir_ops.edge.cortex_m.quantized_conv2d_s4.default: partial(
        cmsis_nn_conv_buffer_size, packed_int4=True
    ),
    exir_ops.edge.cortex_m.quantized_conv2d_s4_nhwc.default: partial(
        cmsis_nn_conv_buffer_size, nhwc_logical=True, packed_int4=True
    ),
    exir_ops.edge.cortex_m.quantized_depthwise_conv2d.default: cmsis_nn_depthwise_conv_buffer_size,
    exir_ops.edge.cortex_m.quantized_depthwise_conv2d_nhwc.default: partial(
        cmsis_nn_depthwise_conv_buffer_size, nhwc_logical=True
    ),
    exir_ops.edge.cortex_m.quantized_batch_matmul.default: cmsis_nn_batch_matmul_buffer_size,
    exir_ops.edge.cortex_m.quantized_transpose_conv2d.default: cmsis_nn_transpose_conv_buffer_size,
    exir_ops.edge.cortex_m.quantized_transpose_conv2d_nhwc.default: partial(
        cmsis_nn_transpose_conv_buffer_size, nhwc_logical=True
    ),
    exir_ops.edge.cortex_m.quantized_avg_pool2d.default: cmsis_nn_avgpool_buffer_size,
    exir_ops.edge.cortex_m.quantized_avg_pool2d_nhwc.default: partial(
        cmsis_nn_avgpool_buffer_size, nhwc_logical=True
    ),
}


def required_cmsis_nn_buffer_sizes(
    node: torch.fx.Node, backend: cmsis_nn.Backend
) -> list[int] | None:
    """Returns a sequence of scratch buffer sizes required by node, in bytes.
    If no function is registered to compute this for the target of the node, return None.
    """
    if node.target not in _target_to_buffer_sizes_registry:
        return None

    buffer_size_function = _target_to_buffer_sizes_registry[node.target]
    return buffer_size_function(backend, node)
