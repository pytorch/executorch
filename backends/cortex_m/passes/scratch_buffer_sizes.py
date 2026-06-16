# Copyright 2026 Arm Limited and/or its affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from collections.abc import Callable
from typing import Any, cast

import cmsis_nn  # type: ignore[import-not-found, import-untyped]
import executorch.backends.cortex_m.ops.operators  # noqa

import torch
import torch.fx

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

    # Input is NCHW (PyTorch); CMSIS-NN wants NHWC dims.
    n, c_in, height, width = _shape_from_node(x)

    weight_shape = _shape_from_node(weight)

    # Output is NCHW; convert to NHWC dims.
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


def cmsis_nn_conv_buffer_size(
    backend: cmsis_nn.Backend,
    conv_node: torch.fx.Node,
) -> list[int]:
    (
        input_nhwc,
        weight_shape,
        output_nhwc,
        stride_hw,
        padding_hw,
        dilation_hw,
    ) = _get_common_conv_buffer_size_inputs(conv_node=conv_node)
    input_offset = cast(int, conv_node.args[6])
    output_offset = cast(int, conv_node.args[7])
    output_qmin = cast(int, conv_node.args[10])
    output_qmax = cast(int, conv_node.args[11])

    # Weight is in OHWI layout after conversion.
    c_out, kernel_h, kernel_w, c_in = weight_shape
    filter_nhwc = [c_out, kernel_h, kernel_w, c_in]

    return [
        int(
            cmsis_nn.convolve_wrapper_buffer_size(
                backend,
                cmsis_nn.DataType.A8W8,
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
) -> list[int]:
    (
        input_nhwc,
        weight_shape,
        output_nhwc,
        stride_hw,
        padding_hw,
        dilation_hw,
    ) = _get_common_conv_buffer_size_inputs(conv_node=conv_node)
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

    return [
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
    ]


def cmsis_nn_avgpool_buffer_size(
    backend: cmsis_nn.Backend,
    pool_node: torch.fx.Node,
) -> list[int]:
    x = cast(torch.fx.Node, pool_node.args[0])

    # Input is NCHW (PyTorch); CMSIS-NN's avgpool buffer sizer only needs the
    # input channel count and output width.
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
    exir_ops.edge.cortex_m.quantized_depthwise_conv2d.default: cmsis_nn_depthwise_conv_buffer_size,
    exir_ops.edge.cortex_m.quantized_batch_matmul.default: cmsis_nn_batch_matmul_buffer_size,
    exir_ops.edge.cortex_m.quantized_transpose_conv2d.default: cmsis_nn_transpose_conv_buffer_size,
    exir_ops.edge.cortex_m.quantized_avg_pool2d.default: cmsis_nn_avgpool_buffer_size,
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
