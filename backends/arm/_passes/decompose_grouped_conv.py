# Copyright 2025 Arm Limited and/or its affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from copy import copy

import torch
from executorch.backends.arm._passes.quant_args import QuantArgs
from executorch.exir.dialects._ops import ops as exir_ops
from executorch.exir.pass_base import ExportPass


class DecomposeGroupedConv(ExportPass):
    """
    Splits a grouped convolution which is not supported by TOSA into multiple
    convolutions using slice->conv->cat.

    Before pass:
        x = conv(input, weight, bias, groups = 2)

    After pass:
        input1 = slice(input)
        weight1 = slice(weight)
        bias1 = slice(bias)
        x1 = conv(input1, weight1, bias1)

        input2 = slice(input)
        weight2 = slice(weight)
        bias2 = slice(bias)
        x2 = conv(input2, weight2, bias2)

        x = cat(x1, x2)
    """

    @staticmethod
    def _get_decomposition(op):
        match op:
            case exir_ops.edge.aten.convolution.default:
                return (
                    exir_ops.edge.aten.slice_copy.Tensor,
                    exir_ops.edge.aten.convolution.default,
                    exir_ops.edge.aten.cat.default,
                )
            case torch.ops.aten.conv2d.default:
                return (
                    torch.ops.aten.slice_copy.Tensor,
                    torch.ops.aten.conv2d.default,
                    torch.ops.aten.cat.default,
                )
            case _:
                raise RuntimeError("Invalid op for grouped conv decomposition")

    @staticmethod
    def _split_per_channel_qparams(qarg, index, output_slice_size):
        if qarg is not None and qarg.per_channel:
            start_index = index * output_slice_size
            stop_index = (index + 1) * output_slice_size
            return QuantArgs(
                scale=qarg.scale[start_index:stop_index],
                zp=qarg.zp[start_index:stop_index],
                qmin=qarg.qmin,
                qmax=qarg.qmax,
                dtype=qarg.dtype,
                axis=qarg.axis,
                per_channel=qarg.per_channel,
            )
        return qarg

    @staticmethod
    def _get_meta_copy(meta, i, output_slice_size):
        meta_copy = meta.copy()
        if "input_qparams" in meta.data and len(meta.data["input_qparams"]) > 0:
            # Handle per-channel quantization by splitting quantization params
            # similarly to how activations/weights/biases are split.
            new_qparams = meta.data.get("input_qparams").copy()
            # Get quantization params of the weights and slice them.
            qarg = new_qparams[1]
            new_qparams[1] = DecomposeGroupedConv._split_per_channel_qparams(
                qarg, index=i, output_slice_size=output_slice_size
            )

            meta_copy.data["input_qparams"] = new_qparams

        return meta_copy

    def call_operator(self, op, args, kwargs, meta):
        if op == exir_ops.edge.aten.convolution.default:
            groups = args[8]
            transposed = args[6]
        elif op == torch.ops.aten.conv2d.default:
            groups = args[6]
            transposed = False
        else:
            return super().call_operator(op, args, kwargs, meta)

        if groups == 1 or transposed:
            return super().call_operator(op, args, kwargs, meta)

        input_node = args[0]
        if input_node.data.shape[1] == groups:
            # This is a depthwise convolution which is handled elsewhere
            return super().call_operator(op, args, kwargs, meta)

        weight_node = args[1]
        bias_node = args[2]

        input_slice_size = weight_node.data.shape[1]
        output_slice_size = weight_node.data.shape[0] // groups

        no_q_dq_meta = copy(meta)
        no_q_dq_meta.data = {}
        no_q_dq_meta.data = {}

        slice_op, conv_op, cat_op = DecomposeGroupedConv._get_decomposition(op)

        input_slices = []
        for i in range(groups):
            start_index = i * input_slice_size
            stop_index = (i + 1) * input_slice_size
            slice_args = (input_node, 1, start_index, stop_index)

            input_slices.append(
                super().call_operator(slice_op, slice_args, kwargs, no_q_dq_meta)
            )

        filter_slices = []
        for i in range(groups):
            start_index = i * output_slice_size
            stop_index = (i + 1) * output_slice_size
            slice_args = (weight_node, 0, start_index, stop_index)

            filter_slices.append(
                super().call_operator(slice_op, slice_args, kwargs, no_q_dq_meta)
            )

        bias_slices = []
        for i in range(groups):
            if bias_node is None:
                bias_slices.append(None)
            else:
                start_index = i * output_slice_size
                stop_index = (i + 1) * output_slice_size
                slice_args = (bias_node, 0, start_index, stop_index)

                bias_slices.append(
                    super().call_operator(slice_op, slice_args, kwargs, no_q_dq_meta)
                )

        output_slices = []
        for i, (input_slice, filter_slice, bias_slice) in enumerate(
            zip(input_slices, filter_slices, bias_slices)
        ):

            meta_copy = DecomposeGroupedConv._get_meta_copy(meta, i, output_slice_size)

            if op == exir_ops.edge.aten.convolution.default:
                conv_args = (input_slice, filter_slice, bias_slice, *args[3:8], 1)
            elif op == torch.ops.aten.conv2d.default:
                conv_args = (input_slice, filter_slice, bias_slice, *args[3:6], 1)
            else:
                raise RuntimeError("Invalid op for grouped conv decomposition")

            output_slices.append(
                super().call_operator(conv_op, conv_args, kwargs, meta_copy)
            )

        cat_args = (output_slices, 1)
        # propagate original metadata (including quantization params) to the concatenated output
        return super().call_operator(cat_op, cat_args, kwargs, meta)
