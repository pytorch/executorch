# Copyright 2025-2026 Arm Limited and/or its affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from copy import copy
from typing import Literal, Protocol, Set, Type, TypeGuard

import torch
from executorch.backends.arm._passes.arm_pass import ArmPass
from executorch.backends.arm._passes.conv1d_unsqueeze_pass import Conv1dUnsqueezePass
from executorch.backends.arm._passes.quant_args import QuantArgs
from executorch.exir.dialects._ops import ops as exir_ops
from executorch.exir.pass_base import ExportPass


class _PerChannelQuantArgs(Protocol):
    scale: list[float]
    zp: list[int]
    qmin: int
    qmax: int
    dtype: torch.dtype
    axis: int
    per_channel: Literal[True]


class DecomposeGroupedConvPass(ArmPass):
    """Splits a grouped convolution which is not supported by TOSA into multiple
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

    _passes_required_after: Set[Type[ExportPass]] = {Conv1dUnsqueezePass}

    @staticmethod
    def _get_decomposition(op):
        match op:
            case exir_ops.edge.aten.convolution.default:
                return (
                    exir_ops.edge.aten.slice_copy.Tensor,
                    exir_ops.edge.aten.convolution.default,
                    exir_ops.edge.aten.cat.default,
                )
            case torch.ops.aten.conv_transpose2d.input:
                return (
                    torch.ops.aten.slice_copy.Tensor,
                    torch.ops.aten.conv_transpose2d.input,
                    torch.ops.aten.cat.default,
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
    def _get_groups_and_transposed(op, args):
        if op == exir_ops.edge.aten.convolution.default:
            return args[8], args[6]
        if op == torch.ops.aten.conv_transpose2d.input:
            return args[6], True
        if op == torch.ops.aten.conv2d.default:
            return args[6], False
        return None, None

    @staticmethod
    def _is_depthwise_conv(input_node, groups, transposed):
        return (not transposed) and input_node.data.shape[1] == groups

    @staticmethod
    def _get_slice_sizes(weight_node, groups, transposed):
        if transposed:
            input_slice_size = weight_node.data.shape[0] // groups
            output_slice_size = weight_node.data.shape[1]
        else:
            input_slice_size = weight_node.data.shape[1]
            output_slice_size = weight_node.data.shape[0] // groups
        return input_slice_size, output_slice_size

    def _slice_inputs(
        self, slice_op, input_node, input_slice_size, groups, meta, kwargs
    ):
        input_slices = []
        for i in range(groups):
            start_index = i * input_slice_size
            stop_index = (i + 1) * input_slice_size
            slice_args = (input_node, 1, start_index, stop_index)
            input_slices.append(
                super().call_operator(slice_op, slice_args, kwargs, meta, updated=True)
            )
        return input_slices

    def _slice_weights(
        self,
        slice_op,
        weight_node,
        groups,
        input_slice_size,
        output_slice_size,
        transposed,
        meta,
        kwargs,
    ):
        weight_slices = []
        for i in range(groups):
            if transposed:
                start_index = i * input_slice_size
                stop_index = (i + 1) * input_slice_size
            else:
                start_index = i * output_slice_size
                stop_index = (i + 1) * output_slice_size
            slice_args = (weight_node, 0, start_index, stop_index)
            weight_slices.append(
                super().call_operator(slice_op, slice_args, kwargs, meta, updated=True)
            )
        return weight_slices

    def _slice_biases(
        self, slice_op, bias_node, groups, output_slice_size, meta, kwargs
    ):
        bias_slices = []
        for i in range(groups):
            if bias_node is None:
                bias_slices.append(None)
                continue
            start_index = i * output_slice_size
            stop_index = (i + 1) * output_slice_size
            slice_args = (bias_node, 0, start_index, stop_index)
            bias_slices.append(
                super().call_operator(slice_op, slice_args, kwargs, meta, updated=True)
            )
        return bias_slices

    @staticmethod
    def _build_conv_args(op, args, input_slice, filter_slice, bias_slice):
        if op == exir_ops.edge.aten.convolution.default:
            return (input_slice, filter_slice, bias_slice, *args[3:8], 1)
        if op == torch.ops.aten.conv_transpose2d.input:
            return (
                input_slice,
                filter_slice,
                bias_slice,
                args[3],
                args[4],
                args[5],
                1,
                args[7],
            )
        if op == torch.ops.aten.conv2d.default:
            return (input_slice, filter_slice, bias_slice, *args[3:6], 1)
        raise RuntimeError("Invalid op for grouped conv decomposition")

    @staticmethod
    def _is_per_channel_qparams(
        qarg: QuantArgs | None,
    ) -> TypeGuard[_PerChannelQuantArgs]:
        return qarg is not None and qarg.per_channel

    @staticmethod
    def _split_per_channel_qparams(
        qarg: _PerChannelQuantArgs, start_index, stop_index
    ) -> QuantArgs:
        return QuantArgs(
            scale=qarg.scale[start_index:stop_index],
            zp=qarg.zp[start_index:stop_index],
            qmin=qarg.qmin,
            qmax=qarg.qmax,
            dtype=qarg.dtype,
            axis=qarg.axis,
            per_channel=qarg.per_channel,
        )

    @staticmethod
    def _get_meta_copy(
        meta,
        i,
        input_slice_size,
        output_slice_size,
        transposed,
    ):
        meta_copy = meta.copy()

        if "input_qparams" in meta.data and len(meta.data["input_qparams"]) > 0:
            # Handle per-channel quantization by splitting quantization params
            # similarly to how activations/weights/biases are split.
            new_qparams = meta.data.get("input_qparams").copy()

            # Get quantization params of the weights and slice them.
            w_qarg = new_qparams[1]
            if DecomposeGroupedConvPass._is_per_channel_qparams(w_qarg):

                # For transpose conv, axis=1 corresponds to output channels and
                # does not align with grouped slicing.
                # Per-channel quantization on axis=0 on the other hand could align here but
                # per-channel quant on axis 0 is very uncommon.
                if transposed:
                    raise RuntimeError(
                        "Grouped transpose conv with per-channel quantization is unsupported"
                    )

                slice_size = output_slice_size
                start_index = i * slice_size
                stop_index = (i + 1) * slice_size
                new_qparams[1] = DecomposeGroupedConvPass._split_per_channel_qparams(
                    w_qarg, start_index=start_index, stop_index=stop_index
                )

            # Split per-channel bias qparams to match per-group output slices.
            if len(new_qparams) > 2:
                b_qarg = new_qparams[2]
                if DecomposeGroupedConvPass._is_per_channel_qparams(b_qarg):
                    start_index = i * output_slice_size
                    stop_index = (i + 1) * output_slice_size
                    new_qparams[2] = (
                        DecomposeGroupedConvPass._split_per_channel_qparams(
                            b_qarg, start_index=start_index, stop_index=stop_index
                        )
                    )

            meta_copy.data["input_qparams"] = new_qparams

        return meta_copy

    def call_operator(self, op, args, kwargs, meta):
        groups, transposed = DecomposeGroupedConvPass._get_groups_and_transposed(
            op, args
        )
        if groups is None:
            return super().call_operator(op, args, kwargs, meta)

        if groups == 1:
            return super().call_operator(op, args, kwargs, meta)

        input_node = args[0]
        if DecomposeGroupedConvPass._is_depthwise_conv(input_node, groups, transposed):
            # This is a depthwise convolution which is handled elsewhere
            return super().call_operator(op, args, kwargs, meta)

        weight_node = args[1]
        bias_node = args[2]

        input_slice_size, output_slice_size = DecomposeGroupedConvPass._get_slice_sizes(
            weight_node, groups, transposed
        )

        no_q_dq_meta = copy(meta)
        no_q_dq_meta.data = {}

        slice_op, conv_op, cat_op = DecomposeGroupedConvPass._get_decomposition(op)

        input_slices = self._slice_inputs(
            slice_op, input_node, input_slice_size, groups, no_q_dq_meta, kwargs
        )
        weight_slices = self._slice_weights(
            slice_op,
            weight_node,
            groups,
            input_slice_size,
            output_slice_size,
            transposed,
            no_q_dq_meta,
            kwargs,
        )
        bias_slices = self._slice_biases(
            slice_op, bias_node, groups, output_slice_size, no_q_dq_meta, kwargs
        )

        output_slices = []
        for i, (input_slice, filter_slice, bias_slice) in enumerate(
            zip(input_slices, weight_slices, bias_slices)
        ):

            meta_copy = DecomposeGroupedConvPass._get_meta_copy(
                meta,
                i,
                input_slice_size,
                output_slice_size,
                transposed,
            )

            conv_args = DecomposeGroupedConvPass._build_conv_args(
                op, args, input_slice, filter_slice, bias_slice
            )

            output_slices.append(
                super().call_operator(
                    conv_op, conv_args, kwargs, meta_copy, updated=True
                )
            )

        cat_args = (output_slices, 1)
        # propagate original metadata (including quantization params) to the concatenated output
        return super().call_operator(cat_op, cat_args, kwargs, meta, updated=True)
