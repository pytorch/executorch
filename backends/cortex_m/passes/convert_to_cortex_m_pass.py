# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
# Copyright 2025-2026 Arm Limited and/or its affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.


from typing import Sequence

import executorch.backends.cortex_m.ops.operators  # noqa
import executorch.exir as exir

import torch
import torch.fx

from cmsisnn_sizes import convolve_wrapper_s8_buffer_size_mve

from executorch.backends.arm._passes.arm_pass_utils import get_first_fake_tensor
from executorch.backends.cortex_m.passes.passes_utils import quantize_multiplier_aot

from executorch.backends.transforms.utils import (
    create_constant_placeholder,
    get_param_tensor,
)

from executorch.backends.xnnpack._passes.xnnpack_pass import XNNPACKPass
from executorch.exir.dialects._ops import ops as exir_ops
from executorch.exir.tensor import TensorSpec
from torch.export.graph_signature import InputKind
from torch.fx.passes.infra.pass_manager import PassResult


def shape_from_node(n: torch.fx.Node) -> list[int]:
    spec = n.meta.get("spec", None)
    if spec is not None and getattr(spec, "shape", None) is not None:
        return [int(s) for s in spec.shape]

    v = n.meta.get("val", None)
    if v is not None:
        if isinstance(v, (tuple, list)):
            v = v[0]
        return [int(s) for s in v.shape]

    raise KeyError(f"No shape meta on node {n.format_node()} (need spec or val)")


def cmsisnn_conv_s8_required_bytes_mve(
    *,
    x: torch.fx.Node,
    conv_node: torch.fx.Node,
    weight_shape: Sequence[int],
    stride: Sequence[int],
    padding: Sequence[int],
    dilation: Sequence[int],
    input_zero_point: int,
    output_zero_point: int,
    output_qmin: int,
    output_qmax: int,
) -> int:
    # Input is NCHW (PyTorch); CMSIS-NN wants NHWC dims.
    N, C_in, H, W = shape_from_node(x)

    # Weight is (C_out, C_in/groups, kH, kW) in PyTorch
    C_out, _, kH, kW = map(int, weight_shape)

    # Output is NCHW; convert to NHWC dims.
    N2, C_out2, H_out, W_out = shape_from_node(conv_node)

    input_nhwc = [N, H, W, C_in]
    filter_nhwc = [
        C_out,
        kH,
        kW,
        C_in,
    ]  # CMSIS-NN convention: n=out_ch, h=kH, w=kW, c=in_ch
    output_nhwc = [N2, H_out, W_out, C_out2]

    stride_hw = [int(stride[0]), int(stride[1])]
    padding_hw = [int(padding[0]), int(padding[1])]
    dilation_hw = [int(dilation[0]), int(dilation[1])]

    # CMSIS-NN conv_params offsets are "negative of zero point"
    input_offset = -int(input_zero_point)
    output_offset = -int(output_zero_point)

    return int(
        convolve_wrapper_s8_buffer_size_mve(
            input_nhwc=input_nhwc,
            filter_nhwc=filter_nhwc,
            output_nhwc=output_nhwc,
            padding_hw=padding_hw,
            stride_hw=stride_hw,
            dilation_hw=dilation_hw,
            input_offset=input_offset,
            output_offset=output_offset,
            activation_min=int(output_qmin),
            activation_max=int(output_qmax),
        )
    )


class ConvertToCortexMPass(XNNPACKPass):
    """
    Cortex-M backend pass for replacing supported quantized kernels with Cortex-M
    accelerated kernels.

    Used for ops which require changes to input tensors which is not supported
    by call_operator.
    """

    def _compute_kernel_sum(self, weights, bias, input_offset, weight_offset):
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

    def _get_batch_size_from_conv(self, conv_node: torch.fx.Node):
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

    def _get_linear_replacement(self, node):
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
        input_scale = node.meta["input_qparams"][0].scale
        input_zp = node.meta["input_qparams"][0].zp
        weight_scale = node.meta["input_qparams"][1].scale
        weight_zp = node.meta["input_qparams"][1].zp
        output_scale = node.meta["output_qparams"][0].scale
        output_zp = node.meta["output_qparams"][0].zp
        output_min = node.meta["output_qparams"][0].qmin
        output_max = node.meta["output_qparams"][0].qmax

        quantized_multiplier, quantized_shift = quantize_multiplier_aot(
            (input_scale * weight_scale) / output_scale
        )

        # TODO: Add support for configuring the backend to support other extensions.
        # Kernel sum is only used in the CMSIS-NN implementation for the MVE extension,
        # so this should be optional.
        weights = node.args[1]
        weights_tensor = get_param_tensor(self.exported_program, weights)
        bias_tensor = (
            get_param_tensor(self.exported_program, node.args[2])
            if len(node.args) > 2
            else None
        )
        kernel_sum_tensor = self._compute_kernel_sum(
            weights_tensor, bias_tensor, -input_zp, -weight_zp
        )
        with node.graph.inserting_after(weights):
            kernel_sum = create_constant_placeholder(
                self.exported_program,
                node.graph,
                node.name + "_kernel_sum",
                InputKind.PARAMETER,
                kernel_sum_tensor,
            )

        args = (
            node.args[0],
            weights,
            None,
            kernel_sum,
            -input_zp,
            -weight_zp,
            output_zp,
            [quantized_multiplier],
            [quantized_shift],
            output_max,
            output_min,
        )

        return exir_ops.edge.cortex_m.quantized_linear.default, args

    def _get_convolution_replacement(self, node) -> int:
        (
            x,
            weight,
            bias,
            stride,
            padding,
            dilation,
            transposed,
            output_padding,
            groups,
        ) = node.args

        # Extract values
        input_scale = node.meta["input_qparams"][0].scale
        input_zero_point = node.meta["input_qparams"][0].zp
        weight_scales = node.meta["input_qparams"][1].scale
        if not isinstance(weight_scales, list):
            weight_tensor = get_first_fake_tensor(weight)
            weight_scales = [weight_scales] * weight_tensor.shape[0]

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

        weight_tensor = get_param_tensor(self.exported_program, weight)

        # Detect depthwise convolution:
        # Depthwise means groups == in_channels, out_channels == K * in_channels
        # Weight shape is [out_ch, in_ch_per_group, H, W]
        in_channels = weight_tensor.shape[1] * groups
        out_channels = weight_tensor.shape[0]
        is_depthwise = (in_channels == groups) and (out_channels % in_channels == 0)

        # Only use DW path if batch_size==1, as CMSIS-NN DW falls back to
        # unoptimized implementation otherwise.
        batch_size = self._get_batch_size_from_conv(node)

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
            weight_permuted = weight_tensor.permute(1, 2, 3, 0).contiguous(
                memory_format=torch.channels_last
            )
        else:
            # For regular conv: OIHW -> OHWI
            weight_permuted = weight_tensor.permute(0, 2, 3, 1).contiguous(
                memory_format=torch.channels_last
            )

        with node.graph.inserting_after(weight):
            weight_nhwc = create_constant_placeholder(
                self.exported_program,
                node.graph,
                node.name + "_weight_nhwc",
                InputKind.PARAMETER,
                weight_permuted,
            )

            quantized_multiplier_tensor = create_constant_placeholder(
                self.exported_program,
                node.graph,
                node.name + "_quantized_multiplier",
                InputKind.PARAMETER,
                torch.tensor(quantized_multipliers, dtype=torch.int32),
            )

            quantized_shift_tensor = create_constant_placeholder(
                self.exported_program,
                node.graph,
                node.name + "_quantized_shift",
                InputKind.PARAMETER,
                torch.tensor(quantized_shifts, dtype=torch.int32),
            )

        fake_size = 2000  # TODO add DW conv get buffer size function
        if use_depthwise_conv:
            required_bytes = fake_size
        else:
            weight_shape = get_first_fake_tensor(weight).shape
            required_bytes = cmsisnn_conv_s8_required_bytes_mve(
                x=x,
                conv_node=node,
                weight_shape=weight_shape,
                stride=stride,
                padding=padding,
                dilation=dilation,
                input_zero_point=input_zero_point,
                output_zero_point=output_zero_point,
                output_qmin=output_qmin,
                output_qmax=output_qmax,
            )
        print("required_bytes = ", required_bytes)

        graph = self.exported_program.graph_module.graph
        with graph.inserting_before(node):
            scratch = graph.call_function(
                exir.memory.alloc, args=(((required_bytes,), torch.uint8),), kwargs={}
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

            new_args = (
                x,
                scratch,
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
            )
            return exir_ops.edge.cortex_m.quantized_depthwise_conv2d.default, new_args
        else:
            # Use regular convolution operator
            new_args = (
                x,
                scratch,
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
            )
            return exir_ops.edge.cortex_m.quantized_conv2d.default, new_args

    def call(self, graph_module: torch.fx.GraphModule) -> PassResult:
        modified = False
        for node in graph_module.graph.nodes:
            if node.op != "call_function":
                continue
            if (
                node.meta.get("input_qparams", {}) == {}
                or node.meta.get("output_qparams", {}) == {}
            ):
                continue

            match node.target:
                case exir_ops.edge.aten.linear.default:
                    op, args = self._get_linear_replacement(node)
                case exir_ops.edge.aten.convolution.default:
                    op, args = self._get_convolution_replacement(node)
                case _:
                    continue

            with graph_module.graph.inserting_before(node):
                cortex_m_op = graph_module.graph.create_node(
                    "call_function",
                    target=op,
                    args=args,
                    kwargs={},
                )

                cortex_m_op.meta.update(node.meta)  # preserve shape for get buffer size

                node.replace_all_uses_with(cortex_m_op)
                graph_module.graph.erase_node(node)

            modified = True

        if modified:
            graph_module.graph.eliminate_dead_code()
            graph_module.recompile()
            graph_module = super().call(graph_module).graph_module

        return PassResult(graph_module, modified)
