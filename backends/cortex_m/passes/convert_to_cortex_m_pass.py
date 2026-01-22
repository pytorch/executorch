# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
# Copyright 2025 Arm Limited and/or its affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.


import executorch.backends.cortex_m.ops.operators  # noqa

import torch
import torch.fx
from executorch.backends.arm._passes.arm_pass_utils import get_first_fake_tensor
from executorch.backends.cortex_m.passes.passes_utils import quantize_multiplier_aot

from executorch.backends.transforms.utils import (
    create_constant_placeholder,
    get_param_tensor,
)

from executorch.backends.xnnpack._passes.xnnpack_pass import XNNPACKPass
from executorch.exir.dialects._ops import ops as exir_ops
from torch.export.graph_signature import InputKind
from torch.fx.passes.infra.pass_manager import PassResult


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
            # The permute achieves the desired logical layout (IHWO). CMSIS-NN expects
            # weights in physically contiguous memory after the permute (not in channels-last)
            # so we use contiguous() here.
            weight_permuted = weight_tensor.permute(1, 2, 3, 0).contiguous()
        else:
            # For regular conv: OIHW -> OHWI
            # The permute achieves the desired logical layout (OHWI). CMSIS-NN expects
            # weights in physically contiguous memory after the permute (not in channels-last)
            # so we use contiguous() here.
            weight_permuted = weight_tensor.permute(0, 2, 3, 1).contiguous()

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

    def _get_transpose_conv2d_replacement(self, node) -> tuple:
        """
        Transform aten.convolution with transposed=True to cortex_m.quantized_transpose_conv2d
        """
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

        input_scale = node.meta["input_qparams"][0].scale
        input_zero_point = node.meta["input_qparams"][0].zp
        weight_scales = node.meta["input_qparams"][1].scale

        # For transposed conv: weight shape is (in_channels, out_channels/groups, H, W)
        # We need requantization params for each output channel
        weight_tensor = get_first_fake_tensor(weight)
        if not isinstance(weight_scales, list):
            # weight_tensor.shape[1] is out_channels for transposed conv
            num_output_channels = weight_tensor.shape[1]
            weight_scales = [weight_scales] * num_output_channels

        output_qparams = node.meta["output_qparams"][0]
        output_scale = output_qparams.scale
        output_zero_point = output_qparams.zp
        output_qmin = output_qparams.qmin
        output_qmax = output_qparams.qmax

        # Compute per-channel requantization parameters
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
        weight_tensor = get_param_tensor(self.exported_program, weight)
        weight_permuted = weight_tensor.permute(1, 2, 3, 0).contiguous(
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
        )
        return exir_ops.edge.cortex_m.quantized_transpose_conv2d.default, new_args

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
                    # Check if it's transposed convolution (arg index 6)
                    transposed = node.args[6] if len(node.args) > 6 else False
                    if transposed:
                        op, args = self._get_transpose_conv2d_replacement(node)
                    else:
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

                node.replace_all_uses_with(cortex_m_op)
                graph_module.graph.erase_node(node)

            modified = True

        if modified:
            graph_module.graph.eliminate_dead_code()
            graph_module.recompile()
            graph_module = super().call(graph_module).graph_module

        return PassResult(graph_module, modified)
