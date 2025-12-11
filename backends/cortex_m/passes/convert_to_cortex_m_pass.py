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
        # PyTorch depthwise weight is [out_ch, 1, H, W] where dimension 1 is 1
        # and groups == input_channels (groups > 1)
        is_depthwise = weight_tensor.shape[1] == 1 and groups > 1

        if is_depthwise:
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

        if is_depthwise:
            # Compute depth_multiplier for depthwise convolution
            # For depthwise: output_channels = input_channels * depth_multiplier
            # PyTorch depthwise weight is [C_OUT, 1, H, W]
            output_channels = weight_tensor.shape[0]
            input_channels = groups  # For depthwise, groups == input_channels

            if output_channels % input_channels != 0:
                raise ValueError(
                    f"Depthwise conv: output_channels ({output_channels}) must be "
                    f"divisible by input_channels ({input_channels})"
                )
            depth_multiplier = output_channels // input_channels

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

                node.replace_all_uses_with(cortex_m_op)
                graph_module.graph.erase_node(node)

            modified = True

        if modified:
            graph_module.graph.eliminate_dead_code()
            graph_module.recompile()
            graph_module = super().call(graph_module).graph_module

        return PassResult(graph_module, modified)
