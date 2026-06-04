# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
# Copyright 2025-2026 Arm Limited and/or its affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import executorch.backends.cortex_m.ops.operators  # noqa
import executorch.exir as exir

import torch
import torch.fx
from executorch.backends.arm._passes.arm_pass_utils import get_first_fake_tensor

from executorch.backends.cortex_m.passes.cortex_m_pass import CortexMPass
from executorch.backends.cortex_m.passes.passes_utils import (
    build_activation_lut,
    quantize_multiplier_aot,
    to_physical_order,
)
from executorch.backends.cortex_m.passes.scratch_buffer_sizes import (
    required_cmsis_nn_buffer_sizes,
)

from executorch.backends.transforms.utils import (
    create_constant_placeholder,
    get_param_tensor,
    is_param_node,
)
from executorch.exir.dialects._ops import ops as exir_ops
from executorch.exir.passes import make_alloc_node
from torch._subclasses.fake_tensor import FakeTensorMode

from torch.export.graph_signature import InputKind
from torch.fx.passes.infra.pass_manager import PassResult


class ConvertToCortexMPass(CortexMPass):
    """
    Cortex-M backend pass for replacing supported quantized kernels with Cortex-M
    accelerated kernels.

    Used for ops which require changes to input tensors which is not supported
    by call_operator.
    """

    def _create_uninitialized_alloc_node(self):
        """Create an unitialized alloc node to be initialize at a later point."""
        with FakeTensorMode() as mode:
            return make_alloc_node(
                self.exported_program.graph_module,
                mode.from_tensor(torch.empty(0)),
                None,
            )

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

    def _get_convolution_replacement(self, node):
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

        param_weight_tensor = get_param_tensor(self.exported_program, weight)
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
            weight_permuted = param_weight_tensor.permute(1, 2, 3, 0).contiguous()
        else:
            # For regular conv: OIHW -> OHWI
            # The permute achieves the desired logical layout (OHWI). CMSIS-NN expects
            # weights in physically contiguous memory after the permute (not in channels-last)
            # so we use contiguous() here.
            weight_permuted = param_weight_tensor.permute(0, 2, 3, 1).contiguous()

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

        with node.graph.inserting_before(node):
            scratch = self._create_uninitialized_alloc_node()

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
                scratch,
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
                scratch,
            )
            return exir_ops.edge.cortex_m.quantized_conv2d.default, new_args

    def _initialize_alloc_node_size(self, node: torch.fx.Node) -> None:
        """For nodes with a registered buffer size function for node.target, set the buffer sizes
        of the last n args, which should be exir.memory.alloc nodes. For nodes without a
        registered function, do nothing.
        """

        scratch_buffer_sizes = required_cmsis_nn_buffer_sizes(
            node, self.target_config.backend
        )
        if scratch_buffer_sizes is None:
            return

        # Assume that scratch_buffer_sizes are given from left to right in the call signature of node.target.
        for i, scratch_buffer_size in enumerate(reversed(scratch_buffer_sizes)):
            scratch_arg = node.args[-(i + 1)]
            if (
                not isinstance(scratch_arg, torch.fx.Node)
                or scratch_arg.target != exir.memory.alloc
            ):
                raise RuntimeError(
                    f"Expected scratch alloc node as final argument(s) for {node.target}, got {scratch_arg}."
                )

            # buffer size is given in bytes, always use uint8 as dtype.
            scratch_arg.args = (((scratch_buffer_size,), torch.uint8),)

    def _get_transpose_conv2d_replacement(self, node):
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
        weight_tensor_param = get_param_tensor(self.exported_program, weight)
        if weight_tensor_param is None:
            raise RuntimeError(
                f"Expected transpose conv weight parameter tensor for node {node.name}."
            )
        weight_permuted = weight_tensor_param.permute(1, 2, 3, 0).contiguous()

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

        with node.graph.inserting_before(node):
            scratch = self._create_uninitialized_alloc_node()
            output_scratch = self._create_uninitialized_alloc_node()

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
        return exir_ops.edge.cortex_m.quantized_transpose_conv2d.default, new_args

    def _get_bmm_replacement(self, node):
        lhs_scale = node.meta["input_qparams"][0].scale
        lhs_zp = node.meta["input_qparams"][0].zp
        rhs_scale = node.meta["input_qparams"][1].scale
        rhs_zp = node.meta["input_qparams"][1].zp
        output_scale = node.meta["output_qparams"][0].scale
        output_zp = node.meta["output_qparams"][0].zp

        output_mult, output_shift = quantize_multiplier_aot(
            (lhs_scale * rhs_scale) / output_scale
        )

        lhs_node = node.args[0]
        rhs_node = node.args[1]

        is_constant_rhs = is_param_node(self.exported_program, rhs_node)
        if is_constant_rhs:
            rhs_tensor = get_param_tensor(self.exported_program, rhs_node)
            rhs_transposed_tensor = rhs_tensor.permute(0, 2, 1).contiguous()
            with node.graph.inserting_after(rhs_node):
                rhs_transposed = create_constant_placeholder(
                    self.exported_program,
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

        with node.graph.inserting_before(node):
            scratch = self._create_uninitialized_alloc_node()

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
        return exir_ops.edge.cortex_m.quantized_batch_matmul.default, args

    def _get_avg_pool2d_replacement(self, node):
        kernel_size = [int(v) for v in node.args[1]]
        stride_arg = node.args[2] if len(node.args) > 2 else None
        stride = [int(v) for v in stride_arg] if stride_arg else list(kernel_size)
        padding_arg = node.args[3] if len(node.args) > 3 else None
        padding = [int(v) for v in padding_arg] if padding_arg else [0, 0]
        ceil_mode = bool(node.args[4]) if len(node.args) > 4 else False
        count_include_pad = bool(node.args[5]) if len(node.args) > 5 else True
        divisor_override = node.args[6] if len(node.args) > 6 else None

        # CMSIS-NN avg pool does not model ceil_mode or divisor_override; leave
        # the aten op in place so a portable kernel can handle it instead.
        if ceil_mode or divisor_override is not None:
            return None

        input_node = node.args[0]
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

        with node.graph.inserting_before(node):
            scratch = self._create_uninitialized_alloc_node()

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
        return exir_ops.edge.cortex_m.quantized_avg_pool2d.default, new_args

    def _get_activation_replacement(self, node):
        """Lower a standalone quantized sigmoid / tanh / silu to a single
        cortex_m.quantized_activation call backed by an AoT-built 256-entry
        int8 LUT. The kernel is shape-agnostic; the LUT encodes both the
        activation function and the input/output qparams.
        """
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
                self.exported_program,
                node.graph,
                node.name + "_lut",
                InputKind.PARAMETER,
                lut_tensor,
            )

        new_args = (node.args[0], lut_node)
        return exir_ops.edge.cortex_m.quantized_activation.default, new_args

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
                case exir_ops.edge.aten.bmm.default:
                    op, args = self._get_bmm_replacement(node)
                case exir_ops.edge.aten.avg_pool2d.default:
                    result = self._get_avg_pool2d_replacement(node)
                    if result is None:
                        continue
                    op, args = result
                case (
                    exir_ops.edge.aten.sigmoid.default
                    | exir_ops.edge.aten.tanh.default
                    | exir_ops.edge.aten.silu.default
                ):
                    op, args = self._get_activation_replacement(node)
                case _:
                    continue

            with graph_module.graph.inserting_before(node):
                cortex_m_op = graph_module.graph.create_node(
                    "call_function",
                    target=op,
                    args=args,
                    kwargs={},
                )
                self._initialize_alloc_node_size(cortex_m_op)

                node.replace_all_uses_with(cortex_m_op)
                graph_module.graph.erase_node(node)

            modified = True

        if modified:
            graph_module.graph.eliminate_dead_code()
            graph_module.recompile()
            graph_module = super().call(graph_module).graph_module

        return PassResult(graph_module, modified)
