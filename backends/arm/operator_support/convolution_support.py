# Copyright 2025-2026 Arm Limited and/or its affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.
"""Declare operator support for ``aten.convolution`` in TOSA.

Provide general checks and hardware-specific constraints (e.g., U55 subset) for
convolution nodes prior to delegation to the TOSA backend.

"""

from typing import cast

import torch
import torch.fx as fx
from executorch.backends.arm._passes.arm_pass_utils import (
    expand_around_channel,
    get_first_fake_tensor,
)
from executorch.backends.arm.operator_support.tosa_supported_operators import (
    register_tosa_support_check,
    SupportedTOSAOperatorCheck,
)
from executorch.backends.arm.tosa import TosaSpecification

from executorch.exir.dialects._ops import ops as exir_ops


@register_tosa_support_check
class ConvolutionSupported(SupportedTOSAOperatorCheck):
    """Provide TOSA support check for convolutions."""

    targets = [exir_ops.edge.aten.convolution.default]

    tosa_specs = [
        TosaSpecification.create_from_string("TOSA-1.0+INT"),
        TosaSpecification.create_from_string("TOSA-1.0+FP"),
        TosaSpecification.create_from_string("TOSA-1.1+INT"),
        TosaSpecification.create_from_string("TOSA-1.1+FP"),
    ]

    def is_node_tosa_supported(
        self, node: fx.Node, tosa_spec: TosaSpecification
    ) -> bool:
        """Return True if the node is supported by TOSA.

        Reject transposed convolutions and convolutions with non-zero output
        padding. Apply additional hardware-specific constraints for U55.

        """
        transposed = cast(bool, node.args[6])
        output_padding = cast(list[int], node.args[7])
        groups = cast(int, node.args[8])

        if transposed:
            if groups != 1:
                self.reporter.report_reject(
                    node, "Grouped transpose convolutions are not supported."
                )
                return False

            dilation = expand_around_channel(cast(list[int], node.args[5]), 2)
            if any(d != 1 for d in dilation):
                self.reporter.report_reject(
                    node, "Transpose convolutions with dilation are not supported."
                )
                return False

            pad = expand_around_channel(cast(list[int], node.args[4]), 2)
            out_pad = expand_around_channel(output_padding, 2)
            weight_shape = get_first_fake_tensor(cast(fx.Node, node.args[1])).shape
            if len(weight_shape) != 4:
                self.reporter.report_reject(
                    node, "Only 2D transpose convolutions are supported."
                )
                return False
            kernel_h = weight_shape[2]
            kernel_w = weight_shape[3]

            out_pad_top = -pad[0]
            out_pad_bottom = -pad[0] + out_pad[0]
            out_pad_left = -pad[1]
            out_pad_right = -pad[1] + out_pad[1]

            if out_pad_top <= -kernel_h or out_pad_bottom <= -kernel_h:
                self.reporter.report_reject(
                    node, "Transpose convolution out_pad exceeds kernel height."
                )
                return False
            if out_pad_left <= -kernel_w or out_pad_right <= -kernel_w:
                self.reporter.report_reject(
                    node, "Transpose convolution out_pad exceeds kernel width."
                )
                return False
        else:
            for output_pad in output_padding:
                if output_pad != 0:
                    self.reporter.report_reject(
                        node,
                        "Convolutions with non-zero output padding not implemented.",
                    )
                    return False

        # Hardware specific constraints
        if tosa_spec.is_U55_subset:
            return self._is_node_supported_u55(node)
        else:
            return True

    def _is_node_supported_u55(self, node: fx.Node) -> bool:
        """Enforce Ethos-U55-specific constraints (Vela 4.2.0).

        Check channel dimensions, kernel sizes, and stride/pad/dilation
        combinations permitted on U55.

        Args:
            node (fx.Node): Convolution node to validate.

        Returns:
            bool: True if supported; otherwise, False.

        """
        transposed = cast(bool, node.args[6])
        if transposed:
            kernel = cast(fx.Node, node.args[1]).meta["val"].shape
            kernel_h = kernel[2]
            kernel_w = kernel[3] if len(kernel) > 3 else 1
            if kernel_h != kernel_w:
                self.reporter.report_reject(
                    node,
                    f"Transpose convolution on U55 requires square kernels, got ({kernel_w}, {kernel_h}).",
                )
                return False

            strides = expand_around_channel(cast(list[int], node.args[3]), 2)
            if strides[0] != strides[1]:
                self.reporter.report_reject(
                    node,
                    f"Transpose convolution on U55 requires equal strides, got {strides}.",
                )
                return False

        shape_in = cast(torch.Tensor, node.all_input_nodes[0].meta["val"]).shape
        shape_out = node.meta["val"].shape
        kernel = cast(fx.Node, node.args[1]).meta["val"].shape
        group = cast(int, node.args[8])

        C_in = shape_in[1]
        C_out = shape_out[1]
        if (C_in == group) and (C_out % C_in) == 0 and len(shape_in) <= 4:
            # Depthwise convolution
            for dim in shape_in[1:]:
                if not 1 <= dim <= 65536:
                    self.reporter.report_reject(
                        node,
                        f"Depthwise convolution must have CWH <= 65536, got {dim})",
                    )
                    return False
        else:
            # Convolution
            if not 1 <= C_in <= 65536:
                self.reporter.report_reject(
                    node, f"Convolution must have C <= 65536, got {C_in})"
                )
                return False

        kernel_w = kernel[2]
        kernel_h = kernel[3] if len(kernel) > 3 else 1
        kernel_z = kernel[4] if len(kernel) > 4 else 1
        # Kernel condition misses constraint on sum of absolute weights
        if not 1 <= kernel_h <= 64 or not 1 <= kernel_w * kernel_h <= 4096:
            self.reporter.report_reject(
                node,
                f"Convolution needs to have kernel_y<=64, kernel_x*kernel_y<=4096, got kernel ({kernel_w}, {kernel_h})",
            )
            return False
        if kernel_z != 1:
            self.reporter.report_reject(
                node, f"Convolution3d needs to have kernel_z==1, got {kernel_z}."
            )
            return False

        if not self._stride_condition(node):
            self.reporter.report_reject(
                node, "Failed condition on stride, pad and dilation combination."
            )
            return False

        return True

    def _stride_condition(self, node: fx.Node) -> bool:
        """Check a simplified stride/padding/dilation constraint.

        Disallow strides greater than 3 unless there is no padding and the
        dilation is 1. For 3D convolutions, enforce ``stride_z <= 1``.

        Args:
            node (fx.Node): Convolution node to evaluate.

        Returns:
            bool: True if the condition is satisfied.

        """
        strides = cast(list[int], node.args[3])
        has_padding = any(pad > 0 for pad in cast(list[int], node.args[4]))
        dilations = cast(list[int], node.args[5])
        if len(dilations) == 1:
            dilations = [dilations[0]] * 2
        if len(strides) == 1:
            strides = [strides[0]] * 2

        if len(strides) > 2:
            stride_z = strides[2]
            if stride_z > 1:
                self.reporter.report_reject(
                    node, f"Convolution3d only supports stride_z<=1, got {stride_z}."
                )
                return False

        for stride, dilation in zip(strides, dilations):
            stride_condition = 1 <= stride <= 3
            dilation_condition = (not has_padding) and (dilation == 1)
            if (not stride_condition) and (not dilation_condition):
                return False

        return True
