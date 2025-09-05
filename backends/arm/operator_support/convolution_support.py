# Copyright 2025 Arm Limited and/or its affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from typing import cast

import torch
import torch.fx as fx
from executorch.backends.arm.operator_support.tosa_supported_operators import (
    register_tosa_support_check,
    SupportedTOSAOperatorCheck,
)
from executorch.backends.arm.tosa import TosaSpecification

from executorch.exir.dialects._ops import ops as exir_ops


@register_tosa_support_check
class ConvolutionSupported(SupportedTOSAOperatorCheck):
    targets = [exir_ops.edge.aten.convolution.default]

    tosa_specs = [
        TosaSpecification.create_from_string("TOSA-1.0+INT"),
        TosaSpecification.create_from_string("TOSA-1.0+FP"),
    ]

    def is_node_tosa_supported(self, node: fx.Node, tosa_spec: TosaSpecification):

        # Not implemented
        transposed = cast(bool, node.args[6])
        output_padding = cast(list[int], node.args[7])
        if transposed:
            return False

        for pad in output_padding:
            if pad != 0:
                self.reporter.report_reject(
                    node, "Convolutions with non-zero output padding not implemented."
                )
                return False

        # Hardware specific constraints
        if tosa_spec.is_U55_subset:
            return self._is_node_supported_u55(node)
        else:
            return True

    def _is_node_supported_u55(self, node: fx.Node):
        """Hardware constraints for Ethos-U-55 case, Vela 4.2.0 (25.02 release)"""

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
        """This condition is somewhat complex but boils down
        to not supporting stride > 3, unless we have some special conditions.
        This condition is a simplified, relaxed version of the hardware constraint,
        since the actual constraint requires information not available
        here (without a lot of work).

        This means that we might accept ops that are not actually supported.
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
