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
from executorch.backends.arm.tosa_specification import Tosa_0_80, TosaSpecification
from executorch.exir.dialects._ops import ops as exir_ops


def kernel_check(kernel: tuple[int, int]) -> bool:
    if not (1 <= kernel[0] * kernel[1] <= 65536):
        return False
    return 1 <= kernel[1] <= 256


def stride_check(strides: tuple[int, int]) -> bool:
    return all(1 <= stride <= 3 for stride in strides)


def dim_check(shape=torch.Size) -> bool:
    check = True
    for dim in shape[1:]:
        check &= 1 <= dim <= 65536
    return check


@register_tosa_support_check
class AvgPool2dSupported(SupportedTOSAOperatorCheck):
    targets = [
        exir_ops.edge.aten.avg_pool2d.default,
    ]

    tosa_specs = [
        TosaSpecification.create_from_string("TOSA-0.80+BI"),
        TosaSpecification.create_from_string("TOSA-0.80+MI"),
        TosaSpecification.create_from_string("TOSA-1.0+INT"),
        TosaSpecification.create_from_string("TOSA-1.0+FP"),
    ]

    def is_node_tosa_supported(self, node: fx.Node, tosa_spec: TosaSpecification):
        if not (isinstance(tosa_spec, Tosa_0_80) and tosa_spec.is_U55_subset):
            return True

        # U55 case, Vela 4.2.0 (25.02 release)
        shape = cast(torch.Tensor, node.all_input_nodes[0].meta["val"]).shape
        kernel = cast(tuple[int, int], node.args[1])
        stride = cast(tuple[int, int], node.args[2])
        if len(node.args) > 3:
            # Padding case
            if not all(1 <= k <= 8 for k in kernel):
                self.reporter.report_reject(
                    node, f"Avgpool2d with padding needs kernel dims < 8, got {kernel}"
                )
                return False
        else:
            if not kernel_check(kernel):
                self.reporter.report_reject(
                    node,
                    f"Avgpool2d needs kernel_y < 256, kernel_x*kernel_y<=65536, got {kernel}",
                )
                return False

        if not dim_check(shape):
            self.reporter.report_reject(
                node,
                f"Avgpool2d needs N == 1, rest dims <= 65536, got shape {list(shape)}",
            )
            return False
        if not stride_check(stride):
            self.reporter.report_reject(
                node, f"Avgpool2d needs stride <= 3, got {stride}"
            )
            return False
        if not shape[0] == 1:
            self.reporter.report_reject(
                node, f"Avgpool2d needs N==1, got N=={shape[0]}"
            )
            return False
        return True


@register_tosa_support_check
class MaxPool2dSupported(SupportedTOSAOperatorCheck):
    targets = [
        exir_ops.edge.aten.max_pool2d_with_indices.default,
    ]

    tosa_specs = [
        TosaSpecification.create_from_string("TOSA-0.80+BI"),
        TosaSpecification.create_from_string("TOSA-0.80+MI"),
        TosaSpecification.create_from_string("TOSA-1.0+INT"),
        TosaSpecification.create_from_string("TOSA-1.0+FP"),
    ]

    def is_node_tosa_supported(self, node: fx.Node, tosa_spec: TosaSpecification):
        if not (isinstance(tosa_spec, Tosa_0_80) and tosa_spec.is_U55_subset):
            return True

        # U55 case, Vela 4.2.0 (25.02 release)
        shape = cast(torch.Tensor, node.all_input_nodes[0].meta["val"]).shape
        kernel = cast(tuple[int, int], node.args[1])
        stride = cast(tuple[int, int], node.args[2])

        if not kernel_check(kernel):
            self.reporter.report_reject(
                node,
                f"Maxpool2d needs kernel_y < 256, kernel_x*kernel_y<=65536, got {kernel}",
            )
            return False
        if not dim_check(shape):
            self.reporter.report_reject(
                node,
                f"Maxpool2d needs N == 1, rest dims <= 65536, got shape {list(shape)}",
            )
            return False
        if not stride_check(stride):
            self.reporter.report_reject(
                node, f"Maxpool2d needs stride <= 3, got {stride}"
            )
            return False
        return True
