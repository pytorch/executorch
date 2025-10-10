# Copyright 2025 Arm Limited and/or its affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.
"""Provide TOSA support checks for 2D pooling.

Validate ``avg_pool2d`` and ``max_pool2d_with_indices`` against U55 profile
constraints including kernel size, stride, padding, and dimensionality.

"""

from typing import cast

import torch
import torch.fx as fx
from executorch.backends.arm._passes.arm_pass_utils import get_first_fake_tensor
from executorch.backends.arm.operator_support.tosa_supported_operators import (
    register_tosa_support_check,
    SupportedTOSAOperatorCheck,
)
from executorch.backends.arm.operators.operator_validation_utils import (
    adjust_pooling_pad_if_needed,
)
from executorch.backends.arm.tosa import TosaSpecification
from executorch.exir.dialects._ops import ops as exir_ops


def kernel_check(kernel: tuple[int, int]) -> bool:
    """Check if kernel size is within U55 constraints.

    Checks that ``kernel_x * kernel_y`` is in ``[1, 65536]`` and
    ``kernel_y`` is in ``[1, 256]`` as required by the U55 profile.

    Args:
        kernel (tuple[int, int]): Kernel height and width ``(kh, kw)``.

    Returns:
        bool: True if the kernel passes validation.

    """
    if not (1 <= kernel[0] * kernel[1] <= 65536):
        return False
    return 1 <= kernel[1] <= 256


def stride_check(strides: tuple[int, int]) -> bool:
    """Check if strides are within U55 constraints.

    Args:
        strides (tuple[int, int]): Vertical and horizontal strides.

    Returns:
        bool: True if each stride is in ``[1, 3]``.

    """
    return all(1 <= stride <= 3 for stride in strides)


def dim_check(shape=torch.Size) -> bool:
    """Check if non-batch dims are within U55 constraints.

    Verifies that all dimensions except batch are in ``[1, 65536]``.

    Args:
        shape (torch.Size): Input tensor shape.

    Returns:
        bool: True if all checked dimensions pass.

    """
    check = True
    for dim in shape[1:]:
        check &= 1 <= dim <= 65536
    return check


@register_tosa_support_check
class AvgPool2dSupported(SupportedTOSAOperatorCheck):
    """Provide TOSA support checks for ``aten.avg_pool2d``.

    Applies additional constraints when targeting the U55 subset, including
    limits on kernel size, stride, padding behavior, and tensor ranks.

    """

    targets = [
        exir_ops.edge.aten.avg_pool2d.default,
    ]

    tosa_specs = [
        TosaSpecification.create_from_string("TOSA-1.0+INT"),
        TosaSpecification.create_from_string("TOSA-1.0+FP"),
    ]

    def is_node_tosa_supported(self, node: fx.Node, tosa_spec: TosaSpecification):
        """Return True if ``avg_pool2d`` satisfies U55 constraints.

        Computes the effective TOSA padding (depending on ``count_include_pad``
        and ``divisor_override``) and validates kernel, stride, and shape limits.

        """
        if not tosa_spec.is_U55_subset:
            return True

        # U55 case, Vela 4.2.0 (25.02 release)
        input_arg = node.args[0]
        if isinstance(input_arg, torch.fx.Node):
            input_arg = get_first_fake_tensor(input_arg)
        shape = input_arg.data.shape  # type: ignore[union-attr]

        # Calculate padding used in the final TOSA operator
        kernel = cast(tuple[int, int], node.args[1])
        stride = cast(tuple[int, int], node.args[2])
        padding = cast(tuple[int, int], node.args[3]) if len(node.args) > 3 else (0, 0)
        ceil_mode = cast(bool, node.args[4]) if len(node.args) > 4 else False
        count_include_pad = cast(bool, node.args[5]) if len(node.args) > 5 else True
        divisor_override = cast(int, node.args[6]) if len(node.args) > 6 else None

        # If count_include_pad is True or divior_override is given, padding is applied
        # by concating zero-elements rather than setting it in the avg_pool op.
        if count_include_pad or divisor_override is not None:
            tosa_padding = (0, 0, 0, 0)
        # Otherwise, calculate the padding as done in the node visitor
        else:
            post_pad_h = adjust_pooling_pad_if_needed(
                shape[2], kernel[0], stride[0], padding[0], ceil_mode
            )
            post_pad_w = adjust_pooling_pad_if_needed(
                shape[3], kernel[1], stride[1], padding[1], ceil_mode
            )
            tosa_padding = (padding[0], post_pad_h, padding[1], post_pad_w)

        if not all(1 <= k <= 8 for k in kernel) and not all(
            v == 0 for v in tosa_padding
        ):
            self.reporter.report_reject(
                node, f"Avgpool2d with padding needs kernel dims < 8, got {kernel}"
            )
            return False

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
    """Provide TOSA support checks for ``aten.max_pool2d_with_indices``.

    Applies additional constraints when targeting the U55 subset, including
    limits on kernel size, stride, and tensor ranks.

    """

    targets = [
        exir_ops.edge.aten.max_pool2d_with_indices.default,
    ]

    tosa_specs = [
        TosaSpecification.create_from_string("TOSA-1.0+INT"),
        TosaSpecification.create_from_string("TOSA-1.0+FP"),
    ]

    def is_node_tosa_supported(self, node: fx.Node, tosa_spec: TosaSpecification):
        """Return True if ``max_pool2d_with_indices`` satisfies U55
        constraints.
        """
        if not tosa_spec.is_U55_subset:
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
