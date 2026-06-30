# Copyright 2026 Arm Limited and/or its affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.
"""Provide TOSA support checks for upsample operators."""

import torch.fx as fx
from executorch.backends.arm._passes.arm_pass_utils import get_first_fake_tensor
from executorch.backends.arm._passes.rewrite_upsample import RewriteUpsamplePass
from executorch.backends.arm.common.type import ensure_type
from executorch.backends.arm.operator_support.tosa_supported_operators import (
    register_tosa_support_check,
    SupportedTOSAOperatorCheck,
)
from executorch.backends.arm.tosa import TosaSpecification
from executorch.backends.arm.tosa.resize_utils import get_tosa_resize_validation_error
from executorch.exir.dialects._ops import ops as exir_ops


def _is_upsample_node_tosa_supported(
    support_check: SupportedTOSAOperatorCheck,
    node: fx.Node,
    tosa_spec: TosaSpecification,
    *,
    align_corners: bool,
) -> bool:
    input_node = ensure_type(fx.Node, node.args[0])
    input_size_yx = get_first_fake_tensor(input_node).shape[2:]
    output_size_yx = get_first_fake_tensor(node).shape[2:]

    try:
        scale_y_n, scale_y_d, offset_y, border_y = (
            RewriteUpsamplePass.get_resize_parameters_1d(
                input_size_yx[0], output_size_yx[0], align_corners
            )
        )
        scale_x_n, scale_x_d, offset_x, border_x = (
            RewriteUpsamplePass.get_resize_parameters_1d(
                input_size_yx[1], output_size_yx[1], align_corners
            )
        )
    except RuntimeError as err:
        support_check.reporter.report_reject(node, str(err))
        return False

    # Validate the exact TOSA RESIZE parameters that RewriteUpsamplePass will
    # emit so support checks and fake-op validation reject the same cases.
    validation_error = get_tosa_resize_validation_error(
        input_hw=input_size_yx,
        output_hw=output_size_yx,
        scale=[scale_y_n, scale_y_d, scale_x_n, scale_x_d],
        offset=[offset_y, offset_x],
        border=[border_y, border_x],
        tosa_spec=tosa_spec,
    )
    if validation_error is not None:
        support_check.reporter.report_reject(node, validation_error)
        return False

    return True


@register_tosa_support_check
class UpsampleNearest2dSupported(SupportedTOSAOperatorCheck):
    """Provide the explicit TOSA support gate for nearest upsample."""

    targets = [exir_ops.edge.aten.upsample_nearest2d.vec]

    def is_node_tosa_supported(
        self, node: fx.Node, tosa_spec: TosaSpecification
    ) -> bool:  # type: ignore[override, misc]
        return _is_upsample_node_tosa_supported(
            self, node, tosa_spec, align_corners=False
        )


@register_tosa_support_check
class UpsampleBilinear2dSupported(SupportedTOSAOperatorCheck):
    """Reject bilinear upsample cases that cannot lower to a valid TOSA
    RESIZE.
    """

    targets = [exir_ops.edge.aten.upsample_bilinear2d.vec]

    def is_node_tosa_supported(
        self, node: fx.Node, tosa_spec: TosaSpecification
    ) -> bool:  # type: ignore[override, misc]
        align_corners = ensure_type(bool, node.args[2])
        return _is_upsample_node_tosa_supported(
            self, node, tosa_spec, align_corners=align_corners
        )
