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
from executorch.exir.dialects._ops import ops as exir_ops


@register_tosa_support_check
class UpsampleNearest2dSupported(SupportedTOSAOperatorCheck):
    """Provide the explicit TOSA support gate for nearest upsample."""

    targets = [exir_ops.edge.aten.upsample_nearest2d.vec]

    def is_node_tosa_supported(
        self, _node: fx.Node, _tosa_spec: TosaSpecification
    ) -> bool:  # type: ignore[override, misc]
        return True


@register_tosa_support_check
class UpsampleBilinear2dSupported(SupportedTOSAOperatorCheck):
    """Reject bilinear upsample cases that cannot lower to a valid TOSA
    RESIZE.
    """

    targets = [exir_ops.edge.aten.upsample_bilinear2d.vec]

    def is_node_tosa_supported(
        self, node: fx.Node, _tosa_spec: TosaSpecification
    ) -> bool:  # type: ignore[override, misc]
        input_node = ensure_type(fx.Node, node.args[0])
        align_corners = ensure_type(bool, node.args[2])
        input_size_yx = get_first_fake_tensor(input_node).shape[2:]
        output_size_yx = get_first_fake_tensor(node).shape[2:]

        try:
            scale_y_n, scale_y_d, _, _ = RewriteUpsamplePass.get_resize_parameters_1d(
                input_size_yx[0], output_size_yx[0], align_corners
            )
            scale_x_n, scale_x_d, _, _ = RewriteUpsamplePass.get_resize_parameters_1d(
                input_size_yx[1], output_size_yx[1], align_corners
            )
        except RuntimeError as err:
            self.reporter.report_reject(node, str(err))
            return False

        # get_resize_parameters_1d() returns the TOSA RESIZE scale fraction for
        # each spatial dimension. For align_corners=False, this is the effective
        # output_size / input_size ratio, so the 1/16 boundary is checked
        # directly in the same representation that RESIZE lowering will use.
        if scale_y_d >= 16 * scale_y_n or scale_x_d >= 16 * scale_x_n:
            self.reporter.report_reject(
                node,
                "Bilinear RESIZE downscale must be strictly greater than 1/16",
            )
            return False

        return True
