# Copyright 2026 Arm Limited and/or its affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import torch
import torch.fx as fx

from executorch.backends.arm.operator_support.tosa_supported_operators import (
    register_tosa_support_check,
    SupportedTOSAOperatorCheck,
)
from executorch.backends.arm.tosa import TosaSpecification
from executorch.exir.dialects._ops import ops as exir_ops


@register_tosa_support_check
class BoolBitwiseSupported(SupportedTOSAOperatorCheck):
    """Allow boolean bitwise ops, which are lowered to logical ops."""

    targets = [
        exir_ops.edge.aten.bitwise_and.Tensor,
        exir_ops.edge.aten.bitwise_and.Scalar,
        exir_ops.edge.aten.bitwise_or.Tensor,
        exir_ops.edge.aten.bitwise_or.Scalar,
        exir_ops.edge.aten.bitwise_xor.Tensor,
        exir_ops.edge.aten.bitwise_xor.Scalar,
        exir_ops.edge.aten.bitwise_not.default,
    ]

    tosa_specs = TosaSpecification.all_versions_and_profiles()

    def is_node_tosa_supported(
        self, node: fx.Node, tosa_spec: TosaSpecification
    ) -> bool:  # type: ignore[override, misc]
        if node.meta["val"].dtype == torch.bool:
            return True

        return False
