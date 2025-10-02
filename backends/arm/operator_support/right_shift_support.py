# Copyright 2024-2025 Arm Limited and/or its affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.
"""Declare operator support for bitwise right-shift in TOSA.

Provide support checks for ``aten.bitwise_right_shift`` and ``__rshift__``
targets across integer and float TOSA profiles.

"""

# pyre-unsafe


import logging

import torch.fx as fx
from executorch.backends.arm.operator_support.tosa_supported_operators import (
    register_tosa_support_check,
    SupportedTOSAOperatorCheck,
)
from executorch.backends.arm.tosa import TosaSpecification
from executorch.exir.dialects._ops import ops as exir_ops

logger = logging.getLogger(__name__)


@register_tosa_support_check
class RightShiftSupported(SupportedTOSAOperatorCheck):
    """Provide TOSA support check for right-shift operations."""

    targets = [
        exir_ops.edge.aten.bitwise_right_shift.Tensor,
        exir_ops.edge.aten.__rshift__.Scalar,
    ]

    tosa_specs = [
        TosaSpecification.create_from_string("TOSA-1.0+INT"),
        TosaSpecification.create_from_string("TOSA-1.0+FP"),
    ]

    def is_node_tosa_supported(
        self, node: fx.Node, tosa_spec: TosaSpecification
    ) -> bool:
        """Return True if the node is supported by TOSA.

        Emit a warning on U55 subsets where one-off errors may occur. Otherwise
        accept all matching targets.

        """
        # TODO MLETORCH-525 Remove warning
        if tosa_spec.is_U55_subset:
            logging.warning(f"{node.target} may introduce one-off errors.")
        return True
