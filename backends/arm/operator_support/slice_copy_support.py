# Copyright 2025 Arm Limited and/or its affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.
"""Declare operator support for ``aten.slice_copy`` in TOSA.

Support slicing with unit step only; emit a warning and reject otherwise.

"""

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
class SliceCopySupported(SupportedTOSAOperatorCheck):
    """Provide TOSA support check for ``aten.slice_copy``."""

    targets = [exir_ops.edge.aten.slice_copy.Tensor]

    tosa_specs = [
        TosaSpecification.create_from_string("TOSA-1.0+INT"),
        TosaSpecification.create_from_string("TOSA-1.0+FP"),
    ]

    def is_node_tosa_supported(
        self, node: fx.Node, tosa_spec: TosaSpecification
    ) -> bool:  # type: ignore[override, misc]
        """Return True if the node is supported by TOSA.

        Accept slice_copy when the step is 1 (or unspecified). Warn and reject
        non-unit step sizes.

        """
        if tosa_spec not in self.tosa_specs:
            return False

        args = node.args
        if len(args) == 5 and (step := args[4]) != 1:
            logger.warning(f"{node.target} with step size of {step} not supported.")
            return False
        return True
