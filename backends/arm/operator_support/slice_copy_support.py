# Copyright 2025 Arm Limited and/or its affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.


import logging

import torch.fx as fx
from executorch.backends.arm.operator_support.tosa_supported_operators import (
    register_tosa_support_check,
    SupportedTOSAOperatorCheck,
)
from executorch.backends.arm.tosa_specification import TosaSpecification
from executorch.backends.arm.tosa_utils import getNodeArgs
from executorch.exir.dialects._ops import ops as exir_ops

logger = logging.getLogger(__name__)
logger.setLevel(logging.WARNING)


@register_tosa_support_check
class SliceCopySupported(SupportedTOSAOperatorCheck):
    targets = [exir_ops.edge.aten.slice_copy.Tensor]

    tosa_specs = [
        TosaSpecification.create_from_string("TOSA-0.80+BI"),
        TosaSpecification.create_from_string("TOSA-0.80+MI"),
    ]

    def is_node_tosa_supported(self, node: fx.Node, tosa_spec: TosaSpecification) -> bool:  # type: ignore[override, misc]
        if tosa_spec not in self.tosa_specs:
            return False

        inputs = getNodeArgs(node)
        if len(inputs) == 5 and (step := inputs[4].number) != 1:
            logging.warning(f"{node.target} with step size of {step} not supported.")
            return False
        return True
