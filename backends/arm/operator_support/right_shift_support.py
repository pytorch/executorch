# Copyright 2024 Arm Limited and/or its affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.


import logging

import torch.fx as fx
from executorch.backends.arm.operator_support.tosa_supported_operators import (
    register_tosa_support_check,
    SupportedTOSAOperatorCheck,
)
from executorch.backends.arm.tosa_specification import Tosa_0_80, TosaSpecification
from executorch.exir.dialects._ops import ops as exir_ops

logger = logging.getLogger(__name__)
logger.setLevel(logging.WARNING)


@register_tosa_support_check
class RightShiftSupported(SupportedTOSAOperatorCheck):
    targets = [exir_ops.edge.aten.__rshift__.Scalar]

    tosa_specs = [
        TosaSpecification.create_from_string("TOSA-0.80.0+BI"),
        TosaSpecification.create_from_string("TOSA-0.80.0+MI"),
    ]

    def is_node_supported(self, node: fx.Node, tosa_spec: TosaSpecification):

        # TODO MLETORCH-525 Remove warning
        if isinstance(tosa_spec, Tosa_0_80) and tosa_spec.is_U55_subset:
            logging.warning(f"{node.target} may introduce one-off errors.")
        return True
