# Copyright 2024 Arm Limited and/or its affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-unsafe

from typing import cast

import torch.fx as fx

from executorch.backends.arm.operator_support.tosa_supported_operators import (
    register_tosa_support_check,
    SupportedTOSAOperatorCheck,
)
from executorch.backends.arm.tosa_specification import TosaSpecification
from executorch.exir.dialects._ops import ops as exir_ops


@register_tosa_support_check
class VarCorrectionSupported(SupportedTOSAOperatorCheck):
    targets = [exir_ops.edge.aten.var.correction]

    tosa_specs = [
        TosaSpecification.create_from_string("TOSA-0.80.0+BI"),
        TosaSpecification.create_from_string("TOSA-0.80.0+MI"),
    ]

    def is_node_supported(self, node: fx.Node, tosa_spec: TosaSpecification) -> bool:
        assert node.target in self.targets

        keep_dim = node.kwargs.get("keepdim", False)
        return cast(bool, keep_dim)
