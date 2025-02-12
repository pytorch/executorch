# Copyright 2025 Arm Limited and/or its affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from typing import cast

import torch.fx as fx
from executorch.backends.arm.operator_support.tosa_supported_operators import (
    register_tosa_support_check,
    SupportedTOSAOperatorCheck,
)
from executorch.backends.arm.tosa_specification import Tosa_0_80, TosaSpecification
from executorch.exir.dialects._ops import ops as exir_ops


@register_tosa_support_check
class SumSupported(SupportedTOSAOperatorCheck):
    targets = [exir_ops.edge.aten.sum.dim_IntList]

    tosa_specs = [
        TosaSpecification.create_from_string("TOSA-0.80+BI"),
        TosaSpecification.create_from_string("TOSA-0.80+MI"),
    ]

    def is_node_tosa_supported(self, node: fx.Node, tosa_spec: TosaSpecification):
        if not (isinstance(tosa_spec, Tosa_0_80) and tosa_spec.is_U55_subset):
            return True

        # U55 case, Vela 4.2.0 (25.02 release)
        input_shape = node.all_input_nodes[0].meta["val"].shape
        dim_list = cast(list[int], node.args[1])
        dim_list = [dim % len(input_shape) for dim in dim_list]

        for dim in dim_list:
            if not 1 <= input_shape[dim] <= 65536:
                return False

            # We can't be certain of which dim is the last in memory yet,
            # Always go for stricter condition.
            pre_R_product = 1.0
            for length in input_shape[:dim]:
                pre_R_product *= length
            post_R_product = 1.0
            for length in input_shape[dim + 1 :]:
                post_R_product *= length
            if not 1 <= pre_R_product <= 65536:
                return False
            if not 1 <= post_R_product <= 65536:
                return False
        return True
