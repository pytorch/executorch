# Copyright 2026 Arm Limited and/or its affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import operator

import torch
import torch.fx as fx
from executorch.backends.arm.operator_support.tosa_supported_operators import (
    register_tosa_support_check,
    SupportedTOSAOperatorCheck,
)
from executorch.backends.arm.tosa.specification import TosaSpecification


@register_tosa_support_check
class SymIntArithmeticSupport(SupportedTOSAOperatorCheck):
    """Allow symbolic integer arithmetic with the TOSA shape extension."""

    targets = [
        operator.add,
        operator.sub,
        operator.mul,
        operator.mod,
        operator.floordiv,
    ]
    tosa_specs = TosaSpecification.all_profiles_for_version("1.1")

    def is_node_tosa_supported(
        self, node: fx.Node, tosa_spec: TosaSpecification
    ) -> bool:
        if not tosa_spec.support_extension("shape") or not isinstance(
            node.meta.get("val"), torch.SymInt
        ):
            return False
        return all(
            (
                isinstance(arg.meta.get("val"), torch.SymInt)
                if isinstance(arg, fx.Node)
                else isinstance(arg, (int, torch.SymInt))
            )
            for arg in node.args
        )
