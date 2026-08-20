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
from executorch.backends.arm.tosa.specification import TosaSpecification


@register_tosa_support_check
class SymSizeIntSupport(SupportedTOSAOperatorCheck):
    """Allow ``aten.sym_size.int`` when the TOSA shape extension is enabled."""

    targets = [torch.ops.aten.sym_size.int]
    tosa_specs = TosaSpecification.all_profiles_for_version("1.1")

    def is_node_tosa_supported(
        self, node: fx.Node, tosa_spec: TosaSpecification
    ) -> bool:
        return tosa_spec.support_extension("shape")
