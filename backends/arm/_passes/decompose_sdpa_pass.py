# Copyright 2025-2026 Arm Limited and/or its affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from typing import Set, Type

import torch
from executorch.backends.arm._passes.arm_pass import ArmPass
from executorch.backends.transforms import decompose_sdpa
from executorch.exir.pass_base import ExportPass


class DecomposeScaledDotProductAttentionPass(
    ArmPass, decompose_sdpa.DecomposeScaledDotProductAttention
):
    """Pass that expands `aten.scaled_dot_product_attention` into primitive
    ops.
    """

    _passes_required_after: Set[Type[ExportPass]] = set()

    def _should_decompose(self, node: torch.fx.Node) -> bool:
        return self.allowed_to_transform(node.meta) and super()._should_decompose(node)
