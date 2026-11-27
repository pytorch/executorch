# Copyright 2026 Arm Limited and/or its affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.
"""Provide TOSA support checks for statically decomposable rolls."""

import torch
import torch.fx as fx
from executorch.backends.arm._passes.arm_pass_utils import get_first_fake_tensor
from executorch.backends.arm._passes.decompose_roll_pass import can_decompose_roll
from executorch.backends.arm.common.type import ensure_type
from executorch.backends.arm.constants import MAX_RANK
from executorch.backends.arm.operator_support.tosa_supported_operators import (
    is_quantized,
    register_tosa_support_check,
    SupportedTOSAOperatorCheck,
)
from executorch.backends.arm.tosa import TosaSpecification
from executorch.exir.dialects._ops import ops as exir_ops


def is_decomposable_roll_node(node: fx.Node, tosa_spec: TosaSpecification) -> bool:
    """Return whether backend preprocessing can decompose a roll node."""
    if node.target not in {
        torch.ops.aten.roll.default,
        exir_ops.edge.aten.roll.default,
    }:
        return False
    if (
        tosa_spec.support_integer()
        and not tosa_spec.support_float()
        and not is_quantized(node)
    ):
        return False

    input_node = ensure_type(fx.Node, node.args[0])
    input_tensor = get_first_fake_tensor(input_node)
    if not 0 < len(input_tensor.shape) <= MAX_RANK:
        return False
    if input_tensor.dtype not in {torch.float16, torch.float32} and not (
        input_tensor.dtype == torch.bfloat16 and tosa_spec.support_extension("bf16")
    ):
        return False

    dims = node.args[2] if len(node.args) > 2 else ()
    return can_decompose_roll(input_tensor.shape, node.args[1], dims)


@register_tosa_support_check
class RollSupported(SupportedTOSAOperatorCheck):
    """Provide TOSA support checks for rolls decomposed during preprocessing."""

    targets = [exir_ops.edge.aten.roll.default]

    def is_node_tosa_supported(
        self, node: fx.Node, tosa_spec: TosaSpecification
    ) -> bool:
        """Return True when backend preprocessing can decompose the roll."""
        return is_decomposable_roll_node(node, tosa_spec)
