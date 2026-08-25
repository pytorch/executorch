# Copyright 2026 Arm Limited and/or its affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-unsafe

"""TOSA configuration of the shared permute/view propagation passes."""

from collections.abc import Sequence

import torch
from executorch.backends.arm.tosa.mapping import TosaSpecialDtype
from executorch.backends.arm.tosa.specification import get_context_spec
from executorch.backends.transforms.propagate_view_copy_permute_pass import (
    PropagateViewCopyPermuteDownPass as _DownPass,
    PropagateViewCopyPermuteUpPass as _UpPass,
)
from executorch.exir.dialects._ops import ops as exir_ops
from executorch.exir.pass_base import ExportPass

from .arm_pass import ArmPass
from .fuse_duplicate_users_pass import TOSA_EXCLUDED_TARGETS
from .remove_permutes_around_elementwise_tosa_ops import (
    RemovePermutesAroundElementwiseTosaOps,
)


class TosaPropagationOverrides:
    """TOSA-specific answers to the propagation pass's extension points.

    Kept apart from the pass itself so the algorithm can be shared with backends
    that have no TOSA dialect.

    """

    def duplicate_user_fusion_exclusions(self) -> frozenset:
        return TOSA_EXCLUDED_TARGETS

    def make_fusion_pass(self) -> ExportPass | None:
        if self.exported_program is None:
            return None
        return RemovePermutesAroundElementwiseTosaOps(self.exported_program)

    def should_propagate(self) -> bool:
        # Do not run for Ethos-U85 since this exposes a numerical issue.
        # There is no target meta-data at this stage so use INT+cf as proxy.
        # To be removed after MLBEDSW-11805.
        return not self._is_u85_like_tosa_int_cf()

    def is_transparent(self, node: torch.fx.Node) -> bool:
        return (
            super().is_transparent(node)
            or node.target == exir_ops.backend.tosa.RESCALE.default
        )

    def is_multi_input_elementwise(self, node: torch.fx.Node) -> bool:
        return node.target == exir_ops.backend.tosa.TABLE.default

    def blocks_crossing(self, node: torch.fx.Node) -> bool:
        return node.target == exir_ops.backend.tosa.SCATTER.default

    def is_elementwise(self, node: torch.fx.Node) -> bool:
        if node.target == exir_ops.backend.tosa.RESCALE.default:
            return self._is_per_tensor_rescale(node)
        if node.target == exir_ops.backend.tosa.TABLE.default:
            return True
        return super().is_elementwise(node)

    def _is_u85_like_tosa_int_cf(self) -> bool:
        if self.compile_spec is not None:
            tosa_spec = self.compile_spec.tosa_spec
        else:
            try:
                tosa_spec = get_context_spec()
            except RuntimeError:
                return False

        return (
            tosa_spec.support_integer()
            and not tosa_spec.support_float()
            and tosa_spec.support_extension("cf")
        )

    def _is_per_tensor_rescale(self, node: torch.fx.Node) -> bool:
        if len(node.args) < 3:
            return False
        input_nodes = node.all_input_nodes
        if len(input_nodes) != 1:
            return False
        special_dtype_key = TosaSpecialDtype.meta_key()
        if input_nodes[0].meta.get(special_dtype_key) != node.meta.get(
            special_dtype_key
        ):
            return False
        scales = node.args[2]
        return not isinstance(scales, Sequence) or len(scales) == 1




class PropagateViewCopyPermuteUpPass(TosaPropagationOverrides, _UpPass, ArmPass):
    pass


class PropagateViewCopyPermuteDownPass(TosaPropagationOverrides, _DownPass, ArmPass):
    pass
