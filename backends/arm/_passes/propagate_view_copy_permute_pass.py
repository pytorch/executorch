# Copyright 2026 Arm Limited and/or its affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-unsafe
"""TOSA configuration of the shared permute/view propagation passes."""

from collections.abc import Sequence
from typing import Any

import torch
from executorch.backends.arm._passes.fold_qdq_with_annotated_qparams_pass import (
    get_input_qparams,
    get_output_qparams,
)
from executorch.backends.arm.tosa.mapping import TosaSpecialDtype
from executorch.backends.transforms.propagate_view_copy_permute_pass import (
    PropagateViewCopyPermuteDownPass as _DownPass,
    PropagateViewCopyPermutePass as _BasePass,
    PropagateViewCopyPermuteUpPass as _UpPass,
)
from executorch.exir.dialects._ops import ops as exir_ops
from executorch.exir.pass_base import ExportPass

from .arm_pass import ArmPass
from .fuse_duplicate_users_pass import TOSA_EXCLUDED_TARGETS
from .remove_permutes_around_elementwise_tosa_ops import (
    RemovePermutesAroundElementwiseTosaOps,
)


class TosaPropagationOverrides(_BasePass):
    """TOSA-specific answers to the propagation pass's extension points.

    Kept apart from the pass itself so the algorithm can be shared with backends
    that have no TOSA dialect.

    """

    def duplicate_user_fusion_exclusions(self) -> frozenset:
        return TOSA_EXCLUDED_TARGETS

    def duplicate_user_fusion_key(self, node: torch.fx.Node) -> Any:
        return (
            get_input_qparams(node) if node.meta.get("input_qparams") else {},
            get_output_qparams(node) if node.meta.get("output_qparams") else {},
        )

    _REDUCTION_TARGETS = {
        exir_ops.edge.aten.mean.dim,
    }
    _ARG_UPDATE_TARGETS = {
        *_REDUCTION_TARGETS,
        exir_ops.edge.aten.slice_copy.Tensor,
    }

    def blocks_moving(
        self,
        moving_node: torch.fx.Node,
        frontier: torch.fx.Node,
        next_nodes: Sequence[torch.fx.Node],
    ) -> bool:
        # INT48 storage is not laid out like the int32 fake tensor used for
        # metadata, so permuting it would address the packed data incorrectly.
        return moving_node.target in self._permute_targets and any(
            self._node_or_inputs_are_int48(candidate)
            for candidate in (frontier, *next_nodes)
        )

    def tolerates_shape_after_move(self, next_node: torch.fx.Node, shape: Any) -> bool:
        # Moving a view across a TABLE changes the shape the lookup table
        # operates on. Ethos-U85 miscompiles a TABLE fed by a MAC op when that
        # operand is [N, 1, 1, C] with N > 1 (MLBEDSW-11805), so keep the view
        # on the other side of the table in that case.
        if next_node.target != exir_ops.backend.tosa.TABLE.default:
            return True
        return not self._is_unsafe_table_operand_shape(shape)

    @staticmethod
    def _is_unsafe_table_operand_shape(shape: Any) -> bool:
        """Whether a TABLE reading ``shape`` is miscompiled on Ethos-U85."""
        return (
            shape is not None
            and len(shape) == 4
            and all(isinstance(dim, int) for dim in shape)
            and shape[0] > 1
            and shape[1] == 1
            and shape[2] == 1
        )

    @staticmethod
    def _node_or_inputs_are_int48(node: torch.fx.Node) -> bool:
        special_dtype_key = TosaSpecialDtype.meta_key()
        return node.meta.get(special_dtype_key) == TosaSpecialDtype.INT48 or any(
            input_node.meta.get(special_dtype_key) == TosaSpecialDtype.INT48
            for input_node in node.all_input_nodes
        )

    def make_fusion_pass(self) -> ExportPass | None:
        if self.exported_program is None:
            return None
        return RemovePermutesAroundElementwiseTosaOps(self.exported_program)

    def is_transparent(self, node: torch.fx.Node) -> bool:
        return (
            super().is_transparent(node)
            or node.target == exir_ops.backend.tosa.RESCALE.default
        )

    def is_swappable(self, next_node: torch.fx.Node) -> bool:
        # Arm normalizes reductions to keepdim=True before this pass runs, so a
        # non-keepdim reduction here means the pipeline is misordered. The
        # shared pass merely declines; Arm wants to hear about it.
        if next_node.target in self._REDUCTION_TARGETS:
            keep_dim = (
                next_node.args[2]
                if len(next_node.args) > 2
                else next_node.kwargs.get("keepdim")
            )
            if keep_dim is not True:
                raise RuntimeError(
                    f"{self.__class__.__name__} expects keep_dim=True for "
                    f"reduction ops to simplify propagation logic, got "
                    f"{keep_dim} for node {next_node.name}."
                )
        return super().is_swappable(next_node)

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
