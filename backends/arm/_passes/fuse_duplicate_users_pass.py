# Copyright 2026 Arm Limited and/or its affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from typing import Any

from executorch.backends.arm._passes.arm_pass import ArmPass
from executorch.backends.arm._passes.fold_qdq_with_annotated_qparams_pass import (
    get_input_qparams,
    get_output_qparams,
)
from executorch.backends.transforms.fuse_duplicate_users_pass import (
    FuseDuplicateUsersPass as _FuseDuplicateUsersPass,
)
from executorch.exir.dialects._ops import ops as exir_ops
from torch.fx import Node

# A fused RESCALE feeds its single output tensor to every original consumer.
# Later passes and the Vela compiler assume each integer RESCALE feeds one
# consumer, so collapsing duplicate RESCALE users onto a shared node corrupts
# integer outputs (observed as all-zero results on Ethos-U). Exclude RESCALE
# from fusion; the op-count wins this pass targets are on FP graphs, which
# carry no rescales.
TOSA_EXCLUDED_TARGETS = frozenset({exir_ops.backend.tosa.RESCALE.default})


def quantization_metadata_key(node: Node) -> tuple[Any, Any]:
    return (
        get_input_qparams(node) if node.meta.get("input_qparams") else {},
        get_output_qparams(node) if node.meta.get("output_qparams") else {},
    )


class FuseDuplicateUsersPass(_FuseDuplicateUsersPass, ArmPass):
    """TOSA-aware configuration of the shared duplicate-user fusion."""

    _recompile_before_retrace = False

    def __init__(
        self,
        excluded_targets: frozenset | None = None,
        may_alias_outputs: bool = False,
    ) -> None:
        super().__init__(
            excluded_targets or TOSA_EXCLUDED_TARGETS,
            may_alias_outputs=may_alias_outputs,
            semantic_key=quantization_metadata_key,
        )
