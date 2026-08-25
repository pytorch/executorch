# Copyright 2026 Arm Limited and/or its affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from executorch.backends.transforms.fuse_duplicate_users_pass import (
    FuseDuplicateUsersPass as _FuseDuplicateUsersPass,
)
from executorch.exir.dialects._ops import ops as exir_ops

# A fused RESCALE feeds its single output tensor to every original consumer.
# Later passes and the Vela compiler assume each integer RESCALE feeds one
# consumer, so collapsing duplicate RESCALE users onto a shared node corrupts
# integer outputs (observed as all-zero results on Ethos-U). Exclude RESCALE
# from fusion; the op-count wins this pass targets are on FP graphs, which
# carry no rescales.
TOSA_EXCLUDED_TARGETS = frozenset({exir_ops.backend.tosa.RESCALE.default})


class FuseDuplicateUsersPass(_FuseDuplicateUsersPass):
    """TOSA-aware configuration of the shared duplicate-user fusion."""

    def __init__(self, excluded_targets: frozenset | None = None) -> None:
        super().__init__(excluded_targets or TOSA_EXCLUDED_TARGETS)
