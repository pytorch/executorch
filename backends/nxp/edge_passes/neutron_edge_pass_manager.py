# Copyright 2025-2026 NXP
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from executorch.backends.nxp.edge_passes.move_auxiliary_operator_into_separate_qdq_cluster_pass import (
    MoveLeadingAuxiliaryOperatorIntoSeparateQDQClusterPass,
    MoveTrailingAuxiliaryOperatorIntoSeparateQDQClusterPass,
)
from executorch.backends.nxp.edge_passes.neutron_edge_pass import NeutronEdgePass
from executorch.backends.nxp.edge_passes.remove_as_strided_copy_nodes import (
    RemoveUselessAsStridedCopyNodes,
)
from torch.fx.passes.infra.pass_manager import PassManager


class NeutronEdgePassManager(PassManager):

    def __init__(self, passes: list[NeutronEdgePass] = None):
        passes: list[NeutronEdgePass] = passes or [
            MoveLeadingAuxiliaryOperatorIntoSeparateQDQClusterPass(),
            MoveTrailingAuxiliaryOperatorIntoSeparateQDQClusterPass(),
            RemoveUselessAsStridedCopyNodes(),
        ]

        super().__init__(
            passes,
            steps=10,  # Empirical value. At most 10 cycles of passes will be run.
        )
