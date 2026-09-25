# Copyright 2025-2026 NXP
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from executorch.backends.nxp.edge_passes.convert_reshaping_nodes_to_view import (
    ConvertReshapingNodesToViewPass,
)
from executorch.backends.nxp.edge_passes.move_auxiliary_operator_into_separate_qdq_cluster_pass import (
    MoveLeadingAuxiliaryOperatorIntoSeparateQDQClusterPass,
    MoveTrailingAuxiliaryOperatorIntoSeparateQDQClusterPass,
)
from executorch.backends.nxp.edge_passes.remove_as_strided_copy_nodes import (
    RemoveUselessAsStridedCopyNodes,
)
from executorch.exir.pass_base import ExportPass
from executorch.exir.pass_manager import PassManager
from executorch.exir.passes.fold_redundant_qdq_pass import (
    FoldRedundantDequantizeQuantizePass,
)


class NeutronEdgePassManager(PassManager):

    def __init__(self, passes: list[ExportPass] = None):
        passes: list[ExportPass] = passes or [
            MoveLeadingAuxiliaryOperatorIntoSeparateQDQClusterPass(),
            MoveTrailingAuxiliaryOperatorIntoSeparateQDQClusterPass(),
            RemoveUselessAsStridedCopyNodes(),
            ConvertReshapingNodesToViewPass(),
            # Auxiliary-op splitting can introduce DQ -> Q on fanout branches.
            FoldRedundantDequantizeQuantizePass(),
        ]

        super().__init__(
            passes,
            steps=10,  # Empirical value. At most 10 cycles of passes will be run.
        )
