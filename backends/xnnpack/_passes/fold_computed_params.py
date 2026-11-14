# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
# Copyright 2026 Arm Limited and/or its affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import torch
from executorch.backends.xnnpack._passes.constant_folding import fold_constants
from executorch.backends.xnnpack._passes.xnnpack_pass import XNNPACKPass
from executorch.backends.xnnpack.utils.utils import (
    COMPUTED_PARAM_TAG,
    get_param_tensor,
    is_param_node,
)
from executorch.exir.pass_base import PassResult
from torch.export.graph_signature import InputKind


class FoldComputedParamsPass(XNNPACKPass):
    """
    Folds the nodes that the partitioner tagged with COMPUTED_PARAM_TAG, which
    compute a GEMM weight or bias from params, into a param holding the result.
    """

    def call(self, graph_module: torch.fx.GraphModule) -> PassResult:
        graph = graph_module.graph
        names = {node.name for node in graph.nodes}
        values = {
            node: get_param_tensor(self.exported_program, node)
            for node in graph.nodes
            if is_param_node(self.exported_program, node)
        }
        selected = set()
        for node in graph.nodes:
            # A GEMM rejected after its check, possibly by an earlier partitioner,
            # leaves tags behind, which delegate outputs and inputs inherit. So
            # fold a tagged node only next to its GEMM, and only from params.
            if (
                node.op != "call_function"
                or not names & node.meta.get(COMPUTED_PARAM_TAG, set())
                or not all(n in values or n in selected for n in node.all_input_nodes)
            ):
                continue
            selected.add(node)

        changed = fold_constants(
            self.exported_program, selected, values, InputKind.PARAMETER
        )
        return PassResult(graph_module, changed)
