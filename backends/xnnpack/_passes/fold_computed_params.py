# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import torch
from executorch.backends.transforms.utils import create_constant_placeholder
from executorch.backends.xnnpack._passes.xnnpack_pass import XNNPACKPass
from executorch.backends.xnnpack.utils.utils import COMPUTED_PARAM_TAG, get_param_tensor
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
        values = {}
        for node in graph.nodes:
            # Checking a GEMM tags the nodes computing its weight or bias before
            # the GEMM is accepted, so they are folded only next to that GEMM.
            if not names & node.meta.get(COMPUTED_PARAM_TAG, set()):
                continue
            args, kwargs = torch.fx.node.map_arg(
                (node.args, node.kwargs),
                lambda n: (
                    values[n]
                    if n in values
                    else get_param_tensor(self.exported_program, n)
                ),
            )
            with torch.no_grad():
                values[node] = node.target(*args, **kwargs)

        if not values:
            return PassResult(graph_module, False)

        taken = (
            names
            | self.exported_program.state_dict.keys()
            | self.exported_program.constants.keys()
        )
        for node, value in values.items():
            if all(user in values for user in node.users):
                continue
            name = f"{node.name}_folded"
            while name in taken:
                name += "_"
            taken.add(name)
            with graph.inserting_before(next(iter(graph.nodes))):
                param = create_constant_placeholder(
                    exp_program=self.exported_program,
                    graph=graph,
                    kind=InputKind.PARAMETER,
                    name=name,
                    data=value,
                )
            node.replace_all_uses_with(param)

        for node in reversed(values):
            graph.erase_node(node)

        graph_module.recompile()
        return PassResult(graph_module, True)
