# Copyright 2025 Arm Limited and/or its affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from copy import copy
from typing import cast

from executorch.backends.arm._passes.arm_pass_utils import create_node
from executorch.backends.arm._passes.quant_args import QuantArgs
from executorch.backends.arm.constants import DQ_OPS, Q_OPS
from executorch.exir.dialects._ops import ops as exir_ops
from executorch.exir.pass_base import ExportPass, PassResult
from torch.fx import GraphModule, Node


class InsertRescalePass(ExportPass):
    """Finds patterns of dq -> q, and replaces them
    with backend dialect tosa::RESCALE op.

    Does not guarantee that the dtypes and zero points are valid
    in TOSA, that is the job of the quantization annotator that
    produced the dq and q nodes. The TOSA constraints are validated
    in the fake implementation of.
    """

    def fold_dq_q_to_rescale(self, node: Node, user: Node, graph_module: GraphModule):
        dq_args = QuantArgs.from_operator(node.target, node.args)
        q_args = QuantArgs.from_operator(user.target, user.args)
        new_scale = dq_args.scale / q_args.scale

        with graph_module.graph.inserting_before(node):
            rescale_node = create_node(
                graph_module.graph,
                exir_ops.backend.tosa.RESCALE.default,
                (
                    node.all_input_nodes[0],
                    q_args.dtype,
                    new_scale,
                    dq_args.zp,
                    q_args.zp,
                ),
            )
            rescale_node.meta = copy(user.meta)
            user.replace_all_uses_with(rescale_node)
            graph_module.graph.erase_node(user)

    def call(self, graph_module: GraphModule) -> PassResult:
        modified = False
        for node in graph_module.graph.nodes:
            node = cast(Node, node)

            if node.target not in DQ_OPS:
                continue
            # Copy users since we remove them while iterating, modyfing the node.users list.
            for user in copy(node.users):
                if user.target in Q_OPS:
                    self.fold_dq_q_to_rescale(node, user, graph_module)
                    modified = True
            if len(node.users) == 0:
                graph_module.graph.erase_node(node)

        graph_module = super().call(graph_module).graph_module
        graph_module.recompile()
        return PassResult(graph_module, modified)
