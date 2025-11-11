# Copyright (c) Qualcomm Innovation Center, Inc.
# All rights reserved
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.
import torch
from executorch.exir.dialects._ops import ops as exir_ops
from executorch.exir.pass_base import ExportPass, PassResult
from executorch.exir.passes import dead_code_elimination_pass


class RemoveRedundancy(ExportPass):
    """
    Trim certain operators to reduce unnecessary overhead.
    """

    def __init__(self, quantization_capture=False):
        super(RemoveRedundancy, self).__init__()
        self.redundant_ops_general = {
            torch.clone: self._default_condition,
            torch.ops.aten.clone.default: self._default_condition,
            exir_ops.edge.dim_order_ops._clone_dim_order.default: self._default_condition,
            torch.ops.aten.alias.default: self._default_condition,
            exir_ops.edge.aten.alias.default: self._default_condition,
            exir_ops.edge.aten.alias_copy.default: self._default_condition,
            exir_ops.edge.aten.lift_fresh_copy.default: self._default_condition,
            # remove this target if '_skip_dim_order' is set to False
            exir_ops.edge.dim_order_ops._to_dim_order_copy.default: self._dim_order_op_condition,
            # remove channel_last / contiguous _to_copy if '_skip_dim_order' is set to True
            exir_ops.edge.aten._to_copy.default: self._to_copy_op_condition,
            torch.ops.aten._assert_tensor_metadata.default: self._default_condition,
            torch.ops.aten._assert_scalar.default: self._default_condition,
        }
        self.redundant_ops_annotation = {
            torch.ops.aten._assert_tensor_metadata.default: self._default_condition,
        }
        self.redundant_ops = (
            self.redundant_ops_annotation
            if quantization_capture
            else self.redundant_ops_general
        )

    def _dim_order_op_condition(self, node):
        return node.meta["val"].dtype == node.args[0].meta["val"].dtype

    def _to_copy_op_condition(self, node):
        return "memory_format" in node.kwargs

    def _default_condition(self, ndoe):
        return True

    def _remove(self, graph_module: torch.fx.GraphModule) -> torch.fx.GraphModule:
        for n in graph_module.graph.nodes:
            if n.target in self.redundant_ops and self.redundant_ops[n.target](n):
                to_be_remove = n
                # assert_tensor_metadata op has no user
                if len(n.users.keys()) == 0:
                    n.args = ()
                # normal case
                for user_n in list(n.users.keys()):
                    user_n.replace_input_with(n, n.args[0])
                graph_module.graph.erase_node(to_be_remove)

    def call(self, graph_module: torch.fx.GraphModule):
        self._remove(graph_module)
        graph_module.recompile()
        dead_code_elimination_pass(graph_module)
        return PassResult(graph_module, True)
