# Copyright (c) 2025 Samsung Electronics Co. LTD
# All rights reserved
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import executorch.exir.passes.constant_prop_pass as constant_prop_module
import torch
from executorch.exir import ExportedProgram
from executorch.exir.pass_base import ExportPass, PassResult
from executorch.exir.passes.constant_prop_pass import (
    constant_prop_pass,
    get_propagated_const_tensor_dict,
)
from torch.fx import GraphModule


class _constant_prop_context:
    def __init__(self):
        self.backup_skip_targets = constant_prop_module._DEFAULT_SKIP_TARGETS
        self.backup_skip_targets_no_quant = (
            constant_prop_module._DEFAULT_SKIP_TARGETS_NO_QUANT
        )

    def __enter__(self):
        constant_prop_module._DEFAULT_SKIP_TARGETS = set()
        constant_prop_module._DEFAULT_SKIP_TARGETS_NO_QUANT = {}

    def __exit__(self, exc_type, exc_val, exc_tb):
        constant_prop_module._DEFAULT_SKIP_TARGETS = self.backup_skip_targets
        constant_prop_module._DEFAULT_SKIP_TARGETS_NO_QUANT = (
            self.backup_skip_targets_no_quant
        )


def get_nodes_in_const_subgraph(exported_program: torch.export.ExportedProgram):
    with _constant_prop_context():
        remove_memory_format = RemoveMemoryFormat(exported_program.graph_module)
        remove_memory_format.remove_memory_format()
        const_node_to_tensor = get_propagated_const_tensor_dict(exported_program, None)
        remove_memory_format.restore_nodes()
        return const_node_to_tensor.keys()


class RemoveMemoryFormat:
    def __init__(self, graph_module: GraphModule):
        self.replaced_node_dict = {}
        self.graph_module = graph_module

    def remove_memory_format(self) -> None:
        """
        Remove memory_format parameter.
        constant_prop_pass cannot handle some ops because memory_format is not torch.fx.Node.
        """
        for node in list(self.graph_module.graph.nodes):
            if node.op == "call_function":
                if "memory_format" in node.kwargs:
                    self.replaced_node_dict[node] = node.kwargs
                    # Create a new kwargs dict without memory_format
                    new_kwargs = {
                        k: v for k, v in node.kwargs.items() if k != "memory_format"
                    }
                    node.kwargs = new_kwargs

    def restore_nodes(self) -> None:
        for node in list(self.graph_module.graph.nodes):
            if node in self.replaced_node_dict.keys():
                node.kwargs = self.replaced_node_dict[node]


class ConstantPropPass(ExportPass):
    """
    Official constant_prop_pass will not fold Q-DQ
    But we need to fold quantized constant tensor as well as non-quantized one
    """

    def __init__(self, edge_program: ExportedProgram):
        super().__init__()
        self.edge_program = edge_program

    def call(self, graph_module: GraphModule):
        with _constant_prop_context():
            remove_memory_format = RemoveMemoryFormat(self.edge_program.graph_module)
            remove_memory_format.remove_memory_format()
            ep = constant_prop_pass(self.edge_program)
            remove_memory_format.restore_nodes()
        self._set_constant_quantize_attrs()
        return PassResult(ep.graph_module, True)

    def _set_constant_quantize_attrs(self) -> None:
        """Set quantize_attrs for constant input nodes.

        This function iterates through all "call_function" nodes and checks if their
        inputs are constant nodes with name prefix "_prop_tensor_constant". If so,
        it sets the quantize_attrs from the current node's meta["in_quantize_attrs"]
        to the input constant node's meta["quantize_attrs"].
        """
        prefix = "_prop_tensor_constant"
        for node in self.edge_program.graph_module.graph.nodes:
            if node.op != "call_function":
                continue
            in_quantize_attrs = node.meta.get("in_quantize_attrs", [])
            if not in_quantize_attrs:
                continue
            for idx, quantize_attrs in in_quantize_attrs:
                if idx < len(node.args):
                    input_node = node.args[idx]
                    if isinstance(
                        input_node, torch.fx.Node
                    ) and input_node.name.startswith(prefix):
                        if input_node.meta.get("quantize_attrs", []):
                            continue
                        input_node.meta["quantize_attrs"] = quantize_attrs
