# Copyright (c) Samsung Electronics Co. LTD
# All rights reserved
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import operator
from collections import OrderedDict

import torch
from executorch.exir.pass_base import ExportPass, PassResult
from executorch.exir.sym_util import eval_shape
from torch._ops import OpOverload


class ComputeConstAttrs(ExportPass):
    def __init__(self) -> None:
        super().__init__()

    def call(self, graph_module: torch.fx.GraphModule) -> PassResult:
        graph = graph_module.graph
        node_to_value = self.get_propagated_attrs(graph)
        self.replace_attrs(node_to_value, graph)
        graph.eliminate_dead_code()
        graph_module.recompile()
        return PassResult(graph_module, True)

    def _collect_sym_size_int(self, graph: torch.fx.Graph):
        node_to_value: OrderedDict[torch.fx.Node, int] = OrderedDict()

        for node in graph.find_nodes(
            op="call_function", target=torch.ops.aten.sym_size.int
        ):
            tensor_node = node.args[0]
            dim_arg = node.args[1]

            if not isinstance(dim_arg, int):
                continue

            if isinstance(tensor_node, torch.fx.Node):
                if "val" in tensor_node.meta:
                    fake_tensor = tensor_node.meta["val"]
                    if hasattr(fake_tensor, "shape") and dim_arg < len(
                        fake_tensor.shape
                    ):
                        size_value = eval_shape(fake_tensor.shape)[dim_arg]
                        node_to_value[node] = size_value
        return node_to_value

    def get_propagated_attrs(self, graph: torch.fx.Graph):
        """
        Calculate the actual value for torch.ops.aten.sym_size.int nodes and operator nodes.

        Returns an OrderedDict mapping nodes to their computed values.
        """
        node_to_value = self._collect_sym_size_int(graph)
        for node in graph.nodes:
            if node.op != "call_function" or isinstance(node.target, OpOverload):
                continue

            # Check if target is a Python operator function
            if node.target not in (
                operator.add,
                operator.sub,
                operator.mul,
                operator.floordiv,
                operator.mod,
            ):
                continue

            resolved_args = []

            for arg in node.args:
                if isinstance(arg, (int, float, bool)):
                    resolved_args.append(arg)
                elif isinstance(arg, torch.fx.Node):
                    if arg in node_to_value:
                        resolved_args.append(node_to_value[arg])

            if len(resolved_args) != len(node.args):
                continue

            try:
                result = node.target(*resolved_args)
                node_to_value[node] = result
                with graph.inserting_before(node):
                    node.args = tuple(resolved_args)
            except Exception as e:
                print(f"    Failed to calculate {node.target}: {e}")

        return node_to_value

    def replace_attrs(
        self, node_to_value: OrderedDict[torch.fx.Node, int], graph: torch.fx.Graph
    ):
        """
        Replace node arguments with their computed values if they are in node_to_value.
        Then delete all nodes in node_to_value from the graph.
        """

        def replace_in_structure(obj):
            """Recursively replace nodes with their computed values in nested structures."""
            if isinstance(obj, torch.fx.Node) and obj in node_to_value:
                return node_to_value[obj]
            elif isinstance(obj, (list, tuple)):
                replaced = [replace_in_structure(item) for item in obj]
                return type(obj)(replaced)
            else:
                return obj

        # replace arguments in ATen ops with computed values
        for node in graph.nodes:
            if node.op != "call_function" or not isinstance(node.target, OpOverload):
                continue

            args_modified = False
            new_args = tuple(replace_in_structure(arg) for arg in node.args)

            if new_args != node.args:
                args_modified = True

            kwargs_modified = False
            new_kwargs = {}
            for key, value in node.kwargs.items():
                new_value = replace_in_structure(value)
                if new_value != value:
                    kwargs_modified = True
                new_kwargs[key] = new_value

            if args_modified or kwargs_modified:
                with graph.inserting_before(node):
                    node.args = new_args
                    node.kwargs = new_kwargs

        # delete all nodes in node_to_value from the graph
        for node in node_to_value.keys():
            if node in graph.nodes:
                graph.erase_node(node)
