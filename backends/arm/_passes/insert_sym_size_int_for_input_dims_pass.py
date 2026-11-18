# Copyright 2026 Arm Limited and/or its affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from typing import Set, Type

import torch
from executorch.backends.arm._passes import ArmPass
from executorch.backends.arm._passes.symbolic_to_tosa_shape_pass import (
    SymbolicToTosaShapesPass,
)
from executorch.exir.pass_base import ExportPass, PassResult


class InsertSymSizeIntForInputDimsPass(ArmPass):
    _passes_required_after: Set[Type[ExportPass]] = {SymbolicToTosaShapesPass}

    def _get_tensor_meta(self, node: torch.fx.Node) -> torch.Tensor | None:
        for meta_key in ("val", "example_value"):
            meta_value = node.meta.get(meta_key)
            if isinstance(meta_value, torch.Tensor):
                return meta_value
        return None

    def _has_sym_size_int(
        self, graph: torch.fx.Graph, placeholder: torch.fx.Node, dim: int
    ) -> bool:
        for node in graph.nodes:
            if node.op not in ("placeholder", "call_function"):
                break
            if (
                node.op == "call_function"
                and node.target != torch.ops.aten.sym_size.int
            ):
                break
            if node.op != "call_function":
                continue
            if node.args == (placeholder, dim):
                return True
        return False

    def call(self, graph_module: torch.fx.GraphModule) -> PassResult:
        graph = graph_module.graph
        modified = False
        use_example_value = any("example_value" in node.meta for node in graph.nodes)

        for node in list(graph.nodes):
            if node.op != "placeholder":
                continue

            tensor = self._get_tensor_meta(node)
            if tensor is None:
                continue

            if use_example_value and "example_value" not in node.meta:
                node.meta["example_value"] = tensor

            for dim, dim_value in enumerate(tensor.shape):
                if not isinstance(dim_value, torch.SymInt):
                    continue
                if self._has_sym_size_int(graph, node, dim):
                    continue

                with graph.inserting_after(node):
                    sym_size_node = graph.call_function(
                        torch.ops.aten.sym_size.int, (node, dim)
                    )
                sym_size_node.meta["val"] = dim_value
                if use_example_value:
                    sym_size_node.meta["example_value"] = dim_value
                modified = True

        if modified:
            graph_module.recompile()

        return PassResult(graph_module, modified)
