# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

from typing import Set, Union

import torch

from executorch.exir.dialects.edge._ops import EdgeOpOverload
from executorch.exir.pass_base import ExportPass, PassResult
from executorch.exir.program._program import _get_updated_graph_signature

from torch.export.exported_program import ExportedProgram

OpType = Union[str, torch._ops.OpOverload, EdgeOpOverload]


class RemoveAssertsTransform(ExportPass):
    """
    Remove operators which perform assertions. These are not possible to execute in
    Vulkan since GLSL shaders cannot abort execution at runtime. Therefore, remove these
    operators.
    """

    assert_ops: Set[OpType] = {
        torch.ops.aten._assert_scalar.default,
        torch.ops.aten.sym_constrain_range_for_size.default,
    }

    def call(self, graph_module: torch.fx.GraphModule) -> PassResult:
        for node in graph_module.graph.nodes:
            if node.target in self.assert_ops:
                graph_module.graph.erase_node(node)

        graph_module.graph.eliminate_dead_code()
        graph_module.recompile()
        return PassResult(graph_module, True)


def remove_asserts(edge_program: ExportedProgram) -> ExportedProgram:
    graph_module = edge_program.graph_module
    RemoveAssertsTransform()(graph_module)

    edge_program._graph_signature = _get_updated_graph_signature(
        edge_program.graph_signature, graph_module
    )
    edge_program._validate()
    return edge_program
