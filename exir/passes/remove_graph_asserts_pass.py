# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

import torch

from torch.fx.passes.infra.pass_base import PassBase, PassResult


def _erase_asserts_from_modules(
    graph_module: torch.fx.GraphModule,
    targets: tuple,
) -> bool:
    modified = False
    for module in graph_module.modules():
        if not isinstance(module, torch.fx.GraphModule):
            continue
        module_modified = False
        for node in module.graph.nodes:
            if node.op == "call_function" and node.target in targets:
                module.graph.erase_node(node)
                module_modified = True
        if module_modified:
            module.recompile()
            module.graph.eliminate_dead_code()
            modified = True
    return modified


_CORE_ASSERT_TARGETS: tuple = (
    torch.ops.aten._assert_async.msg,
    torch.ops.aten._assert_scalar.default,
    torch.ops.aten.sym_constrain_range_for_size.default,
    torch.ops.aten.sym_constrain_range.default,
    torch.ops.aten._assert_tensor_metadata.default,
)


class RemoveGraphAssertsPass(PassBase):
    """
    Temporary pass to remove all the assert ops until runtime decides to address it.
    """

    def call(self, graph_module: torch.fx.GraphModule) -> PassResult:
        return PassResult(
            graph_module,
            _erase_asserts_from_modules(graph_module, _CORE_ASSERT_TARGETS),
        )


class RemoveNonCoreAtenOpGraphAssertsPass(PassBase):
    """
    Remove assert ops from the graph that're not Aten Canonical.
    """

    def call(self, graph_module: torch.fx.GraphModule) -> PassResult:
        return PassResult(
            graph_module,
            _erase_asserts_from_modules(
                graph_module,
                (torch.ops.aten._assert_tensor_metadata.default,),
            ),
        )
