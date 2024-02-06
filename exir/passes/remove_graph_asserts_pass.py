# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

import torch

from torch.fx.passes.infra.pass_base import PassBase, PassResult


class RemoveGraphAssertsPass(PassBase):
    """
    Temporary pass to remove all the assert ops until runtime decides to address it.
    """

    def call(self, graph_module: torch.fx.GraphModule) -> PassResult:
        for module in graph_module.modules():
            if not isinstance(module, torch.fx.GraphModule):
                continue

            for node in module.graph.nodes:
                if node.op == "call_function" and (
                    node.target
                    in (
                        torch.ops.aten._assert_async.msg,
                        torch.ops.aten._assert_scalar.default,
                        torch.ops.aten.sym_constrain_range_for_size.default,
                        torch.ops.aten.sym_constrain_range.default,
                    )
                ):
                    module.graph.erase_node(node)

            module.recompile()
            module.graph.eliminate_dead_code()

        return PassResult(graph_module, True)
