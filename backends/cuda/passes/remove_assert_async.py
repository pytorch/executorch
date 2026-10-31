# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import torch
from torch.fx import GraphModule
from torch.fx.passes.infra.pass_base import PassBase, PassResult


class RemoveAssertAsyncPass(PassBase):
    """
    Removes tensor checks (``aten._assert_async.msg``) from the program the
    CUDA backend compiles, along with the math that only fed them.

    ``torch._assert_async(cond, msg)`` exports as this op, and the argument
    checks in ``torch.distributions`` use it. Inductor compiles each one into
    a device assert inside a Triton kernel. AOTInductor then runs every kernel
    once at compile time to pick its launch settings, on made-up buffers:
    random floats, and zeros for integers and bools. A check that reads a bool
    computed by an earlier kernel sees false there and fails, even when it
    holds for every real input. A failed device assert also breaks the CUDA
    context, so the export cannot go on.

    ExecuTorch already removes these checks from the rest of a program before
    it is saved (see ``RemoveGraphAssertsPass``), so the delegate now does the
    same.
    """

    def call(self, graph_module: GraphModule) -> PassResult:
        modified = False
        for module in graph_module.modules():
            if not isinstance(module, GraphModule):
                continue
            asserts = [
                node
                for node in module.graph.nodes
                if node.op == "call_function"
                and node.target == torch.ops.aten._assert_async.msg
            ]
            for node in asserts:
                module.graph.erase_node(node)
            if asserts:
                module.graph.eliminate_dead_code()
                module.recompile()
                modified = True
        return PassResult(graph_module, modified)
