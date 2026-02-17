# Copyright 2026 Arm Limited and/or its affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from typing import Set, Type

from executorch.backends.arm._passes.arm_pass import ArmPass
from executorch.backends.transforms.utils import is_get_attr_node
from executorch.exir.dialects._ops import ops as exir_ops
from executorch.exir.graph_module import get_cond_while_submodules

from executorch.exir.pass_base import ExportPass, PassResult
from torch.fx import GraphModule


class ControlFlowConstInlinePass(ArmPass):
    """
        When we lift out each control flow body as its own GraphModule, any scalar constants that were captured in Python become module attributes.  FX represents those as get_attr nodes in the
    submodule graph. These become getattr nodes submodule graph.

        This pass ensures that Scalar tensors in control flow operation are converted from getattr operators to expected call_function full ops.
    """

    _passes_required_after: Set[Type[ExportPass]] = set()

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def call(self, graph_module: GraphModule) -> PassResult:
        modified = False

        for _, submodule, _ in get_cond_while_submodules(graph_module):
            for submodule_node in submodule.graph.nodes:
                if is_get_attr_node(submodule_node):
                    val = getattr(
                        submodule_node.graph.owning_module, submodule_node.target
                    )
                    with submodule.graph.inserting_before(submodule_node):
                        const_node = submodule.graph.create_node(
                            op="call_function",
                            target=exir_ops.edge.aten.full.default,
                            args=(val.shape, val.item()),
                            kwargs={
                                "device": submodule_node.meta["val"].device,
                                "dtype": submodule_node.meta["val"].dtype,
                            },
                        )
                    const_node.meta = submodule_node.meta
                    submodule_node.replace_all_uses_with(const_node)
                    submodule.graph.erase_node(submodule_node)
                    modified = True

        if modified:
            graph_module.recompile()
            graph_module.graph.eliminate_dead_code()

        return PassResult(graph_module, modified)
