# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

from copy import deepcopy

import executorch.backends.vulkan.custom_ops_lib  # noqa

import torch

from executorch.backends.vulkan.op_registry import handles_own_prepacking
from executorch.backends.vulkan.utils import is_param_node

from executorch.exir.dialects._ops import ops as exir_ops

from torch.export import ExportedProgram


def insert_prepack_nodes(program: ExportedProgram) -> ExportedProgram:
    """
    Insert `et_vk.prepack` nodes for constant tensors in the graph. The prepack operator
    is responsible for transferring the tensor data, which is serialized with the model,
    to a GPU tensor object during the prepacking stage of model execution.

    Some operators are performance sensitive and will prefer to handle prepacking within
    the operator. For these ops, the constant tensor data will be passed directly as an
    argument into the operator implementation.
    """

    def prepack_not_required(node: torch.fx.Node) -> bool:
        if not is_param_node(program, node):
            return True

        # Annotate that this node is going to represented as a tensorref in the Vulkan
        # compute graph. This will be useful for later graph passes.
        node.meta["vkdg_tensorref"] = True

        for user in node.users:
            if user.op == "call_function" and handles_own_prepacking(
                # pyre-ignore
                user.target
            ):
                return True

        return False

    for node in program.graph_module.graph.nodes:
        if prepack_not_required(node):
            continue

        with program.graph_module.graph.inserting_after(node):
            prepack_node = program.graph_module.graph.create_node(
                "call_function",
                exir_ops.edge.et_vk.prepack.default,
                (node,),
            )
            # This pass assumes that the SpecPropPass() has already been applied
            assert "spec" in node.meta
            # Validate that the original node is marked as a constant. Constant tensors
            # do not participate in memory planning.
            assert node.meta["spec"].const
            prepack_node.meta["val"] = node.meta["val"]
            prepack_node.meta["spec"] = deepcopy(node.meta["spec"])
            # Set the mem_obj_id to -1 to indicate that this node requires a dedicated
            # memory object.
            prepack_node.meta["spec"].mem_obj_id = -1
            node.replace_all_uses_with(prepack_node, lambda x, y=prepack_node: x != y)

    program.graph.eliminate_dead_code()
    return program
