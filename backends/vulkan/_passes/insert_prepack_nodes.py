# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

from copy import deepcopy

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

    for node in program.graph_module.graph.nodes:
        # Prepacking is only needed for constant tensors. Only nodes corresponding to
        # constant tensors will proceed beyond this point.
        if not is_param_node(program, node):
            continue

        # Mark that this node is going to be represented as a TensorRef type in the
        # Vulkan compute graph. This annotation is used in later graph passes.
        node.meta["etvk_tensorref"] = True

        # Get the list of node users that do not handle their own prepacking
        nodes_to_replace_input = []
        for user in node.users:
            if user.op == "call_function" and not handles_own_prepacking(user.target):
                nodes_to_replace_input.append(user)

        if len(nodes_to_replace_input) == 0:
            continue

        replace_all_uses = len(nodes_to_replace_input) == len(node.users)

        with program.graph_module.graph.inserting_after(node):
            prepack_node = program.graph_module.graph.create_node(
                "call_function",
                exir_ops.edge.et_vk.prepack.default,
                (node,),
            )
            # This pass assumes that the SpecPropPass() has already been applied
            assert "spec" in node.meta
            # Mutable buffers will not be marked as constant, but it might as well be
            # for the purposes of memory planning. Mark it as a constant tensor so that
            # it is handled correctly by the memory planning pass.
            if not node.meta["spec"].const:
                assert is_param_node(program, node)
                node.meta["spec"].const = True
            # Validate that the original node is marked as a constant. Constant tensors
            # do not participate in memory planning.
            assert node.meta["spec"].const
            prepack_node.meta["val"] = node.meta["val"]
            prepack_node.meta["spec"] = deepcopy(node.meta["spec"])
            # Set the mem_obj_id to -1 to indicate that this node requires a dedicated
            # memory object.
            prepack_node.meta["spec"].mem_obj_id = -1
            if replace_all_uses:
                node.replace_all_uses_with(
                    prepack_node,
                    lambda x, y=prepack_node: (x != y and x.op != "output"),
                )
            else:
                for user_node in nodes_to_replace_input:
                    user_node.replace_input_with(node, prepack_node)

    program.graph.eliminate_dead_code()
    return program
