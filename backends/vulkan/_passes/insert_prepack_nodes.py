# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

from typing import List

import executorch.backends.vulkan._passes.custom_ops_defs  # noqa

import torch

from executorch.exir.dialects._ops import ops as exir_ops

from torch._export.utils import is_buffer, is_param
from torch.export import ExportedProgram

USES_WEIGHTS: List[torch._ops.OpOverload] = [
    exir_ops.edge.aten.embedding.default,
    exir_ops.edge.aten.convolution.default,
    exir_ops.edge.et_vk.conv_with_clamp.default,
    exir_ops.edge.aten.linear.default,
    exir_ops.edge.aten._weight_int8pack_mm.default,
    exir_ops.edge.et_vk.linear_weight_int4.default,
    exir_ops.edge.aten._native_batch_norm_legit_no_training.default,
    exir_ops.edge.aten.native_layer_norm.default,
    "llama::sdpa_with_kv_cache",
]


def insert_prepack_nodes(program: ExportedProgram) -> ExportedProgram:
    """
    Insert `et_vk.prepack` nodes for constant tensors in the graph. The prepack operator
    is responsible for transferring the tensor data, which is serialized with the model,
    to a GPU tensor object during the prepacking stage of model execution.

    Some operators, listed in `USES_WEIGHTS` above, are performance sensitive and will
    prefer to handle prepacking within the operator. For these ops, the constant tensor
    data will be passed directly as an argument into the operator implementation.
    """

    def is_get_attr_node(node: torch.fx.Node) -> bool:
        return isinstance(node, torch.fx.Node) and node.op == "get_attr"

    def is_constant(node: torch.fx.Node) -> bool:
        return node.name in program.graph_signature.inputs_to_lifted_tensor_constants

    def is_param_node(node: torch.fx.Node) -> bool:
        """
        Check if the given node is a parameter within the exported program
        """
        return (
            is_get_attr_node(node)
            or is_param(program, node)
            or is_buffer(program, node)
            or is_constant(node)
        )

    def is_non_weight_param_tensor(node: torch.fx.Node) -> bool:
        if not is_param_node(node):
            return False

        for user in node.users:
            if user.op == "call_function" and (
                # pyre-ignore [16]
                user.target in USES_WEIGHTS
                or user.target.name() in USES_WEIGHTS
            ):
                return False

        return True

    for node in program.graph_module.graph.nodes:
        if not is_non_weight_param_tensor(node):
            continue

        with program.graph_module.graph.inserting_after(node):
            prepack_node = program.graph_module.graph.create_node(
                "call_function",
                exir_ops.edge.et_vk.prepack.default,
                (node,),
            )
            prepack_node.meta["spec"] = node.meta["spec"]
            # Set the mem_obj_id to -1 to indicate that this node requires a dedicated
            # memory object. This pass must be executed AFTER the memory planning pass.
            prepack_node.meta["spec"].mem_obj_id = -1
            node.replace_all_uses_with(prepack_node, lambda x, y=prepack_node: x != y)

    program.graph.eliminate_dead_code()
    return program
