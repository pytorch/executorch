# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

import torch
from executorch.exir.dialects._ops import ops as exir_ops
from executorch.exir.pass_base import ExportPass, PassResult

from torch._subclasses.fake_tensor import FakeTensor


def node_is_local_scalar_dense_chain(node: torch.fx.Node) -> bool:
    """
    Converting a tensor to a scalar via tensor[0].item() creates a index_select +
    local_scalar_dense pattern in the graph. Check if a node is the start of this pattern.
    """
    if (
        node.op == "call_function"
        and node.target == exir_ops.edge.aten.select_copy.int
        and len(node.users) == 1
    ):
        user = list(node.users.keys())[0]
        return user.target == torch.ops.aten._local_scalar_dense.default

    return False


def tag_node_if_scalar_tensor(node: torch.fx.Node) -> None:
    """
    A scalar tensor in the Vulkan backend is a tensor that can be represented as a scalar
    value instead of a Tensor object. The criteria for identifying a tensor as a scalar
    tensor are as follows:

    1. The tensor has only 1 element
    2. One of the node's uses is converting it to a scalar via `tensor[0].item()`, which
       creates a index_select + local_scalar_dense pattern in the graph

    If any of these criteria are fulfilled, then tag the node for the tensor to mark it
    so that it is added as a scalar value during serialization.
    """
    tensor_val = node.meta["val"]
    if not isinstance(tensor_val, FakeTensor):
        return

    # Scalar tensors must have only one element
    if tensor_val.numel() != 1:
        return

    for user in node.users:
        if node_is_local_scalar_dense_chain(user):
            node.meta["etvk_is_scalar_tensor"] = True


def remove_local_scalar_dense_chain(graph: torch.fx.Graph, node: torch.fx.Node) -> None:
    """
    Remove the index_select + local_scalar_dense pattern in the graph in favor of passing
    the original scalar tensor directly.
    """
    replace_node = node.args[0]
    assert isinstance(replace_node, torch.fx.Node)
    # If the argument to the local_scalar_dense op is a select op with only
    # one user, and the argument to the select op is a tensor with only one
    # element (i.e. a scalar tensor), then replace the entire pattern with the
    # scalar tensor.
    if (
        replace_node.op == "call_function"
        and replace_node.target == exir_ops.edge.aten.select_copy.int
    ):
        # pyre-ignore
        if replace_node.args[0].meta["val"].numel() == 1:
            replace_node = replace_node.args[0]
            assert isinstance(replace_node, torch.fx.Node)
            assert replace_node.meta.get("etvk_is_scalar_tensor", True)

    with graph.inserting_after(node):
        node.replace_all_uses_with(replace_node)


def remove_local_scalar_dense_ops(graph: torch.fx.Graph) -> torch.fx.Graph:
    """
    The purpose of this pass is twofold:
    1. Tag scalar tensors (see `tag_node_if_scalar_tensor()` for the criteria)
    2. Remove the index_select + local_scalar_dense pattern in the graph in favor of
       passing the original scalar tensor directly (see `remove_local_scalar_dense_chain()`)

    This makes it easier to deal with scalar tensors in the Vulkan backend. In particular,
    it allows serializing scalar tensors as SymInt objects instead of Tensor objects.
    Because scalar tensors are often used to inform tensor shapes, their values need to
    be easily accessed by the CPU during resizing logic, while also being able to reflect
    updates to their value in any GPU shaders that reference them.
    """
    target_op = torch.ops.aten._local_scalar_dense.default
    for node in graph.nodes:
        tag_node_if_scalar_tensor(node)

        if node.op == "call_function" and node.target == target_op:
            remove_local_scalar_dense_chain(graph, node)

    graph.eliminate_dead_code()
    return graph


class RemoveLocalScalarDenseOpsTransform(ExportPass):
    def call(self, graph_module: torch.fx.GraphModule) -> PassResult:
        graph_module.graph = remove_local_scalar_dense_ops(graph_module.graph)
        return PassResult(graph_module, True)
