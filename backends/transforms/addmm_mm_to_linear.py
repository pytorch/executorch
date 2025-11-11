# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import torch
from executorch.exir.dialects._ops import ops as exir_ops
from executorch.exir.pass_base import ExportPass, PassResult

from executorch.exir.sym_util import eval_shape, eval_shape_upper_bound


_int64_max_dim_val = torch.iinfo(torch.int64).max - 1


def get_shape(input_node: torch.fx.Node):
    """
    If shape is symbolic then evaluate shape, otherwise if it has upperbound
    shape, then return upperbound shape.
    Note that we must check for upperbound because by default upperbound is int64_max
    """
    input_val = input_node.meta["val"]
    upper_bound_shape = eval_shape_upper_bound(input_val.shape)
    for i in range(len(input_val.shape)):
        # Unbounded shape get int64 max values assigned to it.
        # This is just hacking around it when export with dynamic shape
        # does not use constraint api but instead just traces the
        # modelw with tensors of the max size
        if upper_bound_shape[i] >= _int64_max_dim_val:
            return eval_shape(input_val.shape)
    return upper_bound_shape


def get_dqlinear_input(node: torch.fx.Node):
    ops = exir_ops.edge
    node_to_backtrack = node
    # First find the activation input
    # Then trace it backwards through all view copies
    # Until you find dequant node.
    # if any of the nodes, during backtracking, is not view_copy
    # then break
    while node_to_backtrack.op != "placeholder":
        if (
            node_to_backtrack.op == "call_function"
            and node_to_backtrack.target
            == ops.quantized_decomposed.dequantize_per_tensor.tensor
        ):
            return node_to_backtrack
        if (
            node_to_backtrack.op == "call_function"
            and node_to_backtrack.target == ops.aten.view_copy.default
        ):
            node_to_backtrack = node_to_backtrack.args[0]
        else:
            return None
    return None


def replace_linear_view_copy_input_output(graph: torch.fx.Graph) -> torch.fx.Graph:
    """
    Replaces pattern: x -> view_copy -> view_copy -> linear -> view_copy -> y
    with
    x -> linear -> y
    Linear nodes can handle input tensor with > 2 dimensions.
    """
    ops = exir_ops.edge
    for node in graph.nodes:
        if node.op == "call_function" and (node.target == ops.aten.linear.default):
            input_node = node.args[0]
            dqlinear_input = get_dqlinear_input(input_node)
            if dqlinear_input is not None and dqlinear_input != input_node:
                if len(input_node.args[0].users) == 1:
                    input_node.replace_all_uses_with(dqlinear_input)
                else:
                    print(
                        f"{input_node} has more than one user. Users: {input_node.users}"
                    )
            if len(node.users) == 1:
                users = list(node.users)
                maybe_view_copy = users[0]
                if maybe_view_copy.op == "call_function" and (
                    maybe_view_copy.target == ops.aten.view_copy.default
                ):
                    # Must update the input node since replaced the original node
                    input_node = node.args[0]
                    input_shape = list(get_shape(input_node))
                    weight_node = node.args[1]
                    if "val" not in weight_node.meta:
                        raise ValueError(f"Val not found meta of node {weight_node}")
                    weight_val = weight_node.meta["val"]
                    output_channels = weight_val.shape[0]
                    output_shape = input_shape
                    output_shape[-1] = output_channels
                    view_copy_out_shape = list(get_shape(maybe_view_copy))
                    if output_shape == view_copy_out_shape:
                        maybe_view_copy.replace_all_uses_with(node)
    graph.eliminate_dead_code()
    return graph


def replace_addmm_mm_with_linear(graph: torch.fx.Graph) -> torch.fx.Graph:
    """
    Replace calls to addmm/mm with linear node
    Reason is that it simplifies the downstream logic of lowering to just linear node.
    Furthermore it also removes various view_copy nodes. These nodes have been absorbed
    by delegated by ignoring them entirely.
    Furthermore, removing view_copy nodes has the advantage of not having to match
    against view copies which simplifies the pattern that has to be matched.
    Simplified patterns will be less brittle since symbolic ints and sizes creeping into
    the graph was making them harder to match.
    """
    ops = exir_ops.edge
    for node in graph.nodes:
        if node.op == "call_function" and (
            node.target == ops.aten.mm.default or node.target == ops.aten.addmm.default
        ):
            with graph.inserting_after(node):
                if node.target == ops.aten.addmm.default:
                    weight_t_node = node.args[2]
                    if weight_t_node.target not in [
                        ops.aten.t_copy.default,
                        ops.aten.permute_copy.default,
                    ]:
                        # Skip this node as it appears to be a standalone `addmm`
                        continue
                    weight_node = weight_t_node.args[0]
                    args = (node.args[1], weight_node, node.args[0])
                    linear_node = graph.create_node(
                        "call_function", ops.aten.linear.default, args
                    )
                    node.replace_all_uses_with(linear_node)
                    output_val = linear_node.target(  # pyre-fixme[29]
                        args[0].meta["val"], args[1].meta["val"], args[2].meta["val"]
                    )
                else:
                    weight_t_node = node.args[1]
                    if weight_t_node.target not in [
                        ops.aten.t_copy.default,
                        ops.aten.permute_copy.default,
                    ]:
                        # Skip this node as it appears to be a standalone `mm`
                        continue
                    weight_node = weight_t_node.args[0]
                    args = (node.args[0], weight_node)
                    linear_node = graph.create_node(
                        "call_function", ops.aten.linear.default, args
                    )
                    node.replace_all_uses_with(linear_node)
                    output_val = linear_node.target(  # pyre-fixme[29]
                        args[0].meta["val"], args[1].meta["val"]
                    )
                linear_node.meta = node.meta
                # Val contain in this meta and corresponding shape will not be accurate
                # Sub
                linear_node.meta["val"] = output_val
    graph.eliminate_dead_code()
    return graph


def apply_addmm_mm_to_linear_transform(graph: torch.fx.Graph) -> torch.fx.Graph:
    graph = replace_addmm_mm_with_linear(graph)
    graph = replace_linear_view_copy_input_output(graph)
    return graph


class AddmmToLinearTransform(ExportPass):
    def call(self, graph_module: torch.fx.GraphModule) -> PassResult:
        graph_module.graph = apply_addmm_mm_to_linear_transform(graph_module.graph)
        return PassResult(graph_module, True)
