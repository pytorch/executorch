# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import torch
from executorch.exir.dialects._ops import ops as exir_ops
from executorch.exir.pass_base import ExportPass, PassResult

from executorch.exir.sym_util import eval_shape, eval_shape_upper_bound
from torch.fx.experimental.symbolic_shapes import statically_known_true, sym_eq


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
    Removes the view_copy after a linear when the linear already produces its
    shape, so x -> linear -> view_copy -> y becomes x -> linear -> y. A linear
    with other readers besides the view is left as is.

    When the linear reads a dequantize_per_tensor.tensor node through view_copy
    nodes and has no bias of more than one dim, it reads that node instead if
    that makes its output match the view, so
    dq -> view_copy -> linear -> view_copy -> y becomes dq -> linear -> y.
    Linear nodes can handle input tensor with > 2 dimensions.

    A view goes only when the shapes are known to be equal. With dynamic shapes,
    the linear's input and the view need symbolic shapes in meta["val"].
    """
    ops = exir_ops.edge
    for node in graph.nodes:
        if node.op != "call_function" or node.target != ops.aten.linear.default:
            continue
        if len(node.users) != 1:
            continue
        view_copy = next(iter(node.users))
        if view_copy.target != ops.aten.view_copy.default:
            continue
        weight_node = node.args[1]
        if "val" not in weight_node.meta:
            raise ValueError(f"Val not found meta of node {weight_node}")
        output_channels = weight_node.meta["val"].shape[0]
        input_node = node.args[0]
        dqlinear_input = get_dqlinear_input(input_node)
        bias_node = node.args[2] if len(node.args) > 2 else None
        if bias_node is not None and len(bias_node.meta["val"].shape) > 1:
            # A linear kept from export can have a bias of more than one dim
            # that follows the rows of the flattened input, and then it would
            # not broadcast to the dequantized shape.
            dqlinear_input = None
        # Reading the dequantized input before its views changes the shape of the
        # linear output, so it is done only when the view after the linear goes.
        for new_input in (dqlinear_input, input_node):
            if new_input is None:
                continue
            output_shape = list(new_input.meta["val"].shape)
            output_shape[-1] = output_channels
            # Compare the symbolic shapes, since different dynamic shapes can
            # have equal upper bounds.
            if statically_known_true(
                sym_eq(output_shape, list(view_copy.meta["val"].shape))
            ):
                node.update_arg(0, new_input)
                node.meta["val"] = view_copy.meta["val"]
                view_copy.replace_all_uses_with(node)
                break
    graph.eliminate_dead_code()
    return graph


def is_transpose(node: torch.fx.Node) -> bool:
    """
    Returns whether node transposes a 2-D tensor, so that a linear can take the
    tensor before it as its weight.
    """
    ops = exir_ops.edge
    if node.target == ops.aten.permute_copy.default:
        return [d % 2 for d in node.args[1]] == [1, 0]
    return node.target == ops.aten.t_copy.default


def addmm_can_be_linear(node: torch.fx.Node) -> bool:
    """
    Returns whether linear(x, w, b) can replace addmm(b, x, w.t()). The weight
    must be transposed, linear cannot scale its bias or its product, and
    ExecuTorch's linear kernel takes only a 1-D bias.
    """
    return (
        is_transpose(node.args[2])
        and node.kwargs.get("beta", 1) == 1
        and node.kwargs.get("alpha", 1) == 1
        and len(node.args[0].meta["val"].shape) == 1
    )


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
                    if not addmm_can_be_linear(node):
                        continue
                    weight_node = node.args[2].args[0]
                    args = (node.args[1], weight_node, node.args[0])
                    linear_node = graph.create_node(
                        "call_function", ops.aten.linear.default, args
                    )
                    node.replace_all_uses_with(linear_node)
                    output_val = linear_node.target(  # pyre-fixme[29]
                        args[0].meta["val"], args[1].meta["val"], args[2].meta["val"]
                    )
                else:
                    if not is_transpose(node.args[1]):
                        continue
                    weight_node = node.args[1].args[0]
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
