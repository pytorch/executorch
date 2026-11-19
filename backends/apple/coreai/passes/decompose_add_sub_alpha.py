# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Decompose add/sub with a non-unit ``alpha`` for the Core AI backend."""

import torch
from executorch.exir.dialects._ops import ops as exir_ops
from executorch.exir.dialects.edge._ops import EdgeOpOverload
from executorch.exir.pass_base import PassResult
from torch.fx import GraphModule, Node

_ADD_SUB = (
    torch.ops.aten.add.Tensor,
    torch.ops.aten.sub.Tensor,
    torch.ops.aten.add.Scalar,
    torch.ops.aten.sub.Scalar,
)


class DecomposeAddSubAlphaPass:
    """Rewrite ``x ± alpha * y`` into ``x ± mul(y, alpha)`` when alpha != 1.

    coreai-torch lowers add/sub from their first two operands and ignores
    ``alpha``. A tensor ``y`` is multiplied by a 0-dim ``alpha`` tensor stored
    as a ``get_attr`` constant, which ExecuTorch's ``lift_constant_tensor_pass``
    turns into a constant buffer placeholder (not a user input). A scalar ``y``
    is folded with ``alpha`` instead.

    Runs on ATen or edge graphs. Run it after :class:`NarrowToCoreAIDtypesPass`,
    whose fake-tensor propagation can't evaluate the inserted constant.
    """

    def __call__(self, graph_module: GraphModule) -> PassResult:
        modified = False
        for node in list(graph_module.graph.nodes):
            if node.op != "call_function" or _aten_op(node.target) not in _ADD_SUB:
                continue
            alpha = node.kwargs.get("alpha", 1)
            if isinstance(alpha, Node) or alpha == 1:
                continue

            other = node.args[1]
            if isinstance(other, Node):
                node.args = (node.args[0], _scale_tensor(graph_module, node, alpha))
            else:
                node.args = (node.args[0], _scale_scalar(other, alpha))
            node.kwargs = {k: v for k, v in node.kwargs.items() if k != "alpha"}
            modified = True

        if modified:
            graph_module.graph.lint()
            graph_module.recompile()
        return PassResult(graph_module, modified)


def _aten_op(target):
    return target._op if isinstance(target, EdgeOpOverload) else target


def _scale_scalar(other, alpha):
    if isinstance(other, bool) and isinstance(alpha, bool):
        return other and alpha
    return other * alpha


def _scale_tensor(graph_module: GraphModule, node: Node, alpha) -> Node:
    graph = graph_module.graph
    out_dtype = node.meta["val"].dtype
    other = node.args[1]
    other_val = other.meta["val"]
    fake_mode = other_val.fake_mode

    if isinstance(node.target, EdgeOpOverload):
        mul_op = exir_ops.edge.aten.mul.Tensor
        to_copy_op = exir_ops.edge.aten._to_copy.default
    else:
        mul_op = torch.ops.aten.mul.Tensor
        to_copy_op = torch.ops.aten._to_copy.default

    # 0-dim so the product keeps other's shape and a 0-dim result stays 0-dim.
    alpha_tensor = torch.tensor(alpha, dtype=out_dtype)
    alpha_val = fake_mode.from_tensor(alpha_tensor)
    op_meta = {k: v for k, v in node.meta.items() if k not in ("val", "tensor_meta")}

    with graph.inserting_before(node):
        # A dimensioned lower-precision other (e.g. f16 against an f32 result)
        # would win promotion against the 0-dim alpha and round alpha * other
        # to f16, so cast it to the result dtype first.
        if torch.result_type(other_val, alpha_val) != out_dtype:
            other_val = other_val.to(out_dtype)
            other = graph.call_function(to_copy_op, (other,), {"dtype": out_dtype})
            other.meta = {**op_meta, "val": other_val}

        # create_node, not get_attr: an ExportedProgram's graph has two owning
        # modules, so get_attr can't see the buffer and warns spuriously.
        alpha_node = graph.create_node(
            "get_attr", _register_constant(graph_module, alpha_tensor)
        )
        alpha_node.meta["val"] = alpha_val

        scaled = graph.call_function(mul_op, (other, alpha_node))
        with fake_mode:
            scaled.meta = {**op_meta, "val": other_val * alpha_val}
    return scaled


def _register_constant(graph_module: GraphModule, tensor: torch.Tensor) -> str:
    index = 0
    while hasattr(graph_module, f"_coreai_alpha_constant{index}"):
        index += 1
    name = f"_coreai_alpha_constant{index}"
    graph_module.register_buffer(name, tensor)
    return name
