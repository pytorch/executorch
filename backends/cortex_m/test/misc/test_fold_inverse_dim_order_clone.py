# Copyright 2026 Arm Limited and/or its affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import torch
import torch.fx
from executorch.backends.cortex_m.passes.fold_inverse_dim_order_clone_pass import (
    FoldInverseDimOrderClonePass,
)
from executorch.exir.dialects._ops import ops as exir_ops


_CLONE = exir_ops.edge.dim_order_ops._clone_dim_order.default


def _count(graph_module: torch.fx.GraphModule, target) -> int:
    return sum(
        1
        for n in graph_module.graph.nodes
        if n.op == "call_function" and n.target == target
    )


def _make_clone_pair_graph(
    first_dim_order: tuple[int, ...],
    second_dim_order: tuple[int, ...],
) -> torch.fx.GraphModule:
    """Hand-build a graph: placeholder -> clone(first) -> clone(second) -> output.
    The input tensor is contiguous (dim_order = (0, 1, 2, 3)).
    """
    shape = (1, 4, 1, 8)
    input_tensor = torch.empty(shape)

    graph = torch.fx.Graph()
    x = graph.placeholder("x")
    x.meta["val"] = input_tensor

    first = graph.create_node(
        "call_function",
        target=_CLONE,
        args=(x,),
        kwargs={"dtype": torch.float32, "dim_order": list(first_dim_order)},
    )
    first.meta["val"] = input_tensor

    second = graph.create_node(
        "call_function",
        target=_CLONE,
        args=(first,),
        kwargs={"dtype": torch.float32, "dim_order": list(second_dim_order)},
    )
    second.meta["val"] = input_tensor

    graph.output(second)

    return torch.fx.GraphModule(torch.nn.Module(), graph)


def test_fold_removes_inverse_pair():
    # Input is contiguous (dim_order 0,1,2,3); first clone reorders to NHWC,
    # second clone reorders back to the original -> net identity.
    gm = _make_clone_pair_graph(
        first_dim_order=(0, 2, 3, 1),
        second_dim_order=(0, 1, 2, 3),
    )
    assert _count(gm, _CLONE) == 2

    result = FoldInverseDimOrderClonePass()(gm)
    assert result.modified
    assert _count(result.graph_module, _CLONE) == 0


def test_fold_preserves_non_identity_pair():
    # Second clone's target is not the input's original dim_order, so the
    # composition isn't identity -- fold must not fire.
    gm = _make_clone_pair_graph(
        first_dim_order=(0, 2, 3, 1),
        second_dim_order=(0, 3, 1, 2),
    )
    assert _count(gm, _CLONE) == 2

    result = FoldInverseDimOrderClonePass()(gm)
    assert not result.modified
    assert _count(result.graph_module, _CLONE) == 2


def test_fold_respects_fanout():
    # If the first clone has another consumer, folding would orphan that
    # consumer's view of the reordered data. The pass must refuse.
    gm = _make_clone_pair_graph(
        first_dim_order=(0, 2, 3, 1),
        second_dim_order=(0, 1, 2, 3),
    )
    first_clone = next(
        n for n in gm.graph.nodes if n.op == "call_function" and n.target == _CLONE
    )
    output_node = next(n for n in gm.graph.nodes if n.op == "output")
    with gm.graph.inserting_before(output_node):
        extra_user = gm.graph.create_node(
            "call_function",
            target=torch.ops.aten.relu.default,
            args=(first_clone,),
        )
        extra_user.meta["val"] = first_clone.meta["val"]
    output_node.args = ((output_node.args[0], extra_user),)
    gm.recompile()

    result = FoldInverseDimOrderClonePass()(gm)
    assert not result.modified
    assert _count(result.graph_module, _CLONE) == 2
