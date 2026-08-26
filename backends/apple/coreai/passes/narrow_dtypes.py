# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Boundary dtype-narrowing pass for the Core AI backend."""

import torch
from executorch.exir.dialects.edge._ops import EdgeOpOverload
from executorch.exir.pass_base import PassResult
from torch.fx import GraphModule
from torch.fx.passes.fake_tensor_prop import FakeTensorProp

# Core AI represents only up-to-32-bit dtypes (coreai-torch narrows int64->si32
# and float64->f32 internally), so it can't carry a 64-bit tensor across the
# delegate boundary.
_NARROW = {torch.int64: torch.int32, torch.float64: torch.float32}
_SIXTY_FOUR_BIT = (torch.int64, torch.float64)


class NarrowToCoreAIDtypesPass:
    """Narrow int64/float64 to 32-bit inside the graph while preserving I/O.

    Casts int64/float64 graph inputs to 32-bit right after the placeholder,
    re-propagates dtypes, then casts 32-bit values feeding int64/float64 outputs
    back to 64-bit. The model's external input/output dtypes stay unchanged,
    while the interior (what Core AI sees) becomes 32-bit.

    The inserted ``_to_copy`` casts keep a 64-bit operand/result, so
    :class:`CoreAIPartitioner` (which rejects nodes with int64/float64 tensors)
    leaves them outside the delegate; the delegate boundary then sees only 32-bit
    tensors that match what coreai declares.

    Runs pre-partition (e.g. via ``get_default_passes``) at either the ATen or
    edge level (it picks the matching ``_to_copy`` overload).
    """

    def __call__(self, graph_module: GraphModule) -> PassResult:
        graph = graph_module.graph
        placeholders = [n for n in graph.nodes if n.op == "placeholder"]
        wide_inputs = [p for p in placeholders if _is_64bit(p.meta.get("val"))]
        externalized = _externalized_nodes(graph)
        if not wide_inputs and not externalized:
            return PassResult(graph_module, False)

        uses_edge = any(
            isinstance(node.target, EdgeOpOverload)
            for node in graph.nodes
            if node.op == "call_function"
        )
        if uses_edge:
            from executorch.exir.dialects._ops import ops as exir_ops

            to_copy = exir_ops.edge.aten._to_copy.default
        else:
            to_copy = torch.ops.aten._to_copy.default

        output_node = next(n for n in graph.nodes if n.op == "output")
        # Remember which outputs were 64-bit so we can restore them afterwards.
        orig_output_dtypes = [
            val.dtype if _is_64bit(val := _node_val(a)) else None
            for a in output_node.args[0]
        ]

        # 1. Narrow 64-bit inputs right after the placeholder.
        for placeholder in wide_inputs:
            val = placeholder.meta.get("val")
            narrow_dtype = _NARROW[val.dtype]
            with graph.inserting_after(placeholder):
                cast = graph.call_function(
                    to_copy, (placeholder,), {"dtype": narrow_dtype}
                )
            placeholder.replace_all_uses_with(cast)
            cast.args = (placeholder,)  # restore the cast's own (detached) input

        # 2. Re-propagate dtypes through the (now-32-bit) interior.
        FakeTensorProp(graph_module).propagate(*[p.meta["val"] for p in placeholders])

        # 2b. Narrow 64-bit operands of externalized ops. These are always
        # claimed by the delegate (only coreai can lower them), so their
        # operands always cross the boundary, but a 64-bit value produced
        # inside the graph -- indices from an argmax, say -- is not covered by
        # the input narrowing above.
        for node in externalized:
            for operand in _wide_tensor_operands(node):
                narrow_dtype = _NARROW[operand.meta["val"].dtype]
                with graph.inserting_before(node):
                    cast = graph.call_function(
                        to_copy, (operand,), {"dtype": narrow_dtype}
                    )
                # Set by hand: this runs after the propagate above.
                cast.meta["val"] = operand.meta["val"].to(narrow_dtype)
                node.replace_input_with(operand, cast)

        # 3. Widen 32-bit values feeding originally-64-bit outputs back.
        out_args = list(output_node.args[0])
        for i, arg in enumerate(out_args):
            orig_dtype = orig_output_dtypes[i]
            if orig_dtype is None or not isinstance(arg, torch.fx.Node):
                continue
            cur = _node_val(arg)
            if isinstance(cur, torch.Tensor) and cur.dtype != orig_dtype:
                with graph.inserting_before(output_node):
                    widen = graph.call_function(to_copy, (arg,), {"dtype": orig_dtype})
                # Set by hand: this runs after the propagate above, so nothing
                # else fills it in.
                widen.meta["val"] = cur.to(orig_dtype)
                out_args[i] = widen
        output_node.args = (tuple(out_args),)

        graph_module.recompile()
        return PassResult(graph_module, True)


def _externalized_nodes(graph):
    """Call sites of coreai's temporary externalized ops, if any.

    Imported lazily and defensively: this pass otherwise needs only torch and
    exir, while ``externalize`` pulls in coreai-torch. Without that package no
    externalized op can exist, so an empty list is the right answer.
    """
    try:
        from executorch.backends.apple.coreai.externalize import is_externalize_target
    except ImportError:
        return []

    return [
        n
        for n in graph.nodes
        if n.op == "call_function" and is_externalize_target(n.target)
    ]


def _wide_tensor_operands(node):
    """Node operands whose value is a 64-bit tensor."""
    return [
        arg
        for arg in node.args
        if isinstance(arg, torch.fx.Node) and _is_64bit(arg.meta.get("val"))
    ]


def _node_val(arg):
    return arg.meta.get("val") if isinstance(arg, torch.fx.Node) else None


def _is_64bit(val) -> bool:
    return isinstance(val, torch.Tensor) and val.dtype in _SIXTY_FOUR_BIT
