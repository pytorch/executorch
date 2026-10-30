# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import logging
import operator

import torch
from executorch.exir.dialects._ops import ops as exir_ops
from executorch.exir.passes.constant_prop_pass import _nodes_written_in_place
from torch.fx.passes.infra.pass_base import PassBase, PassResult
from torch.utils import _pytree as pytree


def _reads_only_shape(node: torch.fx.Node) -> bool:
    """full_like with a constant fill value reads only the shape and dtype of
    its input. With a static shape its result is a constant, and OpenVINO
    folds it into one too.
    """
    if node.target != exir_ops.edge.aten.full_like.default:
        return False
    if any(
        isinstance(a, torch.fx.Node) for a in (*node.args[1:], *node.kwargs.values())
    ):
        return False
    val = node.meta.get("val")
    return isinstance(val, torch.Tensor) and all(isinstance(d, int) for d in val.shape)


def _inputs(node: torch.fx.Node) -> list:
    return [] if _reads_only_shape(node) else node.all_input_nodes


def _on_cpu(value):
    if isinstance(value, torch.Tensor):
        return value.cpu()
    if isinstance(value, torch.device):
        return torch.device("cpu")
    return value


def _constant_nodes(graph_module: torch.fx.GraphModule) -> set:
    """Nodes that depend only on constants. Graph order puts inputs first."""
    constants = set()
    written = _nodes_written_in_place(graph_module)
    for node in graph_module.graph.nodes:
        # Only tensors: a higher-order op such as cond reads its subgraphs,
        # which may hold a random op, through get_attr nodes too.
        if node.op == "get_attr":
            value = operator.attrgetter(node.target)(graph_module)
            if isinstance(value, torch.Tensor) and node not in written:
                constants.add(node)
        elif node.op == "call_function" and (
            _reads_only_shape(node)
            or (
                torch.Tag.nondeterministic_seeded
                not in getattr(node.target, "tags", ())
                and all(n in constants for n in node.all_input_nodes)
            )
        ):
            constants.add(node)
    return constants


def _evaluate(graph_module: torch.fx.GraphModule, targets: list) -> dict:
    """Values of targets, computing only what they read, on the CPU."""
    needed = set()
    stack = list(targets)
    while stack:
        node = stack.pop()
        if node not in needed:
            needed.add(node)
            stack.extend(_inputs(node))
    order = [n for n in graph_module.graph.nodes if n in needed]
    last_use = {i: n for n in order for i in _inputs(n)}
    values = {}
    for node in order:
        if node.op == "get_attr":
            values[node] = operator.attrgetter(node.target)(graph_module)
            continue
        if _reads_only_shape(node):
            like = node.args[0].meta["val"]
            args = (torch.empty(like.shape, dtype=like.dtype), *node.args[1:])
            kwargs = dict(node.kwargs)
        else:
            args, kwargs = torch.fx.node.map_arg(
                (node.args, node.kwargs), lambda n: values[n]
            )
        # The values are the same on any device, and lowering for OpenVINO
        # must not need the GPU a constant was made for.
        args, kwargs = pytree.tree_map(_on_cpu, (args, kwargs))
        if kwargs.get("pin_memory"):
            kwargs = {**kwargs, "pin_memory": False}
        values[node] = node.target(*args, **kwargs)
        for i in _inputs(node):
            if last_use[i] is node and i not in targets:
                del values[i]
    return values


class FoldConstantCumsumPass(PassBase):
    """Replace an integer cumsum that reads only constants with its result.

    With an integer cumsum over a constant, OpenVINO's CPU plugin (2025.4 to
    2026.4) gives wrong results from the second inference on: masks built from
    it keep only what earlier calls allowed. pi0 builds its attention masks
    this way. The OpenVINO nightly no longer does this, so the pass can go once
    the oldest supported OpenVINO release has the fix. Only these cumsums are
    folded; constants that feed no such cumsum, such as compressed weights, are
    left as they are. Runs on an unlifted module, where constants are get_attr
    nodes; buffers the graph writes to are not constants, and full_like of an
    input with a static shape is one.
    """

    def call(self, graph_module: torch.fx.GraphModule) -> PassResult:
        graph = graph_module.graph
        constants = _constant_nodes(graph_module)
        cumsums = [
            n
            for n in graph.nodes
            if n in constants
            and n.target == exir_ops.edge.aten.cumsum.default
            and not n.meta["val"].dtype.is_floating_point
            and not n.meta["val"].dtype.is_complex
        ]
        modified = False
        # One at a time, so a cumsum that cannot be evaluated does not keep
        # the others from being folded.
        for node in cumsums:
            try:
                value = _evaluate(graph_module, [node])[node]
            except RuntimeError as e:
                # Some ops have modes the CPU lacks.
                logging.warning(
                    "OpenVINO: could not fold %s, so on a CPU some OpenVINO "
                    "releases give wrong results for it from the second "
                    "inference on: %s",
                    node.name,
                    e,
                )
                continue
            name = f"_folded_cumsum_{node.name}"
            while hasattr(graph_module, name):
                name += "_"
            graph_module.register_buffer(name, value)
            with graph.inserting_before(node):
                folded = graph.get_attr(name)
            folded.meta = node.meta.copy()
            node.replace_all_uses_with(folded)
            modified = True
        if modified:
            graph.eliminate_dead_code()
            graph_module.recompile()
        return PassResult(graph_module, modified)
