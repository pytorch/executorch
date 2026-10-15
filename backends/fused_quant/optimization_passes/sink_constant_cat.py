# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.


from __future__ import annotations

from dataclasses import dataclass

import executorch.backends.fused_quant.ops  # noqa: F401
import torch
from executorch.backends.transforms.permute_pass_utils import get_arg
from executorch.exir.dialects._ops import ops as exir_ops
from executorch.exir.pass_base import ExportedProgramPassBase, ExportedProgramPassResult
from executorch.exir.passes.constant_prop_pass import constant_prop_pass
from torch import fx
from torch._dynamo.utils import detect_fake_mode
from torch.export import ExportedProgram
from torch.export.graph_signature import InputKind
from torch.fx.passes.fake_tensor_prop import FakeTensorProp


def _propagate_fake_tensors(graph_module: fx.GraphModule) -> None:
    """Recompute meta["val"] for all nodes via FakeTensorProp."""
    inputs = []
    for node in graph_module.graph.find_nodes(op="placeholder"):
        val = node.meta.get("val")
        if val is not None:
            inputs.append(val)
        else:
            inputs.append(node)

    fake_mode = detect_fake_mode(inputs)
    FakeTensorProp(graph_module, mode=fake_mode).propagate_dont_convert_inputs(*inputs)


@dataclass
class CatDimTracker:
    """Tracks the cat dimension through a chain of downstream ops.

    After a cat on dimension `dim`, this tracks where that dimension ends up
    as ops like permute, view, unsqueeze, squeeze, and slice are applied.
    If a reshape merges or splits the cat dimension, the tracker is invalidated.
    """

    dim: int
    valid: bool = True

    def update_permute(self, dims: list[int]) -> None:
        if not self.valid:
            return
        try:
            self.dim = dims.index(self.dim)
        except ValueError:
            self.valid = False

    def update_view(self, old_shape: list[int], new_shape: list[int]) -> None:
        if not self.valid:
            return

        i = 0  # index into new_shape
        j = 0  # index into old_shape
        new_cat_dim = None

        while i < len(new_shape) and j < len(old_shape):
            if new_shape[i] == old_shape[j]:
                if j == self.dim:
                    new_cat_dim = i
                    break
                i += 1
                j += 1
            elif new_shape[i] == 1:
                # Inserted unary dim in new_shape — skip it.
                i += 1
            else:
                # Sizes don't match — accumulate product on the old side.
                # If the cat dim is consumed in this merge, bail.
                product = old_shape[j]
                j += 1
                while product != new_shape[i]:
                    if j >= len(old_shape):
                        self.valid = False
                        return
                    if j == self.dim:
                        self.valid = False
                        return
                    product *= old_shape[j]
                    j += 1
                i += 1

        if new_cat_dim is None:
            self.valid = False
        else:
            self.dim = new_cat_dim

    def update_unsqueeze(self, unsqueeze_dim: int, ndim: int) -> None:
        if not self.valid:
            return
        if unsqueeze_dim < 0:
            unsqueeze_dim = ndim + 1 + unsqueeze_dim
        if unsqueeze_dim <= self.dim:
            self.dim += 1

    def update_squeeze(self, squeeze_dims: list[int]) -> None:
        if not self.valid:
            return
        if self.dim in squeeze_dims:
            self.valid = False
            return
        shift = sum(1 for d in squeeze_dims if d < self.dim)
        self.dim -= shift


_UNARY_ELEMENTWISE_OPS: frozenset[torch._ops.OpOverload] = frozenset(
    {
        exir_ops.edge.aten.relu.default,
        exir_ops.edge.aten.sigmoid.default,
        exir_ops.edge.aten.hardswish.default,
        exir_ops.edge.aten.clamp.default,
        exir_ops.edge.fused_quant.relu.default,
        exir_ops.edge.fused_quant.hardswish.default,
    }
)


def _track_dim_through_op(
    node: fx.Node,
    tracker: CatDimTracker,
) -> bool:
    """Update the tracker for a single op. Returns True if the op is safe."""
    if not tracker.valid:
        return False

    target = node.target

    if target in _UNARY_ELEMENTWISE_OPS:
        return True

    if target == exir_ops.edge.aten.slice_copy.Tensor:
        slice_dim = get_arg(node, "dim", int)
        input_node = node.args[0]
        assert isinstance(input_node, fx.Node)
        ndim = input_node.meta["val"].ndim
        if slice_dim < 0:
            slice_dim = ndim + slice_dim
        if slice_dim == tracker.dim:
            return False
        return True

    if target == exir_ops.edge.aten.permute_copy.default:
        dims = get_arg(node, "dims", list[int])
        tracker.update_permute(dims)
        return tracker.valid

    if target == exir_ops.edge.aten.unsqueeze_copy.default:
        dim = get_arg(node, "dim", int)
        input_node = get_arg(node, "input", fx.Node)
        tracker.update_unsqueeze(dim, input_node.meta["val"].ndim)
        return tracker.valid

    if target == exir_ops.edge.aten.squeeze_copy.dims:
        dims = get_arg(node, "dims", list[int])
        tracker.update_squeeze(dims)
        return tracker.valid

    if target == exir_ops.edge.aten.view_copy.default:
        input_node = get_arg(node, "input", fx.Node)
        old_shape = list(input_node.meta["val"].shape)
        new_shape = list(node.meta["val"].shape)
        tracker.update_view(old_shape, new_shape)
        return tracker.valid

    # As we support more data movement patterns, we can add them here.
    return False


@dataclass
class SinkPath:
    """A single user path that the cat can be sunk through."""

    chain: list[fx.Node]
    cat_dim_per_node: dict[fx.Node, int]
    new_cat_dim: int


def _find_sink_paths(
    cat_node: fx.Node,
    cat_dim: int,
) -> list[SinkPath]:
    """Find sink paths for each user of the cat node.

    Walks forward from each user along a single-user chain, collecting
    ops that the cat can be sunk through. Returns a SinkPath per user
    that has at least one sinkable op.
    """
    paths: list[SinkPath] = []
    for user in list(cat_node.users.keys()):
        tracker = CatDimTracker(dim=cat_dim)
        cursor = user
        chain: list[fx.Node] = []
        cat_dim_per_node: dict[fx.Node, int] = {}

        while True:
            if cursor.op != "call_function":
                break
            cat_dim_per_node[cursor] = tracker.dim
            if not _track_dim_through_op(cursor, tracker):
                break
            chain.append(cursor)

            # While we support sinking cats if there are multiple users, we
            # stop in the chain if we hit a fork. We can run this pass iteratively
            # to address multiple forks if they appear.
            if len(cursor.users) != 1:
                break
            cursor = next(iter(cursor.users.keys()))

        if chain:
            paths.append(
                SinkPath(
                    chain=chain,
                    cat_dim_per_node=cat_dim_per_node,
                    new_cat_dim=tracker.dim,
                )
            )
    return paths


def _is_constant_node(node: fx.Node, exported_program: ExportedProgram) -> bool:
    for spec in exported_program.graph_signature.input_specs:
        if spec.arg.name == node.name:
            if spec.kind in (InputKind.PARAMETER, InputKind.CONSTANT_TENSOR):
                return True
            if spec.kind == InputKind.BUFFER:
                assert spec.target is not None
                return (
                    spec.target
                    not in exported_program.graph_signature.buffers_to_mutate.values()
                )
            return False
    return False


def _adjust_view_shape(
    original_shape: list[int],
    cat_dim: int,
    new_input: fx.Node,
) -> list[int]:
    """Compute the correct view shape for a replicated branch.

    The original view shape was computed for the full cat output. The
    replicated branch has a different size on the cat dim, so we scale
    the cat-dim component proportionally.
    """
    original_input_shape = new_input.meta["val"].shape
    new_cat_dim_size = original_input_shape[cat_dim]

    adjusted = list(original_shape)
    adjusted[cat_dim] = new_cat_dim_size
    return adjusted


def _replicate_chain(
    chain: list[fx.Node],
    node_map: dict[fx.Node, fx.Node],
    cat_dim_per_node: dict[fx.Node, int],
    graph: fx.Graph,
    target_node: fx.Node,
) -> None:
    """Replicate a chain of ops, fixing view shapes for the new input size."""
    for op_node in chain:
        new_args = _replace_nodes_in_args(op_node.args, node_map)

        if op_node.target == exir_ops.edge.aten.view_copy.default:
            cat_dim = cat_dim_per_node[op_node]
            input_in_map = op_node.args[0]
            assert isinstance(input_in_map, fx.Node)
            new_input = node_map[input_in_map]
            original_output_shape = list(op_node.meta["val"].shape)
            adjusted_shape = _adjust_view_shape(
                original_output_shape, cat_dim, new_input
            )
            new_args = (new_args[0], adjusted_shape)

        with graph.inserting_before(target_node):
            new_node = graph.call_function(
                op_node.target,  # pyre-ignore[6]
                args=new_args,  # pyre-ignore[6]
                kwargs=op_node.kwargs,
            )
        new_node.meta = op_node.meta.copy()
        node_map[op_node] = new_node


def _replace_nodes_in_args(
    args: tuple[object, ...],
    node_map: dict[fx.Node, fx.Node],
) -> tuple[object, ...]:
    result: list[object] = []
    for arg in args:
        if isinstance(arg, fx.Node) and arg in node_map:
            result.append(node_map[arg])
        elif isinstance(arg, (list, tuple)):
            replaced = [
                node_map[a] if (isinstance(a, fx.Node) and a in node_map) else a
                for a in arg
            ]
            result.append(type(arg)(replaced))
        else:
            result.append(arg)
    return tuple(result)


def _sink_cat_along_paths(
    cat_node: fx.Node,
    paths: list[SinkPath],
    constant_inputs: list[fx.Node],
    activation_inputs: list[fx.Node],
    graph: fx.Graph,
) -> None:
    """Push cat past downstream ops, splitting only constant inputs out.

    Only the constant inputs get their own replicated chain. Activation
    inputs stay together in the original cat (or a reduced cat if some
    constants were removed). At the sink point, a new cat merges the
    activation path with each constant path.

    Before:
        [act_a, act_b, const] -> cat -> op1 -> op2 -> consumers

    After (before constant folding):
        [act_a, act_b] -> cat -> op1 -> op2 -> ─┐
                                                  ├─> cat -> consumers
        const -> op1 -> op2 -> ─────────────────┘

    After constant folding:
        [act_a, act_b] -> cat -> op1 -> op2 -> ─┐
                                                  ├─> cat -> consumers
        [folded_const] ────────────────────────┘
    """
    cat_dim = get_arg(cat_node, "dim", int)
    all_chain_nodes: list[fx.Node] = []

    for path in paths:
        chain = path.chain
        target_node = chain[-1]
        all_chain_nodes.extend(chain)

        # The activation path: keep activations in a (possibly smaller) cat,
        # then run the chain on that cat's output.
        if len(activation_inputs) == 1:
            act_head: fx.Node = activation_inputs[0]
        else:
            with graph.inserting_before(cat_node):
                act_head = graph.call_function(
                    exir_ops.edge.aten.cat.default,
                    args=(activation_inputs, cat_dim),
                )

        act_node_map: dict[fx.Node, fx.Node] = {cat_node: act_head}
        _replicate_chain(chain, act_node_map, path.cat_dim_per_node, graph, target_node)

        sink_outputs: list[fx.Node] = [act_node_map[chain[-1]]]

        # Each constant input gets its own replicated chain.
        for const_inp in constant_inputs:
            const_node_map: dict[fx.Node, fx.Node] = {cat_node: const_inp}
            _replicate_chain(
                chain, const_node_map, path.cat_dim_per_node, graph, target_node
            )
            sink_outputs.append(const_node_map[chain[-1]])

        with graph.inserting_before(target_node):
            new_cat = graph.call_function(
                exir_ops.edge.aten.cat.default,
                args=(sink_outputs, path.new_cat_dim),
            )
        target_node.replace_all_uses_with(new_cat)

    for op_node in reversed(all_chain_nodes):
        if len(op_node.users) == 0:
            graph.erase_node(op_node)
    if len(cat_node.users) == 0:
        graph.erase_node(cat_node)


class SinkConstantCat(ExportedProgramPassBase):
    """Sink cat ops past downstream ops to separate constant from activation paths.

    When a cat combines constant and activation inputs, downstream ops
    (view, permute, slice, etc.) operate on the full concatenated tensor,
    preventing the constant subgraph from being folded. This pass pushes the
    cat as far down as possible, applying downstream ops to each cat input
    independently. The constant path then becomes a pure constant subgraph
    that a subsequent constant propagation pass can fold away entirely,
    eliminating redundant runtime compute.
    """

    def call(self, exported_program: ExportedProgram) -> ExportedProgramPassResult:
        graph_module = exported_program.graph_module
        graph = graph_module.graph
        modified = False

        cat_nodes = graph.find_nodes(
            op="call_function", target=exir_ops.edge.aten.cat.default
        )

        for cat_node in cat_nodes:
            tensors = get_arg(cat_node, "tensors", list[fx.Node])
            constant_inputs = [
                t for t in tensors if _is_constant_node(t, exported_program)
            ]
            if not constant_inputs:
                continue
            activation_inputs = [
                t for t in tensors if not _is_constant_node(t, exported_program)
            ]

            cat_dim = get_arg(cat_node, "dim", int)
            paths = _find_sink_paths(cat_node, cat_dim)
            if len(paths) != len(list(cat_node.users)):
                # We require that for each user we should be able
                # to sink the cat node, since if we cannot for
                # every user, we still end up
                continue

            _sink_cat_along_paths(
                cat_node, paths, constant_inputs, activation_inputs, graph
            )

            # Make sure meta['val'] is correct after these types of modifications.
            # Shouldn't be too bad in practice, since only updates after we handle
            # all users of a cat node.
            _propagate_fake_tensors(graph_module)
            modified = True

        if modified:
            graph_module.recompile()
            exported_program = constant_prop_pass(exported_program)

        return ExportedProgramPassResult(
            exported_program=exported_program,
            modified=modified,
        )
