# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
# Copyright 2026 Arm Limited and/or its affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-unsafe

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any, cast

import torch
import torch.fx
from executorch.backends.transforms.channels_last_layout import PERMUTE_COPY_TARGETS
from executorch.backends.transforms.permute_pass_utils import get_arg, set_arg
from executorch.exir.dialects._ops import ops as exir_ops
from executorch.exir.pass_base import ExportPass, PassResult


class RemovePermutesAroundElementwiseOps(ExportPass):
    """
    Looks for subgraphs of elementwise ops sandwiched between permutes and removes those
    permutes if possible.
    Allows special handling for certain non-elementwise ops that can be easily updated
    based on the permute's parameter such as mean, cat, and slice.
    The repeat_interleave idiom (unsqueeze -> expand_copy -> merging view_copy) is
    recognised as a single rank-preserving unit; see _interleave_triple.

    ``extra_permutable_ops`` must be layout-equivariant without argument
    remapping. ``can_propagate`` lets a backend reject nodes that the shared
    pass would otherwise treat as layout-equivariant.

    ``permute_targets`` is the closed family of layout-copy operators the
    backend considers equivalent. A rewritten region must use one member of
    that family consistently, and any synthesized copies retain that member.
    """

    @dataclass()
    class Subgraph:
        start_permute: list[int]
        end_permute: list[int]
        permute_target: Any
        # Nodes in the subgraph, does not include permutes.
        nodes: set[torch.fx.Node] = field(default_factory=set)
        # Incoming edges to the subgraph from permute nodes.
        edges_in: set[tuple[torch.fx.Node, torch.fx.Node]] = field(default_factory=set)
        # Outgoing edges of the subgraph to permute nodes.
        edges_out: set[tuple[torch.fx.Node, torch.fx.Node]] = field(default_factory=set)
        # Outgoing edges to permutes that do not match end_permute. Those are
        # kept and their permutation rewritten to absorb the removed start
        # permute, as (producer, permute node, new permutation).
        edges_out_to_update: set[
            tuple[torch.fx.Node, torch.fx.Node, tuple[int, ...]]
        ] = field(default_factory=set)
        # Incoming edges from constant nodes that need a compensating permute.
        constant_edges_in: set[tuple[torch.fx.Node, torch.fx.Node]] = field(
            default_factory=set
        )
        # Per-node expected end permutation (may differ from end_permute
        # when the subgraph contains rank-changing views).
        node_end_permute: dict[torch.fx.Node, list[int]] = field(default_factory=dict)
        # Per-node expected start permutation for upstream traversal.
        node_start_permute: dict[torch.fx.Node, list[int]] = field(default_factory=dict)
        # repeat_interleave triples keyed by their unit-dim-inserting head node,
        # mapping to (dim, scale, expand_node, view_node). See _interleave_triple.
        interleaves: dict[
            torch.fx.Node, tuple[int, int, torch.fx.Node, torch.fx.Node]
        ] = field(default_factory=dict)

    def __init__(
        self,
        extra_permutable_ops: set | None = None,
        *,
        can_propagate: Callable[[torch.fx.Node], bool] | None = None,
        permute_targets: set | frozenset | None = None,
    ) -> None:
        super().__init__()
        self.can_propagate = can_propagate
        self._permute_targets = frozenset(permute_targets or PERMUTE_COPY_TARGETS)
        self._permutable_ops = {
            exir_ops.edge.aten.add.Tensor,
            exir_ops.edge.aten.mul.Tensor,
            exir_ops.edge.aten.sub.Tensor,
            exir_ops.edge.aten.hardtanh.default,
            exir_ops.edge.aten.clamp.default,
            exir_ops.edge.aten.cat.default,
            exir_ops.edge.aten.constant_pad_nd.default,
            exir_ops.edge.aten.mean.dim,
            exir_ops.edge.aten.pad.default,
            exir_ops.edge.aten.sum.dim_IntList,
            exir_ops.edge.aten.slice_copy.Tensor,
        }
        try:
            self._permutable_ops.add(
                exir_ops.edge.quantized_decomposed.quantize_per_tensor.default
            )
            self._permutable_ops.add(
                exir_ops.edge.quantized_decomposed.dequantize_per_tensor.default
            )
        except AttributeError:
            pass
        if extra_permutable_ops:
            self._permutable_ops |= extra_permutable_ops
        self._sq_unsq_cache: dict[torch.fx.Node, bool] = {}
        self._interleave_cache: dict[
            torch.fx.Node,
            tuple[int, int, torch.fx.Node, torch.fx.Node] | None,
        ] = {}

    _VIEW_OPS = (
        exir_ops.edge.aten.view_copy.default,
        exir_ops.edge.aten.view.default,
    )

    @staticmethod
    def _concrete_shape(node: torch.fx.Node) -> list[int] | None:
        """Return a node's shape, or None if it is missing or symbolic."""
        val = node.meta.get("val")
        if val is None:
            return None
        shape = val.shape
        if not all(isinstance(d, int) for d in shape):
            return None
        return [int(d) for d in shape]

    def _view_shapes(self, node: torch.fx.Node) -> tuple[list[int], list[int]] | None:
        """Concrete (input, output) shapes of a view op, else None."""
        if node.target not in self._VIEW_OPS:
            return None
        inp = node.args[0] if node.args else None
        if not isinstance(inp, torch.fx.Node):
            return None
        in_shape = self._concrete_shape(inp)
        out_shape = self._concrete_shape(node)
        if in_shape is None or out_shape is None:
            return None
        return in_shape, out_shape

    _PAD_OPS = (
        exir_ops.edge.aten.constant_pad_nd.default,
        exir_ops.edge.aten.pad.default,
    )

    @staticmethod
    def _find_extra_ones(longer: list[int], shorter: list[int]) -> list[int] | None:
        """Positions in ``longer`` whose removal yields ``shorter``.

        Every such position must hold a size-1 dim, so the two shapes differ by
        unit dims alone. Returns None when no such set of positions exists.
        """
        extra: list[int] = []
        j = 0
        for i, dim in enumerate(longer):
            if j < len(shorter) and dim == shorter[j]:
                j += 1
                continue
            if dim != 1:
                return None
            extra.append(i)
        return extra if j == len(shorter) else None

    def _is_squeeze_unsqueeze_view(self, node: torch.fx.Node) -> bool:
        """Check if a node is a view_copy that only adds or removes dims of
        size 1."""
        if node in self._sq_unsq_cache:
            return self._sq_unsq_cache[node]
        result = self._check_squeeze_unsqueeze_view(node)
        self._sq_unsq_cache[node] = result
        return result

    def _check_squeeze_unsqueeze_view(self, node: torch.fx.Node) -> bool:
        shapes = self._view_shapes(node)
        if shapes is None:
            return False
        in_shape, out_shape = shapes
        if len(out_shape) > len(in_shape):
            return self._find_extra_ones(out_shape, in_shape) is not None
        if len(in_shape) > len(out_shape):
            return self._find_extra_ones(in_shape, out_shape) is not None
        return False

    def _is_permutation_sink_view(self, node: torch.fx.Node) -> bool:
        """True if ``node`` is a reshape whose input has at most one non-unit dim.

        Flattening such a tensor -- e.g. the ``[1, C, 1, 1] -> [1, C]`` after a
        global pool -- is permutation-invariant: every layout of the input
        produces the identical output (the single non-unit run of elements is
        contiguous regardless of which axis holds it). The region may terminate
        here without a compensating permute when downstream consumers do not use
        the output shape for layout-dependent broadcasting.
        """
        if node.target not in self._VIEW_OPS:
            return False
        inp = node.args[0]
        assert isinstance(inp, torch.fx.Node)
        shape = inp.meta["val"].shape
        # Count a dim as non-unit unless it is a concrete size-1 (symbolic dims
        # are treated as non-unit, i.e. conservatively not a sink).
        non_unit = [d for d in shape if not (isinstance(d, int) and d == 1)]
        return len(non_unit) <= 1

    def _sink_users_are_layout_invariant(self, sink: torch.fx.Node) -> bool:
        """Return whether dropping layout at ``sink`` is safe for its consumers."""
        frontier = [(user, sink) for user in sink.users]
        visited: set[torch.fx.Node] = set()
        while frontier:
            node, producer = frontier.pop()
            if node in visited:
                continue
            visited.add(node)

            if node.op == "output":
                continue
            if node.target in self._permute_targets:
                # This explicit transform re-establishes the downstream layout,
                # so consumers beyond it do not depend on the sink's layout.
                continue
            if self._is_permutation_sink_view(node):
                continue

            tensor_inputs = [
                input_node
                for input_node in node.all_input_nodes
                if input_node.meta.get("val") is not None
            ]
            if any(
                input_node is not producer and input_node.meta["val"].numel() != 1
                for input_node in tensor_inputs
            ):
                return False
            if not self.is_node_permutable(node):
                return False
            frontier.extend((user, node) for user in node.users)
        return True

    def _inserted_unit_dim(self, node: torch.fx.Node) -> int | None:
        """Position of the size-1 dim ``node`` inserts, else None.

        Accepts both an explicit unsqueeze and a view_copy that only adds a
        single unit dim, matching how the rest of the pass treats the two
        spellings interchangeably.
        """
        is_unsqueeze = node.target == exir_ops.edge.aten.unsqueeze_copy.default
        if not is_unsqueeze and node.target not in self._VIEW_OPS:
            return None
        inp = node.args[0]
        if not isinstance(inp, torch.fx.Node):
            return None
        in_shape = self._concrete_shape(inp)
        out_shape = self._concrete_shape(node)
        if in_shape is None or out_shape is None:
            return None
        if len(out_shape) != len(in_shape) + 1:
            return None
        if is_unsqueeze:
            dim = get_arg(node, "dim", int)
            pos = dim if dim >= 0 else dim + len(out_shape)
            if not 0 <= pos < len(out_shape) or out_shape[pos] != 1:
                return None
            return pos
        positions = self._find_extra_ones(out_shape, in_shape)
        if positions is None or len(positions) != 1:
            return None
        return positions[0]

    def _interleave_triple(
        self, node: torch.fx.Node
    ) -> tuple[int, int, torch.fx.Node, torch.fx.Node] | None:
        """Recognise a repeat_interleave and return (dim, scale, expand, view)."""
        if node not in self._interleave_cache:
            self._interleave_cache[node] = self._match_interleave_triple(node)
        return self._interleave_cache[node]

    def _match_interleave_triple(
        self, node: torch.fx.Node
    ) -> tuple[int, int, torch.fx.Node, torch.fx.Node] | None:
        """Match a repeat_interleave lowered to three shape operations.

        ``repeat_interleave(scale, dim)`` lowers to::

            unsqueeze(dim + 1) -> expand_copy(scale at dim + 1)
                               -> view_copy(merge dim, dim + 1)

        (e.g. torchaudio's Stretch2d). The triple is rank-preserving overall, so
        a permutation flows through it unchanged and only the dim it acts on has
        to be remapped -- unlike the merging view_copy on its own, which is not
        layout-invariant. Handling the three nodes as one unit also avoids having
        to pick an un-permuted position for the intermediate unit dim, a choice
        that would otherwise decide whether the merge stays legal.
        """
        pos = self._inserted_unit_dim(node)
        if pos is None or pos == 0 or len(node.users) != 1:
            return None
        dim = pos - 1

        expand_node = next(iter(node.users))
        if (
            expand_node.target != exir_ops.edge.aten.expand_copy.default
            or len(expand_node.users) != 1
        ):
            return None

        unsq_shape = self._concrete_shape(node)
        if unsq_shape is None:
            return None
        size = get_arg(expand_node, "size")
        if not isinstance(size, (list, tuple)) or len(size) != len(unsq_shape):
            return None
        size = list(size)
        if not all(isinstance(s, int) for s in size):
            return None
        # Every dim other than the inserted one must pass through untouched.
        if any(s != -1 and s != unsq_shape[k] for k, s in enumerate(size) if k != pos):
            return None
        scale = size[pos]
        if scale < 1:
            return None

        view_node = next(iter(expand_node.users))
        if view_node.target not in self._VIEW_OPS:
            return None
        in_shape = self._concrete_shape(cast(torch.fx.Node, node.args[0]))
        if in_shape is None:
            return None
        merged = list(in_shape)
        merged[dim] *= scale
        if self._concrete_shape(view_node) != merged:
            return None
        return dim, scale, expand_node, view_node

    def _adapt_permute_across_view(
        self, permute: list[int], node: torch.fx.Node
    ) -> list[int] | None:
        """Adjust a permutation across a squeeze/unsqueeze boundary.

        Adapts from input-rank to output-rank space (downstream direction).
        Returns the adjusted permutation, or None if not possible.
        """
        shapes = self._view_shapes(node)
        if shapes is None:
            return None
        in_shape, out_shape = shapes
        # ``permute`` must live in the view's input-rank space. It does not when
        # the view is reached by upstream traversal, where the permutation is
        # expressed at the view's output rank.
        if len(permute) != len(in_shape):
            return None

        if len(out_shape) > len(in_shape):
            # unsqueeze: insert an identity mapping at each added position.
            # Positions are ascending and index into the output, so inserting
            # them in order lands each one at its final index.
            positions = self._find_extra_ones(out_shape, in_shape)
            if positions is None:
                return None
            new_perm = list(permute)
            for index in positions:
                new_perm = [x + 1 if x >= index else x for x in new_perm]
                new_perm.insert(index, index)
            return new_perm

        if len(in_shape) > len(out_shape):
            positions = self._find_extra_ones(in_shape, out_shape)
            # Positions index into the node's input rank, which can differ from
            # the permutation's rank when the view is reached via upstream
            # traversal after an earlier rank change.
            if positions is None or positions[-1] >= len(permute):
                return None
            # A position is a POSITION in the tensor; the permutation VALUE at
            # that position is the logical dim being removed.
            squeezed_values = {permute[p] for p in positions}
            return [
                x - sum(1 for v in squeezed_values if v < x)
                for x in permute
                if x not in squeezed_values
            ]

        return None

    def call(self, graph_module: torch.fx.GraphModule) -> PassResult:  # noqa: C901
        self._sq_unsq_cache.clear()
        self._interleave_cache.clear()
        subgraphs_found: list[RemovePermutesAroundElementwiseOps.Subgraph] = []
        processed_nodes: set[torch.fx.Node] = set()
        for node in graph_module.graph.nodes:
            if node.target not in self._permute_targets:
                continue
            start_permute = self.get_permutation(node)
            if start_permute is None:
                continue
            # Expected end permutation for the subgraph.
            end_permute = [start_permute.index(i) for i in range(len(start_permute))]

            for user in node.users:
                if (
                    not self.is_node_permutable(user)
                    and self._interleave_triple(user) is None
                ):
                    continue
                subgraph = self.Subgraph(start_permute, end_permute, node.target)
                if self.visit(user, subgraph, processed_nodes):
                    subgraphs_found.append(subgraph)
                    for n in subgraph.nodes:
                        processed_nodes.add(n)

        modified = False
        for subgraph in subgraphs_found:
            if self.permute_subgraph(subgraph):
                modified = True

        if modified:
            graph_module.graph.eliminate_dead_code()
            graph_module.recompile()
            return super().call(graph_module)

        return PassResult(graph_module, False)

    def visit(  # noqa: C901
        self,
        node: torch.fx.Node,
        subgraph: Subgraph,
        processed_nodes: set[torch.fx.Node],
        current_end_permute: list[int] | None = None,
        current_start_permute: list[int] | None = None,
    ) -> bool:
        if current_end_permute is None:
            current_end_permute = subgraph.end_permute
        if current_start_permute is None:
            current_start_permute = subgraph.start_permute

        if node in subgraph.nodes:
            return True
        if node in processed_nodes:
            return False
        # Explicit unsqueeze nodes are not generally permutable after shape-op
        # canonicalization, but a guarded repeat_interleave triple is handled as
        # one rank-preserving unit.
        triple = self._interleave_triple(node)
        if triple is None and not self.is_node_permutable(node):
            return False
        if triple is not None:
            inp = node.args[0] if node.args else None
            if not isinstance(inp, torch.fx.Node):
                return False
            in_shape = self._concrete_shape(inp)
            if in_shape is None or len(current_start_permute) != len(in_shape):
                return False
        else:
            # A permutable op can still change rank via broadcasting (e.g.
            # [8, 1] + [1, 1, 1] -> [1, 8, 1]), which would leave the node
            # carrying a permutation of the wrong rank. Squeeze/unsqueeze views
            # are exempt because _adapt_permute_across_view checks their ranks.
            if not self._is_squeeze_unsqueeze_view(node):
                node_shape = getattr(node.meta.get("val"), "shape", None)
                if node_shape is not None and len(node_shape) != len(
                    current_start_permute
                ):
                    return False
        subgraph.nodes.add(node)
        subgraph.node_end_permute[node] = current_end_permute
        subgraph.node_start_permute[node] = current_start_permute

        # A repeat_interleave triple is absorbed whole: its interior nodes are
        # not layout-invariant individually, but the triple is rank-preserving.
        users_source = node
        if triple is not None:
            users_source = self._absorb_interleave(
                node,
                triple,
                subgraph,
                processed_nodes,
                current_end_permute,
                current_start_permute,
            )
            if users_source is None:
                return False

        # If this is a squeeze/unsqueeze view, adapt permutations for
        # traversal across the rank change boundary.
        downstream_end = current_end_permute
        downstream_start = current_start_permute
        if triple is None and self._is_squeeze_unsqueeze_view(node):
            # Adapt start permute for downstream (input-rank → output-rank)
            adapted_start = self._adapt_permute_across_view(current_start_permute, node)
            if adapted_start is None:
                return False
            downstream_start = adapted_start

            # Derive end permute as the inverse of adapted start to ensure
            # consistency.  Computing start and end independently via
            # _adapt_permute_across_view can produce mismatched results for
            # squeeze views because the formula differs for "forward" vs
            # "inverse" permutations.
            downstream_end = [adapted_start.index(i) for i in range(len(adapted_start))]

        # Traverse downstream:
        for user in users_source.users:
            if user.target in self._permute_targets:
                if user.target != subgraph.permute_target:
                    return False
                user_perm = self.get_permutation(user)
                if user_perm == downstream_end:
                    subgraph.edges_out.add((users_source, user))
                else:
                    # Non-matching permute: keep it and fold the start permute into it
                    # rather than discarding the region.
                    if user_perm is None or len(user_perm) != len(downstream_start):
                        return False
                    subgraph.edges_out_to_update.add(
                        (
                            users_source,
                            user,
                            tuple(downstream_start[d] for d in user_perm),
                        )
                    )
            elif user.op == "output":
                return False
            elif self.can_propagate is not None and not self.can_propagate(user):
                return False
            elif self._is_permutation_sink_view(user):
                # The tensor's element order is invariant at this reshape, but
                # its output shape can still carry broadcast-axis meaning.
                if not self._sink_users_are_layout_invariant(user):
                    return False
                continue
            elif not self.visit(
                user, subgraph, processed_nodes, downstream_end, downstream_start
            ):
                return False

        # Traverse upstream:
        for inp in node.all_input_nodes:
            if inp.target in self._permute_targets:
                if (
                    inp.target != subgraph.permute_target
                    or self.get_permutation(inp) != current_start_permute
                ):
                    return False
                subgraph.edges_in.add((inp, node))
            elif (inp_val := inp.meta.get("val")) is not None and inp_val.numel() == 1:
                # A numel-1 input (per-tensor quant scale / zero_point, scalar
                # constant, ...) is layout-invariant: it broadcasts identically
                # under any permutation, so it needs no compensating permute and
                # stays wired directly. Notably this keeps lifted per-tensor
                # qparam placeholders as placeholders, which lowering requires.
                continue
            elif self._is_constant(inp):
                const_rank = self._get_node_rank(inp)
                permute_rank = len(current_end_permute)
                if const_rank is None:
                    return False
                if const_rank > permute_rank:
                    return False
                if const_rank < permute_rank and inp.meta.get("val") is None:
                    return False
                subgraph.constant_edges_in.add((inp, node))
            elif not self.visit(
                inp,
                subgraph,
                processed_nodes,
                current_end_permute,
                current_start_permute,
            ):
                return False

        return True

    def _absorb_interleave(
        self,
        head: torch.fx.Node,
        triple: tuple[int, int, torch.fx.Node, torch.fx.Node],
        subgraph: Subgraph,
        processed_nodes: set[torch.fx.Node],
        current_end_permute: list[int],
        current_start_permute: list[int],
    ) -> torch.fx.Node | None:
        """Add a matched interleave's interior nodes and return its tail."""
        _, _, expand_node, view_node = triple
        if expand_node in processed_nodes or view_node in processed_nodes:
            return None
        for interior in (expand_node, view_node):
            subgraph.nodes.add(interior)
            subgraph.node_end_permute[interior] = current_end_permute
            subgraph.node_start_permute[interior] = current_start_permute
        subgraph.interleaves[head] = triple
        return view_node

    def _is_constant(self, node: torch.fx.Node) -> bool:
        """Check if a node's value is available at compile time.
        Only considers direct constants (get_attr, parameter/buffer/constant
        placeholders, full ops producing scalar constants) — does not recurse
        into call_function chains to avoid stack overflow on deep graphs."""
        if node.op == "get_attr":
            return True
        if node.op == "placeholder":
            target = str(node.target)
            return target.startswith(("b_", "p_", "c_"))
        # full.default creates scalar constants (e.g. epsilon in LayerNorm)
        if (
            node.op == "call_function"
            and node.target == exir_ops.edge.aten.full.default
        ):
            return True
        return False

    def _get_node_rank(self, node: torch.fx.Node) -> int | None:
        """Return the tensor rank of a node's output, or None if unknown."""
        val = node.meta.get("val")
        if val is None:
            return None
        return len(val.shape)

    @staticmethod
    def _is_pointwise(target) -> bool:
        """Check if a target op is tagged as pointwise in ATen."""
        op = getattr(target, "_op", None)
        if op is not None and hasattr(op, "tags"):
            return torch.Tag.pointwise in op.tags
        return False

    def is_node_permutable(self, node: torch.fx.Node) -> bool:
        if self.can_propagate is not None and not self.can_propagate(node):
            return False
        if node.target in self._PAD_OPS and not self._is_constant_pad(node):
            return False
        if node.target in self._permutable_ops:
            if node.target in (
                exir_ops.edge.aten.mean.dim,
                exir_ops.edge.aten.sum.dim_IntList,
            ):
                if not get_arg(node, "keepdim", bool):
                    return False
            return True
        if self._is_squeeze_unsqueeze_view(node):
            return True
        return self._is_pointwise(node.target)

    def _is_constant_pad(self, node: torch.fx.Node) -> bool:
        if len(node.args) < 2:
            return False

        pad = node.args[1]
        if not isinstance(pad, (list, tuple)):
            return False

        if len(pad) % 2 != 0:
            return False

        if node.target == exir_ops.edge.aten.pad.default:
            mode = node.args[2] if len(node.args) > 2 else node.kwargs.get("mode")
            if mode not in (None, "constant"):
                return False

        return True

    def _removes_a_permute(self, subgraph: Subgraph) -> bool:
        """Whether rewriting this region reduces the number of permutes."""
        if subgraph.edges_out:
            return True
        rewired = set(subgraph.edges_in)
        for permute, _ in subgraph.edges_in:
            if all((permute, user) in rewired for user in permute.users):
                return True
        return False

    def permute_subgraph(self, subgraph: Subgraph) -> bool:  # noqa: C901
        # Ensure that the subgraph's edges have not been modified by an earlier rewrite before applying changes.
        if not self._subgraph_edges_are_current(subgraph):
            return False

        # Folding an end permute only pays for itself if some permute goes away.
        # Otherwise the region is rewritten for nothing, and the composed
        # permutation is a worse fusion candidate for the passes downstream.
        if subgraph.edges_out_to_update and not self._removes_a_permute(subgraph):
            return False

        # Nodes belonging to a repeat_interleave triple are rewritten as a unit
        # below, so they must skip the per-node dim handling and the view rank
        # check (the triple's interior ranks intentionally differ from the
        # region's permutation rank).
        interleave_nodes: set[torch.fx.Node] = set()
        for head, (_, _, expand_node, view_node) in subgraph.interleaves.items():
            interleave_nodes.update((head, expand_node, view_node))
            perm = subgraph.node_start_permute.get(head, subgraph.start_permute)
            inp = head.args[0] if head.args else None
            if not isinstance(inp, torch.fx.Node):
                return False
            in_shape = self._concrete_shape(inp)
            if in_shape is None or len(perm) != len(in_shape):
                return False

        # Validate: every view_copy node's permutation rank must match its
        # input tensor rank.  A mismatch can occur when a squeeze/unsqueeze
        # view is reached via upstream traversal with a permutation that was
        # already adapted to a different rank.  Applying the optimisation in
        # this case would produce an invalid graph, so skip the subgraph.
        for node in subgraph.nodes:
            if node in interleave_nodes:
                continue
            if node.target in self._VIEW_OPS:
                perm = subgraph.node_start_permute.get(node, subgraph.start_permute)
                inp = node.args[0]
                if isinstance(inp, torch.fx.Node) and inp.meta.get("val") is not None:
                    if len(perm) != len(inp.meta["val"].shape):
                        return False

        # Handle dimension related node arguments FIRST, before
        # bypassing permutes (which changes node inputs/metadata).
        for node in subgraph.nodes:
            if node in interleave_nodes:
                continue
            node_start_perm = subgraph.node_start_permute.get(
                node, subgraph.start_permute
            )
            if node.target == exir_ops.edge.aten.cat.default:
                self.update_cat(node, node_start_perm)
            elif node.target in (
                exir_ops.edge.aten.mean.dim,
                exir_ops.edge.aten.sum.dim_IntList,
            ):
                self.update_mean_dim(node, node_start_perm)
            elif node.target == exir_ops.edge.aten.slice_copy.Tensor:
                self.update_slice_copy(node, node_start_perm)
            elif node.target in self._PAD_OPS:
                self.update_pad(node, node_start_perm)
            elif node.target in self._VIEW_OPS:
                self.update_view_copy(node, node_start_perm)

        for head, triple in subgraph.interleaves.items():
            self.update_interleave(
                head,
                triple,
                subgraph.node_start_permute.get(head, subgraph.start_permute),
            )

        # Skip incoming permutes.
        for inp, out in subgraph.edges_in:
            assert inp.target in self._permute_targets
            if len(inp.args) >= 1:
                out.replace_input_with(inp, cast(torch.fx.Node, inp.args[0]))
            else:
                out.replace_input_with(inp, cast(torch.fx.Node, inp.kwargs["input"]))

        # Insert compensating permute on constant inputs.
        for const_node, user_node in subgraph.constant_edges_in:
            graph = const_node.graph
            const_rank = self._get_node_rank(const_node)
            # Use the node-specific end_permute for the correct rank
            node_end_perm = subgraph.node_end_permute.get(
                user_node, subgraph.end_permute
            )
            permute_rank = len(node_end_perm)

            with graph.inserting_after(const_node):
                if const_rank is not None and const_rank == permute_rank:
                    new_node = graph.create_node(
                        "call_function",
                        subgraph.permute_target,
                        args=(const_node, node_end_perm),
                    )
                elif (
                    const_rank is not None
                    and const_rank < permute_rank
                    and const_node.meta.get("val") is not None
                ):
                    # Broadcasting widens the constant to the region's rank
                    # before the permutation applies.
                    original_shape = list(const_node.meta["val"].shape)
                    padded = [1] * (permute_rank - const_rank) + original_shape
                    target_shape = [padded[dim] for dim in node_end_perm]

                    # Where each non-unit axis ends up. Unit axes carry no
                    # elements, so only the order of these decides whether the
                    # permutation rearranges data or merely reshapes.
                    destinations = [
                        node_end_perm.index(axis)
                        for axis, size in enumerate(padded)
                        if size != 1
                    ]
                    if destinations == sorted(destinations):
                        # Only unit extents moved, so this is a pure reshape and
                        # a view says it exactly -- and says it for free, since
                        # view_copy later becomes a memory.view alias.
                        new_node = graph.create_node(
                            "call_function",
                            exir_ops.edge.aten.view_copy.default,
                            args=(const_node, target_shape),
                        )
                    else:
                        # Reordering a non-unit extent moves data. A view would
                        # reinterpret the strides and read different elements,
                        # so widen with a view and permute at full rank.
                        widened = graph.create_node(
                            "call_function",
                            exir_ops.edge.aten.view_copy.default,
                            args=(const_node, padded),
                        )
                        with graph.inserting_after(widened):
                            new_node = graph.create_node(
                                "call_function",
                                subgraph.permute_target,
                                args=(widened, node_end_perm),
                            )
                else:
                    continue
            user_node.replace_input_with(const_node, new_node)

        # Skip outgoing permutes.
        for inp, out in subgraph.edges_out:
            assert out.target in self._permute_targets
            out.replace_all_uses_with(inp)

        # Update outgoing permutes that can't be eliminated.
        for _, out, new_permutation in subgraph.edges_out_to_update:
            assert out.target in PERMUTE_COPY_TARGETS
            set_arg(out, "dims", list(new_permutation))

        return True

    def _subgraph_edges_are_current(self, subgraph: Subgraph) -> bool:
        """Return false if an earlier rewrite invalidated this candidate."""
        for inp, out in subgraph.edges_in:
            if (
                inp.target not in self._permute_targets
                or inp not in out.all_input_nodes
            ):
                return False

            # edges_out_to_update can rewrite a permute in place, leaving it wired.
            if self.get_permutation(inp) != subgraph.node_start_permute.get(
                out, subgraph.start_permute
            ):
                return False

        for inp, out in subgraph.edges_out:
            if out.target not in self._permute_targets or out not in inp.users:
                return False

        for inp, out, _ in subgraph.edges_out_to_update:
            if out.target not in PERMUTE_COPY_TARGETS or out not in inp.users:
                return False

        for const_node, user_node in subgraph.constant_edges_in:
            if const_node not in user_node.all_input_nodes:
                return False

        for head, (_, _, expand_node, view_node) in subgraph.interleaves.items():
            if (
                len(head.users) != 1
                or len(expand_node.users) != 1
                or expand_node not in head.users
                or view_node not in expand_node.users
            ):
                return False

        return True

    def update_interleave(
        self,
        head: torch.fx.Node,
        triple: tuple[int, int, torch.fx.Node, torch.fx.Node],
        start_permute: list[int],
    ) -> None:
        """Retarget a repeat_interleave triple at the un-permuted layout.

        After the boundary permutes are removed the triple's input is in the
        original layout, so the dim it interleaves moves from ``dim`` to
        ``start_permute[dim]`` and all three shape arguments are rebuilt there.
        """
        dim, scale, expand_node, view_node = triple
        inp = cast(torch.fx.Node, head.args[0])
        in_shape = [int(d) for d in inp.meta["val"].shape]
        inverse_permute = [start_permute.index(i) for i in range(len(start_permute))]
        unpermuted_in = [in_shape[inverse_permute[i]] for i in range(len(in_shape))]
        target_dim = start_permute[dim]

        if head.target == exir_ops.edge.aten.unsqueeze_copy.default:
            set_arg(head, "dim", target_dim + 1)
        else:
            unsqueezed = list(unpermuted_in)
            unsqueezed.insert(target_dim + 1, 1)
            set_arg(head, "size", unsqueezed)

        expand_size = list(unpermuted_in)
        expand_size.insert(target_dim + 1, scale)
        set_arg(expand_node, "size", expand_size)

        merged = list(unpermuted_in)
        merged[target_dim] *= scale
        set_arg(view_node, "size", merged)

    def update_cat(self, node: torch.fx.Node, start_permute: list[int]) -> None:
        dim = get_arg(node, "dim", int)
        set_arg(node, "dim", start_permute[dim])

    def update_mean_dim(self, node: torch.fx.Node, start_permute: list[int]) -> None:
        dims = get_arg(node, "dim")
        set_arg(node, "dim", [start_permute[d] for d in cast(list[int], dims)])

    def update_slice_copy(self, node: torch.fx.Node, start_permute: list[int]) -> None:
        dim = get_arg(node, "dim", int)
        set_arg(node, "dim", start_permute[dim])

    def update_pad(self, node: torch.fx.Node, start_permute: list[int]) -> None:
        pad = list(cast(list[int], node.args[1]))
        rank = len(start_permute)
        pad_pairs = [[0, 0] for _ in range(rank)]
        for pair_idx, pair_start in enumerate(range(0, len(pad), 2)):
            dim = rank - 1 - pair_idx
            pad_pairs[dim] = [pad[pair_start], pad[pair_start + 1]]

        remapped_pairs = [pad_pairs[start_permute.index(dim)] for dim in range(rank)]
        remapped_pad = []
        for pair in reversed(remapped_pairs):
            remapped_pad.extend(pair)

        while len(remapped_pad) > 2 and remapped_pad[-2:] == [0, 0]:
            remapped_pad = remapped_pad[:-2]

        node.update_arg(1, remapped_pad)

    def update_view_copy(self, node: torch.fx.Node, start_permute: list[int]) -> None:
        """Adjust view_copy shape arg after permute removal.

        After removing the start permute, the view's input is in the original
        (un-permuted) layout. Recompute the view's target shape accordingly.
        """
        shapes = self._view_shapes(node)
        if shapes is None:
            return
        in_shape, out_shape = shapes

        # Compute un-permuted input shape
        inverse_permute = [start_permute.index(i) for i in range(len(start_permute))]
        unpermuted_in = [in_shape[inverse_permute[i]] for i in range(len(in_shape))]

        if len(out_shape) > len(in_shape):
            # unsqueeze: the added unit dims keep their output positions in the
            # un-permuted layout too
            positions = self._find_extra_ones(out_shape, in_shape)
            if positions is None:
                return
            new_shape = list(unpermuted_in)
            for index in positions:
                new_shape.insert(index, 1)
            node.update_arg(1, new_shape)
        elif len(in_shape) > len(out_shape):
            # squeeze: map each removed dim from permuted to un-permuted space,
            # deleting from the back so earlier indices stay valid
            positions = self._find_extra_ones(in_shape, out_shape)
            if positions is None or positions[-1] >= len(start_permute):
                return
            new_shape = list(unpermuted_in)
            for index in sorted((start_permute[p] for p in positions), reverse=True):
                del new_shape[index]
            node.update_arg(1, new_shape)

    def get_permutation(self, permute_node: torch.fx.Node) -> list[int] | None:
        assert permute_node.target in self._permute_targets
        raw_permute: list[int]
        if len(permute_node.args) >= 2:
            raw_permute = list(cast(list[int], permute_node.args[1]))
        else:
            raw_dims = permute_node.kwargs.get("dims", permute_node.kwargs.get("dim"))
            if raw_dims is None:
                return None
            raw_permute = list(cast(list[int], raw_dims))

        rank = len(raw_permute)
        normalized_permute = [d + rank if d < 0 else d for d in raw_permute]

        if not all(0 <= d < rank for d in normalized_permute):
            return None
        if sorted(normalized_permute) != list(range(rank)):
            return None
        return normalized_permute
