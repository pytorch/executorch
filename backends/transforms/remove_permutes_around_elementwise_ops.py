# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
# Copyright 2026 Arm Limited and/or its affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-unsafe

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import cast

import torch
import torch.fx
from executorch.backends.transforms.channels_last_layout import (
    ATEN_PERMUTE_COPY,
    is_layout_copy,
    is_permute_copy,
    LAYOUT_PERMUTE_COPY,
    PERMUTE_COPY_TARGETS,
)
from executorch.backends.transforms.permute_pass_utils import get_arg, set_arg
from executorch.exir import ExportedProgram
from executorch.exir.dialects._ops import ops as exir_ops
from executorch.exir.pass_base import ExportPass, PassResult
from torch.fx.node import Target


class RemovePermutesAroundElementwiseOps(ExportPass):
    """
    Looks for subgraphs of elementwise ops sandwiched between permutes and removes those
    permutes if possible.
    Allows special handling for certain non-elementwise ops that can be easily updated
    based on the permute's parameter such as mean, cat, and slice.
    The repeat_interleave idiom (unsqueeze -> expand_copy -> merging view_copy) is
    recognised as a single rank-preserving unit; see _interleave_triple.

    ``extra_permutable_ops`` must be layout-equivariant without argument remapping.
    Layout-boundary propagation applies only to layout-owned dialect copies.
    ``layout_pad_target`` opts into retargeting rank-4 constant pads after their
    layout-dependent pad argument has been remapped.
    """

    @dataclass()
    class Subgraph:
        start_permute: list[int]
        end_permute: list[int]
        # Nodes in the subgraph, does not include permutes.
        nodes: set[torch.fx.Node] = field(default_factory=set)
        # Incoming edges to the subgraph from permute nodes.
        edges_in: set[tuple[torch.fx.Node, torch.fx.Node]] = field(default_factory=set)
        # Outgoing edges of the subgraph to permute nodes.
        edges_out: set[tuple[torch.fx.Node, torch.fx.Node]] = field(default_factory=set)
        # Incoming edges from constant nodes that need a compensating permute.
        constant_edges_in: set[tuple[torch.fx.Node, torch.fx.Node]] = field(
            default_factory=set
        )
        # Open boundaries used only for structural layout-copy propagation.
        input_boundaries: set[tuple[torch.fx.Node, torch.fx.Node, tuple[int, ...]]] = (
            field(default_factory=set)
        )
        output_boundaries: set[tuple[torch.fx.Node, torch.fx.Node, tuple[int, ...]]] = (
            field(default_factory=set)
        )
        layout_region: bool = False
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
        exported_program: ExportedProgram | None = None,
        allow_layout_boundary_propagation: bool = False,
        layout_pad_target: Target | None = None,
        can_propagate: Callable[[torch.fx.Node], bool] | None = None,
    ) -> None:
        super().__init__()
        self.exported_program = exported_program
        self.allow_layout_boundary_propagation = allow_layout_boundary_propagation
        self.layout_pad_target = layout_pad_target
        self.can_propagate = can_propagate
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
        exir_ops.edge.channels_last.constant_pad_nd.default,
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
        contiguous regardless of which axis holds it). A permutation propagating
        into it therefore simply dies, so the region can terminate here with no
        compensating permute.
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
        permute_nodes = [
            node for node in graph_module.graph.nodes if is_permute_copy(node)
        ]
        for node in permute_nodes:
            start_permute = self.get_permutation(node)
            if start_permute is None:
                continue
            layout_region = self.allow_layout_boundary_propagation and is_layout_copy(
                node
            )
            # Expected end permutation for the subgraph.
            end_permute = [start_permute.index(i) for i in range(len(start_permute))]

            if layout_region:
                users = list(node.users)
                if users and all(
                    self.is_node_permutable(user)
                    or self._interleave_triple(user) is not None
                    for user in users
                ):
                    subgraph = self.Subgraph(
                        start_permute,
                        end_permute,
                        layout_region=True,
                    )
                    if all(
                        self.visit(user, subgraph, processed_nodes) for user in users
                    ):
                        subgraphs_found.append(subgraph)
                        processed_nodes.update(subgraph.nodes)
                # Layout boundary movement is atomic across every direct user.
                continue

            # Try direct users first (same-rank matching)
            for user in node.users:
                if (
                    not self.is_node_permutable(user)
                    and self._interleave_triple(user) is None
                ):
                    continue
                subgraph = self.Subgraph(
                    start_permute,
                    end_permute,
                    layout_region=layout_region,
                )
                if self.visit(user, subgraph, processed_nodes):
                    subgraphs_found.append(subgraph)
                    for n in subgraph.nodes:
                        processed_nodes.add(n)

            # Also try: permute → view(squeeze/unsqueeze) → chain → ...
            # If the permute's sole user is a squeeze/unsqueeze view,
            # adapt the permutation across the view and search for a
            # matching end permute at the new rank.
            users = list(node.users.keys())
            if (
                len(users) == 1
                and self._is_squeeze_unsqueeze_view(users[0])
                and node not in processed_nodes
            ):
                view_node = users[0]
                adapted_start = self._adapt_permute_across_view(
                    start_permute, view_node
                )
                if adapted_start is not None:
                    adapted_end = [
                        adapted_start.index(i) for i in range(len(adapted_start))
                    ]
                    for view_user in view_node.users:
                        if (
                            not self.is_node_permutable(view_user)
                            and self._interleave_triple(view_user) is None
                        ):
                            continue
                        subgraph = self.Subgraph(
                            adapted_start,
                            adapted_end,
                            layout_region=layout_region,
                        )
                        # Include the view in the subgraph
                        subgraph.nodes.add(view_node)
                        subgraph.node_end_permute[view_node] = adapted_end
                        # Use the ORIGINAL start_permute for the view node
                        # so update_view_copy can remap its shape correctly
                        subgraph.node_start_permute[view_node] = start_permute
                        # The start permute feeds into the view
                        subgraph.edges_in.add((node, view_node))
                        if self.visit(
                            view_user,
                            subgraph,
                            processed_nodes,
                            adapted_end,
                            adapted_start,
                        ):
                            subgraphs_found.append(subgraph)
                            for n in subgraph.nodes:
                                processed_nodes.add(n)

        if self.allow_layout_boundary_propagation:
            for node in permute_nodes:
                end_permute = self.get_permutation(node)
                if end_permute is None:
                    continue
                if not is_layout_copy(node):
                    continue
                producer = node.args[0] if node.args else None
                if not isinstance(producer, torch.fx.Node):
                    continue
                if (
                    not self.is_node_permutable(producer)
                    and self._interleave_triple(producer) is None
                ):
                    continue
                start_permute = [end_permute.index(i) for i in range(len(end_permute))]
                subgraph = self.Subgraph(
                    start_permute,
                    end_permute,
                    layout_region=True,
                )
                if self.visit(producer, subgraph, processed_nodes):
                    subgraphs_found.append(subgraph)
                    processed_nodes.update(subgraph.nodes)

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
            if user.target in PERMUTE_COPY_TARGETS:
                user_perm = self.get_permutation(user)
                if user_perm == downstream_end:
                    subgraph.edges_out.add((users_source, user))
                else:
                    # Check if permute → view(squeeze/unsqueeze) forms an
                    # end boundary at a different rank.
                    user_users = list(user.users.keys())
                    if len(user_users) == 1 and self._is_squeeze_unsqueeze_view(
                        user_users[0]
                    ):
                        view_after: torch.fx.Node = user_users[0]
                        # Adapt the start permute across the view and derive
                        # the expected end permute as its inverse.
                        adapted_start_after = self._adapt_permute_across_view(
                            downstream_start, view_after
                        )
                        if adapted_start_after is not None:
                            adapted = [
                                adapted_start_after.index(i)
                                for i in range(len(adapted_start_after))
                            ]
                            if user_perm == adapted:
                                # Include both the permute and the view as end edges
                                subgraph.edges_out.add((users_source, user))
                                # Mark the view for inclusion so it gets preserved
                                continue
                    return False
            elif user.op == "output":
                if (
                    not self.allow_layout_boundary_propagation
                    or not subgraph.layout_region
                ):
                    return False
                subgraph.output_boundaries.add(
                    (users_source, user, tuple(downstream_start))
                )
            elif self._is_permutation_sink_view(user):
                # The permutation dies at this reshape (see
                # _is_permutation_sink_view), so terminate the region here with
                # no compensating permute and no further downstream traversal.
                # Checked before the rank-change handling below: a sink always
                # terminates cleanly, whereas crossing it would leave the region
                # hunting for an end permute that layout-invariance made moot.
                continue
            elif (
                self.allow_layout_boundary_propagation
                and subgraph.layout_region
                and self.can_propagate is not None
                and not self.can_propagate(user)
            ):
                subgraph.output_boundaries.add(
                    (users_source, user, tuple(downstream_start))
                )
            elif not self.visit(
                user, subgraph, processed_nodes, downstream_end, downstream_start
            ):
                return False

        # Traverse upstream:
        for inp in node.all_input_nodes:
            if inp.target in PERMUTE_COPY_TARGETS:
                if self.get_permutation(inp) != current_start_permute:
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
            elif self._is_user_input(inp):
                if (
                    not self.allow_layout_boundary_propagation
                    or not subgraph.layout_region
                    or self._get_node_rank(inp) != len(current_end_permute)
                ):
                    return False
                subgraph.input_boundaries.add((inp, node, tuple(current_end_permute)))
            elif (
                self.allow_layout_boundary_propagation
                and subgraph.layout_region
                and self.can_propagate is not None
                and not self.can_propagate(inp)
                and self._get_node_rank(inp) == len(current_end_permute)
            ):
                subgraph.input_boundaries.add((inp, node, tuple(current_end_permute)))
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

    def _is_user_input(self, node: torch.fx.Node) -> bool:
        if node.op != "placeholder":
            return False
        if self.exported_program is not None:
            return node.name in self.exported_program.graph_signature.user_inputs
        return not self._is_constant(node)

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

    def permute_subgraph(self, subgraph: Subgraph) -> bool:  # noqa: C901
        # Ensure that the subgraph's edges have not been modified by an earlier rewrite before applying changes.
        if not self._subgraph_edges_are_current(subgraph):
            return False
        if subgraph.layout_region and (
            not self._boundary_permutations_are_layout_copies(subgraph)
            or not self._boundary_rewrite_is_cost_safe(subgraph)
        ):
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
            elif node.target in (
                exir_ops.edge.aten._softmax.default,
                exir_ops.edge.aten.softmax.int,
            ):
                self.update_dim(node, node_start_perm)
            elif node.target in self._PAD_OPS:
                self.update_pad(node, node_start_perm, subgraph.layout_region)
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
            assert inp.target in PERMUTE_COPY_TARGETS
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
                        (
                            LAYOUT_PERMUTE_COPY
                            if subgraph.layout_region
                            else ATEN_PERMUTE_COPY
                        ),
                        args=(const_node, node_end_perm),
                    )
                    new_node.meta = {}
                elif (
                    const_rank is not None
                    and const_rank < permute_rank
                    and const_node.meta.get("val") is not None
                ):
                    original_shape = list(const_node.meta["val"].shape)
                    padded = [1] * (permute_rank - const_rank) + original_shape
                    target_shape = [padded[d] for d in node_end_perm]
                    target_shape = target_shape[permute_rank - const_rank :]
                    new_node = graph.create_node(
                        "call_function",
                        exir_ops.edge.aten.view_copy.default,
                        args=(const_node, target_shape),
                    )
                else:
                    continue
            user_node.replace_input_with(const_node, new_node)

        # Skip outgoing permutes.
        for inp, out in subgraph.edges_out:
            assert out.target in PERMUTE_COPY_TARGETS
            out.replace_all_uses_with(inp)

        self._insert_input_boundary_permutations(subgraph)
        self._insert_output_boundary_permutations(subgraph)

        return True

    def _subgraph_edges_are_current(self, subgraph: Subgraph) -> bool:  # noqa: C901
        """Return false if an earlier rewrite invalidated this candidate."""
        for inp, out in subgraph.edges_in:
            if inp.target not in PERMUTE_COPY_TARGETS or inp not in out.all_input_nodes:
                return False

        for inp, out in subgraph.edges_out:
            if out.target not in PERMUTE_COPY_TARGETS or out not in inp.users:
                return False

        for const_node, user_node in subgraph.constant_edges_in:
            if const_node not in user_node.all_input_nodes:
                return False

        for input_node, user_node, _ in subgraph.input_boundaries:
            if input_node not in user_node.all_input_nodes:
                return False
            future_occurrences = self._node_argument_count(user_node, input_node)
            future_occurrences += sum(
                self._node_argument_count(user_node, permute)
                for permute, user in subgraph.edges_in
                if user is user_node
                and len(permute.args) >= 1
                and permute.args[0] is input_node
            )
            if future_occurrences != 1:
                return False

        for producer, output_node, _ in subgraph.output_boundaries:
            if producer not in output_node.all_input_nodes:
                return False
            future_occurrences = self._node_argument_count(output_node, producer)
            future_occurrences += sum(
                self._node_argument_count(output_node, permute)
                for source, permute in subgraph.edges_out
                if source is producer
            )
            if future_occurrences != 1:
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

    @staticmethod
    def _node_argument_count(node: torch.fx.Node, target: torch.fx.Node) -> int:
        count = 0

        def visit(argument):
            nonlocal count
            if argument is target:
                count += 1
            return argument

        torch.fx.map_arg((node.args, node.kwargs), visit)
        return count

    def _boundary_permutations_are_layout_copies(self, subgraph: Subgraph) -> bool:
        return all(is_layout_copy(permute) for permute, _ in subgraph.edges_in) and all(
            is_layout_copy(permute) for _, permute in subgraph.edges_out
        )

    def _boundary_rewrite_is_cost_safe(self, subgraph: Subgraph) -> bool:
        for const_node, user_node in subgraph.constant_edges_in:
            node_end_perm = subgraph.node_end_permute.get(
                user_node, subgraph.end_permute
            )
            if self._constant_transform_requires_data_copy(const_node, node_end_perm):
                return False

        removed_copies = {
            permute
            for permute, _ in subgraph.edges_in
            if all(user in subgraph.nodes for user in permute.users)
        } | {permute for _, permute in subgraph.edges_out}
        new_copy_sources = {
            (input_node, permutation)
            for input_node, _, permutation in subgraph.input_boundaries
        } | {
            (producer, permutation)
            for producer, _, permutation in subgraph.output_boundaries
        }

        if not new_copy_sources:
            return True

        removed_bytes = [self._static_tensor_bytes(node) for node in removed_copies]
        new_bytes = [self._static_tensor_bytes(node) for node, _ in new_copy_sources]
        if any(size is None for size in removed_bytes + new_bytes):
            return False
        return sum(cast(int, size) for size in new_bytes) <= sum(
            cast(int, size) for size in removed_bytes
        )

    def _constant_transform_requires_data_copy(
        self, node: torch.fx.Node, permutation: list[int]
    ) -> bool:
        val = node.meta.get("val")
        if not isinstance(val, torch.Tensor):
            return True
        shape = list(val.shape)
        if len(shape) > len(permutation) or not all(
            isinstance(dim, int) for dim in shape
        ):
            return True
        rank_difference = len(permutation) - len(shape)
        padded_shape = [1] * rank_difference + shape
        output_shape = [padded_shape[dim] for dim in permutation]
        if any(size != 1 for size in output_shape[:rank_difference]):
            return True
        old_order = [dim for dim, size in enumerate(padded_shape) if size != 1]
        new_order = [dim for dim, size in zip(permutation, output_shape) if size != 1]
        return old_order != new_order

    @staticmethod
    def _static_tensor_bytes(node: torch.fx.Node) -> int | None:
        val = node.meta.get("val")
        if not isinstance(val, torch.Tensor) or not all(
            isinstance(dim, int) for dim in val.shape
        ):
            return None
        return val.numel() * val.element_size()

    def _insert_input_boundary_permutations(self, subgraph: Subgraph) -> None:
        if not subgraph.input_boundaries:
            return
        assert subgraph.layout_region
        groups: dict[tuple[torch.fx.Node, tuple[int, ...]], list[torch.fx.Node]] = {}
        for input_node, user_node, permutation in subgraph.input_boundaries:
            groups.setdefault((input_node, permutation), []).append(user_node)

        graph = next(iter(subgraph.input_boundaries))[0].graph
        node_order = {node: index for index, node in enumerate(graph.nodes)}
        for (input_node, permutation), users in groups.items():
            first_user = min(users, key=node_order.__getitem__)
            with input_node.graph.inserting_before(first_user):
                new_permute = input_node.graph.call_function(
                    LAYOUT_PERMUTE_COPY,
                    args=(input_node, list(permutation)),
                )
            new_permute.meta = dict(input_node.meta)
            for user in users:
                user.replace_input_with(input_node, new_permute)

    def _insert_output_boundary_permutations(self, subgraph: Subgraph) -> None:
        if not subgraph.output_boundaries:
            return
        assert subgraph.layout_region
        groups: dict[tuple[torch.fx.Node, tuple[int, ...]], list[torch.fx.Node]] = {}
        for producer, output_node, permutation in subgraph.output_boundaries:
            groups.setdefault((producer, permutation), []).append(output_node)

        graph = next(iter(subgraph.output_boundaries))[0].graph
        node_order = {node: index for index, node in enumerate(graph.nodes)}
        for (producer, permutation), outputs in groups.items():
            first_output = min(outputs, key=node_order.__getitem__)
            with producer.graph.inserting_before(first_output):
                new_permute = producer.graph.call_function(
                    LAYOUT_PERMUTE_COPY,
                    args=(producer, list(permutation)),
                )
            new_permute.meta = dict(producer.meta)
            for output in outputs:
                output.replace_input_with(producer, new_permute)

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

    def update_dim(self, node: torch.fx.Node, start_permute: list[int]) -> None:
        dim = get_arg(node, "dim", int) % len(start_permute)
        set_arg(node, "dim", start_permute[dim])

    def update_pad(
        self,
        node: torch.fx.Node,
        start_permute: list[int],
        layout_region: bool,
    ) -> None:
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
        if (
            layout_region
            and len(start_permute) == 4
            and node.target == exir_ops.edge.aten.constant_pad_nd.default
            and self.layout_pad_target is not None
        ):
            node.target = self.layout_pad_target

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
        assert permute_node.target in PERMUTE_COPY_TARGETS
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
