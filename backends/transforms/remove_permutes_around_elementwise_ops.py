# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
# Copyright 2026 Arm Limited and/or its affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-unsafe

from dataclasses import dataclass, field
from typing import cast

import torch
import torch.fx
from executorch.backends.transforms.permute_pass_utils import get_arg, set_arg
from executorch.exir.dialects._ops import ops as exir_ops
from executorch.exir.pass_base import ExportPass, PassResult


class RemovePermutesAroundElementwiseOps(ExportPass):
    """
    Looks for subgraphs of elementwise ops sandwiched between permutes and removes those
    permutes if possible.
    Allows special handling for certain non-elementwise ops that can be easily updated
    based on the permute's parameter such as mean, cat, and slice.
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
        # Per-node expected end permutation (may differ from end_permute
        # when the subgraph contains rank-changing views).
        node_end_permute: dict[torch.fx.Node, list[int]] = field(default_factory=dict)
        # Per-node expected start permutation for upstream traversal.
        node_start_permute: dict[torch.fx.Node, list[int]] = field(default_factory=dict)

    def __init__(self, extra_permutable_ops: set | None = None) -> None:
        super().__init__()
        self._permutable_ops = {
            exir_ops.edge.aten.add.Tensor,
            exir_ops.edge.aten.mul.Tensor,
            exir_ops.edge.aten.sub.Tensor,
            exir_ops.edge.aten.hardtanh.default,
            exir_ops.edge.aten.clamp.default,
            exir_ops.edge.aten.cat.default,
            exir_ops.edge.aten.mean.dim,
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

    _VIEW_OPS = (
        exir_ops.edge.aten.view_copy.default,
        exir_ops.edge.aten.view.default,
    )

    _UNSQUEEZE_OPS = (exir_ops.edge.aten.unsqueeze_copy.default,)

    _SQUEEZE_OPS = (exir_ops.edge.aten.squeeze_copy.dim,)

    @staticmethod
    def _find_extra_one(longer: list[int], shorter: list[int]) -> int:
        """If longer has exactly one more element of value 1, return its index. Else -1."""
        if len(longer) != len(shorter) + 1:
            return -1
        for i in range(len(shorter)):
            if longer[i] != shorter[i]:
                if longer[i] == 1 and shorter[i:] == longer[i + 1 :]:
                    return i
                return -1
        return len(shorter) if longer[-1] == 1 else -1

    def _is_squeeze_unsqueeze_view(self, node: torch.fx.Node) -> bool:
        """Check if a node is a squeeze, unsqueeze, or view_copy that only
        adds or removes a single dim of size 1."""
        if node in self._sq_unsq_cache:
            return self._sq_unsq_cache[node]
        result = self._check_squeeze_unsqueeze_view(node)
        self._sq_unsq_cache[node] = result
        return result

    def _check_squeeze_unsqueeze_view(self, node: torch.fx.Node) -> bool:
        if node.target in self._UNSQUEEZE_OPS or node.target in self._SQUEEZE_OPS:
            return True
        if node.target not in self._VIEW_OPS:
            return False
        inp = node.args[0]
        assert isinstance(inp, torch.fx.Node)
        in_shape = inp.meta["val"].shape
        out_shape = node.meta["val"].shape
        if len(out_shape) == len(in_shape) + 1:
            return self._find_extra_one(out_shape, in_shape) != -1
        if len(in_shape) == len(out_shape) + 1:
            return self._find_extra_one(in_shape, out_shape) != -1
        return False

    def _adapt_permute_across_view(
        self, permute: list[int], node: torch.fx.Node
    ) -> list[int] | None:
        """Adjust a permutation across a squeeze/unsqueeze boundary.

        Adapts from input-rank to output-rank space (downstream direction).
        Returns the adjusted permutation, or None if not possible.
        """
        # Handle explicit unsqueeze_copy(dim)
        if node.target in self._UNSQUEEZE_OPS:
            dim = cast(int, node.args[1])
            rank = len(permute)
            index = dim if dim >= 0 else dim + rank + 1
            new_perm = [x + 1 if x >= index else x for x in permute]
            new_perm.insert(index, index)
            return new_perm

        # Handle explicit squeeze_copy(dim)
        if node.target in self._SQUEEZE_OPS:
            dim = cast(int, node.args[1])
            rank = len(permute)
            index = dim if dim >= 0 else dim + rank
            # index is a POSITION in the tensor; the permutation VALUE at
            # that position is the logical dim being removed.
            squeezed_value = permute[index]
            new_perm = [
                x - 1 if x > squeezed_value else x
                for x in permute
                if x != squeezed_value
            ]
            return new_perm

        # Handle view_copy (squeeze/unsqueeze-like reshape)
        inp = node.args[0]
        assert isinstance(inp, torch.fx.Node)
        in_shape = inp.meta["val"].shape
        out_shape = node.meta["val"].shape

        if len(out_shape) == len(in_shape) + 1:
            # unsqueeze: insert identity mapping at the new dim
            index = self._find_extra_one(out_shape, in_shape)
            new_perm = [x + 1 if x >= index else x for x in permute]
            new_perm.insert(index, index)
            return new_perm
        elif len(in_shape) == len(out_shape) + 1:
            # squeeze via view_copy: find the squeezed dim and remove it
            index = self._find_extra_one(in_shape, out_shape)
            # index is a POSITION in in_shape; the permutation VALUE at
            # that position is the logical dim being removed.
            squeezed_value = permute[index]
            new_perm = [
                x - 1 if x > squeezed_value else x
                for x in permute
                if x != squeezed_value
            ]
            return new_perm
        return None

    def call(self, graph_module: torch.fx.GraphModule) -> PassResult:  # noqa: C901
        self._sq_unsq_cache.clear()
        subgraphs_found: list[RemovePermutesAroundElementwiseOps.Subgraph] = []
        processed_nodes: set[torch.fx.Node] = set()
        for node in graph_module.graph.find_nodes(
            op="call_function", target=exir_ops.edge.aten.permute_copy.default
        ):
            start_permute = self.get_permutation(node)
            if start_permute is None:
                continue
            # Expected end permutation for the subgraph.
            end_permute = [start_permute.index(i) for i in range(len(start_permute))]

            # Try direct users first (same-rank matching)
            for user in node.users:
                if not self.is_node_permutable(user):
                    continue
                subgraph = self.Subgraph(start_permute, end_permute)
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
                        if not self.is_node_permutable(view_user):
                            continue
                        subgraph = self.Subgraph(adapted_start, adapted_end)
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
        if node in processed_nodes or not self.is_node_permutable(node):
            return False
        subgraph.nodes.add(node)
        subgraph.node_end_permute[node] = current_end_permute
        subgraph.node_start_permute[node] = current_start_permute

        # If this is a squeeze/unsqueeze view, adapt permutations for
        # traversal across the rank change boundary.
        downstream_end = current_end_permute
        downstream_start = current_start_permute
        if self._is_squeeze_unsqueeze_view(node):
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
        for user in node.users:
            if user.target == exir_ops.edge.aten.permute_copy.default:
                user_perm = self.get_permutation(user)
                if user_perm == downstream_end:
                    subgraph.edges_out.add((node, user))
                else:
                    # Check if permute → view(squeeze/unsqueeze) forms an
                    # end boundary at a different rank.
                    user_users = list(user.users.keys())
                    if len(user_users) == 1 and self._is_squeeze_unsqueeze_view(
                        user_users[0]
                    ):
                        view_after = user_users[0]
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
                                subgraph.edges_out.add((node, user))
                                # Mark the view for inclusion so it gets preserved
                                continue
                    return False
            elif user.op == "output":
                return False
            elif not self.visit(
                user, subgraph, processed_nodes, downstream_end, downstream_start
            ):
                return False

        # Traverse upstream:
        for inp in node.all_input_nodes:
            if inp.target == exir_ops.edge.aten.permute_copy.default:
                if self.get_permutation(inp) != current_start_permute:
                    return False
                subgraph.edges_in.add((inp, node))
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

    def permute_subgraph(self, subgraph: Subgraph) -> bool:  # noqa: C901
        # Validate: every view_copy node's permutation rank must match its
        # input tensor rank.  A mismatch can occur when a squeeze/unsqueeze
        # view is reached via upstream traversal with a permutation that was
        # already adapted to a different rank.  Applying the optimisation in
        # this case would produce an invalid graph, so skip the subgraph.
        for node in subgraph.nodes:
            if node.target in self._VIEW_OPS:
                perm = subgraph.node_start_permute.get(node, subgraph.start_permute)
                inp = node.args[0]
                if isinstance(inp, torch.fx.Node) and inp.meta.get("val") is not None:
                    if len(perm) != len(inp.meta["val"].shape):
                        return False

        # Handle dimension related node arguments FIRST, before
        # bypassing permutes (which changes node inputs/metadata).
        for node in subgraph.nodes:
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
            elif node.target in self._VIEW_OPS:
                self.update_view_copy(node, node_start_perm)
            elif node.target in self._UNSQUEEZE_OPS:
                # unsqueeze dim is in output space (rank + 1)
                dim = cast(int, node.args[1])
                rank = len(node_start_perm)
                index = dim if dim >= 0 else dim + rank + 1
                if index < rank:
                    node.update_arg(1, node_start_perm[index])
                else:
                    # Inserting at or beyond existing dims — position unchanged
                    node.update_arg(1, index)
            elif node.target in self._SQUEEZE_OPS:
                # squeeze dim is in input space (rank)
                dim = get_arg(node, "dim", int)
                set_arg(node, "dim", node_start_perm[dim])

        # Skip incoming permutes.
        for inp, out in subgraph.edges_in:
            assert inp.target == exir_ops.edge.aten.permute_copy.default
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
                        exir_ops.edge.aten.permute_copy.default,
                        args=(const_node, node_end_perm),
                    )
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
            assert out.target == exir_ops.edge.aten.permute_copy.default
            out.replace_all_uses_with(inp)

        return True

    def update_cat(self, node: torch.fx.Node, start_permute: list[int]) -> None:
        dim = get_arg(node, "dim", int)
        set_arg(node, "dim", start_permute[dim])

    def update_mean_dim(self, node: torch.fx.Node, start_permute: list[int]) -> None:
        dims = get_arg(node, "dim")
        set_arg(node, "dim", [start_permute[d] for d in cast(list[int], dims)])

    def update_slice_copy(self, node: torch.fx.Node, start_permute: list[int]) -> None:
        dim = get_arg(node, "dim", int)
        set_arg(node, "dim", start_permute[dim])

    def update_view_copy(self, node: torch.fx.Node, start_permute: list[int]) -> None:
        """Adjust view_copy shape arg after permute removal.

        After removing the start permute, the view's input is in the original
        (un-permuted) layout. Recompute the view's target shape accordingly.
        """
        inp = node.args[0]
        assert isinstance(inp, torch.fx.Node)

        in_shape = inp.meta["val"].shape
        out_shape = node.meta["val"].shape

        # Compute un-permuted input shape
        inverse_permute = [start_permute.index(i) for i in range(len(start_permute))]
        unpermuted_in = [in_shape[inverse_permute[i]] for i in range(len(in_shape))]

        if len(out_shape) == len(in_shape) + 1:
            # unsqueeze: find the inserted dim in the permuted output,
            # then determine where it goes in the un-permuted layout
            index = self._find_extra_one(out_shape, in_shape)
            if index != -1:
                new_shape = list(unpermuted_in)
                new_shape.insert(index, 1)
                node.update_arg(1, new_shape)
        elif len(in_shape) == len(out_shape) + 1:
            # squeeze: find the removed dim in the permuted input,
            # map it to the un-permuted position, and remove it
            index = self._find_extra_one(in_shape, out_shape)
            if index != -1:
                # Map the squeezed dim from permuted to un-permuted space
                unpermuted_index = start_permute[index]
                new_shape = list(unpermuted_in)
                del new_shape[unpermuted_index]
                node.update_arg(1, new_shape)

    def get_permutation(self, permute_node: torch.fx.Node) -> list[int] | None:
        assert permute_node.target == exir_ops.edge.aten.permute_copy.default
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
