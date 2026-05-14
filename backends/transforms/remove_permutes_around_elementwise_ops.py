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
from executorch.exir.dialects._ops import ops as exir_ops
from executorch.exir.dialects.edge._ops import EdgeOpOverload
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

    # Ops explicitly listed as permutable. This includes non-pointwise ops
    # that need special dimension-argument handling (cat, mean, sum, slice)
    # and quantize/dequantize ops not tagged as pointwise in ATen.
    # In addition to this set, any op tagged with torch.Tag.pointwise is
    # automatically considered permutable (see is_node_permutable).
    permutable_ops: set[EdgeOpOverload] = {
        exir_ops.edge.quantized_decomposed.quantize_per_tensor.default,
        exir_ops.edge.quantized_decomposed.dequantize_per_tensor.default,
        # Ops that require special handling of dimension arguments.
        exir_ops.edge.aten.cat.default,
        exir_ops.edge.aten.mean.dim,
        exir_ops.edge.aten.sum.dim_IntList,
        exir_ops.edge.aten.slice_copy.Tensor,
    }

    def call(self, graph_module: torch.fx.GraphModule) -> PassResult:
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

            for user in node.users:
                if user.target not in self.permutable_ops and not self._is_pointwise(
                    user.target
                ):
                    continue
                # Create a separate subgraph for each user since there may be cases
                # where only a portion of the users are permutable.
                subgraph = self.Subgraph(start_permute, end_permute)
                if self.visit(user, subgraph, processed_nodes):
                    subgraphs_found.append(subgraph)
                    for node in subgraph.nodes:
                        processed_nodes.add(node)

        modified = False
        for subgraph in subgraphs_found:
            self.permute_subgraph(subgraph)
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
    ) -> bool:
        if node in subgraph.nodes:
            return True
        if node in processed_nodes or not self.is_node_permutable(node):
            return False
        subgraph.nodes.add(node)

        # Traverse downstream:
        for user in node.users:
            # Output should either go to a matching permute or another permutable op.
            if user.target == exir_ops.edge.aten.permute_copy.default:
                if self.get_permutation(user) != subgraph.end_permute:
                    return False
                subgraph.edges_out.add((node, user))
            elif user.op == "output":
                # Graph output requires the data in its original layout.
                # Removing permutes here would silently change the output
                # format, so treat this as an invalid subgraph boundary.
                return False
            elif not self.visit(user, subgraph, processed_nodes):
                return False

        # Traverse upstream:
        for inp in node.all_input_nodes:
            # Input should either come from a matching permute or another permutable op.
            if inp.target == exir_ops.edge.aten.permute_copy.default:
                if self.get_permutation(inp) != subgraph.start_permute:
                    return False
                subgraph.edges_in.add((inp, node))
            elif self._is_constant(inp):
                # Only accept the constant if we can insert a compensating
                # permute or view. Otherwise reject the subgraph.
                const_rank = self._get_node_rank(inp)
                permute_rank = len(subgraph.end_permute)
                if const_rank is None:
                    return False
                if const_rank > permute_rank:
                    return False
                if const_rank < permute_rank and inp.meta.get("val") is None:
                    return False
                subgraph.constant_edges_in.add((inp, node))
            elif not self.visit(inp, subgraph, processed_nodes):
                return False

        return True

    def _is_constant(self, node: torch.fx.Node) -> bool:
        """Check if a node's value is available at compile time.
        Only considers direct constants (get_attr, parameter/buffer/constant
        placeholders) — does not recurse into call_function chains to avoid
        stack overflow on deep graphs."""
        if node.op == "get_attr":
            return True
        if node.op == "placeholder":
            target = str(node.target)
            return target.startswith(("b_", "p_", "c_"))
        return False

    def _get_node_rank(self, node: torch.fx.Node) -> int | None:
        """Return the tensor rank of a node's output, or None if unknown."""
        val = node.meta.get("val")
        if val is not None and hasattr(val, "shape"):
            return len(val.shape)
        return None

    @staticmethod
    def _is_pointwise(target) -> bool:
        """Check if a target op is tagged as pointwise in ATen."""
        op = getattr(target, "_op", None)
        if op is not None and hasattr(op, "tags"):
            return torch.Tag.pointwise in op.tags
        return False

    def is_node_permutable(self, node: torch.fx.Node) -> bool:
        if node.target in self.permutable_ops:
            # Special-case validation for dim-based ops.
            if node.target in (
                exir_ops.edge.aten.mean.dim,
                exir_ops.edge.aten.sum.dim_IntList,
            ):
                # keepdim should be True.
                if len(node.args) >= 3:
                    if not node.args[2]:
                        return False
                elif "keepdim" in node.kwargs:
                    if not node.kwargs["keepdim"]:
                        return False
                else:
                    # Default keepdim is False.
                    return False
            return True
        # Accept any op tagged as pointwise in ATen (elementwise).
        return self._is_pointwise(node.target)

    def permute_subgraph(self, subgraph: Subgraph) -> None:
        # Skip incoming permutes.
        for inp, out in subgraph.edges_in:
            assert inp.target == exir_ops.edge.aten.permute_copy.default
            if len(inp.args) >= 1:
                out.replace_input_with(inp, cast(torch.fx.Node, inp.args[0]))
            else:
                out.replace_input_with(inp, cast(torch.fx.Node, inp.kwargs["input"]))

        # Insert compensating permute on constant inputs.
        # Since the subgraph's start permutes are being removed, the subgraph
        # will operate in the un-permuted (original) layout. Constants that
        # were in the permuted layout need end_permute (the inverse of
        # start_permute) to convert back to the original layout.
        for const_node, user_node in subgraph.constant_edges_in:
            graph = const_node.graph
            const_rank = self._get_node_rank(const_node)
            permute_rank = len(subgraph.end_permute)

            with graph.inserting_after(const_node):
                if const_rank is not None and const_rank == permute_rank:
                    new_node = graph.create_node(
                        "call_function",
                        exir_ops.edge.aten.permute_copy.default,
                        args=(const_node, subgraph.end_permute),
                    )
                elif (
                    const_rank is not None
                    and const_rank < permute_rank
                    and const_node.meta.get("val") is not None
                ):
                    # Rank mismatch (e.g. rank-1 bias with rank-4 permute).
                    # The constant is broadcastable and its shape is smaller
                    # than the permute rank, so we can't apply the permute
                    # directly. Instead, use view_copy to rearrange the
                    # shape according to the end_permute restricted to
                    # the trailing dimensions.
                    original_shape = list(const_node.meta["val"].shape)
                    # Pad shape to match permute rank for reordering
                    padded = [1] * (permute_rank - const_rank) + original_shape
                    target_shape = [padded[d] for d in subgraph.end_permute]
                    # Strip leading 1s back to original rank
                    target_shape = target_shape[permute_rank - const_rank :]
                    new_node = graph.create_node(
                        "call_function",
                        exir_ops.edge.aten.view_copy.default,
                        args=(const_node, target_shape),
                    )
                else:
                    # Cannot determine rank or handle this case; skip.
                    continue
            user_node.replace_input_with(const_node, new_node)

        # Skip outgoing permutes.
        for inp, out in subgraph.edges_out:
            assert out.target == exir_ops.edge.aten.permute_copy.default
            out.replace_all_uses_with(inp)

        # Handle dimension related node arguments.
        for node in subgraph.nodes:
            if node.target == exir_ops.edge.aten.cat.default:
                self.update_cat(node, subgraph.start_permute)
            elif node.target in (
                exir_ops.edge.aten.mean.dim,
                exir_ops.edge.aten.sum.dim_IntList,
            ):
                self.update_mean_dim(node, subgraph.start_permute)
            elif node.target == exir_ops.edge.aten.slice_copy.Tensor:
                self.update_slice_copy(node, subgraph.start_permute)

    def update_cat(self, node: torch.fx.Node, start_permute: list[int]) -> None:
        if len(node.args) >= 2:
            node.update_arg(1, start_permute[cast(int, node.args[1])])
        elif "dim" in node.kwargs:
            node.update_kwarg("dim", start_permute[cast(int, node.kwargs["dim"])])
        else:
            # Default cat dim is 0.
            node.update_kwarg("dim", start_permute[0])

    def update_mean_dim(self, node: torch.fx.Node, start_permute: list[int]) -> None:
        if len(node.args) >= 2:
            node.update_arg(
                1, [start_permute[dim] for dim in cast(list[int], node.args[1])]
            )
        else:
            node.update_kwarg(
                "dim",
                [start_permute[dim] for dim in cast(list[int], node.kwargs["dim"])],
            )

    def update_slice_copy(self, node: torch.fx.Node, start_permute: list[int]) -> None:
        if len(node.args) >= 2:
            node.update_arg(1, start_permute[cast(int, node.args[1])])
        else:
            node.update_kwarg("dim", start_permute[cast(int, node.kwargs["dim"])])

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
