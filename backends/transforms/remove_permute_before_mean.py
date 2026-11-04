# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-unsafe

from typing import Optional

import torch
import torch.fx
from executorch.backends.transforms.permute_pass_utils import (
    get_arg,
    RemoveOrReplacePassInterface,
)
from executorch.exir.dialects._ops import ops as exir_ops
from executorch.exir.dialects.edge._ops import EdgeOpOverload
from torch.fx.node import Node
from torch.utils import _pytree as pytree


class RemovePermuteBeforeMeanPass(RemoveOrReplacePassInterface):
    """Remove or sink permute ops that precede mean reductions through unary chains.

    When a permute feeds into a mean (possibly through unary ops like
    dequantize/quantize), two optimizations apply:

    1. If non-reduced dims maintain their relative order and positions, the
       permute is fully removed and the mean's reduction dims are remapped.
    2. Otherwise, the permute is moved after the mean so it operates on
       smaller data.

    Cost model
    ----------
    Let S_in = input size in bytes, S_out = output size in bytes,
    R = S_in / S_out (reduction ratio), C_c = contiguous mean compute
    cost, C_s = strided mean compute cost, and Delta = C_s - C_c.

    Original graph (permute -> mean):
        Cost_orig = 3*S_in + S_out + C_c
        Breakdown: permute reads and writes S_in (2*S_in), mean reads
        S_in and writes S_out.

    Case 1 -- full removal (mean with remapped dims, no permute):
        Cost_remove = S_in + S_out + C_s
        Profitable when: Delta < 2*S_in
        The strided access penalty must be less than twice the full
        input tensor I/O (the eliminated permute cost).

    Case 2 -- reorder (mean with remapped dims -> small permute):
        Cost_reorder = S_in + 3*S_out + C_s
        Profitable when: Delta < 2*(S_in - S_out) = 2*S_in*(R-1)/R
        At R = 4 the threshold is 1.5*S_in; at R = 16 it approaches
        2*S_in, converging to the full-removal bound as S_out becomes
        negligible.

    Full removal always dominates reorder when both are structurally
    possible (Cost_remove < Cost_reorder since S_out < 3*S_out), so
    removal is applied without a cost gate.

    Additionally, if the original permute does not place the reduction
    dims as the trailing (innermost) dimensions, the mean is already
    strided in the original graph. Removing or sinking the permute
    saves the permute I/O (2*S_in) while the mean performance can only
    improve or stay the same. This makes the transformation
    unconditionally profitable without needing the cost model.
    """

    _UNARY_TARGETS: frozenset[EdgeOpOverload] = frozenset(
        {
            exir_ops.edge.quantized_decomposed.dequantize_per_tensor.default,
            exir_ops.edge.quantized_decomposed.quantize_per_tensor.default,
            exir_ops.edge.aten.clone.default,
            exir_ops.edge.aten.relu.default,
            exir_ops.edge.aten.neg.default,
            exir_ops.edge.aten.abs.default,
        }
    )

    _MIN_REDUCTION_RATIO_FOR_REORDER: int = 4

    @property
    def targets(self) -> list[EdgeOpOverload]:
        return [exir_ops.edge.aten.mean.dim]

    def _find_permute_through_unary_chain(self, mean_node: Node) -> Optional[Node]:
        """Walk backward from mean through single-user unary ops to find a permute."""
        current = mean_node.args[0]
        if not isinstance(current, Node):
            return None
        while True:
            if current.target == exir_ops.edge.aten.permute_copy.default:
                return current
            if current.target not in self._UNARY_TARGETS:
                return None
            if len(current.users) != 1:
                return None
            parent = current.args[0]
            if not isinstance(parent, Node):
                return None
            current = parent

    @staticmethod
    def _reduction_dims_are_trailing(reduction_dims: list[int], ndim: int) -> bool:
        """Check whether all reduction dims are the trailing (innermost) dims."""
        canonical = sorted(d % ndim for d in reduction_dims)
        return canonical == list(range(ndim - len(canonical), ndim))

    def _is_reorder_profitable(
        self,
        reduction_dims: list[int],
        new_reduction_dims: list[int],
        ndim: int,
        input_shape: torch.Size,
    ) -> bool:
        """Determine whether sinking the permute past the mean is profitable.

        Three cases, in order of evaluation:

        1. The original permute does not place the reduction dims as trailing
           (innermost) dimensions. The mean is already strided, so removing
           the permute saves 2*S_in with no mean degradation. Always
           profitable.

        2. The new (remapped) reduction dims are trailing in the original
           layout. The mean stays contiguous after the transformation
           (Delta ~ 0). Always profitable.

        3. The mean becomes strided. The reorder is profitable when
           Delta < 2*S_in*(R-1)/R. We approximate this by requiring
           R >= _MIN_REDUCTION_RATIO_FOR_REORDER, ensuring the I/O savings
           from operating on a smaller tensor outweigh the strided access
           penalty.
        """
        if not self._reduction_dims_are_trailing(reduction_dims, ndim):
            return True

        if self._reduction_dims_are_trailing(new_reduction_dims, ndim):
            return True

        reduction_ratio = 1
        canonical_reduction = set(new_reduction_dims)
        for d in canonical_reduction:
            reduction_ratio *= input_shape[d]

        return reduction_ratio >= self._MIN_REDUCTION_RATIO_FOR_REORDER

    def maybe_remove_or_replace(self, node: Node) -> bool:
        reduction_dims = get_arg(node, "dim", list[int])

        permute_node = self._find_permute_through_unary_chain(node)
        if permute_node is None:
            return False

        perm = get_arg(permute_node, "dims", list[int])
        ndim = len(perm)

        if len(permute_node.users) != 1:
            return False

        permute_input = permute_node.args[0]
        assert isinstance(permute_input, Node)

        new_reduction_dims = [perm[d] for d in reduction_dims]
        keepdim = get_arg(node, "keepdim", bool)

        # Determine if the permute can be fully removed (post-mean permute
        # would be a no-op) vs needing to be sunk after the mean.
        canonical_reduction = set(new_reduction_dims)
        if keepdim:
            can_remove = all(
                perm[d] == d for d in range(ndim) if d not in canonical_reduction
            )
        else:
            non_reduced_in_perm_order = [
                d for d in perm if d not in canonical_reduction
            ]
            can_remove = non_reduced_in_perm_order == sorted(non_reduced_in_perm_order)

        # Full removal is almost always profitable. Reorder requires a
        # tighter cost bound; verify via the cost model before proceeding.
        if not can_remove:
            input_shape = permute_input.meta["val"].shape
            if not self._is_reorder_profitable(
                reduction_dims, new_reduction_dims, ndim, input_shape
            ):
                return False

        # Rewire: the permute's single user (either the mean itself or the
        # first unary op in the chain) should read from the permute's input.
        permute_user = next(iter(permute_node.users))
        permute_user.replace_input_with(permute_node, permute_input)
        node.args = (node.args[0], new_reduction_dims) + node.args[2:]

        # Re-derive the mean's meta since its reduction dims changed.
        fake_args = pytree.tree_map(
            lambda x: x.meta["val"] if isinstance(x, Node) else x,
            node.args,
        )
        node.meta["val"] = node.target(*fake_args)  # pyre-ignore[29]

        if not can_remove:
            # Compute the post-mean permute on the reduced output.
            if keepdim:
                post_perm = list(perm)
            else:
                non_reduced_original = sorted(
                    d for d in range(ndim) if d not in canonical_reduction
                )
                non_reduced_permuted = [d for d in perm if d not in canonical_reduction]
                post_perm = [
                    non_reduced_original.index(d) for d in non_reduced_permuted
                ]

            graph = node.graph
            with graph.inserting_after(node):
                new_permute = graph.create_node(
                    "call_function",
                    exir_ops.edge.aten.permute_copy.default,
                    args=(node, post_perm),
                )
                new_permute.meta["val"] = exir_ops.edge.aten.permute_copy.default(
                    node.meta["val"], post_perm
                )
            for user in list(node.users):
                if user is not new_permute:
                    user.replace_input_with(node, new_permute)

        return True
