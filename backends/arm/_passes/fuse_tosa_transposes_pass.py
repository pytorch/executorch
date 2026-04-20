# Copyright 2026 Arm Limited and/or its affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.
# pyre-strict

import logging
from typing import Set, Type

import torch
from executorch.backends.arm._passes import ArmPass
from executorch.backends.arm._passes.arm_pass_utils import get_first_fake_tensor
from executorch.exir.dialects._ops import ops as exir_ops
from executorch.exir.pass_base import ExportPass, PassResult
from torch.fx import GraphModule

logger = logging.getLogger(__name__)

# Layout-agnostic TOSA/edge ops that transposes can propagate through.
# These ops process each element independently, so reordering memory
# layout before or after them produces identical results.
#
# IMPORTANT: Only unary or broadcast-safe ops belong here. For binary ops
# (ADD, MUL, SUB, etc.), propagation is only safe when the other operand
# is broadcast-compatible in both the original and transposed layout.
#
# NOTE: We use string-based target matching because TOSA backend ops
# (exir_ops.backend.tosa.*) are registered lazily and may not be available
# at module import time.
_UNARY_ELEMENTWISE_TARGET_NAMES: set[str] = {
    # Native TOSA ops (matched by string name)
    "RESCALE",
    # Edge ATen unary ops (matched by string name)
    "abs.default",
    "ceil.default",
    "floor.default",
    "neg.default",
    "clamp.default",
    "sigmoid.default",  # TABLE op, pure elementwise
    "tanh.default",  # TABLE op, pure elementwise
}

# Binary elementwise ops where transpose propagation is safe when the
# non-primary operand is a scalar or 1-element tensor (broadcast-safe).
_BINARY_ELEMENTWISE_TARGET_NAMES: set[str] = {
    "add.Tensor",
    "sub.Tensor",
    "mul.Tensor",
}


def _target_name(target: object) -> str:
    """Extract a recognizable name from a node target for string matching."""
    name = str(target)
    # Handle exir_ops.backend.tosa.RESCALE.default → "RESCALE"
    # Handle exir_ops.edge.aten.add.Tensor → "add.Tensor"
    parts = name.rsplit(".", 2)
    if len(parts) >= 2:
        # For "backend__ops_tosa_RESCALE_default" patterns
        if "RESCALE" in name:
            return "RESCALE"
        # Return the last two parts for ATen ops: "add.Tensor", "clamp.default", etc.
        return ".".join(parts[-2:])
    return name


class FuseTosaTransposesPass(ArmPass):
    """
    Eliminate redundant TOSA TRANSPOSE operations.

    This pass runs after ToTosaMemoryFormatPass and performs four optimizations:
    1. Identity elimination: Remove TRANSPOSE with identity permutation [0,1,2,3]
    2. Inverse-pair cancellation: Remove TRANSPOSE → TRANSPOSE pairs that compose to identity
    3. Composition: Fuse consecutive TRANSPOSE ops into a single TRANSPOSE
    4. Propagation: Move TRANSPOSE through layout-agnostic ops to enable more cancellations

    The propagation pattern (4) handles the common case where ToTosaMemoryFormatPass
    inserts TRANSPOSE pairs around view_copy rank boundaries, with RESCALE and
    elementwise ops in between that prevent direct inverse-pair cancellation:

        TRANSPOSE(p) → RESCALE → relu → RESCALE → TRANSPOSE(inv(p))
        →  RESCALE → relu → RESCALE  (transposes cancelled)
    """

    _passes_required_after: Set[Type[ExportPass]] = set()

    @staticmethod
    def _is_identity_permutation(perms: list[int]) -> bool:
        """Check if a permutation is an identity (e.g., [0,1,2,3])."""
        return perms == list(range(len(perms)))

    @staticmethod
    def _compose_permutations(perm1: list[int], perm2: list[int]) -> list[int]:
        """
        Compose two permutations: result[i] = perm1[perm2[i]].

        Example:
            perm1 = [0, 2, 3, 1]  # NCHW → NHWC
            perm2 = [0, 3, 1, 2]  # NHWC → NCHW
            result = [0, 1, 2, 3]  # Identity
        """
        return [perm1[p] for p in perm2]

    @staticmethod
    def _is_inverse_pair(perm1: list[int], perm2: list[int]) -> bool:
        """Check if two permutations compose to identity."""
        composed = FuseTosaTransposesPass._compose_permutations(perm1, perm2)
        return FuseTosaTransposesPass._is_identity_permutation(composed)

    @staticmethod
    def _is_transpose(node: torch.fx.Node) -> bool:
        return (
            node.op == "call_function"
            and "TRANSPOSE" in str(node.target)
        )

    @staticmethod
    def _get_transpose_perm(node: torch.fx.Node) -> list[int] | None:
        if not FuseTosaTransposesPass._is_transpose(node):
            return None
        if len(node.args) < 2:
            return None
        perms = node.args[1]
        if not isinstance(perms, (list, tuple)):
            return None
        return list(perms)

    @staticmethod
    def _is_scalar_or_broadcast_safe(node: torch.fx.Node) -> bool:
        """Check if a node produces a scalar or 1-element tensor (broadcast-safe)."""
        if node.op in ("get_attr",):
            # Weight/param nodes — check shape from metadata
            if "val" in node.meta:
                val = node.meta["val"]
                if hasattr(val, "shape"):
                    return val.numel() == 1
            return False
        if "val" in node.meta:
            val = node.meta["val"]
            if hasattr(val, "shape"):
                shape = val.shape
                # Scalar (0-dim) or single-element
                if len(shape) == 0 or all(s == 1 for s in shape):
                    return True
                # Also safe: 1D tensor with size matching channel dim (broadcasts)
                # But we conservatively limit to true scalars/singletons
            return False
        return False

    def _eliminate_identity_transposes(self, graph_module: GraphModule) -> bool:
        """
        Pattern 1: Identity Transpose Elimination.
        Remove any TRANSPOSE where the permutation is identity.
        """
        modified = False
        for node in list(graph_module.graph.nodes):
            if self._is_transpose(node):
                if len(node.args) < 2:
                    continue

                perms = node.args[1]
                if not isinstance(perms, (list, tuple)):
                    continue

                if self._is_identity_permutation(list(perms)):
                    input_node = node.args[0]
                    logger.debug(
                        f"Eliminating identity TRANSPOSE {node.name} with perm {perms}"
                    )

                    # Replace all uses of the TRANSPOSE with its input
                    node.replace_all_uses_with(input_node)
                    graph_module.graph.erase_node(node)
                    modified = True

        return modified

    def _cancel_inverse_pairs(self, graph_module: GraphModule) -> bool:
        """
        Pattern 2: Inverse-Pair Cancellation.
        Remove adjacent TRANSPOSE pairs where the second is the inverse of the first.
        """
        modified = False
        for node in list(graph_module.graph.nodes):
            if self._is_transpose(node):
                if len(node.args) < 2:
                    continue

                input_node = node.args[0]
                if self._is_transpose(input_node):
                    if len(input_node.args) < 2:
                        continue

                    perm1 = input_node.args[1]
                    perm2 = node.args[1]

                    if not isinstance(perm1, (list, tuple)) or not isinstance(
                        perm2, (list, tuple)
                    ):
                        continue

                    perm1_list = list(perm1)
                    perm2_list = list(perm2)

                    if self._is_inverse_pair(perm1_list, perm2_list):
                        original_input = input_node.args[0]
                        logger.debug(
                            f"Cancelling inverse TRANSPOSE pair: {input_node.name} {perm1_list} → {node.name} {perm2_list}"
                        )

                        # Replace all uses of the second TRANSPOSE with the original input
                        node.replace_all_uses_with(original_input)

                        # Erase both TRANSPOSE nodes if the first has no other users
                        graph_module.graph.erase_node(node)
                        if len(input_node.users) == 0:
                            graph_module.graph.erase_node(input_node)

                        modified = True

        return modified

    def _compose_consecutive_transposes(self, graph_module: GraphModule) -> bool:
        """
        Pattern 3: Composition (Non-Inverse Consecutive Transposes).
        Fuse consecutive TRANSPOSE ops into a single TRANSPOSE.
        """
        modified = False
        for node in list(graph_module.graph.nodes):
            if self._is_transpose(node):
                if len(node.args) < 2:
                    continue

                input_node = node.args[0]
                if self._is_transpose(input_node):
                    if len(input_node.args) < 2:
                        continue

                    perm1 = input_node.args[1]
                    perm2 = node.args[1]

                    if not isinstance(perm1, (list, tuple)) or not isinstance(
                        perm2, (list, tuple)
                    ):
                        continue

                    perm1_list = list(perm1)
                    perm2_list = list(perm2)

                    # Only compose if they don't already cancel out
                    if not self._is_inverse_pair(perm1_list, perm2_list):
                        composed_perm = self._compose_permutations(perm1_list, perm2_list)
                        logger.debug(
                            f"Composing TRANSPOSE pair: {input_node.name} {perm1_list} → {node.name} {perm2_list} = {composed_perm}"
                        )

                        original_input = input_node.args[0]

                        # Create a new TRANSPOSE with the composed permutation
                        with graph_module.graph.inserting_before(node):
                            new_transpose = graph_module.graph.call_function(
                                exir_ops.backend.tosa.TRANSPOSE.default,
                                args=(original_input, composed_perm),
                            )

                            # Copy metadata
                            new_transpose.meta.update(node.meta)
                            if "tosa_dim_order" in node.meta:
                                new_transpose.meta["tosa_dim_order"] = node.meta[
                                    "tosa_dim_order"
                                ]
                            if "tosa_spatial_rank" in node.meta:
                                new_transpose.meta["tosa_spatial_rank"] = node.meta[
                                    "tosa_spatial_rank"
                                ]

                        # Replace second TRANSPOSE with the new composed one
                        node.replace_all_uses_with(new_transpose)
                        graph_module.graph.erase_node(node)

                        # Remove first TRANSPOSE if it has no other users
                        if len(input_node.users) == 0:
                            graph_module.graph.erase_node(input_node)

                        modified = True

        return modified

    def _propagate_transpose_through_elementwise(
        self, graph_module: GraphModule
    ) -> bool:
        """
        Pattern 4: Propagate TRANSPOSE through layout-agnostic ops.

        Detects chains of the form:
            TRANSPOSE(p) → [elementwise ops] → TRANSPOSE(inv(p))
        where all intermediate ops are layout-agnostic (RESCALE, ADD, MUL, etc.)
        with single users, and eliminates both transposes.

        This is safe because elementwise ops produce identical results regardless
        of memory layout — TRANSPOSE(f(x)) == f(TRANSPOSE(x)) for any
        elementwise f.

        For binary ops (ADD, MUL, SUB), propagation is only performed when
        the non-primary operand is broadcast-safe (scalar or 1-element tensor).
        """
        modified = False

        for node in list(graph_module.graph.nodes):
            perm = self._get_transpose_perm(node)
            if perm is None:
                continue

            # Only consider single-user TRANSPOSE nodes to ensure safe rewiring
            if len(node.users) != 1:
                continue

            # Walk forward through single-user elementwise chain
            chain = self._walk_elementwise_chain(node, perm)
            if chain is None:
                continue

            # chain is (list_of_intermediate_nodes, final_inverse_transpose)
            intermediate_nodes, end_transpose = chain

            original_input = node.args[0]  # Input before the first TRANSPOSE

            logger.debug(
                f"Propagating TRANSPOSE through {len(intermediate_nodes)} elementwise ops: "
                f"{node.name} → {[n.name for n in intermediate_nodes]} → {end_transpose.name}"
            )

            # Rewire: first elementwise op takes original_input instead of transpose output
            if intermediate_nodes:
                first_elem = intermediate_nodes[0]
                first_elem.replace_input_with(node, original_input)
            else:
                # Direct inverse pair with no intermediate ops (handled by _cancel_inverse_pairs)
                continue

            # Rewire: end_transpose's users take last elementwise output
            last_elem = intermediate_nodes[-1]
            end_transpose.replace_all_uses_with(last_elem)

            # Update tosa_dim_order for intermediate nodes to match
            # original_input, guarding by rank to avoid propagating across
            # rank boundaries.
            if "tosa_dim_order" in original_input.meta:
                orig_dim_order = original_input.meta["tosa_dim_order"]
                for inode in intermediate_nodes:
                    if "val" in inode.meta:
                        inode_rank = get_first_fake_tensor(inode).dim()
                        if len(orig_dim_order) == inode_rank:
                            inode.meta["tosa_dim_order"] = orig_dim_order

            # Erase both transposes
            graph_module.graph.erase_node(end_transpose)
            if len(node.users) == 0:
                graph_module.graph.erase_node(node)

            modified = True

        return modified

    def _walk_elementwise_chain(
        self,
        start_transpose: torch.fx.Node,
        start_perm: list[int],
    ) -> tuple[list[torch.fx.Node], torch.fx.Node] | None:
        """
        Walk forward from a TRANSPOSE node through a chain of single-user
        elementwise ops, looking for an inverse TRANSPOSE at the end.

        Returns (intermediate_nodes, end_transpose) or None if no valid chain found.
        """
        current = start_transpose
        intermediates: list[torch.fx.Node] = []
        max_chain_length = 20  # Safety limit

        while len(intermediates) < max_chain_length:
            # Current node must have exactly one user
            if len(current.users) != 1:
                return None

            next_node = list(current.users)[0]

            # Check if next node is the closing inverse TRANSPOSE
            next_perm = self._get_transpose_perm(next_node)
            if next_perm is not None:
                if self._is_inverse_pair(start_perm, next_perm):
                    return (intermediates, next_node)
                else:
                    # Non-inverse TRANSPOSE in chain — stop
                    return None

            # Check if next node is a layout-agnostic elementwise op
            if not self._is_propagation_safe(next_node, chain_input=current):
                return None

            intermediates.append(next_node)
            current = next_node

        return None  # Chain too long

    def _is_propagation_safe(
        self, node: torch.fx.Node, chain_input: torch.fx.Node
    ) -> bool:
        """
        Check if a TRANSPOSE can safely propagate through this node.

        A node is safe for transpose propagation if:
        1. It's a unary elementwise op (RESCALE, abs, ceil, floor, neg, clamp), OR
        2. It's a binary elementwise op (add, mul, sub) where the non-chain
           operand is broadcast-safe (scalar or 1-element tensor)
        3. It has exactly one user (ensures we don't break other consumers)

        Args:
            node: The candidate node to propagate through.
            chain_input: The predecessor node in the chain (used to identify
                which operand of a binary op is the chain operand vs the other).
        """
        if node.op != "call_function":
            return False

        # Must have single user for safe chain walking
        if len(node.users) != 1:
            return False

        tname = _target_name(node.target)

        # Unary elementwise ops are always safe
        if tname in _UNARY_ELEMENTWISE_TARGET_NAMES:
            return True

        # Binary elementwise ops: the non-chain operand must be broadcast-safe
        if tname in _BINARY_ELEMENTWISE_TARGET_NAMES:
            if len(node.args) >= 2:
                arg0, arg1 = node.args[0], node.args[1]
                # Identify the non-chain operand (the one NOT from chain_input)
                if arg0 is chain_input:
                    other = arg1
                elif arg1 is chain_input:
                    other = arg0
                else:
                    # Neither arg is the chain input — can't determine safety
                    return False
                # Non-chain operand must be broadcast-safe
                if isinstance(other, (int, float)):
                    return True
                if isinstance(other, torch.fx.Node) and self._is_scalar_or_broadcast_safe(other):
                    return True
            return False

        return False

    @staticmethod
    def _count_tosa_transposes(graph_module: GraphModule) -> int:
        """Count TOSA TRANSPOSE nodes using string-based matching.

        Uses string matching instead of direct object comparison because
        TOSA backend ops are lazily registered and the target object
        may not compare equal via '==' even when it's the same op.
        """
        count = 0
        for node in graph_module.graph.nodes:
            if node.op == "call_function" and "TRANSPOSE" in str(node.target):
                count += 1
        return count

    def call(self, graph_module: GraphModule) -> PassResult:
        """
        Entry point for the pass. Use fixed-point iteration to eliminate
        all redundant transposes.
        """
        before_count = self._count_tosa_transposes(graph_module)

        modified_overall = False
        iteration = 0
        max_iterations = 10

        while iteration < max_iterations:
            modified = False

            # Apply all four patterns
            modified |= self._eliminate_identity_transposes(graph_module)
            modified |= self._cancel_inverse_pairs(graph_module)
            modified |= self._compose_consecutive_transposes(graph_module)
            modified |= self._propagate_transpose_through_elementwise(graph_module)

            if not modified:
                break

            modified_overall = True
            iteration += 1

        if iteration >= max_iterations:
            logger.warning(
                f"FuseTosaTransposesPass reached max iterations ({max_iterations})"
            )

        graph_module.graph.eliminate_dead_code()
        graph_module.recompile()

        after_count = self._count_tosa_transposes(graph_module)
        if before_count != after_count:
            logger.info(
                f"FuseTosaTransposesPass: TOSA_TRANSPOSE {before_count} -> {after_count} "
                f"(eliminated {before_count - after_count}), iterations={iteration}"
            )

        return PassResult(graph_module, modified_overall)
