# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
INT4 Weight-Only Quantized Matmul Fusion Pass

This pass fuses multiple int4pack_mm operations that share the same input tensor
into a single fused operation, reducing kernel launch overhead.

ALGORITHM:

The fusion transforms:
    input → int4mm(input, W_q, block_size, S_q) → Q
    input → int4mm(input, W_k, block_size, S_k) → K
    input → int4mm(input, W_v, block_size, S_v) → V

Into:
    fused_W = cat([W_q, W_k, W_v], dim=0)
    fused_S = cat([S_q, S_k, S_v], dim=1)
    fused_output = int4mm(input, fused_W, block_size, fused_S)
    [Q, K, V] = split(fused_output, dim=-1)

CORRECTNESS:

This transformation is mathematically valid due to matrix multiplication's
distributive property over concatenation:

    X @ [W_1 | W_2 | W_3] = [X @ W_1 | X @ W_2 | X @ W_3]

Where [A | B] denotes horizontal concatenation along the output dimension.

For INT4 quantized operations:
    int4mm(X, W, bs, S) computes: X @ dequantize(W, S, bs)

Therefore:
    int4mm(X, cat([W_1, W_2, W_3]), bs, cat([S_1, S_2, S_3]))
    = cat([int4mm(X, W_1, bs, S_1), int4mm(X, W_2, bs, S_2), int4mm(X, W_3, bs, S_3)])

PREREQUISITES:

This pass requires Common Subexpression Elimination (CSE) to run first:
- CSE merges duplicate preprocessing chains (reshape, cast, pad, etc.)
- After CSE, operations with identical preprocessing share the same input node
- This allows simple grouping by checking node.args[0] equality

EXAMPLES:

1. Attention QKV projection:     3 int4mm ops → 1 fused op
2. MLP Gate/Up projection:        2 int4mm ops → 1 fused op
3. Multi-head attention (8 heads): 8 int4mm ops → 3 fused ops (max_fusion_size=3)
"""

import operator
from collections import defaultdict
from typing import Dict, List, Optional, Tuple

import torch
from executorch.exir.pass_base import ExportPass
from torch.fx import GraphModule, Node
from torch.fx.passes.infra.pass_base import PassResult


class FuseInt4WeightOnlyQuantMatmulPass(ExportPass):
    """
    Fuses INT4 weight-only quantized matmul operations sharing the same input.

    This pass identifies groups of aten._weight_int4pack_mm operations that:
    1. Share the same input tensor (after CSE preprocessing)
    2. Have the same block_size parameter
    3. Have compatible dtypes and devices

    For each group, it:
    1. Concatenates weights and scales
    2. Creates a single fused int4mm operation
    3. Splits the output back to individual results

    Args:
        min_fusion_size: Minimum number of operations to fuse (default: 2)
        max_fusion_size: Maximum operations per fused group (default: 3)
    """

    def __init__(self, min_fusion_size: int = 2, max_fusion_size: int = 3):
        super().__init__()
        self.min_fusion_size = min_fusion_size
        self.max_fusion_size = max_fusion_size

    def call(self, graph_module: GraphModule) -> PassResult:
        """Apply fusion pass to the graph."""
        groups = self._find_fuseable_groups(graph_module)
        fusion_results = [self._fuse_group(graph_module, g) for g in groups]
        modified = any(fusion_results)

        if modified:
            graph_module.graph.lint()
            graph_module.graph.eliminate_dead_code()
            graph_module.recompile()
            try:
                graph_module = super().call(graph_module).graph_module
            except Exception:
                # super().call() may fail on mock graphs without proper metadata
                pass

        return PassResult(graph_module, modified)

    def _is_int4mm(self, node: Node) -> bool:
        """Check if node is an int4pack_mm operation.

        Handles both standard torch ops and EdgeOpOverload (edge dialect).
        """
        if node.op != "call_function":
            return False

        target = node.target

        # Direct match for standard torch op
        if target == torch.ops.aten._weight_int4pack_mm.default:
            return True

        # Handle EdgeOpOverload (edge dialect wraps ops)
        # Check if the target's name matches the int4pack_mm op
        target_name = getattr(target, "_name", None) or getattr(target, "name", lambda: "")()
        if "_weight_int4pack_mm" in str(target_name) or "_weight_int4pack_mm" in str(target):
            return True

        return False

    def _get_params(self, node: Node) -> Optional[Tuple[Node, int, Node]]:
        """
        Extract parameters from int4mm node.

        Returns:
            (weight_node, block_size, scale_node) or None if invalid
        """
        if not self._is_int4mm(node) or len(node.args) < 4:
            return None

        w, bs, s = node.args[1], node.args[2], node.args[3]
        if isinstance(w, Node) and isinstance(s, Node) and isinstance(bs, int):
            return (w, bs, s)
        return None

    def _get_out_features(self, node: Node) -> Optional[int]:
        """Extract output dimension from node metadata."""
        val = node.meta.get("val")
        return (
            val.shape[-1]
            if isinstance(val, torch.Tensor) and len(val.shape) >= 2
            else None
        )

    def _validate_group(self, group: List[Node]) -> bool:
        """
        Validate that a group of operations can be safely fused.

        Checks:
        - Group size is within [min_fusion_size, max_fusion_size]
        - All operations have compatible dtypes
        - All operations have compatible devices
        - All weights have the same input dimension (k)
        """
        if len(group) < self.min_fusion_size:
            return False

        has_metadata = all("val" in node.meta for node in group)

        if has_metadata:
            # Verify dtype compatibility
            dtypes = {node.meta["val"].dtype for node in group}
            if len(dtypes) > 1:
                return False

            # Verify device compatibility
            devices = {str(node.meta["val"].device) for node in group}
            if len(devices) > 1:
                return False

            # Verify output dimensions are extractable
            if not all(self._get_out_features(node) for node in group):
                return False

        # Verify all operations have valid parameters
        params_list = [self._get_params(n) for n in group]
        if not all(params_list):
            return False

        weights = [p[0] for p in params_list]

        # Verify weight input dimensions match (required for concatenation)
        if has_metadata and all("val" in w.meta for w in weights):
            k_dims = [w.meta["val"].shape[-1] for w in weights]
            if len(set(k_dims)) > 1:
                return False

        return True

    def _find_fuseable_groups(self, graph_module: GraphModule) -> List[List[Node]]:
        """
        Identify groups of int4mm operations that can be fused together.

        Grouping strategy:
        1. Iterate through all int4mm operations in the graph
        2. Group operations by (input_node, block_size)
           - Same input_node: operations consume the same preprocessed input
           - Same block_size: required for compatible quantization
        3. Split large groups into chunks of max_fusion_size
        4. Validate each group before including it

        Returns:
            List of fuseable groups, where each group is a list of nodes
        """
        groups: Dict[Tuple[Node, int], List[Node]] = defaultdict(list)

        for node in graph_module.graph.nodes:
            if not self._is_int4mm(node):
                continue

            # Extract the immediate input node
            if not node.args or not isinstance(node.args[0], Node):
                continue
            input_node = node.args[0]

            # Extract block_size parameter
            params = self._get_params(node)
            if params:
                groups[(input_node, params[1])].append(node)

        # Split groups by max_fusion_size and validate
        result = []
        for ops in groups.values():
            for i in range(0, len(ops), self.max_fusion_size):
                group = ops[i : i + self.max_fusion_size]
                if len(group) >= self.min_fusion_size and self._validate_group(group):
                    result.append(group)

        return result

    def _get_last_placeholder(self, graph_module: GraphModule) -> Optional[Node]:
        """Find the last placeholder node in the graph."""
        last = None
        for n in graph_module.graph.nodes:
            if n.op == "placeholder":
                last = n
            else:
                break
        return last

    def _compute_cat_metadata(
        self, nodes: List[Node], dim: int
    ) -> Optional[torch.Tensor]:
        """
        Compute metadata for concatenating tensors along a dimension.

        Args:
            nodes: Source nodes to concatenate
            dim: Dimension to concatenate along (0 or 1)

        Returns:
            Fake tensor with concatenated shape, or None if metadata unavailable
        """
        if not all("val" in n.meta for n in nodes):
            return None

        # Get reference properties from first node
        ref_val = nodes[0].meta["val"]
        shapes = [n.meta["val"].shape for n in nodes]

        # Compute concatenated shape
        result_shape = list(shapes[0])
        result_shape[dim] = sum(s[dim] for s in shapes)

        # Use device='meta' to support dynamic shapes with SymInt.
        # Concrete device objects (e.g., 'cuda:0') fail when shape dimensions
        # are symbolic rather than concrete integers.
        return torch.empty(
            tuple(result_shape), dtype=ref_val.dtype, device="meta"
        )

    def _fuse_group(self, graph_module: GraphModule, group: List[Node]) -> bool:
        """
        Fuse a group of int4mm operations into a single operation.

        Transformation:
        1. Concatenate all weights along output dimension (dim=0)
        2. Concatenate all scales along output dimension (dim=1)
        3. Create single fused int4mm with concatenated weights/scales
        4. Split fused output back to individual results
        5. Replace original operations with split results

        Args:
            graph_module: The graph to modify
            group: List of int4mm nodes to fuse

        Returns:
            True if fusion succeeded, False otherwise
        """
        try:
            params = self._get_params(group[0])
            if not params:
                return False
            _, block_size, _ = params

            # Extract weights and scales from all operations
            params_list = [self._get_params(n) for n in group]
            weights = [p[0] for p in params_list]
            scales = [p[2] for p in params_list]

            # Compute output features once at the start for efficiency.
            # These values are used in multiple places: fused_mm metadata,
            # split_points calculation, and validation.
            output_features = [self._get_out_features(n) for n in group]
            if not all(output_features):
                return False

            # Create concatenated weights and scales.
            # Insert before the FIRST original int4mm node to maintain topological order.
            # The original int4mm nodes are already correctly placed after their inputs
            # (shared_input, weights, scales), so inserting near them preserves validity.
            first_int4mm = min(group, key=lambda n: list(graph_module.graph.nodes).index(n))
            with graph_module.graph.inserting_before(first_int4mm):
                # IMPORTANT: Use args, not kwargs, for cat operations.
                # AOT Inductor expects positional arguments and may not correctly
                # handle kwargs for aten::cat, leading to empty tensor outputs.
                fused_weight = graph_module.graph.call_function(
                    torch.ops.aten.cat.default,
                    args=(weights, 0),
                )
                # Compute metadata for concatenated weights
                if (val := self._compute_cat_metadata(weights, dim=0)) is not None:
                    fused_weight.meta["val"] = val

                fused_scale = graph_module.graph.call_function(
                    torch.ops.aten.cat.default,
                    args=(scales, 1),
                )
                # Compute metadata for concatenated scales
                if (val := self._compute_cat_metadata(scales, dim=1)) is not None:
                    fused_scale.meta["val"] = val

                # Create fused matmul operation AFTER fused_weight and fused_scale
                # to maintain topological order (fused_mm depends on fused_weight and fused_scale)
                fused_mm = graph_module.graph.call_function(
                    torch.ops.aten._weight_int4pack_mm.default,
                    args=(group[0].args[0], fused_weight, block_size, fused_scale),
                )

                # Set output metadata with total concatenated output dimension.
                # Use device='meta' to support dynamic shapes with SymInt.
                if "val" in group[0].meta:
                    base_shape = group[0].meta["val"].shape[:-1]
                    total_out = sum(output_features)
                    fused_mm.meta["val"] = torch.empty(
                        base_shape + (total_out,),
                        dtype=group[0].meta["val"].dtype,
                        device='meta',
                    )

            # Calculate split points to divide the fused output back to individual results.
            # For N operations, we need N-1 split points at cumulative output boundaries.
            split_points = []
            offset = 0
            for out_feat in output_features[:-1]:
                offset += out_feat
                split_points.append(offset)

            # Split fused output back to individual results
            with graph_module.graph.inserting_after(fused_mm):
                split_list = graph_module.graph.call_function(
                    torch.ops.aten.tensor_split.indices,
                    args=(fused_mm, split_points, -1),
                )
                # Set metadata for split operation (list of original output tensors)
                if "val" in fused_mm.meta:
                    split_list.meta["val"] = [n.meta["val"] for n in group if "val" in n.meta]

            # Replace each original operation with its corresponding split result.
            # IMPORTANT: tensor_split creates non-contiguous views with incorrect strides.
            # For example, shape [batch, seq, hidden] gets strides [seq*3*hidden, 3*hidden, 1]
            # instead of the expected [seq*hidden, hidden, 1]. This causes issues during
            # AOTI compilation where kernels may assume contiguous memory layout.
            # We add .contiguous() after each getitem to ensure proper memory layout.
            for i, node in enumerate(group):
                with graph_module.graph.inserting_after(split_list):
                    getitem = graph_module.graph.call_function(
                        operator.getitem,
                        args=(split_list, i),
                    )
                    # Set metadata for getitem (non-contiguous view from tensor_split)
                    if "val" in node.meta:
                        getitem.meta["val"] = node.meta["val"]

                # Add contiguous() AFTER getitem to ensure proper memory layout.
                # This is critical for encoder patterns (seq_len > 1) where
                # the non-contiguous strides from tensor_split would cause
                # incorrect memory access in downstream operations.
                with graph_module.graph.inserting_after(getitem):
                    contiguous = graph_module.graph.call_function(
                        torch.ops.aten.contiguous.default,
                        args=(getitem,),
                    )
                    # Set metadata for contiguous output.
                    # The output has the same shape but with proper contiguous strides.
                    if "val" in node.meta:
                        # Create a contiguous version of the metadata tensor
                        orig_val = node.meta["val"]
                        contiguous.meta["val"] = torch.empty(
                            orig_val.shape,
                            dtype=orig_val.dtype,
                            device="meta",
                        )

                node.replace_all_uses_with(contiguous)

            # Remove original operations
            for node in group:
                graph_module.graph.erase_node(node)

            return True

        except Exception as e:
            # Log fusion failures with full context for debugging.
            # Fusion is an optimization, so we gracefully skip failed groups,
            # but we must provide visibility into failures to help developers
            # identify and fix issues (e.g., graph structure problems, metadata bugs).
            import logging
            logger = logging.getLogger(__name__)
            logger.warning(
                f"Failed to fuse INT4 group of {len(group)} operations: {type(e).__name__}: {e}",
                exc_info=True
            )
            return False
