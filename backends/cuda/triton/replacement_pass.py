# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
Graph Transformation Pass for Triton Kernel Replacement.

This pass replaces ATen operators with optimized Triton kernels in the graph.
"""

import logging

import torch
from executorch.backends.cuda.triton.kernels.sdpa import (
    _prepare_mask_params,
    _validate_qkv_shapes,
    _validate_sdpa_inputs,
)
from executorch.exir.dialects._ops import ops as exir_ops

from torch.fx import GraphModule, Node
from torch.fx.passes.infra.pass_base import PassBase, PassResult

logger = logging.getLogger(__name__)
triton = torch.ops.triton

# Global mapping from edge dialect operators to Triton kernel functions
EDGE_TO_TRITON_KERNELS = {
    exir_ops.edge.aten.scaled_dot_product_attention.default: triton.sdpa,
    exir_ops.edge.aten.topk.default: triton.topk,
}


_SPLITK_LKV_THRESHOLD = 256


class ReplaceEdgeOpWithTritonOpPass(PassBase):
    """
    Pass to replace ATen operators with Triton kernels.

    This pass scans the graph for Edge operators that have registered Triton
    replacements using EDGE_TO_TRITON_KERNELS and replaces them with the
    optimized Triton implementations.
    """

    def __init__(self):
        """Initialize the pass."""
        super().__init__()
        self._replacement_count = 0

    def call(self, graph_module: GraphModule) -> PassResult:
        """
        Execute the pass on the graph module.

        Args:
            graph_module: The graph module to transform

        Returns:
            PassResult indicating success/failure and the modified graph module
        """
        self._replacement_count = 0
        modified = False

        if not EDGE_TO_TRITON_KERNELS:
            return PassResult(graph_module, False)

        # Iterate through all nodes in the graph
        for node in graph_module.graph.nodes:
            if self._should_replace_node(node):
                try:
                    self._replace_node_with_triton(graph_module, node)
                    modified = True
                    self._replacement_count += 1
                except Exception as e:
                    logger.warning(f"Failed to replace node {node.name}: {e}")
                    # Continue with other replacements even if one fails

        if modified:
            # Recompile the graph module after modifications
            graph_module.recompile()

        logger.info(f"Replaced {self._replacement_count} nodes with Triton kernels")

        return PassResult(graph_module, modified)

    # The topk kernel loads an entire row into a single thread block via
    # tl.arange(0, BLOCK). For large N (e.g., vocab-sized topk with N=248K),
    # this exceeds Triton's register/shared memory limits. Skip replacement
    # for rows larger than this threshold.
    _TOPK_MAX_N = 4096

    @staticmethod
    def _pick_sdpa_kernel(node: Node):
        """Choose between standard SDPA and split-K flash-decoding.

        Split-K partitions the KV sequence across many CTAs for better GPU
        utilization at decode time (L_q=1). It wins when L_kv is large
        (full-attention KV caches) but loses to the standard kernel for
        small L_kv (sliding-window ring buffers) due to the overhead of
        allocating partial buffers and running the reduction kernel.

        TODO(gasoonjia): Benchmarking to determine the optimal
        implementation for each shape.
        """
        q_shape = node.args[0].meta["val"].shape
        k_shape = node.args[1].meta["val"].shape
        L_q, D = q_shape[2], q_shape[3]
        L_kv = k_shape[2]

        # TODO: Re-enable split-K after validating ROCm Voxtral decode numerics.
        if (
            torch.version.hip is None
            and isinstance(L_q, int)
            and L_q == 1
            and isinstance(L_kv, int)
            and L_kv >= _SPLITK_LKV_THRESHOLD
            and D > 0
            and (D & (D - 1)) == 0  # power of 2
        ):
            return triton.sdpa_decode_splitk

        return triton.sdpa

    def _should_replace_node(self, node: Node) -> bool:
        """
        Check if a node should be replaced with a Triton kernel.

        Args:
            node: The node to check

        Returns:
            True if the node should be replaced
        """
        if node.op != "call_function":
            return False

        if node.target not in EDGE_TO_TRITON_KERNELS:
            return False

        # The topk kernel loads an entire row into one thread block.
        # Skip replacement for large N that would exceed Triton limits.
        if node.target == exir_ops.edge.aten.topk.default:
            input_shape = node.args[0].meta["val"].shape
            dim = node.args[2] if len(node.args) > 2 else -1
            N = input_shape[dim]
            if N > self._TOPK_MAX_N:
                logger.info(f"Skipping topk replacement: N={N} > {self._TOPK_MAX_N}")
                return False

        if node.target == exir_ops.edge.aten.scaled_dot_product_attention.default:
            return self._sdpa_kernel_supports(node)

        return True

    @classmethod
    def _sdpa_kernel_supports(cls, node: Node) -> bool:
        """Whether the Triton SDPA kernel the pass would pick accepts this call.

        Runs the kernels' own input checks on the fake tensors, so calls they
        reject stay with the regular lowering instead of failing the export.
        """
        # The kernel picker reads query and key by position.
        if len(node.args) < 3:
            return False
        names = ("attn_mask", "dropout_p", "is_causal", "scale", "enable_gqa")
        defaults = (None, 0.0, False, None, False)
        attn_mask, dropout_p, is_causal, _, enable_gqa = (
            node.args[3 + i] if 3 + i < len(node.args) else node.kwargs.get(name, d)
            for i, (name, d) in enumerate(zip(names, defaults))
        )
        query, key, value = (arg.meta["val"] for arg in node.args[:3])
        try:
            _validate_sdpa_inputs(query, key, value, dropout_p, enable_gqa)
            splitk = cls._pick_sdpa_kernel(node) is triton.sdpa_decode_splitk
            # Split-K groups query heads over key/value heads on its own.
            B, _, _, L_q, L_kv, _, _ = _validate_qkv_shapes(
                query, key, value, enable_gqa or splitk
            )
            if is_causal and L_q != L_kv and not splitk:
                raise RuntimeError(
                    f"Causal masking requires L_q == L_kv; got L_q={L_q}, L_kv={L_kv}."
                )
            mask = None if attn_mask is None else attn_mask.meta["val"]
            _prepare_mask_params(mask, B, L_q, L_kv)
        except RuntimeError as e:
            logger.info(f"Skipping SDPA replacement: {e}")
            return False
        return True

    def _replace_node_with_triton(self, graph_module: GraphModule, node: Node) -> None:
        """
        Replace an edge dialect node with a Triton kernel call.

        Args:
            graph_module: The graph module containing the node
            node: The node to replace
        """
        # Get the target operator (should be an exir_ops edge dialect op)
        target = node.target

        # Get the replacement kernel
        if target not in EDGE_TO_TRITON_KERNELS:
            raise ValueError(f"No replacement kernel found for {target}")

        triton_kernel_fn = EDGE_TO_TRITON_KERNELS[target]

        if target == exir_ops.edge.aten.scaled_dot_product_attention.default:
            triton_kernel_fn = self._pick_sdpa_kernel(node)
            if triton_kernel_fn is triton.sdpa_decode_splitk:
                L_kv, D = node.args[1].meta["val"].shape[2:]
                logger.info(f"Using split-K decode SDPA (L_kv={L_kv}, D={D})")

        # Create a new node with the Triton kernel
        with graph_module.graph.inserting_before(node):
            # The triton_kernel_fn is already registered as a custom op via @triton_op
            # We can call it directly
            new_node = graph_module.graph.call_function(
                triton_kernel_fn,
                args=node.args,
                kwargs=node.kwargs,
            )

            # Copy metadata from original node
            new_node.meta = node.meta.copy()

        # Replace all uses of the old node with the new node
        node.replace_all_uses_with(new_node)

        # Remove the old node
        graph_module.graph.erase_node(node)
