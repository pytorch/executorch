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

_SPLITK_LKV_THRESHOLD = 2048


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

    # fp8 dtypes a KV cache may be stored in (read by the SDPA kernels, which
    # cast to bf16 on load). When the model feeds fp8 K/V through a bf16 cast to
    # satisfy aten SDPA's same-dtype export check, we rewire the triton kernel to
    # read the fp8 source directly (and DCE the now-dead cast).
    _FP8_DTYPES = (torch.float8_e5m2, torch.float8_e4m3fn)

    @staticmethod
    def _unwrap_fp8_cast(arg):
        """If ``arg`` is an fp8->bf16 dtype-conversion node, return its fp8
        source; otherwise return ``arg`` unchanged.

        The model casts fp8 KV up to bf16 only to pass aten SDPA's same-dtype
        meta check during export. The triton SDPA kernels read fp8 natively, so
        we point them at the fp8 source and let the bf16 cast become dead code.
        """
        if not isinstance(arg, Node) or arg.op != "call_function":
            return arg
        val = arg.meta.get("val")
        if val is None or getattr(val, "dtype", None) != torch.bfloat16:
            return arg
        if not arg.args:
            return arg
        src = arg.args[0]
        if not isinstance(src, Node):
            return arg
        src_val = src.meta.get("val")
        if (
            src_val is not None
            and getattr(src_val, "dtype", None)
            in ReplaceEdgeOpWithTritonOpPass._FP8_DTYPES
        ):
            return src
        return arg

    @staticmethod
    def _pick_sdpa_kernel(node: Node):
        """Choose between standard SDPA and split-K flash-decoding.

        Split-K partitions the KV sequence across many CTAs for better GPU
        utilization at decode time (L_q=1). It uses split-K for decode
        whenever L_kv >= 2048 (both sliding-window ring buffers and full
        caches); the standard kernel underfills the GPU at L_q=1.
        """
        q_shape = node.args[0].meta["val"].shape
        k_shape = node.args[1].meta["val"].shape
        L_q, D = q_shape[2], q_shape[3]
        L_kv = k_shape[2]

        if (
            isinstance(L_q, int)
            and L_q == 1
            and isinstance(L_kv, int)
            and L_kv >= _SPLITK_LKV_THRESHOLD  # >= so sliding L_kv=2048 uses split-K too
            and D > 0
            and (D & (D - 1)) == 0  # power of 2
        ):
            logger.info(f"Using split-K decode SDPA (L_kv={L_kv}, D={D})")
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

        args = node.args
        dead_casts = []
        if target == exir_ops.edge.aten.scaled_dot_product_attention.default:
            triton_kernel_fn = self._pick_sdpa_kernel(node)
            # Rewire fp8 K/V: read the fp8 cache directly instead of the bf16
            # cast the model inserted to satisfy aten SDPA's export dtype check.
            new_args = list(node.args)
            for i in (1, 2):  # key, value
                if i < len(new_args):
                    src = self._unwrap_fp8_cast(new_args[i])
                    if src is not new_args[i]:
                        dead_casts.append(new_args[i])
                        new_args[i] = src
            args = tuple(new_args)
            if dead_casts:
                print(
                    f"[FP8KV_REWIRE] {triton_kernel_fn}: rewired "
                    f"{len(dead_casts)} K/V arg(s) to fp8 source",
                    flush=True,
                )

        # Create a new node with the Triton kernel
        with graph_module.graph.inserting_before(node):
            # The triton_kernel_fn is already registered as a custom op via @triton_op
            # We can call it directly
            new_node = graph_module.graph.call_function(
                triton_kernel_fn,
                args=args,
                kwargs=node.kwargs,
            )

            # Copy metadata from original node
            new_node.meta = node.meta.copy()

        # Replace all uses of the old node with the new node
        node.replace_all_uses_with(new_node)

        # Remove the old node
        graph_module.graph.erase_node(node)

        # Erase the now-dead bf16 casts we bypassed (only if fully unused), so
        # they don't get codegen'd into a full-cache fp8->bf16 copy each step.
        for cast in dead_casts:
            if isinstance(cast, Node) and len(cast.users) == 0:
                graph_module.graph.erase_node(cast)
