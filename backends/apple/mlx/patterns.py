#
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.
#

"""
MLX Pattern Handlers - pattern-based op lowering for fused operations.

This module contains pattern handlers that match multi-node subgraphs and lower
them to optimized MLX operations. Examples include:
- SLICE_UPDATE: In-place slice updates for KV cache
- UPDATE_CACHE: KV cache updates via llama.update_cache custom op
- SDPA: Scaled Dot-Product Attention with optional GQA
- QUANTIZED_LINEAR: Fused dequantize + linear for quantized models
- QUANTIZED_EMBEDDING: Fused dequantize + embedding for quantized models
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import torch
from executorch.backends.apple.mlx.program_builder import (
    emit_stop_position,
    get_aten_target_normalized,
    MLXProgramBuilder,
    PatternHandler,
    REGISTRY,
    Slot,
    torch_dtype_to_scalar_type,
)
from executorch.backends.apple.mlx.serialization.mlx_graph_schema import (
    IdCopyNode,
    IndexUpdateNode,
    IntOrVid,
    QuantizedGatherNode,
    QuantizedLinearNode,
    SdpaNode,
    SliceUpdateNode,
)
from torch.export.exported_program import ExportedProgram
from torch.fx.node import Node

# When True, always serialize the biases tensor for quantized ops (existing behavior).
# When False, use scale_only=True optimization when zero_point is all zeros,
# which avoids serializing the biases tensor (C++ runtime computes: biases = -scales * 2^(bits-1)).
QUANTIZED_SERIALIZE_BIASES = True


# =============================================================================
# Pattern-matching utilities
# =============================================================================


def match_target(node: Node, op) -> bool:
    """Check if a node's normalized aten target matches the given op."""
    return node.op == "call_function" and get_aten_target_normalized(node.target) == op


def has_single_user(node: Node) -> bool:
    """Check if a node has exactly one consumer."""
    return len(node.users) == 1


@dataclass
class OpStep:
    """One step in a backward walk through the graph."""

    op: Any
    optional: bool = False
    args: Optional[Dict[int, Any]] = field(default=None)


def walk_back(
    node: Node,
    steps: List[OpStep],
) -> Optional[Tuple[Node, List[Node]]]:
    """Walk backwards through a chain of single-user ops.

    Starting from *node*, try to match each step against the current node.
    At every matched step the walk advances to ``cur.args[0]``.  Optional
    steps are silently skipped when they don't match.

    Returns:
        ``(base_node, body_nodes)`` if the full chain matches, else ``None``.
        *base_node* is the input to the first (deepest) op in the chain.
        *body_nodes* are the matched intermediate nodes (in walk order).
    """
    body: List[Node] = []
    cur = node

    for step in steps:
        if not isinstance(cur, Node):
            return None

        if match_target(cur, step.op):
            if not has_single_user(cur):
                return None
            if step.args:
                for idx, expected in step.args.items():
                    if len(cur.args) <= idx or cur.args[idx] != expected:
                        return None
            body.append(cur)
            cur = cur.args[0]
        elif step.optional:
            continue
        else:
            return None

    if not isinstance(cur, Node):
        return None

    return cur, body


@REGISTRY.register_pattern(name="SLICE_UPDATE")
class SliceUpdateHandler(PatternHandler):
    """
    Pattern for in-place slice updates (used for KV cache).

    Matches: slice -> copy -> slice_scatter
    Where slice and slice_scatter operate on the same buffer.
    """

    def __init__(
        self,
        head: Node,
        body: List[Node],
        dst: Node,
        update: Node,
        axis: int,
        start: Any,
        stop: Any,
    ):
        super().__init__(head, body)
        self.dst = dst
        self.update = update
        self.axis = axis
        self.start = start
        self.stop = stop

    @classmethod
    def maybe_create(  # noqa: C901
        cls, ep: ExportedProgram, head: Node
    ) -> Optional["SliceUpdateHandler"]:
        slice_scatter_node = head
        if not match_target(slice_scatter_node, torch.ops.aten.slice_scatter.default):
            return None

        # Slice scatter should write to a mutable input/buffer to be a slice update.
        # NOTE: We also check user_inputs_to_mutate for the case where the mutated
        # buffer is passed as a user input (after tag_mutated_buffer tags it).
        if (slice_scatter_node.name not in ep.graph_signature.buffers_to_mutate) and (
            slice_scatter_node.name not in ep.graph_signature.user_inputs_to_mutate
        ):
            return None

        if len(slice_scatter_node.args) != 5:
            return None
        ss_dst, ss_src, ss_axis, ss_start, ss_end = slice_scatter_node.args

        copy_node = ss_src
        if not match_target(copy_node, torch.ops.aten.copy.default):
            return None
        if not has_single_user(copy_node):
            return None
        if len(copy_node.args) != 2:
            return None
        c_dst, c_src = copy_node.args

        slice_node = c_dst
        if not match_target(slice_node, torch.ops.aten.slice.Tensor):
            return None
        if not has_single_user(slice_node):
            return None
        if len(slice_node.args) != 4:
            return None
        s_src, s_axis, s_start, s_end = slice_node.args

        # Slice should be on a buffer/input to be a slice-update.
        # After tag_mutated_buffer runs, the buffer may show up in user_inputs too.
        if (s_src.name not in ep.graph_signature.inputs_to_buffers) and (
            s_src.name not in ep.graph_signature.user_inputs
        ):
            # Partitioned subgraph case: mutation info may be empty but the
            # buffer is passed as a placeholder input. Check that s_src is a placeholder.
            if s_src.op != "placeholder":
                return None

        # We should be slice / slice-scatter the same input/buffer
        if s_src.name in ep.graph_signature.inputs_to_buffers:
            buf = ep.graph_signature.inputs_to_buffers[s_src.name]
            buf_mut = ep.graph_signature.buffers_to_mutate.get(slice_scatter_node.name)
            if buf_mut is not None and buf != buf_mut:
                return None

        if s_src.name in ep.graph_signature.user_inputs:
            # If there's mutation tracking, verify consistency
            # If not (partitioned subgraph), allow the pattern
            pass

        if (
            (s_src != ss_dst)
            or (s_axis != ss_axis)
            or (s_start != ss_start)
            or (s_end != ss_end)
        ):
            return None

        head = slice_scatter_node
        body = [slice_node, copy_node]
        dst = s_src
        update = c_src
        axis = s_axis
        start = s_start
        stop = s_end
        return SliceUpdateHandler(head, body, dst, update, axis, start, stop)

    def __call__(self, P: MLXProgramBuilder, n: Node) -> Slot:
        assert n == self.head
        dst, update, axis, start, stop = P.slot_map(
            [self.dst, self.update, self.axis, self.start, self.stop]
        )
        P.emit(
            SliceUpdateNode(
                dst=P.slot_to_tid(dst),
                update=P.slot_to_tid(update),
                axis=P.to_int_or_vid(axis),
                start=P.to_int_or_vid(start),
                stop=P.to_int_or_vid(stop),
            )
        )
        # The slice_scatter node output is logically the same as the dst buffer
        # (it's an in-place update). If the node already has a slot assigned
        # (e.g., it's an output), we need to emit an ID_COPY to map dst -> output slot.
        existing_slot = P.slot_manager.get_slot(n)
        if existing_slot is not None and existing_slot != dst:
            # Node already has a slot (e.g., Output), need to copy dst to it
            P.emit(
                IdCopyNode(
                    x=P.slot_to_tid(dst),
                    out=P.slot_to_tid(existing_slot),
                )
            )
            return existing_slot
        else:
            # No existing slot or same as dst - just set to dst
            P.set_slot(n, dst)
            return dst


# =============================================================================
# INDEX_UPDATE pattern
# =============================================================================


@REGISTRY.register_pattern(name="INDEX_UPDATE")
class IndexUpdateHandler(PatternHandler):
    """
    Pattern for index-based updates on mutable buffers.

    Matches: aten.index_copy.default on a mutable buffer
    Lowers to IndexUpdateNode which performs in-place update.
    """

    def __init__(
        self,
        head: Node,
        body: List[Node],
        dst: Node,
        update: Node,
        indices: Node,
        axis: int,
    ):
        super().__init__(head, body)
        self.dst = dst
        self.update = update
        self.indices = indices
        self.axis = axis

    @classmethod
    def maybe_create(  # noqa: C901
        cls, ep: ExportedProgram, head: Node
    ) -> Optional["IndexUpdateHandler"]:
        index_copy_node = head
        if not match_target(index_copy_node, torch.ops.aten.index_copy.default):
            return None

        # index_copy should write to a mutable input/buffer to be an index update.
        if (index_copy_node.name not in ep.graph_signature.buffers_to_mutate) and (
            index_copy_node.name not in ep.graph_signature.user_inputs_to_mutate
        ):
            return None

        # index_copy(dst, axis, indices, update)
        if len(index_copy_node.args) != 4:
            return None
        dst, axis, indices, update = index_copy_node.args

        # axis must be a literal int
        if not isinstance(axis, int):
            return None

        return cls(
            head=index_copy_node,
            body=[],
            dst=dst,
            update=update,
            indices=indices,
            axis=axis,
        )

    def __call__(self, P: MLXProgramBuilder, n: Node) -> Slot:
        assert n == self.head
        dst, update, indices = P.slot_map([self.dst, self.update, self.indices])

        P.emit(
            IndexUpdateNode(
                dst=P.slot_to_tid(dst),
                update=P.slot_to_tid(update),
                indices=P.slot_to_tid(indices),
                axis=self.axis,
            )
        )

        # index_copy returns the updated dst (same buffer, in-place update)
        existing_slot = P.slot_manager.get_slot(n)
        if existing_slot is not None and existing_slot != dst:
            P.emit(
                IdCopyNode(
                    x=P.slot_to_tid(dst),
                    out=P.slot_to_tid(existing_slot),
                )
            )
            return existing_slot
        else:
            P.set_slot(n, dst)
            return dst


# =============================================================================
# ET_KV_CACHE_UPDATE pattern
# =============================================================================


@REGISTRY.register_pattern(name="ET_KV_CACHE_UPDATE")
class ETKVCacheUpdateHandler(PatternHandler):
    """
    Pattern for KV cache updates using torch.ops.mlx.kv_cache_update.

    Matches the full chain: auto_functionalized → getitem[1] → alias
    HEAD = aten.alias.default (the final alias node that becomes output)

    Graph structure after run_decompositions({}):
        auto_func = auto_functionalized_v2(mlx.kv_cache_update, new_values=k_val, ...)
        getitem_1 = getitem(auto_func, 1)    # Updated cache alias
        alias_2 = alias(getitem_1)           # HEAD - the output alias

    By making alias the HEAD, we emit IdCopyNode directly and avoid
    the ContiguousNode that _clone_handler would emit.
    """

    def __init__(
        self,
        head: Node,
        body: List[Node],
        cache: Node,
        update: Node,
        start_pos: Any,
        getitem_node: Node,
    ):
        super().__init__(head, body)
        self.cache = cache  # The cache buffer [B, H, S, D]
        self.update = update  # The update tensor [B, H, S_step, D]
        self.start_pos = start_pos  # Start position (int or SymInt)
        self.getitem_node = getitem_node  # getitem[1] node

    @staticmethod
    def _is_auto_func_et_kv_cache_update(node: Node) -> bool:
        """Check if a node is auto_functionalized_v2 wrapping mlx.kv_cache_update."""
        if node.op != "call_function":
            return False
        target_str = str(node.target)
        if "auto_functionalized" not in target_str:
            return False
        if len(node.args) < 1:
            return False
        func_arg = node.args[0]
        func_str = str(func_arg) if func_arg else ""
        return "kv_cache_update" in func_str and "mlx" in func_str

    @classmethod
    def maybe_create(
        cls, ep: ExportedProgram, head: Node
    ) -> Optional["ETKVCacheUpdateHandler"]:
        """
        Match the ET_KV_CACHE_UPDATE pattern.

        Pattern (HEAD = alias):
            auto_func = auto_functionalized_v2(mlx.kv_cache_update, ...)
            getitem_1 = getitem(auto_func, 1)    # Updated cache
            alias = alias(getitem_1)             # HEAD
        """
        # HEAD must be aten.alias.default or aten.alias_copy.default
        if not match_target(head, torch.ops.aten.alias.default) and not match_target(
            head, torch.ops.aten.alias_copy.default
        ):
            return None

        alias_node = head

        # alias's input should be a getitem node with idx=1
        if len(alias_node.args) < 1 or not isinstance(alias_node.args[0], Node):
            return None

        potential_getitem = alias_node.args[0]
        if potential_getitem.op != "call_function" or "getitem" not in str(
            potential_getitem.target
        ):
            return None

        if len(potential_getitem.args) < 2 or potential_getitem.args[1] != 1:
            return None

        getitem_node = potential_getitem

        # getitem's source should be auto_functionalized_v2 wrapping mlx.kv_cache_update
        if len(getitem_node.args) < 1 or not isinstance(getitem_node.args[0], Node):
            return None

        auto_func_node = getitem_node.args[0]
        if not cls._is_auto_func_et_kv_cache_update(auto_func_node):
            return None

        # Extract info from auto_functionalized_v2 kwargs
        kwargs = auto_func_node.kwargs
        new_values_node = kwargs.get("new_values")
        start_pos_node = kwargs.get("start_pos")
        all_bases = kwargs.get("_all_bases", [])

        if not new_values_node or not all_bases:
            return None

        cache_node = all_bases[0]

        # Build the pattern body: auto_func and getitem nodes
        body = [auto_func_node, getitem_node]

        return ETKVCacheUpdateHandler(
            head=head,
            body=body,
            cache=cache_node,
            update=new_values_node,
            start_pos=start_pos_node,
            getitem_node=getitem_node,
        )

    def __call__(self, P: MLXProgramBuilder, n: Node) -> Slot:
        assert n == self.head

        # Get slots for cache and update
        # cache is [B, H, S, D] (BHSD layout - mutable buffer)
        # update is [B, H, S_step, D] (new values to insert)
        cache_slot = P.slot_map([self.cache])[0]
        update_slot = P.slot_map([self.update])[0]

        # start_pos could be an int, SymInt, or a Node (from item())
        if isinstance(self.start_pos, Node):
            start_slot = P.slot_map([self.start_pos])[0]
        else:
            start_slot = self.start_pos

        # Get the output slot for this node (the alias HEAD)
        out_slot = P.make_or_get_slot(n)

        # Calculate stop = start + seq_len
        # update is [B, H, S_step, D], so seq_len is dim 2
        update_meta = self.update.meta.get("val")
        stop_slot = emit_stop_position(
            P,
            start=start_slot,
            length_tensor=update_slot,
            length_dim=2,  # S_step is dim 2 in [B, H, S_step, D]
            length_meta=update_meta,
        )

        # SliceUpdateNode on axis=2
        # cache is [B, H, S, D], update is [B, H, S_step, D]
        # This updates cache[:, :, start:stop, :] = update
        P.emit(
            SliceUpdateNode(
                dst=P.slot_to_tid(cache_slot),
                update=P.slot_to_tid(update_slot),
                axis=IntOrVid.from_literal(2),  # S dimension in [B, H, S, D]
                start=P.to_int_or_vid(start_slot),
                stop=P.to_int_or_vid(stop_slot),
            )
        )

        # Emit IdCopyNode from cache to output (NO ContiguousNode!)
        P.emit(
            IdCopyNode(
                x=P.slot_to_tid(cache_slot),
                out=P.slot_to_tid(out_slot),
            )
        )

        return out_slot


# =============================================================================
# SDPA pattern
# =============================================================================


@REGISTRY.register_pattern(name="SDPA")
class SDPAHandler(PatternHandler):
    """
    Pattern for Scaled Dot Product Attention with optional GQA.

    Matches: scaled_dot_product_attention
    Optionally with repeat_interleave for grouped query attention.
    """

    def __init__(
        self,
        head: Node,
        body: List[Node],
        q_node: Node,
        k_node: Node,
        v_node: Node,
    ):
        super().__init__(head, body)
        self.q_node = q_node
        self.k_node = k_node
        self.v_node = v_node

    @classmethod
    def _parse_sdpa_args_and_kwargs(cls, sdpa_node: Node):
        q, k, v = sdpa_node.args[0:3]
        attn_mask = sdpa_node.args[3] if len(sdpa_node.args) > 3 else None
        dropout_p = sdpa_node.args[4] if len(sdpa_node.args) > 4 else 0.0
        is_causal = sdpa_node.args[5] if len(sdpa_node.args) > 5 else False
        enable_gqa = sdpa_node.args[6] if len(sdpa_node.args) > 6 else False
        scale = sdpa_node.kwargs.get("scale", None)
        return q, k, v, attn_mask, dropout_p, is_causal, scale, enable_gqa

    @classmethod
    def _try_unwrap_repeat_kv(cls, node: Node) -> Optional[Tuple[Node, List[Node]]]:
        """Try to unwrap a HuggingFace repeat_kv pattern.

        HuggingFace's repeat_kv expands KV heads for grouped query attention:
            hidden_states[:, :, None, :, :].expand(B, n_kv, n_rep, T, D)
            .clone().reshape(B, n_heads, T, D)

        In Edge IR this becomes:
            unsqueeze_copy(x, 2) → expand_copy → clone → view_copy

        Returns:
            (base_node, body_nodes) if pattern matches, else None.
            base_node is the original [B, n_kv, T, D] tensor.
            body_nodes are the intermediate nodes to absorb.
        """
        return walk_back(
            node,
            [
                OpStep(torch.ops.aten.view.default),
                OpStep(torch.ops.aten.clone.default, optional=True),
                OpStep(torch.ops.aten.expand.default),
                OpStep(torch.ops.aten.unsqueeze.default, args={1: 2}),
            ],
        )

    @classmethod
    def maybe_create(cls, ep: ExportedProgram, head: Node) -> Optional["SDPAHandler"]:
        sdpa_node = head
        if not match_target(
            sdpa_node, torch.ops.aten.scaled_dot_product_attention.default
        ):
            return None

        q, k, v, _, _, _, _, _ = cls._parse_sdpa_args_and_kwargs(sdpa_node)

        # Detect grouped kv attention pattern with repeat_interleave before SDPA
        is_grouped_kv = False
        k_base = k
        v_base = v
        body: List[Node] = []
        if (
            match_target(k, torch.ops.aten.repeat_interleave.self_int)
            and has_single_user(k)
            and (len(k.args) == 3)
            and (len(k.kwargs) == 0)
            and match_target(v, torch.ops.aten.repeat_interleave.self_int)
            and has_single_user(v)
            and (len(v.args) == 3)
            and (len(v.kwargs) == 0)
        ):
            k_unrepeated, k_reps, k_dim = k.args
            v_unrepeated, v_reps, v_dim = v.args

            if (k_dim == 1 and v_dim == 1) and (k_reps == v_reps):
                is_grouped_kv = True
                k_base = k_unrepeated
                v_base = v_unrepeated
                body = [k, v]

        # Detect HuggingFace repeat_kv pattern:
        # unsqueeze(dim=2) → expand → clone → view
        if not is_grouped_kv:
            k_unwrap = cls._try_unwrap_repeat_kv(k)
            v_unwrap = cls._try_unwrap_repeat_kv(v)
            if k_unwrap is not None and v_unwrap is not None:
                k_base, k_body = k_unwrap
                v_base, v_body = v_unwrap
                is_grouped_kv = True
                body = k_body + v_body

        head = sdpa_node
        if not is_grouped_kv:
            body = []
        return SDPAHandler(head, body, q_node=q, k_node=k_base, v_node=v_base)

    def __call__(self, P: MLXProgramBuilder, n: Node) -> Slot:
        assert n == self.head
        q, k, v, attn_mask, dropout_p, is_causal, scale, enable_gqa = (
            SDPAHandler._parse_sdpa_args_and_kwargs(n)
        )
        head_dim = q.meta["val"].shape[-1]
        if scale is None:
            scale = head_dim**-0.5

        q = self.q_node
        k = self.k_node
        v = self.v_node

        assert dropout_p == 0.0, "SDPA with dropout is not supported"

        q, k, v, attn_mask = P.slot_map([q, k, v, attn_mask])
        out = P.make_or_get_slot(n)
        P.emit(
            SdpaNode(
                q=P.slot_to_tid(q),
                k=P.slot_to_tid(k),
                v=P.slot_to_tid(v),
                out=P.slot_to_tid(out),
                scale=scale,
                mask=P.slot_to_tid(attn_mask) if attn_mask else None,
                causal=is_causal,
            )
        )
        return out


# =============================================================================
# MLX_CUSTOM_SDPA pattern (mlx::custom_sdpa)
# =============================================================================


@REGISTRY.register_pattern(name="MLX_CUSTOM_SDPA")
class MLXCustomSdpaHandler(PatternHandler):
    """
    Pattern handler for mlx::custom_sdpa custom op.

    This op follows the optimum-executorch pattern:
    - Input: Q, K, V in BHSD format [B, num_heads, seq_len, head_dim]
    - start_pos: FIRST position of current query batch (not last!)
    - stop_pos: computed as start_pos + query_seq_len
    - K/V are FULL cache, sliced internally to [:, :, :stop_pos, :]

    For prefill with 7 tokens at positions [0,1,2,3,4,5,6]: start_pos=0, stop_pos=7
    For decode at position 10: start_pos=10, stop_pos=11

    Decomposes into:
    - SliceNode (K): slice to [:, :, :stop_pos, :]
    - SliceNode (V): slice to [:, :, :stop_pos, :]
    - SdpaNode: scaled dot-product attention (handles GQA internally)
    """

    def __init__(
        self,
        head: Node,
        body: List[Node],
        query: Node,
        key: Node,
        value: Node,
        start_pos: Any,  # int or Node (SymInt)
        scale: Optional[float],
        is_causal: bool,
    ):
        super().__init__(head, body)
        self.query = query
        self.key = key
        self.value = value
        self.start_pos = start_pos
        self.scale = scale
        self.is_causal = is_causal

    @classmethod
    def maybe_create(
        cls, ep: ExportedProgram, head: Node
    ) -> Optional["MLXCustomSdpaHandler"]:
        """Match the mlx::custom_sdpa custom op."""
        if head.op != "call_function":
            return None

        target_str = str(head.target)
        if "custom_sdpa" not in target_str or "mlx" not in target_str:
            return None

        # Op signature: custom_sdpa(query, key, value, start_pos, attn_mask, dropout_p, is_causal, scale)
        # start_pos is a SymInt (int), not a Tensor
        args = head.args
        kwargs = head.kwargs

        if len(args) < 4:
            return None

        query = args[0]
        key = args[1]
        value = args[2]
        start_pos = args[3]  # int or SymInt (Node)

        # Get optional args
        attn_mask = args[4] if len(args) > 4 else kwargs.get("attn_mask", None)
        dropout_p = args[5] if len(args) > 5 else kwargs.get("dropout_p", 0.0)
        is_causal = args[6] if len(args) > 6 else kwargs.get("is_causal", False)
        scale = args[7] if len(args) > 7 else kwargs.get("scale", None)

        # We only support causal attention without explicit mask
        if attn_mask is not None and not isinstance(attn_mask, type(None)):
            return None
        if dropout_p != 0.0:
            return None

        return MLXCustomSdpaHandler(
            head=head,
            body=[],
            query=query,
            key=key,
            value=value,
            start_pos=start_pos,
            scale=scale,
            is_causal=is_causal,
        )

    def __call__(self, P: MLXProgramBuilder, n: Node) -> Slot:
        from executorch.backends.apple.mlx.serialization.mlx_graph_schema import (
            IntOrVid,
            SdpaNode,
            SliceNode,
        )

        assert n == self.head

        # Get slots for Q, K, V
        q_slot, k_slot, v_slot = P.slot_map([self.query, self.key, self.value])

        # Get scale from metadata if not provided
        q_meta = self.query.meta.get("val")
        head_dim = q_meta.shape[-1]
        scale = self.scale if self.scale is not None else head_dim**-0.5

        # Resolve start_pos to int or Slot (same pattern as KVCacheUpdateHandler)
        if isinstance(self.start_pos, Node):
            start_slot = P.slot_map([self.start_pos])[0]
        else:
            start_slot = self.start_pos

        # Compute stop = start_pos + seq_len using emit_stop_position,
        # which handles static/dynamic seq_len (SymInt) and start_pos correctly.
        # BHSD layout: q is [B, num_heads, seq_len, head_dim], seq_len is dim 2.
        stop = emit_stop_position(
            P,
            start=start_slot,
            length_tensor=q_slot,
            length_dim=2,
            length_meta=q_meta,
        )
        slice_stop = P.to_int_or_vid(stop)

        # Step 1: Slice K to [:, :, :stop_pos, :] where stop_pos = start_pos + query_seq_len
        _, k_sliced_slot = P.make_tmp_slot()
        P.emit(
            SliceNode(
                x=P.slot_to_tid(k_slot),
                out=P.slot_to_tid(k_sliced_slot),
                axis=IntOrVid.from_literal(2),
                start=IntOrVid.from_literal(0),
                stop=slice_stop,
            )
        )

        # Step 2: Slice V to [:, :, :stop_pos, :] where stop_pos = start_pos + query_seq_len
        _, v_sliced_slot = P.make_tmp_slot()
        P.emit(
            SliceNode(
                x=P.slot_to_tid(v_slot),
                out=P.slot_to_tid(v_sliced_slot),
                axis=IntOrVid.from_literal(2),
                start=IntOrVid.from_literal(0),
                stop=slice_stop,
            )
        )

        # Step 3: SDPA (handles GQA internally) - outputs BHSD
        out_slot = P.make_or_get_slot(n)
        P.emit(
            SdpaNode(
                q=P.slot_to_tid(q_slot),
                k=P.slot_to_tid(k_sliced_slot),
                v=P.slot_to_tid(v_sliced_slot),
                out=P.slot_to_tid(out_slot),
                scale=scale,
                mask=None,
                causal=self.is_causal,
            )
        )

        return out_slot


# =============================================================================
# Quantization helpers
# =============================================================================


def _to_mlx_qparams(
    qdata: torch.Tensor,
    scale: torch.Tensor,
    zero_point: torch.Tensor,
    bits: int,
    compute_biases: bool = True,
) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
    """
    Convert TorchAO quantization params to MLX format.

    TorchAO uses: s * (q - z), with q signed
    MLX uses: S * Q + B, with Q unsigned

    s * (q - z)
      = s ((q + offset) - (z + offset))
      = s Q + B,
    where Q = q + offset, B = -s * (z + offset)

    Args:
        compute_biases: If False, skip bias computation (for scale_only mode).
                       Returns (Q, None) in this case. This is valid when
                       zero_point is all zeros, as the C++ runtime will compute
                       biases = -scales * 2^(bits-1).
    """
    assert qdata.dtype == torch.int8
    offset = 2 ** (bits - 1)
    Q = qdata.to(torch.int32) + offset

    # Pack data tightly into uint32
    assert 32 % bits == 0
    vals_per_uint32 = 32 // bits
    assert qdata.shape[1] % vals_per_uint32 == 0

    Q = Q.reshape(-1, vals_per_uint32)
    shifts = torch.arange(0, 32, bits, dtype=torch.int64)

    # Convert to int64 for shift/packing
    Q = Q.to(torch.int64)
    Q = (Q << shifts).sum(dim=-1)
    Q = Q.to(torch.uint32)
    Q = Q.reshape(qdata.shape[0], -1)

    if compute_biases:
        B = -scale * (zero_point.to(scale.dtype) + offset)
        return Q, B
    else:
        return Q, None


def _parse_dequant_node(
    node: Node,
) -> Optional[Tuple[Node, Node, Node, int, int, Optional[torch.dtype]]]:
    """Parse a torchao.dequantize_affine node."""
    qdata, block_size, scale, zero_point, dtype, qmin, qmax = node.args[0:7]
    out_dtype = node.kwargs.get("output_dtype", None)
    if dtype != torch.int8:
        return None
    if len(block_size) != 2 or block_size[0] != 1 or block_size[1] not in [32, 64, 128]:
        return None
    group_size = block_size[1]
    if qmin == -8 and qmax == 7:
        bits = 4
    elif qmin == -128 and qmax == 127:
        bits = 8
    else:
        return None
    return qdata, scale, zero_point, group_size, bits, out_dtype


# =============================================================================
# QUANTIZED_LINEAR pattern
# =============================================================================


@REGISTRY.register_pattern(name="QUANTIZED_LINEAR")
class QuantizedLinearHandler(PatternHandler):
    """
    Pattern for quantized linear: dequantize_affine + linear.
    """

    def __init__(
        self,
        head: Node,
        body: List[Node],
        qdata: Node,
        scale: Node,
        zero_point: Node,
        group_size: int,
        bits: int,
        out_dtype: torch.dtype,
    ):
        super().__init__(head, body)
        self.qdata = qdata
        self.scale = scale
        self.zero_point = zero_point
        self.group_size = group_size
        self.bits = bits
        self.out_dtype = out_dtype

    @classmethod
    def maybe_create(
        cls, ep: ExportedProgram, head: Node
    ) -> Optional["QuantizedLinearHandler"]:
        linear_node = head
        if not match_target(linear_node, torch.ops.aten.linear.default):
            return None

        x, w = linear_node.args[0:2]
        dequant_node = w
        if not match_target(dequant_node, torch.ops.torchao.dequantize_affine.default):
            return None
        if not has_single_user(dequant_node):
            return None

        parsed = _parse_dequant_node(dequant_node)
        if parsed is None:
            return None
        qdata, scale, zero_point, group_size, bits, out_dtype = parsed
        out_dtype = x.meta["val"].dtype if out_dtype is None else out_dtype

        head = linear_node
        body = [dequant_node]
        return QuantizedLinearHandler(
            head,
            body,
            qdata=qdata,
            scale=scale,
            zero_point=zero_point,
            group_size=group_size,
            bits=bits,
            out_dtype=out_dtype,
        )

    def __call__(self, P: MLXProgramBuilder, n: Node) -> Slot:
        assert n == self.head

        x, w = n.args[0:2]
        b = n.args[2] if len(n.args) > 2 else None

        qdata_target, qdata = P.get_placeholder_target_and_tensor(self.qdata)
        zero_point_target, zero_point = P.get_placeholder_target_and_tensor(
            self.zero_point
        )
        _, scale = P.get_placeholder_target_and_tensor(self.scale)

        out_scalar_type = torch_dtype_to_scalar_type(self.out_dtype)

        # Check if we can use scale_only optimization:
        # When zero_point is all zeros, biases = -scales * 2^(bits-1)
        # which can be computed at runtime instead of serialized.
        # Note: During partitioning, tensors are FakeTensors so we skip the check.
        # The optimization is only applied during preprocess when we have real tensors.
        use_scale_only = False
        if not QUANTIZED_SERIALIZE_BIASES:
            from torch._subclasses.fake_tensor import FakeTensor

            if not isinstance(zero_point, FakeTensor):
                if torch.sum(torch.abs(zero_point)).item() == 0:
                    use_scale_only = True

        Q, B = _to_mlx_qparams(
            qdata, scale, zero_point, self.bits, compute_biases=not use_scale_only
        )
        w = P.make_or_get_constant(f"{qdata_target}_to_packed", Q)

        if use_scale_only:
            biases_tid = None
        else:
            biases = P.make_or_get_constant(f"{zero_point_target}_to_biases", B)
            biases_tid = P.slot_to_tid(biases)

        x, scale_slot, b = P.slot_map([x, self.scale, b])
        out = P.make_or_get_slot(n)
        P.emit(
            QuantizedLinearNode(
                x=P.slot_to_tid(x),
                w=P.slot_to_tid(w),
                scales=P.slot_to_tid(scale_slot),
                out=P.slot_to_tid(out),
                biases=biases_tid,
                bias=P.slot_to_tid(b) if b else None,
                group_size=self.group_size,
                bits=self.bits,
                mode="affine",
                out_scalar_type=out_scalar_type,
                scale_only=use_scale_only,
            )
        )
        return out


# =============================================================================
# QUANTIZED_EMBEDDING pattern
# ============================================================================


@REGISTRY.register_pattern(name="QUANTIZED_EMBEDDING")
class QuantizedEmbeddingHandler(PatternHandler):
    """
    Pattern for quantized embedding: dequantize_affine + embedding.
    """

    def __init__(
        self,
        head: Node,
        body: List[Node],
        qdata: Node,
        scale: Node,
        zero_point: Node,
        group_size: int,
        bits: int,
        out_dtype: torch.dtype,
    ):
        super().__init__(head, body)
        self.qdata = qdata
        self.scale = scale
        self.zero_point = zero_point
        self.group_size = group_size
        self.bits = bits
        self.out_dtype = out_dtype

    @classmethod
    def maybe_create(
        cls, ep: ExportedProgram, head: Node
    ) -> Optional["QuantizedEmbeddingHandler"]:
        embedding_node = head
        if not match_target(embedding_node, torch.ops.aten.embedding.default):
            return None

        w, x = embedding_node.args[0:2]

        dequant_node = w
        if not match_target(dequant_node, torch.ops.torchao.dequantize_affine.default):
            return None
        if not has_single_user(dequant_node):
            return None

        parsed = _parse_dequant_node(dequant_node)
        if parsed is None:
            return None
        qdata, scale, zero_point, group_size, bits, out_dtype = parsed
        out_dtype = scale.meta["val"].dtype if out_dtype is None else out_dtype

        head = embedding_node
        body = [dequant_node]
        return QuantizedEmbeddingHandler(
            head,
            body,
            qdata=qdata,
            scale=scale,
            zero_point=zero_point,
            group_size=group_size,
            bits=bits,
            out_dtype=out_dtype,
        )

    def __call__(self, P: MLXProgramBuilder, n: Node) -> Slot:
        assert n == self.head
        w, x = n.args[0:2]

        qdata_target, qdata = P.get_placeholder_target_and_tensor(self.qdata)
        zero_point_target, zero_point = P.get_placeholder_target_and_tensor(
            self.zero_point
        )
        _, scale = P.get_placeholder_target_and_tensor(self.scale)

        Q, B = _to_mlx_qparams(qdata, scale, zero_point, self.bits)
        out_scalar_type = torch_dtype_to_scalar_type(self.out_dtype)

        w = P.make_or_get_constant(f"{qdata_target}_to_packed", Q)
        biases = P.make_or_get_constant(f"{zero_point_target}_to_biases", B)

        x, scale_slot = P.slot_map([x, self.scale])
        out = P.make_or_get_slot(n)
        P.emit(
            QuantizedGatherNode(
                table_q=P.slot_to_tid(w),
                scales=P.slot_to_tid(scale_slot),
                ids=P.slot_to_tid(x),
                out=P.slot_to_tid(out),
                biases=P.slot_to_tid(biases),
                group_size=self.group_size,
                bits=self.bits,
                mode="affine",
                out_scalar_type=out_scalar_type,
            )
        )
        return out
