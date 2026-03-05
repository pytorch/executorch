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
them to optimized MLX operations.
"""

from __future__ import annotations

from typing import Any, List, Optional, Tuple

import torch
from executorch.backends.mlx.builder.op_helpers import (
    emit_quantized_biases,
    emit_stop_position,
    parse_dequant_node,
    parse_dequant_nvfp4_node,
    to_mlx_qparams,
    torch_dtype_to_scalar_type,
)
from executorch.backends.mlx.builder.op_registry import PatternHandler, REGISTRY
from executorch.backends.mlx.builder.program_builder import MLXProgramBuilder
from executorch.backends.mlx.builder.slot_manager import Slot
from executorch.backends.mlx.pattern_utils import (
    has_single_user,
    match_target,
    OpStep,
    walk_back,
)
from executorch.backends.mlx.serialization.mlx_graph_schema import (
    AddIntNode,
    AddNode,
    AsTypeNode,
    DequantizeNode,
    IndexCopyNode,
    IntOrVid,
    IntOrVidOrTid,
    ModIntNode,
    MultiplyNode,
    QuantizedMatmulNode,
    SdpaNode,
    SliceNode,
    SliceUpdateNode,
    SubtractIntNode,
    SymSizeNode,
    TakeNode,
)
from torch.export.exported_program import ExportedProgram
from torch.fx.node import Node


@REGISTRY.register_pattern(name="INDEX_COPY")
class IndexCopyHandler(PatternHandler):
    """
    Pattern for index-based updates on mutable buffers.
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
    ) -> Optional["IndexCopyHandler"]:
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
            IndexCopyNode(
                dst=P.slot_to_tid(dst),
                update=P.slot_to_tid(update),
                indices=P.slot_to_tid(indices),
                out=P.slot_to_tid(dst),
                axis=self.axis,
            )
        )

        P.set_slot(n, dst)
        return dst


@REGISTRY.register_pattern(name="ET_KV_CACHE_UPDATE")
class ETKVCacheUpdateHandler(PatternHandler):
    """
    Pattern for KV cache updates using torch.ops.mlx.kv_cache_update.

    Matches: auto_functionalized → getitem[1]
    HEAD = getitem[1] (no alias_copy required)

    Graph structure:
        auto_func = auto_functionalized_v2(mlx.kv_cache_update, new_values=k_val, ...)
        getitem_1 = getitem(auto_func, 1)    # HEAD - updated cache
    """

    def __init__(
        self,
        head: Node,
        body: List[Node],
        cache: Node,
        update: Node,
        start_pos: Any,
        ring_size: int = 0,
    ):
        super().__init__(head, body)
        self.cache = cache
        self.update = update
        self.start_pos = start_pos
        self.ring_size = ring_size

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

        Pattern (HEAD = getitem):
            auto_func = auto_functionalized_v2(mlx.kv_cache_update, ...)
            getitem_1 = getitem(auto_func, 1)    # HEAD
        """

        # HEAD must be getitem with idx=1
        if head.op != "call_function" or "getitem" not in str(head.target):
            return None

        if len(head.args) < 2 or head.args[1] != 1:
            return None

        # getitem's source should be auto_functionalized_v2 wrapping mlx.kv_cache_update
        if not isinstance(head.args[0], Node):
            return None

        auto_func_node = head.args[0]
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

        body = [auto_func_node]

        return cls(
            head=head,
            body=body,
            cache=cache_node,
            update=new_values_node,
            start_pos=start_pos_node,
            ring_size=kwargs.get("ring_size", 0),
        )

    def __call__(self, P: "MLXProgramBuilder", n: Node) -> Slot:
        assert n == self.head

        cache_slot, update_slot, start_slot = P.slot_map(
            [self.cache, self.update, self.start_pos]
        )

        if self.ring_size > 0:
            self._emit_ring_buffer(P, cache_slot, update_slot, start_slot)
        else:
            self._emit_linear(P, cache_slot, update_slot, start_slot)

        P.set_slot(n, cache_slot)
        return cache_slot

    def _emit_linear(self, P: "MLXProgramBuilder", cache_slot, update_slot, start_slot):
        """Emit a single SliceUpdate for linear (non-ring) cache."""
        update_meta = self.update.meta.get("val")
        stop_slot = emit_stop_position(
            P,
            start=start_slot,
            length_tensor=update_slot,
            length_dim=2,  # S_step is dim 2 in [B, H, S_step, D]
            length_meta=update_meta,
        )

        # This updates cache[:, :, start:stop, :] = update
        # SliceUpdateNode on axis=2
        # cache is [B, H, S, D], update is [B, H, S_step, D]
        P.emit(
            SliceUpdateNode(
                dst=P.slot_to_tid(cache_slot),
                update=P.slot_to_tid(update_slot),
                out=P.slot_to_tid(cache_slot),
                axis=IntOrVid.from_literal(2),  # S dimension in [B, H, S, D]
                start=P.to_int_or_vid(start_slot),
                stop=P.to_int_or_vid(stop_slot),
            )
        )

    def _emit_ring_buffer(
        self, P: "MLXProgramBuilder", cache_slot, update_slot, start_slot
    ):
        """
        Emit two unconditional SliceUpdates for ring buffer wrapping.

        write_pos    = start_pos % ring_size
        first_len    = ring_size - write_pos
        first_chunk  = update[:, :, :first_len, :]      (Slice clamps to seq_len)
        actual_first = first_chunk.shape[2]              (min(first_len, seq_len))
        rest_chunk   = update[:, :, actual_first:seq_len, :]
        overflow     = seq_len - actual_first
        SliceUpdate(cache, first_chunk, write_pos, write_pos + actual_first)
        SliceUpdate(cache, rest_chunk,  0, overflow)

        When no wrap: actual_first == seq_len, rest_chunk is zero-length,
        second SliceUpdate is a no-op (guarded in exec_slice_update).
        """
        ring_size = self.ring_size

        # write_pos = start_pos % ring_size
        _, write_pos_slot = P.slot_manager.make_tmp_value_slot()
        P.emit(
            ModIntNode(
                a=P.to_int_or_vid(start_slot),
                b=IntOrVid.from_literal(ring_size),
                out=P.slot_to_vid(write_pos_slot),
            )
        )

        # seq_len = update.shape[2]
        _, seq_len_slot = P.slot_manager.make_tmp_value_slot()
        P.emit(
            SymSizeNode(
                a=P.slot_to_tid(update_slot),
                dim=2,
                out=P.slot_to_vid(seq_len_slot),
            )
        )

        # first_len = ring_size - write_pos (may be > seq_len)
        _, first_len_slot = P.slot_manager.make_tmp_value_slot()
        P.emit(
            SubtractIntNode(
                a=IntOrVid.from_literal(ring_size),
                b=P.to_int_or_vid(write_pos_slot),
                out=P.slot_to_vid(first_len_slot),
            )
        )

        # first_chunk = update[:, :, :first_len, :]  (Slice clamps to seq_len)
        _, first_chunk_slot = P.make_tmp_slot()
        P.emit(
            SliceNode(
                x=P.slot_to_tid(update_slot),
                out=P.slot_to_tid(first_chunk_slot),
                axis=IntOrVid.from_literal(2),
                start=IntOrVid.from_literal(0),
                stop=P.to_int_or_vid(first_len_slot),
            )
        )

        # actual_first = first_chunk.shape[2]  (= min(first_len, seq_len))
        _, actual_first_slot = P.slot_manager.make_tmp_value_slot()
        P.emit(
            SymSizeNode(
                a=P.slot_to_tid(first_chunk_slot),
                dim=2,
                out=P.slot_to_vid(actual_first_slot),
            )
        )

        # rest_chunk = update[:, :, actual_first:seq_len, :]
        _, rest_chunk_slot = P.make_tmp_slot()
        P.emit(
            SliceNode(
                x=P.slot_to_tid(update_slot),
                out=P.slot_to_tid(rest_chunk_slot),
                axis=IntOrVid.from_literal(2),
                start=P.to_int_or_vid(actual_first_slot),
                stop=P.to_int_or_vid(seq_len_slot),
            )
        )

        # stop1 = write_pos + actual_first
        _, stop1_slot = P.slot_manager.make_tmp_value_slot()
        P.emit(
            AddIntNode(
                a=P.to_int_or_vid(write_pos_slot),
                b=P.to_int_or_vid(actual_first_slot),
                out=P.slot_to_vid(stop1_slot),
            )
        )

        # overflow = seq_len - actual_first
        _, overflow_slot = P.slot_manager.make_tmp_value_slot()
        P.emit(
            SubtractIntNode(
                a=P.to_int_or_vid(seq_len_slot),
                b=P.to_int_or_vid(actual_first_slot),
                out=P.slot_to_vid(overflow_slot),
            )
        )

        # SliceUpdate 1: cache[:, :, write_pos:stop1, :] = first_chunk
        P.emit(
            SliceUpdateNode(
                dst=P.slot_to_tid(cache_slot),
                update=P.slot_to_tid(first_chunk_slot),
                out=P.slot_to_tid(cache_slot),
                axis=IntOrVid.from_literal(2),
                start=P.to_int_or_vid(write_pos_slot),
                stop=P.to_int_or_vid(stop1_slot),
            )
        )

        # SliceUpdate 2: cache[:, :, 0:overflow, :] = rest_chunk
        # Zero-length no-op when no wrap (overflow=0)
        P.emit(
            SliceUpdateNode(
                dst=P.slot_to_tid(cache_slot),
                update=P.slot_to_tid(rest_chunk_slot),
                out=P.slot_to_tid(cache_slot),
                axis=IntOrVid.from_literal(2),
                start=IntOrVid.from_literal(0),
                stop=P.to_int_or_vid(overflow_slot),
            )
        )


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
        result = walk_back(
            node,
            [
                OpStep(op=torch.ops.aten.view.default, nargs=2),
                OpStep(op=torch.ops.aten.clone.default, optional=True),
                OpStep(op=torch.ops.aten.expand.default, nargs=2),
                OpStep(op=torch.ops.aten.unsqueeze.default, nargs=2),
            ],
        )
        if result is None:
            return None

        base, entries = result
        _view, _clone, _expand, unsqueeze = entries

        # unsqueeze must be on dim=2
        if unsqueeze.args[1] != 2:
            return None

        body = [e for e in entries if e is not None]
        return base, body

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


@REGISTRY.register_pattern(name="NVFP4_QUANTIZED_EMBEDDING")
class NVFP4QuantizedEmbeddingHandler(PatternHandler):
    """Fuse dequantize_nvfp4 + embedding into gather + DequantizeNode(mode="nvfp4").

    Matches:
        embedding(dequantize_nvfp4(qdata, scale, per_tensor_scale, ...), indices)

    Emits:
        TakeNode(qdata) → TakeNode(scales) → DequantizeNode(mode="nvfp4")
        [→ MultiplyNode(per_tensor_scale)] [→ AsTypeNode]
    """

    def __init__(self, head, body, qdata, scale, per_tensor_scale, output_dtype):
        super().__init__(head, body)
        self.qdata = qdata
        self.scale = scale
        self.per_tensor_scale = per_tensor_scale
        self.output_dtype = output_dtype

    @classmethod
    def maybe_create(cls, ep, head):
        if not match_target(head, torch.ops.aten.embedding.default):
            return None

        w, x = head.args[0:2]
        if not isinstance(w, Node):
            return None
        if not has_single_user(w):
            return None
        parsed = parse_dequant_nvfp4_node(w)
        if parsed is None:
            return None
        qdata, scale, per_tensor_scale, output_dtype = parsed
        return cls(head, [w], qdata, scale, per_tensor_scale, output_dtype)

    def __call__(self, P: MLXProgramBuilder, n: Node) -> Slot:
        assert n == self.head
        w_node, x_node = n.args[0:2]

        has_per_tensor_scale = True
        _, per_tensor_scale_value = P.get_placeholder_target_and_tensor(
            self.per_tensor_scale
        )
        from torch._subclasses.fake_tensor import FakeTensor

        if not isinstance(per_tensor_scale_value, FakeTensor):
            if per_tensor_scale_value.item() == 1.0:
                has_per_tensor_scale = False

        x_dtype = x_node.meta["val"].dtype
        needs_cast = self.output_dtype != x_dtype

        x, scales_slot, per_tensor_scale, qdata_slot = P.slot_map(
            [x_node, self.scale, self.per_tensor_scale, self.qdata]
        )

        ids_index = IntOrVidOrTid.from_tid(P.slot_to_tid(x))

        # Gather quantized weights by indices
        _, wq_sel = P.make_tmp_slot()
        P.emit(
            TakeNode(
                x=P.slot_to_tid(qdata_slot),
                index=ids_index,
                out=P.slot_to_tid(wq_sel),
                axis=0,
            )
        )

        # Gather scales by indices
        _, sc_sel = P.make_tmp_slot()
        P.emit(
            TakeNode(
                x=P.slot_to_tid(scales_slot),
                index=ids_index,
                out=P.slot_to_tid(sc_sel),
                axis=0,
            )
        )

        # Dequantize the gathered slices
        out = P.make_or_get_slot(n)
        P.emit(
            DequantizeNode(
                w=P.slot_to_tid(wq_sel),
                scales=P.slot_to_tid(sc_sel),
                out=P.slot_to_tid(out),
                biases=None,
                group_size=16,
                bits=4,
                mode="nvfp4",
                dtype=torch_dtype_to_scalar_type(self.output_dtype),
            )
        )

        if has_per_tensor_scale:
            P.emit(
                MultiplyNode(
                    a=P.slot_to_tid(out),
                    b=P.slot_to_tid(per_tensor_scale),
                    out=P.slot_to_tid(out),
                )
            )

        if needs_cast:
            P.emit(
                AsTypeNode(
                    x=P.slot_to_tid(out),
                    out=P.slot_to_tid(out),
                    scalar_type=torch_dtype_to_scalar_type(self.output_dtype),
                )
            )

        return out


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
        attn_mask: Optional[Node],
        scale: Optional[float],
        is_causal: bool,
    ):
        super().__init__(head, body)
        self.query = query
        self.key = key
        self.value = value
        self.start_pos = start_pos
        self.attn_mask = attn_mask
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

        if dropout_p != 0.0:
            return None

        return MLXCustomSdpaHandler(
            head=head,
            body=[],
            query=query,
            key=key,
            value=value,
            start_pos=start_pos,
            attn_mask=attn_mask,
            scale=scale,
            is_causal=is_causal,
        )

    def __call__(self, P: MLXProgramBuilder, n: Node) -> Slot:
        from executorch.backends.mlx.serialization.mlx_graph_schema import (
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
                mask=(
                    P.slot_to_tid(P.slot_map([self.attn_mask])[0])
                    if self.attn_mask is not None
                    else None
                ),
                causal=self.is_causal,
            )
        )

        return out_slot


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

        parsed = parse_dequant_node(dequant_node)
        if parsed is None:
            return None
        qdata, scale, zero_point, group_size, bits, out_dtype, _quantized_dim = parsed
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

        x_node, w_node = n.args[0:2]
        b_node = n.args[2] if len(n.args) > 2 else None

        qdata_target, qdata = P.get_placeholder_target_and_tensor(self.qdata)
        zero_point_target, zero_point = P.get_placeholder_target_and_tensor(
            self.zero_point
        )
        _, scale = P.get_placeholder_target_and_tensor(self.scale)

        x_slot, scale_slot, b_slot = P.slot_map([x_node, self.scale, b_node])

        Q, B = to_mlx_qparams(qdata, scale, zero_point, self.bits)
        w = P.make_or_get_constant(f"{qdata_target}_to_packed", Q)
        biases = emit_quantized_biases(
            P, zero_point_target, scale, zero_point, self.bits, B, scale_slot
        )

        out = P.make_or_get_slot(n)
        has_bias = b_node is not None
        x_dtype = x_node.meta["val"].dtype
        needs_cast = self.out_dtype != x_dtype

        P.emit(
            QuantizedMatmulNode(
                x=P.slot_to_tid(x_slot),
                w=P.slot_to_tid(w),
                scales=P.slot_to_tid(scale_slot),
                out=P.slot_to_tid(out),
                biases=P.slot_to_tid(biases),
                group_size=self.group_size,
                bits=self.bits,
                mode="affine",
                transpose=True,
            )
        )

        if has_bias:
            P.emit(
                AddNode(
                    a=P.slot_to_tid(out),
                    b=P.slot_to_tid(b_slot),
                    out=P.slot_to_tid(out),
                )
            )

        if needs_cast:
            P.emit(
                AsTypeNode(
                    x=P.slot_to_tid(out),
                    out=P.slot_to_tid(out),
                    scalar_type=torch_dtype_to_scalar_type(self.out_dtype),
                )
            )

        return out


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

        parsed = parse_dequant_node(dequant_node)
        if parsed is None:
            return None
        qdata, scale, zero_point, group_size, bits, out_dtype, _quantized_dim = parsed
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

        Q, B = to_mlx_qparams(qdata, scale, zero_point, self.bits)
        out_scalar_type = torch_dtype_to_scalar_type(self.out_dtype)

        w = P.make_or_get_constant(f"{qdata_target}_to_packed", Q)

        x, scale_slot = P.slot_map([x, self.scale])
        biases = emit_quantized_biases(
            P, zero_point_target, scale, zero_point, self.bits, B, scale_slot
        )
        ids_index = IntOrVidOrTid.from_tid(P.slot_to_tid(x))

        # Gather quantized weights by ids
        _, wq_sel = P.make_tmp_slot()
        P.emit(
            TakeNode(
                x=P.slot_to_tid(w),
                index=ids_index,
                out=P.slot_to_tid(wq_sel),
                axis=0,
            )
        )

        # Gather scales by ids
        _, sc_sel = P.make_tmp_slot()
        P.emit(
            TakeNode(
                x=P.slot_to_tid(scale_slot),
                index=ids_index,
                out=P.slot_to_tid(sc_sel),
                axis=0,
            )
        )

        # Gather biases by ids
        _, b_sel = P.make_tmp_slot()
        P.emit(
            TakeNode(
                x=P.slot_to_tid(biases),
                index=ids_index,
                out=P.slot_to_tid(b_sel),
                axis=0,
            )
        )

        # Dequantize the gathered slices
        out = P.make_or_get_slot(n)
        P.emit(
            DequantizeNode(
                w=P.slot_to_tid(wq_sel),
                scales=P.slot_to_tid(sc_sel),
                out=P.slot_to_tid(out),
                biases=P.slot_to_tid(b_sel),
                group_size=self.group_size,
                bits=self.bits,
                mode="affine",
                dtype=out_scalar_type,
            )
        )
        return out


@REGISTRY.register_pattern(name="NVFP4_QUANTIZED_LINEAR")
class NVFP4QuantizedLinearHandler(PatternHandler):
    """Fuse dequantize_nvfp4 + linear into QuantizedMatmulNode(mode="nvfp4").

    Matches:
        linear(x, dequantize_nvfp4(qdata, scale, block_size, [per_tensor_scale]), bias)

    Emits:
        QuantizedMatmulNode [→ MultiplyNode(per_tensor_scale)] [→ AddNode(bias)]
    """

    def __init__(self, head, body, qdata, scale, per_tensor_scale, output_dtype):
        super().__init__(head, body)
        self.qdata = qdata
        self.scale = scale
        self.per_tensor_scale = per_tensor_scale
        self.output_dtype = output_dtype

    @classmethod
    def maybe_create(cls, ep, head):
        if not match_target(head, torch.ops.aten.linear.default):
            return None
        x, dequant = head.args[0:2]
        if not isinstance(dequant, Node):
            return None
        if not has_single_user(dequant):
            return None
        parsed = parse_dequant_nvfp4_node(dequant)
        if parsed is None:
            return None
        qdata, scale, per_tensor_scale, output_dtype = parsed
        return cls(head, [dequant], qdata, scale, per_tensor_scale, output_dtype)

    def __call__(self, P, n):
        assert n == self.head

        x_node, w_node = n.args[0:2]
        b_node = n.args[2] if len(n.args) > 2 else None

        needs_cast = x_node.meta["val"].dtype != self.output_dtype
        has_bias = b_node is not None
        has_per_tensor_scale = True

        _, per_tensor_scale_value = P.get_placeholder_target_and_tensor(
            self.per_tensor_scale
        )
        from torch._subclasses.fake_tensor import FakeTensor

        if not isinstance(per_tensor_scale_value, FakeTensor):
            if per_tensor_scale_value.item() == 1.0:
                has_per_tensor_scale = False

        x, w, scales, bias, per_tensor_scale = P.slot_map(
            [x_node, self.qdata, self.scale, b_node, self.per_tensor_scale]
        )

        out = P.make_or_get_slot(n)
        P.emit(
            QuantizedMatmulNode(
                x=P.slot_to_tid(x),
                w=P.slot_to_tid(w),
                scales=P.slot_to_tid(scales),
                out=P.slot_to_tid(out),
                biases=None,
                group_size=16,
                bits=4,
                mode="nvfp4",
                transpose=True,
            )
        )

        if has_per_tensor_scale:
            P.emit(
                MultiplyNode(
                    a=P.slot_to_tid(out),
                    b=P.slot_to_tid(per_tensor_scale),
                    out=P.slot_to_tid(out),
                )
            )

        if has_bias:
            P.emit(
                AddNode(
                    a=P.slot_to_tid(out),
                    b=P.slot_to_tid(bias),
                    out=P.slot_to_tid(out),
                )
            )

        if needs_cast:
            P.emit(
                AsTypeNode(
                    x=P.slot_to_tid(out),
                    out=P.slot_to_tid(out),
                    scalar_type=torch_dtype_to_scalar_type(self.output_dtype),
                )
            )

        return out
