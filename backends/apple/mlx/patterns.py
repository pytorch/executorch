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

from typing import Any, List, Optional, Tuple

import torch
from executorch.backends.apple.mlx.program_builder import (
    _torch_dtype_to_dtypeid,
    emit_stop_position,
    get_aten_target_normalized,
    MLXProgramBuilder,
    PatternHandler,
    REGISTRY,
    Slot,
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
# SLICE_UPDATE pattern
# =============================================================================


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
        _op_namespace = torch.ops.aten

        slice_scatter_node = head
        if (
            get_aten_target_normalized(slice_scatter_node.target)
            != _op_namespace.slice_scatter.default
        ):
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
        if get_aten_target_normalized(copy_node.target) != _op_namespace.copy.default:
            return None
        if copy_node.users != {slice_scatter_node: None}:
            return None
        if len(copy_node.args) != 2:
            return None
        c_dst, c_src = copy_node.args

        slice_node = c_dst
        # In Edge IR, slice.Tensor becomes slice_copy.Tensor
        # Use get_aten_target_normalized to normalize both to slice.Tensor for comparison
        slice_target = get_aten_target_normalized(slice_node.target)
        if slice_target != _op_namespace.slice.Tensor:
            return None
        if slice_node.users != {copy_node: None}:
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
        _op_namespace = torch.ops.aten

        index_copy_node = head
        if (
            get_aten_target_normalized(index_copy_node.target)
            != _op_namespace.index_copy.default
        ):
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
# UPDATE_CACHE pattern
# =============================================================================


def _is_auto_func_update_cache(node: Node) -> bool:
    """Check if a node is auto_functionalized_v2 wrapping llama.update_cache."""
    if node.op != "call_function":
        return False
    target_str = str(node.target)
    if "auto_functionalized" not in target_str:
        return False
    if len(node.args) < 1:
        return False
    func_arg = node.args[0]
    func_str = str(func_arg) if func_arg else ""
    return "update_cache" in func_str and "llama" in func_str


@REGISTRY.register_pattern(name="UPDATE_CACHE")
class UpdateCacheHandler(PatternHandler):
    """
    Pattern for KV cache updates using torch.ops.llama.update_cache.

    Matches: transpose -> auto_functionalized(update_cache) -> getitem -> transpose
    Where the transposes convert between [B, H, S, D] and [B, S, H, D].

    This pattern is used by CustomKVCache which transposes from [B, H, S, D]
    to [B, S, H, D] for update_cache, then transposes back.

    We recognize this pattern and lower directly to SliceUpdateNode operating
    on the [B, H, S, D] layout (dim=2), eliminating the transpose overhead.

    Graph structure after functionalization (is_edge_ir=False):
        transpose = aten.transpose.int(k_val, 1, 2)
        auto_func = auto_functionalized_v2(llama.update_cache, value=transpose, ...)
        getitem = getitem(auto_func, 1)
        transpose_out = aten.transpose.int(getitem, 1, 2)  <-- HEAD

    Graph structure after to_edge (is_edge_ir=True):
        permute = aten.permute_copy.default(k_val, [0, 2, 1, 3])
        auto_func = auto_functionalized_v2(llama.update_cache, value=permute, ...)
        getitem = getitem(auto_func, 1)
        permute_out = aten.permute_copy.default(getitem, [0, 2, 1, 3])  <-- HEAD
    """

    def __init__(
        self,
        head: Node,
        body: List[Node],
        cache: Node,
        update: Node,
        start_pos: Any,
    ):
        super().__init__(head, body)
        self.cache = cache  # The cache buffer [B, H, S, D]
        self.update = update  # The update tensor [B, H, S_step, D]
        self.start_pos = start_pos  # Start position (int or SymInt)

    @classmethod
    def maybe_create(
        cls, ep: ExportedProgram, head: Node
    ) -> Optional["UpdateCacheHandler"]:
        """
        Match the UPDATE_CACHE pattern.

        Pattern (HEAD = getitem):
            transpose_in = transpose(update, 1, 2)
            auto_func = auto_functionalized_v2(llama.update_cache, value=transpose_in, ...)
            getitem = getitem(auto_func, 1)  <-- HEAD

        Uses get_aten_target_normalized to handle both ATen IR
        (transpose.int) and Edge IR (transpose_copy.int).
        """
        _op_ns = torch.ops.aten

        # Only check getitem nodes
        if head.op != "call_function" or "getitem" not in str(head.target):
            return None

        # Check getitem is extracting idx=1 (the updated cache)
        if len(head.args) < 2 or head.args[1] != 1:
            return None

        getitem_node = head

        # getitem's source should be auto_functionalized_v2
        if len(getitem_node.args) < 1 or not isinstance(getitem_node.args[0], Node):
            return None

        potential_auto_func = getitem_node.args[0]
        if "auto_functionalized" not in str(potential_auto_func.target):
            return None

        auto_func_node = potential_auto_func

        # Extract info from auto_functionalized_v2 kwargs
        kwargs = auto_func_node.kwargs
        value_node = kwargs.get("value")
        start_pos_node = kwargs.get("start_pos")
        all_bases = kwargs.get("_all_bases", [])

        if not value_node or not all_bases:
            return None

        cache_node = all_bases[0]

        # value should be transpose(x, 1, 2)
        if not isinstance(value_node, Node) or value_node.op != "call_function":
            return None

        # Check for transpose - use get_aten_target_normalized to handle both
        # ATen IR (transpose.int) and Edge IR (transpose_copy.int)
        value_target = get_aten_target_normalized(value_node.target)

        if value_target != _op_ns.transpose.int:
            return None

        if len(value_node.args) < 3:
            return None

        dim0_in, dim1_in = value_node.args[1], value_node.args[2]
        if not ((dim0_in == 1 and dim1_in == 2) or (dim0_in == 2 and dim1_in == 1)):
            return None

        input_transpose_node = value_node
        # Get the actual update tensor (input to the transpose) - this is [B, H, S_step, D]
        update_input_node = value_node.args[0] if value_node.args else None

        # Build the pattern body
        body = [auto_func_node, input_transpose_node]

        return UpdateCacheHandler(
            head=head,
            body=body,
            cache=cache_node,
            update=update_input_node,
            start_pos=start_pos_node,
        )

    def __call__(self, P: MLXProgramBuilder, n: Node) -> Slot:
        assert n == self.head

        # Get slots for cache and update
        # cache is [B, H, S, D] (SDPA convention - mutable buffer created with this shape)
        # update is [B, H, S_step, D] (the input BEFORE the permute in the pattern)
        cache_slot = P.slot_map([self.cache])[0]
        update_slot = P.slot_map([self.update])[0]

        # start_pos could be an int, SymInt, or a Node (from item())
        if isinstance(self.start_pos, Node):
            start_slot = P.slot_map([self.start_pos])[0]
        else:
            start_slot = self.start_pos

        # Get the output slot for this node (the head)
        # The head is getitem, which outputs the mutated cache [B, H, S, D]
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
        # No transpose needed! Both are in SDPA format [B, H, S, D]
        P.emit(
            SliceUpdateNode(
                dst=P.slot_to_tid(cache_slot),
                update=P.slot_to_tid(update_slot),
                axis=IntOrVid.from_literal(2),  # S dimension in [B, H, S, D]
                start=P.to_int_or_vid(start_slot),
                stop=P.to_int_or_vid(stop_slot),
            )
        )

        # SliceUpdate mutates dst in-place and returns the updated dst
        # The output is cache_slot which is [B, H, S, D] - exactly what we need!
        # Copy cache to out_slot for the output
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
    def maybe_create(cls, ep: ExportedProgram, head: Node) -> Optional["SDPAHandler"]:
        _op_namespace = torch.ops.aten

        sdpa_node = head
        if (
            get_aten_target_normalized(sdpa_node.target)
            != _op_namespace.scaled_dot_product_attention.default
        ):
            return None

        q, k, v, _, _, _, _, _ = cls._parse_sdpa_args_and_kwargs(sdpa_node)

        # Detect grouped kv attention pattern with repeat_interleave before SDPA
        is_grouped_kv = False
        k_base = k
        v_base = v
        if (
            get_aten_target_normalized(k.target)
            == _op_namespace.repeat_interleave.self_int
            and (k.users == {sdpa_node: None})
            and (len(k.args) == 3)
            and (len(k.kwargs) == 0)
            and get_aten_target_normalized(v.target)
            == _op_namespace.repeat_interleave.self_int
            and (v.users == {sdpa_node: None})
            and (len(v.args) == 3)
            and (len(v.kwargs) == 0)
        ):
            k_unrepeated, k_reps, k_dim = k.args
            v_unrepeated, v_reps, v_dim = v.args

            if (k_dim == 1 and v_dim == 1) and (k_reps == v_reps):
                is_grouped_kv = True
                k_base = k_unrepeated
                v_base = v_unrepeated

        head = sdpa_node
        body = [k, v] if is_grouped_kv else []
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
        _op_namespace = torch.ops.aten

        linear_node = head
        if (
            get_aten_target_normalized(linear_node.target)
            != _op_namespace.linear.default
        ):
            return None

        x, w = linear_node.args[0:2]
        dequant_node = w
        if (
            get_aten_target_normalized(dequant_node.target)
            != torch.ops.torchao.dequantize_affine.default
        ):
            return None

        if dequant_node.users != {linear_node: None}:
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

        out_dtype = _torch_dtype_to_dtypeid(self.out_dtype)

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
                out_dtype=out_dtype,
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
        _op_namespace = torch.ops.aten

        embedding_node = head
        if (
            get_aten_target_normalized(embedding_node.target)
            != _op_namespace.embedding.default
        ):
            return None

        w, x = embedding_node.args[0:2]

        dequant_node = w
        if (
            get_aten_target_normalized(dequant_node.target)
            != torch.ops.torchao.dequantize_affine.default
        ):
            return None
        if dequant_node.users != {embedding_node: None}:
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
        out_dtype = _torch_dtype_to_dtypeid(self.out_dtype)

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
                out_dtype=out_dtype,
            )
        )
        return out
