# Copyright (c) Samsung Electronics Co. LTD
# All rights reserved
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import torch
from executorch.exir import ExportedProgram
from executorch.exir.passes import dead_code_elimination_pass
from torch.fx import Node


_INT4_KERNEL_BIAS = 8  # embedding_4bit kernel reconstructs nibble - 8
_INT2_KERNEL_BIAS = 2  # embedding_2bit kernel reconstructs 2-bit field - 2


def _pick_bits(quant_min: int, quant_max: int) -> int:
    """Bit width (2 or 4) whose kernel maps the range exactly, or 0 if none.

    The `embedding_Nbit` kernels reconstruct each packed field as
    `field - BIAS`, so a value `v` must be stored as `v - quant_min` with
    `quant_min == -BIAS` and fit in N bits. Only ranges that start at the
    matching bias and whose span fits are packable; anything else is rejected.
    """
    span = int(quant_max) - int(quant_min) + 1
    for bits, bias in ((4, _INT4_KERNEL_BIAS), (2, _INT2_KERNEL_BIAS)):
        if int(quant_min) == -bias and 0 < span <= (1 << bits):
            return bits
    return 0


def _pack_int4(
    int8_weight: torch.Tensor, quant_min: int, quant_max: int
) -> torch.Tensor:
    """Pack an int8 table holding values in [quant_min, quant_max] into uint8.

    Two consecutive columns (2k, 2k+1) share one byte; the even column goes in
    the high nibble, the odd column in the low nibble. Each value is encoded as
    an unsigned nibble `v - quant_min`, so `quant_min` maps to 0 and the stored
    nibble spans the whole 4-bit range. This layout matches the `embedding_4bit`
    kernel (kernels/quantized/cpu/embeddingxb.cpp, `weight_value`).

    Range-aware: the clamp and bias are derived from the data's actual
    quant_min/quant_max instead of assuming a fixed symmetric [-8, 7] table.
    """
    hidden = int8_weight.shape[-1]
    if hidden % 2 != 0:
        raise ValueError("int4 embedding requires an even embedding dim")
    span = int(quant_max) - int(quant_min) + 1
    if not (0 < span <= 16):
        raise ValueError(
            f"int4 embedding range [{quant_min}, {quant_max}] does not fit a nibble"
        )
    # The embedding_4bit kernel hardcodes `nibble - BIAS`; reconstructing the
    # original value therefore requires the stored nibble == v - quant_min with
    # quant_min == -BIAS. Refuse to produce silently-wrong numerics otherwise.
    if int(quant_min) != -_INT4_KERNEL_BIAS:
        raise ValueError(
            "embedding_4bit kernel only reconstructs data whose quant_min == -8 "
            f"(got {quant_min})"
        )
    shifted = (
        int8_weight.to(torch.int64).clamp(int(quant_min), int(quant_max))
        - int(quant_min)
    ).to(torch.uint8)
    pairs = shifted.view(*shifted.shape[:-1], hidden // 2, 2)
    packed = (pairs[..., 0] << 4) | pairs[..., 1]
    return packed.contiguous()


def _pack_int2(
    int8_weight: torch.Tensor, quant_min: int, quant_max: int
) -> torch.Tensor:
    """Pack an int8 table holding values in [quant_min, quant_max] into uint8.

    Four consecutive columns (4k .. 4k+3) share one byte, with column 4k+`s`
    in the `2*s`-bit field. Each value is encoded as an unsigned field
    `v - quant_min`. This matches the `embedding_2bit` kernel
    (kernels/quantized/cpu/embeddingxb.cpp, `weight_value`).
    """
    hidden = int8_weight.shape[-1]
    if hidden % 4 != 0:
        raise ValueError("int2 embedding requires embedding dim divisible by 4")
    if int(quant_min) != -_INT2_KERNEL_BIAS:
        raise ValueError(
            "embedding_2bit kernel only reconstructs data whose quant_min == -2 "
            f"(got {quant_min})"
        )
    shifted = (
        int8_weight.to(torch.int64).clamp(int(quant_min), int(quant_max))
        - int(quant_min)
    ).to(torch.uint8)
    quads = shifted.view(*shifted.shape[:-1], hidden // 4, 4)
    packed = (
        quads[..., 0]
        | (quads[..., 1] << 2)
        | (quads[..., 2] << 4)
        | (quads[..., 3] << 6)
    )
    return packed.contiguous()


def _repackable(weight: torch.Tensor, bits: int) -> bool:
    per_byte = 8 // bits  # 8 // bits values share one byte
    return (
        weight.dtype == torch.int8
        and weight.dim() == 2
        and weight.shape[1] % per_byte == 0
    )


def _fake_for(shape, dtype, device, fake_mode):
    """Create a fake tensor reusing the graph's fake mode so to_edge sees
    consistent fake modes across all inputs."""
    with fake_mode:
        return torch.empty(shape, dtype=dtype, device=device)


class FuseEmbeddingDequantPass:
    """Fuse `dequantize_per_channel -> aten.embedding` into a single
    `embedding_2bit.dtype` / `embedding_4bit.dtype` op with a dense packed weight.

    Gemma 4 text decoder embeddings are quantized per vocab row (axis=0,
    qmin/qmax = -8/7, i.e. int4 values stored unpacked in an int8 table). The
    exported program dequantizes the *entire* vocab table to fp32 before the
    embedding gather. Folding the pattern into one `embedding_Nbit` op gathers
    and dequantizes only the selected tokens, keeping the whole embedding table
    as packed N-bit on device. It is equivalent because per-row dequant commutes
    with gather and zero points are all zero (symmetric), so they are dropped.

    The repack is range-aware: `_pack_int4` / `_pack_int2` clamp to and bias by
    the data's actual quant_min/quant_max (read from the dequant node) rather
    than assuming a fixed table. Because the `embedding_Nbit` kernels hardcode a
    bias (8 for int4, 2 for int2), fusion only applies when the range maps onto
    the matching N-bit field (see `_pick_bits`); otherwise the pattern is left
    untouched (skipped) instead of being silently mis-packed.

    Operates on an `ExportedProgram` because repacking the weight requires
    mutating its backing constant. Weight buffers appear either as `get_attr`
    nodes (strict export) or `placeholder` nodes (non-strict / loaded `.pt2`).
    """

    _DEQUANT = torch.ops.quantized_decomposed.dequantize_per_channel.default
    _EMBEDDING = torch.ops.aten.embedding.default
    _EMBEDDING_2BIT = torch.ops.quantized_decomposed.embedding_2bit.dtype
    _EMBEDDING_4BIT = torch.ops.quantized_decomposed.embedding_4bit.dtype

    def _embedding_op(self, bits: int):
        return self._EMBEDDING_2BIT if bits == 2 else self._EMBEDDING_4BIT

    def __call__(self, ep: ExportedProgram) -> ExportedProgram:
        gm = ep.graph_module
        # Iterate over a snapshot: `_fuse` can erase the node we're visiting.
        for node in list(gm.graph.nodes):
            if node.op != "call_function" or node.target != self._EMBEDDING:
                continue
            dequant = node.args[0]
            if not (
                isinstance(dequant, Node)
                and dequant.op == "call_function"
                and dequant.target == self._DEQUANT
            ):
                continue
            if not self._fuse(ep, node, dequant):
                continue

        gm.recompile()
        dead_code_elimination_pass(gm)
        return ep

    def _fuse(self, ep: ExportedProgram, embedding: Node, dequant: Node) -> bool:
        qmin, qmax = dequant.args[4], dequant.args[5]
        weight = dequant.args[0]
        if not isinstance(weight, Node):
            return False
        if not self._supported(ep, dequant):
            return False
        bits = _pick_bits(qmin, qmax)
        if bits == 0:
            return False
        if not self._repack_weight(ep, weight, qmin, qmax, bits):
            return False

        scales = dequant.args[1]
        indices = embedding.args[1]

        out_dtype = torch.float32
        if (val := embedding.meta.get("val")) is not None:
            out_dtype = val.dtype

        graph = embedding.graph
        with graph.inserting_before(embedding):
            fused = graph.call_function(
                self._embedding_op(bits),
                (weight, scales, None, qmin, qmax, indices),
                {"dtype": out_dtype},
            )
        # Preserve shape/dtype (and node metadata) so downstream edge passes
        # that read `node.meta["val"]` keep working.
        fused.meta.update(embedding.meta.items())
        embedding.replace_all_uses_with(fused)
        graph.erase_node(embedding)
        return True

    def _supported(self, ep: ExportedProgram, dequant: Node) -> bool:
        """Reject patterns that would silently dequantize incorrectly."""
        # Per-row dequant (axis=0) is required for the gather to commute with
        # the per-embedding-row dequant. Other axes do not commute.
        if int(dequant.args[3]) != 0:
            return False
        # The fused op drops zero points; only safe when they are all zero
        # (symmetric quantization).
        zp = dequant.args[2]
        if zp is None:
            return True
        if isinstance(zp, Node):
            zp_val = self._node_value(ep, zp)
            return zp_val is not None and bool(torch.all(zp_val == 0))
        return int(zp) == 0

    def _node_value(self, ep: ExportedProgram, node: Node):
        if node.op == "get_attr":
            return getattr(ep.graph_module, node.target)
        if node.op == "placeholder":
            return _placeholder_value(ep, node)
        return None

    def _repack_weight(
        self,
        ep: ExportedProgram,
        weight: Node,
        quant_min: int,
        quant_max: int,
        bits: int,
    ) -> bool:
        from torch._guards import detect_fake_mode

        gm = ep.graph_module
        # Only repack when the weight feeds exclusively this dequant pattern.
        # Repacking mutates the single backing constant, so any other consumer
        # (another op, an output, or a second dequant with different qmin)
        # would observe the corrupted packed bytes.
        if not all(u.target == self._DEQUANT for u in weight.users):
            return False

        if weight.op == "get_attr":
            t = getattr(gm, weight.target)
        elif weight.op == "placeholder":
            t = _placeholder_value(ep, weight)
        else:
            return False
        if not isinstance(t, torch.Tensor) or t.dim() != 2:
            return False
        if not _repackable(t, bits):
            return False

        packed = (
            _pack_int4(t, quant_min, quant_max)
            if bits == 4
            else _pack_int2(t, quant_min, quant_max)
        )
        val = weight.meta.get("val")
        fake_mode = detect_fake_mode([val]) if val is not None else None
        weight.meta["val"] = _fake_for(
            packed.shape, packed.dtype, packed.device, fake_mode
        )
        if weight.op == "get_attr":
            setattr(gm, weight.target, packed)
        else:
            _replace_placeholder_value(ep, weight, packed)
        return True


def _placeholder_value(ep: ExportedProgram, node: Node):
    """Resolve the backing tensor for a `placeholder` weight node."""
    from torch.export.graph_signature import InputKind

    for spec in ep.graph_signature.input_specs:
        arg = getattr(spec, "arg", None)
        name = getattr(arg, "name", arg)
        if str(name) != node.name:
            continue
        target = spec.target
        if getattr(spec, "kind", None) == InputKind.CONSTANT_TENSOR:
            return (ep.constants or {}).get(target)
        return ep.state_dict.get(target)
    return None


def _replace_placeholder_value(ep: ExportedProgram, node: Node, value: torch.Tensor):
    from torch.export.graph_signature import InputKind

    for spec in ep.graph_signature.input_specs:
        arg = getattr(spec, "arg", None)
        name = getattr(arg, "name", arg)
        if str(name) != node.name:
            continue
        target = spec.target
        if getattr(spec, "kind", None) == InputKind.CONSTANT_TENSOR:
            ep.constants[target] = value.detach()
        else:
            ep.state_dict[target] = value.detach()
        return
