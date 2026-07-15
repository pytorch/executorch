# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""ExecuTorch-internal dp4a-planar INT6 tensor for the CUDA W6A8 dp4a decode kernel.

``CudaDp4aPlanarInt6Tensor`` is an ExecuTorch-internal tensor subclass that stores a
genuine 6-bit packed weight (0.75 B/elem) in a dp4a-friendly *planar* layout: the
6 bits are split into two bit-planes (``ql``/``qh``) so the decode kernel can run
the W6A8 dp4a matvec directly. Used for GGUF Q6_K weights. Unlike
the int8 path (``IntxUnpackedToInt8Tensor``, one int8 per 6-bit value), this
format wastes no bits and carries no zero tensor — Q6_K is symmetric.

Build one with :meth:`from_exportable_gguf` (from a native Q6_K
``ExportableGGUFTensor`` — it reuses the shared Q6_K block decode in
``extension/llm/export/gguf.py`` and then feeds the internal ql/qh packer
:meth:`_from_intx_int8`). This class owns only the 6-bit pack, never the Q6_K
block decode.

The stored value is ``u = q + 32`` in ``[0, 63]`` (``q`` in ``[-32, 31]``); the
constant ``-32`` offset is applied in the decode kernel. The 6 bits are split
into two planes that mirror the INT4 nibble layout so the kernel can reuse the
INT4 even/odd extraction verbatim:

    ql    : (N, K/2) uint8 — low-nibble plane, nibble-packed even/odd
            (``ql[:, j] = lo[:, 2j] | (lo[:, 2j+1] << 4)``, ``lo = u & 0xF``).
    qh    : (N, K/4) uint8 — high-2-bit plane, 4 values/byte, arranged per
            32-weight chunk as ``hi_even_packed[4]`` then ``hi_odd_packed[4]``;
            each byte holds the four 2-bit highs (``hi = (u >> 4) & 0x3``) of one
            8-weight dp4a word, bit field ``j`` (bits ``2j..2j+1``) = the high 2
            bits of that word's ``j``-th even/odd weight.
    scale : (N, K/gs) int8 — per-group signed scale *codes*, row-major (already
            coalesced; the decode kernel reads it row-for-row, no transpose).
    steps : (N, K/256) fp16 — per-256-super-block scale step; the real per-group
            scale is ``scale_code * steps[:, g // (256 // gs)]``.

Metadata encoding: the per-group scale is a signed int8 *code* with a
**per-256-super-block fp16 step** (``scale = code * step``). group_size is 16
(GGUF Q6_K), so a 256-weight super-block spans 16 groups and there are ``K/256``
scale steps per row. This mirrors GGUF Q6_K's own per-super-block fp16 ``d``: the
finer per-256 step (vs a single per-row step) recovers that native granularity
and lifts whole-weight dequant SNR. The scale step MUST be fp16 (mirrors int4,
where bf16 for the per-256 step cost ~0.39 dB); the int8 code width is kept (NOT
narrowed to 6-bit). The weight stays symmetric (no zero) and the planar ql/qh
layout is UNCHANGED.

The pack/unpack helpers (:func:`pack_int6`, :func:`unpack_int6`) must stay in
lockstep with ``int6_plain_mm.cuh`` (the decode kernel) — the per-32-weight
``hi_even``/``hi_odd`` byte order is the single most error-prone detail and is
covered by the pack round-trip and the C++ gtest.
"""

from typing import List, Optional, Tuple

import torch
from torchao.utils import TorchAOBaseTensor

__all__ = [
    "CudaDp4aPlanarInt6Tensor",
    "pack_int6",
    "unpack_int6",
]

_CODE_ABSMAX = 127  # int8 signed code range [-127, 127]
_SUPER_BLOCK = 256  # weights per super-block (GGUF Q6_K QK_K); scale step is per this


def pack_int6(q: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """Pack symmetric Q6_K int values into the (ql, qh) planes.

    Args:
        q: (N, K) integer tensor with values in ``[-32, 31]``.

    Returns:
        ``(ql, qh)`` where ``ql`` is ``(N, K/2)`` uint8 and ``qh`` is
        ``(N, K/4)`` uint8 (see the module docstring for the layout).
    """
    if q.dim() != 2:
        raise ValueError(f"pack_int6 expects a 2-D tensor, got shape {tuple(q.shape)}")
    N, K = int(q.shape[0]), int(q.shape[1])
    if K % 32 != 0:
        raise ValueError(f"K={K} must be a multiple of 32 for INT6 packing")

    # All intermediates are uint8 (values fit in a byte) to keep peak memory low
    # — important for the ~1.4B-element tied token embedding.
    u = (q.to(torch.int16) + 32).to(torch.uint8)  # [0, 63]
    lo = u & 0xF  # low nibble (uint8)
    hi = (u >> 4) & 0x3  # high 2 bits (uint8)

    # ql: nibble-pack the low plane even/odd, exactly like the INT4 path.
    ql = lo[:, 0::2] | (lo[:, 1::2] << 4)  # (N, K/2) uint8

    # qh: per 32-weight chunk -> [hi_even_packed[4], hi_odd_packed[4]]; each byte
    # packs the four 2-bit highs of one 8-weight dp4a word, field j at bits 2j.
    chunks = K // 32
    hw = hi.reshape(N, chunks, 4, 8)  # (N, chunk, word, pos-in-word)
    even = hw[..., 0::2]  # (N, chunk, 4, 4) positions 0,2,4,6
    odd = hw[..., 1::2]  # (N, chunk, 4, 4) positions 1,3,5,7
    # Explicit OR (not sum) keeps the result uint8 (torch.sum would promote).
    hi_even_byte = (
        even[..., 0] | (even[..., 1] << 2) | (even[..., 2] << 4) | (even[..., 3] << 6)
    )  # (N, chunk, 4) uint8
    hi_odd_byte = (
        odd[..., 0] | (odd[..., 1] << 2) | (odd[..., 2] << 4) | (odd[..., 3] << 6)
    )
    qh = torch.cat([hi_even_byte, hi_odd_byte], dim=-1)  # (N, chunk, 8) uint8
    qh = qh.reshape(N, K // 4)
    return ql.contiguous(), qh.contiguous()


def unpack_int6(ql: torch.Tensor, qh: torch.Tensor, N: int, K: int) -> torch.Tensor:
    """Inverse of :func:`pack_int6`. Returns ``(N, K)`` int16 q in ``[-32, 31]``.

    Intermediates are uint8 to keep peak memory low; only the final ``- 32`` shift
    (which produces negatives) widens to int16.
    """
    qlu = ql.to(torch.uint8)
    lo_even = qlu & 0xF  # low nibble -> even weights
    lo_odd = (qlu >> 4) & 0xF  # high nibble -> odd weights
    lo = torch.stack([lo_even, lo_odd], dim=-1).reshape(N, K)  # uint8

    chunks = K // 32
    qhu = qh.to(torch.uint8).reshape(N, chunks, 8)
    hi_even_byte = qhu[:, :, 0:4]  # (N, chunk, 4) word w
    hi_odd_byte = qhu[:, :, 4:8]  # (N, chunk, 4)
    hi_even = torch.stack(
        [(hi_even_byte >> s) & 0x3 for s in (0, 2, 4, 6)], dim=-1
    )  # (N, chunk, 4, 4) uint8
    hi_odd = torch.stack([(hi_odd_byte >> s) & 0x3 for s in (0, 2, 4, 6)], dim=-1)
    hi = torch.empty(N, chunks, 4, 8, dtype=torch.uint8, device=ql.device)
    hi[..., 0::2] = hi_even
    hi[..., 1::2] = hi_odd
    hi = hi.reshape(N, K)

    u = lo | (hi << 4)  # [0, 63] uint8
    return u.to(torch.int16) - 32


def _encode_int8_per_super(
    x: torch.Tensor,
    group_size: int,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Encode a (N, n_groups) signed per-group scale to per-256-super-block int8 codes.

    Mirrors int4's per-256 scale step, but SIGNED (Q6_K sub-scales can be
    negative) and int8-code-width: a super-block is ``_SUPER_BLOCK`` (256)
    weights = ``groups_per_super = 256 // group_size`` groups (16 for
    group_size=16). Returns ``(codes, step)`` where ``codes`` is
    ``(N, n_groups)`` int8 and ``step`` is ``(N, n_super)`` fp16 with
    ``n_super = n_groups // groups_per_super = K // 256``, such that
    ``code * step[:, g // groups_per_super] ≈ x``. The step is the
    per-256-super-block absmax / 127 rounded to fp16; rounding uses the
    fp16-rounded step (what the kernel reads) so encode and decode agree.

    The finer per-256 step (vs a single per-row step) matches GGUF Q6_K's own
    per-super-block fp16 ``d`` granularity, improving whole-weight dequant SNR.
    """
    xf = x.contiguous().float()  # (N, n_groups), signed
    N, n_groups = int(xf.shape[0]), int(xf.shape[1])
    groups_per_super = _SUPER_BLOCK // int(group_size)
    if groups_per_super < 1:
        raise ValueError(
            f"group_size={group_size} must be <= {_SUPER_BLOCK} for the per-256 "
            "scale step"
        )
    if n_groups % groups_per_super != 0:
        raise ValueError(
            f"n_groups={n_groups} must be a multiple of {groups_per_super} "
            f"(K must be a multiple of {_SUPER_BLOCK}) for group_size={group_size}"
        )
    n_super = n_groups // groups_per_super
    xb = xf.reshape(N, n_super, groups_per_super)  # (N, n_super, gps)
    block_absmax = (
        xb.abs().amax(dim=2, keepdim=True).clamp_min(1e-30)
    )  # (N, n_super, 1)
    step = (block_absmax / _CODE_ABSMAX).to(torch.float16)  # (N, n_super, 1) fp16
    step_f = step.float().clamp_min(1e-30)
    codes = torch.round(xb / step_f).clamp_(-127, 127).to(torch.int8)
    codes = codes.reshape(N, n_groups).contiguous()
    return codes, step.squeeze(2).contiguous()


class CudaDp4aPlanarInt6Tensor(TorchAOBaseTensor):
    """Dp4a-planar 6-bit weight (ql/qh split bit-planes + per-group scale), symmetric.

    ExecuTorch-internal; see the module docstring. The CUDA decode/prefill
    dispatch (``int6_dispatch.py``) is selected by *type* — it is registered on
    this class only.
    """

    tensor_data_names = ["ql", "qh", "scale", "steps"]
    tensor_attribute_names = ["block_size", "shape"]

    def __new__(
        cls,
        ql: torch.Tensor,
        qh: torch.Tensor,
        scale: torch.Tensor,
        steps: torch.Tensor,
        block_size: List[int],
        shape: torch.Size,
    ):
        kwargs = {}
        kwargs["device"] = ql.device
        # The weight represents a bf16 tensor; pin the wrapper dtype to bf16
        # (decoupled from ``steps``, which is now fp16 for the per-256 scale
        # step) so F.linear / the tied embedding see a bf16 weight as before.
        kwargs["dtype"] = torch.bfloat16
        kwargs["requires_grad"] = False
        return torch.Tensor._make_wrapper_subclass(cls, shape, **kwargs)  # type: ignore[attr-defined]

    def __init__(
        self,
        ql: torch.Tensor,
        qh: torch.Tensor,
        scale: torch.Tensor,
        steps: torch.Tensor,
        block_size: List[int],
        shape: torch.Size,
    ):
        super().__init__()
        self.ql = ql
        self.qh = qh
        self.scale = scale
        self.steps = steps
        self.block_size = block_size

    def _quantization_type(self):
        return (
            f"shape={self.shape}, block_size={self.block_size}, "
            f"device={self.device}"
        )

    @classmethod
    def _from_intx_int8(cls, t: torch.Tensor) -> "CudaDp4aPlanarInt6Tensor":
        """Internal ql/qh packer: build from a symmetric int8 ``IntxUnpackedToInt8Tensor`` decoded from Q6_K.

        The source is symmetric (zero_point == 0), ``qdata`` is int8 in
        ``[-32, 31]`` and ``scale`` is ``(N, K/16)``. The ql/qh bit-pack is baked
        into the serialized weight constant here, once at pack time.
        """
        q = t.qdata
        if not bool(torch.all(t.zero_point == 0)):
            raise ValueError(
                "CudaDp4aPlanarInt6Tensor._from_intx_int8 requires symmetric Q6_K "
                "weights (zero_point == 0)"
            )
        q_min, q_max = int(q.min()), int(q.max())
        if q_min < -32 or q_max > 31:
            raise ValueError(
                f"Q6_K values must be in [-32, 31], got [{q_min}, {q_max}]"
            )
        ql, qh = pack_int6(q)
        scale_codes, steps = _encode_int8_per_super(t.scale, int(t.block_size[-1]))
        return cls(
            ql,
            qh,
            scale_codes,
            steps,
            list(t.block_size),
            t.shape,
        )

    @classmethod
    def from_exportable_gguf(cls, gt) -> "CudaDp4aPlanarInt6Tensor":
        """Build from a native Q6_K ``ExportableGGUFTensor``.

        Reuses the shared Q6_K block decode in
        ``extension/llm/export/gguf.py`` (``to_intx_unpacked_to_int8_tensor`` ->
        a symmetric int8 tensor in ``[-32, 31]``), then bit-packs into the ql/qh
        planes via :meth:`_from_intx_int8`. The Q6_K decode lives in one place;
        this class only owns the 6-bit pack.
        """
        if gt.ggml_type != "q6_k":
            raise ValueError(
                "CudaDp4aPlanarInt6Tensor.from_exportable_gguf requires a q6_k "
                f"ExportableGGUFTensor, got {gt.ggml_type!r}"
            )
        return cls._from_intx_int8(gt.to_intx_unpacked_to_int8_tensor())

    def dequantize(self, output_dtype: Optional[torch.dtype] = None) -> torch.Tensor:
        """Dequantize to a dense tensor (symmetric: ``w = q * scale``).

        Reconstructs the per-group scale from the int8 codes and the per-256
        fp16 step (broadcast over the ``groups_per_super`` groups in each
        super-block). Used for the tied lm_head / token embedding (which can't
        gather a packed tensor) and as the numerical reference.
        """
        dtype = output_dtype if output_dtype is not None else torch.bfloat16
        N, K = int(self.shape[0]), int(self.shape[1])
        gs = self.block_size[-1]
        n_groups = K // gs
        n_super = int(self.steps.shape[1])
        groups_per_super = n_groups // n_super
        q = unpack_int6(self.ql, self.qh, N, K).to(dtype)
        step_g = self.steps.to(dtype).repeat_interleave(groups_per_super, dim=1)
        scale = (self.scale.to(dtype) * step_g).repeat_interleave(gs, dim=-1)
        return (q * scale).to(dtype)


# Allow a model with CudaDp4aPlanarInt6Tensor weights to be loaded with
# `weights_only=True` (mirrors torchao quantized tensors).
torch.serialization.add_safe_globals([CudaDp4aPlanarInt6Tensor])
