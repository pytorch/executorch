# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""ExecuTorch-internal dp4a-planar INT5 tensor for the CUDA W5A8 dp4a decode kernel.

``CudaDp4aPlanarInt5Tensor`` is an ExecuTorch-internal tensor subclass that stores
a genuine 5-bit packed weight (0.625 B/elem) in a dp4a-friendly *planar* layout:
the 5 bits are split into a low nibble plane (``ql``) and a 1-bit high plane
(``qh``) so the decode kernel can run the W5A8 dp4a matvec directly. Used for
GGUF Q5_K weights. Unlike the symmetric INT6 path (``CudaDp4aPlanarInt6Tensor``),
Q5_K is *asymmetric* (it has both ``d`` and ``dmin``), so this format carries a
zero tensor exactly like the INT4 path.

Build one with :meth:`from_exportable_gguf` (from a native Q5_K
``ExportableGGUFTensor`` — it reuses the shared Q5_K block decode in
``extension/llm/export/gguf.py`` and then feeds the internal ql/qh packer
:meth:`_from_intx_int8`). This class owns only the 5-bit pack and the metadata
re-encoding, never the Q5_K block decode.

The stored value is the raw unsigned ``u = q`` in ``[0, 31]`` (no offset — the
per-group zero point is subtracted in the decode kernel, mirroring INT4). The 5
bits are split into two planes that mirror the INT4 nibble layout so the kernel
can reuse the INT4 even/odd extraction verbatim:

    ql         : (N, K/2) uint8 — low-nibble plane, nibble-packed even/odd
                 (``ql[:, j] = lo[:, 2j] | (lo[:, 2j+1] << 4)``, ``lo = u & 0xF``).
    qh         : (N, K/8) uint8 — high-1-bit plane, 8 values/byte, arranged per
                 32-weight chunk as 4 bytes (one per dp4a word); each byte holds
                 the four 1-bit highs of that word's even weights in the low
                 nibble and its odd weights in the high nibble
                 (``hi_even_nibble | (hi_odd_nibble << 4)``, ``hi = (u >> 4) & 1``).
    scale      : (N, n_groups) uint8 — per-group scale *codes*, coalesced.
    scale_step : (N, K/256) fp16 — per-256-super-block scale step; the real
                 per-group scale is ``scale_code * scale_step[:, g // 8]``.
    zero_point : (N, n_groups) uint8 — per-group zero *codes*, coalesced.
    zero_step  : (N, K/256) fp16 — per-256-super-block zero step; the real
                 per-group zero is ``zero_code * zero_step[:, g // 8]``. Both
                 per-256 fp16 steps mirror GGUF Q5_K's per-super-block fp16
                 ``d`` / ``dmin`` and are packed into ONE 32-bit warp-shuffle
                 word by the decode kernel (z_pack), exactly like the INT4 path.
                 The finer per-256 step (vs the previous per-row step) improves
                 whole-weight dequant SNR at ~5.625 bpw.

The pack/unpack helpers (:func:`pack_int5`, :func:`unpack_int5`) must stay in
lockstep with ``int5_plain_mm.cuh`` (the decode kernel) — the per-32-weight
even/odd high-bit byte order is the single most error-prone detail and is
covered by the pack round-trip and the C++ gtest.
"""

from typing import List, Optional, Tuple

import torch
from torchao.utils import TorchAOBaseTensor

__all__ = [
    "CudaDp4aPlanarInt5Tensor",
    "pack_int5",
    "unpack_int5",
]

_CODE_MAX = 255  # uint8 code range [0, 255] (both scale and zero)
_SUPER_BLOCK = 256  # weights per super-block (GGUF Q5_K QK_K); steps are per this


def pack_int5(q: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """Pack unsigned Q5_K values ``u`` in ``[0, 31]`` into the (ql, qh) planes.

    Args:
        q: (N, K) integer tensor with values in ``[0, 31]``.

    Returns:
        ``(ql, qh)`` where ``ql`` is ``(N, K/2)`` uint8 and ``qh`` is
        ``(N, K/8)`` uint8 (see the module docstring for the layout).
    """
    if q.dim() != 2:
        raise ValueError(f"pack_int5 expects a 2-D tensor, got shape {tuple(q.shape)}")
    N, K = int(q.shape[0]), int(q.shape[1])
    if K % 32 != 0:
        raise ValueError(f"K={K} must be a multiple of 32 for INT5 packing")

    # All intermediates are uint8 (values fit in a byte) to keep peak memory low.
    u = q.to(torch.uint8)  # [0, 31]
    lo = u & 0xF  # low nibble (uint8)
    hi = (u >> 4) & 0x1  # high 1 bit (uint8)

    # ql: nibble-pack the low plane even/odd, exactly like the INT4 path.
    ql = lo[:, 0::2] | (lo[:, 1::2] << 4)  # (N, K/2) uint8

    # qh: per 32-weight chunk -> 4 bytes (one per dp4a word); each byte packs the
    # four 1-bit highs of that word's even weights (low nibble) and odd weights
    # (high nibble), field j at bit j.
    chunks = K // 32
    hw = hi.reshape(N, chunks, 4, 8)  # (N, chunk, word, pos-in-word)
    even = hw[..., 0::2]  # (N, chunk, 4, 4) positions 0,2,4,6
    odd = hw[..., 1::2]  # (N, chunk, 4, 4) positions 1,3,5,7
    # Explicit OR (not sum) keeps the result uint8 (torch.sum would promote).
    hi_even_nib = (
        even[..., 0] | (even[..., 1] << 1) | (even[..., 2] << 2) | (even[..., 3] << 3)
    )  # (N, chunk, 4) uint8, one nibble per word
    hi_odd_nib = (
        odd[..., 0] | (odd[..., 1] << 1) | (odd[..., 2] << 2) | (odd[..., 3] << 3)
    )
    qh = hi_even_nib | (hi_odd_nib << 4)  # (N, chunk, 4) uint8
    qh = qh.reshape(N, K // 8)
    return ql.contiguous(), qh.contiguous()


def unpack_int5(ql: torch.Tensor, qh: torch.Tensor, N: int, K: int) -> torch.Tensor:
    """Inverse of :func:`pack_int5`. Returns ``(N, K)`` uint8 ``u`` in ``[0, 31]``.

    All intermediates stay uint8 — the value is unsigned (the zero point is
    applied in dequant, not here).
    """
    qlu = ql.to(torch.uint8)
    lo_even = qlu & 0xF  # low nibble -> even weights
    lo_odd = (qlu >> 4) & 0xF  # high nibble -> odd weights
    lo = torch.stack([lo_even, lo_odd], dim=-1).reshape(N, K)  # uint8

    chunks = K // 32
    qhu = qh.to(torch.uint8).reshape(N, chunks, 4)  # one byte per word
    hi_even_nib = qhu & 0xF  # (N, chunk, 4)
    hi_odd_nib = (qhu >> 4) & 0xF
    hi_even = torch.stack(
        [(hi_even_nib >> s) & 0x1 for s in range(4)], dim=-1
    )  # (N, chunk, 4, 4) uint8
    hi_odd = torch.stack([(hi_odd_nib >> s) & 0x1 for s in range(4)], dim=-1)
    hi = torch.empty(N, chunks, 4, 8, dtype=torch.uint8, device=ql.device)
    hi[..., 0::2] = hi_even
    hi[..., 1::2] = hi_odd
    hi = hi.reshape(N, K)

    return lo | (hi << 4)  # [0, 31] uint8


def _encode_uint8_per_super(
    x: torch.Tensor,
    group_size: int,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Encode a (N, n_groups) non-negative per-group tensor to per-256 uint8 codes.

    Used for both the scale and the zero. A super-block is ``_SUPER_BLOCK``
    (256) weights = ``groups_per_super = 256 // group_size`` groups (8 for
    group_size=32). Q5_K per-group scales/zeros are non-negative (``d``/``dmin``
    >= 0), so an unsigned code is exact at the endpoints. Returns
    ``(codes, step)`` where ``codes`` is ``(N, n_groups)`` uint8 and ``step`` is
    ``(N, n_super)`` fp16 with ``n_super = n_groups // groups_per_super = K //
    256``, such that ``code * step[:, g // groups_per_super] ≈ x``. The step is
    the per-256-super-block max / 255 rounded to fp16 (what the kernel reads), so
    encode and decode agree. Mirrors the INT4 ``_encode_uint8_per_super`` (the
    int5 scale/zero are already row-major (N, n_groups), so no transpose).
    """
    xf = x.contiguous().float()  # (N, n_groups), non-negative
    N, n_groups = int(xf.shape[0]), int(xf.shape[1])
    groups_per_super = _SUPER_BLOCK // int(group_size)
    if groups_per_super < 1:
        raise ValueError(
            f"group_size={group_size} must be <= {_SUPER_BLOCK} for the per-256 step"
        )
    if n_groups % groups_per_super != 0:
        raise ValueError(
            f"n_groups={n_groups} must be a multiple of {groups_per_super} "
            f"(K must be a multiple of {_SUPER_BLOCK}) for group_size={group_size}"
        )
    n_super = n_groups // groups_per_super
    xb = xf.reshape(N, n_super, groups_per_super)  # (N, n_super, gps)
    block_max = xb.amax(dim=2, keepdim=True).clamp_min(1e-30)  # (N, n_super, 1)
    step = (block_max / _CODE_MAX).to(torch.float16)  # (N, n_super, 1) fp16
    step_f = step.float().clamp_min(1e-30)
    codes = torch.round(xb / step_f).clamp_(0, _CODE_MAX).to(torch.uint8)
    codes = codes.reshape(N, n_groups).contiguous()
    return codes, step.squeeze(2).contiguous()


class CudaDp4aPlanarInt5Tensor(TorchAOBaseTensor):
    """Dp4a-planar 5-bit weight (ql/qh split bit-planes + per-group scale/zero), asymmetric.

    ExecuTorch-internal; see the module docstring. The CUDA decode/prefill
    dispatch (``int5_dispatch.py``) is selected by *type* — it is registered on
    this class only.
    """

    tensor_data_names = ["ql", "qh", "scale", "scale_step", "zero_point", "zero_step"]
    tensor_attribute_names = ["block_size", "shape"]

    def __new__(
        cls,
        ql: torch.Tensor,
        qh: torch.Tensor,
        scale: torch.Tensor,
        scale_step: torch.Tensor,
        zero_point: torch.Tensor,
        zero_step: torch.Tensor,
        block_size: List[int],
        shape: torch.Size,
    ):
        kwargs = {}
        kwargs["device"] = ql.device
        kwargs["dtype"] = torch.bfloat16
        kwargs["requires_grad"] = False
        return torch.Tensor._make_wrapper_subclass(cls, shape, **kwargs)  # type: ignore[attr-defined]

    def __init__(
        self,
        ql: torch.Tensor,
        qh: torch.Tensor,
        scale: torch.Tensor,
        scale_step: torch.Tensor,
        zero_point: torch.Tensor,
        zero_step: torch.Tensor,
        block_size: List[int],
        shape: torch.Size,
    ):
        super().__init__()
        self.ql = ql
        self.qh = qh
        self.scale = scale
        self.scale_step = scale_step
        self.zero_point = zero_point
        self.zero_step = zero_step
        self.block_size = block_size

    def _quantization_type(self):
        return f"shape={self.shape}, block_size={self.block_size}, device={self.device}"

    @classmethod
    def _from_intx_int8(cls, t: torch.Tensor) -> "CudaDp4aPlanarInt5Tensor":
        """Build from an asymmetric int5 ``IntxUnpackedToInt8Tensor`` decoded from Q5_K.

        The source stores centered quants ``qdata`` in ``[-16, 15]`` (int8) with a
        (float) bf16 ``zero_point`` and per-group ``scale``, both ``(N, n_groups)``
        (dequant = ``scale * (qdata - zero_point)``). We shift back to the raw
        unsigned ``u = qdata + 16`` in ``[0, 31]`` and the unsigned zero
        ``zero_u = zero_point + 16`` (both non-negative), bit-pack the ql/qh
        planes, and re-encode scale/zero to per-group uint8 codes each with a
        per-256-super-block fp16 step (mirroring the INT4 z_pack). All of this is
        baked into the serialized weight constant here, once at pack time.
        """
        q_centered = t.qdata
        q_min, q_max = int(q_centered.min()), int(q_centered.max())
        if q_min < -16 or q_max > 15:
            raise ValueError(
                f"Q5_K centered values must be in [-16, 15], got [{q_min}, {q_max}]"
            )
        u = (q_centered.to(torch.int16) + 16).to(torch.uint8)  # [0, 31]
        # Unsigned zero in q-space (>= 0 for Q5_K's non-negative affine min).
        zero_u = (t.zero_point.float() + 16.0).clamp_min(0.0)  # (N, n_groups)
        scale = t.scale.float()  # (N, n_groups), non-negative

        gs = int(t.block_size[-1])
        ql, qh = pack_int5(u)
        scale_codes, scale_step = _encode_uint8_per_super(scale, gs)
        zero_codes, zero_step = _encode_uint8_per_super(zero_u, gs)
        return cls(
            ql,
            qh,
            scale_codes,
            scale_step,
            zero_codes,
            zero_step,
            list(t.block_size),
            t.shape,
        )

    @classmethod
    def from_exportable_gguf(cls, gt) -> "CudaDp4aPlanarInt5Tensor":
        """Build from a native Q5_K ``ExportableGGUFTensor``.

        Reuses the shared Q5_K block decode in
        ``extension/llm/export/gguf.py`` (``to_intx_unpacked_to_int8_tensor`` ->
        an asymmetric int5 tensor centered in ``[-16, 15]``), then bit-packs into
        the ql/qh planes via :meth:`_from_intx_int8`. The Q5_K decode lives in one
        place; this class only owns the 5-bit pack + metadata re-encode.
        """
        if gt.ggml_type != "q5_k":
            raise ValueError(
                "CudaDp4aPlanarInt5Tensor.from_exportable_gguf requires a q5_k "
                f"ExportableGGUFTensor, got {gt.ggml_type!r}"
            )
        return cls._from_intx_int8(gt.to_intx_unpacked_to_int8_tensor())

    def dequantize(self, output_dtype: Optional[torch.dtype] = None) -> torch.Tensor:
        """Dequantize to a dense tensor (asymmetric: ``w = scale * (u - zero)``).

        Reconstructs the per-group scale/zero from the uint8 codes and their
        per-256-super-block fp16 steps (broadcast over the 8 groups per
        super-block). Used for the tied lm_head / token embedding (which can't
        gather a packed tensor) and as the numerical reference.
        """
        dtype = output_dtype if output_dtype is not None else torch.bfloat16
        N, K = int(self.shape[0]), int(self.shape[1])
        gs = self.block_size[-1]
        n_groups = K // gs
        n_super = int(self.scale_step.shape[1])
        groups_per_super = n_groups // n_super

        u = unpack_int5(self.ql, self.qh, N, K).to(torch.float32)
        scale_code = self.scale.to(torch.float32)  # (N, n_groups)
        scale_step = self.scale_step.float().repeat_interleave(groups_per_super, dim=1)
        scale = (scale_code * scale_step).repeat_interleave(gs, dim=1)  # (N, K)

        zero_code = self.zero_point.to(torch.float32)  # (N, n_groups)
        zero_step = self.zero_step.float().repeat_interleave(groups_per_super, dim=1)
        zero = (zero_code * zero_step).repeat_interleave(gs, dim=1)  # (N, K)
        return (scale * (u - zero)).to(dtype)


# Allow a model with CudaDp4aPlanarInt5Tensor weights to be loaded with
# `weights_only=True` (mirrors torchao quantized tensors).
torch.serialization.add_safe_globals([CudaDp4aPlanarInt5Tensor])
