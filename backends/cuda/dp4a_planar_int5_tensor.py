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
    zero_point : (N, n_groups) uint8 — per-group zero *codes*, coalesced.
    steps      : (N, 2) bf16 — per-row super-scales (scale_step, zero_step); the
                 real per-group values are ``code * step``. This compacts the
                 metadata from bf16 scale + bf16 zero (4 B/group, 5.625 bpw) to
                 uint8 scale + uint8 zero + a tiny per-row step (2 B/group,
                 5.125 bpw) at ~baseline accuracy — Q5_K group scales/zeros are
                 non-negative and fit an 8-bit per-row-normalized code (measured
                 dequant SNR ~50 dB, matching the bf16 metadata it replaces).

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


def _encode_uint8_per_row(
    x: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Encode a (N, n_groups) non-negative per-group tensor to per-row uint8 codes.

    Q5_K per-group scales and zeros are both non-negative (``d``/``dmin`` >= 0),
    so an unsigned code is exact at the endpoints and ~baseline elsewhere.
    Returns ``(codes, step)`` where ``codes`` is ``(N, n_groups)`` uint8 and
    ``step`` is ``(N, 1)`` bf16, with ``code * step`` reconstructing the value.
    The step is the per-row max / 255 rounded to bf16, so the largest group in
    each row maps to ~255 and the 8-bit code spans the row's dynamic range.
    """
    xf = x.contiguous().float()  # (N, n_groups), non-negative
    row_max = xf.amax(dim=1, keepdim=True).clamp_min(1e-30)  # (N, 1)
    step = (row_max / 255.0).to(torch.bfloat16)  # (N, 1) bf16
    step_f = step.float().clamp_min(1e-30)
    codes = torch.round(xf / step_f).clamp_(0, 255).to(torch.uint8)
    return codes.contiguous(), step.contiguous()


class CudaDp4aPlanarInt5Tensor(TorchAOBaseTensor):
    """Dp4a-planar 5-bit weight (ql/qh split bit-planes + per-group scale/zero), asymmetric.

    ExecuTorch-internal; see the module docstring. The CUDA decode/prefill
    dispatch (``int5_dispatch.py``) is selected by *type* — it is registered on
    this class only.
    """

    tensor_data_names = ["ql", "qh", "scale", "zero_point", "steps"]
    tensor_attribute_names = ["block_size", "shape"]

    def __new__(
        cls,
        ql: torch.Tensor,
        qh: torch.Tensor,
        scale: torch.Tensor,
        zero_point: torch.Tensor,
        steps: torch.Tensor,
        block_size: List[int],
        shape: torch.Size,
    ):
        kwargs = {}
        kwargs["device"] = ql.device
        kwargs["dtype"] = steps.dtype
        kwargs["requires_grad"] = False
        return torch.Tensor._make_wrapper_subclass(cls, shape, **kwargs)  # type: ignore[attr-defined]

    def __init__(
        self,
        ql: torch.Tensor,
        qh: torch.Tensor,
        scale: torch.Tensor,
        zero_point: torch.Tensor,
        steps: torch.Tensor,
        block_size: List[int],
        shape: torch.Size,
    ):
        super().__init__()
        self.ql = ql
        self.qh = qh
        self.scale = scale
        self.zero_point = zero_point
        self.steps = steps
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
        planes, and re-encode scale/zero to per-row uint8 codes. All of this is
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

        ql, qh = pack_int5(u)
        scale_codes, scale_step = _encode_uint8_per_row(scale)
        zero_codes, zero_step = _encode_uint8_per_row(zero_u)
        steps = torch.cat([scale_step, zero_step], dim=1).contiguous()  # (N, 2)
        return cls(
            ql,
            qh,
            scale_codes,
            zero_codes,
            steps,
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

        Used for the tied lm_head / token embedding (which can't gather a packed
        tensor) and as the numerical reference.
        """
        dtype = output_dtype if output_dtype is not None else self.steps.dtype
        N, K = int(self.shape[0]), int(self.shape[1])
        gs = self.block_size[-1]
        u = unpack_int5(self.ql, self.qh, N, K).to(dtype)
        scale_step = self.steps[:, 0].to(dtype).reshape(N, 1)
        zero_step = self.steps[:, 1].to(dtype).reshape(N, 1)
        scale = (self.scale.to(dtype) * scale_step).repeat_interleave(gs, dim=-1)
        zero = (self.zero_point.to(dtype) * zero_step).repeat_interleave(gs, dim=-1)
        return ((u - zero) * scale).to(dtype)


# Allow a model with CudaDp4aPlanarInt5Tensor weights to be loaded with
# `weights_only=True` (mirrors torchao quantized tensors).
torch.serialization.add_safe_globals([CudaDp4aPlanarInt5Tensor])
