# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""ExecuTorch-internal packed-INT6 tensor for the CUDA W6A8 dp4a decode kernel.

``CudaPackedInt6Tensor`` is an ExecuTorch-internal tensor subclass that stores a
genuine 6-bit packed weight (0.75 B/elem), used for GGUF Q6_K weights. Unlike
the int8 path (``IntxUnpackedToInt8Tensor``, one int8 per 6-bit value), this
format wastes no bits and carries no zero tensor — Q6_K is symmetric.

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
    scale : (N, K/gs) bf16 — per-group scales, row-major (already coalesced; the
            decode kernel reads it row-for-row, no transpose).

The pack/unpack helpers (:func:`pack_int6`, :func:`unpack_int6`) must stay in
lockstep with ``int6_plain_mm.cuh`` (the decode kernel) — the per-32-weight
``hi_even``/``hi_odd`` byte order is the single most error-prone detail and is
covered by the pack round-trip and the C++ gtest.
"""

from typing import List, Optional, Tuple

import torch
from torchao.utils import TorchAOBaseTensor

__all__ = [
    "CudaPackedInt6Tensor",
    "pack_int6",
    "unpack_int6",
]


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


class CudaPackedInt6Tensor(TorchAOBaseTensor):
    """Packed 6-bit weight (ql/qh planes + per-group scale), symmetric.

    ExecuTorch-internal; see the module docstring. The CUDA decode/prefill
    dispatch (``int6_dispatch.py``) is selected by *type* — it is registered on
    this class only.
    """

    tensor_data_names = ["ql", "qh", "scale"]
    tensor_attribute_names = ["block_size", "shape"]

    def __new__(
        cls,
        ql: torch.Tensor,
        qh: torch.Tensor,
        scale: torch.Tensor,
        block_size: List[int],
        shape: torch.Size,
    ):
        kwargs = {}
        kwargs["device"] = ql.device
        kwargs["dtype"] = scale.dtype
        kwargs["requires_grad"] = False
        return torch.Tensor._make_wrapper_subclass(cls, shape, **kwargs)  # type: ignore[attr-defined]

    def __init__(
        self,
        ql: torch.Tensor,
        qh: torch.Tensor,
        scale: torch.Tensor,
        block_size: List[int],
        shape: torch.Size,
    ):
        super().__init__()
        self.ql = ql
        self.qh = qh
        self.scale = scale
        self.block_size = block_size

    def _quantization_type(self):
        return (
            f"shape={self.shape}, block_size={self.block_size}, "
            f"device={self.device}"
        )

    @classmethod
    def from_intx_int8(cls, t: torch.Tensor) -> "CudaPackedInt6Tensor":
        """Build from a torchao ``IntxUnpackedToInt8Tensor`` decoded from Q6_K.

        The source is symmetric (zero_point == 0), ``qdata`` is int8 in
        ``[-32, 31]`` and ``scale`` is ``(N, K/16)``. The ql/qh bit-pack is baked
        into the serialized weight constant here, once at pack time.
        """
        q = t.qdata
        if not bool(torch.all(t.zero_point == 0)):
            raise ValueError(
                "CudaPackedInt6Tensor.from_intx_int8 requires symmetric Q6_K "
                "weights (zero_point == 0)"
            )
        q_min, q_max = int(q.min()), int(q.max())
        if q_min < -32 or q_max > 31:
            raise ValueError(
                f"Q6_K values must be in [-32, 31], got [{q_min}, {q_max}]"
            )
        ql, qh = pack_int6(q)
        return cls(
            ql,
            qh,
            t.scale.contiguous(),
            list(t.block_size),
            t.shape,
        )

    def dequantize(self, output_dtype: Optional[torch.dtype] = None) -> torch.Tensor:
        """Dequantize to a dense tensor (symmetric: ``w = q * scale``).

        Used for the tied lm_head / token embedding (which can't gather a packed
        tensor) and as the numerical reference.
        """
        dtype = output_dtype if output_dtype is not None else self.scale.dtype
        N, K = int(self.shape[0]), int(self.shape[1])
        gs = self.block_size[-1]
        q = unpack_int6(self.ql, self.qh, N, K).to(dtype)
        scale = self.scale.to(dtype).repeat_interleave(gs, dim=-1)
        return (q * scale).to(dtype)


# Allow a model with CudaPackedInt6Tensor weights to be loaded with
# `weights_only=True` (mirrors torchao quantized tensors).
torch.serialization.add_safe_globals([CudaPackedInt6Tensor])
