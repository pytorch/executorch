#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Export-time GGUF quantized weights.

``ExportableGGUFTensor`` wraps the *raw* GGUF block bytes for one tensor and
defers all unpacking, serving as the canonical GGUF loading representation:

* ``iter_gguf(path)`` -> streams ``(name, ExportableGGUFTensor | Tensor)`` pairs
  (quantized tensors become subclasses; F32/F16 stay plain). No unpacking at load.
* As a weight, a k-quant tensor dequantizes via the ``torchao::dequantize_gguf``
  custom op (gguf-package eager body) then a plain ``linear`` / ``embedding`` --
  a backend can pattern-match ``dequantize_gguf`` -> linear/embedding to fuse.
* ``.to_exportable_int4_tensor()`` / ``.to_intx_unpacked_to_int8_tensor()`` /
  ``.to_exportable_nvfp4_tensor()`` convert into ExecuTorch/torchao subclasses
  (``ExportableInt4Tensor`` / ``IntxUnpackedToInt8Tensor`` /
  ``ExportableNVFP4Tensor``) instead.

The quant type is a string (``"q4_k"`` / ``"q5_k"`` / ``"q6_k"`` / ``"nvfp4"``);
the ``gguf`` package's integer ``GGMLQuantizationType`` ids are an internal
lookup detail. Which tensors to convert is the caller's policy.

NVFP4 blocks are not self-contained: GGUF stores the per-tensor scale of
``<name>.weight`` in a separate F32 ``<name>.scale`` tensor. An NVFP4
``ExportableGGUFTensor`` is therefore not a complete weight: ``dequantize`` and
the linear/embedding/fusion paths raise, and callers pair ``.scale`` by name
and convert with ``to_exportable_nvfp4_tensor(per_tensor_scale=...)``.
``torchao::dequantize_gguf(raw, "nvfp4")`` decodes the blocks alone.

Attribution: Q4_K / Q5_K / Q6_K / NVFP4 block layouts follow llama.cpp / gguf-py
(``ggml-common.h``), MIT-licensed (Copyright (c) 2023-2024 The ggml authors).
"""

from __future__ import annotations

from typing import Iterator, Optional, Tuple, Union

import numpy as np
import torch
from torch import Tensor
from torchao.prototype.mx_formats.kernels import f4_unpacked_to_f32
from torchao.utils import TorchAOBaseTensor

aten = torch.ops.aten

# GGUF k-quant constants

QK_K = 256  # super-block size for k-quants

Q4_K_GROUP_SIZE = QK_K // 8  # 32  (8 sub-blocks per super-block)
Q5_K_GROUP_SIZE = QK_K // 8  # 32  (8 sub-blocks per super-block, like Q4_K)
Q6_K_GROUP_SIZE = QK_K // 16  # 16 (16 sub-blocks per super-block)

_Q4_K_BLOCK_BYTES = 2 + 2 + 12 + QK_K // 2  # 144
_Q5_K_BLOCK_BYTES = 2 + 2 + 12 + QK_K // 8 + QK_K // 2  # 176
_Q6_K_BLOCK_BYTES = 2 + QK_K // 2 + QK_K // 4 + QK_K // 16  # 210

# GGML NVFP4: four K16 groups per block, four UE4M3 scale bytes then E2M1 codes.
QK_NVFP4 = 64
NVFP4_GROUP_SIZE = 16
_NVFP4_GROUPS = QK_NVFP4 // NVFP4_GROUP_SIZE  # 4
_NVFP4_BLOCK_BYTES = _NVFP4_GROUPS + QK_NVFP4 // 2  # 36

# ``gguf.GGMLQuantizationType`` integer ids.
GGML_F32 = 0
GGML_F16 = 1
GGML_Q8_0 = 8
QK8_0 = 32
_Q8_0_BLOCK_BYTES = 2 + QK8_0  # 34: fp16 scale + 32 int8
GGML_Q4_K = 12
GGML_Q5_K = 13
GGML_Q6_K = 14
GGML_NVFP4 = 40

# String quant-type names are the user-facing identifier (op arg + subclass attr).
# These dicts map names to the internal ids / block sizes.
_GGML_ID_BY_TYPE = {
    "q4_k": GGML_Q4_K,
    "q5_k": GGML_Q5_K,
    "q6_k": GGML_Q6_K,
    "q8_0": GGML_Q8_0,
    "nvfp4": GGML_NVFP4,
}
_TYPE_BY_GGML_ID = {v: k for k, v in _GGML_ID_BY_TYPE.items()}
_BLOCK_BYTES_BY_TYPE = {
    "q4_k": _Q4_K_BLOCK_BYTES,
    "q5_k": _Q5_K_BLOCK_BYTES,
    "q6_k": _Q6_K_BLOCK_BYTES,
    "q8_0": _Q8_0_BLOCK_BYTES,
    "nvfp4": _NVFP4_BLOCK_BYTES,
}

_BLOCK_ELEMENTS_BY_TYPE = {
    "q4_k": QK_K,
    "q5_k": QK_K,
    "q6_k": QK_K,
    "q8_0": QK8_0,
    "nvfp4": QK_NVFP4,
}

_NVFP4_UNPAIRED = (
    "GGUF NVFP4 blocks exclude the per-tensor scale stored in <name>.scale; "
    "convert with to_exportable_nvfp4_tensor(per_tensor_scale=...) first"
)


def _q8_0_fields(raw: Tensor) -> Tuple[Tensor, Tensor]:
    """Q8_0 rows ``(N, row_bytes)`` -> int8 values ``(N, K)`` and fp16 scales
    ``(N, K / 32)``."""
    n, row_bytes = raw.shape
    blocks = raw.reshape(n, row_bytes // _Q8_0_BLOCK_BYTES, _Q8_0_BLOCK_BYTES)
    scale = blocks[..., :2].contiguous().view(torch.float16).reshape(n, -1)
    q = blocks[..., 2:].contiguous().view(torch.int8).reshape(n, -1)
    return q, scale


def _read_f16(raw: Tensor, col_start: int, col_end: int) -> Tensor:
    """Read an fp16 field from per-block bytes, return float32."""
    return raw[:, col_start:col_end].contiguous().view(torch.float16).float()


def _dequantize_gguf(raw: Tensor, ggml_type: str, output_dtype: torch.dtype) -> Tensor:
    """Dequantize a raw GGUF block blob to a float tensor (k-quants via the
    ``gguf`` package, NVFP4 in torch without its per-tensor scale).

    ``raw`` is ``(N, row_bytes)`` uint8; the result is ``(N, K)`` in
    ``output_dtype``.
    """
    if ggml_type == "q8_0":
        q, scale = _q8_0_fields(raw)
        n = q.shape[0]
        values = q.float().reshape(n, -1, QK8_0) * scale.float()[..., None]
        return values.reshape(n, -1).to(output_dtype)

    if ggml_type == "nvfp4":
        n = raw.shape[0]
        _, scale, codes = _nvfp4_fields(raw)
        values = f4_unpacked_to_f32(codes).reshape(n, -1, NVFP4_GROUP_SIZE)
        return (values * scale[..., None]).reshape(n, -1).to(output_dtype)

    import gguf

    if ggml_type not in _GGML_ID_BY_TYPE:
        raise NotImplementedError(f"unsupported GGUF quant type {ggml_type!r}")
    qtype = gguf.GGMLQuantizationType(_GGML_ID_BY_TYPE[ggml_type])
    np_raw = raw.detach().cpu().contiguous().numpy()
    deq = gguf.dequantize(np_raw, qtype)
    return torch.from_numpy(np.ascontiguousarray(deq)).to(
        device=raw.device, dtype=output_dtype
    )


# Fused ops (eager = dequantize + torch op; a backend may lower to kernels)


@torch.library.custom_op("torchao::dequantize_gguf", mutates_args=())
def dequantize_gguf(
    weight: Tensor,
    ggml_type: str,
    output_dtype: torch.dtype = torch.bfloat16,
) -> Tensor:
    """Dequantize a raw GGUF block blob (``(N, row_bytes)`` uint8) to ``(N, K)``.

    NVFP4 output excludes the per-tensor scale (the GGUF ``<name>.scale``).
    """
    return _dequantize_gguf(weight, ggml_type, output_dtype)


@dequantize_gguf.register_fake
def _(weight, ggml_type, output_dtype=torch.bfloat16):
    K = (weight.shape[1] // _BLOCK_BYTES_BY_TYPE[ggml_type]) * _BLOCK_ELEMENTS_BY_TYPE[
        ggml_type
    ]
    return torch.empty((weight.shape[0], K), dtype=output_dtype, device=weight.device)


# Per-type field extraction (used by the to_*_tensor conversions)


def _nvfp4_fields(raw: Tensor) -> Tuple[Tensor, Tensor, Tensor]:
    """Decode NVFP4 rows ``(N, row_bytes)``.

    Returns ``(e4m3, scale, codes)``: the ``(N, K // 16)`` scale bytes as
    ``float8_e4m3fn`` bits and as the float32 values GGML decodes, and the
    ``(N, K)`` uint8 E2M1 codes in element order (GGML packs eight lows then
    eight highs per K16 group) with code 8 (-0) canonicalized to +0. GGML UE4M3
    ignores the sign bit and treats 0x7f as zero, but checks 0x7f before
    masking, so 0xff decodes to 480: E4M3 has no such value, and that byte is
    the only one whose two scale forms differ.
    """
    n, row_bytes = raw.shape
    blocks = raw.reshape(n, row_bytes // _NVFP4_BLOCK_BYTES, _NVFP4_BLOCK_BYTES)
    raw_scale = blocks[..., :_NVFP4_GROUPS].reshape(n, -1)
    e4m3 = torch.where(raw_scale == 0x7F, 0, raw_scale & 0x7F).to(torch.uint8)
    scale = torch.where(
        raw_scale == 0xFF, 480.0, e4m3.view(torch.float8_e4m3fn).float()
    )
    packed = blocks[..., _NVFP4_GROUPS:].reshape(n, -1, NVFP4_GROUP_SIZE // 2)
    codes = torch.cat((packed & 15, packed >> 4), dim=-1).reshape(n, -1)
    return e4m3, scale, torch.where(codes == 8, 0, codes).to(torch.uint8)


def _q4_k_fields(raw: Tensor, N: int, K: int) -> Tuple[Tensor, Tensor, Tensor]:
    """Decode Q4_K blocks for conversion to ``Int4Tensor``.

    Returns ``(q, eff_scale, eff_min)`` where ``q`` is ``(N, K)`` uint8 in
    [0, 15], and ``eff_scale`` / ``eff_min`` are ``(N, K // 32)`` float32.
    """
    n_blocks = N * (K // QK_K)
    blk = raw.reshape(n_blocks, _Q4_K_BLOCK_BYTES)

    d = _read_f16(blk, 0, 2)
    dmin = _read_f16(blk, 2, 4)
    s = blk[:, 4:16]
    qs = blk[:, 16:144]

    sc = torch.empty(n_blocks, 8, dtype=torch.float32)
    mn = torch.empty(n_blocks, 8, dtype=torch.float32)
    sc[:, :4] = (s[:, :4] & 0x3F).float()
    mn[:, :4] = (s[:, 4:8] & 0x3F).float()
    sc[:, 4:] = ((s[:, 8:12] & 0xF) | ((s[:, :4] >> 6) << 4)).float()
    mn[:, 4:] = ((s[:, 8:12] >> 4) | ((s[:, 4:8] >> 6) << 4)).float()

    eff_scale = (d * sc).reshape(N, -1)
    eff_min = (dmin * mn).reshape(N, -1)

    # GGUF Q4_K nibble order: 32 lows then 32 highs per sub-block pair.
    low = (qs & 0x0F).to(torch.uint8)
    high = ((qs >> 4) & 0x0F).to(torch.uint8)
    q = torch.cat(
        [
            low[:, :32],
            high[:, :32],
            low[:, 32:64],
            high[:, 32:64],
            low[:, 64:96],
            high[:, 64:96],
            low[:, 96:128],
            high[:, 96:128],
        ],
        dim=-1,
    ).reshape(N, K)
    return q, eff_scale, eff_min


def _q5_k_fields(raw: Tensor, N: int, K: int) -> Tuple[Tensor, Tensor, Tensor]:
    """Decode Q5_K blocks for conversion to a genuine 5-bit tensor.

    Q5_K is Q4_K plus a 1-bit high plane: same fp16 ``d`` / ``dmin`` super-scales
    and the same 6-bit packed sub-block scales/mins, but each weight is 5 bits
    (a low nibble in ``qs`` plus one high bit in ``qh``). Returns
    ``(q, eff_scale, eff_min)`` where ``q`` is ``(N, K)`` uint8 in [0, 31] and
    ``eff_scale`` / ``eff_min`` are ``(N, K // 32)`` float32, so
    ``w = eff_scale * q - eff_min`` (asymmetric, exactly like the Q4_K affine
    form). Verified bit-exact against ``gguf.dequantize``.
    """
    n_blocks = N * (K // QK_K)
    blk = raw.reshape(n_blocks, _Q5_K_BLOCK_BYTES)

    d = _read_f16(blk, 0, 2)
    dmin = _read_f16(blk, 2, 4)
    s = blk[:, 4:16]  # 12B packed 6-bit sub-block scales/mins (same as Q4_K)
    qh = blk[:, 16:48]  # 32B high-bit plane (one bit per weight)
    qs = blk[:, 48:176]  # 128B low-nibble plane (same layout as Q4_K qs)

    sc = torch.empty(n_blocks, 8, dtype=torch.float32)
    mn = torch.empty(n_blocks, 8, dtype=torch.float32)
    sc[:, :4] = (s[:, :4] & 0x3F).float()
    mn[:, :4] = (s[:, 4:8] & 0x3F).float()
    sc[:, 4:] = ((s[:, 8:12] & 0xF) | ((s[:, :4] >> 6) << 4)).float()
    mn[:, 4:] = ((s[:, 8:12] >> 4) | ((s[:, 4:8] >> 6) << 4)).float()

    eff_scale = (d * sc).reshape(N, -1)
    eff_min = (dmin * mn).reshape(N, -1)

    # Low 4 bits: same nibble order as Q4_K (32 lows then 32 highs per pair).
    low = (qs & 0x0F).to(torch.uint8)
    high = ((qs >> 4) & 0x0F).to(torch.uint8)
    q4 = torch.cat(
        [
            low[:, :32],
            high[:, :32],
            low[:, 32:64],
            high[:, 32:64],
            low[:, 64:96],
            high[:, 64:96],
            low[:, 96:128],
            high[:, 96:128],
        ],
        dim=-1,
    )  # (n_blocks, QK_K) in [0, 15], weight order matches Q4_K
    # High bit: sub-block g (32 consecutive weights) reads bit g of each qh byte.
    hb = (
        torch.stack([(qh >> g) & 1 for g in range(8)], dim=1)
        .reshape(n_blocks, QK_K)
        .to(torch.uint8)
    )
    q = (q4 | (hb << 4)).reshape(N, K)  # [0, 31]
    return q, eff_scale, eff_min


def _q6_k_fields(raw: Tensor, N: int, K: int) -> Tuple[Tensor, Tensor]:
    """Decode Q6_K blocks for conversion to ``IntxUnpackedToInt8Tensor``.

    Returns ``(q, eff_scale)`` where ``q`` is ``(N, K)`` int8 in [-32, 31] and
    ``eff_scale`` is ``(N, K // 16)`` float32.
    """
    n_blocks = N * (K // QK_K)
    blk = raw.reshape(n_blocks, _Q6_K_BLOCK_BYTES)

    ql = blk[:, 0:128]
    qh = blk[:, 128:192]
    sc = blk[:, 192:208]
    d = _read_f16(blk, 208, 210)

    qh0 = qh[:, :32]
    qh1 = qh[:, 32:64]
    q = torch.empty(n_blocks, QK_K, dtype=torch.int16)
    q[:, 0:32] = (ql[:, :32] & 0x0F) | ((qh0 & 0x03) << 4)
    q[:, 32:64] = (ql[:, 32:64] & 0x0F) | (((qh0 >> 2) & 0x03) << 4)
    q[:, 64:96] = ((ql[:, :32] >> 4) & 0x0F) | (((qh0 >> 4) & 0x03) << 4)
    q[:, 96:128] = ((ql[:, 32:64] >> 4) & 0x0F) | (((qh0 >> 6) & 0x03) << 4)
    q[:, 128:160] = (ql[:, 64:96] & 0x0F) | ((qh1 & 0x03) << 4)
    q[:, 160:192] = (ql[:, 96:128] & 0x0F) | (((qh1 >> 2) & 0x03) << 4)
    q[:, 192:224] = ((ql[:, 64:96] >> 4) & 0x0F) | (((qh1 >> 4) & 0x03) << 4)
    q[:, 224:256] = ((ql[:, 96:128] >> 4) & 0x0F) | (((qh1 >> 6) & 0x03) << 4)
    q -= 32

    # ``sc`` bytes are signed int8 sub-block scales.
    eff_scale = (d * sc.to(torch.int8).float()).reshape(N, -1)
    return q.reshape(N, K).to(torch.int8), eff_scale


def _max_uniform_group_size(
    eff_scale: Tensor,
    eff_min: Tensor,
    base_group_size: int,
    max_group_size: int,
) -> Tuple[int, Tensor, Tensor]:
    """Merge adjacent identical sub-block groups into the largest lossless group.

    ``eff_scale`` / ``eff_min`` are ``(N, K // base_group_size)`` per-sub-block
    affine params. Returns ``(group_size, eff_scale, eff_min)`` where
    ``group_size`` is the largest power-of-two multiple of ``base_group_size``
    that is ``<= max_group_size``, divides the per-row group count, and for which
    every run of ``group_size // base_group_size`` consecutive sub-blocks has an
    identical scale *and* min across all rows. Because a run is merged only when
    both params match exactly, ``scale * (q - zero_point)`` is bit-identical, so
    the upgrade is lossless. The returned params are subsampled to
    ``(N, K // group_size)``; when no merge applies the inputs are returned
    unchanged at ``base_group_size``.

    Merging is data-dependent (it inspects the actual scale/min values), so under
    fake tensors -- e.g. the partitioner's support check, which runs on a fake
    program -- the values are unknown and the base group size is kept. The real
    merge happens later during preprocess, when the weight holds real data.
    """
    from torch._subclasses.fake_tensor import FakeTensor

    if isinstance(eff_scale, FakeTensor):
        return base_group_size, eff_scale, eff_min

    n_rows, n_groups = eff_scale.shape

    # Largest power-of-two factor that fits under max_group_size and divides the
    # group count evenly.
    factor = 1
    while (
        base_group_size * factor * 2 <= max_group_size and n_groups % (factor * 2) == 0
    ):
        factor *= 2

    # Step down to the largest factor whose runs are actually uniform (a larger
    # uniform run implies its sub-runs are too, so the first match is the max).
    while factor > 1:
        s = eff_scale.reshape(n_rows, n_groups // factor, factor)
        m = eff_min.reshape(n_rows, n_groups // factor, factor)
        if bool((s == s[:, :, :1]).all()) and bool((m == m[:, :, :1]).all()):
            return (
                base_group_size * factor,
                s[:, :, 0].contiguous(),
                m[:, :, 0].contiguous(),
            )
        factor //= 2

    return base_group_size, eff_scale, eff_min


# Tensor subclass


class ExportableGGUFTensor(TorchAOBaseTensor):
    """Wraps the raw GGUF block bytes for one quantized weight.

    Stores the exact GGUF block byte layout (no repacking) plus the quant type
    string (``"q4_k"`` / ``"q5_k"`` / ``"q6_k"`` / ``"nvfp4"``). ``aten.linear`` /
    ``aten.embedding`` dequantize via the ``torchao::dequantize_gguf`` op (then a
    plain linear/embedding); :meth:`to_exportable_int4_tensor` /
    :meth:`to_intx_unpacked_to_int8_tensor` / :meth:`to_exportable_nvfp4_tensor`
    convert to ExecuTorch/torchao subclasses instead. An NVFP4 tensor holds only
    the blocks; its per-tensor scale is the separate GGUF ``.scale`` tensor, so
    it must be converted with :meth:`to_exportable_nvfp4_tensor` before use.
    """

    tensor_data_names = ["raw"]
    tensor_attribute_names = ["ggml_type", "orig_dtype"]

    def __new__(cls, raw: Tensor, ggml_type: str, orig_dtype: torch.dtype):
        if raw.dim() != 2 or raw.dtype != torch.uint8:
            raise ValueError(
                f"ExportableGGUFTensor: raw must be 2-D uint8 (N, row_bytes); got "
                f"shape {tuple(raw.shape)} dtype {raw.dtype}"
            )
        if ggml_type not in _BLOCK_BYTES_BY_TYPE:
            raise NotImplementedError(
                f"ExportableGGUFTensor: unsupported quant type {ggml_type!r}; "
                f"supported: {sorted(_BLOCK_BYTES_BY_TYPE)}"
            )
        n, row_bytes = int(raw.shape[0]), int(raw.shape[1])
        block_bytes = _BLOCK_BYTES_BY_TYPE[ggml_type]
        if row_bytes % block_bytes != 0:
            raise ValueError(
                f"ExportableGGUFTensor: row bytes {row_bytes} not a multiple of "
                f"block bytes {block_bytes} for quant type {ggml_type!r}"
            )
        K = (row_bytes // block_bytes) * _BLOCK_ELEMENTS_BY_TYPE[ggml_type]
        self = torch.Tensor._make_wrapper_subclass(
            cls, (n, K), dtype=orig_dtype, device=raw.device, requires_grad=False
        )
        self.raw = raw
        self.ggml_type = ggml_type
        self.orig_dtype = orig_dtype
        return self

    @classmethod
    def from_raw(
        cls,
        raw: Tensor,
        ggml_type: str,
        orig_dtype: torch.dtype = torch.bfloat16,
    ) -> "ExportableGGUFTensor":
        """Build from a ``(N, row_bytes)`` uint8 GGUF block blob."""
        return cls(raw.contiguous(), ggml_type, orig_dtype)

    def to(self, *args, **kwargs) -> "ExportableGGUFTensor":
        """Move device and/or set the output dtype *without* dequantizing.

        Mirrors ``IntxUnpackedToInt8Tensor.to``: the raw GGUF blocks are only
        moved across devices; ``dtype`` becomes the dtype :meth:`dequantize`
        will produce. Casting via ``.to`` therefore preserves the quantized
        payload (unlike ``aten._to_copy``, which dequantizes for export).
        """
        kwargs = self._get_to_kwargs(*args, **kwargs)
        device = kwargs.pop("device")
        dtype = kwargs.pop("dtype")
        assert dtype.is_floating_point, f"expected a floating dtype; got {dtype}"
        return ExportableGGUFTensor(self.raw.to(device), self.ggml_type, dtype)

    def dequantize(self, output_dtype: Optional[torch.dtype] = None) -> Tensor:
        """Dequantize to a plain float tensor; NVFP4 raises (see module doc)."""
        if self.ggml_type == "nvfp4":
            raise ValueError(_NVFP4_UNPAIRED)
        return torch.ops.torchao.dequantize_gguf(
            self.raw, self.ggml_type, output_dtype or self.orig_dtype
        )

    def to_exportable_nvfp4_tensor(
        self,
        *,
        per_tensor_scale: Union[Tensor, float],
        output_dtype: Optional[torch.dtype] = None,
        act_per_tensor_scale: Union[Tensor, float, None] = None,
    ) -> Tensor:
        """Losslessly repack GGML NVFP4 blocks to the portable torchao layout.

        ``per_tensor_scale`` is the scalar from the GGUF ``<name>.scale``
        companion and ``act_per_tensor_scale`` the optional scalar
        ``<name>.input_scale`` (stored as a Python float, see
        ``ExportableNVFP4Tensor``); the caller pairs them by name. GGUF stores
        both as F32, so tensor scales must still be float32: pair them before
        any dtype cast (e.g. a loader's ``dtype=``), which would round them.

        GGML packs eight lows followed by eight highs in each K16 group;
        torchao packs the even element in the low nibble. Scales E4M3 cannot
        represent (see ``_nvfp4_fields``) are rejected.
        """
        from executorch.extension.llm.export.nvfp4 import ExportableNVFP4Tensor
        from torch._subclasses.fake_tensor import is_fake
        from torchao.prototype.mx_formats.kernels import pack_uint4

        if self.ggml_type != "nvfp4":
            raise NotImplementedError(
                f"to_exportable_nvfp4_tensor only supports nvfp4; got {self.ggml_type!r}"
            )
        e4m3, scale, codes = _nvfp4_fields(self.raw)
        if not is_fake(e4m3) and not torch.equal(
            e4m3.view(torch.float8_e4m3fn).float(), scale
        ):
            raise ValueError(
                "GGML NVFP4 scale 0xff decodes to 480, which cannot be represented losslessly as E4M3"
            )
        scales = (per_tensor_scale, act_per_tensor_scale)
        if any(isinstance(s, Tensor) and s.dtype != torch.float32 for s in scales):
            raise ValueError(
                "NVFP4 per-tensor scales must be float32 (GGUF F32); a cast "
                "before pairing rounds them"
            )
        pts = torch.as_tensor(
            per_tensor_scale, dtype=torch.float32, device=self.raw.device
        )
        if pts.numel() != 1:
            raise ValueError("NVFP4 per_tensor_scale must be a scalar")
        return ExportableNVFP4Tensor(
            pack_uint4(codes).contiguous().view(torch.uint32),
            e4m3,
            pts.reshape(()),
            NVFP4_GROUP_SIZE,
            output_dtype or self.orig_dtype,
            act_per_tensor_scale,
        )

    def to_exportable_int4_tensor(
        self, output_dtype: Optional[torch.dtype] = None
    ) -> Tensor:
        """Convert a Q4_K tensor to an ``ExportableInt4Tensor``.

        Threads ``output_dtype`` (default ``orig_dtype``) into the scale/zero_point
        and the dequantized output dtype, so callers targeting fp16 (e.g. MLX) get
        fp16 params while CUDA can pass bf16. The packed int nibbles are unchanged.
        """
        from executorch.extension.llm.export.int4 import ExportableInt4Tensor

        if self.ggml_type != "q4_k":
            raise NotImplementedError(
                f"to_exportable_int4_tensor only supports q4_k; got {self.ggml_type!r}"
            )
        dtype = output_dtype or self.orig_dtype
        N, K = int(self.shape[0]), int(self.shape[1])
        q, eff_scale, eff_min = _q4_k_fields(self.raw, N, K)

        zero = torch.where(
            eff_scale != 0, eff_min / eff_scale, torch.zeros_like(eff_min)
        )
        # Nibble-pack: even index -> low nibble, odd -> high.
        packed = q[:, ::2] | (q[:, 1::2] << 4)
        return ExportableInt4Tensor(
            packed,
            # ExportableInt4Tensor scale/zero layout is (K // gs, N) -- transposed.
            eff_scale.to(dtype).t().contiguous(),
            zero.to(dtype).t().contiguous(),
            Q4_K_GROUP_SIZE,
            dtype,
        )

    def to_intx_unpacked_to_int8_tensor(
        self,
        max_group_size: Optional[int] = None,
        scale_dtype: Optional[torch.dtype] = None,
    ) -> Tensor:
        """Convert to a torchao ``IntxUnpackedToInt8Tensor`` (Q4_K, Q5_K or Q6_K).

        Q6_K maps to a symmetric int8 tensor (values [-32, 31], zero-point 0).
        Q4_K maps to a 4-bit tensor: values are centered to [-8, 7] and the
        affine min is folded into a (float) zero-point, so the rewrite is exact.
        Q5_K maps to a 5-bit tensor: values are centered to [-16, 15] and the
        affine min folded into the zero-point, exactly like Q4_K.

        ``max_group_size``: when set, adjacent sub-blocks whose scale *and* min
        are identical are merged into the largest lossless group size
        ``<= max_group_size`` (see :func:`_max_uniform_group_size`). Q6_K is
        symmetric (no min), so its merge decision is scale-equality only.
        Defaults to ``None`` (keep the native groups: 32 for Q4_K/Q5_K, 16 for
        Q6_K).
        """
        from torchao.quantization import IntxUnpackedToInt8Tensor

        # Scale/zero-point dtype. GGUF stores the super-block scale as float16,
        # so float16 is the faithful default. Callers pass their own compute
        # dtype: the MLX repack passes the activation dtype so quantized_matmul
        # scales match activations and MLX does not promote (e.g. bf16 + f16 ->
        # f32); the CUDA integrations pass bfloat16.
        sdt = scale_dtype if scale_dtype is not None else torch.float16

        N, K = int(self.shape[0]), int(self.shape[1])
        if self.ggml_type == "q8_0":
            q, scale = _q8_0_fields(self.raw)
            return IntxUnpackedToInt8Tensor(
                qdata=q,
                scale=scale.to(sdt),
                zero_point=torch.zeros_like(scale, dtype=torch.int8),
                target_dtype=torch.int8,
                block_size=(1, QK8_0),
                dtype=sdt,
                activation_quantization=None,
            )
        if self.ggml_type == "q6_k":
            q, eff_scale = _q6_k_fields(self.raw, N, K)
            group_size = Q6_K_GROUP_SIZE
            if max_group_size is not None:
                # Symmetric (zero-point 0, no min): an all-zeros min always
                # matches, so the merge reduces to adjacent scale equality.
                group_size, eff_scale, _ = _max_uniform_group_size(
                    eff_scale,
                    torch.zeros_like(eff_scale),
                    Q6_K_GROUP_SIZE,
                    max_group_size,
                )
            return IntxUnpackedToInt8Tensor(
                qdata=q,
                scale=eff_scale.to(sdt),
                zero_point=torch.zeros_like(eff_scale, dtype=torch.int8),
                target_dtype=torch.int8,
                block_size=(1, group_size),
                dtype=sdt,
                activation_quantization=None,
            )
        if self.ggml_type == "q5_k":
            q, eff_scale, eff_min = _q5_k_fields(self.raw, N, K)
            group_size = Q5_K_GROUP_SIZE
            if max_group_size is not None:
                group_size, eff_scale, eff_min = _max_uniform_group_size(
                    eff_scale, eff_min, Q5_K_GROUP_SIZE, max_group_size
                )
            zero = torch.where(
                eff_scale != 0, eff_min / eff_scale, torch.zeros_like(eff_min)
            )
            # Center quants [0, 31] -> [-16, 15] and shift the zero-point to
            # match (dequant = scale * (q - zp) is preserved).
            return IntxUnpackedToInt8Tensor(
                qdata=q.to(torch.int8) - 16,
                scale=eff_scale.to(sdt),
                zero_point=(zero - 16).to(sdt),
                target_dtype=torch.int5,
                block_size=(1, group_size),
                dtype=sdt,
                activation_quantization=None,
            )
        if self.ggml_type == "q4_k":
            q, eff_scale, eff_min = _q4_k_fields(self.raw, N, K)
            group_size = Q4_K_GROUP_SIZE
            if max_group_size is not None:
                group_size, eff_scale, eff_min = _max_uniform_group_size(
                    eff_scale, eff_min, Q4_K_GROUP_SIZE, max_group_size
                )
            zero = torch.where(
                eff_scale != 0, eff_min / eff_scale, torch.zeros_like(eff_min)
            )
            # Center quants [0, 15] -> [-8, 7] and shift the zero-point to match
            # (dequant = scale * (q - zp) is preserved).
            return IntxUnpackedToInt8Tensor(
                qdata=q.to(torch.int8) - 8,
                scale=eff_scale.to(sdt),
                zero_point=(zero - 8).to(sdt),
                target_dtype=torch.int4,
                block_size=(1, group_size),
                dtype=sdt,
                activation_quantization=None,
            )
        raise NotImplementedError(
            f"to_intx_unpacked_to_int8_tensor supports q4_k/q5_k/q6_k/q8_0; "
            f"got {self.ggml_type!r}"
        )

    __torch_function__ = torch._C._disabled_torch_function_impl


implements = ExportableGGUFTensor.implements


@implements([aten.linear.default])
def _(func, types, args, kwargs):
    input_tensor, weight = args[0], args[1]
    bias = args[2] if len(args) > 2 else None
    return torch.nn.functional.linear(
        input_tensor, weight.dequantize(input_tensor.dtype), bias
    )


@implements([aten.embedding.default])
def _(func, types, args, kwargs):
    weight, indices = args[0], args[1]
    return torch.nn.functional.embedding(indices, weight.dequantize())


@implements([aten.t.default])
def _(func, types, args, kwargs):
    return args[0].dequantize().t()


@implements([aten.detach.default, aten.alias.default])
def _(func, types, args, kwargs):
    return args[0]


@implements([aten._to_copy.default])
def _(func, types, args, kwargs):
    return args[0].dequantize(output_dtype=kwargs.get("dtype", args[0].orig_dtype))


# Loader


def iter_gguf(
    path: str,
) -> Iterator[Tuple[str, Union[ExportableGGUFTensor, Tensor]]]:
    """Stream ``(name, value)`` for every tensor in a GGUF file (low peak mem).

    Quantized tensors (Q4_K, Q5_K, Q6_K, Q8_0, NVFP4) are wrapped as
    ``ExportableGGUFTensor`` with the raw block bytes; F32/F16/BF16 are returned
    as plain float tensors in their native dtype. GGUF shapes are reversed to
    PyTorch ``(N, K)`` convention. The per-tensor scale of an NVFP4
    ``<name>.weight`` is yielded as its own ``<name>.scale`` F32 tensor; callers
    pair it by name before any dtype cast (see
    :meth:`ExportableGGUFTensor.to_exportable_nvfp4_tensor`).
    """
    from gguf import GGMLQuantizationType, GGUFReader

    reader = GGUFReader(path)
    for tensor in reader.tensors:
        shape = list(reversed(tensor.shape.tolist()))
        ttype = int(tensor.tensor_type)
        flat = torch.frombuffer(memoryview(tensor.data), dtype=torch.uint8)
        if ttype in _TYPE_BY_GGML_ID:
            N = shape[0]
            row_bytes = flat.numel() // N
            raw = flat.reshape(N, row_bytes).clone()
            yield (
                tensor.name,
                ExportableGGUFTensor.from_raw(raw, _TYPE_BY_GGML_ID[ttype]),
            )
        elif tensor.tensor_type == GGMLQuantizationType.F32:
            yield tensor.name, flat.view(torch.float32).reshape(shape).clone()
        elif tensor.tensor_type == GGMLQuantizationType.F16:
            yield tensor.name, flat.view(torch.float16).reshape(shape).clone()
        elif tensor.tensor_type == GGMLQuantizationType.BF16:
            yield tensor.name, flat.view(torch.bfloat16).reshape(shape).clone()
        else:
            raise ValueError(f"Unsupported GGUF quant type: {tensor.tensor_type}")
