#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Export-time GGUF quantized weights.

``ExportableGGUFTensor`` is the canonical **loading representation** for a
GGUF-quantized weight: it wraps the *raw* GGUF block bytes for one tensor and
defers all unpacking. The intended flow:

1. **Load**: ``load_gguf(path)`` -> ``dict[name -> ExportableGGUFTensor | Tensor]``
   (quantized tensors become ``ExportableGGUFTensor``; F32/F16 become plain
   tensors). No unpacking happens at load.
2. **Lower (dequantize)**: used as a weight, the subclass dequantizes via the
   ``torchao::gguf_dequantize`` custom op (gguf-package eager body) and runs the
   plain torch ``linear`` / ``embedding`` (NVFP4-style). A backend can
   pattern-match ``gguf_dequantize`` -> linear/embedding to fuse.
3. **Convert**: ``.to_int4_tensor()`` / ``.to_intx_unpacked_to_int8_tensor()``
   unpack into torchao tensor subclasses (``Int4Tensor`` for Q4_K,
   ``IntxUnpackedToInt8Tensor`` for Q4_K or Q6_K) to take the non-fused
   (affine-dequant) path instead.

The GGUF quant type is identified by a **string** (``"q4_k"``, ``"q6_k"``)
everywhere user-facing (subclass attribute + ``gguf_dequantize`` op argument); the
``gguf`` package's integer ``GGMLQuantizationType`` ids are an internal lookup
detail.

Backend-agnostic; depends on ``torch``, ``torchao``, ``numpy``, and the ``gguf``
package. The *policy* of which tensors to convert is left to the caller.

Attribution: the Q4_K / Q6_K block layouts follow llama.cpp / gguf-py
(``ggml-common.h``), MIT-licensed (Copyright (c) 2023-2024 The ggml authors).
"""

from __future__ import annotations

from typing import Dict, Iterator, Optional, Tuple, Union

import numpy as np
import torch
from torch import Tensor
from torchao.utils import TorchAOBaseTensor

aten = torch.ops.aten

# ---------------------------------------------------------------------------
# GGUF k-quant constants
# ---------------------------------------------------------------------------

QK_K = 256  # super-block size for k-quants

Q4_K_GROUP_SIZE = QK_K // 8  # 32  (8 sub-blocks per super-block)
Q6_K_GROUP_SIZE = QK_K // 16  # 16 (16 sub-blocks per super-block)

_Q4_K_BLOCK_BYTES = 2 + 2 + 12 + QK_K // 2  # 144
_Q6_K_BLOCK_BYTES = 2 + QK_K // 2 + QK_K // 4 + QK_K // 16  # 210

# ``gguf.GGMLQuantizationType`` integer ids.
GGML_F32 = 0
GGML_F16 = 1
GGML_Q4_K = 12
GGML_Q6_K = 14

# String quant-type names are the user-facing identifier (op arg + subclass attr).
# These dicts map names to the internal ids / block sizes.
_GGML_ID_BY_TYPE = {"q4_k": GGML_Q4_K, "q6_k": GGML_Q6_K}
_TYPE_BY_GGML_ID = {v: k for k, v in _GGML_ID_BY_TYPE.items()}
_BLOCK_BYTES_BY_TYPE = {"q4_k": _Q4_K_BLOCK_BYTES, "q6_k": _Q6_K_BLOCK_BYTES}


def _read_f16(raw: Tensor, col_start: int, col_end: int) -> Tensor:
    """Read an fp16 field from per-block bytes, return float32."""
    return raw[:, col_start:col_end].contiguous().view(torch.float16).float()


def _gguf_dequantize(raw: Tensor, ggml_type: str, output_dtype: torch.dtype) -> Tensor:
    """Dequantize a raw GGUF block blob to a float tensor via the ``gguf`` package.

    ``raw`` is ``(N, row_bytes)`` uint8; the result is ``(N, K)`` in
    ``output_dtype``.
    """
    import gguf

    if ggml_type not in _GGML_ID_BY_TYPE:
        raise NotImplementedError(f"unsupported GGUF quant type {ggml_type!r}")
    qtype = gguf.GGMLQuantizationType(_GGML_ID_BY_TYPE[ggml_type])
    np_raw = raw.detach().cpu().contiguous().numpy()
    deq = gguf.dequantize(np_raw, qtype)
    return torch.from_numpy(np.ascontiguousarray(deq)).to(
        device=raw.device, dtype=output_dtype
    )


# ---------------------------------------------------------------------------
# Fused ops (eager = gguf.dequantize + torch op; a backend may lower to kernels)
# ---------------------------------------------------------------------------


@torch.library.custom_op("torchao::gguf_dequantize", mutates_args=())
def gguf_dequantize(
    weight: Tensor,
    ggml_type: str,
    output_dtype: torch.dtype = torch.bfloat16,
) -> Tensor:
    """Dequantize a raw GGUF block blob (``(N, row_bytes)`` uint8) to ``(N, K)``."""
    return _gguf_dequantize(weight, ggml_type, output_dtype)


@gguf_dequantize.register_fake
def _(weight, ggml_type, output_dtype=torch.bfloat16):
    K = (weight.shape[1] // _BLOCK_BYTES_BY_TYPE[ggml_type]) * QK_K
    return torch.empty((weight.shape[0], K), dtype=output_dtype, device=weight.device)


# ---------------------------------------------------------------------------
# Per-type field extraction (used by the to_*_tensor conversions)
# ---------------------------------------------------------------------------


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


# ---------------------------------------------------------------------------
# Tensor subclass
# ---------------------------------------------------------------------------


class ExportableGGUFTensor(TorchAOBaseTensor):
    """Wraps the raw GGUF block bytes for one quantized weight.

    Stores the exact GGUF ``block_q*_K`` byte layout (no repacking) plus the
    quant type string (``"q4_k"`` / ``"q6_k"``). ``aten.linear`` / ``aten.embedding``
    dequantize via the ``torchao::gguf_dequantize`` op (then a plain
    linear/embedding); :meth:`to_int4_tensor` / :meth:`to_intx_unpacked_to_int8_tensor`
    convert to torchao subclasses instead.
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
        K = (row_bytes // block_bytes) * QK_K
        self = torch.Tensor._make_wrapper_subclass(
            cls, (n, K), dtype=orig_dtype, device=raw.device, requires_grad=False
        )
        self.raw = raw
        self.ggml_type = ggml_type
        self.orig_dtype = orig_dtype
        return self

    # -- construction --------------------------------------------------------

    @classmethod
    def from_raw(
        cls,
        raw: Tensor,
        ggml_type: str,
        orig_dtype: torch.dtype = torch.bfloat16,
    ) -> "ExportableGGUFTensor":
        """Build from a ``(N, row_bytes)`` uint8 GGUF block blob."""
        return cls(raw.contiguous(), ggml_type, orig_dtype)

    # -- dequant (via gguf package) ------------------------------------------

    def dequantize(self, output_dtype: Optional[torch.dtype] = None) -> Tensor:
        """Dequantize to a plain float tensor using the ``gguf`` package."""
        return torch.ops.torchao.gguf_dequantize(
            self.raw, self.ggml_type, output_dtype or self.orig_dtype
        )

    # -- conversions (unpack lives here) -------------------------------------

    def to_int4_tensor(self) -> Tensor:
        """Convert a Q4_K tensor to a torchao ``Int4Tensor``."""
        from torchao.quantization.quantize_.workflows.int4.int4_tensor import (
            Int4Tensor,
        )

        if self.ggml_type != "q4_k":
            raise NotImplementedError(
                f"to_int4_tensor only supports q4_k; got {self.ggml_type!r}"
            )
        N, K = int(self.shape[0]), int(self.shape[1])
        q, eff_scale, eff_min = _q4_k_fields(self.raw, N, K)

        zero = torch.where(
            eff_scale != 0, eff_min / eff_scale, torch.zeros_like(eff_min)
        )
        # Nibble-pack for Int4Tensor: even index -> low nibble, odd -> high.
        packed = q[:, ::2] | (q[:, 1::2] << 4)
        return Int4Tensor(
            qdata=packed,
            # Int4Tensor scale/zero layout is (K // gs, N) -- transposed.
            scale=eff_scale.to(torch.bfloat16).t().contiguous(),
            zero_point=zero.to(torch.bfloat16).t().contiguous(),
            block_size=[1, Q4_K_GROUP_SIZE],
            shape=torch.Size([N, K]),
        )

    def to_intx_unpacked_to_int8_tensor(self) -> Tensor:
        """Convert to a torchao ``IntxUnpackedToInt8Tensor`` (Q4_K or Q6_K).

        Q6_K maps to a symmetric int8 tensor (values [-32, 31], zero-point 0).
        Q4_K maps to a 4-bit tensor: values are centered to [-8, 7] and the
        affine min is folded into a (float) zero-point, so the rewrite is exact.
        """
        from torchao.quantization import IntxUnpackedToInt8Tensor

        N, K = int(self.shape[0]), int(self.shape[1])
        if self.ggml_type == "q6_k":
            q, eff_scale = _q6_k_fields(self.raw, N, K)
            return IntxUnpackedToInt8Tensor(
                qdata=q,
                scale=eff_scale.to(torch.bfloat16),
                zero_point=torch.zeros_like(eff_scale, dtype=torch.int8),
                target_dtype=torch.int8,
                block_size=(1, Q6_K_GROUP_SIZE),
                dtype=torch.bfloat16,
                activation_quantization=None,
            )
        if self.ggml_type == "q4_k":
            q, eff_scale, eff_min = _q4_k_fields(self.raw, N, K)
            zero = torch.where(
                eff_scale != 0, eff_min / eff_scale, torch.zeros_like(eff_min)
            )
            # Center quants [0, 15] -> [-8, 7] and shift the zero-point to match
            # (dequant = scale * (q - zp) is preserved).
            return IntxUnpackedToInt8Tensor(
                qdata=q.to(torch.int8) - 8,
                scale=eff_scale.to(torch.bfloat16),
                zero_point=(zero - 8).to(torch.bfloat16),
                target_dtype=torch.int4,
                block_size=(1, Q4_K_GROUP_SIZE),
                dtype=torch.bfloat16,
                activation_quantization=None,
            )
        raise NotImplementedError(
            f"to_intx_unpacked_to_int8_tensor supports q4_k/q6_k; "
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


# ---------------------------------------------------------------------------
# Loader
# ---------------------------------------------------------------------------


def iter_gguf(
    path: str,
) -> Iterator[Tuple[str, Union[ExportableGGUFTensor, Tensor]]]:
    """Stream ``(name, value)`` for every tensor in a GGUF file (low peak mem).

    Quantized tensors (Q4_K, Q6_K) are wrapped as ``ExportableGGUFTensor`` with
    the raw block bytes; F32/F16 are returned as plain float tensors (bf16 for
    F16). GGUF shapes are reversed to PyTorch ``(N, K)`` convention.
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
            yield tensor.name, ExportableGGUFTensor.from_raw(
                raw, _TYPE_BY_GGML_ID[ttype]
            )
        elif tensor.tensor_type == GGMLQuantizationType.F32:
            yield tensor.name, flat.view(torch.float32).reshape(shape).clone()
        elif tensor.tensor_type == GGMLQuantizationType.F16:
            yield tensor.name, flat.view(torch.float16).reshape(shape).to(
                torch.bfloat16
            )
        else:
            raise ValueError(f"Unsupported GGUF quant type: {tensor.tensor_type}")


def load_gguf(path: str) -> Dict[str, Union[ExportableGGUFTensor, Tensor]]:
    """Load a GGUF file into ``{name -> ExportableGGUFTensor | Tensor}``.

    Holds all tensors at once; use :func:`iter_gguf` for low peak memory.
    """
    return dict(iter_gguf(path))
