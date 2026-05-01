# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Unpack GGUF quantized tensors to torchao tensor subclasses.

Supports Q4_K, Q6_K, F32, and F16 tensor types. Two public APIs:

  - ``unpack_gguf_tensor`` — convert a single tensor
  - ``iter_gguf_tensors`` — stream all tensors from a file (low peak memory)

Model-agnostic. For Gemma 4 31B key mapping and model loading, see
``gguf_loader.py``.
"""

from collections.abc import Iterator

import torch

QK_K = 256  # super-block size for k-quants
Q4_K_GROUPS = 8  # sub-blocks per Q4_K super-block
Q4_K_GROUP_SIZE = QK_K // Q4_K_GROUPS  # 32
Q6_K_GROUPS = 16  # sub-blocks per Q6_K super-block
Q6_K_GROUP_SIZE = QK_K // Q6_K_GROUPS  # 16


def _raw_tensor(data: bytes) -> torch.Tensor:
    """Wrap a numpy mmap view as a uint8 torch tensor (zero-copy)."""
    return torch.frombuffer(memoryview(data), dtype=torch.uint8)


def _read_f16(raw: torch.Tensor, col_start: int, col_end: int) -> torch.Tensor:
    """Read fp16 field from block bytes, return float32."""
    return raw[:, col_start:col_end].contiguous().view(torch.float16).float()


def _unpack_q4_k(data, shape: list[int]) -> torch.Tensor:
    """Unpack Q4_K super-blocks into an ``Int4Tensor``.

    Q4_K block layout (144 bytes per 256 values):
      - d     (2B, fp16): super-block scale
      - dmin  (2B, fp16): super-block min
      - scales (12B): 8 sub-block scales + 8 sub-block mins, 6-bit packed
      - qs    (128B): 256 4-bit values, two per byte

    Dequant: weight = d * sub_scale * q - dmin * sub_min
    """
    from torchao.quantization.quantize_.workflows.int4.int4_tensor import Int4Tensor

    N, K = shape
    assert K % QK_K == 0, f"Q4_K requires K divisible by {QK_K}, got {K}"
    n_blocks = N * (K // QK_K)
    block_bytes = 2 + 2 + 12 + QK_K // 2  # 144
    raw = _raw_tensor(data).reshape(n_blocks, block_bytes)

    d = _read_f16(raw, 0, 2)
    dmin = _read_f16(raw, 2, 4)
    s = raw[:, 4:16]
    qs = raw[:, 16:144]

    sc = torch.empty(n_blocks, 8, dtype=torch.float32)
    mn = torch.empty(n_blocks, 8, dtype=torch.float32)
    sc[:, :4] = (s[:, :4] & 0x3F).float()
    mn[:, :4] = (s[:, 4:8] & 0x3F).float()
    sc[:, 4:] = ((s[:, 8:12] & 0xF) | ((s[:, :4] >> 6) << 4)).float()
    mn[:, 4:] = ((s[:, 8:12] >> 4) | ((s[:, 4:8] >> 6) << 4)).float()
    del s

    eff_scale = (d * sc).reshape(N, -1)
    eff_min = (dmin * mn).reshape(N, -1)
    del d, dmin, sc, mn

    zero_std = torch.where(
        eff_scale != 0, eff_min / eff_scale, torch.zeros_like(eff_min)
    )
    del eff_min

    # GGUF Q4_K nibble order: 32 lows then 32 highs per sub-block pair
    low = (qs & 0x0F).to(torch.uint8)
    high = ((qs >> 4) & 0x0F).to(torch.uint8)
    qdata_unpacked = torch.cat(
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
    del qs, low, high

    # Nibble-pack for Int4Tensor: even=LOW, odd=HIGH
    packed = qdata_unpacked[:, ::2] | (qdata_unpacked[:, 1::2] << 4)

    # Int4Tensor scale/zero layout: (K//gs, N) — transposed
    return Int4Tensor(
        qdata=packed,
        scale=eff_scale.to(torch.bfloat16).t().contiguous(),
        zero_point=zero_std.to(torch.bfloat16).t().contiguous(),
        block_size=[1, Q4_K_GROUP_SIZE],
        shape=torch.Size([N, K]),
    )


def _unpack_q6_k(data, shape: list[int]) -> torch.Tensor:
    """Unpack Q6_K super-blocks into an ``IntxUnpackedToInt8Tensor``.

    Q6_K block layout (210 bytes per 256 values):
      - ql    (128B): lower 4 bits of 256 6-bit values
      - qh    (64B): upper 2 bits of 256 6-bit values
      - scales (16B): 16 int8 sub-block scales (groups of 16)
      - d     (2B, fp16): super-block scale

    Dequant: weight = d * scale_j * (q - 32)
    Values are 6-bit [-32, 31], widened to INT8.
    """
    from torchao.quantization import IntxUnpackedToInt8Tensor

    N, K = shape
    assert K % QK_K == 0, f"Q6_K requires K divisible by {QK_K}, got {K}"
    n_blocks = N * (K // QK_K)
    block_bytes = 2 + QK_K // 2 + QK_K // 4 + QK_K // 16  # 210
    raw = _raw_tensor(data).reshape(n_blocks, block_bytes)

    ql = raw[:, 0:128]
    qh = raw[:, 128:192]
    sc = raw[:, 192:208]
    d = _read_f16(raw, 208, 210)

    qh0 = qh[:, :32]
    qh1 = qh[:, 32:64]
    qdata = torch.empty(n_blocks, QK_K, dtype=torch.int16)
    qdata[:, 0:32] = (ql[:, :32] & 0x0F) | ((qh0 & 0x03) << 4)
    qdata[:, 32:64] = (ql[:, 32:64] & 0x0F) | (((qh0 >> 2) & 0x03) << 4)
    qdata[:, 64:96] = ((ql[:, :32] >> 4) & 0x0F) | (((qh0 >> 4) & 0x03) << 4)
    qdata[:, 96:128] = ((ql[:, 32:64] >> 4) & 0x0F) | (((qh0 >> 6) & 0x03) << 4)
    qdata[:, 128:160] = (ql[:, 64:96] & 0x0F) | ((qh1 & 0x03) << 4)
    qdata[:, 160:192] = (ql[:, 96:128] & 0x0F) | (((qh1 >> 2) & 0x03) << 4)
    qdata[:, 192:224] = ((ql[:, 64:96] >> 4) & 0x0F) | (((qh1 >> 4) & 0x03) << 4)
    qdata[:, 224:256] = ((ql[:, 96:128] >> 4) & 0x0F) | (((qh1 >> 6) & 0x03) << 4)
    qdata -= 32
    del ql, qh, qh0, qh1

    # sc bytes are signed int8 scales; reinterpret from uint8
    eff_scale = (d * sc.to(torch.int8).float()).reshape(N, -1)
    del d, sc

    return IntxUnpackedToInt8Tensor(
        qdata=qdata.reshape(N, K).to(torch.int8),
        scale=eff_scale.to(torch.bfloat16),
        zero_point=torch.zeros_like(eff_scale, dtype=torch.int8),
        target_dtype=torch.int8,
        block_size=(1, Q6_K_GROUP_SIZE),
        dtype=torch.bfloat16,
        activation_quantization=None,
    )


def unpack_gguf_tensor(
    tensor_data,
    tensor_type,
    shape: list[int],
) -> torch.Tensor:
    """Unpack a single GGUF tensor.

    Returns an ``Int4Tensor`` for Q4_K, ``IntxUnpackedToInt8Tensor`` for Q6_K,
    or a plain ``torch.Tensor`` for F32/F16.
    """
    from gguf import GGMLQuantizationType

    if tensor_type == GGMLQuantizationType.Q4_K:
        return _unpack_q4_k(tensor_data, shape)
    elif tensor_type == GGMLQuantizationType.Q6_K:
        return _unpack_q6_k(tensor_data, shape)
    elif tensor_type == GGMLQuantizationType.F32:
        return _raw_tensor(tensor_data).view(torch.float32).reshape(shape).clone()
    elif tensor_type == GGMLQuantizationType.F16:
        return (
            _raw_tensor(tensor_data)
            .view(torch.float16)
            .reshape(shape)
            .to(torch.bfloat16)
        )
    else:
        raise ValueError(f"Unsupported GGUF quant type: {tensor_type}")


def iter_gguf_tensors(
    path: str,
) -> Iterator[tuple[str, torch.Tensor]]:
    """Yield ``(name, result)`` for each tensor in a GGUF file.

    Processes one tensor at a time for low peak memory. Tensor names are
    GGUF names (e.g., ``blk.0.attn_q.weight``); the caller handles key
    remapping. GGUF shapes are reversed to PyTorch convention automatically.
    """
    from gguf import GGUFReader

    reader = GGUFReader(path)
    for tensor in reader.tensors:
        shape = list(reversed(tensor.shape.tolist()))
        result = unpack_gguf_tensor(tensor.data, tensor.tensor_type, shape)
        yield tensor.name, result
