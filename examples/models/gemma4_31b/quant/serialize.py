# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Serialize and persist quantized weights.

Two layers:

  - **serialize / deserialize** — convert between ``CanonicalQuantizedWeight``
    objects and plain tensors + JSON metadata. Pure logic, no I/O. The output
    is a ``(tensors_dict, metadata_dict)`` pair that any file writer can
    consume.
  - **save / load** — write/read the serialized form to/from safetensors on
    disk. Thin I/O wrappers around ``safetensors.save_file`` /
    ``safetensors.safe_open``.

For 4-bit weights, qdata is nibble-packed (two values per byte) during
serialization to keep file size at ~0.5 bytes/param.
"""

import json
from dataclasses import dataclass
from typing import Optional

import torch
from safetensors import safe_open
from safetensors.torch import save_file

from .recipe import QuantConfig

# Bump when the on-disk layout changes in a backward-incompatible way
# (e.g., different nibble-pack convention, renamed keys, new required fields).
# The loader rejects files with an unsupported version rather than silently
# producing corrupt data.
FORMAT_VERSION = "1"
_SUPPORTED_VERSIONS = {FORMAT_VERSION}


@dataclass
class CanonicalQuantizedWeight:
    """Packing-free quantized weight representation.

    ``qdata``  int8 values: [0, 15] for 4-bit (both symmetric and asymmetric
               are stored as unsigned after shifting), [-128, 127] for 8-bit.
    ``scale``  bf16 per-group scales, shape ``[*weight_shape[:-1], K // group_size]``.
    ``zero``   bf16 per-group zero points (``None`` when symmetric).
    ``config`` the ``QuantConfig`` that produced this.
    """

    qdata: torch.Tensor
    scale: torch.Tensor
    zero: Optional[torch.Tensor]
    config: QuantConfig


# ---------------------------------------------------------------------------
# Nibble packing for 4-bit on-disk storage.
#
# Two 4-bit values are packed into one byte to halve file size. The
# convention is: even-indexed values go into the LOW nibble (bits 0-3),
# odd-indexed values go into the HIGH nibble (bits 4-7).
#
#   values:  [v0, v1, v2, v3, ...]   (each in [0, 15])
#   packed:  [v0 | (v1 << 4), v2 | (v3 << 4), ...]
#   byte 0:  bits 0-3 = v0,  bits 4-7 = v1
#
# To unpack: low = byte & 0x0F, high = (byte >> 4) & 0x0F.
#
# This matches the Triton fused_moe kernel's unpack convention
# ((byte >> (k%2)*4) & 0xF) and Qwen's _quantize_experts_int4 packing.
# Note: tinygemm uses the OPPOSITE convention (even=HIGH, odd=LOW) — the
# CUDA packer in pack_cuda.py handles that conversion separately.


def _nibble_pack(qdata: torch.Tensor) -> torch.Tensor:
    """Pack int8 values (each in [0, 15]) into half the last dim.

    Even-indexed values → low nibble, odd-indexed → high nibble.
    """
    assert qdata.shape[-1] % 2 == 0, f"Last dim must be even, got {qdata.shape}"
    low = qdata[..., ::2].to(torch.uint8)
    high = qdata[..., 1::2].to(torch.uint8)
    return (low | (high << 4)).to(torch.int8).contiguous()


def _nibble_unpack(packed: torch.Tensor, orig_last_dim: int) -> torch.Tensor:
    """Unpack nibble-packed int8 → original last dim.

    Low nibble (bits 0-3) → even indices, high nibble (bits 4-7) → odd indices.
    """
    p = packed.to(torch.uint8)
    low = (p & 0x0F).to(torch.int8)
    high = ((p >> 4) & 0x0F).to(torch.int8)
    return torch.stack([low, high], dim=-1).reshape(*packed.shape[:-1], orig_last_dim)


# ---------------------------------------------------------------------------
# Serialize / deserialize (pure logic, no I/O)


def serialize(
    quantized: dict[str, CanonicalQuantizedWeight],
    unquantized: dict[str, torch.Tensor],
) -> tuple[dict[str, torch.Tensor], dict[str, str]]:
    """Convert quantized + unquantized weights to plain tensors + metadata.

    Returns ``(tensors, header)`` ready for any file writer. Quantized
    weights become ``{fqn}.qdata``, ``{fqn}.scale``, optionally
    ``{fqn}.zero``. For 4-bit, qdata is nibble-packed.
    """
    tensors: dict[str, torch.Tensor] = {}
    quant_meta: dict[str, dict] = {}

    for fqn, cw in quantized.items():
        qdata = cw.qdata
        if cw.config.bits == 4:
            qdata = _nibble_pack(qdata)
        tensors[f"{fqn}.qdata"] = qdata.contiguous()
        tensors[f"{fqn}.scale"] = cw.scale.contiguous()
        if cw.zero is not None:
            tensors[f"{fqn}.zero"] = cw.zero.contiguous()
        quant_meta[fqn] = {
            "bits": cw.config.bits,
            "group_size": cw.config.group_size,
            "symmetric": cw.config.symmetric,
            "method": cw.config.method,
            "shape": list(cw.qdata.shape),
        }

    for fqn, tensor in unquantized.items():
        tensors[fqn] = tensor.contiguous()

    header = {"format_version": FORMAT_VERSION}
    if quant_meta:
        header["quant"] = json.dumps(quant_meta)

    return tensors, header


def deserialize(
    tensors: dict[str, torch.Tensor],
    header: dict[str, str],
) -> tuple[dict[str, CanonicalQuantizedWeight], dict[str, torch.Tensor]]:
    """Reconstruct quantized + unquantized weights from plain tensors + metadata.

    Inverse of ``serialize``. Returns ``(quantized, unquantized)`` dicts.
    """
    version = header.get("format_version", "1")
    if version not in _SUPPORTED_VERSIONS:
        raise ValueError(
            f"Unsupported format version {version!r}. "
            f"This code supports {sorted(_SUPPORTED_VERSIONS)}. "
            f"Update the quant package or re-quantize the model."
        )

    quant_meta = json.loads(header.get("quant", "{}"))

    quantized: dict[str, CanonicalQuantizedWeight] = {}
    consumed_keys: set[str] = set()

    for fqn, meta in quant_meta.items():
        config = QuantConfig(
            bits=meta["bits"],
            group_size=meta["group_size"],
            symmetric=meta["symmetric"],
            method=meta["method"],
        )
        qdata = tensors[f"{fqn}.qdata"]
        consumed_keys.add(f"{fqn}.qdata")

        original_shape = meta["shape"]
        if config.bits == 4:
            qdata = _nibble_unpack(qdata, original_shape[-1])

        scale = tensors[f"{fqn}.scale"]
        consumed_keys.add(f"{fqn}.scale")

        zero = tensors.get(f"{fqn}.zero")
        if zero is not None:
            consumed_keys.add(f"{fqn}.zero")

        quantized[fqn] = CanonicalQuantizedWeight(
            qdata=qdata, scale=scale, zero=zero, config=config
        )

    unquantized = {k: v for k, v in tensors.items() if k not in consumed_keys}

    return quantized, unquantized


# ---------------------------------------------------------------------------
# Save / load (I/O wrappers)


def save(
    quantized: dict[str, CanonicalQuantizedWeight],
    unquantized: dict[str, torch.Tensor],
    path: str,
) -> int:
    """Serialize and write to safetensors. Returns the number of tensors written."""
    tensors, header = serialize(quantized, unquantized)
    save_file(tensors, path, metadata=header)
    return len(tensors)


def load(
    path: str,
) -> tuple[dict[str, CanonicalQuantizedWeight], dict[str, torch.Tensor]]:
    """Read safetensors and deserialize. Returns ``(quantized, unquantized)``."""
    with safe_open(path, framework="pt", device="cpu") as f:
        header = f.metadata()
        tensors = {k: f.get_tensor(k) for k in f.keys()}
    return deserialize(tensors, header)
