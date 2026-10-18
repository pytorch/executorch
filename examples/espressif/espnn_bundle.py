# Copyright 2026 Mwaniki Njagi.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.
#
# SPDX-License-Identifier: BSD-3-Clause

"""
On-device test bundles for examples/espressif/espnn_runner.

Bundle (little-endian), magic "XTB1" for XTS3 int8 graphs or "XTP1" for .pte programs:
    magic u32 version=1 u32 count
    count x { char name[32]; u32 blob_offset; u32 blob_size; u32 vectors_offset; u32 vectors_size }
    payloads aligned to 128 bytes

XTV2 vectors (XTS3 graphs; inputs and outputs in device layout, bit-exact):
    "XTV2" u32 version=2 u32 samples u32 inputs u32 outputs
    u32 input_bytes[inputs] u32 output_bytes[outputs]
    samples x { int8 inputs..., int8 expected outputs... }

XTE1 vectors (.pte programs; ExecuTorch tensors in logical layout):
    "XTE1" u32 version=1 u32 samples u32 inputs u32 outputs f32 min_sqnr_db u32 flags
    {u32 scalar_type, u32 numel} x (inputs + outputs)
    samples x { inputs..., expected outputs... }
    flags bit 0: outputs must be bit-identical; bit 1: outputs are stored as CRC32 only.
"""

from __future__ import annotations

import math
import struct
import zlib
from pathlib import Path

import numpy as np
import torch

from executorch.backends.espressif.reference import esp_nn_reference as reference

NAME_BYTES = 32
ALIGNMENT = 128
FLAG_EXACT = 1
FLAG_CRC = 2
SCALAR_TYPES = {torch.float32: 6, torch.int8: 1, torch.int64: 4, torch.int32: 3}


def sqnr_db(reference_values: torch.Tensor, actual: torch.Tensor) -> float:
    reference_values = reference_values.to(torch.float64)
    noise = (reference_values - actual.to(torch.float64)).pow(2).sum().item()
    if noise == 0:
        return math.inf
    return 10 * math.log10(reference_values.pow(2).sum().item() / noise)


def xtv2_vectors(graph, samples: list[list[np.ndarray]]) -> bytes:
    """Quantize float samples and record the bit-exact outputs of the ESP-NN reference."""
    interpreter = reference.ReferenceInterpreter(graph)
    input_tensors = [graph.tensors[i] for i in interpreter.inputs]
    output_tensors = [graph.tensors[i] for i in interpreter.outputs]
    input_bytes = [int(np.prod(reference.device_shape(t))) for t in input_tensors]
    output_bytes = [int(np.prod(reference.device_shape(t))) for t in output_tensors]
    body = bytearray()
    for sample in samples:
        device_inputs = [
            reference.to_device_layout(t, reference.quantize(t, np.asarray(value)))
            for t, value in zip(input_tensors, sample)
        ]
        outputs = interpreter.run_outputs(device_inputs)
        for value in device_inputs + outputs:
            body += value.astype(np.int8).tobytes()
    header = struct.pack(
        "<4sIIII", b"XTV2", 2, len(samples), len(input_tensors), len(output_tensors)
    )
    sizes = struct.pack(
        f"<{len(input_bytes) + len(output_bytes)}I", *input_bytes, *output_bytes
    )
    return header + sizes + bytes(body)


def xte1_vectors(rows, exact: bool, min_sqnr_db: float, crc: bool = False) -> bytes:
    """rows: list of (inputs, outputs), each a list of tensors."""
    first_inputs, first_outputs = rows[0]
    descriptors = [
        (SCALAR_TYPES[t.dtype], t.numel())
        for t in list(first_inputs) + list(first_outputs)
    ]
    flags = (FLAG_EXACT if exact else 0) | (FLAG_CRC if exact and crc else 0)
    header = struct.pack(
        "<4sIIIIfI",
        b"XTE1",
        1,
        len(rows),
        len(first_inputs),
        len(first_outputs),
        float(min_sqnr_db),
        flags,
    )
    table = b"".join(struct.pack("<II", dtype, numel) for dtype, numel in descriptors)
    body = bytearray()
    for inputs, outputs in rows:
        for t in inputs:
            body += t.contiguous().numpy().tobytes()
        for t in outputs:
            data = t.contiguous().numpy().tobytes()
            body += struct.pack("<I", zlib.crc32(data)) if flags & FLAG_CRC else data
    return header + table + bytes(body)


def write_bundle(path: Path, magic: bytes, entries) -> int:
    """entries: list of (name, blob_bytes, vectors_bytes). Returns the bundle size in bytes."""
    if magic not in (b"XTB1", b"XTP1"):
        raise ValueError(f"Unknown bundle magic {magic!r}")
    payload = bytearray(b"\0" * (12 + len(entries) * (NAME_BYTES + 16)))
    records = []
    for name, blob, vectors in entries:
        payload.extend(b"\0" * ((-len(payload)) % ALIGNMENT))
        blob_offset = len(payload)
        payload += blob
        payload.extend(b"\0" * ((-len(payload)) % ALIGNMENT))
        vectors_offset = len(payload)
        payload += vectors
        records.append((name, blob_offset, len(blob), vectors_offset, len(vectors)))
    header = struct.pack("<4sII", magic, 1, len(records))
    table = b"".join(
        struct.pack(f"<{NAME_BYTES}sIIII", name.encode()[: NAME_BYTES - 1], *rest)
        for name, *rest in records
    )
    payload[: len(header) + len(table)] = header + table
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(bytes(payload))
    return len(payload)


def read_bundle(path: Path) -> tuple[bytes, list[tuple[str, bytes, bytes]]]:
    data = path.read_bytes()
    magic, version, count = struct.unpack_from("<4sII", data, 0)
    entries = []
    for index in range(count):
        name, blob_offset, blob_size, vectors_offset, vectors_size = struct.unpack_from(
            f"<{NAME_BYTES}sIIII", data, 12 + index * (NAME_BYTES + 16)
        )
        entries.append(
            (
                name.rstrip(b"\0").decode(),
                data[blob_offset : blob_offset + blob_size],
                data[vectors_offset : vectors_offset + vectors_size],
            )
        )
    return magic, entries


__all__ = ["read_bundle", "sqnr_db", "write_bundle", "xte1_vectors", "xtv2_vectors"]
