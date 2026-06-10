# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Model metadata storage for PTE files.

Embeds model metadata (tokenizer config, chat templates, architecture info)
directly in PTE files via the NamedData mechanism. Replaces the current
constant_methods approach (which creates full ExecutionPlan entries for
simple constant values).

Keys use a dotted namespace.field convention:
    tokenizer.bos_id, tokenizer.eos_ids, context.max_seq_len, etc.

Wire format: each value is prefixed with a 1-byte type tag:
    0x01 = int64  (8 bytes, little-endian)
    0x02 = float64 (8 bytes, little-endian)
    0x03 = string (UTF-8, no length prefix - length is implicit from buffer size)
    0x04 = int_list (uint32 count + count * int64, all little-endian)
    0x05 = bytes (raw, no framing)
"""

from __future__ import annotations

import struct
from typing import Dict, List, Sequence, TYPE_CHECKING, Union

if TYPE_CHECKING:
    from executorch.exir import EdgeProgramManager

METADATA_PREFIX = "metadata."

# Type tag constants (1 byte, prefixed to every encoded value).
TAG_INT: int = 0x01
TAG_FLOAT: int = 0x02
TAG_STRING: int = 0x03
TAG_INT_LIST: int = 0x04
TAG_BYTES: int = 0x05

MetadataValue = Union[str, int, float, bytes, Sequence[int]]


def _encode_value(key: str, value: MetadataValue) -> bytes:
    if isinstance(value, bool):
        raise TypeError(f"bool not supported for key '{key}', use int (0/1) instead")
    if isinstance(value, str):
        return bytes([TAG_STRING]) + value.encode("utf-8")
    elif isinstance(value, (list, tuple)):
        for i, elem in enumerate(value):
            if not isinstance(elem, int) or isinstance(elem, bool):
                raise TypeError(
                    f"list element {i} for key '{key}' must be int, got {type(elem)}"
                )
        return bytes([TAG_INT_LIST]) + struct.pack(f"<I{len(value)}q", len(value), *value)
    elif isinstance(value, int):
        return bytes([TAG_INT]) + struct.pack("<q", value)
    elif isinstance(value, float):
        return bytes([TAG_FLOAT]) + struct.pack("<d", value)
    elif isinstance(value, bytes):
        return bytes([TAG_BYTES]) + value
    raise TypeError(f"Unsupported metadata type {type(value)} for key '{key}'")


def add_metadata(
    edge_manager: EdgeProgramManager,
    metadata: Dict[str, MetadataValue],
) -> None:
    """Add metadata KV pairs to a PTE file during export.

    Call BEFORE edge_manager.to_executorch().

    Args:
        edge_manager: The EdgeProgramManager from to_edge() or
            to_edge_transform_and_lower().
        metadata: Dict mapping string keys to values (str, int, float, bytes,
            or list[int]). Keys are automatically prefixed with "metadata." to
            avoid collision with backend named data.
    """
    for key, value in metadata.items():
        edge_manager._named_data_store.add_named_data(
            key=f"{METADATA_PREFIX}{key}",
            data=_encode_value(key, value),
        )


def read_metadata(pte_path: str) -> Dict[str, bytes]:
    """Read all metadata entries from a PTE file.

    Returns raw bytes (including type tag prefix) for each key (without the
    "metadata." prefix). Use get_string/get_int/get_float for typed access.

    WARNING: Loads the entire PTE file into memory. Not suitable for
    large model files in production; intended for testing and debugging.
    """
    from executorch.exir._serialize._program import deserialize_pte_binary

    with open(pte_path, "rb") as f:
        pte_data = f.read()

    pte_file = deserialize_pte_binary(pte_data)

    result = {}
    if pte_file.named_data is not None:
        for key, entry in pte_file.named_data.pte_data.items():
            if key.startswith(METADATA_PREFIX):
                short_key = key[len(METADATA_PREFIX) :]
                result[short_key] = pte_file.named_data.buffers[entry.buffer_index]

    return result


def _check_tag(data: bytes, key: str, expected_tag: int, type_name: str) -> None:
    """Validate the type tag prefix."""
    if len(data) == 0:
        raise ValueError(f"Empty data for key '{key}'")
    actual_tag = data[0]
    if actual_tag != expected_tag:
        tag_names = {
            TAG_INT: "int",
            TAG_FLOAT: "float",
            TAG_STRING: "string",
            TAG_INT_LIST: "int_list",
            TAG_BYTES: "bytes",
        }
        actual_name = tag_names.get(actual_tag, f"unknown(0x{actual_tag:02x})")
        raise TypeError(
            f"Type mismatch for key '{key}': expected {type_name}, "
            f"got {actual_name} (tag=0x{actual_tag:02x})"
        )


def get_string(metadata: Dict[str, bytes], key: str) -> str:
    data = metadata[key]
    _check_tag(data, key, TAG_STRING, "string")
    return data[1:].decode("utf-8")


def get_int(metadata: Dict[str, bytes], key: str) -> int:
    data = metadata[key]
    _check_tag(data, key, TAG_INT, "int")
    return struct.unpack("<q", data[1:])[0]


def get_float(metadata: Dict[str, bytes], key: str) -> float:
    data = metadata[key]
    _check_tag(data, key, TAG_FLOAT, "float")
    return struct.unpack("<d", data[1:])[0]


def get_bytes(metadata: Dict[str, bytes], key: str) -> bytes:
    data = metadata[key]
    _check_tag(data, key, TAG_BYTES, "bytes")
    return data[1:]


def get_int_list(metadata: Dict[str, bytes], key: str) -> List[int]:
    data = metadata[key]
    _check_tag(data, key, TAG_INT_LIST, "int_list")
    payload = data[1:]
    (count,) = struct.unpack_from("<I", payload, 0)
    return list(struct.unpack_from(f"<{count}q", payload, 4))
