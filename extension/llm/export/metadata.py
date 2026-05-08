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
"""

import struct
from typing import Dict, List, Sequence, Union

METADATA_PREFIX = "metadata."

MetadataValue = Union[str, int, float, bytes, Sequence[int]]


# Type tags for self-describing metadata values
_TAG_INT = b'\x01'
_TAG_FLOAT = b'\x02'
_TAG_STRING = b'\x03'
_TAG_INT_LIST = b'\x04'
_TAG_BYTES = b'\x05'


def _encode_value(key: str, value: MetadataValue) -> bytes:
    if isinstance(value, str):
        return _TAG_STRING + value.encode("utf-8")
    elif isinstance(value, (list, tuple)):
        return _TAG_INT_LIST + struct.pack(f"<I{len(value)}q", len(value), *value)
    elif isinstance(value, int):
        return _TAG_INT + struct.pack("<q", value)
    elif isinstance(value, float):
        return _TAG_FLOAT + struct.pack("<d", value)
    elif isinstance(value, bytes):
        return _TAG_BYTES + value
    raise TypeError(f"Unsupported metadata type {type(value)} for key \'{key}\'")


def add_metadata(
    edge_manager,  # EdgeProgramManager
    metadata: Dict[str, MetadataValue],
) -> None:
    """Add metadata KV pairs to a PTE file during export.

    Call BEFORE edge_manager.to_executorch().

    Args:
        edge_manager: The EdgeProgramManager from to_edge() or
            to_edge_transform_and_lower().
        metadata: Dict mapping string keys to values (str, int, float, or bytes).
            Keys are automatically prefixed with "metadata." to avoid collision
            with backend named data.
    """
    for key, value in metadata.items():
        edge_manager._named_data_store.add_named_data(
            key=f"{METADATA_PREFIX}{key}",
            data=_encode_value(key, value),
        )


def read_metadata(pte_path: str) -> Dict[str, bytes]:
    """Read all metadata entries from a PTE file.

    Returns raw bytes for each key (without the "metadata." prefix).
    Use get_string/get_int/get_float for typed access.
    """
    from executorch.exir._serialize._program import deserialize_pte_binary

    with open(pte_path, "rb") as f:
        pte_data = f.read()

    pte_file = deserialize_pte_binary(pte_data)

    result = {}
    if pte_file.named_data is not None:
        for key, entry in pte_file.named_data.pte_data.items():
            if key.startswith(METADATA_PREFIX):
                short_key = key[len(METADATA_PREFIX):]
                result[short_key] = pte_file.named_data.buffers[entry.buffer_index]

    return result


def get_string(metadata: Dict[str, bytes], key: str) -> str:
    data = metadata[key]
    if data[0:1] == _TAG_STRING:
        return data[1:].decode("utf-8")
    # Legacy format (no tag) - treat entire buffer as string
    return data.decode("utf-8")


def get_int(metadata: Dict[str, bytes], key: str) -> int:
    data = metadata[key]
    if data[0:1] == _TAG_INT:
        return struct.unpack("<q", data[1:9])[0]
    # Legacy format (no tag)
    return struct.unpack("<q", data)[0]


def get_float(metadata: Dict[str, bytes], key: str) -> float:
    data = metadata[key]
    if data[0:1] == _TAG_FLOAT:
        return struct.unpack("<d", data[1:9])[0]
    # Legacy format (no tag)
    return struct.unpack("<d", data)[0]


def get_int_list(metadata: Dict[str, bytes], key: str) -> List[int]:
    data = metadata[key]
    offset = 0
    if data[0:1] == _TAG_INT_LIST:
        offset = 1
    (count,) = struct.unpack_from("<I", data, offset)
    return list(struct.unpack_from(f"<{count}q", data, offset + 4))
