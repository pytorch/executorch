# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
Fat PTE: packs multiple backend specializations into one delegate payload.
Runtime selects the best specialization at load time. All specializations
share a single PTD (named data store).

Binary layout:
  [4B] magic "NFAT"
  [4B] version (1)
  [4B] num_specializations
  Per specialization:
    [32B] backend_id (utf-8, null-padded)
    [8B]  offset into data section
    [8B]  length
  [payload bytes ...]
"""

import struct
from typing import Dict, List, Optional, Tuple

from executorch.exir._serialize._named_data_store import NamedDataStore
from executorch.exir.backend.backend_details import PreprocessResult

FAT_MAGIC = b"NFAT"
FAT_VERSION = 1
_ENTRY_FMT = "32sQQ"
_ENTRY_SIZE = struct.calcsize(_ENTRY_FMT)


def pack_fat_blob(specializations: List[Tuple[str, bytes]]) -> bytes:
    """Pack (backend_id, payload) pairs into a fat blob."""
    header = struct.pack("<4sII", FAT_MAGIC, FAT_VERSION, len(specializations))

    entries = []
    offset = 0
    for backend_id, payload in specializations:
        encoded = backend_id.encode("ascii")
        if len(encoded) > 32:
            raise ValueError(
                f"Backend ID '{backend_id}' is {len(encoded)} bytes; max is 32"
            )
        bid = encoded.ljust(32, b"\x00")
        entries.append(struct.pack("<" + _ENTRY_FMT, bid, offset, len(payload)))
        offset += len(payload)

    return header + b"".join(entries) + b"".join(p for _, p in specializations)


def build_fat_result(
    results: List[Tuple[str, PreprocessResult]],
    debug_handle_map: Optional[Dict] = None,
) -> PreprocessResult:
    """Merge (backend_id, PreprocessResult) pairs into a single fat result."""
    fat_entries: List[Tuple[str, bytes]] = []
    merged_data_store = NamedDataStore()

    for backend_id, result in results:
        fat_entries.append((backend_id, result.processed_bytes))

        if result.data_store_output:
            for key, entry in result.data_store_output.pte_data.items():
                buf = result.data_store_output.buffers[entry.buffer_index]
                merged_data_store.add_named_data(key, buf, alignment=entry.alignment)

    if debug_handle_map is None and results:
        debug_handle_map = results[0][1].debug_handle_map

    return PreprocessResult(
        processed_bytes=pack_fat_blob(fat_entries),
        debug_handle_map=debug_handle_map,
        data_store_output=merged_data_store.get_named_data_store_output(),
    )
