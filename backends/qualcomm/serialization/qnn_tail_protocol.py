# Copyright (c) Qualcomm Innovation Center, Inc.
# All rights reserved
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.
r"""QNN Tail Protocol — extensible metadata appendix for processed_bytes.

Embeds host-side metadata (schematics, future: calibration, timing hints, etc.)
inside the .PTE's ``processed_bytes`` buffer *after* the QNN context binary,
so the .PTE is fully self-contained for profiling and debugging tools.

Safety Guarantee
----------------
The on-device runtime never sees the tail.  The QNN context binary is fronted by
a 256-byte custom protocol header (see ``QnnCustomProtocol.h``) whose
``binary_size_`` field tells the runtime exactly how many bytes to consume.
Anything appended past that boundary is invisible to the DSP loader.

Wire Format (version 1)
------------------------
The tail is appended immediately after the QNN Custom Protocol Buffer (header +
context binary).  It is read from the END of ``processed_bytes`` — the fixed-
size footer at the very tail tells the reader where the sections begin.

::

    +========================================================================+
    |  QNN Custom Protocol Buffer (256-byte header + context binary)         |
    |  [Untouched — runtime reads only this part]                            |
    +========================================================================+
    |                                                                        |
    |  ┌──────────────────────────────────────────────────────────────────┐  |
    |  │  Section 0                                                       │  |
    |  │  ┌────────────────┬───────────────────────────────────────────┐  │  |
    |  │  │ Type    (4B LE)│ uint32 — identifies payload kind (enum)   │  │  |
    |  │  ├────────────────┼───────────────────────────────────────────┤  │  |
    |  │  │ Length  (8B LE)│ uint64 — byte length of Payload below     │  │  |
    |  │  ├────────────────┼───────────────────────────────────────────┤  │  |
    |  │  │ Payload        │ `Length` bytes, opaque to this layer       │  │  |
    |  │  └────────────────┴───────────────────────────────────────────┘  │  |
    |  ├──────────────────────────────────────────────────────────────────┤  |
    |  │  Section 1  (same layout)                                        │  |
    |  ├──────────────────────────────────────────────────────────────────┤  |
    |  │  ...                                                             │  |
    |  └──────────────────────────────────────────────────────────────────┘  |
    |                                                                        |
    +------------------------------------------------------------------------+
    |  FOOTER (fixed 22 bytes, always at the very end of processed_bytes)    |
    |  ┌──────────────────────────────────────────────────────────────────┐  |
    |  │ Total Sections Length  (8B LE) — sum of all section bytes above  │  |
    |  │ Section Count          (4B LE) — number of sections              │  |
    |  │ Version                (2B LE) — protocol version (currently 1)  │  |
    |  │ Magic                  (8B)    — b"QNNTAIL\\x00"                 │  |
    |  └──────────────────────────────────────────────────────────────────┘  |
    +------------------------------------------------------------------------+

Section Types (uint32 enum)
---------------------------
=====  ===========  ===========================================================
Value  Name         Payload format
=====  ===========  ===========================================================
0x01   SCHEMATIC    Sequence of named blobs (see below).  Used by optrace
                    tooling to locate per-graph ``*_schematic.bin`` files.
=====  ===========  ===========================================================

Reserve 0x00 as invalid.  New types are added by appending to this table and
updating the reader — unknown types are skipped by length.

SCHEMATIC Payload (type 0x01)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Zero or more entries packed sequentially:

::

    ┌────────────────────┬────────────────────┬────────────────┬──────────────┐
    │ Name Length (4B LE)│ Name (UTF-8 bytes) │ Data Len (8B LE)│ Data Bytes  │
    └────────────────────┴────────────────────┴────────────────┴──────────────┘

``Name`` is typically the graph name (e.g. ``"forward"``).
``Data`` is the raw schematic binary produced by the QNN compiler.

Extensibility
-------------
To add a new payload type:

  1. Define a new ``SECTION_*`` constant below.
  2. Add a packer helper that returns ``bytes`` for the payload.
  3. Add an unpacker case in ``unpack_tail_sections()``.
  4. Update this docstring's table.

The reader skips any section type it does not recognise, so older readers
gracefully ignore newer section types (forward-compatible), and newer readers
handle the absence of newer sections (backward-compatible).
"""

from __future__ import annotations

import struct
from typing import Dict, List, Tuple

# ==============================================================================
# Constants
# ==============================================================================

TAIL_MAGIC: bytes = b"QNNTAIL\x00"
TAIL_VERSION: int = 1
FOOTER_SIZE: int = 8 + 4 + 2 + 8  # total_len + count + version + magic = 22

# Section type tags (uint32).  0x00 is reserved/invalid.
SECTION_SCHEMATIC: int = 0x01


# ==============================================================================
# Binary encoding helpers
#
# These wrap Python's `struct` module to give each wire-level read/write a
# semantic name.  All integers are little-endian (LE) to match the QNN SDK
# convention and ARM host byte order.
#
# struct format reference used below:
#   "<" = little-endian byte order
#   "I" = unsigned 32-bit integer (4 bytes)
#   "Q" = unsigned 64-bit integer (8 bytes)
#   "H" = unsigned 16-bit integer (2 bytes)
# ==============================================================================


def _encode_u16(value: int) -> bytes:
    """Encode an unsigned 16-bit integer as 2 bytes, little-endian."""
    return struct.pack("<H", value)


def _encode_u32(value: int) -> bytes:
    """Encode an unsigned 32-bit integer as 4 bytes, little-endian."""
    return struct.pack("<I", value)


def _encode_u64(value: int) -> bytes:
    """Encode an unsigned 64-bit integer as 8 bytes, little-endian."""
    return struct.pack("<Q", value)


def _decode_u16(data: bytes, offset: int) -> int:
    """Read an unsigned 16-bit integer from 2 bytes at offset, little-endian."""
    return struct.unpack("<H", data[offset : offset + 2])[0]


def _decode_u32(data: bytes, offset: int) -> int:
    """Read an unsigned 32-bit integer from 4 bytes at offset, little-endian."""
    return struct.unpack("<I", data[offset : offset + 4])[0]


def _decode_u64(data: bytes, offset: int) -> int:
    """Read an unsigned 64-bit integer from 8 bytes at offset, little-endian."""
    return struct.unpack("<Q", data[offset : offset + 8])[0]


# ==============================================================================
# Packing (AoT — qnn_preprocess.py calls these)
# ==============================================================================


def pack_schematic_payload(named_blobs: List[Tuple[str, bytes]]) -> bytes:
    """Pack a list of (name, data) pairs into a SCHEMATIC section payload.

    Each entry is serialized as:
      [name_length: u32] [name: utf-8 bytes] [data_length: u64] [data: raw bytes]

    Args:
        named_blobs: Each entry is (graph_name, schematic_binary_bytes).

    Returns:
        The raw payload bytes for a SCHEMATIC section.
    """
    parts: List[bytes] = []
    for name, data in named_blobs:
        name_encoded = name.encode("utf-8")
        parts.append(_encode_u32(len(name_encoded)))  # 4B: name byte length
        parts.append(name_encoded)                     # variable: UTF-8 name
        parts.append(_encode_u64(len(data)))           # 8B: blob byte length
        parts.append(data)                             # variable: raw blob
    return b"".join(parts)


def pack_tail(sections: List[Tuple[int, bytes]]) -> bytes:
    """Pack typed sections into a complete tail appendix (sections + footer).

    Each section is serialized as:
      [type_tag: u32] [payload_length: u64] [payload: raw bytes]

    The footer is appended last:
      [total_sections_length: u64] [section_count: u32] [version: u16] [magic: 8B]

    Args:
        sections: List of (type_tag, payload_bytes) pairs.

    Returns:
        Bytes to append directly after the QNN context binary.
        Returns empty bytes if ``sections`` is empty.
    """
    if not sections:
        return b""

    section_parts: List[bytes] = []
    for type_tag, payload in sections:
        section_parts.append(_encode_u32(type_tag))      # 4B: section type
        section_parts.append(_encode_u64(len(payload)))  # 8B: payload size
        section_parts.append(payload)                    # variable: payload
    section_bytes = b"".join(section_parts)

    # Fixed 22-byte footer — reader parses from the end of the buffer
    footer = (
        _encode_u64(len(section_bytes))  # 8B: total length of all sections above
        + _encode_u32(len(sections))     # 4B: number of sections
        + _encode_u16(TAIL_VERSION)      # 2B: protocol version
        + TAIL_MAGIC                     # 8B: magic identifier for detection
    )
    return section_bytes + footer


# ==============================================================================
# Unpacking (Host-side tooling — utils/utils.py calls these)
# ==============================================================================


def has_tail(processed_bytes: bytes) -> bool:
    """Check whether processed_bytes carries a QNN tail appendix.

    Detection: the last 8 bytes must equal TAIL_MAGIC and the buffer must be at
    least FOOTER_SIZE (22) bytes long.
    """
    return (
        len(processed_bytes) >= FOOTER_SIZE
        and processed_bytes[-8:] == TAIL_MAGIC
    )


def unpack_tail_sections(
    processed_bytes: bytes,
) -> Dict[int, List[bytes]]:
    """Unpack all tail sections from processed_bytes.

    Reads the fixed footer from the buffer's tail to locate and walk through
    each section.  Unknown section types are preserved in the output — callers
    simply ignore types they do not handle.

    Returns:
        Dict mapping section type tag → list of payloads (a type may appear
        more than once).

    Raises:
        ValueError: If the magic is present but the buffer is truncated or
            corrupted.
    """
    if not has_tail(processed_bytes):
        return {}

    # --- Parse the fixed 22-byte footer (at the very end) ---
    footer = processed_bytes[-FOOTER_SIZE:]
    total_sections_len = _decode_u64(footer, 0)   # bytes 0..7:  total sections length
    section_count = _decode_u32(footer, 8)         # bytes 8..11: section count
    # _version = _decode_u16(footer, 12)           # bytes 12..13: version (reserved)
    # bytes 14..21: magic (already verified by has_tail)

    # --- Validate that the buffer is large enough ---
    expected_min = FOOTER_SIZE + total_sections_len
    if len(processed_bytes) < expected_min:
        raise ValueError(
            f"QNN tail protocol: buffer too short. "
            f"Need {expected_min} bytes for tail, got {len(processed_bytes)}."
        )

    # --- Slice out the sections block (sits just before the footer) ---
    sections_block = processed_bytes[
        -(FOOTER_SIZE + total_sections_len) : -FOOTER_SIZE
    ]

    # --- Walk sections sequentially ---
    result: Dict[int, List[bytes]] = {}
    offset = 0
    for _ in range(section_count):
        if offset + 12 > len(sections_block):
            raise ValueError("QNN tail protocol: truncated section header.")

        type_tag = _decode_u32(sections_block, offset)       # 4B: section type
        payload_len = _decode_u64(sections_block, offset + 4)  # 8B: payload size
        offset += 12  # advance past the 12-byte section header

        if offset + payload_len > len(sections_block):
            raise ValueError(
                f"QNN tail protocol: section type=0x{type_tag:02X} claims "
                f"{payload_len} bytes but only {len(sections_block) - offset} "
                f"remain."
            )
        payload = sections_block[offset : offset + payload_len]
        offset += payload_len
        result.setdefault(type_tag, []).append(payload)

    return result


def unpack_schematic_payload(
    payload: bytes,
) -> List[Tuple[str, bytes]]:
    """Decode a SCHEMATIC section payload into (name, data) pairs.

    Walks the payload sequentially, reading each entry as:
      [name_length: u32] [name: utf-8 bytes] [data_length: u64] [data: raw bytes]

    Args:
        payload: Raw payload bytes from a SECTION_SCHEMATIC entry.

    Returns:
        List of (graph_name, schematic_binary) tuples.
    """
    entries: List[Tuple[str, bytes]] = []
    offset = 0
    while offset < len(payload):
        if offset + 4 > len(payload):
            raise ValueError("QNN tail SCHEMATIC: truncated name length.")
        name_len = _decode_u32(payload, offset)  # 4B: how many bytes the name uses
        offset += 4

        if offset + name_len > len(payload):
            raise ValueError("QNN tail SCHEMATIC: truncated name.")
        name = payload[offset : offset + name_len].decode("utf-8")
        offset += name_len

        if offset + 8 > len(payload):
            raise ValueError("QNN tail SCHEMATIC: truncated data length.")
        data_len = _decode_u64(payload, offset)  # 8B: how many bytes the blob uses
        offset += 8

        if offset + data_len > len(payload):
            raise ValueError("QNN tail SCHEMATIC: truncated data.")
        data = payload[offset : offset + data_len]
        offset += data_len

        entries.append((name, data))
    return entries
