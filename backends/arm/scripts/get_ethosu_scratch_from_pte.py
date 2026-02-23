# Copyright 2026 Arm Limited and/or its affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

#!/usr/bin/env python3

import struct
from pathlib import Path

from executorch.devtools.bundled_program.serialize import (
    deserialize_from_flatbuffer_to_bundled_program,
)
from executorch.exir._serialize._program import deserialize_pte_binary


def iter_vela_blocks(blob: bytes):
    off = 0
    while off + 32 <= len(blob):
        name = blob[off : off + 16].split(b"\x00", 1)[0].decode("ascii", "ignore")
        off += 16
        # header: 4 x int32, first is size
        (size,) = struct.unpack("<i", blob[off : off + 4])
        off += 16
        data = blob[off : off + size]
        off += (size + 15) // 16 * 16  # pad to 16 bytes
        yield name, data
        if name == "vela_end_stream":
            break


def get_scratch_size_from_delegate(blob: bytes) -> int | None:
    for name, data in iter_vela_blocks(blob):
        if name == "scratch_size":
            return struct.unpack("<I", data[:4])[0]
    return None


def _extract_pte_from_bundle(pte_data: bytes) -> bytes:
    try:
        bundled = deserialize_from_flatbuffer_to_bundled_program(pte_data)
    except Exception:
        return pte_data
    return bundled.program if bundled.program else pte_data


def get_scratch_from_pte(pte_path: str) -> int | None:
    pte_data = _extract_pte_from_bundle(Path(pte_path).read_bytes())
    pte = deserialize_pte_binary(pte_data)
    program = pte.program

    sizes = []
    for plan in program.execution_plan:
        for delegate in plan.delegates:
            # delegate.id is the backend key (e.g. "EthosUBackend")
            idx = delegate.processed.index
            blob = program.backend_delegate_data[idx].data
            scratch = get_scratch_size_from_delegate(blob)
            if scratch is not None:
                sizes.append((delegate.id, scratch))

    if not sizes:
        return None

    for did, s in sizes:
        print(f"{did}: scratch_size={s} bytes")
    max_size = max(s for _, s in sizes)
    print(f"max_scratch_size={max_size} bytes")
    return max_size


if __name__ == "__main__":
    import sys

    if len(sys.argv) != 2:
        raise SystemExit(f"Usage: {sys.argv[0]} model.pte")
    get_scratch_from_pte(sys.argv[1])
