# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
# Copyright 2023-2026 Arm Limited and/or its affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.


import hashlib
import os
import struct
import tempfile

from collections.abc import Mapping
from dataclasses import dataclass
from typing import List

import numpy as np

try:
    from ethosu.vela import vela  # type: ignore

    has_vela = True
except ImportError:
    has_vela = False


_BLOCK_NAME_LENGTH = 16
_BLOCK_HEADER = struct.Struct(f"<{_BLOCK_NAME_LENGTH}sIB11s")
_BLOCK_ALIGNMENT = 16
_STREAM_HEADER = "vela_bin_stream"
_STREAM_FOOTER = "vela_end_stream"
_EXTERNAL_REFERENCE = 1
_RESERVED_BYTES = b"\x00" * 11


@dataclass(frozen=True)
class VelaExternalBlock:
    """Named-data payload emitted by Vela compilation."""

    key: str
    payload: bytes
    alignment: int
    placement: str


@dataclass(frozen=True)
class VelaCompileResult:
    """Binary stream and external payloads produced by Vela."""

    processed_bytes: bytes
    external_blocks: tuple[VelaExternalBlock, ...] = ()


def _as_int32(value, name: str) -> int:
    """Convert numpy scalars to signed int32 with a clear error on overflow."""
    arr = np.asarray(value)
    if np.issubdtype(arr.dtype, np.unsignedinteger):
        # Interpret unsigned values as signed (e.g., uint64 max -> -1).
        arr = arr.astype(np.int64)
    v = int(arr)
    if v < -(2**31) or v > 2**31 - 1:
        raise ValueError(f"{name} out of int32 range: {v}")
    return v


# Pack either input or output tensor block, compose the related arrays into
# per-io structs to simplify runtime use.
def vela_bin_pack_io(prefix, data):
    vela_input_shapes = data[prefix + "_shape"]
    # Vela input/output shape is fixed to 6D
    vela_io_shape_dims = 6

    ios = struct.pack("<i", len(vela_input_shapes))
    for i in range(len(vela_input_shapes)):
        io_shape = vela_input_shapes[i]
        io_elem_size = _as_int32(data[prefix + "_elem_size"][i], f"{prefix}_elem_size")
        io_offset = _as_int32(data[prefix + "_offset"][i], f"{prefix}_offset")
        io_region = _as_int32(data[prefix + "_region"][i], f"{prefix}_region")
        if len(io_shape) != vela_io_shape_dims:
            raise ValueError(
                f"Expected {vela_io_shape_dims}D shape, got {len(io_shape)}D"
            )
        inp_pad = io_shape.tolist()
        io_struct = struct.pack(
            "<iiiiiiiii", *inp_pad, io_elem_size, io_offset, io_region
        )
        ios += io_struct
    return ios


# Output via Vela to binary stream for ArmBackendEthosU
# WARNING: Do not change this without changing VelaBinStream.cpp as that
#          function consumes this format and the two need to align.
def vela_compile(
    tosa_flatbuffer: bytes,
    args: List[str],
    verbose: bool = False,
    intermediate_path: str | None = None,
    block_placements: Mapping[str, str] | None = None,
) -> VelaCompileResult:
    """Compile a TOSA graph to a binary stream for ArmBackendEthosU using
    Vela.
    """
    if not has_vela:
        raise RuntimeError(
            "ethos-u-vela pip package couldn't be imported. Make sure it's installed!"
        )
    resolved_block_placements: Mapping[str, str] = block_placements or {}

    def run(dir: str) -> VelaCompileResult:
        tosaname = "out.tosa"
        tosa_path = os.path.join(dir, tosaname)
        with open(tosa_path, "wb") as f:
            f.write(tosa_flatbuffer)

        # invoke vela
        output_dir = os.path.join(dir, "output")
        args.append(f"--output-dir={output_dir}")
        args.append(tosa_path)
        if verbose:
            args.append("--verbose-all")
        vela.main(" ".join(args).split(" "))

        np_path = os.path.join(dir, "output", "out_vela.npz")

        with np.load(np_path, allow_pickle=False) as data:
            # Construct our modified output_blocks with data in a form easily
            # digested on the device side
            bin_blocks = {_STREAM_HEADER: b""}

            # copy command data through unmodified
            bin_blocks["cmd_data"] = data["cmd_data"].tobytes()

            # copy weight data through unmodified
            bin_blocks["weight_data"] = data["weight_data"].tobytes()

            # Add a block for scratch, inputs and outputs;  scratch shape is a 1 element
            # array giving us size in bytes so extract this and add a block of 0's.
            # Currently we preallocated this on the host to provide SRAM for computation.
            if not isinstance(data["scratch_shape"][0], np.int64):
                raise RuntimeError("Expected scratch to be int64")
            block_length = int(data["scratch_shape"][0])
            bin_blocks["scratch_size"] = struct.pack("<I", block_length)

            # Capture inputs and outputs
            bin_blocks["inputs"] = vela_bin_pack_io("input", data)
            bin_blocks["outputs"] = vela_bin_pack_io("output", data)

            bin_blocks[_STREAM_FOOTER] = b""

            unknown_blocks = resolved_block_placements.keys() - bin_blocks.keys()
            if unknown_blocks:
                raise ValueError(
                    "External Vela block placements reference blocks that were not "
                    f"emitted: {sorted(unknown_blocks)}"
                )

            # Emit the NPZ regions as:
            #  - 16 byte block name null terminated string (padded to 16 if name shorter)
            #  - 4 bytes of int32 block length, 1 byte external flag, and 11 reserved bytes
            #  - block data (padded to 16 byte alignment at end)
            # Repeat for all blocks
            blocks = b""
            external_blocks: list[VelaExternalBlock] = []
            for key in bin_blocks.keys():
                block_name = bytes(key, "utf8")[:15]
                block_name = block_name + b"\x00" * (16 - len(block_name))
                block_data = bin_blocks[key]
                placement = resolved_block_placements.get(key)
                external_reference = 0
                if placement is not None:
                    digest = hashlib.sha256(
                        placement.encode("ascii") + b"\0" + block_data
                    ).hexdigest()
                    external_blocks.append(
                        VelaExternalBlock(
                            key=digest,
                            payload=block_data,
                            alignment=_BLOCK_ALIGNMENT,
                            placement=placement,
                        )
                    )
                    block_data = digest.encode("ascii")
                    external_reference = _EXTERNAL_REFERENCE

                # We need the acual unpadded block lengths for hw setup
                block_header = _BLOCK_HEADER.pack(
                    block_name,
                    len(block_data),
                    external_reference,
                    _RESERVED_BYTES,
                )

                # Pad block data to multiple of 16 bytes
                block_data = block_data + b"\x00" * (15 - (len(block_data) - 1) % 16)

                block = block_header + block_data
                blocks = blocks + block

            return VelaCompileResult(blocks, tuple(external_blocks))

    if intermediate_path is not None:
        return run(intermediate_path)
    else:
        with tempfile.TemporaryDirectory() as tmpdir:
            return run(tmpdir)
