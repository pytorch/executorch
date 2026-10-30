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
# Vela region of persistent state with --separate-persistent-region. Must match
# PERSISTENT_REGION in regor/compiler/raw_writer.cpp.
_PERSISTENT_REGION = 5


def _persistent_region_size(data: Mapping) -> int:
    """Size in bytes of the persistent region (base address 5).

    The highest (offset + byte size) over the region-5 variables, rounded up to
    the 16-byte block alignment. 0 when the model has no persistent state, in
    which case Vela omits the variable_* keys entirely.

    """
    keys = (
        "variable_offset",
        "variable_shape",
        "variable_elem_size",
        "variable_region",
    )
    if any(k not in data for k in keys):
        return 0
    size = 0
    for offset, shape, elem_size, region in zip(
        data["variable_offset"],
        data["variable_shape"],
        data["variable_elem_size"],
        data["variable_region"],
    ):
        if int(region) != _PERSISTENT_REGION:
            continue
        nbytes = int(np.prod(np.asarray(shape))) * int(elem_size)
        size = max(size, int(offset) + nbytes)
    return (size + _BLOCK_ALIGNMENT - 1) & ~(_BLOCK_ALIGNMENT - 1)


def _persistent_init_image(data: Mapping, persistent_size: int) -> bytes | None:
    """Initial contents of the persistent region, or None when it is all zero.

    Vela gives each variable its own initial data: variable_data[i], to be
    copied to variable_offset[i], empty for variables without initial data.
    Empty entries are skipped rather than zero-filled, as a variable's read and
    write tensors share an address and only one of them carries the data.
    np.savez stores the entries as fixed-width byte strings, which drop
    trailing zero bytes on load, so each is padded back to
    variable_data_size[i]. None when there is no initial data or it is all
    zero, in which case the runtime's own zeroing is equivalent and the block
    is left out.

    """
    if "variable_data" not in data or "variable_data_size" not in data:
        return None
    image = bytearray(persistent_size)
    for offset, region, entry, nbytes in zip(
        data["variable_offset"],
        data["variable_region"],
        data["variable_data"],
        data["variable_data_size"],
    ):
        nbytes = int(nbytes)
        if nbytes == 0:
            continue
        if int(region) != _PERSISTENT_REGION:
            raise ValueError(
                f"Variable at offset {int(offset)} has initial data in region "
                f"{int(region)}; only the persistent region (region "
                f"{_PERSISTENT_REGION}) is initialised"
            )
        start = int(offset)
        if start + nbytes > persistent_size:
            raise ValueError(
                f"Variable initial data at {start}+{nbytes} exceeds the "
                f"{persistent_size} byte persistent region"
            )
        image[start : start + nbytes] = bytes(entry).ljust(nbytes, b"\x00")
    if not any(image):
        return None
    return bytes(image)


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
    max_scratch_size: int | None = None,
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
            if max_scratch_size is not None and block_length > max_scratch_size:
                raise RuntimeError(
                    f"Ethos-U delegate scratch arena requires {block_length} bytes, "
                    f"exceeding the configured capacity of {max_scratch_size} bytes "
                    f"by {block_length - max_scratch_size} bytes. "
                    "Reduce the model's memory requirements or target a platform "
                    "with sufficient scratch memory. See Vela's documentation "
                    "for memory-mode and arena-cache-size options: "
                    "https://gitlab.arm.com/artificial-intelligence/ethos-u/"
                    "ethos-u-vela/-/blob/main/OPTIONS.md"
                )
            bin_blocks["scratch_size"] = struct.pack("<I", block_length)

            # Delegate-owned streaming (persistent/variable) state lives in its
            # own region (region 5), with 0-based offsets independent of scratch
            # and weights. The runtime allocates a buffer of this size for base
            # address 5. Assumes a single model owns the persistent region.
            #
            # Only emitted when the model actually has persistent state. The
            # block name is unknown to older runtimes, and VelaBinStream rejects
            # unrecognised names outright rather than skipping them, so emitting
            # it unconditionally would break every model on any runtime that
            # predates it. A model without streaming state must keep producing
            # the byte-identical blob it produced before.
            persistent_size = _persistent_region_size(data)
            if persistent_size:
                bin_blocks["persistent_size"] = struct.pack("<I", persistent_size)

                # Initial contents of the region, when the state does not start
                # from the all-zero bit pattern. Vela gives each variable its own
                # initial data; it is laid out here at the variables' offsets as
                # one image, so the runtime installs it with a single copy and
                # never needs to know the layout.
                #
                # Zeroing is only the right initial value when every state's
                # quantized zero point is 0. Under asymmetric quantization a
                # real-valued zero encodes as the zero point, not as 0, and a
                # state may legitimately start from a non-zero value in any
                # case. Both are just "the buffer's contents", which is what
                # this carries.
                #
                # Skipped when the image is all zeroes: the runtime already
                # zeroes the region, so the block would cost blob size to say
                # nothing.
                persistent_init = _persistent_init_image(data, persistent_size)
                if persistent_init is not None:
                    bin_blocks["persistent_init"] = persistent_init

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
