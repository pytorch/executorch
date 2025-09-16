# Copyright 2023-2025 Arm Limited and/or its affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-unsafe

import os
import struct
import tempfile

from typing import List

import numpy as np

try:
    from ethosu.vela import vela  # type: ignore

    has_vela = True
except ImportError:
    has_vela = False


# Pack either input or output tensor block, compose the related arrays into
# per-io structs to simplify runtime use.
def vela_bin_pack_io(prefix, data):
    vela_input_shapes = data[prefix + "_shape"]
    # Vela input/output shape is fixed to 6D
    vela_io_shape_dims = 6

    ios = struct.pack("<i", len(vela_input_shapes))
    for i in range(len(vela_input_shapes)):
        io_shape = vela_input_shapes[i]
        io_elem_size = data[prefix + "_elem_size"][i]
        io_offset = data[prefix + "_offset"][i]
        io_region = data[prefix + "_region"][i]
        assert len(io_shape) == vela_io_shape_dims
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
):
    """
    Compile a TOSA graph to a binary stream for ArmBackendEthosU using Vela.
    """
    if not has_vela:
        raise RuntimeError(
            "ethos-u-vela pip package couldn't be imported. Make sure it's installed!"
        )

    def run(dir: str) -> bytes:
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

        if any("ethos-u85" in arg for arg in args) or any(
            "debug-force-regor" in arg for arg in args
        ):
            np_path = os.path.join(dir, "output", "out_vela.npz")
        else:
            np_path = os.path.join(dir, "output", "out_sg0_vela.npz")

        blocks = b""
        with np.load(np_path, allow_pickle=False) as data:
            # Construct our modified output_blocks with data in a form easily
            # digested on the device side
            bin_blocks = {"vela_bin_stream": b""}

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

            bin_blocks["vela_end_stream"] = b""

            # Emit the NPZ regions as:
            #  - 16 byte block name null terminated string (padded to 16 if name shorter)
            #  - 4 bytes of int32 block length and 12 bytes of 0's
            #  - block data (padded to 16 byte alignment at end)
            # Repeat for all blocks
            for key in bin_blocks.keys():
                block_name = bytes(key, "utf8")[:15]
                block_name = block_name + b"\x00" * (16 - len(block_name))

                # We need the acual unpadded block lengths for hw setup
                block_length_bytes = struct.pack("<iiii", len(bin_blocks[key]), 0, 0, 0)

                # Pad block data to multiple of 16 bytes
                block_data = bin_blocks[key]
                block_data = block_data + b"\x00" * (15 - (len(block_data) - 1) % 16)

                block = block_name + block_length_bytes + block_data
                blocks = blocks + block

        return blocks

    if intermediate_path is not None:
        return run(intermediate_path)
    else:
        with tempfile.TemporaryDirectory() as tmpdir:
            return run(tmpdir)
