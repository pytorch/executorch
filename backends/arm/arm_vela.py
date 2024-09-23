# Copyright 2023-2024 Arm Limited and/or its affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-unsafe

import os
import struct
import tempfile

from typing import List

import numpy as np
from ethosu.vela import vela


# Pack either input or output tensor block, compose the related arrays into
# per-io structs to simplify runtime use.
def vela_bin_pack_io(prefix, data):
    ios = struct.pack("<i", len(data[prefix + "_shape"]))
    for i in range(len(data[prefix + "_shape"])):
        io_shape = data[prefix + "_shape"][i]
        io_elem_size = data[prefix + "_elem_size"][i]
        io_offset = data[prefix + "_offset"][i]
        io_region = data[prefix + "_region"][i]
        assert len(io_shape) <= 4
        inp_pad = io_shape.tolist() + [0] * (4 - len(io_shape))
        io_struct = struct.pack(
            "<iiiiiii", *inp_pad, io_elem_size, io_offset, io_region
        )
        ios += io_struct
    return ios


# Output via Vela to binary stream for ArmBackendEthosU
# WARNING: Do not change this without changing VelaBinStream.cpp as that
#          function consumes this format and the two need to align.
def vela_compile(tosa_graph, args: List[str]):
    with tempfile.TemporaryDirectory() as tmpdir:
        tosaname = "out.tosa"
        flatbuffer = tosa_graph.serialize()
        tosa_path = os.path.join(tmpdir, tosaname)
        with open(tosa_path, "wb") as f:
            f.write(flatbuffer)

        # invoke vela
        output_dir = os.path.join(tmpdir, "output")
        args.append(f"--output-dir={output_dir}")
        args.append(tosa_path)
        vela.main(" ".join(args).split(" "))

        if any("ethos-u85" in arg for arg in args) or any(
            "debug-force-regor" in arg for arg in args
        ):
            np_path = os.path.join(tmpdir, "output", "out_vela.npz")
        else:
            np_path = os.path.join(tmpdir, "output", "out_sg0_vela.npz")
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
            bin_blocks["scratch_data"] = b"\x00" * block_length

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
                block_length = struct.pack("<iiii", len(bin_blocks[key]), 0, 0, 0)

                # Pad block data to multiple of 16 bytes
                block_data = bin_blocks[key]
                block_data = block_data + b"\x00" * (15 - (len(block_data) - 1) % 16)

                block = block_name + block_length + block_data
                blocks = blocks + block

        return blocks
