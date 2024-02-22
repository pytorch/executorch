# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import os
import random
import unittest
from typing import List, Tuple

from executorch.backends.xnnpack.serialization.xnnpack_graph_schema import (
    Buffer,
    XNNGraph,
)

from executorch.backends.xnnpack.serialization.xnnpack_graph_serialize import (
    _HEADER_BYTEORDER,
    serialize_xnnpack_binary,
    XNNHeader,
)


class TestSerialization(unittest.TestCase):
    def _generate_random_const_buffers(
        self, num_tensors: int
    ) -> Tuple[List[Buffer], List[int]]:
        """
        Helper function to generate `num_tensor` buffers of random sizes and random contents,
        we return a tuple of (list_of_buffers, list_of_mem_sizes),
        """
        buffers = []
        mem_sizes = []
        for _ in range(num_tensors):
            buffer_size = random.randint(1, 1000)
            buffer = bytearray(os.urandom(buffer_size))
            buffers.append(Buffer(storage=bytes(buffer)))
            mem_sizes.append(buffer_size)

        return buffers, mem_sizes

    def test_serialize_xnnpack_binary(self):
        xnn_graph = XNNGraph(
            version="0",
            xnodes=[],
            xvalues=[],
            num_externs=0,
            input_ids=[],
            output_ids=[],
            constant_buffer=[Buffer(storage=b"")],
            mem_buffer_sizes=[0],
            constant_data=[],
        )
        buffers, sizes = self._generate_random_const_buffers(5)
        xnn_graph.constant_buffer.extend(buffers)
        xnn_graph.mem_buffer_sizes.extend(sizes)
        buffers = xnn_graph.constant_buffer

        serialized_binary = serialize_xnnpack_binary(xnn_graph)
        offsets = xnn_graph.constant_data

        # Check header
        self.assertEqual(serialized_binary[0:4], b"\x00\x00\x00\x00")
        self.assertEqual(serialized_binary[XNNHeader.MAGIC_OFFSET], b"XH00")
        flatbuffer_offset_bytes = serialized_binary[XNNHeader.FLATBUFFER_OFFSET_OFFSET]
        constant_data_offset_bytes = serialized_binary[
            XNNHeader.CONSTANT_DATA_OFFSET_OFFSET
        ]

        # Check flatbuffer is at flatbuffer offset
        flatbuffer_offset = int.from_bytes(
            flatbuffer_offset_bytes, byteorder=_HEADER_BYTEORDER
        )
        # Flatbuffer magic should be in the same spot as the Header's magic
        self.assertEqual(
            serialized_binary[flatbuffer_offset:][XNNHeader.MAGIC_OFFSET], b"XN01"
        )

        # Check constant data
        # Check that constant buffers have been moved to constant data
        self.assertEqual(len(offsets), len(buffers))
        self.assertEqual(len(xnn_graph.constant_buffer), 0)

        constant_data_offset = int.from_bytes(
            constant_data_offset_bytes, byteorder=_HEADER_BYTEORDER
        )
        constant_data_payload = serialized_binary[constant_data_offset:]

        # We check that constant data indexes stored in the xnn_graph correctly index
        # into the correct buffer in the constant data section
        for idx in range(1, len(offsets)):
            offset = offsets[idx].offset
            size = offsets[idx].size

            constant_data_bytes = constant_data_payload[offset : offset + size]
            constant_buffer_bytes = buffers[idx].storage

            self.assertEqual(constant_data_bytes, constant_buffer_bytes)
