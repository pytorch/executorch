# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import unittest

from executorch.backends.xnnpack.serialization.xnnpack_graph_schema import (
    ConstantDataOffset,
    XNNGraph,
)

from executorch.backends.xnnpack.serialization.xnnpack_graph_serialize import (
    _HEADER_BYTEORDER,
    serialize_xnnpack_binary,
    XNNHeader,
)


class TestSerialization(unittest.TestCase):
    def test_serialize_xnnpack_binary(self):
        xnn_graph = XNNGraph(
            version="0",
            xnodes=[],
            xvalues=[],
            num_externs=0,
            input_ids=[],
            output_ids=[],
            constant_data=[ConstantDataOffset(0, 0)],
        )

        constant_data_bytes = b"\x00" * 24
        serialized_binary = serialize_xnnpack_binary(
            xnn_graph, bytearray(constant_data_bytes)
        )

        # Check header
        self.assertEqual(serialized_binary[0:4], b"\x00\x00\x00\x00")
        self.assertEqual(serialized_binary[XNNHeader.MAGIC_OFFSET], b"XH00")
        flatbuffer_offset_bytes = serialized_binary[XNNHeader.FLATBUFFER_OFFSET_OFFSET]

        # Check flatbuffer is at flatbuffer offset
        flatbuffer_offset = int.from_bytes(
            flatbuffer_offset_bytes, byteorder=_HEADER_BYTEORDER
        )
        # Flatbuffer magic should be in the same spot as the Header's magic
        self.assertEqual(
            serialized_binary[flatbuffer_offset:][XNNHeader.MAGIC_OFFSET], b"XN01"
        )
