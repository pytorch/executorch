# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import unittest

import torch
from executorch.backends.xnnpack.operators.node_visitor import NodeVisitor
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

    def test_convert_to_qc2w(self):
        # Two's complement crumbs, four per byte from the low bits up:
        # -2 = 0b10, -1 = 0b11, 0 = 0b00, 1 = 0b01. Rows run on across byte
        # boundaries and only the last byte is zero-padded.
        weights = torch.tensor([[-2, -1, 0, 1, 1], [0, -2, 1, -1, 0]], dtype=torch.int8)
        packed = NodeVisitor.convert_to_qc2w(weights)
        self.assertEqual(packed.dtype, torch.uint8)
        self.assertEqual(packed.tolist(), [0b01_00_11_10, 0b01_10_00_01, 0b00_00_00_11])

    def test_convert_to_qc2w_rejects_out_of_range(self):
        with self.assertRaises(AssertionError):
            NodeVisitor.convert_to_qc2w(torch.tensor([[0, 2]], dtype=torch.int8))
