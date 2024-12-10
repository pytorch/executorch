# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import typing
import unittest
from typing import List

from executorch.exir._serialize.data_serializer import (
    DataSerializer,
    SerializationInfo,
    TensorLayout,
)

from executorch.exir.schema import ScalarType
from executorch.extension.flat_tensor.serialize.flat_tensor_schema import TensorMetadata

from executorch.extension.flat_tensor.serialize.serialize import (
    _convert_to_flat_tensor,
    FlatTensorConfig,
    FlatTensorHeader,
    FlatTensorSerializer,
)

# Test artifacts
TEST_TENSOR_BUFFER = [b"\x11" * 4, b"\x22" * 32]
TEST_TENSOR_MAP = {
    "fqn1": 0,
    "fqn2": 0,
    "fqn3": 1,
}

TEST_TENSOR_LAYOUT = {
    "fqn1": TensorLayout(
        scalar_type=ScalarType.FLOAT,
        dim_sizes=[1, 1, 1],
        dim_order=typing.cast(List[bytes], [0, 1, 2]),
    ),
    "fqn2": TensorLayout(
        scalar_type=ScalarType.FLOAT,
        dim_sizes=[1, 1, 1],
        dim_order=typing.cast(List[bytes], [0, 1, 2]),
    ),
    "fqn3": TensorLayout(
        scalar_type=ScalarType.INT,
        dim_sizes=[2, 2, 2],
        dim_order=typing.cast(List[bytes], [0, 1]),
    ),
}


class TestSerialize(unittest.TestCase):
    def check_tensor_metadata(
        self, tensor_layout: TensorLayout, tensor_metadata: TensorMetadata
    ) -> None:
        self.assertEqual(tensor_layout.scalar_type, tensor_metadata.scalar_type)
        self.assertEqual(tensor_layout.dim_sizes, tensor_metadata.dim_sizes)
        self.assertEqual(tensor_layout.dim_order, tensor_metadata.dim_order)

    def test_serialize(self) -> None:
        config = FlatTensorConfig()
        serializer: DataSerializer = FlatTensorSerializer(config)

        data = bytes(
            serializer.serialize_tensors(
                SerializationInfo(
                    TEST_TENSOR_BUFFER, TEST_TENSOR_MAP, TEST_TENSOR_LAYOUT
                )
            )
        )

        # Check header.
        header = FlatTensorHeader.from_bytes(data[0 : FlatTensorHeader.EXPECTED_LENGTH])
        self.assertTrue(header.is_valid())

        self.assertEqual(header.flatbuffer_offset, 48)
        self.assertEqual(header.flatbuffer_size, 288)
        self.assertEqual(header.segment_base_offset, 336)
        self.assertEqual(header.data_size, 48)

        self.assertEqual(
            data[header.flatbuffer_offset + 4 : header.flatbuffer_offset + 8], b"FT01"
        )

        # Check flat tensor data.
        flat_tensor_bytes = data[
            header.flatbuffer_offset : header.flatbuffer_offset + header.flatbuffer_size
        ]

        flat_tensor = _convert_to_flat_tensor(flat_tensor_bytes)

        self.assertEqual(flat_tensor.version, 0)
        self.assertEqual(flat_tensor.tensor_alignment, config.tensor_alignment)

        tensors = flat_tensor.tensors
        self.assertEqual(len(tensors), 3)
        self.assertEqual(tensors[0].fully_qualified_name, "fqn1")
        self.check_tensor_metadata(TEST_TENSOR_LAYOUT["fqn1"], tensors[0])
        self.assertEqual(tensors[0].segment_index, 0)
        self.assertEqual(tensors[0].offset, 0)

        self.assertEqual(tensors[1].fully_qualified_name, "fqn2")
        self.check_tensor_metadata(TEST_TENSOR_LAYOUT["fqn2"], tensors[1])
        self.assertEqual(tensors[1].segment_index, 0)
        self.assertEqual(tensors[1].offset, 0)

        self.assertEqual(tensors[2].fully_qualified_name, "fqn3")
        self.check_tensor_metadata(TEST_TENSOR_LAYOUT["fqn3"], tensors[2])
        self.assertEqual(tensors[2].segment_index, 0)
        self.assertEqual(tensors[2].offset, config.tensor_alignment)

        segments = flat_tensor.segments
        self.assertEqual(len(segments), 1)
        self.assertEqual(segments[0].offset, 0)
        self.assertEqual(segments[0].size, config.tensor_alignment * 3)

        # Check segment data.
        segment_data = data[
            header.segment_base_offset : header.segment_base_offset + segments[0].size
        ]

        t0_start = 0
        t0_len = len(TEST_TENSOR_BUFFER[0])
        t0_end = config.tensor_alignment
        self.assertEqual(
            segment_data[t0_start : t0_start + t0_len], TEST_TENSOR_BUFFER[0]
        )
        padding = b"\x00" * (t0_end - t0_len)
        self.assertEqual(segment_data[t0_start + t0_len : t0_end], padding)

        t1_start = config.tensor_alignment
        t1_len = len(TEST_TENSOR_BUFFER[1])
        t1_end = config.tensor_alignment * 3
        self.assertEqual(
            segment_data[t1_start : t1_start + t1_len],
            TEST_TENSOR_BUFFER[1],
        )
        padding = b"\x00" * (t1_end - (t1_len + t1_start))
        self.assertEqual(segment_data[t1_start + t1_len : t1_end], padding)
