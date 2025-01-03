# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import unittest

from executorch.exir._serialize.data_serializer import (
    DataPayload,
    DataSerializer,
    TensorEntry,
    TensorLayout,
)

from executorch.exir._serialize.padding import aligned_size

from executorch.exir.schema import ScalarType

from executorch.extension.flat_tensor.serialize.serialize import (
    FlatTensorConfig,
    FlatTensorHeader,
    FlatTensorSerializer,
)

# Test artifacts.
TEST_TENSOR_BUFFER = [b"tensor"]
TEST_TENSOR_MAP = {
    "fqn1": TensorEntry(
        buffer_index=0,
        layout=TensorLayout(
            scalar_type=ScalarType.FLOAT,
            sizes=[1, 1, 1],
            dim_order=[0, 1, 2],
        ),
    ),
    "fqn2": TensorEntry(
        buffer_index=0,
        layout=TensorLayout(
            scalar_type=ScalarType.FLOAT,
            sizes=[1, 1, 1],
            dim_order=[0, 1, 2],
        ),
    ),
}
TEST_DATA_PAYLOAD = DataPayload(
    buffers=TEST_TENSOR_BUFFER,
    fqn_to_tensor=TEST_TENSOR_MAP,
)


class TestSerialize(unittest.TestCase):
    def test_serialize(self) -> None:
        config = FlatTensorConfig()
        serializer: DataSerializer = FlatTensorSerializer(config)

        data = bytes(serializer.serialize(TEST_DATA_PAYLOAD))

        header = FlatTensorHeader.from_bytes(data[0 : FlatTensorHeader.EXPECTED_LENGTH])
        self.assertTrue(header.is_valid())

        # Header is aligned to config.segment_alignment, which is where the flatbuffer starts.
        self.assertEqual(
            header.flatbuffer_offset,
            aligned_size(FlatTensorHeader.EXPECTED_LENGTH, config.segment_alignment),
        )

        # Flatbuffer is non-empty.
        self.assertTrue(header.flatbuffer_size > 0)

        # Segment base offset is aligned to config.segment_alignment.
        expected_segment_base_offset = aligned_size(
            header.flatbuffer_offset + header.flatbuffer_size, config.segment_alignment
        )
        self.assertTrue(header.segment_base_offset, expected_segment_base_offset)

        # TEST_TENSOR_BUFFER is aligned to config.segment_alignment.
        self.assertEqual(header.segment_data_size, config.segment_alignment)

        # Confirm the flatbuffer magic is present.
        self.assertEqual(
            data[header.flatbuffer_offset + 4 : header.flatbuffer_offset + 8], b"FT01"
        )
