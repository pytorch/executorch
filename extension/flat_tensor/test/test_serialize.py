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

from executorch.extension.flat_tensor.serialize.serialize import (
    FlatTensorHeader,
    FlatTensorSerializer,
)

# Test artifacts
TEST_TENSOR_BUFFER = [b"tensor"]
TEST_TENSOR_MAP = {
    "fqn1": 0,
    "fqn2": 0,
}

TEST_TENSOR_LAYOUT = {
    "fqn1": TensorLayout(
        scalar_type=ScalarType.FLOAT,
        sizes=[1, 1, 1],
        dim_order=typing.cast(List[bytes], [0, 1, 2]),
    ),
    "fqn2": TensorLayout(
        scalar_type=ScalarType.FLOAT,
        sizes=[1, 1, 1],
        dim_order=typing.cast(List[bytes], [0, 1, 2]),
    ),
}


class TestSerialize(unittest.TestCase):
    def test_serialize(self) -> None:
        serializer: DataSerializer = FlatTensorSerializer()

        data = bytes(
            serializer.serialize_tensors(
                SerializationInfo(
                    TEST_TENSOR_BUFFER, TEST_TENSOR_MAP, TEST_TENSOR_LAYOUT
                )
            )
        )

        header = FlatTensorHeader.from_bytes(data[0 : FlatTensorHeader.EXPECTED_LENGTH])
        self.assertTrue(header.is_valid())

        self.assertEqual(header.flatbuffer_offset, 48)
        self.assertEqual(header.flatbuffer_size, 200)
        self.assertEqual(header.segment_base_offset, 256)
        self.assertEqual(header.data_size, 16)

        self.assertEqual(
            data[header.flatbuffer_offset + 4 : header.flatbuffer_offset + 8], b"FT01"
        )
