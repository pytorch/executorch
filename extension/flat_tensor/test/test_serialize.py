# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-unsafe

import dataclasses
import math
import unittest

from typing import Dict, List, Optional

import torch

from executorch.exir._serialize._cord import Cord
from executorch.exir._serialize._named_data_store import NamedDataStore

from executorch.exir._serialize.data_serializer import (
    DataEntry,
    DataPayload,
    DataSerializer,
)
from executorch.exir._serialize.padding import aligned_size

from executorch.exir.schema import ScalarType
from executorch.exir.tensor_layout import TensorLayout

from executorch.extension.flat_tensor.serialize.serialize import (
    _deserialize_to_flat_tensor,
    FlatTensorConfig,
    FlatTensorHeader,
    FlatTensorSerializer,
)

# The raw data stored in the serialized file segments.
TEST_BUFFER: List[bytes] = [b"\x11" * 4, b"\x22" * 32, b"\x33" * 17]

# Items serialized into FlatTensor.named_data.
# fqn1 and fqn2 are tensors that point to the same buffer index.
# fqn3 is a single tensor.
# key0 is a named_data entry.
TEST_NAMED_DATA = {
    "fqn1": DataEntry(
        buffer_index=0,
        alignment=0,
        tensor_layout=TensorLayout(
            scalar_type=ScalarType.FLOAT,
            sizes=[1, 1, 1],
            dim_order=[0, 1, 2],
        ),
    ),
    "fqn2": DataEntry(
        buffer_index=0,
        alignment=0,
        tensor_layout=TensorLayout(
            scalar_type=ScalarType.FLOAT,
            sizes=[1, 1, 1],
            dim_order=[0, 1, 2],
        ),
    ),
    "fqn3": DataEntry(
        buffer_index=1,
        alignment=0,
        tensor_layout=TensorLayout(
            scalar_type=ScalarType.INT,
            sizes=[2, 2, 2],
            dim_order=[0, 1],
        ),
    ),
    "key0": DataEntry(
        buffer_index=2,
        alignment=64,
        tensor_layout=None,
    ),
}

TEST_DATA_PAYLOAD = DataPayload(
    buffers=TEST_BUFFER,
    named_data=TEST_NAMED_DATA,
)


class TestSerialize(unittest.TestCase):
    # TODO(T211851359): improve test coverage.
    def check_tensor_layout(
        self, expected: Optional[TensorLayout], actual: Optional[TensorLayout]
    ) -> None:
        self.assertIsNotNone(expected)
        self.assertIsNotNone(actual)
        self.assertEqual(expected.scalar_type, actual.scalar_type)
        self.assertEqual(expected.sizes, actual.sizes)
        self.assertEqual(expected.dim_order, actual.dim_order)

    def _check_named_data_entries(
        self, reference: Dict[str, DataEntry], actual: Dict[str, DataEntry]
    ) -> None:
        self.assertEqual(reference.keys(), actual.keys())
        SKIP_FIELDS = {"alignment"}  # Fields to ignore in comparison.
        for key in reference.keys():
            ref_entry = reference[key]
            actual_entry = actual[key]
            for field in dataclasses.fields(ref_entry):
                if field.name not in SKIP_FIELDS:
                    self.assertEqual(
                        getattr(ref_entry, field.name),
                        getattr(actual_entry, field.name),
                        f"Named data record {key}.{field.name} does not match.",
                    )

    def test_serialize(self) -> None:
        config = FlatTensorConfig()
        serializer: DataSerializer = FlatTensorSerializer(config)
        serialized_data = bytes(serializer.serialize(TEST_DATA_PAYLOAD))

        # Ensure valid header.
        header = FlatTensorHeader.from_bytes(
            serialized_data[8 : FlatTensorHeader.EXPECTED_LENGTH + 8]
        )
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

        # Confirm the flatbuffer magic is present.
        self.assertEqual(
            serialized_data[4:8],
            b"FT01",
        )

        # Extract the flatbuffer.
        flat_tensor_bytes = serialized_data[
            0 : header.flatbuffer_offset + header.flatbuffer_size
        ]
        flat_tensor = _deserialize_to_flat_tensor(flat_tensor_bytes)

        # Check FlatTensor.version.
        self.assertEqual(flat_tensor.version, 0)

        # Check FlatTensor.named_data; key, segment_index, tensor_layout.
        named_data = flat_tensor.named_data
        self.assertEqual(len(named_data), 4)

        self.assertEqual(named_data[0].key, "fqn1")
        self.assertEqual(named_data[0].segment_index, 0)
        self.check_tensor_layout(
            TEST_NAMED_DATA["fqn1"].tensor_layout, named_data[0].tensor_layout
        )

        self.assertEqual(named_data[1].key, "fqn2")
        self.assertEqual(named_data[1].segment_index, 0)
        self.check_tensor_layout(
            TEST_NAMED_DATA["fqn2"].tensor_layout, named_data[1].tensor_layout
        )

        self.assertEqual(named_data[2].key, "fqn3")
        self.assertEqual(named_data[2].segment_index, 1)
        self.check_tensor_layout(
            TEST_NAMED_DATA["fqn3"].tensor_layout, named_data[2].tensor_layout
        )

        self.assertEqual(named_data[3].key, "key0")
        self.assertEqual(named_data[3].segment_index, 2)
        self.assertEqual(named_data[3].tensor_layout, None)

        # Check FlatTensor.segments.
        segments = flat_tensor.segments
        self.assertEqual(len(segments), 3)

        # Segment 0 contains fqn1, fqn2; 4 bytes, aligned to config.tensor_alignment.
        self.assertEqual(segments[0].offset, 0)
        self.assertEqual(segments[0].size, len(TEST_BUFFER[0]))

        # Segment 1 contains fqn3; 32 bytes, aligned to config.tensor_alignment.
        self.assertEqual(segments[1].offset, config.tensor_alignment)
        self.assertEqual(segments[1].size, len(TEST_BUFFER[1]))

        # Segment 2 contains key0; 17 bytes, aligned to 64.
        custom_alignment = math.lcm(
            config.segment_alignment, TEST_NAMED_DATA["key0"].alignment
        )
        self.assertEqual(
            segments[2].offset,
            aligned_size(config.tensor_alignment * 3, custom_alignment),
        )
        self.assertEqual(segments[2].size, len(TEST_BUFFER[2]))

        # Length of serialized_data matches segment_base_offset + segment_data_size.
        self.assertEqual(
            header.segment_base_offset + header.segment_data_size, len(serialized_data)
        )
        self.assertTrue(segments[0].size <= header.segment_data_size)

        # Check the contents of the segment. Expecting two tensors and one blob
        # from TEST_BUFFER = [b"\x11" * 4, b"\x22" * 32, b"\x33" * 17]
        segment_data = serialized_data[
            header.segment_base_offset : header.segment_base_offset
            + header.segment_data_size
        ]

        # Tensor: b"\x11" * 4
        self.assertEqual(
            segment_data[segments[0].offset : segments[0].offset + segments[0].size],
            TEST_BUFFER[0],
        )

        # Tensor: b"\x22" * 32
        padding = b"\x00" * (
            segments[1].offset - (segments[0].offset + segments[0].size)
        )
        self.assertEqual(
            segment_data[segments[0].offset + segments[0].size : segments[1].offset],
            padding,
        )
        self.assertEqual(
            segment_data[segments[1].offset : segments[1].offset + segments[1].size],
            TEST_BUFFER[1],
        )

        # Named data: b"\x33" * 17
        padding = b"\x00" * (
            segments[2].offset - (segments[1].offset + segments[1].size)
        )
        self.assertEqual(
            segment_data[segments[1].offset + segments[1].size : segments[2].offset],
            padding,
        )
        self.assertEqual(
            segment_data[segments[2].offset : segments[2].offset + segments[2].size],
            TEST_BUFFER[2],
        )

        self.assertEqual(segments[2].offset + segments[2].size, len(segment_data))

    def test_round_trip(self) -> None:
        # Serialize and then deserialize the test payload. Make sure it's reconstructed
        # properly.
        config = FlatTensorConfig()
        serializer: DataSerializer = FlatTensorSerializer(config)

        # Round trip the data.
        serialized_data = bytes(serializer.serialize(TEST_DATA_PAYLOAD))
        deserialized_payload = serializer.deserialize(Cord(serialized_data))

        # Validate the deserialized payload. Since alignment isn't serialized, we need to
        # do this somewhat manually.
        for i in range(len(deserialized_payload.buffers)):
            self.assertEqual(
                TEST_DATA_PAYLOAD.buffers[i],
                deserialized_payload.buffers[i],
                f"Buffer at index {i} does not match.",
            )

        self._check_named_data_entries(
            TEST_DATA_PAYLOAD.named_data, deserialized_payload.named_data
        )

    def test_deserialize_to_named_data_store_output(self) -> None:
        store = NamedDataStore()
        external_tag = "model"

        tensor_layout = TensorLayout(ScalarType.FLOAT, [1, 2], [0, 1])
        store.add_named_data(
            "key0",
            b"data0",
            alignment=1,
            external_tag=external_tag,
            tensor_layout=tensor_layout,
        )
        store.add_named_data(
            "key1",
            torch.tensor([[1, 2], [3, 4]], dtype=torch.float32),
            alignment=1,
            external_tag=external_tag,
        )

        output = store.get_named_data_store_output()
        self.assertEqual(len(output.buffers), 2)
        self.assertEqual(len(output.pte_data), 0)
        self.assertEqual(len(output.external_data), 1)
        self.assertEqual(len(output.external_data[external_tag]), 2)

        # Serialize and deserialize.
        config = FlatTensorConfig()
        serializer: DataSerializer = FlatTensorSerializer(config)
        data_payload = DataPayload(
            buffers=output.buffers, named_data=output.external_data[external_tag]
        )
        serialized_data = serializer.serialize(data_payload)

        output2 = serializer.deserialize_to_named_data_store_output(
            bytes(serialized_data), external_tag
        )

        self.assertEqual(output.buffers, output2.buffers)
        self.assertEqual(len(output.pte_data), 0)
        self.assertEqual(len(output2.pte_data), 0)

        self._check_named_data_entries(
            output.external_data[external_tag], output2.external_data[external_tag]
        )
