# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-unsafe

import dataclasses
import io
import math
import os
import tempfile
import unittest

from typing import Dict, List, Optional
from unittest import mock

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
    _FLAT_TENSOR_VERSION,
    _FLATBUFFER_ALIGNMENT,
    FlatTensorConfig,
    FlatTensorHeader,
    FlatTensorSerializer,
    load_ptd,
    save_ptd,
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

    def _serialize_with_alignment(self, config: FlatTensorConfig) -> None:
        serializer: DataSerializer = FlatTensorSerializer(config)
        serialized_data = bytes(serializer.serialize(TEST_DATA_PAYLOAD))

        # Ensure valid header.
        header = FlatTensorHeader.from_bytes(
            serialized_data[8 : FlatTensorHeader.EXPECTED_LENGTH + 8]
        )
        self.assertTrue(header.is_valid())

        # Flatbuffer is non-empty.
        self.assertTrue(header.flatbuffer_size > 0)

        # Align the flatbuffer to _FLATBUFFER_ALIGNMENT.
        self.assertEqual(
            header.flatbuffer_offset,
            aligned_size(FlatTensorHeader.EXPECTED_LENGTH, _FLATBUFFER_ALIGNMENT),
        )

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

        # Segment 0 contains fqn1, fqn2; 4 bytes, aligned to config.segment_alignment.
        self.assertEqual(segments[0].offset, 0)
        self.assertEqual(segments[0].size, len(TEST_BUFFER[0]))

        # Segment 1 contains fqn3; 32 bytes, aligned to config.segment_alignment.
        self.assertEqual(segments[1].offset, config.segment_alignment)
        self.assertEqual(segments[1].size, len(TEST_BUFFER[1]))

        # Segment 2 contains key0; 17 bytes, aligned to 64.
        custom_alignment = math.lcm(
            config.segment_alignment, TEST_NAMED_DATA["key0"].alignment
        )
        self.assertEqual(
            segments[2].offset,
            aligned_size(config.segment_alignment * 2, custom_alignment),
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

    def test_serialize_default_alignment(self) -> None:
        config = FlatTensorConfig()
        self._serialize_with_alignment(config)

    def test_serialize_align_4096(self) -> None:
        config = FlatTensorConfig(segment_alignment=4096)
        self._serialize_with_alignment(config)

    def test_serialize_align_1024(self) -> None:
        config = FlatTensorConfig(segment_alignment=1024)
        self._serialize_with_alignment(config)

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

    def test_deserialize_refuses_newer_version(self) -> None:
        # A file whose version is newer than this reader understands must be
        # refused, rather than silently misread. Serialize with a bumped
        # version, then deserialize with the real (lower) reader version.
        config = FlatTensorConfig()
        serializer: DataSerializer = FlatTensorSerializer(config)

        with mock.patch(
            "executorch.extension.flat_tensor.serialize.serialize._FLAT_TENSOR_VERSION",
            _FLAT_TENSOR_VERSION + 1,
        ):
            serialized_data = bytes(serializer.serialize(TEST_DATA_PAYLOAD))

        with self.assertRaises(NotImplementedError):
            serializer.deserialize(Cord(serialized_data))

    def test_deserialize_accepts_older_version(self) -> None:
        # Back-compat: a file older than the reader must still load, and load
        # correctly. Serialize at the current version, then deserialize with the
        # reader's version bumped higher, so the file looks older than the
        # reader. This is what `>` buys over the previous `!=`, which rejected
        # any mismatch including older.
        config = FlatTensorConfig()
        serializer: DataSerializer = FlatTensorSerializer(config)
        serialized_data = bytes(serializer.serialize(TEST_DATA_PAYLOAD))

        with mock.patch(
            "executorch.extension.flat_tensor.serialize.serialize._FLAT_TENSOR_VERSION",
            _FLAT_TENSOR_VERSION + 1,
        ):
            deserialized_payload = serializer.deserialize(Cord(serialized_data))

        # The older file must not just load without raising; it must round-trip
        # intact, so a regression that loads but drops data can't pass silently.
        for i in range(len(deserialized_payload.buffers)):
            self.assertEqual(
                TEST_DATA_PAYLOAD.buffers[i],
                deserialized_payload.buffers[i],
                f"Buffer at index {i} does not match.",
            )
        self._check_named_data_entries(
            TEST_DATA_PAYLOAD.named_data, deserialized_payload.named_data
        )

    def test_file_backed_data_matches_bytes(self) -> None:
        from executorch.exir._serialize._cord import FileBackedData

        with tempfile.TemporaryDirectory() as directory:
            path = os.path.join(directory, "weights.bin")
            with open(path, "wb") as f:
                f.write(TEST_BUFFER[0])
            file_data = FileBackedData.move_from(path)
            payload = DataPayload(
                buffers=[file_data],
                named_data={"weight": DataEntry(0, 16, None)},
            )
            bytes_payload = DataPayload(
                buffers=[TEST_BUFFER[0]],
                named_data={"weight": DataEntry(0, 16, None)},
            )
            serializer = FlatTensorSerializer(FlatTensorConfig())
            self.assertEqual(
                bytes(serializer.serialize(bytes_payload)),
                bytes(serializer.serialize(payload)),
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


class TestSaveLoadPtd(unittest.TestCase):
    def _round_trip(self, tensors: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        buffer = io.BytesIO()
        save_ptd(buffer, tensors)
        buffer.seek(0)
        return load_ptd(buffer)

    def _check_tensors_match(
        self, expected: Dict[str, torch.Tensor], actual: Dict[str, torch.Tensor]
    ) -> None:
        self.assertEqual(expected.keys(), actual.keys())
        for key, expected_tensor in expected.items():
            actual_tensor = actual[key]
            self.assertEqual(actual_tensor.dtype, expected_tensor.dtype, key)
            self.assertEqual(actual_tensor.shape, expected_tensor.shape, key)
            self.assertEqual(actual_tensor.stride(), expected_tensor.stride(), key)
            self.assertTrue(torch.equal(actual_tensor, expected_tensor), key)

    def test_round_trip_dtypes(self) -> None:
        tensors = {
            "float32": torch.rand(2, 3),
            "float64": torch.rand(4, dtype=torch.float64),
            "float16": torch.rand(2, 2).to(torch.float16),
            "bfloat16": torch.rand(3, 3).to(torch.bfloat16),
            "int64": torch.arange(6, dtype=torch.int64).reshape(3, 2),
            "int8": torch.tensor([-1, 0, 1], dtype=torch.int8),
            "uint8": torch.tensor([0, 128, 255], dtype=torch.uint8),
            "bool": torch.tensor([True, False, True]),
        }
        self._check_tensors_match(tensors, self._round_trip(tensors))

    def test_round_trip_shapes(self) -> None:
        tensors = {
            "scalar": torch.tensor(3.5),
            "empty": torch.empty(0),
            "empty_with_shape": torch.empty(0, 3),
            "high_rank": torch.rand(2, 1, 3, 1, 2),
        }
        self._check_tensors_match(tensors, self._round_trip(tensors))

    def test_round_trip_preserves_channels_last(self) -> None:
        tensors = {
            "channels_last": torch.rand(2, 3, 4, 5).to(
                memory_format=torch.channels_last
            )
        }
        loaded = self._round_trip(tensors)
        self._check_tensors_match(tensors, loaded)
        self.assertTrue(
            loaded["channels_last"].is_contiguous(memory_format=torch.channels_last)
        )

    def test_round_trip_to_file(self) -> None:
        tensors = {"weight": torch.rand(3, 4), "bias": torch.rand(4)}
        with tempfile.TemporaryDirectory() as directory:
            path = os.path.join(directory, "params.ptd")
            save_ptd(path, tensors)
            self._check_tensors_match(tensors, load_ptd(path))

    def test_round_trip_empty(self) -> None:
        self.assertEqual(self._round_trip({}), {})

    def test_save_detaches_parameters(self) -> None:
        # named_parameters() and state_dict(keep_vars=True) hand back
        # Parameters that require grad, and .numpy() raises on one of those
        # unless it is detached first.
        parameter = torch.nn.Parameter(torch.rand(2, 2))
        loaded = self._round_trip({"weight": parameter})
        self._check_tensors_match({"weight": parameter.detach()}, loaded)
        self.assertFalse(loaded["weight"].requires_grad)

    def test_tensor_alignment(self) -> None:
        tensor_alignment = 256
        buffer = io.BytesIO()
        save_ptd(
            buffer,
            {"a": torch.rand(3), "b": torch.rand(5)},
            tensor_alignment=tensor_alignment,
        )
        serialized_data = buffer.getvalue()

        header = FlatTensorHeader.from_bytes(
            serialized_data[8 : FlatTensorHeader.EXPECTED_LENGTH + 8]
        )
        self.assertTrue(header.is_valid())
        self.assertEqual(header.segment_base_offset % tensor_alignment, 0)

        flat_tensor = _deserialize_to_flat_tensor(
            serialized_data[0 : header.flatbuffer_offset + header.flatbuffer_size]
        )
        self.assertEqual(len(flat_tensor.segments), 2)
        for segment in flat_tensor.segments:
            self.assertEqual(segment.offset % tensor_alignment, 0)

    def test_save_rejects_invalid_alignment(self) -> None:
        with self.assertRaises(ValueError):
            save_ptd(io.BytesIO(), {"a": torch.rand(3)}, tensor_alignment=0)

    def test_save_rejects_non_contiguous(self) -> None:
        with self.assertRaises(ValueError):
            save_ptd(io.BytesIO(), {"a": torch.rand(4, 4).t()})

    def test_save_rejects_view_into_larger_storage(self) -> None:
        # A channels-last slice is contiguous in its memory format, so nothing
        # upstream rejects it, but its storage holds the whole base tensor.
        base = torch.rand(4, 3, 4, 5).to(memory_format=torch.channels_last)
        with self.assertRaises(ValueError):
            save_ptd(io.BytesIO(), {"a": base[2:4]})

        # .clone() preserves the memory format, so the copy that owns its
        # storage still round trips as channels-last.
        tensors = {"a": base[2:4].clone()}
        self._check_tensors_match(tensors, self._round_trip(tensors))

    def test_tied_tensors_share_one_buffer(self) -> None:
        # One tensor under two keys, as with an embedding tied to the output
        # projection, is stored once rather than copied per key.
        shared = torch.rand(32, 4)
        tied = io.BytesIO()
        copied = io.BytesIO()
        save_ptd(tied, {"embedding": shared, "output": shared})
        save_ptd(copied, {"embedding": shared, "output": shared.clone()})
        self.assertLess(len(tied.getvalue()), len(copied.getvalue()))

        tied.seek(0)
        self._check_tensors_match(
            {"embedding": shared, "output": shared}, load_ptd(tied)
        )

    def test_save_rejects_non_tensor(self) -> None:
        with self.assertRaises(TypeError):
            save_ptd(io.BytesIO(), {"a": [1.0, 2.0]})

    def test_load_skips_opaque_blobs(self) -> None:
        # Backends store opaque data alongside tensors in the same file, and
        # only the tensors belong in a tensor dict.
        tensor = torch.rand(2, 2)
        store = NamedDataStore()
        store.add_named_data("blob", b"opaque", external_tag="model")
        store.add_named_data("tensor", tensor, external_tag="model")
        output = store.get_named_data_store_output()

        serialized_data = FlatTensorSerializer(FlatTensorConfig()).serialize(
            DataPayload(
                buffers=output.buffers, named_data=output.external_data["model"]
            )
        )

        loaded = load_ptd(io.BytesIO(bytes(serialized_data)))
        self._check_tensors_match({"tensor": tensor}, loaded)

    def test_loaded_tensors_do_not_alias(self) -> None:
        # Equal data may share one segment in the file; the loaded tensors must
        # still be independent, so writing to one is not seen through the other.
        tensors = {"a": torch.zeros(4), "b": torch.zeros(4)}
        loaded = self._round_trip(tensors)
        loaded["a"][0] = 1.0
        self.assertEqual(loaded["b"][0].item(), 0.0)
