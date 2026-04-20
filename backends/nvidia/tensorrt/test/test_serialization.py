# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Tests for TensorRT blob serialization format."""

import unittest


class SerializationTest(unittest.TestCase):
    """Tests for TensorRT blob serialization format."""

    def test_serialize_and_deserialize_header(self) -> None:
        """Test that serialize/deserialize round-trips correctly."""
        from executorch.backends.nvidia.tensorrt.serialization import (
            deserialize_blob_header,
            serialize_blob,
            TENSORRT_MAGIC,
        )

        engine_bytes = b"fake_engine_data_12345"
        blob = serialize_blob(engine_bytes)
        header = deserialize_blob_header(blob)

        self.assertIsNotNone(header)
        self.assertEqual(header.magic, TENSORRT_MAGIC)
        self.assertTrue(header.is_valid())
        self.assertEqual(header.engine_size, len(engine_bytes))

    def test_header_size(self) -> None:
        """Test that header size constant is correct."""
        from executorch.backends.nvidia.tensorrt.serialization import (
            deserialize_blob_header,
            HEADER_FORMAT,
            HEADER_SIZE,
            serialize_blob,
        )
        import struct

        # Verify HEADER_SIZE matches the struct format
        self.assertEqual(struct.calcsize(HEADER_FORMAT), HEADER_SIZE)

        # Verify blob layout: header starts at 0, metadata at HEADER_SIZE
        engine_bytes = b"test"
        blob = serialize_blob(engine_bytes)
        header = deserialize_blob_header(blob)

        self.assertEqual(header.metadata_offset, HEADER_SIZE)

    def test_get_engine_from_blob(self) -> None:
        """Test engine extraction from blob."""
        from executorch.backends.nvidia.tensorrt.serialization import (
            get_engine_from_blob,
            serialize_blob,
        )

        engine_bytes = b"test_engine_bytes_here"
        blob = serialize_blob(engine_bytes)
        extracted = get_engine_from_blob(blob)

        self.assertEqual(extracted, engine_bytes)

    def test_invalid_blob_header(self) -> None:
        """Test that invalid data returns None."""
        from executorch.backends.nvidia.tensorrt.serialization import (
            deserialize_blob_header,
        )

        result = deserialize_blob_header(b"short")
        self.assertIsNone(result)

    def test_invalid_magic(self) -> None:
        """Test that invalid magic returns False for is_valid."""
        from executorch.backends.nvidia.tensorrt.serialization import (
            TensorRTBlobHeader,
        )

        header = TensorRTBlobHeader(
            magic=b"XXXX",
            metadata_offset=32,
            metadata_size=0,
            engine_offset=32,
            engine_size=100,
        )
        self.assertFalse(header.is_valid())

    def test_io_binding_to_dict(self) -> None:
        """Test TensorRTIOBinding serialization to dict."""
        from executorch.backends.nvidia.tensorrt.serialization import (
            TensorRTIOBinding,
        )

        binding = TensorRTIOBinding(
            name="input_0",
            dtype="float32",
            shape=[1, 3, 224, 224],
            is_input=True,
        )
        d = binding.to_dict()

        self.assertEqual(d["name"], "input_0")
        self.assertEqual(d["dtype"], "float32")
        self.assertEqual(d["shape"], [1, 3, 224, 224])
        self.assertTrue(d["is_input"])

    def test_io_binding_from_dict(self) -> None:
        """Test TensorRTIOBinding deserialization from dict."""
        from executorch.backends.nvidia.tensorrt.serialization import (
            TensorRTIOBinding,
        )

        d = {
            "name": "output_0",
            "dtype": "float16",
            "shape": [1, 1000],
            "is_input": False,
        }
        binding = TensorRTIOBinding.from_dict(d)

        self.assertEqual(binding.name, "output_0")
        self.assertEqual(binding.dtype, "float16")
        self.assertEqual(binding.shape, [1, 1000])
        self.assertFalse(binding.is_input)

    def test_metadata_roundtrip(self) -> None:
        """Test TensorRTBlobMetadata JSON round-trip."""
        from executorch.backends.nvidia.tensorrt.serialization import (
            TensorRTBlobMetadata,
            TensorRTIOBinding,
        )

        original = TensorRTBlobMetadata(
            io_bindings=[
                TensorRTIOBinding("x", "float32", [2, 3], True),
                TensorRTIOBinding("y", "float32", [2, 3], True),
                TensorRTIOBinding("output", "float32", [2, 3], False),
            ]
        )

        json_bytes = original.to_json()
        restored = TensorRTBlobMetadata.from_json(json_bytes)

        self.assertEqual(len(restored.io_bindings), 3)
        self.assertEqual(restored.io_bindings[0].name, "x")
        self.assertEqual(restored.io_bindings[1].name, "y")
        self.assertEqual(restored.io_bindings[2].name, "output")
        self.assertTrue(restored.io_bindings[0].is_input)
        self.assertFalse(restored.io_bindings[2].is_input)

    def test_blob_with_metadata(self) -> None:
        """Test full blob serialization with metadata."""
        from executorch.backends.nvidia.tensorrt.serialization import (
            get_engine_from_blob,
            get_metadata_from_blob,
            serialize_blob,
            TensorRTBlobMetadata,
            TensorRTIOBinding,
        )

        engine_bytes = b"fake_tensorrt_engine"
        metadata = TensorRTBlobMetadata(
            io_bindings=[
                TensorRTIOBinding("input", "float32", [1, 3, 224, 224], True),
                TensorRTIOBinding("output", "float32", [1, 1000], False),
            ]
        )

        blob = serialize_blob(engine_bytes, metadata)

        # Verify engine extraction
        extracted_engine = get_engine_from_blob(blob)
        self.assertEqual(extracted_engine, engine_bytes)

        # Verify metadata extraction
        extracted_metadata = get_metadata_from_blob(blob)
        self.assertIsNotNone(extracted_metadata)
        self.assertEqual(len(extracted_metadata.io_bindings), 2)
        self.assertEqual(extracted_metadata.io_bindings[0].name, "input")
        self.assertEqual(extracted_metadata.io_bindings[1].name, "output")

    def test_blob_alignment(self) -> None:
        """Test that engine data is 16-byte aligned."""
        from executorch.backends.nvidia.tensorrt.serialization import (
            deserialize_blob_header,
            serialize_blob,
            TensorRTBlobMetadata,
            TensorRTIOBinding,
        )

        # Create metadata of varying sizes
        for num_bindings in [1, 2, 5, 10]:
            bindings = [
                TensorRTIOBinding(f"tensor_{i}", "float32", [1, i + 1], i == 0)
                for i in range(num_bindings)
            ]
            metadata = TensorRTBlobMetadata(io_bindings=bindings)
            blob = serialize_blob(b"engine", metadata)
            header = deserialize_blob_header(blob)

            # Engine offset must be 16-byte aligned
            self.assertEqual(header.engine_offset % 16, 0)
