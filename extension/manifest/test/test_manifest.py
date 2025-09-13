# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import unittest

import torch

from executorch.exir import to_edge
from executorch.extension.manifest._manifest import (
    _MANIFEST_BYTEORDER,
    _ManifestLayout,
    append_manifest,
    Manifest,
)
from executorch.extension.pybindings.portable_lib import (
    _load_for_executorch_from_buffer,
)
from torch.export import export


class TestManifestLayout(unittest.TestCase):
    """Test cases for _ManifestLayout class."""

    def setUp(self):
        """Set up test fixtures."""
        self.valid_signature = b"test_signature_123"
        self.valid_signature_size = len(self.valid_signature)
        self.valid_program_offset = 1008
        self.valid_padding_size = 8
        self.manifest_layout = _ManifestLayout(
            signature=self.valid_signature,
            program_offset=self.valid_program_offset,
            padding_size=self.valid_padding_size,
        )

    def test_init_with_defaults(self):
        """Test initialization with default values."""
        layout = _ManifestLayout(signature=b"test123")
        self.assertEqual(layout.signature, b"test123")
        self.assertEqual(layout.signature_size, 7)
        self.assertEqual(layout.magic, _ManifestLayout.EXPECTED_MAGIC)

    def test_is_valid_true(self):
        """Test is_valid returns True for valid manifest."""
        self.assertTrue(self.manifest_layout.is_valid())

    def test_is_valid_false_wrong_magic(self):
        """Test is_valid returns False for wrong magic."""
        self.manifest_layout.magic = b"bad!"
        self.assertFalse(self.manifest_layout.is_valid())

    def test_is_valid_false_short_length(self):
        """Test is_valid returns False for length too short."""
        self.manifest_layout.length = _ManifestLayout.EXPECTED_MIN_LENGTH - 1
        self.assertFalse(self.manifest_layout.is_valid())

    def test_to_bytes(self):
        """Test converting manifest layout to bytes."""
        data = self.manifest_layout.to_bytes()

        # Should be exactly EXPECTED_MIN_LENGTH + signature_size bytes
        expected_length = (
            _ManifestLayout.EXPECTED_MIN_LENGTH + self.valid_signature_size
        )
        self.assertEqual(len(data), expected_length)

        # Check magic at the end
        self.assertEqual(data[-4:], _ManifestLayout.EXPECTED_MAGIC)

        # Check length field
        expected_length_bytes = expected_length.to_bytes(
            4, byteorder=_MANIFEST_BYTEORDER
        )
        self.assertEqual(data[-8:-4], expected_length_bytes)

        # Check program offset (8 bytes)
        expected_program_offset = self.valid_program_offset.to_bytes(
            8, byteorder=_MANIFEST_BYTEORDER
        )
        self.assertEqual(data[-16:-8], expected_program_offset)

        # Check padding size (4 bytes)
        expected_padding_size = self.valid_padding_size.to_bytes(
            4, byteorder=_MANIFEST_BYTEORDER
        )
        self.assertEqual(data[-20:-16], expected_padding_size)

        # Check signature size (4 bytes)
        expected_signature_size = self.valid_signature_size.to_bytes(
            4, byteorder=_MANIFEST_BYTEORDER
        )
        self.assertEqual(data[-24:-20], expected_signature_size)

        # Check signature (variable length bytes at the beginning)
        self.assertEqual(data[: self.valid_signature_size], self.valid_signature)

    def test_from_bytes_valid_data(self):
        """Test creating manifest layout from valid bytes."""
        original_data = self.manifest_layout.to_bytes()

        # Add some extra data at the beginning to simulate real usage
        full_data = b"some_data_here" + original_data

        layout = _ManifestLayout.from_bytes(full_data)

        self.assertEqual(layout.signature, self.valid_signature)
        self.assertEqual(layout.signature_size, self.valid_signature_size)
        self.assertEqual(
            layout.length,
            _ManifestLayout.EXPECTED_MIN_LENGTH + self.valid_signature_size,
        )
        self.assertEqual(layout.magic, _ManifestLayout.EXPECTED_MAGIC)

    def test_from_bytes_insufficient_data(self):
        """Test from_bytes raises ValueError with insufficient data."""
        short_data = b"short"

        with self.assertRaises(ValueError) as cm:
            _ManifestLayout.from_bytes(short_data)

        self.assertIn("Not enough data for the manifest", str(cm.exception))

    def test_from_bytes_success(self):
        """Test from_bytes works with exactly the required length."""
        data = self.manifest_layout.to_bytes()
        layout = _ManifestLayout.from_bytes(data)

        self.assertEqual(layout.signature, self.valid_signature)
        self.assertEqual(layout.signature_size, self.valid_signature_size)

    def test_from_manifest(self):
        """Test creating manifest layout from Manifest object."""
        manifest = Manifest(signature=self.valid_signature)
        layout = _ManifestLayout.from_manifest(manifest)

        self.assertEqual(layout.signature, self.valid_signature)

    def test_roundtrip_to_bytes_from_bytes(self):
        """Test that to_bytes and from_bytes are inverse operations."""
        original = self.manifest_layout
        data = original.to_bytes()
        reconstructed = _ManifestLayout.from_bytes(data)

        self.assertEqual(original.signature, reconstructed.signature)
        self.assertEqual(original.signature_size, reconstructed.signature_size)
        self.assertEqual(original.program_offset, reconstructed.program_offset)
        self.assertEqual(original.magic, reconstructed.magic)


class TestManifest(unittest.TestCase):
    """Test cases for Manifest class."""

    def setUp(self):
        """Set up test fixtures."""
        self.valid_signature = b"manifest_test_sig"
        self.manifest = Manifest(signature=self.valid_signature)

    def test_from_bytes_valid(self):
        """Test creating Manifest from valid bytes."""
        layout = _ManifestLayout(signature=self.valid_signature)
        data = layout.to_bytes()

        manifest = Manifest.from_bytes(data)
        print("foobar")
        print(layout)
        print(manifest)

        self.assertEqual(manifest.signature, self.valid_signature)

    def test_from_bytes_invalid_data(self):
        """Test from_bytes raises ValueError with invalid data."""
        # Create invalid layout with wrong magic
        layout = _ManifestLayout(signature=self.valid_signature)
        layout.magic = b"bad!"
        # Can't easily construct invalid data since to_bytes() enforces validity
        # Instead, create some invalid bytes manually
        data = b"invalid_data_too_short"

        with self.assertRaises(ValueError) as cm:
            Manifest.from_bytes(data)

        self.assertIn("Not enough data for the manifest", str(cm.exception))

    def test_from_bytes_insufficient_data(self):
        """Test from_bytes raises ValueError with insufficient data."""
        short_data = b"short"

        with self.assertRaises(ValueError):
            Manifest.from_bytes(short_data)


class TestAppendManifest(unittest.TestCase):
    """Test cases for append_manifest function."""

    def setUp(self):
        """Set up test fixtures."""
        self.signature = b"test_append_sig"
        self.manifest = Manifest(signature=self.signature)
        self.test_data = b"12345678"

    def test_append_manifest_default_alignment(self):
        """Test append_manifest with default alignment."""
        result = append_manifest(self.test_data, self.manifest)

        # Check that data was appended
        self.assertGreater(len(result), len(self.test_data))

        # The manifest should be at the end
        expected_manifest_length = _ManifestLayout.EXPECTED_MIN_LENGTH + len(
            self.signature
        )
        manifest_data = result[-expected_manifest_length:]
        reconstructed_manifest_layout = _ManifestLayout.from_bytes(manifest_data)
        reconstructed_manifest = Manifest.from_bytes(manifest_data)

        self.assertEqual(reconstructed_manifest.signature, self.signature)
        self.assertEqual(
            reconstructed_manifest_layout.signature_size, len(self.signature)
        )
        self.assertEqual(
            reconstructed_manifest_layout.program_offset, len(self.test_data)
        )


class TestEndToEndManifestIntegration(unittest.TestCase):
    """End-to-end integration test for ExecutorTorch models with manifests."""

    def test_simple_nn_module_with_manifest(self):
        """Test creating a simple nn.Module, lowering to ExecutorTorch, adding manifest, and running."""

        # Create a simple neural network module
        class SimpleLinearModel(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.linear = torch.nn.Linear(4, 2)
                self.relu = torch.nn.ReLU()

            def forward(self, x):
                x = self.linear(x)
                x = self.relu(x)
                return x

        # Create model and sample input
        model = SimpleLinearModel()
        sample_input = (torch.ones(1, 4, dtype=torch.float32),)

        # Lower to ExecutorTorch
        exported_program = export(model, sample_input, strict=True)
        edge_program = to_edge(exported_program)
        executorch_program = edge_program.to_executorch()

        # Get the serialized program data
        program_data = executorch_program.buffer
        self.assertGreater(len(program_data), 0, "Program data should not be empty")

        # Create a manifest with a unique signature
        test_signature = b"integration_test_sig"
        manifest = Manifest(signature=test_signature)

        # Add manifest to the program data
        final_data = append_manifest(program_data, manifest, alignment=16)

        # Check that the manifest was appended
        self.assertGreater(
            len(final_data),
            len(program_data),
            "Data should be larger after adding manifest",
        )

        # Verify manifest fields are correct
        expected_manifest_length = _ManifestLayout.EXPECTED_MIN_LENGTH + len(
            test_signature
        )
        manifest_bytes = final_data[-expected_manifest_length:]
        reconstructed_layout = _ManifestLayout.from_bytes(manifest_bytes)
        reconstructed_manifest = Manifest.from_bytes(manifest_bytes)

        # Check manifest fields
        self.assertTrue(reconstructed_layout.is_valid(), "Manifest should be valid")
        self.assertEqual(
            reconstructed_manifest.signature, test_signature, "Signature should match"
        )
        self.assertEqual(
            reconstructed_layout.magic,
            _ManifestLayout.EXPECTED_MAGIC,
            "Magic should match",
        )
        self.assertEqual(
            reconstructed_layout.signature_size,
            len(test_signature),
            "Signature size should match",
        )

        # Check program offset calculation
        self.assertEqual(
            reconstructed_layout.program_offset,
            len(program_data),
            "Program offset should be correct",
        )

        # Load and execute the model with ExecutorTorch runtime
        module = _load_for_executorch_from_buffer(final_data)

        test_input = torch.ones(1, 4, dtype=torch.float32)
        output = module.run_method("forward", (test_input,))

        with torch.no_grad():
            expected_output = model(test_input)

        torch.testing.assert_close(output[0], expected_output, atol=1e-5, rtol=1e-5)


if __name__ == "__main__":
    unittest.main()
