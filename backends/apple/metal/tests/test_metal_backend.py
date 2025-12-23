# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import unittest

from executorch.backends.apple.metal.metal_backend import (
    COMPILE_SPEC_KEYS,
    MetalBackend,
)
from executorch.exir.backend.compile_spec_schema import CompileSpec


class TestMetalBackend(unittest.TestCase):
    """Test Metal backend utility functions."""

    def test_generate_method_name_compile_spec(self):
        """Test that compile spec is generated correctly with method name."""
        method_name = "forward"
        compile_spec = MetalBackend.generate_method_name_compile_spec(method_name)

        # Verify compile spec structure
        self.assertIsInstance(compile_spec, CompileSpec)
        self.assertEqual(compile_spec.key, COMPILE_SPEC_KEYS.METHOD_NAME.value)
        self.assertEqual(compile_spec.value, method_name.encode("utf-8"))

    def test_method_name_from_compile_specs(self):
        """Test extracting method name from compile specs."""
        method_name = "forward"
        compile_specs = [MetalBackend.generate_method_name_compile_spec(method_name)]

        # Extract method name
        extracted_name = MetalBackend.method_name_from_compile_specs(compile_specs)

        self.assertEqual(extracted_name, method_name)

    def test_method_name_from_compile_specs_with_multiple_specs(self):
        """Test extracting method name when there are multiple compile specs."""
        method_name = "forward"
        compile_specs = [
            CompileSpec("other_key", b"other_value"),
            MetalBackend.generate_method_name_compile_spec(method_name),
            CompileSpec("another_key", b"another_value"),
        ]

        # Extract method name
        extracted_name = MetalBackend.method_name_from_compile_specs(compile_specs)

        self.assertEqual(extracted_name, method_name)

    def test_method_name_from_compile_specs_missing(self):
        """Test that RuntimeError is raised when method name is missing."""
        compile_specs = [
            CompileSpec("other_key", b"other_value"),
        ]

        # Should raise RuntimeError when method name is not found
        with self.assertRaises(RuntimeError) as context:
            MetalBackend.method_name_from_compile_specs(compile_specs)

        self.assertIn("Could not find method name", str(context.exception))

    def test_compile_spec_roundtrip(self):
        """Test that method name survives encode/decode roundtrip."""
        original_name = "my_custom_method"

        # Generate compile spec
        compile_spec = MetalBackend.generate_method_name_compile_spec(original_name)

        # Extract from compile specs list
        extracted_name = MetalBackend.method_name_from_compile_specs([compile_spec])

        self.assertEqual(original_name, extracted_name)


if __name__ == "__main__":
    unittest.main()
