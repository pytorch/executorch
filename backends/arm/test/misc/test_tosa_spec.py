# Copyright 2024-2025 Arm Limited and/or its affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import unittest

from executorch.backends.arm.tosa.specification import (
    get_tosa_spec,
    Tosa_1_00,
    TosaSpecification,
)

from executorch.exir.backend.compile_spec_schema import CompileSpec
from parameterized import parameterized  # type: ignore[import-untyped]

test_valid_strings = [
    "TOSA-1.0.0+INT+FP+fft",
    "TOSA-1.0.0+FP+bf16+fft",
    "TOSA-1.0.0+INT+int4+cf",
    "TOSA-1.0.0+FP+cf+bf16+8k",
    "TOSA-1.0.0+FP+INT+bf16+fft+int4+cf",
    "TOSA-1.0.0+FP+INT+fft+int4+cf+8k",
    "TOSA-1.0+INT+FP+fft",
    "TOSA-1.0+FP+bf16+fft",
    "TOSA-1.0+INT+int4+cf",
    "TOSA-1.0+FP+cf+bf16+8k",
    "TOSA-1.0+FP+INT+bf16+fft+int4+cf",
    "TOSA-1.0+FP+INT+fft+int4+cf+8k",
]

test_valid_extensions = {
    "INT": ["int16", "int4", "var", "cf"],
    "FP": ["bf16", "fp8e4m3", "fp8e5m2", "fft", "var", "cf"],
}

test_invalid_strings = [
    "TOSA-1.0.0+fft",
    "TOSA-1.0.0+fp+bf16+fft",
    "TOSA-1.0.0+INT+INT4+cf",
    "TOSA-1.0.0+FP+FP+INT",
    "TOSA-1.0.0+FP+CF+bf16",
    "TOSA-1.0.0+BF16+fft+int4+cf+INT",
]

test_compile_specs = [
    ([CompileSpec("tosa_spec", "TOSA-1.0.0+INT".encode())],),
]

test_compile_specs_no_version = [
    ([CompileSpec("other_key", "some_value".encode())],),
]


class TestTosaSpecification(unittest.TestCase):
    """Tests the TOSA specification class"""

    @parameterized.expand(test_valid_strings)  # type: ignore[misc]
    def test_version_string(self, version_string: str):
        tosa_spec = TosaSpecification.create_from_string(version_string)
        assert isinstance(tosa_spec, Tosa_1_00)
        assert [profile in ["INT", "FP"] for profile in tosa_spec.profiles].count(
            True
        ) > 0

        for profile in tosa_spec.profiles:
            assert [e in test_valid_extensions[profile] for e in tosa_spec.extensions]

    @parameterized.expand(test_invalid_strings)  # type: ignore[misc]
    def test_invalid_version_strings(self, version_string: str):
        tosa_spec = None
        with self.assertRaises(ValueError):
            tosa_spec = TosaSpecification.create_from_string(version_string)

        assert tosa_spec is None

    @parameterized.expand(test_compile_specs)  # type: ignore[misc]
    def test_create_from_compilespec(self, compile_specs: list[CompileSpec]):
        tosa_spec = get_tosa_spec(compile_specs)
        assert isinstance(tosa_spec, TosaSpecification)

    @parameterized.expand(test_compile_specs_no_version)  # type: ignore[misc]
    def test_create_from_invalid_compilespec(self, compile_specs: list[CompileSpec]):
        tosa_spec = None
        with self.assertRaises(ValueError):
            tosa_spec = get_tosa_spec(compile_specs)

        assert tosa_spec is None

    @parameterized.expand(test_valid_strings)
    def test_correct_string_representation(self, version_string: str):
        tosa_spec = TosaSpecification.create_from_string(version_string)
        assert isinstance(tosa_spec, Tosa_1_00)
        assert f"{tosa_spec}" == version_string
