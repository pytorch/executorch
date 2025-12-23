# Copyright 2024-2025 Arm Limited and/or its affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import unittest

from executorch.backends.arm.tosa.specification import (
    Tosa_1_00,
    TosaSpecification,
    TosaSpecMapping,
)

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


class TestTosaSpecification(unittest.TestCase):
    """Tests the TOSA specification class"""

    @parameterized.expand(test_valid_strings)  # type: ignore[misc]
    def test_version_string_no_target(self, version_string: str):
        tosa_spec = TosaSpecification.create_from_string(version_string)
        assert isinstance(tosa_spec, Tosa_1_00)
        assert [profile in ["INT", "FP"] for profile in tosa_spec.profiles].count(
            True
        ) > 0

        for profile in tosa_spec.profiles:
            assert [e in test_valid_extensions[profile] for e in tosa_spec.extensions]

    @parameterized.expand(test_invalid_strings)  # type: ignore[misc]
    def test_invalid_version_strings_no_target(self, version_string: str):
        tosa_spec = None
        with self.assertRaises(ValueError):
            tosa_spec = TosaSpecification.create_from_string(version_string)

        assert tosa_spec is None

    @parameterized.expand(test_valid_strings)
    def test_correct_string_representation_no_target(self, version_string: str):
        tosa_spec = TosaSpecification.create_from_string(version_string)
        assert isinstance(tosa_spec, Tosa_1_00)
        assert f"{tosa_spec}" == version_string


class TestTosaSpecMapping(unittest.TestCase):
    """Tests the TosaSpecMapping class"""

    def test_mapping_no_target(self):
        mapping = TosaSpecMapping()
        mapping.add(TosaSpecification.create_from_string("TOSA-1.0+INT"), "A")
        # check that the mapping is correct
        vals = mapping.get(TosaSpecification.create_from_string("TOSA-1.0+INT"))

        assert vals == ["A"]
        assert len(vals) == 1

    def test_mapping_multiple_no_target(self):
        mapping = TosaSpecMapping()
        mapping.add(TosaSpecification.create_from_string("TOSA-1.0+INT"), "A")
        mapping.add(TosaSpecification.create_from_string("TOSA-1.0+INT"), "B")
        # check that the mapping is correct
        vals = mapping.get(TosaSpecification.create_from_string("TOSA-1.0+INT"))

        assert vals == ["A", "B"]
        assert len(vals) == 2

    def test_mapping_different_profiles_no_target(self):
        mapping = TosaSpecMapping()
        mapping.add(TosaSpecification.create_from_string("TOSA-1.0+INT"), "A")
        mapping.add(TosaSpecification.create_from_string("TOSA-1.0+FP"), "B")
        # check that the mapping is correct
        vals_int = mapping.get(TosaSpecification.create_from_string("TOSA-1.0+INT"))
        vals_fp = mapping.get(TosaSpecification.create_from_string("TOSA-1.0+FP"))

        assert vals_int == ["A"]
        assert vals_fp == ["B"]
        assert len(vals_int) == 1
        assert len(vals_fp) == 1

    def test_mapping_different_profiles_combined_consumer_no_target(self):
        mapping = TosaSpecMapping()
        mapping.add(TosaSpecification.create_from_string("TOSA-1.0+INT"), "A")
        mapping.add(TosaSpecification.create_from_string("TOSA-1.0+FP"), "B")
        # check that the mapping is correct
        combined_vals = mapping.get(
            TosaSpecification.create_from_string("TOSA-1.0+INT+FP")
        )

        assert "A" in combined_vals
        assert "B" in combined_vals
        assert len(combined_vals) == 2

    def test_mapping_no_spec_no_target(self):
        mapping = TosaSpecMapping()
        with self.assertRaises(KeyError):
            mapping.get(TosaSpecification.create_from_string("TOSA-1.0+INT"))

    def test_mapping_no_values_for_spec_no_target(self):
        mapping = TosaSpecMapping()
        mapping.add(TosaSpecification.create_from_string("TOSA-1.0+FP"), "A")
        with self.assertRaises(KeyError):
            mapping.get(TosaSpecification.create_from_string("TOSA-1.0+INT"))

    def test_spec_with_different_profiles_no_target(self):
        mapping = TosaSpecMapping()
        mapping.add(TosaSpecification.create_from_string("TOSA-1.0+FP"), "A")
        mapping.add(TosaSpecification.create_from_string("TOSA-1.0+INT"), "B")
        # check that the mapping is correct
        vals_int = mapping.get(TosaSpecification.create_from_string("TOSA-1.0+INT"))
        vals_fp = mapping.get(TosaSpecification.create_from_string("TOSA-1.0+FP"))
        vals_int_fp = mapping.get(
            TosaSpecification.create_from_string("TOSA-1.0+INT+FP")
        )

        assert vals_fp == ["A"]
        assert vals_int == ["B"]
        assert len(vals_int) == 1
        assert len(vals_fp) == 1
        assert len(vals_int_fp) == 2

    def test_combined_profiles_no_target(self):
        mapping = TosaSpecMapping()
        with self.assertRaises(ValueError):
            # Don't allow multiple profiles in a single spec
            mapping.add(TosaSpecification.create_from_string("TOSA-1.0+INT+FP"), "A")

    def test_spec_add_with_extension_no_target(self):
        mapping = TosaSpecMapping()
        with self.assertRaises(ValueError):
            mapping.add(
                TosaSpecification.create_from_string("TOSA-1.0.0+INT+int16"), "A"
            )

    def test_spec_non_canonical_key_no_target(self):
        mapping = TosaSpecMapping()
        mapping.add(TosaSpecification.create_from_string("TOSA-1.0+INT"), "A")

        val = mapping.get(TosaSpecification.create_from_string("TOSA-1.0+INT+u55"))
        assert val == ["A"]
