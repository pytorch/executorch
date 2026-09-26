# Copyright 2026 Arm Limited and/or its affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-unsafe

import unittest
from typing import Any
from unittest import mock

import torch
from executorch.backends.arm.test import common as arm_common
from executorch.backends.arm.test.tester.arm_tester import ArmTester
from executorch.backends.test.harness.tester import Tester as BackendTester

from .test_facto import _selected_op_names, cp, FactoTestsBase, Spec


ARM_TOSA_FP_TENSOR_CONSTRAINTS = [
    cp.Dtype.In(lambda deps: [torch.float32]),
]

ARM_TOSA_INT_TENSOR_CONSTRAINTS = [
    # Arm TOSA FACTO runs start from float tensors. INT flows quantize from
    # float inputs instead of treating generated integer tensors as graph inputs.
    cp.Dtype.In(lambda deps: [torch.float32]),
]

NON_FLOAT_TENSOR_INPUT_NAMES = {
    "condition",
    "index",
    "indices",
    "mask",
    "offsets",
}


def _should_force_float_tensor_input(input_name: str) -> bool:
    return input_name.lower() not in NON_FLOAT_TENSOR_INPUT_NAMES


class _ArmFactoTester(ArmTester):
    def __init__(
        self,
        model: torch.nn.Module,
        example_inputs: tuple[Any, ...],
        *,
        quantize_before_export: bool = False,
        qtol: int = 0,
    ):
        tosa_profile = "TOSA-1.0+INT" if quantize_before_export else "TOSA-1.0+FP"
        super().__init__(
            model,
            example_inputs=example_inputs,
            compile_spec=arm_common.get_tosa_compile_spec(tosa_profile),
        )
        self._quantize_before_export = quantize_before_export
        self._qtol = qtol

    def export(self, *args, **kwargs):
        if self._quantize_before_export and not self.is_quantized():
            self.quantize()
        return super().export(*args, **kwargs)

    def run_method_and_compare_outputs(self, *args, **kwargs):
        kwargs.setdefault("qtol", self._qtol)
        return super().run_method_and_compare_outputs(*args, **kwargs)


def _make_arm_tosa_fp_tester(
    model: torch.nn.Module, example_inputs: tuple[Any, ...]
) -> BackendTester:
    return _ArmFactoTester(model, example_inputs)


def _make_arm_tosa_int_tester(
    model: torch.nn.Module, example_inputs: tuple[Any, ...]
) -> BackendTester:
    return _ArmFactoTester(
        model,
        example_inputs,
        quantize_before_export=True,
        qtol=1,
    )


class _FactoTestsArmMixin:
    _tensor_constraints = ARM_TOSA_FP_TENSOR_CONSTRAINTS

    def _patch_spec_for_backend(self, spec: Spec, _op_name: str) -> Spec:
        for inspec in spec.inspec:
            if inspec.type.is_tensor() and _should_force_float_tensor_input(
                inspec.name
            ):
                inspec.constraints.extend(self._tensor_constraints)
        return spec

    def _run_delegated_case(self, tester: BackendTester) -> None:
        tester.to_executorch().run_method_and_compare_outputs(
            inputs=tester.example_inputs
        )

    def _should_fail_on_failures(self) -> bool:
        return True

    def _count_as_test_failures(
        self,
        *,
        eager_fail_count: int,
        export_fail_count: int,
        fail_count: int,
    ) -> int:
        return fail_count + export_fail_count


class _FactoTestsArmSerializedRunnerMixin(_FactoTestsArmMixin):
    def _run_delegated_case(self, tester: BackendTester) -> None:
        tester.to_executorch().serialize().run_method_and_compare_outputs(
            inputs=tester.example_inputs
        )


class FactoTestsArmTOSA_FP(_FactoTestsArmMixin, FactoTestsBase):
    __test__ = True

    def __init__(self, *args, **kwargs):
        super().__init__(_make_arm_tosa_fp_tester, *args, **kwargs)


class FactoTestsArmTOSA_INT(_FactoTestsArmMixin, FactoTestsBase):
    __test__ = True
    _tensor_constraints = ARM_TOSA_INT_TENSOR_CONSTRAINTS

    def __init__(self, *args, **kwargs):
        super().__init__(_make_arm_tosa_int_tester, *args, **kwargs)


class TestArmFactoTensorConstraintSelection(unittest.TestCase):
    def test_arm_shared_bases_are_not_unittest_cases(self) -> None:
        self.assertFalse(issubclass(_FactoTestsArmMixin, unittest.TestCase))
        self.assertFalse(
            issubclass(_FactoTestsArmSerializedRunnerMixin, unittest.TestCase)
        )

    def test_delegated_runtime_uses_facto_inputs(self) -> None:
        tester = mock.Mock()
        tester.example_inputs = ("facto-input",)

        _FactoTestsArmMixin()._run_delegated_case(tester)

        (
            tester.to_executorch.return_value.run_method_and_compare_outputs.assert_called_once_with(
                inputs=tester.example_inputs
            )
        )

    def test_serialized_runtime_uses_facto_inputs(self) -> None:
        tester = mock.Mock()
        tester.example_inputs = ("facto-input",)

        _FactoTestsArmSerializedRunnerMixin()._run_delegated_case(tester)

        (
            tester.to_executorch.return_value.serialize.return_value.run_method_and_compare_outputs.assert_called_once_with(
                inputs=tester.example_inputs
            )
        )

    def test_arm_counts_export_failures_as_test_failures(self) -> None:
        self.assertEqual(
            _FactoTestsArmMixin()._count_as_test_failures(
                eager_fail_count=2,
                export_fail_count=3,
                fail_count=4,
            ),
            7,
        )

    def test_value_tensor_inputs_are_forced_to_float(self) -> None:
        self.assertTrue(_should_force_float_tensor_input("self"))
        self.assertTrue(_should_force_float_tensor_input("other"))
        self.assertTrue(_should_force_float_tensor_input("weight"))

    def test_auxiliary_tensor_inputs_preserve_original_dtype(self) -> None:
        self.assertFalse(_should_force_float_tensor_input("condition"))
        self.assertFalse(_should_force_float_tensor_input("index"))
        self.assertFalse(_should_force_float_tensor_input("indices"))
        self.assertFalse(_should_force_float_tensor_input("mask"))
        self.assertFalse(_should_force_float_tensor_input("offsets"))


for op_name in _selected_op_names():
    FactoTestsArmTOSA_FP._generate_test(op_name)
    FactoTestsArmTOSA_INT._generate_test(op_name)
