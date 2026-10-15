# Copyright 2026 Arm Limited and/or its affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-unsafe

import unittest
from typing import Any
from unittest import mock

import torch
from executorch.backends.arm.common.arm_compile_spec import ArmCompileSpec
from executorch.backends.arm.test import common as arm_common
from executorch.backends.arm.test.tester.arm_tester import ArmTester
from executorch.backends.test.harness.tester import Tester as BackendTester

from .test_facto import (
    _facto_config,
    _generated_facto_tests_enabled,
    _selected_arm_op_names,
    cp,
    FACTO_AVAILABLE,
    FactoTestsBase,
    Spec,
)


if FACTO_AVAILABLE:
    ARM_TOSA_FP_TENSOR_CONSTRAINTS = [
        cp.Dtype.In(lambda deps: [torch.float32]),
    ]

    ARM_TOSA_INT_TENSOR_CONSTRAINTS = [
        # Arm TOSA FACTO runs start from float tensors. INT flows quantize from
        # float inputs instead of treating generated integer tensors as graph inputs.
        cp.Dtype.In(lambda deps: [torch.float32]),
    ]
else:
    ARM_TOSA_FP_TENSOR_CONSTRAINTS = []
    ARM_TOSA_INT_TENSOR_CONSTRAINTS = []

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
        compile_spec: ArmCompileSpec,
        quantize_before_export: bool = False,
        qtol: int = 0,
    ):
        super().__init__(
            model,
            example_inputs=example_inputs,
            compile_spec=compile_spec,
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
    return _ArmFactoTester(
        model,
        example_inputs,
        compile_spec=arm_common.get_tosa_compile_spec("TOSA-1.0+FP"),
    )


def _make_arm_tosa_int_tester(
    model: torch.nn.Module, example_inputs: tuple[Any, ...]
) -> BackendTester:
    return _ArmFactoTester(
        model,
        example_inputs,
        compile_spec=arm_common.get_tosa_compile_spec("TOSA-1.0+INT"),
        quantize_before_export=True,
        qtol=1,
    )


def _make_arm_vgf_fp_tester(
    model: torch.nn.Module, example_inputs: tuple[Any, ...]
) -> BackendTester:
    return _ArmFactoTester(
        model,
        example_inputs,
        compile_spec=arm_common.get_vgf_compile_spec("TOSA-1.0+FP"),
    )


def _make_arm_vgf_int_tester(
    model: torch.nn.Module, example_inputs: tuple[Any, ...]
) -> BackendTester:
    return _ArmFactoTester(
        model,
        example_inputs,
        compile_spec=arm_common.get_vgf_compile_spec("TOSA-1.0+INT"),
        quantize_before_export=True,
        qtol=1,
    )


def _vgf_serialized_runner_available() -> bool:
    return (
        arm_common.model_converter_installed()
        and arm_common.vkml_emulation_layer_installed()
        and arm_common.arm_executor_runner_exists("vkml_emulation_layer")
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
        return eager_fail_count + fail_count + export_fail_count


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


@unittest.skipUnless(
    _vgf_serialized_runner_available(),
    "Did not find VGF model-converter or runtime environment",
)
class FactoTestsArmVGF_FP(_FactoTestsArmSerializedRunnerMixin, FactoTestsBase):
    __test__ = True

    def __init__(self, *args, **kwargs):
        super().__init__(_make_arm_vgf_fp_tester, *args, **kwargs)


@unittest.skipUnless(
    _vgf_serialized_runner_available(),
    "Did not find VGF model-converter or runtime environment",
)
class FactoTestsArmVGF_INT(_FactoTestsArmSerializedRunnerMixin, FactoTestsBase):
    __test__ = True
    _tensor_constraints = ARM_TOSA_INT_TENSOR_CONSTRAINTS

    def __init__(self, *args, **kwargs):
        super().__init__(_make_arm_vgf_int_tester, *args, **kwargs)


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

    def test_arm_counts_all_pre_runtime_failures_as_test_failures(self) -> None:
        self.assertEqual(
            _FactoTestsArmMixin()._count_as_test_failures(
                eager_fail_count=2,
                export_fail_count=3,
                fail_count=4,
            ),
            9,
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

    def test_vgf_int_tester_uses_int_only_profile(self) -> None:
        tester = _make_arm_vgf_int_tester(torch.nn.Identity(), (torch.ones(1),))
        profiles = tester.compile_spec.tosa_spec.profiles

        self.assertIn("INT", profiles)
        self.assertNotIn("FP", profiles)

    def test_vgf_serialized_runner_requires_converter_and_runtime(self) -> None:
        with (
            mock.patch.object(
                arm_common, "model_converter_installed", return_value=True
            ),
            mock.patch.object(
                arm_common, "vkml_emulation_layer_installed", return_value=True
            ),
            mock.patch.object(
                arm_common,
                "arm_executor_runner_exists",
                return_value=True,
            ),
        ):
            self.assertTrue(_vgf_serialized_runner_available())

        with (
            mock.patch.object(
                arm_common, "model_converter_installed", return_value=True
            ),
            mock.patch.object(
                arm_common, "vkml_emulation_layer_installed", return_value=False
            ),
            mock.patch.object(
                arm_common,
                "arm_executor_runner_exists",
                return_value=True,
            ),
        ):
            self.assertFalse(_vgf_serialized_runner_available())


_ARM_CONFIG = _facto_config()
_ARM_SELECTED_OP_NAMES = _selected_arm_op_names()

for op_name in _ARM_SELECTED_OP_NAMES:
    FactoTestsArmTOSA_FP._generate_test(op_name)
    FactoTestsArmTOSA_INT._generate_test(op_name)
    FactoTestsArmVGF_FP._generate_test(op_name)
    FactoTestsArmVGF_INT._generate_test(op_name)

if _generated_facto_tests_enabled():
    if not FACTO_AVAILABLE:
        FactoTestsArmTOSA_FP._generate_missing_facto_test()
    elif _ARM_CONFIG.selected_ops_error is not None:
        FactoTestsArmTOSA_FP._generate_configuration_error_test(
            "test_facto_ops_filter_matches_ops",
            _ARM_CONFIG.selected_ops_error,
        )
