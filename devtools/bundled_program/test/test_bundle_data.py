# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

import unittest
from typing import List

import executorch.devtools.bundled_program.schema as bp_schema

import torch
from executorch.devtools.bundled_program.config import ConfigValue
from executorch.devtools.bundled_program.core import BundledProgram
from executorch.devtools.bundled_program.util.test_util import (
    get_common_executorch_program,
)
from executorch.exir._serialize import _serialize_pte_binary


class TestBundle(unittest.TestCase):
    def assertIOsetDataEqual(
        self,
        program_ioset_data: List[bp_schema.Value],
        config_ioset_data: List[ConfigValue],
    ) -> None:
        self.assertEqual(len(program_ioset_data), len(config_ioset_data))
        for program_element, config_element in zip(
            program_ioset_data, config_ioset_data
        ):
            if isinstance(program_element.val, bp_schema.Tensor):
                # TODO: Update to check the bundled input share the same type with the config input after supporting multiple types.
                self.assertTrue(isinstance(config_element, torch.Tensor))
                self.assertEqual(program_element.val.sizes, list(config_element.size()))
                # TODO(gasoonjia): Check the inner data.
            elif type(program_element.val) is bp_schema.Int:
                self.assertEqual(program_element.val.int_val, config_element)
            elif type(program_element.val) is bp_schema.Double:
                self.assertEqual(program_element.val.double_val, config_element)
            elif type(program_element.val) is bp_schema.Bool:
                self.assertEqual(program_element.val.bool_val, config_element)

    def test_bundled_program(self) -> None:
        executorch_program, method_test_suites = get_common_executorch_program()

        bundled_program = BundledProgram(executorch_program, method_test_suites)

        method_test_suites = sorted(method_test_suites, key=lambda t: t.method_name)

        for plan_id in range(len(executorch_program.executorch_program.execution_plan)):
            bundled_plan_test = (
                bundled_program.serialize_to_schema().method_test_suites[plan_id]
            )
            method_test_suite = method_test_suites[plan_id]

            self.assertEqual(
                len(bundled_plan_test.test_cases), len(method_test_suite.test_cases)
            )
            for bundled_program_ioset, method_test_case in zip(
                bundled_plan_test.test_cases, method_test_suite.test_cases
            ):
                self.assertIOsetDataEqual(
                    bundled_program_ioset.inputs, method_test_case.inputs
                )
                self.assertIOsetDataEqual(
                    bundled_program_ioset.expected_outputs,
                    method_test_case.expected_outputs,
                )

        self.assertEqual(
            bundled_program.serialize_to_schema().program,
            bytes(_serialize_pte_binary(executorch_program.executorch_program)),
        )

    def test_bundled_miss_methods(self) -> None:
        executorch_program, method_test_suites = get_common_executorch_program()

        # only keep the testcases for the first method to mimic the case that user only creates testcases for the first method.
        method_test_suites = method_test_suites[:1]

        _ = BundledProgram(executorch_program, method_test_suites)

    def test_bundled_wrong_method_name(self) -> None:
        executorch_program, method_test_suites = get_common_executorch_program()

        method_test_suites[-1].method_name = "wrong_method_name"
        self.assertRaises(
            AssertionError,
            BundledProgram,
            executorch_program,
            method_test_suites,
        )

    def test_bundle_wrong_input_type(self) -> None:
        executorch_program, method_test_suites = get_common_executorch_program()

        # pyre-ignore[8]: Use a wrong type on purpose. Should raise an error when creating a bundled program using method_test_suites.
        method_test_suites[0].test_cases[-1].inputs = ["WRONG INPUT TYPE"]
        self.assertRaises(
            AssertionError,
            BundledProgram,
            executorch_program,
            method_test_suites,
        )

    def test_bundle_wrong_output_type(self) -> None:
        executorch_program, method_test_suites = get_common_executorch_program()

        method_test_suites[0].test_cases[-1].expected_outputs = [
            0,
            0.0,
        ]
        self.assertRaises(
            AssertionError,
            BundledProgram,
            executorch_program,
            method_test_suites,
        )
