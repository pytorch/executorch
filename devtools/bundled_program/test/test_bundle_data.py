# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
# Copyright 2026 Arm Limited and/or its affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

import tempfile
import unittest
from typing import List
from unittest.mock import patch

import executorch.devtools.bundled_program.schema as bp_schema

import torch
from executorch.devtools.bundled_program.config import (
    ConfigValue,
    MethodTestCase,
    MethodTestSuite,
)
from executorch.devtools.bundled_program.core import BundledProgram
from executorch.devtools.bundled_program.util.test_util import (
    get_common_executorch_program,
)
from executorch.exir import to_edge
from executorch.exir._serialize import _PTEFile, _serialize_pte_binary
from executorch.exir._serialize._program import _get_extended_header
from executorch.exir.tensor import stride_from_dim_order
from torch.export import export


class TestBundle(unittest.TestCase):
    def _serialize_tensor_test_case(
        self, input_tensor: torch.Tensor, output_tensor: torch.Tensor
    ) -> bp_schema.BundledMethodTestCase:
        class Model(torch.nn.Module):
            def forward(self, x):
                return x * 2

        program = to_edge(export(Model(), (input_tensor,))).to_executorch()
        with tempfile.TemporaryDirectory() as tmp_dir:
            pte_path = f"{tmp_dir}/model.pte"
            with open(pte_path, "wb") as pte_file:
                pte_file.write(program.buffer)
            bundled_program = BundledProgram(
                executorch_program=None,
                method_test_suites=[
                    MethodTestSuite(
                        "forward", [MethodTestCase((input_tensor,), (output_tensor,))]
                    )
                ],
                pte_file_path=pte_path,
            )
            return (
                bundled_program.serialize_to_schema()
                .method_test_suites[0]
                .test_cases[0]
            )

    def test_channels_last_input_and_output(self) -> None:
        input_tensor = (
            torch.arange(2 * 3 * 4 * 5, dtype=torch.float32)
            .reshape(2, 3, 4, 5)
            .contiguous(memory_format=torch.channels_last)
        )
        output_tensor = (input_tensor * 2).contiguous(memory_format=torch.channels_last)
        test_case = self._serialize_tensor_test_case(input_tensor, output_tensor)

        for bundled_value, source in zip(
            test_case.inputs + test_case.expected_outputs,
            (input_tensor, output_tensor),
        ):
            bundled_tensor = bundled_value.val
            assert isinstance(bundled_tensor, bp_schema.Tensor)
            self.assertEqual(bundled_tensor.dim_order, [0, 2, 3, 1])
            self.assertEqual(bundled_tensor.sizes, list(source.shape))
            self.assertEqual(
                bundled_tensor.data,
                source.permute(0, 2, 3, 1).contiguous().numpy().tobytes(),
            )

    def test_channels_last_view_with_storage_offset(self) -> None:
        base = (
            torch.arange(3 * 3 * 4 * 5, dtype=torch.float32)
            .reshape(3, 3, 4, 5)
            .contiguous(memory_format=torch.channels_last)
        )
        view = base[1:3]
        self.assertGreater(view.storage_offset(), 0)
        test_case = self._serialize_tensor_test_case(view, view)

        for bundled_value in test_case.inputs + test_case.expected_outputs:
            bundled_tensor = bundled_value.val
            assert isinstance(bundled_tensor, bp_schema.Tensor)
            self.assertEqual(bundled_tensor.dim_order, [0, 2, 3, 1])
            self.assertEqual(len(bundled_tensor.data), view.nbytes)
            self.assertEqual(
                bundled_tensor.data,
                view.permute(0, 2, 3, 1).contiguous().numpy().tobytes(),
            )

    def test_non_dense_view_uses_only_its_own_elements(self) -> None:
        base = torch.arange(2 * 3 * 4 * 5, dtype=torch.float32).reshape(2, 3, 4, 5)
        view = base[:, :, ::2, :]
        test_case = self._serialize_tensor_test_case(view, view)

        for bundled_value in test_case.inputs + test_case.expected_outputs:
            bundled_tensor = bundled_value.val
            assert isinstance(bundled_tensor, bp_schema.Tensor)
            self.assertEqual(len(bundled_tensor.data), view.nbytes)
            decoded = torch.frombuffer(
                bytearray(bundled_tensor.data), dtype=view.dtype
            ).as_strided(
                bundled_tensor.sizes,
                stride_from_dim_order(bundled_tensor.sizes, bundled_tensor.dim_order),
            )
            self.assertTrue(torch.equal(decoded, view))

    def test_bundle_uses_program_dim_order(self) -> None:
        class Permute(torch.nn.Module):
            def forward(self, x):
                return x.permute(0, 2, 3, 1)

        model = Permute()
        export_input = torch.zeros(2, 3, 4, 5)
        program = to_edge(export(model, (export_input,))).to_executorch()
        input_tensor = torch.arange(120, dtype=torch.float32).reshape(2, 4, 3, 5)
        input_tensor = input_tensor.transpose(1, 2)
        expected = model(input_tensor)
        self.assertFalse(input_tensor.is_contiguous())
        self.assertFalse(expected.is_contiguous())
        suites = [
            MethodTestSuite("forward", [MethodTestCase((input_tensor,), (expected,))])
        ]
        with tempfile.TemporaryDirectory() as tmp_dir:
            pte_path = f"{tmp_dir}/model.pte"
            with open(pte_path, "wb") as pte_file:
                pte_file.write(program.buffer)
            for from_file in (False, True):
                with self.subTest(from_file=from_file):
                    bundle = (
                        BundledProgram(None, suites, pte_file_path=pte_path)
                        if from_file
                        else BundledProgram(program, suites)
                    )
                    case = (
                        bundle.serialize_to_schema().method_test_suites[0].test_cases[0]
                    )
                    for value, source in zip(
                        case.inputs + case.expected_outputs, (input_tensor, expected)
                    ):
                        self.assertEqual(value.val.dim_order, [0, 1, 2, 3])
                        self.assertEqual(
                            value.val.data, source.contiguous().numpy().tobytes()
                        )

    def test_pte_reader_matches_in_memory_program(self) -> None:
        program, suites = get_common_executorch_program()
        expected = BundledProgram(program, suites).serialize_to_schema()
        with tempfile.TemporaryDirectory() as tmp_dir:
            pte_path = f"{tmp_dir}/model.pte"
            with open(pte_path, "wb") as pte_file:
                pte_file.write(program.buffer)
            actual = BundledProgram(
                None, suites, pte_file_path=pte_path
            ).serialize_to_schema()
        self.assertEqual(actual, expected)

    def test_pte_reader_does_not_unpack_program_payloads(self) -> None:
        class Model(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.register_buffer("weight", torch.arange(6).reshape(2, 3).float())

            def forward(self, x):
                return x + self.weight

        model = Model()
        tensor = torch.arange(6).reshape(3, 2).t().float()
        expected = model(tensor)
        program = to_edge(export(model, (torch.zeros(2, 3),))).to_executorch()
        program_bytes = program.buffer
        header = _get_extended_header(program_bytes)
        assert header is not None
        self.assertGreater(header.segment_data_size, 0)
        suites = [MethodTestSuite("forward", [MethodTestCase((tensor,), (expected,))])]
        with tempfile.TemporaryDirectory() as tmp_dir:
            pte_path = f"{tmp_dir}/model.pte"
            with open(pte_path, "wb") as pte_file:
                pte_file.write(program_bytes)
            with (
                patch(
                    "executorch.exir._serialize._flatbuffer_program.ProgramT.InitFromPackedBuf",
                    side_effect=AssertionError("Full program must not be unpacked"),
                ),
                patch(
                    "executorch.exir._serialize._program._restore_segments",
                    side_effect=AssertionError("Program payloads must not be restored"),
                ),
            ):
                bundle = BundledProgram(None, suites, pte_file_path=pte_path)
                schema = bundle.serialize_to_schema()

            self.assertEqual(schema.program, program_bytes)
            case = schema.method_test_suites[0].test_cases[0]
            for value, source in zip(
                case.inputs + case.expected_outputs, (tensor, expected)
            ):
                self.assertEqual(value.val.dim_order, [0, 1])
                self.assertEqual(value.val.sizes, [2, 3])
                self.assertEqual(value.val.data, source.contiguous().numpy().tobytes())

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
            bytes(
                _serialize_pte_binary(
                    pte_file=_PTEFile(program=executorch_program.executorch_program)
                )
            ),
        )

    def test_bundled_program_from_pte(self) -> None:
        executorch_program, method_test_suites = get_common_executorch_program()

        with tempfile.TemporaryDirectory() as tmp_dir:
            executorch_model_path = f"{tmp_dir}/executorch_model.pte"
            with open(executorch_model_path, "wb") as f:
                f.write(executorch_program.buffer)

            bundled_program = BundledProgram(
                executorch_program=None,
                method_test_suites=method_test_suites,
                pte_file_path=executorch_model_path,
            )

            method_test_suites = sorted(method_test_suites, key=lambda t: t.method_name)

            for plan_id in range(
                len(executorch_program.executorch_program.execution_plan)
            ):
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
                executorch_program.buffer,
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
