# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
# Copyright 2026 Arm Limited and/or its affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import ctypes
from typing import Dict, List, Optional, Sequence, Union

import executorch.devtools.bundled_program.schema as bp_schema

import executorch.exir.schema as core_schema

import torch
import torch.fx
from executorch.devtools.bundled_program.config import ConfigValue, MethodTestSuite

from executorch.devtools.bundled_program.version import BUNDLED_PROGRAM_SCHEMA_VERSION

from executorch.exir import ExecutorchProgram, ExecutorchProgramManager
from executorch.exir._serialize import deserialize_program_method_io, ProgramMethodIO
from executorch.exir.tensor import get_scalar_type, scalar_type_enum

supported_program_type_table: Dict[type, type] = {
    core_schema.Tensor: torch.Tensor,
    core_schema.Int: int,
    core_schema.Double: float,
    core_schema.Bool: bool,
}


class BundledProgram:
    """
    Bundled program contains all information needed to execute and verify the program on device.

    Public Attributes:
        method_test_suites: All test suites for verifying methods.
        executorch_program: ExecutorchProgram-like variable, containing the Program to be verified by method_test_suites, including
                            ExecutorchProgram, MultiMethodExecutorchProgram or ExecutorchProgramManager.
    """

    def __init__(
        self,
        executorch_program: Optional[
            Union[
                ExecutorchProgram,
                ExecutorchProgramManager,
            ]
        ],
        method_test_suites: Sequence[MethodTestSuite],
        pte_file_path: Optional[str] = None,
    ):
        """Create BundledProgram by bundling the given program and method_test_suites together.

        Args:
            executorch_program: The program to be bundled.
            method_test_suites: The testcases for certain methods to be bundled.
            pte_file_path: The path to pte file to deserialize program if executorch_program is not provided.
        """
        if not executorch_program and not pte_file_path:
            raise RuntimeError(
                "Either executorch_program or pte_file_path must be provided"
            )

        if executorch_program and pte_file_path:
            raise RuntimeError(
                "Only one of executorch_program or pte_file_path can be used"
            )

        method_test_suites = sorted(method_test_suites, key=lambda x: x.method_name)
        if executorch_program:
            self._assert_valid_bundle(
                self._get_method_io(self._extract_program(executorch_program)),
                method_test_suites,
            )
        self.executorch_program: Optional[
            Union[
                ExecutorchProgram,
                ExecutorchProgramManager,
            ]
        ] = executorch_program
        self._pte_file_path: Optional[str] = pte_file_path

        self.method_test_suites = method_test_suites

        # This is the cache for bundled program in schema type.
        # User should not access this field directly. Please Use `serialize_to_schema` function instead.
        self._bundled_program_in_schema: Optional[bp_schema.BundledProgram] = None

    def serialize_to_schema(self) -> bp_schema.BundledProgram:
        """Serialize the current Bundled Program into its schema format for further serialization.."""
        # Return cached value if exists
        if self._bundled_program_in_schema is not None:
            return self._bundled_program_in_schema

        if self.executorch_program:
            program = self._extract_program(self.executorch_program)
            program_bytes = self.executorch_program.buffer
        else:
            assert self._pte_file_path is not None
            with open(self._pte_file_path, "rb") as f:
                program_bytes = f.read()
            program = program_bytes
        method_io = self._get_method_io(program)
        self._assert_valid_bundle(method_io, self.method_test_suites)

        bundled_method_test_suites: List[bp_schema.BundledMethodTestSuite] = []

        # Emit data and metadata of bundled tensor
        for method_test_suite in self.method_test_suites:
            plan = method_io[method_test_suite.method_name]
            bundled_test_cases: List[bp_schema.BundledMethodTestCase] = []

            # emit I/O sets for each method test case
            for test_case in method_test_suite.test_cases:
                inputs: List[bp_schema.Value] = []
                expected_outputs: List[bp_schema.Value] = []

                for input_idx, input_val in enumerate(test_case.inputs):
                    if type(input_val) is torch.Tensor:
                        self._emit_bundled_tensor(
                            input_val,
                            plan.inputs[input_idx].dim_order,
                            inputs,
                        )
                    else:
                        self._emit_prim(
                            input_val,
                            inputs,
                        )
                for output_idx, expected_output_tensor in enumerate(
                    test_case.expected_outputs
                ):
                    assert (
                        type(expected_output_tensor) is torch.Tensor
                    ), "Only tensor outputs are currently supported."
                    self._emit_bundled_tensor(
                        expected_output_tensor,
                        plan.outputs[output_idx].dim_order,
                        expected_outputs,
                    )
                bundled_test_cases.append(
                    bp_schema.BundledMethodTestCase(
                        inputs=inputs, expected_outputs=expected_outputs
                    )
                )

            # emit the whole execution plan test
            bundled_method_test_suites.append(
                bp_schema.BundledMethodTestSuite(
                    method_name=method_test_suite.method_name,
                    test_cases=bundled_test_cases,
                )
            )

        self._bundled_program_in_schema = bp_schema.BundledProgram(
            version=BUNDLED_PROGRAM_SCHEMA_VERSION,
            method_test_suites=bundled_method_test_suites,
            program=program_bytes,
        )
        return self._bundled_program_in_schema

    @staticmethod
    def _emit_bundled_tensor(
        tensor: torch.Tensor,
        dim_order: Optional[List[int]],
        bundled_values: List[bp_schema.Value],
    ) -> None:
        assert (
            dim_order is not None
        ), "Program I/O tensor is missing its dimension order"
        sizes = list(tensor.shape)
        # Store bytes in the physical order declared by the method's I/O tensor.
        tensor = tensor.permute(dim_order).contiguous().resolve_conj().resolve_neg()
        tensor_data = (
            ctypes.string_at(tensor.data_ptr(), tensor.nbytes) if tensor.nbytes else b""
        )
        bundled_values.append(
            bp_schema.Value(
                val=bp_schema.Tensor(
                    scalar_type=scalar_type_enum(tensor.dtype),
                    sizes=sizes,
                    data=tensor_data,
                    dim_order=list(dim_order),
                ),
            )
        )

    def _emit_prim(self, val: ConfigValue, bundled_values: List[bp_schema.Value]):
        if type(val) is int:
            bundled_values.append(bp_schema.Value(val=bp_schema.Int(int_val=val)))
        elif type(val) is bool:
            bundled_values.append(bp_schema.Value(val=bp_schema.Bool(bool_val=val)))
        elif type(val) is float:
            bundled_values.append(bp_schema.Value(val=bp_schema.Double(double_val=val)))
        else:
            assert 0, "Unsupported primitive type received."

    def _get_method_io(
        self, program: Union[core_schema.Program, bytes]
    ) -> Dict[str, ProgramMethodIO]:
        if isinstance(program, core_schema.Program):
            return {
                plan.name: ProgramMethodIO(
                    inputs=[plan.values[index].val for index in plan.inputs],
                    outputs=[plan.values[index].val for index in plan.outputs],
                )
                for plan in program.execution_plan
            }

        return deserialize_program_method_io(program)

    def _assert_valid_bundle(
        self,
        method_io: Dict[str, ProgramMethodIO],
        method_test_suites: Sequence[MethodTestSuite],
    ) -> None:
        """Check that test cases match the program's method I/O metadata."""
        for suite in method_test_suites:
            assert (
                suite.method_name in method_io
            ), f"Bundled method {suite.method_name!r} was not found in the program."
            plan = method_io[suite.method_name]
            assert all(
                type(value) in supported_program_type_table for value in plan.inputs
            ), "The type of program's input isn't supported."
            assert all(
                type(value) is core_schema.Tensor for value in plan.outputs
            ), "Only supports program with output in Tensor type."

            for test_case in suite.test_cases:
                assert len(test_case.inputs) == len(plan.inputs), (
                    f"The number of bundled inputs shall be {len(plan.inputs)}, "
                    f"but got {len(test_case.inputs)}."
                )
                # Verification indexes one expected value per method output.
                # An empty list allows profiling without verification.
                if test_case.expected_outputs:
                    assert len(test_case.expected_outputs) == len(plan.outputs), (
                        f"The number of bundled expected outputs shall be "
                        f"{len(plan.outputs)}, but got {len(test_case.expected_outputs)}."
                    )
                for name, values, specs in (
                    ("input", test_case.inputs, plan.inputs),
                    ("output", test_case.expected_outputs, plan.outputs),
                ):
                    for index, (value, spec) in enumerate(zip(values, specs)):
                        expected_type = supported_program_type_table[type(spec)]
                        assert type(value) is expected_type, (
                            f"The type of {name} {index} in method "
                            f"{suite.method_name!r} shall be {expected_type}, "
                            f"but got {type(value)}."
                        )
                        if isinstance(spec, core_schema.Tensor):
                            expected_dtype = get_scalar_type(spec.scalar_type)
                            assert value.dtype == expected_dtype, (
                                f"The {name} tensor {index} dtype shall be "
                                f"{expected_dtype}, "
                                f"but got {value.dtype}."
                            )
                            assert value.dim() == len(spec.sizes), (
                                f"The {name} tensor {index} rank shall be "
                                f"{len(spec.sizes)}, but got {value.dim()}."
                            )

    def _extract_program(
        self,
        executorch_program: Union[
            ExecutorchProgram,
            ExecutorchProgramManager,
        ],
    ):
        if isinstance(executorch_program, ExecutorchProgramManager):
            program = executorch_program.executorch_program
        else:
            assert isinstance(executorch_program, ExecutorchProgram)
            program = executorch_program.program
        return program
