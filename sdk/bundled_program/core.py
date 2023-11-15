# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import ctypes
import typing
from dataclasses import dataclass
from typing import Dict, List, Sequence, Type, Union

import executorch.exir.schema as core_schema

import executorch.sdk.bundled_program.schema as bp_schema

import torch
import torch.fx

from executorch.exir import (
    ExecutorchProgram,
    ExecutorchProgramManager,
    MultiMethodExecutorchProgram,
)
from executorch.exir._serialize import _serialize_pte_binary
from executorch.exir.tensor import get_scalar_type, scalar_type_enum, TensorSpec
from executorch.sdk.bundled_program.config import ConfigValue, MethodTestSuite

from executorch.sdk.bundled_program.version import BUNDLED_PROGRAM_SCHEMA_VERSION

# pyre-ignore
supported_program_type_table: Dict[Type[core_schema.KernelTypes], ConfigValue] = {
    core_schema.Tensor: torch.Tensor,
    core_schema.Int: int,
    core_schema.Double: float,
    core_schema.Bool: bool,
}


@dataclass
class BundledProgram:
    """
    User should not create an instance of BundledProgram directly. Instead, use `create_bundled_program` API.

    Bundled program contains all information needed to execute and verify the program on device.

    Attributes:
        method_test_suites: All test suites for verifying methods.
        executorch_program: ExecutorchProgram-like variable for the program to be verified, including
                            ExecutorchProgram, MultiMethodExecutorchProgram or ExecutorchProgramManager.
        _bundled_program: information will be serialized into a binary format.
    """

    executorch_program: Union[
        ExecutorchProgram,
        MultiMethodExecutorchProgram,
        ExecutorchProgramManager,
    ]
    method_test_suites: Sequence[MethodTestSuite]
    _bundled_program: bp_schema.BundledProgram


def emit_bundled_tensor(
    spec: TensorSpec, bundled_values: List[bp_schema.Value]
) -> None:
    # QuantizedSchema in tensor has deprecated and may not be used anymore.
    # So here we don't emit it.

    if spec.allocated_memory == 0:
        tensor_data: bytes = b""
    else:
        array_type = (
            ctypes.c_char * typing.cast(torch.UntypedStorage, spec.storage).nbytes()
        )
        spec_array = ctypes.cast(
            typing.cast(torch.UntypedStorage, spec.storage).data_ptr(),
            ctypes.POINTER(array_type),
        ).contents
        tensor_data: bytes = bytes(spec_array)

    bundled_values.append(
        bp_schema.Value(
            val=bp_schema.Tensor(
                scalar_type=scalar_type_enum(spec.dtype),
                sizes=spec.shape,
                data=tensor_data,
                dim_order=list(spec.dim_order),
            ),
        )
    )


def emit_prim(val: ConfigValue, bundled_values: List[bp_schema.Value]):
    if type(val) is int:
        bundled_values.append(bp_schema.Value(val=bp_schema.Int(int_val=val)))
    elif type(val) is bool:
        bundled_values.append(bp_schema.Value(val=bp_schema.Bool(bool_val=val)))
    elif type(val) is float:
        bundled_values.append(bp_schema.Value(val=bp_schema.Double(double_val=val)))
    else:
        assert 0, "Unsupported primitive type received."


def get_program_input(
    program: core_schema.Program, plan_idx: int, input_idx: int
) -> core_schema.KernelTypes:
    return (
        program.execution_plan[plan_idx]
        .values[program.execution_plan[plan_idx].inputs[input_idx]]
        .val
    )


def get_program_output(
    program: core_schema.Program, plan_idx: int, output_idx: int
) -> core_schema.KernelTypes:
    return (
        program.execution_plan[plan_idx]
        .values[program.execution_plan[plan_idx].outputs[output_idx]]
        .val
    )


def get_input_dtype(
    program: core_schema.Program, plan_idx: int, input_idx: int
) -> torch.dtype:
    # pyre-fixme[16]: now assert all input and outputs is in tenor type. Support multuple datatypes in the future.
    return get_scalar_type(get_program_input(program, plan_idx, input_idx).scalar_type)


def get_input_type(program: core_schema.Program, plan_idx: int, input_idx: int) -> type:
    type_lookup = {
        core_schema.Int: int,
        core_schema.Bool: bool,
        core_schema.Double: float,
    }
    # pyre-fixme[6]: Incompatible parameter type [6]: In call `dict.__getitem__`, for 1st positional only parameter
    # expected `Type[Union[core_schema.Bool, core_schema.Double, core_schema.Int]]` but got `Type[Union[core_schema.Bool, core_schema.Double, core_schema.Int, core_schema.Tensor, BoolList, DoubleList,
    # IntList, Null, OptionalTensorList, String, TensorList]]`.
    return type_lookup[type(get_program_input(program, plan_idx, input_idx))]


def get_output_dtype(
    program: core_schema.Program, plan_idx: int, output_idx: int
) -> torch.dtype:
    return get_scalar_type(
        # pyre-ignore[16]: now assert all outputs is in tensor type.
        get_program_output(program, plan_idx, output_idx).scalar_type
    )


def assert_valid_bundle(
    program: core_schema.Program,
    method_test_suites: Sequence[MethodTestSuite],
) -> None:
    """Check if the program and method_test_suites matches each other.

    Other checks not related to correspondence are done in config.py

    Args:
        program: The program to be bundled.
        method_test_suites: The testcases for specific methods to be bundled.

    """

    method_name_of_program = {e.name for e in program.execution_plan}
    method_name_of_test_suites = {t.method_name for t in method_test_suites}

    assert method_name_of_test_suites.issubset(
        method_name_of_program
    ), f"All method names in bundled config should be found in program.execution_plan, \
         but {str(method_name_of_test_suites - method_name_of_program)} does not include."

    # check if method_test_suites has been sorted in ascending alphabetical order of method name.
    for test_suite_id in range(1, len(method_test_suites)):
        assert (
            method_test_suites[test_suite_id - 1].method_name
            <= method_test_suites[test_suite_id].method_name
        ), f"The method name of test suite should be sorted in ascending alphabetical \
            order of method name, but {test_suite_id-1}-th and {test_suite_id}-th method_test_suite aren't."

    # Check if the inputs' type meet Program's requirement
    for method_test_suite in method_test_suites:

        # Get the method with same method name as method_test_suite
        program_plan_id = -1
        for plan in program.execution_plan:
            if plan.name == method_test_suite.method_name:
                program_plan_id = program.execution_plan.index(plan)
                break

        # Raise Assertion Error if can not find the method with same method_name as method_test_suite in program.
        assert (
            program_plan_id != -1
        ), f"method_test_suites has testcases for method {method_test_suite.method_name}, but can not find it in the given program. All method names in the program are {', '.join([p.name for p in program.execution_plan])}."

        plan = program.execution_plan[program_plan_id]

        # Check if the type of Program's input is supported
        for index in range(len(plan.inputs)):
            assert (
                type(get_program_input(program, program_plan_id, index))
                in supported_program_type_table
            ), "The type of program's input isn't supported."

        # Check if the type of Program's output is supported
        for index in range(len(plan.outputs)):
            assert (
                type(get_program_output(program, program_plan_id, index))
                == core_schema.Tensor
            ), "Only supports program with output in Tensor type."

        # Check if the I/O sets of each execution plan test match program's requirement.
        for i in range(len(method_test_suite.test_cases)):
            cur_plan_test_inputs = method_test_suite.test_cases[i].inputs
            cur_plan_test_expected_outputs = method_test_suite.test_cases[
                i
            ].expected_outputs

            assert len(plan.inputs) == len(
                cur_plan_test_inputs
            ), "The number of input in each bundled set and Program shall equal, but get {} and {}".format(
                len(plan.inputs),
                len(cur_plan_test_inputs),
            )

            # Check if bundled input in the current exeution plan test share same type as input in Program
            for j in range(len(cur_plan_test_inputs)):
                assert (
                    type(cur_plan_test_inputs[j])
                    is supported_program_type_table[
                        type(get_program_input(program, program_plan_id, j))
                    ]
                ), "The type {}-th input in {}-th test set of {}-th execution plan does not meet Program's requirement: expected {} but get {}".format(
                    j,
                    i,
                    program_plan_id,
                    supported_program_type_table[
                        type(get_program_input(program, program_plan_id, j))
                    ],
                    type(cur_plan_test_inputs[j]),
                )

                # type of tensor input should match execution plan
                if type(cur_plan_test_inputs[j]) is torch.Tensor:
                    # pyre-fixme[16]: Undefined attribute [16]: Item `bool` of `typing.Union[bool, float, int, torch._tensor.Tensor]`
                    # has no attribute `dtype`.
                    assert cur_plan_test_inputs[j].dtype == get_input_dtype(
                        program, program_plan_id, j
                    ), "The input tensor {} dtype shall be {}, but now is {}".format(
                        cur_plan_test_inputs[j],
                        get_input_dtype(program, program_plan_id, j),
                        cur_plan_test_inputs[j].dtype,
                    )
                elif type(cur_plan_test_inputs[j]) in (
                    int,
                    bool,
                    float,
                ):
                    assert type(cur_plan_test_inputs[j]) is get_input_type(
                        program, program_plan_id, j
                    ), "The input primitive dtype shall be {}, but now is {}".format(
                        get_input_type(program, program_plan_id, j),
                        type(cur_plan_test_inputs[j]),
                    )

            # Check if bundled expected output in the current exeution plan test share same type as output in Program
            for j in range(len(cur_plan_test_expected_outputs)):
                assert (
                    type(cur_plan_test_expected_outputs[j]) is torch.Tensor
                ), "The {}-th expected output shall be a tensor, but now is {}".format(
                    j, type(cur_plan_test_expected_outputs[j])
                )

                # pyre-fixme[16]: Undefined attribute [16]: Item `bool` of `typing.Union[bool, float, int, torch._tensor.Tensor]`
                # has no attribute `dtype`.
                assert cur_plan_test_expected_outputs[j].dtype == get_output_dtype(
                    program, program_plan_id, j
                ), "The label tensor {} dtype shall be {}, but now is {}".format(
                    cur_plan_test_expected_outputs[j],
                    get_output_dtype(program, program_plan_id, j),
                    cur_plan_test_expected_outputs[j].dtype,
                )


def create_bundled_program(
    executorch_program: Union[
        ExecutorchProgram,
        MultiMethodExecutorchProgram,
        ExecutorchProgramManager,
    ],
    method_test_suites: Sequence[MethodTestSuite],
) -> BundledProgram:
    """Create bp_schema.BundledProgram by bundling the given program and method_test_suites together.

    Args:
        program: The program to be bundled.
        method_test_suites: The testcases for certain methods to be bundled.

    Returns:
        The `BundledProgram` variable contains given ExecuTorch program and test cases.
    """
    if isinstance(executorch_program, ExecutorchProgramManager):
        program = executorch_program.executorch_program
    elif isinstance(executorch_program, ExecutorchProgram):
        program = executorch_program.program
    else:
        assert isinstance(executorch_program, MultiMethodExecutorchProgram)
        program = executorch_program.program

    method_test_suites = sorted(method_test_suites, key=lambda x: x.method_name)

    assert_valid_bundle(program, method_test_suites)

    bundled_method_test_suites: List[bp_schema.BundledMethodTestSuite] = []

    # Emit data and metadata of bundled tensor
    for method_test_suite in method_test_suites:
        bundled_test_cases: List[bp_schema.BundledMethodTestCase] = []

        # emit I/O sets for each method test case
        for i in range(len(method_test_suite.test_cases)):
            inputs: List[bp_schema.Value] = []
            expected_outputs: List[bp_schema.Value] = []

            cur_plan_test_inputs = method_test_suite.test_cases[i].inputs
            cur_plan_test_expected_outputs = method_test_suite.test_cases[
                i
            ].expected_outputs

            for input_val in cur_plan_test_inputs:
                if type(input_val) is torch.Tensor:
                    emit_bundled_tensor(
                        TensorSpec.from_tensor(input_val, const=True),
                        inputs,
                    )
                else:
                    emit_prim(
                        input_val,
                        inputs,
                    )
            for expected_output_tensor in cur_plan_test_expected_outputs:
                assert (
                    type(expected_output_tensor) is torch.Tensor
                ), "Only tensor outputs are currently supported."
                emit_bundled_tensor(
                    TensorSpec.from_tensor(expected_output_tensor, const=True),
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
                method_name=method_test_suite.method_name, test_cases=bundled_test_cases
            )
        )

    program_bytes: bytes = _serialize_pte_binary(program)

    return BundledProgram(
        executorch_program=executorch_program,
        method_test_suites=method_test_suites,
        _bundled_program=bp_schema.BundledProgram(
            version=BUNDLED_PROGRAM_SCHEMA_VERSION,
            method_test_suites=bundled_method_test_suites,
            program=program_bytes,
        ),
    )
