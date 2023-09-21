# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import ctypes
import typing
from typing import Dict, List, Type

import executorch.bundled_program.schema as bp_schema
import executorch.exir.schema as core_schema

import torch
import torch.fx
from executorch.bundled_program.config import (
    BundledConfig,
    ConfigExecutionPlanTest,
    ConfigValue,
)

from executorch.bundled_program.version import BUNDLED_PROGRAM_SCHEMA_VERSION
from executorch.exir._serialize import _serialize_pte_binary

from executorch.exir.tensor import get_scalar_type, scalar_type_enum, TensorSpec

# pyre-ignore
supported_program_type_table: Dict[Type[core_schema.KernelTypes], ConfigValue] = {
    core_schema.Tensor: torch.Tensor,
    core_schema.Int: int,
    core_schema.Double: float,
    core_schema.Bool: bool,
}


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
    if type(val) == int:
        bundled_values.append(bp_schema.Value(val=bp_schema.Int(int_val=val)))
    elif type(val) == bool:
        bundled_values.append(bp_schema.Value(val=bp_schema.Bool(bool_val=val)))
    elif type(val) == float:
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
    bundled_config: BundledConfig,
) -> None:
    """Check if the program and BundledConfig matches each other.

    Other checks not related to correspondence are done in config.py

    Args:
        program: The program to be bundled.
        bundled_config: The config to be bundled.

    """

    program_plan_id = 0
    bp_plan_id = 0

    method_name_of_program = {e.name for e in program.execution_plan}
    method_name_of_test_suites = {
        t.method_name for t in bundled_config.execution_plan_tests
    }

    assert method_name_of_test_suites.issubset(
        method_name_of_program
    ), f"All methods in method_test_suites should be found in program.execution_plan, \
         but {str(method_name_of_test_suites - method_name_of_program)} does not include."

    # check if method_tesdt_suites has been sorted in ascending alphabetical order of method name.
    for bp_plan_id in range(1, len(bundled_config.execution_plan_tests)):
        assert (
            bundled_config.execution_plan_tests[bp_plan_id - 1].method_name
            <= bundled_config.execution_plan_tests[bp_plan_id].method_name
        ), f"The method name of test suite should be sorted in ascending alphabetical \
            order of method name, but {bp_plan_id-1}-th and {bp_plan_id}-th method_test_suite aren't."

    # Check if the inputs' type meet Program's requirement
    while bp_plan_id < len(bundled_config.execution_plan_tests):

        plan_test: ConfigExecutionPlanTest = bundled_config.execution_plan_tests[
            bp_plan_id
        ]
        plan: core_schema.ExecutionPlan = program.execution_plan[program_plan_id]

        # User does not provide testcases for current plan, skip it
        if plan_test.method_name > plan.name:
            program_plan_id += 1
            continue

        # Check if the method name in user provided test matches the one in the original program
        assert (
            plan_test.method_name == plan.name
        ), f"BundledConfig has testcases for method {plan_test.method_name}, but can not find it in the given program. All method names in the program are {', '.join([p.name for p in program.execution_plan])}."

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
        for i in range(len(plan_test.test_sets)):
            cur_plan_test_inputs = plan_test.test_sets[i].inputs
            cur_plan_test_expected_outputs = plan_test.test_sets[i].expected_outputs

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
                    == supported_program_type_table[
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
                if type(cur_plan_test_inputs[j]) == torch.Tensor:
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
                    assert type(cur_plan_test_inputs[j]) == get_input_type(
                        program, program_plan_id, j
                    ), "The input primitive dtype shall be {}, but now is {}".format(
                        get_input_type(program, program_plan_id, j),
                        type(cur_plan_test_inputs[j]),
                    )

            # Check if bundled expected output in the current exeution plan test share same type as output in Program
            for j in range(len(cur_plan_test_expected_outputs)):
                # pyre-fixme[16]: Undefined attribute [16]: Item `bool` of `typing.Union[bool, float, int, torch._tensor.Tensor]`
                # has no attribute `dtype`.
                assert cur_plan_test_expected_outputs[j].dtype == get_output_dtype(
                    program, program_plan_id, j
                ), "The label tensor {} dtype shall be {}, but now is {}".format(
                    cur_plan_test_expected_outputs[j],
                    get_output_dtype(program, program_plan_id, j),
                    cur_plan_test_expected_outputs[j].dtype,
                )

        program_plan_id += 1
        bp_plan_id += 1


def create_bundled_program(
    program: core_schema.Program,
    bundled_config: BundledConfig,
) -> bp_schema.BundledProgram:
    """Create bp_schema.BundledProgram by bundling the given program and bundled_config together.

    Args:
        program: The program to be bundled.
        bundled_config: The config to be bundled.
    """

    assert_valid_bundle(program, bundled_config)

    execution_plan_tests: List[bp_schema.BundledExecutionPlanTest] = []

    # Emit data and metadata of bundled tensor
    for plan_test in bundled_config.execution_plan_tests:
        test_sets: List[bp_schema.BundledIOSet] = []

        # emit I/O sets for each execution plan test
        for i in range(len(plan_test.test_sets)):
            inputs: List[bp_schema.Value] = []
            expected_outputs: List[bp_schema.Value] = []

            cur_plan_test_inputs = plan_test.test_sets[i].inputs
            cur_plan_test_expected_outputs = plan_test.test_sets[i].expected_outputs

            for input_val in cur_plan_test_inputs:
                if type(input_val) == torch.Tensor:
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
                    type(expected_output_tensor) == torch.Tensor
                ), "Only tensor outputs are currently supported."
                emit_bundled_tensor(
                    TensorSpec.from_tensor(expected_output_tensor, const=True),
                    expected_outputs,
                )
            test_sets.append(
                bp_schema.BundledIOSet(inputs=inputs, expected_outputs=expected_outputs)
            )

        # emit the whole execution plan test
        execution_plan_tests.append(
            bp_schema.BundledExecutionPlanTest(
                method_name=plan_test.method_name, test_sets=test_sets
            )
        )

    program_bytes: bytes = _serialize_pte_binary(program)

    return bp_schema.BundledProgram(
        version=BUNDLED_PROGRAM_SCHEMA_VERSION,
        execution_plan_tests=execution_plan_tests,
        program=program_bytes,
    )
