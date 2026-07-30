# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
# Copyright 2025-2026 Arm Limited and/or its affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

# TODO(T138924864): Refactor to unify the serialization for bundled program and executorch program.

import functools
import importlib.resources as _resources
import json
import os
import re
import tempfile
from typing import Any

import executorch.devtools.bundled_program.schema as bp_schema

import executorch.devtools.bundled_program.serialize as serialization_package

import flatbuffers  # pyre-ignore[21]
from executorch.devtools.bundled_program.core import BundledProgram
from executorch.devtools.bundled_program.serialize.generated.bundled_program_flatbuffer import (
    Bool as _Bool,
    BundledMethodTestCase as _BundledMethodTestCase,
    BundledMethodTestSuite as _BundledMethodTestSuite,
    BundledProgram as _BundledProgram,
    Double as _Double,
    Int as _Int,
    Tensor as _Tensor,
    Value as _Value,
    ValueUnion as _ValueUnion,
)
from executorch.exir._serialize._dataclass import _DataclassEncoder, _json_to_dataclass
from executorch.exir._serialize._flatbuffer import _flatc_compile, _flatc_decompile
from executorch.exir._serialize._flatbuffer_program import (
    _coerce_bytes,
    _create_aligned_byte_vector,
)

# The prefix of schema files used for bundled program
BUNDLED_PROGRAM_SCHEMA_NAME = "bundled_program_schema"
SCALAR_TYPE_SCHEMA_NAME = "scalar_type"


@functools.lru_cache(maxsize=1)
def _bundled_program_file_identifier() -> bytes:
    schema = _resources.read_binary(
        serialization_package, f"{BUNDLED_PROGRAM_SCHEMA_NAME}.fbs"
    )
    match = re.search(rb'file_identifier\s+"([^"]+)"', schema)
    if match is None:
        raise ValueError(
            f"Missing file_identifier in {BUNDLED_PROGRAM_SCHEMA_NAME}.fbs"
        )
    file_identifier = match.group(1)
    if len(file_identifier) != 4:
        raise ValueError(
            f"Invalid file_identifier length {len(file_identifier)} "
            f"in {BUNDLED_PROGRAM_SCHEMA_NAME}.fbs"
        )
    return file_identifier


def write_schema(d: str, schema_name: str) -> None:
    schema_path = os.path.join(d, "{}.fbs".format(schema_name))
    with open(schema_path, "wb") as schema_file:
        schema_file.write(
            _resources.read_binary(serialization_package, f"{schema_name}.fbs")
        )


def serialize_from_bundled_program_to_json(
    bundled_program: bp_schema.BundledProgram,
) -> str:
    return json.dumps(bundled_program, cls=_DataclassEncoder)


def deserialize_from_json_to_bundled_program(
    program_json: bytes,
) -> bp_schema.BundledProgram:
    program_json = json.loads(program_json)
    return _json_to_dataclass(program_json, bp_schema.BundledProgram)


def convert_to_flatbuffer(program_json: str) -> bytes:
    with tempfile.TemporaryDirectory() as d:
        # load given and common schema
        write_schema(d, BUNDLED_PROGRAM_SCHEMA_NAME)
        write_schema(d, SCALAR_TYPE_SCHEMA_NAME)

        schema_path = os.path.join(d, "{}.fbs".format(BUNDLED_PROGRAM_SCHEMA_NAME))
        json_path = os.path.join(d, "{}.json".format(BUNDLED_PROGRAM_SCHEMA_NAME))
        with open(json_path, "wb") as json_file:
            json_file.write(program_json.encode("ascii"))
        _flatc_compile(d, schema_path, json_path)
        output_path = os.path.join(d, "{}.bpte".format(BUNDLED_PROGRAM_SCHEMA_NAME))
        with open(output_path, "rb") as output_file:
            return output_file.read()


def convert_from_flatbuffer(program_flatbuffer: bytes) -> bytes:
    with tempfile.TemporaryDirectory() as d:
        write_schema(d, BUNDLED_PROGRAM_SCHEMA_NAME)
        write_schema(d, SCALAR_TYPE_SCHEMA_NAME)

        schema_path = os.path.join(d, "{}.fbs".format(BUNDLED_PROGRAM_SCHEMA_NAME))
        bin_path = os.path.join(d, "schema.bin")
        with open(bin_path, "wb") as bin_file:
            bin_file.write(program_flatbuffer)
        _flatc_decompile(d, schema_path, bin_path)
        output_path = os.path.join(d, "schema.json")
        with open(output_path, "rb") as output_file:
            return output_file.read()


def _pack_tensor(self: Any, builder: Any) -> int:
    if self.sizes is not None:
        _Tensor.TensorStartSizesVector(builder, len(self.sizes))
        for i in reversed(range(len(self.sizes))):
            builder.PrependInt32(self.sizes[i])
        sizes = builder.EndVector()
    if self.data is not None:
        data = _create_aligned_byte_vector(builder, _coerce_bytes(self.data), 16)
    if self.dimOrder is not None:
        dim_order = _create_aligned_byte_vector(
            builder, _coerce_bytes(self.dimOrder), 1
        )

    _Tensor.TensorStart(builder)
    _Tensor.TensorAddScalarType(builder, self.scalarType)
    if self.sizes is not None:
        _Tensor.TensorAddSizes(builder, sizes)
    if self.data is not None:
        _Tensor.TensorAddData(builder, data)
    if self.dimOrder is not None:
        _Tensor.TensorAddDimOrder(builder, dim_order)
    return _Tensor.TensorEnd(builder)


def _pack_bundled_program(self: Any, builder: Any) -> int:
    if self.methodTestSuites is not None:
        method_test_suites_list = [
            method_test_suite.Pack(builder)
            for method_test_suite in self.methodTestSuites
        ]
        _BundledProgram.BundledProgramStartMethodTestSuitesVector(
            builder, len(self.methodTestSuites)
        )
        for i in reversed(range(len(self.methodTestSuites))):
            builder.PrependUOffsetTRelative(method_test_suites_list[i])
        method_test_suites = builder.EndVector()
    if self.program is not None:
        program = _create_aligned_byte_vector(builder, _coerce_bytes(self.program), 32)

    _BundledProgram.BundledProgramStart(builder)
    _BundledProgram.BundledProgramAddVersion(builder, self.version)
    if self.methodTestSuites is not None:
        _BundledProgram.BundledProgramAddMethodTestSuites(builder, method_test_suites)
    if self.program is not None:
        _BundledProgram.BundledProgramAddProgram(builder, program)
    return _BundledProgram.BundledProgramEnd(builder)


@functools.lru_cache(maxsize=1)
def _install_fast_packers() -> None:
    _Tensor.TensorT.Pack = _pack_tensor
    _BundledProgram.BundledProgramT.Pack = _pack_bundled_program


def _convert_tensor(val: bp_schema.Tensor) -> Any:
    result = _Tensor.TensorT()
    result.scalarType = int(val.scalar_type)
    result.sizes = list(val.sizes)
    result.data = _coerce_bytes(val.data)
    result.dimOrder = _coerce_bytes(val.dim_order)
    return result


def _convert_int(val: bp_schema.Int) -> Any:
    result = _Int.IntT()
    result.intVal = val.int_val
    return result


def _convert_bool(val: bp_schema.Bool) -> Any:
    result = _Bool.BoolT()
    result.boolVal = val.bool_val
    return result


def _convert_double(val: bp_schema.Double) -> Any:
    result = _Double.DoubleT()
    result.doubleVal = val.double_val
    return result


def _convert_value_union(val: bp_schema.ValueUnion) -> tuple[int, Any]:
    if isinstance(val, bp_schema.Tensor):
        return _ValueUnion.ValueUnion.Tensor, _convert_tensor(val)
    if isinstance(val, bp_schema.Int):
        return _ValueUnion.ValueUnion.Int, _convert_int(val)
    if isinstance(val, bp_schema.Bool):
        return _ValueUnion.ValueUnion.Bool, _convert_bool(val)
    if isinstance(val, bp_schema.Double):
        return _ValueUnion.ValueUnion.Double, _convert_double(val)
    return _ValueUnion.ValueUnion.NONE, None


def _convert_value(val: bp_schema.Value) -> Any:
    result = _Value.ValueT()
    result.valType, result.val = _convert_value_union(val.val)
    return result


def _convert_method_test_case(val: bp_schema.BundledMethodTestCase) -> Any:
    result = _BundledMethodTestCase.BundledMethodTestCaseT()
    result.inputs = [_convert_value(value) for value in val.inputs]
    result.expectedOutputs = [_convert_value(value) for value in val.expected_outputs]
    return result


def _convert_method_test_suite(val: bp_schema.BundledMethodTestSuite) -> Any:
    result = _BundledMethodTestSuite.BundledMethodTestSuiteT()
    result.methodName = val.method_name
    result.testCases = [
        _convert_method_test_case(test_case) for test_case in val.test_cases
    ]
    return result


def _convert_bundled_program(val: bp_schema.BundledProgram) -> Any:
    result = _BundledProgram.BundledProgramT()
    result.version = val.version
    result.methodTestSuites = [
        _convert_method_test_suite(suite) for suite in val.method_test_suites
    ]
    result.program = _coerce_bytes(val.program)
    return result


def _bundled_program_schema_to_flatbuffer(
    bundled_program: bp_schema.BundledProgram,
) -> bytes:
    _install_fast_packers()
    bundled_program_t = _convert_bundled_program(bundled_program)
    builder = flatbuffers.Builder()
    bundled_program_offset = bundled_program_t.Pack(builder)
    builder.Finish(
        bundled_program_offset,
        file_identifier=_bundled_program_file_identifier(),
    )
    return bytes(builder.Output())


# from bundled program to flatbuffer
def serialize_from_bundled_program_to_flatbuffer(
    bundled_program: BundledProgram,
) -> bytes:
    """
    Serialize a BundledProgram into FlatBuffer binary format.

    Args:
        bundled_program (BundledProgram): The `BundledProgram` variable to be serialized.

    Returns:
        The serialized FlatBuffer binary data in bytes.
    """

    bundled_program_in_schema = bundled_program.serialize_to_schema()

    return _bundled_program_schema_to_flatbuffer(bundled_program_in_schema)


# From flatbuffer to bundled program in schema.
# Please notice here the bundled program is the one in our schema (bp_schema.BundledProgram),
# not the bundled program user interact with (core.bundled_program).
# However there're two concerns for current design:
# 1. the misalignment of serialization input and deserialization out, which may confuse our user.
# 2. the mis-exposion of schema.bundled_program. all classes in schema should not directly
#    interact with user, but the deserialization api returns one.
# TODO(T170042248): Solve the above issues.
def deserialize_from_flatbuffer_to_bundled_program(
    flatbuffer: bytes,
) -> bp_schema.BundledProgram:
    """
    Deserialize a FlatBuffer binary format into a BundledProgram.

    Args:
        flatbuffer (bytes): The FlatBuffer binary data in bytes.

    Returns:
        A `BundledProgram` instance.
    """
    return deserialize_from_json_to_bundled_program(convert_from_flatbuffer(flatbuffer))
