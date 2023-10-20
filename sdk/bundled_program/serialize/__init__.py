# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

# TODO(T138924864): Refactor to unify the serialization for bundled program and executorch program.

import json
import os
import tempfile

# @manual=fbsource//third-party/pypi/setuptools:setuptools
import pkg_resources

from executorch.exir._serialize._dataclass import _DataclassEncoder, _json_to_dataclass
from executorch.exir._serialize._flatbuffer import _flatc_compile, _flatc_decompile
from executorch.sdk.bundled_program.schema import BundledProgram

# The prefix of schema files used for bundled program
BUNDLED_PROGRAM_SCHEMA_NAME = "bundled_program_schema"
SCALAR_TYPE_SCHEMA_NAME = "scalar_type"


def write_schema(d: str, schema_name: str) -> None:
    schema_path = os.path.join(d, "{}.fbs".format(schema_name))
    with open(schema_path, "wb") as schema_file:
        schema_file.write(
            pkg_resources.resource_string(__name__, "{}.fbs".format(schema_name))
        )


def serialize_from_bundled_program_to_json(bundled_program: BundledProgram) -> str:
    return json.dumps(bundled_program, cls=_DataclassEncoder)


# from json to Bundled Program
def deserialize_from_json_to_bundled_program(program_json: bytes) -> BundledProgram:
    program_json = json.loads(program_json)
    return _json_to_dataclass(program_json, BundledProgram)


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

    return convert_to_flatbuffer(
        serialize_from_bundled_program_to_json(bundled_program)
    )


# from flatbuffer to general program
def deserialize_from_flatbuffer_to_bundled_program(flatbuffer: bytes) -> BundledProgram:
    """
    Deserialize a FlatBuffer binary format into a BundledProgram.

    Args:
        flatbuffer (bytes): The FlatBuffer binary data in bytes.

    Returns:
        A `BundledProgram` instance.
    """
    return deserialize_from_json_to_bundled_program(convert_from_flatbuffer(flatbuffer))
