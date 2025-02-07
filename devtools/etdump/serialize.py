# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

import json
import os
import tempfile

import pkg_resources
from executorch.devtools.etdump.schema_flatcc import ETDumpFlatCC

from executorch.exir._serialize._dataclass import _DataclassEncoder, _json_to_dataclass

from executorch.exir._serialize._flatbuffer import _flatc_compile, _flatc_decompile

# The prefix of schema files used for etdump
ETDUMP_FLATCC_SCHEMA_NAME = "etdump_schema_flatcc"
SCALAR_TYPE_SCHEMA_NAME = "scalar_type"


def _write_schema(d: str, schema_name: str) -> None:
    schema_path = os.path.join(d, "{}.fbs".format(schema_name))
    with open(schema_path, "wb") as schema_file:
        schema_file.write(
            pkg_resources.resource_string(__name__, "{}.fbs".format(schema_name))
        )


def _serialize_from_etdump_to_json(etdump: ETDumpFlatCC) -> str:
    return json.dumps(etdump, cls=_DataclassEncoder, indent=4)


"""
ETDump FlatCC Schema Implementations
"""


# from json to etdump
def _deserialize_from_json_to_etdump_flatcc(etdump_json: bytes) -> ETDumpFlatCC:
    etdump_json = json.loads(etdump_json)
    return _json_to_dataclass(etdump_json, ETDumpFlatCC)


def _convert_to_flatcc(etdump_json: str) -> bytes:
    with tempfile.TemporaryDirectory() as d:
        # load given and common schema
        _write_schema(d, ETDUMP_FLATCC_SCHEMA_NAME)
        _write_schema(d, SCALAR_TYPE_SCHEMA_NAME)

        schema_path = os.path.join(d, "{}.fbs".format(ETDUMP_FLATCC_SCHEMA_NAME))
        json_path = os.path.join(d, "{}.json".format(ETDUMP_FLATCC_SCHEMA_NAME))
        with open(json_path, "wb") as json_file:
            json_file.write(etdump_json.encode("ascii"))

        _flatc_compile(d, schema_path, json_path)
        output_path = os.path.join(d, "{}.etdp".format(ETDUMP_FLATCC_SCHEMA_NAME))
        with open(output_path, "rb") as output_file:
            return output_file.read()


def _convert_from_flatcc(etdump_flatbuffer: bytes, size_prefixed: bool = True) -> bytes:
    with tempfile.TemporaryDirectory() as d:
        _write_schema(d, ETDUMP_FLATCC_SCHEMA_NAME)
        _write_schema(d, SCALAR_TYPE_SCHEMA_NAME)

        schema_path = os.path.join(d, "{}.fbs".format(ETDUMP_FLATCC_SCHEMA_NAME))
        bin_path = os.path.join(d, "schema.bin")
        with open(bin_path, "wb") as bin_file:
            bin_file.write(etdump_flatbuffer)
        additional_args = []
        if size_prefixed:
            additional_args = ["--size-prefixed"]
        _flatc_decompile(d, schema_path, bin_path, additional_args)
        output_path = os.path.join(d, "schema.json")
        with open(output_path, "rb") as output_file:
            return output_file.read()


def serialize_to_etdump_flatcc(
    etdump: ETDumpFlatCC,
) -> bytes:
    """
    Given an ETdump python object this function will return a serialized object
    that can then be written to a file using the FlatCC schema.
    Args:
        etdump: ETDump python object that the user wants to serialize.
    Returns:
        Serialized etdump binary blob using the FlatCC schema
    """
    return _convert_to_flatcc(_serialize_from_etdump_to_json(etdump))


def deserialize_from_etdump_flatcc(
    data: bytes, size_prefixed: bool = True
) -> ETDumpFlatCC:
    """
    Given an etdump binary blob (constructed using the FlatCC schema) this function will deserialize
    it and return the FlatCC python object representation of etdump.
    Args:
        data: Serialized etdump binary blob.
    Returns:
        Deserialized ETDump python object.
    """
    return _deserialize_from_json_to_etdump_flatcc(
        _convert_from_flatcc(data, size_prefixed)
    )
