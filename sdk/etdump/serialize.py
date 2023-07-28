# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

import json
import os
import tempfile

# pyre-ignore[21]: Could not find module `executorch.exir.serialize.bindings`.
import executorch.exir.serialize.bindings as bindings  # @manual=//executorch/exir/serialize:bindings

# @manual=fbsource//third-party/pypi/setuptools:setuptools
import pkg_resources

from executorch.exir.serialize._dataclass import _DataclassEncoder, _json_to_dataclass
from executorch.sdk.etdump.schema import ETDump

# The prefix of schema files used for etdump
ETDUMP_SCHEMA_NAME = "etdump_schema"
SCALAR_TYPE_SCHEMA_NAME = "scalar_type"


def _write_schema(d: str, schema_name: str) -> None:
    schema_path = os.path.join(d, "{}.fbs".format(schema_name))
    with open(schema_path, "wb") as schema_file:
        schema_file.write(
            pkg_resources.resource_string(__name__, "{}.fbs".format(schema_name))
        )


def _serialize_from_etdump_to_json(etdump: ETDump) -> str:
    return json.dumps(etdump, cls=_DataclassEncoder)


# from json to etdump
def _deserialize_from_json_to_etdump(etdump_json: bytes) -> ETDump:
    etdump_json = json.loads(etdump_json)
    return _json_to_dataclass(etdump_json, ETDump)


def _convert_to_flatbuffer(etdump_json: str) -> bytes:
    with tempfile.TemporaryDirectory() as d:
        # load given and common schema
        _write_schema(d, ETDUMP_SCHEMA_NAME)
        _write_schema(d, SCALAR_TYPE_SCHEMA_NAME)

        schema_path = os.path.join(d, "{}.fbs".format(ETDUMP_SCHEMA_NAME))
        json_path = os.path.join(d, "{}.json".format(ETDUMP_SCHEMA_NAME))
        with open(json_path, "wb") as json_file:
            json_file.write(etdump_json.encode("ascii"))

        # pyre-ignore
        bindings.flatc_compile(d, schema_path, json_path)
        output_path = os.path.join(d, "{}.etdp".format(ETDUMP_SCHEMA_NAME))
        with open(output_path, "rb") as output_file:
            return output_file.read()


def _convert_from_flatbuffer(etdump_flatbuffer: bytes) -> bytes:
    with tempfile.TemporaryDirectory() as d:
        _write_schema(d, ETDUMP_SCHEMA_NAME)
        _write_schema(d, SCALAR_TYPE_SCHEMA_NAME)

        schema_path = os.path.join(d, "{}.fbs".format(ETDUMP_SCHEMA_NAME))
        bin_path = os.path.join(d, "schema.bin")
        with open(bin_path, "wb") as bin_file:
            bin_file.write(etdump_flatbuffer)
        # pyre-ignore
        bindings.flatc_decompile(d, schema_path, bin_path)
        output_path = os.path.join(d, "schema.json")
        with open(output_path, "rb") as output_file:
            return output_file.read()


def serialize_to_etdump(
    etdump: ETDump,
) -> bytes:
    """
    Given an ETdump python object this function will return a serialized object
    that can then be written to a file.
    Args:
        etdump: ETDump python object that the user wants to serialize.
    Returns:
        Serialized etdump binary blob.
    """
    return _convert_to_flatbuffer(_serialize_from_etdump_to_json(etdump))


def deserialize_from_etdump(data: bytes) -> ETDump:
    """
    Given an etdump binary blob this function will deserialize it and return
    the python object representation of etdump.
    Args:
        data: Serialized etdump binary blob.
    Returns:
        Deserialized ETDump python object.
    """
    return _deserialize_from_json_to_etdump(_convert_from_flatbuffer(data))
