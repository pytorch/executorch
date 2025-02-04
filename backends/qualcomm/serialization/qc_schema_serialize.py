# Copyright (c) Qualcomm Innovation Center, Inc.
# All rights reserved
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import json
import os
import tempfile

import pkg_resources
from executorch.backends.qualcomm.serialization.qc_schema import QnnExecuTorchOptions
from executorch.exir._serialize._dataclass import _DataclassEncoder, _json_to_dataclass
from executorch.exir._serialize._flatbuffer import _flatc_compile, _flatc_decompile


def _convert_to_flatbuffer(obj, schema: str):
    obj_json = json.dumps(obj, cls=_DataclassEncoder)
    with tempfile.TemporaryDirectory() as d:
        schema_path = os.path.join(d, f"{schema}.fbs")
        with open(schema_path, "wb") as schema_file:
            schema_file.write(pkg_resources.resource_string(__name__, f"{schema}.fbs"))
        json_path = os.path.join(d, f"{schema}.json")
        with open(json_path, "wb") as json_file:
            json_file.write(obj_json.encode("ascii"))

        _flatc_compile(d, schema_path, json_path)
        output_path = os.path.join(d, f"{schema}.bin")
        with open(output_path, "rb") as output_file:
            return output_file.read()


def _convert_to_object(flatbuffers: bytes, obj_type, schema: str):
    with tempfile.TemporaryDirectory() as d:
        json_path = os.path.join(d, f"{schema}.json")
        schema_path = os.path.join(d, f"{schema}.fbs")
        bin_path = os.path.join(d, f"{schema}.bin")
        with open(schema_path, "wb") as schema_file:
            schema_file.write(pkg_resources.resource_string(__name__, f"{schema}.fbs"))
        with open(bin_path, "wb") as bin_file:
            bin_file.write(flatbuffers)

        _flatc_decompile(d, schema_path, bin_path, ["--raw-binary"])
        with open(json_path, "rb") as output_file:
            return _json_to_dataclass(json.load(output_file), obj_type)


def option_to_flatbuffer(qnn_executorch_options: QnnExecuTorchOptions) -> bytes:
    return _convert_to_flatbuffer(qnn_executorch_options, "qc_compiler_spec")


def flatbuffer_to_option(flatbuffers: bytes) -> QnnExecuTorchOptions:
    return _convert_to_object(flatbuffers, QnnExecuTorchOptions, "qc_compiler_spec")
