# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
# Copyright 2025 Arm Limited and/or its affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import importlib.resources as _resources
import json
import os
import tempfile

import executorch.backends.apple.mps.serialization as serialization_package
from executorch.backends.apple.mps.serialization.mps_graph_schema import MPSGraph
from executorch.exir._serialize._dataclass import _DataclassEncoder
from executorch.exir._serialize._flatbuffer import _flatc_compile


def convert_to_flatbuffer(mps_graph: MPSGraph) -> bytes:
    mps_graph_json = json.dumps(mps_graph, cls=_DataclassEncoder)
    with tempfile.TemporaryDirectory() as d:
        schema_path = os.path.join(d, "schema.fbs")
        with open(schema_path, "wb") as schema_file:
            schema_file.write(
                _resources.read_binary(serialization_package, "schema.fbs")
            )
        json_path = os.path.join(d, "schema.json")
        with open(json_path, "wb") as json_file:
            json_file.write(mps_graph_json.encode("ascii"))

        _flatc_compile(d, schema_path, json_path)
        output_path = os.path.join(d, "schema.bin")
        with open(output_path, "rb") as output_file:
            return output_file.read()
