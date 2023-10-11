# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import json
import os
import tempfile

# pyre-ignore[21]: Could not find module `executorch.exir._serialize._bindings`.
import executorch.exir._serialize._bindings as bindings  # @manual=//executorch/exir/_serialize:_bindings

import pkg_resources

from executorch.backends.qnnpack.serialization.qnnpack_graph_schema import (
    QNNDynamicLinear,
)
from executorch.exir._serialize._dataclass import _DataclassEncoder


def convert_to_flatbuffer(qnn_dynamic_linear: QNNDynamicLinear) -> bytes:
    qnnpack_graph_json = json.dumps(qnn_dynamic_linear, cls=_DataclassEncoder)
    with tempfile.TemporaryDirectory() as d:
        schema_path = os.path.join(d, "schema.fbs")
        with open(schema_path, "wb") as schema_file:
            schema_file.write(pkg_resources.resource_string(__name__, "schema.fbs"))
        json_path = os.path.join(d, "schema.json")
        with open(json_path, "wb") as json_file:
            json_file.write(qnnpack_graph_json.encode("ascii"))
        # pyre-ignore
        bindings.flatc_compile(d, schema_path, json_path)
        output_path = os.path.join(d, "schema.bin")
        with open(output_path, "rb") as output_file:
            return output_file.read()
