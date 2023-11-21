# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import json
import os
import tempfile

from dataclasses import fields, is_dataclass

import pkg_resources
from executorch.backends.xnnpack.serialization.xnnpack_graph_schema import XNNGraph
from executorch.exir._serialize._dataclass import _DataclassEncoder

from executorch.exir._serialize._flatbuffer import _flatc_compile


def sanity_check_xnngraph_dataclass(table, name: str = ""):
    """
    Make sure no SymInt sneaked in during the preparation of XNNGraph.
    """
    assert is_dataclass(table), f"Expecting a dataclass but got {type(table)}"

    def get_cls_name(obj, field_name=None):
        return (
            f"<{obj.__class__.__name__}>{field_name}"
            if field_name
            else obj.__class__.__name__
        )

    def check_for_sym(obj, name):
        """
        Basic check against the class name of the given obj and
        if it starts from "Sym" or not to catch SymInt the main culprit.
        """
        class_name = get_cls_name(obj)
        assert (
            "Sym" not in class_name
        ), f"Non serializable type {class_name} found at type {name}"

    _name = name if len(name) else get_cls_name(table)

    for field in fields(table):
        o = getattr(table, field.name)

        # Skip str and bytes
        if isinstance(o, str) or isinstance(o, bytes):
            continue

        _name_field = f"{_name}.{get_cls_name(o, field.name)}"

        # Recurse
        if is_dataclass(o):
            sanity_check_xnngraph_dataclass(o, _name_field)

        # Only handles List type, add more if needed
        elif isinstance(o, list):
            for i, v in enumerate(o):
                _name_field_i = _name_field + f"[{i}]"
                # Recurse
                if is_dataclass(v):
                    sanity_check_xnngraph_dataclass(v, f"{_name_field_i}")
                else:
                    check_for_sym(v, _name_field_i)
        else:
            check_for_sym(o, _name_field)


def convert_to_flatbuffer(xnnpack_graph: XNNGraph) -> bytes:
    sanity_check_xnngraph_dataclass(xnnpack_graph)
    xnnpack_graph_json = json.dumps(xnnpack_graph, cls=_DataclassEncoder)
    with tempfile.TemporaryDirectory() as d:
        schema_path = os.path.join(d, "schema.fbs")
        with open(schema_path, "wb") as schema_file:
            schema_file.write(pkg_resources.resource_string(__name__, "schema.fbs"))
        json_path = os.path.join(d, "schema.json")
        with open(json_path, "wb") as json_file:
            json_file.write(xnnpack_graph_json.encode("ascii"))

        _flatc_compile(d, schema_path, json_path)
        output_path = os.path.join(d, "schema.bin")
        with open(output_path, "rb") as output_file:
            return output_file.read()
