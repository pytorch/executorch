#!/usr/bin/env fbpython
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# Generates a template `functions.yaml` from a model binary. Ignoring all custom ops
import argparse
import os
import sys

from typing import Any, List

import torch
import yaml
from executorch.codegen.tools.yaml_util import BlankLineDumper
from executorch.exir._serialize import _deserialize_pte_binary
from executorch.exir.schema import Operator


def get_operators(model_file: str) -> List[Operator]:
    print("Processing model file: ", model_file)
    with open(model_file, "rb") as f:
        flatbuffer = f.read()
    program = _deserialize_pte_binary(flatbuffer)
    print(f"Program loaded from model file: {model_file}")
    operators = program.execution_plan[0].operators
    return operators


def dump_yaml(model_file: str, output_file: str) -> None:
    ops = get_operators(model_file)
    m = []
    for op in ops:
        if op.name.startswith("aten::"):
            schemas = torch._C._jit_get_schemas_for_operator(op.name)
            m.extend(filter(lambda s: s.overload_name == op.overload, schemas))
        else:
            print(f"Warning: not generating template for {op.name}")
    output = []
    for s in m:
        print(str(s))
        name = s.name.replace("aten::", "torch::executor::")
        output.append(
            {
                "func": str(s),
                "variants": "function",
                "dispatch": {
                    "CPU": f"{name}_{s.overload_name}",
                },
            }
        )
    with open(output_file, "w") as f:
        yaml.dump(
            output,
            f,
            Dumper=BlankLineDumper,
            default_flow_style=False,
            sort_keys=False,
            width=1000,
        )


def main(args: List[Any]) -> None:
    """This binary generates a template functions.yaml which will be consumed by ExecuTorch codegen.
    It reads the model file, deserialize it and dumps all the operators into a new functions.yaml.
    The generated file contains placeholder kernels, it needs to be updated with proper kernel names.
    """
    parser = argparse.ArgumentParser(
        description="Generate operator list from a model file"
    )
    parser.add_argument(
        "--output_path",
        help=("The path to the output yaml file (functions.yaml)"),
        required=True,
    )
    parser.add_argument(
        "--model_file_path",
        help=("Path to an executorch program"),
        required=False,
    )
    options = parser.parse_args(args)
    assert options.model_file_path, "Need to provide a model_file_path."

    assert os.path.isfile(
        options.model_file_path
    ), "The value for --model_file_path needs to be a valid file."
    dump_yaml(
        model_file=options.model_file_path,
        output_file=options.output_path if options.output_path else "./functions.yaml",
    )


if __name__ == "__main__":
    main(sys.argv[1:])
