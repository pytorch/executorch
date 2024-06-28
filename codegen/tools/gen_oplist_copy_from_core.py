#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# This is a simplified copy from //xplat/caffe2/tools/code_analyzer/gen_oplist.py
import argparse
import os
import sys
from functools import reduce
from typing import Any, List

import yaml
from torchgen.selective_build.selector import (
    combine_selective_builders,
    SelectiveBuilder,
)


def throw_if_any_op_includes_overloads(selective_builder: SelectiveBuilder) -> None:
    ops = []
    for op_name, op in selective_builder.operators.items():
        if op.include_all_overloads:
            ops.append(op_name)
    if ops:
        raise Exception(  # noqa: TRY002
            (
                "Operators that include all overloads are "
                + "not allowed since --allow-include-all-overloads "
                + "was specified: {}"
            ).format(", ".join(ops))
        )


def main(argv: List[Any]) -> None:
    """This binary generates 3 files:

    1. selected_mobile_ops.h: Primary operators used by templated selective build and Kernel Function
       dtypes captured by tracing
    2. selected_operators.yaml: Selected root and non-root operators (either via tracing or static analysis)
    """
    parser = argparse.ArgumentParser(description="Generate operator lists")
    parser.add_argument(
        "--output-dir",
        "--output_dir",
        help=(
            "The directory to store the output yaml files (selected_mobile_ops.h, "
            + "selected_kernel_dtypes.h, selected_operators.yaml)"
        ),
        required=True,
    )
    parser.add_argument(
        "--model-file-list-path",
        "--model_file_list_path",
        help=(
            "Path to a file that contains the locations of individual "
            + "model YAML files that contain the set of used operators. This "
            + "file path must have a leading @-symbol, which will be stripped "
            + "out before processing."
        ),
        required=True,
    )
    parser.add_argument(
        "--allow-include-all-overloads",
        "--allow_include_all_overloads",
        help=(
            "Flag to allow operators that include all overloads. "
            + "If not set, operators registered without using the traced style will"
            + "break the build."
        ),
        action="store_true",
        default=False,
        required=False,
    )
    options = parser.parse_args(argv)

    if os.path.isfile(options.model_file_list_path):
        print("Processing model file: ", options.model_file_list_path)
        model_dicts = []
        model_dict = yaml.safe_load(open(options.model_file_list_path))
        model_dicts.append(model_dict)
    else:
        print("Processing model directory: ", options.model_file_list_path)
        assert options.model_file_list_path[0] == "@"
        model_file_list_path = options.model_file_list_path[1:]

        model_dicts = []
        with open(model_file_list_path) as model_list_file:
            model_file_names = model_list_file.read().split()
            for model_file_name in model_file_names:
                with open(model_file_name, "rb") as model_file:
                    model_dict = yaml.safe_load(model_file)
                    model_dicts.append(model_dict)

    selective_builders = [SelectiveBuilder.from_yaml_dict(m) for m in model_dicts]

    # We may have 0 selective builders since there may not be any viable
    # pt_operator_library rule marked as a dep for the pt_operator_registry rule.
    # This is potentially an error, and we should probably raise an assertion
    # failure here. However, this needs to be investigated further.
    selective_builder = SelectiveBuilder.from_yaml_dict({})
    if len(selective_builders) > 0:
        selective_builder = reduce(
            combine_selective_builders,
            selective_builders,
        )

    if not options.allow_include_all_overloads:
        throw_if_any_op_includes_overloads(selective_builder)
    with open(
        os.path.join(options.output_dir, "selected_operators.yaml"), "wb"
    ) as out_file:
        out_file.write(
            yaml.safe_dump(
                selective_builder.to_dict(), default_flow_style=False
            ).encode("utf-8"),
        )


if __name__ == "__main__":
    main(sys.argv[1:])
