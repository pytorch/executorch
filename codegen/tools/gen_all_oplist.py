# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import argparse
import os
import re
import sys
from functools import reduce
from pathlib import Path
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
                + "was not specified: {}"
            ).format(", ".join(ops))
        )


def resolve_model_file_path_to_buck_target(model_file_path: str) -> str:
    real_path = str(Path(model_file_path).resolve(strict=True))
    # try my best to convert to buck target
    prog = re.compile(
        r"/.*/buck-out/.*/(fbsource|fbcode)/[0-9a-f]*/(.*)/__(.*)_et_oplist__/out/selected_operators.yaml"
    )
    match = prog.match(real_path)
    if match:
        return f"{match.group(1)}//{match.group(2)}:{match.group(3)}"
    else:
        return real_path


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
        help=("The directory to store the output yaml file (selected_operators.yaml)"),
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
    parser.add_argument(
        "--check-ops-not-overlapping",
        "--check_ops_not_overlapping",
        help=(
            "Flag to check if the operators in the model file list are overlapping. "
            + "If not set, the script will not error out for overlapping operators."
        ),
        action="store_true",
        default=False,
        required=False,
    )
    options = parser.parse_args(argv)

    # Check if the build has any dependency on any selective build target. If we have a target, BUCK shold give us either:
    # 1. a yaml file containing selected ops (could be empty), or
    # 2. a non-empty list of yaml files in the `model_file_list_path` or
    # 3. a non-empty list of directories in the `model_file_list_path`, with each directory containing a `selected_operators.yaml` file.
    # If none of the 3 things happened, the build target  has no dependency on any selective build and we should error out.
    if os.path.isfile(options.model_file_list_path):
        print("Processing model file: ", options.model_file_list_path)
        model_dicts = []
        model_dict = yaml.safe_load(open(options.model_file_list_path))
        model_dicts.append(model_dict)
    else:
        print(
            "Processing model file list or model directory list: ",
            options.model_file_list_path,
        )
        assert (
            options.model_file_list_path[0] == "@"
        ), "model_file_list_path is not a valid file path, or it doesn't start with '@'. This is likely a BUCK issue."

        model_file_list_path = options.model_file_list_path[1:]

        model_dicts = []
        with open(model_file_list_path) as model_list_file:
            model_file_names = model_list_file.read().split()
            assert (
                len(model_file_names) > 0
            ), "BUCK was not able to find any `et_operator_library` in the dependency graph of the current ExecuTorch "
            "build. Please refer to Selective Build wiki page to add at least one."
            for model_file_name in model_file_names:
                if not os.path.isfile(model_file_name):
                    model_file_name = os.path.join(
                        model_file_name, "selected_operators.yaml"
                    )
                print("Processing model file: ", model_file_name)
                assert os.path.isfile(
                    model_file_name
                ), f"{model_file_name} is not a valid file path. This is likely a BUCK issue."
                with open(model_file_name, "rb") as model_file:
                    model_dict = yaml.safe_load(model_file)
                    resolved = resolve_model_file_path_to_buck_target(model_file_name)
                    for op in model_dict["operators"]:
                        model_dict["operators"][op]["debug_info"] = [resolved]
                    model_dicts.append(model_dict)

    selective_builders = [SelectiveBuilder.from_yaml_dict(m) for m in model_dicts]

    # Optionally check if the operators in the model file list are overlapping.
    if options.check_ops_not_overlapping:
        ops = {}
        for model_dict in model_dicts:
            for op_name in model_dict["operators"]:
                if op_name in ops:
                    debug_info_1 = ",".join(ops[op_name]["debug_info"])
                    debug_info_2 = ",".join(
                        model_dict["operators"][op_name]["debug_info"]
                    )
                    error = f"Operator {op_name} is used in 2 models: {debug_info_1} and {debug_info_2}"
                    if "//" not in debug_info_1 and "//" not in debug_info_2:
                        error += "\nWe can't determine what BUCK targets these model files belong to."
                        tail = "."
                    else:
                        error += "\nPlease run the following commands to find out where is the BUCK target being added as a dependency to your target:\n"
                        error += f'\n   buck2 cquery <mode> "allpaths(<target>, {debug_info_1})"'
                        error += f'\n   buck2 cquery <mode> "allpaths(<target>, {debug_info_2})"'
                        tail = "as well as results from BUCK commands listed above."

                    error += (
                        "\n\nIf issue is not resolved, please post in PyTorch Edge Q&A with this error message"
                        + tail
                    )
                    raise Exception(error)  # noqa: TRY002
                ops[op_name] = model_dict["operators"][op_name]
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
