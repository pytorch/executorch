#!/usr/bin/env fbpython
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import argparse
import os
import sys
from typing import Any, List

from tools_copy.code_analyzer import gen_oplist_copy_from_core


def main(argv: List[Any]) -> None:
    """This binary is a wrapper for //executorch/codegen/tools/gen_oplist_copy_from_core.py.
    This is needed because we intend to error out for the case where `model_file_list_path`
    is empty or invalid, so that the ExecuTorch build will fail when no selective build target
    is provided as a dependency to ExecuTorch build.
    """
    parser = argparse.ArgumentParser(description="Generate operator lists")
    parser.add_argument(
        "--output_dir",
        help=("The directory to store the output yaml file (selected_operators.yaml)"),
        required=True,
    )
    parser.add_argument(
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

    # check if the build has any dependency on any selective build target. If we have a target, BUCK shold give us either:
    # 1. a yaml file containing selected ops (could be empty), or
    # 2. a non-empty list of yaml files in the `model_file_list_path`.
    # If none of the two things happened, the build target  has no dependency on any selective build and we should error out.
    options = parser.parse_args(argv)
    if os.path.isfile(options.model_file_list_path):
        pass
    else:
        assert (
            options.model_file_list_path[0] == "@"
        ), "model_file_list_path is not a valid file path, or it doesn't start with '@'. This is likely a BUCK issue."
        model_file_list_path = options.model_file_list_path[1:]
        with open(model_file_list_path) as model_list_file:
            model_file_names = model_list_file.read().split()
            assert (
                len(model_file_names) > 0
            ), "BUCK was not able to find any `et_operator_library` in the dependency graph of the current ExecuTorch "
            "build. Please refer to Selective Build wiki page to add at least one."
    gen_oplist_copy_from_core.main(argv)


if __name__ == "__main__":
    main(sys.argv[1:])
