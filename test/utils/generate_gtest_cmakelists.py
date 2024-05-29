#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import json
import os
import shutil
import subprocess


def calculate_project_name(path_to_root):
    """
    Get a cmake project name for that path using relative path.
    >>> calculate_project_name("runtime/core/portable_type/test")
    'runtime_core_portable_type_test'
    """
    return path_to_root.replace("/", "_")


def calculate_relative_path(path_to_root):
    """
    Return the relative path from the path_to_root to root (i.e. "..")
    >>> calculate_relative_path("runtime/core/portable_type/test")
    '../../../..'
    """
    return os.path.relpath("/", "/" + path_to_root)


def format_template(path_to_root, test_srcs, additional_libs):
    """
    Format the template with the given path_to_root and test_srcs.
    """
    with open(os.path.dirname(os.path.abspath(__file__)) + "/OSSTest.cmake.in") as f:
        template = f.read()
    return template.format(
        project_name=calculate_project_name(path_to_root),
        path_to_root=calculate_relative_path(path_to_root),
        test_srcs=" ".join(test_srcs),
        additional_libs=" ".join(additional_libs),
    )


def write_template(path_to_root, test_srcs, additional_libs):
    """
    Write the template to the given path_to_root.
    """
    with open(os.path.join(path_to_root, "CMakeLists.txt"), "w") as f:
        f.write(format_template(path_to_root, test_srcs, additional_libs))


def read_config_json(json_path):
    """
    Read the config.json file
    """
    with open(json_path) as f:
        config = json.load(f)
    return config["tests"]


if __name__ == "__main__":
    json_path = os.path.dirname(os.path.abspath(__file__)) + "/OSSTestConfig.json"
    for d in read_config_json(json_path):
        path_to_root = d["directory"]
        test_srcs = d["sources"]
        additional_libs = d.get("additional_libs", [])
        write_template(path_to_root, test_srcs, additional_libs)
        if shutil.which("cmake-format") is not None:
            subprocess.run(
                ["cmake-format", "-i", path_to_root + "/CMakeLists.txt"], check=True
            )
        else:
            print(f"Please run cmake-format -i {path_to_root}/CMakeLists.txt")
