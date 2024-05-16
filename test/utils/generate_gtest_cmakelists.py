#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

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


def format_template(path_to_root, test_srcs):
    """
    Format the template with the given path_to_root and test_srcs.
    """
    with open(os.path.dirname(os.path.abspath(__file__)) + "/OSSTest.cmake.in") as f:
        template = f.read()
    return template.format(
        project_name=calculate_project_name(path_to_root),
        path_to_root=calculate_relative_path(path_to_root),
        test_srcs=" ".join(test_srcs),
    )


CONFIG = [
    (
        "runtime/core/portable_type/test",
        [
            "optional_test.cpp",
            "executor_tensor_test.cpp",
            "half_test.cpp",
            "scalar_test.cpp",
            "tensor_impl_test.cpp",
        ],
    ),
    (
        "runtime/core/test",
        [
            "span_test.cpp",
            "error_handling_test.cpp",
            "event_tracer_test.cpp",
            "freeable_buffer_test.cpp",
            "array_ref_test.cpp",
            "memory_allocator_test.cpp",
            "hierarchical_allocator_test.cpp",
            "evalue_test.cpp",
        ],
    ),
    (
        "runtime/core/exec_aten/util/test",
        [
            "tensor_util_test.cpp",
            "scalar_type_util_test.cpp",
            "operator_impl_example_test.cpp",
            "dim_order_util_test.cpp",
            "../../testing_util/tensor_util.cpp",
        ],
    ),
    (
        "runtime/core/exec_aten/testing_util/test",
        ["tensor_util_test.cpp", "tensor_factory_test.cpp", "../tensor_util.cpp"],
    ),
]


def write_template(path_to_root, test_srcs):
    """
    Write the template to the given path_to_root.
    """
    with open(os.path.join(path_to_root, "CMakeLists.txt"), "w") as f:
        f.write(format_template(path_to_root, test_srcs))


if __name__ == "__main__":
    for path_to_root, test_srcs in CONFIG:
        write_template(path_to_root, test_srcs)
        if shutil.which("cmake-format") is not None:
            subprocess.run(
                ["cmake-format", "-i", path_to_root + "/CMakeLists.txt"], check=True
            )
        else:
            print(f"Please run cmake-format -i {path_to_root}/CMakeLists.txt")
    print("Note: Please update test/run_oss_cpp_tests.sh")
