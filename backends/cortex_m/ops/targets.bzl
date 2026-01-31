# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

load("@fbsource//tools/build_defs:platform_defs.bzl", "CXX")
load("@fbsource//xplat/executorch/build:runtime_wrapper.bzl", "runtime")
load("@fbsource//xplat/executorch/codegen:codegen.bzl", "et_operator_library", "executorch_generated_lib")
load("@fbcode_macros//build_defs:export_files.bzl", "export_file")

def define_operator_target(name: str):
    runtime.cxx_library(
        name = "op_{}".format(name),
        srcs = [
            "op_{}.cpp".format(name),
        ],
        platforms = CXX,
        deps = [
            "//executorch/runtime/kernel:kernel_includes"
        ],
        link_whole = True,
    )

OPERATORS = [
    "quantize_per_tensor",
    "dequantize_per_tensor",
]

def define_common_targets():
    """Defines targets that should be shared between fbcode and xplat.

    The directory containing this targets.bzl file should also contain both
    TARGETS and BUCK files that call this function.
    """
    for op in OPERATORS:
        define_operator_target(op)

    all_op_targets = [":op_{}".format(op) for op in OPERATORS]

    runtime.cxx_library(
        name = "cortex_m_operators",
        srcs = [],
        visibility = ["PUBLIC"],
        platforms = CXX,
        exported_deps = all_op_targets,
    )

    export_file(name = "operators.yaml")

    et_operator_library(
        name = "ops_lib",
        _is_external_target = True,
        ops_schema_yaml_target = ":operators.yaml",
    )

    executorch_generated_lib(
        name = "cortex_m_generated_lib",
        deps = [
            ":ops_lib",
            ":cortex_m_operators",
        ],
        functions_yaml_target = ":operators.yaml",
        platforms = CXX,
        visibility = ["PUBLIC"],
        define_static_targets = True,
    )

    executorch_generated_lib(
        name = "cortex_m_no_except_generated_lib",
        deps = [
            ":ops_lib",
            ":cortex_m_operators",
        ],
        functions_yaml_target = ":operators.yaml",
        platforms = CXX,
        visibility = ["PUBLIC"],
        define_static_targets = True,
        support_exceptions = False,
    )
