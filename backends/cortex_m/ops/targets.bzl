# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

load("@fbsource//tools/build_defs:platform_defs.bzl", "CXX")
load("@fbsource//xplat/executorch/build:runtime_wrapper.bzl", "runtime")
load("@fbsource//xplat/executorch/codegen:codegen.bzl", "et_operator_library", "executorch_generated_lib")
load("@fbcode_macros//build_defs:export_files.bzl", "export_file")

_ARM_EMBEDDED_PLATFORMS = ["ovr_config//cpu:arm32-embedded", "ovr_config//cpu:arm32-embedded-fpu"]

# Operators that compile without CMSIS-NN (portable scalar + optional Helium/MVE).
# These can be built and tested on any platform (including x86_64 CI).
_PORTABLE_OPERATORS = [
    "quantize_per_tensor",
    "dequantize_per_tensor",
]


def define_operator_target(name: str):
    needs_cmsis = name not in _PORTABLE_OPERATORS

    cmsis_deps = [
        "//executorch/kernels/portable/cpu:scalar_utils",
        "//executorch/kernels/portable/cpu/util:broadcast_util",
        "//executorch/kernels/portable/cpu/util:elementwise_util",
        "//executorch/kernels/portable/cpu/util:kernel_ops_util",
        "//executorch/kernels/portable/cpu/util:copy_ops_util",
        "//executorch/kernels/portable/cpu/util:padding_util",
        "fbsource//third-party/cmsis-nn:cmsis_header",
        "fbsource//third-party/cmsis-nn:cmsis_nn",
    ] if needs_cmsis else []

    cmsis_headers = ["cortex_m_ops_common.h"] if needs_cmsis else []

    _compat_kwargs = {}
    if needs_cmsis:
        _compat_kwargs["compatible_with"] = _ARM_EMBEDDED_PLATFORMS

    runtime.cxx_library(
        name = "op_{}".format(name),
        srcs = [
            "op_{}.cpp".format(name),
        ],
        headers = cmsis_headers,
        platforms = CXX,
        deps = [
            "//executorch/runtime/kernel:kernel_includes",
        ] + cmsis_deps,
        link_whole = True,
        **_compat_kwargs
    )

OPERATORS = [
    "quantize_per_tensor",
    "dequantize_per_tensor",
    "quantized_add",
    "quantized_mul",
    "minimum",
    "maximum",
    "quantized_linear",
    "softmax",
    "transpose",
    "pad",
    "quantized_conv2d",
    "quantized_depthwise_conv2d",
    "quantized_transpose_conv2d",
    "quantized_avg_pool2d",
    "quantized_max_pool2d",
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
        compatible_with = _ARM_EMBEDDED_PLATFORMS,
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
        ],
        kernel_deps = [
            ":cortex_m_operators",
        ],
        functions_yaml_target = ":operators.yaml",
        platforms = CXX,
        visibility = ["PUBLIC"],
        define_static_targets = True,
        compatible_with = _ARM_EMBEDDED_PLATFORMS,
    )

    executorch_generated_lib(
        name = "cortex_m_no_except_generated_lib",
        deps = [
            ":ops_lib",
        ],
        kernel_deps = [
            ":cortex_m_operators",
        ],
        functions_yaml_target = ":operators.yaml",
        platforms = CXX,
        visibility = ["PUBLIC"],
        define_static_targets = True,
        support_exceptions = False,
        compatible_with = _ARM_EMBEDDED_PLATFORMS,
    )
