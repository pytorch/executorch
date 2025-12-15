# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

load("@fbsource//xplat/executorch/build:runtime_wrapper.bzl", "runtime")
load("@fbsource//tools/build_defs:platform_defs.bzl", "CXX")

OPERATORS = [
    "quantize_per_tensor",
    "dequantize_per_tensor",
]

def define_operator_test_target(op):
    runtime.cxx_test(
        name = "op_{}_test".format(op),
        srcs = [
            "op_{}_test.cpp".format(op),
        ],
        platforms = CXX,
        deps = [
            "//executorch/runtime/kernel:kernel_includes",
            "//executorch/kernels/test:test_util",
            "//executorch/backends/cortex_m/ops:op_{}".format(op),
            "//executorch/backends/cortex_m/ops:op_quantize_per_tensor",
            "//executorch/backends/cortex_m/ops:op_dequantize_per_tensor",
            "//executorch/backends/cortex_m/ops:cortex_m_generated_lib_headers",
        ]
    )

def define_common_targets():
    """Defines targets that should be shared between fbcode and xplat.

    The directory containing this targets.bzl file should also contain both
    TARGETS and BUCK files that call this function.
    """
    for op in OPERATORS:
        define_operator_test_target(op)

    
