load("@fbsource//tools/build_defs:platform_defs.bzl", "CXX")
load("@fbsource//xplat/executorch/build:runtime_wrapper.bzl", "runtime")

def define_common_targets():
    runtime.cxx_library(
        name = "quant_utils",
        exported_headers = ["quant_utils.h"],
        exported_deps = [
            "//executorch/runtime/core/exec_aten:lib",
        ],
        visibility = ["PUBLIC"],
    )

    runtime.cxx_library(
        name = "op_add",
        srcs = ["op_add.cpp"],
        exported_headers = ["op_add.h"],
        platforms = CXX,
        deps = [
            ":quant_utils",
            "//executorch/runtime/kernel:kernel_includes",
        ],
        visibility = ["PUBLIC"],
    )

    runtime.cxx_library(
        name = "op_mul",
        srcs = ["op_mul.cpp"],
        exported_headers = ["op_mul.h"],
        platforms = CXX,
        deps = [
            ":quant_utils",
            "//executorch/runtime/kernel:kernel_includes",
        ],
        visibility = ["PUBLIC"],
    )

    runtime.cxx_library(
        name = "op_relu",
        srcs = ["op_relu.cpp"],
        exported_headers = ["op_relu.h"],
        platforms = CXX,
        deps = [
            ":quant_utils",
            "//executorch/runtime/kernel:kernel_includes",
        ],
        visibility = ["PUBLIC"],
    )

    runtime.cxx_library(
        name = "op_hardswish",
        srcs = ["op_hardswish.cpp"],
        exported_headers = ["op_hardswish.h"],
        platforms = CXX,
        deps = [
            ":quant_utils",
            "//executorch/runtime/kernel:kernel_includes",
        ],
        visibility = ["PUBLIC"],
    )
