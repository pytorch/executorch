load("@fbsource//tools/build_defs:platform_defs.bzl", "CXX")
load("@fbsource//xplat/executorch/build:runtime_wrapper.bzl", "runtime")

# Deps used by all operators.
# buildifier: keep sorted
COMMON_DEPS = [
    "//executorch/backends/cadence/common:xt_macros",
    "//executorch/kernels/portable/cpu:scalar_utils",
    "//executorch/kernels/portable/cpu/pattern:all_deps",
    "//executorch/kernels/portable/cpu/util:all_deps",
    "//executorch/runtime/kernel:kernel_includes",
    "fbsource//third-party/nnlib-FusionG3/xa_nnlib:libxa_nnlib",
    "fbsource//third-party/nnlib-FusionG3/xa_nnlib:libxa_nnlib_common",
]

def define_common_targets():
    """Defines targets that should be shared between fbcode and xplat.

    The directory containing this targets.bzl file should also contain both
    TARGETS and BUCK files that call this function.
    """

    runtime.cxx_library(
        name = "operators_header",
        exported_headers = ["operators.h"],
        visibility = [
            "//executorch/backends/cadence/...",
        ],
        exported_deps = [
            "//executorch/runtime/core/exec_aten:lib",
            "//executorch/runtime/kernel:kernel_runtime_context",
        ],
    )

    runtime.cxx_library(
        name = "xt_utils",
        exported_headers = ["xt_utils.h"],
        visibility = [
            "//executorch/backends/cadence/...",
        ],
        exported_deps = [
            "//executorch/runtime/core/exec_aten:lib",
            "//executorch/runtime/kernel:kernel_runtime_context",
        ],
    )

    runtime.cxx_library(
        name = "op_add",
        srcs = ["op_add.cpp"],
        platforms = CXX,
        deps = COMMON_DEPS + [
            ":operators_header",
            ":xt_utils",
        ],
        visibility = ["PUBLIC"],
        compatible_with = ["ovr_config//cpu:xtensa"],
    )

    runtime.cxx_library(
        name = "op_cat",
        srcs = ["op_cat.cpp"],
        platforms = CXX,
        deps = COMMON_DEPS + [
            ":operators_header",
            ":xt_utils",
        ],
        visibility = ["PUBLIC"],
        compatible_with = ["ovr_config//cpu:xtensa"],
    )

    runtime.cxx_library(
        name = "op_clamp",
        srcs = ["op_clamp.cpp"],
        platforms = CXX,
        deps = COMMON_DEPS + [
            ":operators_header",
            ":xt_utils",
        ],
        visibility = ["PUBLIC"],
        compatible_with = ["ovr_config//cpu:xtensa"],
    )

    runtime.cxx_library(
        name = "op_dequantize",
        srcs = ["op_dequantize.cpp"],
        platforms = CXX,
        deps = COMMON_DEPS + [
            ":operators_header",
            ":xt_utils",
        ],
        visibility = ["PUBLIC"],
        compatible_with = ["ovr_config//cpu:xtensa"],
    )

    runtime.cxx_library(
        name = "op_div",
        srcs = ["op_div.cpp"],
        platforms = CXX,
        deps = COMMON_DEPS + [
            ":operators_header",
            ":xt_utils",
        ],
        visibility = ["PUBLIC"],
        compatible_with = ["ovr_config//cpu:xtensa"],
    )

    runtime.cxx_library(
        name = "op_exp",
        srcs = ["op_exp.cpp"],
        platforms = CXX,
        deps = COMMON_DEPS + [
            ":operators_header",
            ":xt_utils",
        ],
        visibility = ["PUBLIC"],
        compatible_with = ["ovr_config//cpu:xtensa"],
    )

    runtime.cxx_library(
        name = "op_hardtanh",
        srcs = ["op_hardtanh.cpp"],
        platforms = CXX,
        deps = COMMON_DEPS + [
            ":operators_header",
            ":xt_utils",
        ],
        visibility = ["PUBLIC"],
        compatible_with = ["ovr_config//cpu:xtensa"],
    )

    runtime.cxx_library(
        name = "op_lt",
        srcs = ["op_lt.cpp"],
        platforms = CXX,
        deps = COMMON_DEPS + [
            ":operators_header",
            ":xt_utils",
        ],
        visibility = ["PUBLIC"],
        compatible_with = ["ovr_config//cpu:xtensa"],
    )

    runtime.cxx_library(
        name = "op_mean",
        srcs = ["op_mean.cpp"],
        platforms = CXX,
        deps = COMMON_DEPS + [
            ":operators_header",
            ":xt_utils",
        ],
        visibility = ["PUBLIC"],
        compatible_with = ["ovr_config//cpu:xtensa"],
    )

    runtime.cxx_library(
        name = "op_mul",
        srcs = ["op_mul.cpp"],
        platforms = CXX,
        deps = COMMON_DEPS + [
            ":operators_header",
            ":xt_utils",
        ],
        visibility = ["PUBLIC"],
        compatible_with = ["ovr_config//cpu:xtensa"],
    )

    runtime.cxx_library(
        name = "op_native_layer_norm",
        srcs = ["op_native_layer_norm.cpp"],
        platforms = CXX,
        deps = COMMON_DEPS + [
            ":operators_header",
            ":xt_utils",
        ],
        visibility = ["PUBLIC"],
        compatible_with = ["ovr_config//cpu:xtensa"],
    )

    runtime.cxx_library(
        name = "op_permute_copy",
        srcs = ["op_permute_copy.cpp"],
        platforms = CXX,
        deps = COMMON_DEPS + [
            ":operators_header",
            ":xt_utils",
        ],
        visibility = ["PUBLIC"],
        compatible_with = ["ovr_config//cpu:xtensa"],
    )

    runtime.cxx_library(
        name = "op_quantize",
        srcs = ["op_quantize.cpp"],
        platforms = CXX,
        deps = COMMON_DEPS + [
            ":operators_header",
            ":xt_utils",
        ],
        visibility = ["PUBLIC"],
        compatible_with = ["ovr_config//cpu:xtensa"],
    )

    runtime.cxx_library(
        name = "op_rsqrt",
        srcs = ["op_rsqrt.cpp"],
        platforms = CXX,
        deps = COMMON_DEPS + [
            ":operators_header",
            ":xt_utils",
        ],
        visibility = ["PUBLIC"],
        compatible_with = ["ovr_config//cpu:xtensa"],
    )

    runtime.cxx_library(
        name = "op_sigmoid",
        srcs = ["op_sigmoid.cpp"],
        platforms = CXX,
        deps = COMMON_DEPS + [
            ":operators_header",
            ":xt_utils",
        ],
        visibility = ["PUBLIC"],
        compatible_with = ["ovr_config//cpu:xtensa"],
    )

    runtime.cxx_library(
        name = "op_slice_copy",
        srcs = ["op_slice_copy.cpp"],
        platforms = CXX,
        deps = COMMON_DEPS + [
            ":operators_header",
            ":xt_utils",
        ],
        visibility = ["PUBLIC"],
        compatible_with = ["ovr_config//cpu:xtensa"],
    )

    runtime.cxx_library(
        name = "op_softmax",
        srcs = ["op_softmax.cpp"],
        platforms = CXX,
        deps = COMMON_DEPS + [
            ":operators_header",
            ":xt_utils",
        ],
        visibility = ["PUBLIC"],
        compatible_with = ["ovr_config//cpu:xtensa"],
    )

    runtime.cxx_library(
        name = "op_sqrt",
        srcs = ["op_sqrt.cpp"],
        platforms = CXX,
        deps = COMMON_DEPS + [
            ":operators_header",
            ":xt_utils",
        ],
        visibility = ["PUBLIC"],
        compatible_with = ["ovr_config//cpu:xtensa"],
    )

    runtime.cxx_library(
        name = "op_sub",
        srcs = ["op_sub.cpp"],
        platforms = CXX,
        deps = COMMON_DEPS + [
            ":operators_header",
            ":xt_utils",
        ],
        visibility = ["PUBLIC"],
        compatible_with = ["ovr_config//cpu:xtensa"],
    )

    runtime.cxx_library(
        name = "op_tanh",
        srcs = ["op_tanh.cpp"],
        platforms = CXX,
        deps = COMMON_DEPS + [
            ":operators_header",
            ":xt_utils",
        ],
        visibility = ["PUBLIC"],
        compatible_with = ["ovr_config//cpu:xtensa"],
    )

    runtime.cxx_library(
        name = "op_transpose_copy",
        srcs = ["op_transpose_copy.cpp"],
        platforms = CXX,
        deps = COMMON_DEPS + [
            ":operators_header",
            ":xt_utils",
        ],
        visibility = ["PUBLIC"],
        compatible_with = ["ovr_config//cpu:xtensa"],
    )

    runtime.cxx_library(
        name = "op_where",
        srcs = ["op_where.cpp"],
        platforms = CXX,
        deps = COMMON_DEPS + [
            ":operators_header",
            ":xt_utils",
        ],
        visibility = ["PUBLIC"],
        compatible_with = ["ovr_config//cpu:xtensa"],
    )
