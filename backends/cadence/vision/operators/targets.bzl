load("@fbsource//tools/build_defs:platform_defs.bzl", "CXX")
load("@fbsource//xplat/executorch/build:runtime_wrapper.bzl", "runtime")

# Deps used by all operators.
# buildifier: keep sorted
COMMON_DEPS = [
    "//executorch/backends/cadence/vision/kernels:cadence_kernels",
    "//executorch/backends/cadence/vision/third-party:vision-nnlib",
    "//executorch/kernels/portable/cpu:scalar_utils",
    "//executorch/kernels/portable/cpu/pattern:all_deps",
    "//executorch/kernels/portable/cpu/pattern:bitwise_op",
    "//executorch/kernels/portable/cpu/pattern:comparison_op",
    "//executorch/kernels/portable/cpu/util:all_deps",
    "//executorch/kernels/portable/cpu/util:dtype_util",
    "//executorch/kernels/portable/cpu/util:elementwise_util",
    "//executorch/runtime/kernel:kernel_includes",
]

def define_common_targets():
    """Defines targets that should be shared between fbcode and xplat.

    The directory containing this targets.bzl file should also contain both
    TARGETS and BUCK files that call this function.
    """

    runtime.cxx_library(
        name = "op_add",
        srcs = ["op_add.cpp"],
        exported_headers = ["operators.h"],
        platforms = CXX,
        deps = COMMON_DEPS,
        visibility = ["PUBLIC"],
        compatible_with = ["ovr_config//cpu:xtensa"],
    )

    runtime.cxx_library(
        name = "op_dequantize_per_tensor",
        srcs = ["op_dequantize_per_tensor.cpp"],
        exported_headers = ["operators.h", "quantized_ops.h"],
        platforms = CXX,
        deps = COMMON_DEPS,
        visibility = ["PUBLIC"],
        compatible_with = ["ovr_config//cpu:xtensa"],
    )

    runtime.cxx_library(
        name = "op_embedding",
        srcs = ["op_embedding.cpp"],
        exported_headers = ["operators.h"],
        platforms = CXX,
        deps = COMMON_DEPS,
        visibility = ["PUBLIC"],
        compatible_with = ["ovr_config//cpu:xtensa"],
    )

    runtime.cxx_library(
        name = "op_full",
        srcs = ["op_full.cpp"],
        exported_headers = ["operators.h"],
        platforms = CXX,
        deps = COMMON_DEPS,
        visibility = ["PUBLIC"],
        compatible_with = ["ovr_config//cpu:xtensa"],
    )

    runtime.cxx_library(
        name = "op_im2row_out",
        srcs = ["op_im2row_out.cpp"],
        exported_headers = ["operators.h"],
        platforms = CXX,
        deps = COMMON_DEPS,
        visibility = ["PUBLIC"],
        compatible_with = ["ovr_config//cpu:xtensa"],
    )

    runtime.cxx_library(
        name = "op_quantize_per_tensor",
        srcs = ["op_quantize_per_tensor.cpp"],
        exported_headers = ["operators.h", "quantized_ops.h"],
        platforms = CXX,
        deps = COMMON_DEPS,
        visibility = ["PUBLIC"],
        compatible_with = ["ovr_config//cpu:xtensa"],
    )

    runtime.cxx_library(
        name = "op_quantized_conv_out",
        srcs = ["op_quantized_conv_out.cpp"],
        exported_headers = ["operators.h", "quantized_ops.h"],
        platforms = CXX,
        deps = COMMON_DEPS,
        visibility = ["PUBLIC"],
        compatible_with = ["ovr_config//cpu:xtensa"],
    )

    runtime.cxx_library(
        name = "op_quantized_fully_connected_out",
        srcs = ["op_quantized_fully_connected_out.cpp"],
        exported_headers = ["operators.h", "quantized_ops.h"],
        platforms = CXX,
        deps = COMMON_DEPS,
        visibility = ["PUBLIC"],
        compatible_with = ["ovr_config//cpu:xtensa"],
    )

    runtime.cxx_library(
        name = "op_quantized_layer_norm",
        srcs = ["op_quantized_layer_norm.cpp"],
        exported_headers = ["operators.h", "quantized_ops.h"],
        platforms = CXX,
        deps = COMMON_DEPS,
        visibility = ["PUBLIC"],
        compatible_with = ["ovr_config//cpu:xtensa"],
    )

    runtime.cxx_library(
        name = "op_quantized_linear_out",
        srcs = ["op_quantized_linear_out.cpp"],
        exported_headers = ["operators.h", "quantized_ops.h"],
        platforms = CXX,
        deps = COMMON_DEPS,
        visibility = ["PUBLIC"],
        compatible_with = ["ovr_config//cpu:xtensa"],
    )

    runtime.cxx_library(
        name = "op_quantized_matmul_out",
        srcs = ["op_quantized_matmul_out.cpp"],
        exported_headers = ["operators.h", "quantized_ops.h"],
        platforms = CXX,
        deps = COMMON_DEPS,
        visibility = ["PUBLIC"],
        compatible_with = ["ovr_config//cpu:xtensa"],
    )

    runtime.cxx_library(
        name = "op_quantized_relu_out",
        srcs = ["op_quantized_relu_out.cpp"],
        exported_headers = ["operators.h", "quantized_ops.h"],
        platforms = CXX,
        deps = COMMON_DEPS,
        visibility = ["PUBLIC"],
        compatible_with = ["ovr_config//cpu:xtensa"],
    )

    runtime.cxx_library(
        name = "op_requantize_out",
        srcs = ["op_requantize_out.cpp"],
        exported_headers = ["operators.h", "quantized_ops.h"],
        platforms = CXX,
        deps = COMMON_DEPS,
        visibility = ["PUBLIC"],
        compatible_with = ["ovr_config//cpu:xtensa"],
    )

    runtime.cxx_library(
        name = "op_softmax",
        srcs = ["op_softmax.cpp"],
        exported_headers = ["operators.h"],
        platforms = CXX,
        deps = COMMON_DEPS,
        visibility = ["PUBLIC"],
        compatible_with = ["ovr_config//cpu:xtensa"],
    )

    runtime.cxx_library(
        name = "op_view_copy",
        srcs = ["op_view_copy.cpp"],
        exported_headers = ["operators.h"],
        platforms = CXX,
        deps = COMMON_DEPS,
        visibility = ["PUBLIC"],
        compatible_with = ["ovr_config//cpu:xtensa"],
    )
