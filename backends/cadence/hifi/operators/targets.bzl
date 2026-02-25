load("@fbsource//tools/build_defs:platform_defs.bzl", "CXX")
load("@fbsource//xplat/executorch/build:runtime_wrapper.bzl", "runtime")

# Deps used by all operators.
# buildifier: keep sorted
COMMON_DEPS = [
    "//executorch/backends/cadence/common:xt_macros",
    "//executorch/backends/cadence/hifi/kernels:kernels",
    "//executorch/backends/cadence/hifi/third-party/nnlib:nnlib-extensions",
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
        name = "op_atan2",
        srcs = ["op_atan2.cpp"],
        exported_headers = ["operators.h"],
        platforms = CXX,
        deps = COMMON_DEPS,
        visibility = ["PUBLIC"],
        compatible_with = ["ovr_config//cpu:xtensa"],
    )

    runtime.cxx_library(
        name = "op_bitwise_and",
        srcs = ["op_bitwise_and.cpp"],
        exported_headers = ["operators.h"],
        platforms = CXX,
        deps = COMMON_DEPS,
        visibility = ["PUBLIC"],
        compatible_with = ["ovr_config//cpu:xtensa"],
    )

    runtime.cxx_library(
        name = "op_bitwise_or",
        srcs = ["op_bitwise_or.cpp"],
        exported_headers = ["operators.h"],
        platforms = CXX,
        deps = COMMON_DEPS,
        visibility = ["PUBLIC"],
        compatible_with = ["ovr_config//cpu:xtensa"],
    )

    runtime.cxx_library(
        name = "op_bitwise_xor",
        srcs = ["op_bitwise_xor.cpp"],
        exported_headers = ["operators.h"],
        platforms = CXX,
        deps = COMMON_DEPS,
        visibility = ["PUBLIC"],
        compatible_with = ["ovr_config//cpu:xtensa"],
    )

    runtime.cxx_library(
        name = "op_bmm",
        srcs = ["op_bmm.cpp"],
        exported_headers = ["operators.h"],
        platforms = CXX,
        deps = COMMON_DEPS,
        visibility = ["PUBLIC"],
        compatible_with = ["ovr_config//cpu:xtensa"],
    )

    runtime.cxx_library(
        name = "op_cat",
        srcs = ["op_cat.cpp"],
        exported_headers = ["operators.h"],
        platforms = CXX,
        deps = COMMON_DEPS,
        visibility = ["PUBLIC"],
        compatible_with = ["ovr_config//cpu:xtensa"],
    )

    runtime.cxx_library(
        name = "op_clamp",
        srcs = ["op_clamp.cpp"],
        exported_headers = ["operators.h"],
        platforms = CXX,
        deps = COMMON_DEPS,
        visibility = ["PUBLIC"],
        compatible_with = ["ovr_config//cpu:xtensa"],
    )

    runtime.cxx_library(
        name = "op_dequantize_per_tensor",
        srcs = ["op_dequantize_per_tensor.cpp"],
        exported_headers = ["operators.h"],
        platforms = CXX,
        deps = COMMON_DEPS,
        visibility = ["PUBLIC"],
        compatible_with = ["ovr_config//cpu:xtensa"],
    )

    runtime.cxx_library(
        name = "op_dequantize_per_tensor_asym8s",
        srcs = ["op_dequantize_per_tensor_asym8s.cpp"],
        exported_headers = ["operators.h"],
        platforms = CXX,
        deps = COMMON_DEPS,
        visibility = ["PUBLIC"],
        compatible_with = ["ovr_config//cpu:xtensa"],
    )

    runtime.cxx_library(
        name = "op_div",
        srcs = ["op_div.cpp"],
        exported_headers = ["operators.h"],
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
        name = "op_eq",
        srcs = ["op_eq.cpp"],
        exported_headers = ["operators.h"],
        platforms = CXX,
        deps = COMMON_DEPS,
        visibility = ["PUBLIC"],
        compatible_with = ["ovr_config//cpu:xtensa"],
    )

    runtime.cxx_library(
        name = "op_fmod",
        srcs = ["op_fmod.cpp"],
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
        name = "op_ge",
        srcs = ["op_ge.cpp"],
        exported_headers = ["operators.h"],
        platforms = CXX,
        deps = COMMON_DEPS,
        visibility = ["PUBLIC"],
        compatible_with = ["ovr_config//cpu:xtensa"],
    )

    runtime.cxx_library(
        name = "op_gt",
        srcs = ["op_gt.cpp"],
        exported_headers = ["operators.h"],
        platforms = CXX,
        deps = COMMON_DEPS,
        visibility = ["PUBLIC"],
        compatible_with = ["ovr_config//cpu:xtensa"],
    )

    runtime.cxx_library(
        name = "op_hardtanh",
        srcs = ["op_hardtanh.cpp"],
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
        name = "op_le",
        srcs = ["op_le.cpp"],
        exported_headers = ["operators.h"],
        platforms = CXX,
        deps = COMMON_DEPS,
        visibility = ["PUBLIC"],
        compatible_with = ["ovr_config//cpu:xtensa"],
    )

    runtime.cxx_library(
        name = "op_lt",
        srcs = ["op_lt.cpp"],
        exported_headers = ["operators.h"],
        platforms = CXX,
        deps = COMMON_DEPS,
        visibility = ["PUBLIC"],
        compatible_with = ["ovr_config//cpu:xtensa"],
    )

    runtime.cxx_library(
        name = "op_masked_fill",
        srcs = ["op_masked_fill.cpp"],
        exported_headers = ["operators.h"],
        platforms = CXX,
        deps = COMMON_DEPS,
        visibility = ["PUBLIC"],
        compatible_with = ["ovr_config//cpu:xtensa"],
    )

    runtime.cxx_library(
        name = "op_maximum",
        srcs = ["op_maximum.cpp"],
        exported_headers = ["operators.h"],
        platforms = CXX,
        deps = COMMON_DEPS,
        visibility = ["PUBLIC"],
        compatible_with = ["ovr_config//cpu:xtensa"],
    )

    runtime.cxx_library(
        name = "op_mean",
        srcs = ["op_mean.cpp"],
        exported_headers = ["operators.h"],
        platforms = CXX,
        deps = COMMON_DEPS,
        visibility = ["PUBLIC"],
        compatible_with = ["ovr_config//cpu:xtensa"],
    )

    runtime.cxx_library(
        name = "op_minimum",
        srcs = ["op_minimum.cpp"],
        exported_headers = ["operators.h"],
        platforms = CXX,
        deps = COMMON_DEPS,
        visibility = ["PUBLIC"],
        compatible_with = ["ovr_config//cpu:xtensa"],
    )

    runtime.cxx_library(
        name = "op_mm",
        srcs = ["op_mm.cpp"],
        exported_headers = ["operators.h"],
        platforms = CXX,
        deps = COMMON_DEPS,
        visibility = ["PUBLIC"],
        compatible_with = ["ovr_config//cpu:xtensa"],
    )

    runtime.cxx_library(
        name = "op_mul",
        srcs = ["op_mul.cpp"],
        exported_headers = ["operators.h"],
        platforms = CXX,
        deps = COMMON_DEPS,
        visibility = ["PUBLIC"],
        compatible_with = ["ovr_config//cpu:xtensa"],
    )

    runtime.cxx_library(
        name = "op_ne",
        srcs = ["op_ne.cpp"],
        exported_headers = ["operators.h"],
        platforms = CXX,
        deps = COMMON_DEPS,
        visibility = ["PUBLIC"],
        compatible_with = ["ovr_config//cpu:xtensa"],
    )

    runtime.cxx_library(
        name = "op_permute_copy",
        srcs = ["op_permute_copy.cpp"],
        exported_headers = ["operators.h"],
        platforms = CXX,
        deps = COMMON_DEPS,
        visibility = ["PUBLIC"],
        compatible_with = ["ovr_config//cpu:xtensa"],
    )

    runtime.cxx_library(
        name = "op_pow",
        srcs = ["op_pow.cpp"],
        exported_headers = ["operators.h"],
        platforms = CXX,
        deps = COMMON_DEPS,
        visibility = ["PUBLIC"],
        compatible_with = ["ovr_config//cpu:xtensa"],
    )

    runtime.cxx_library(
        name = "op_quantize_per_tensor",
        srcs = ["op_quantize_per_tensor.cpp"],
        exported_headers = ["operators.h"],
        platforms = CXX,
        deps = COMMON_DEPS,
        visibility = ["PUBLIC"],
        compatible_with = ["ovr_config//cpu:xtensa"],
    )

    runtime.cxx_library(
        name = "op_quantize_per_tensor_asym8s",
        srcs = ["op_quantize_per_tensor_asym8s.cpp"],
        exported_headers = ["operators.h"],
        platforms = CXX,
        deps = COMMON_DEPS,
        visibility = ["PUBLIC"],
        compatible_with = ["ovr_config//cpu:xtensa"],
    )

    runtime.cxx_library(
        name = "op_quantized_add_asym8sxasym8s_asym8s_per_tensor_out",
        srcs = ["op_quantized_add_asym8sxasym8s_asym8s_per_tensor_out.cpp"],
        exported_headers = ["operators.h"],
        platforms = CXX,
        deps = COMMON_DEPS,
        visibility = ["PUBLIC"],
        compatible_with = ["ovr_config//cpu:xtensa"],
    )

    runtime.cxx_library(
        name = "op_quantized_add_asym8uxasym8u_asym8u_per_tensor_out",
        srcs = ["op_quantized_add_asym8uxasym8u_asym8u_per_tensor_out.cpp"],
        exported_headers = ["operators.h"],
        platforms = CXX,
        deps = COMMON_DEPS,
        visibility = ["PUBLIC"],
        compatible_with = ["ovr_config//cpu:xtensa"],
    )

    runtime.cxx_library(
        name = "op_quantized_conv1d_ncl_asym8sxsym8s_asym8s_per_tensor_out",
        srcs = ["op_quantized_conv1d_ncl_asym8sxsym8s_asym8s_per_tensor_out.cpp"],
        exported_headers = ["operators.h"],
        platforms = CXX,
        deps = COMMON_DEPS,
        visibility = ["PUBLIC"],
        compatible_with = ["ovr_config//cpu:xtensa"],
    )

    runtime.cxx_library(
        name = "op_quantized_conv1d_ncl_asym8uxsym8u_asym8u_per_tensor_out",
        srcs = ["op_quantized_conv1d_ncl_asym8uxsym8u_asym8u_per_tensor_out.cpp"],
        exported_headers = ["operators.h"],
        platforms = CXX,
        deps = COMMON_DEPS,
        visibility = ["PUBLIC"],
        compatible_with = ["ovr_config//cpu:xtensa"],
    )

    runtime.cxx_library(
        name = "op_quantized_conv1d_nlc_asym8sxsym8s_asym8s_per_tensor_out",
        srcs = ["op_quantized_conv1d_nlc_asym8sxsym8s_asym8s_per_tensor_out.cpp"],
        exported_headers = ["operators.h"],
        platforms = CXX,
        deps = COMMON_DEPS,
        visibility = ["PUBLIC"],
        compatible_with = ["ovr_config//cpu:xtensa"],
    )

    runtime.cxx_library(
        name = "op_quantized_conv1d_nlc_asym8uxsym8u_asym8u_per_tensor_out",
        srcs = ["op_quantized_conv1d_nlc_asym8uxsym8u_asym8u_per_tensor_out.cpp"],
        exported_headers = ["operators.h"],
        platforms = CXX,
        deps = COMMON_DEPS,
        visibility = ["PUBLIC"],
        compatible_with = ["ovr_config//cpu:xtensa"],
    )

    runtime.cxx_library(
        name = "op_quantized_conv2d_nchw_asym8sxsym8s_asym8s_per_tensor_out",
        srcs = ["op_quantized_conv2d_nchw_asym8sxsym8s_asym8s_per_tensor_out.cpp"],
        exported_headers = ["operators.h"],
        platforms = CXX,
        deps = COMMON_DEPS,
        visibility = ["PUBLIC"],
        compatible_with = ["ovr_config//cpu:xtensa"],
    )

    runtime.cxx_library(
        name = "op_quantized_conv2d_nchw_asym8uxsym8u_asym8u_per_tensor_out",
        srcs = ["op_quantized_conv2d_nchw_asym8uxsym8u_asym8u_per_tensor_out.cpp"],
        exported_headers = ["operators.h"],
        platforms = CXX,
        deps = COMMON_DEPS,
        visibility = ["PUBLIC"],
        compatible_with = ["ovr_config//cpu:xtensa"],
    )

    runtime.cxx_library(
        name = "op_quantized_conv2d_nchw_depthwise_asym8sxsym8s_asym8s_per_tensor_out",
        srcs = ["op_quantized_conv2d_nchw_depthwise_asym8sxsym8s_asym8s_per_tensor_out.cpp"],
        exported_headers = ["operators.h"],
        platforms = CXX,
        deps = COMMON_DEPS,
        visibility = ["PUBLIC"],
        compatible_with = ["ovr_config//cpu:xtensa"],
    )

    runtime.cxx_library(
        name = "op_quantized_conv2d_nchw_depthwise_asym8uxsym8u_asym8u_per_tensor_out",
        srcs = ["op_quantized_conv2d_nchw_depthwise_asym8uxsym8u_asym8u_per_tensor_out.cpp"],
        exported_headers = ["operators.h"],
        platforms = CXX,
        deps = COMMON_DEPS,
        visibility = ["PUBLIC"],
        compatible_with = ["ovr_config//cpu:xtensa"],
    )

    runtime.cxx_library(
        name = "op_quantized_conv2d_nchw_dilated_asym8sxsym8s_asym8s_per_tensor_out",
        srcs = ["op_quantized_conv2d_nchw_dilated_asym8sxsym8s_asym8s_per_tensor_out.cpp"],
        exported_headers = ["operators.h"],
        platforms = CXX,
        deps = COMMON_DEPS,
        visibility = ["PUBLIC"],
        compatible_with = ["ovr_config//cpu:xtensa"],
    )

    runtime.cxx_library(
        name = "op_quantized_conv2d_nchw_dilated_asym8uxsym8u_asym8u_per_tensor_out",
        srcs = ["op_quantized_conv2d_nchw_dilated_asym8uxsym8u_asym8u_per_tensor_out.cpp"],
        exported_headers = ["operators.h"],
        platforms = CXX,
        deps = COMMON_DEPS,
        visibility = ["PUBLIC"],
        compatible_with = ["ovr_config//cpu:xtensa"],
    )

    runtime.cxx_library(
        name = "op_quantized_conv2d_nchw_out",
        srcs = ["op_quantized_conv2d_nchw_out.cpp"],
        exported_headers = ["operators.h"],
        platforms = CXX,
        deps = COMMON_DEPS + [
            "//executorch/backends/cadence/generic/operators:op_quantized_conv2d",
        ],
        visibility = ["PUBLIC"],
        compatible_with = ["ovr_config//cpu:xtensa"],
    )

    runtime.cxx_library(
        name = "op_quantized_conv2d_nhwc_asym8sxsym8s_asym8s_per_tensor_out",
        srcs = ["op_quantized_conv2d_nhwc_asym8sxsym8s_asym8s_per_tensor_out.cpp"],
        exported_headers = ["operators.h"],
        platforms = CXX,
        deps = COMMON_DEPS,
        visibility = ["PUBLIC"],
        compatible_with = ["ovr_config//cpu:xtensa"],
    )

    runtime.cxx_library(
        name = "op_quantized_conv2d_nhwc_asym8uxsym8u_asym8u_per_tensor_out",
        srcs = ["op_quantized_conv2d_nhwc_asym8uxsym8u_asym8u_per_tensor_out.cpp"],
        exported_headers = ["operators.h"],
        platforms = CXX,
        deps = COMMON_DEPS,
        visibility = ["PUBLIC"],
        compatible_with = ["ovr_config//cpu:xtensa"],
    )

    runtime.cxx_library(
        name = "op_quantized_conv2d_nhwc_depthwise_asym8sxsym8s_asym8s_per_tensor_out",
        srcs = ["op_quantized_conv2d_nhwc_depthwise_asym8sxsym8s_asym8s_per_tensor_out.cpp"],
        exported_headers = ["operators.h"],
        platforms = CXX,
        deps = COMMON_DEPS,
        visibility = ["PUBLIC"],
        compatible_with = ["ovr_config//cpu:xtensa"],
    )

    runtime.cxx_library(
        name = "op_quantized_conv2d_nhwc_depthwise_asym8uxsym8u_asym8u_per_tensor_out",
        srcs = ["op_quantized_conv2d_nhwc_depthwise_asym8uxsym8u_asym8u_per_tensor_out.cpp"],
        exported_headers = ["operators.h"],
        platforms = CXX,
        deps = COMMON_DEPS,
        visibility = ["PUBLIC"],
        compatible_with = ["ovr_config//cpu:xtensa"],
    )

    runtime.cxx_library(
        name = "op_quantized_conv2d_nhwc_dilated_asym8sxsym8s_asym8s_per_tensor_out",
        srcs = ["op_quantized_conv2d_nhwc_dilated_asym8sxsym8s_asym8s_per_tensor_out.cpp"],
        exported_headers = ["operators.h"],
        platforms = CXX,
        deps = COMMON_DEPS,
        visibility = ["PUBLIC"],
        compatible_with = ["ovr_config//cpu:xtensa"],
    )

    runtime.cxx_library(
        name = "op_quantized_conv2d_nhwc_dilated_asym8uxsym8u_asym8u_per_tensor_out",
        srcs = ["op_quantized_conv2d_nhwc_dilated_asym8uxsym8u_asym8u_per_tensor_out.cpp"],
        exported_headers = ["operators.h"],
        platforms = CXX,
        deps = COMMON_DEPS,
        visibility = ["PUBLIC"],
        compatible_with = ["ovr_config//cpu:xtensa"],
    )

    runtime.cxx_library(
        name = "op_quantized_conv2d_nhwc_out",
        srcs = ["op_quantized_conv2d_nhwc_out.cpp"],
        exported_headers = ["operators.h"],
        platforms = CXX,
        deps = COMMON_DEPS + [
            "//executorch/backends/cadence/generic/operators:op_quantized_conv2d",
        ],
        visibility = ["PUBLIC"],
        compatible_with = ["ovr_config//cpu:xtensa"],
    )

    runtime.cxx_library(
        name = "op_quantized_fully_connected_asym8sxasym8s_asym8s_per_tensor_out",
        srcs = ["op_quantized_fully_connected_asym8sxasym8s_asym8s_per_tensor_out.cpp"],
        exported_headers = ["operators.h"],
        platforms = CXX,
        deps = COMMON_DEPS,
        visibility = ["PUBLIC"],
        compatible_with = ["ovr_config//cpu:xtensa"],
    )

    runtime.cxx_library(
        name = "op_quantized_fully_connected_asym8uxasym8u_asym8u_per_tensor_out",
        srcs = ["op_quantized_fully_connected_asym8uxasym8u_asym8u_per_tensor_out.cpp"],
        exported_headers = ["operators.h"],
        platforms = CXX,
        deps = COMMON_DEPS,
        visibility = ["PUBLIC"],
        compatible_with = ["ovr_config//cpu:xtensa"],
    )

    runtime.cxx_library(
        name = "op_quantized_fully_connected_out",
        srcs = ["op_quantized_fully_connected_out.cpp"],
        exported_headers = ["operators.h"],
        platforms = CXX,
        deps = COMMON_DEPS,
        visibility = ["PUBLIC"],
        compatible_with = ["ovr_config//cpu:xtensa"],
    )

    runtime.cxx_library(
        name = "op_quantized_layer_norm",
        srcs = ["op_quantized_layer_norm.cpp"],
        exported_headers = ["operators.h"],
        platforms = CXX,
        deps = COMMON_DEPS,
        visibility = ["PUBLIC"],
        compatible_with = ["ovr_config//cpu:xtensa"],
    )

    runtime.cxx_library(
        name = "op_quantized_linear_asym8sxasym8s_asym8s_per_tensor_out",
        srcs = ["op_quantized_linear_asym8sxasym8s_asym8s_per_tensor_out.cpp"],
        exported_headers = ["operators.h"],
        platforms = CXX,
        deps = COMMON_DEPS,
        visibility = ["PUBLIC"],
        compatible_with = ["ovr_config//cpu:xtensa"],
    )

    runtime.cxx_library(
        name = "op_quantized_linear_asym8uxasym8u_asym8u_per_tensor_out",
        srcs = ["op_quantized_linear_asym8uxasym8u_asym8u_per_tensor_out.cpp"],
        exported_headers = ["operators.h"],
        platforms = CXX,
        deps = COMMON_DEPS,
        visibility = ["PUBLIC"],
        compatible_with = ["ovr_config//cpu:xtensa"],
    )

    runtime.cxx_library(
        name = "op_quantized_linear_out",
        srcs = ["op_quantized_linear_out.cpp"],
        exported_headers = ["operators.h"],
        platforms = CXX,
        deps = COMMON_DEPS + [
            "//executorch/backends/cadence/generic/operators:op_quantized_linear",
        ],
        visibility = ["PUBLIC"],
        compatible_with = ["ovr_config//cpu:xtensa"],
    )

    runtime.cxx_library(
        name = "op_quantized_matmul_asym8sxasym8s_asym8s_out",
        srcs = ["op_quantized_matmul_asym8sxasym8s_asym8s_out.cpp"],
        exported_headers = ["operators.h"],
        platforms = CXX,
        deps = COMMON_DEPS,
        visibility = ["PUBLIC"],
        compatible_with = ["ovr_config//cpu:xtensa"],
    )

    runtime.cxx_library(
        name = "op_quantized_matmul_asym8uxasym8u_asym8u_out",
        srcs = ["op_quantized_matmul_asym8uxasym8u_asym8u_out.cpp"],
        exported_headers = ["operators.h"],
        platforms = CXX,
        deps = COMMON_DEPS,
        visibility = ["PUBLIC"],
        compatible_with = ["ovr_config//cpu:xtensa"],
    )

    runtime.cxx_library(
        name = "op_quantized_matmul_out",
        srcs = ["op_quantized_matmul_out.cpp"],
        exported_headers = ["op_quantized_matmul_out.h"],
        platforms = CXX,
        deps = COMMON_DEPS + [
            "//executorch/backends/cadence/generic/operators:op_quantized_matmul",
        ],
        visibility = ["PUBLIC"],
        compatible_with = ["ovr_config//cpu:xtensa"],
    )

    runtime.cxx_library(
        name = "op_quantized_relu_asym8s_asym8s_per_tensor_out",
        srcs = ["op_quantized_relu_asym8s_asym8s_per_tensor_out.cpp"],
        exported_headers = ["operators.h"],
        platforms = CXX,
        deps = COMMON_DEPS,
        visibility = ["PUBLIC"],
        compatible_with = ["ovr_config//cpu:xtensa"],
    )

    runtime.cxx_library(
        name = "op_quantized_relu_asym8u_asym8u_per_tensor_out",
        srcs = ["op_quantized_relu_asym8u_asym8u_per_tensor_out.cpp"],
        exported_headers = ["operators.h"],
        platforms = CXX,
        deps = COMMON_DEPS,
        visibility = ["PUBLIC"],
        compatible_with = ["ovr_config//cpu:xtensa"],
    )

    runtime.cxx_library(
        name = "op_quantized_relu_out",
        srcs = ["op_quantized_relu_out.cpp"],
        exported_headers = ["operators.h"],
        platforms = CXX,
        deps = COMMON_DEPS,
        visibility = ["PUBLIC"],
        compatible_with = ["ovr_config//cpu:xtensa"],
    )

    runtime.cxx_library(
        name = "op_remainder",
        srcs = ["op_remainder.cpp"],
        exported_headers = ["operators.h"],
        platforms = CXX,
        deps = COMMON_DEPS,
        visibility = ["PUBLIC"],
        compatible_with = ["ovr_config//cpu:xtensa"],
    )

    runtime.cxx_library(
        name = "op_rsqrt",
        srcs = ["op_rsqrt.cpp"],
        exported_headers = ["operators.h"],
        platforms = CXX,
        deps = COMMON_DEPS,
        visibility = ["PUBLIC"],
        compatible_with = ["ovr_config//cpu:xtensa"],
    )

    runtime.cxx_library(
        name = "op_select_copy",
        srcs = ["op_select_copy.cpp"],
        exported_headers = ["operators.h"],
        platforms = CXX,
        deps = COMMON_DEPS,
        visibility = ["PUBLIC"],
        compatible_with = ["ovr_config//cpu:xtensa"],
    )

    runtime.cxx_library(
        name = "op_sigmoid",
        srcs = ["op_sigmoid.cpp"],
        exported_headers = ["operators.h"],
        platforms = CXX,
        deps = COMMON_DEPS,
        visibility = ["PUBLIC"],
        compatible_with = ["ovr_config//cpu:xtensa"],
    )

    runtime.cxx_library(
        name = "op_slice_copy",
        srcs = ["op_slice_copy.cpp"],
        exported_headers = ["operators.h"],
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
        name = "op_softmax_f32_f32",
        srcs = ["op_softmax_f32_f32.cpp"],
        exported_headers = ["operators.h"],
        platforms = CXX,
        deps = COMMON_DEPS,
        visibility = ["PUBLIC"],
        compatible_with = ["ovr_config//cpu:xtensa"],
    )

    runtime.cxx_library(
        name = "op_split_with_sizes_copy",
        srcs = ["op_split_with_sizes_copy.cpp"],
        exported_headers = ["operators.h"],
        platforms = CXX,
        deps = COMMON_DEPS,
        visibility = ["PUBLIC"],
        compatible_with = ["ovr_config//cpu:xtensa"],
    )

    runtime.cxx_library(
        name = "op_sub",
        srcs = ["op_sub.cpp"],
        exported_headers = ["operators.h"],
        platforms = CXX,
        deps = COMMON_DEPS,
        visibility = ["PUBLIC"],
        compatible_with = ["ovr_config//cpu:xtensa"],
    )

    runtime.cxx_library(
        name = "op_tanh",
        srcs = ["op_tanh.cpp"],
        exported_headers = ["operators.h"],
        platforms = CXX,
        deps = COMMON_DEPS,
        visibility = ["PUBLIC"],
        compatible_with = ["ovr_config//cpu:xtensa"],
    )

    runtime.cxx_library(
        name = "op_transpose_copy",
        srcs = ["op_transpose_copy.cpp"],
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

    runtime.cxx_library(
        name = "op_where",
        srcs = ["op_where.cpp"],
        exported_headers = ["operators.h"],
        platforms = CXX,
        deps = COMMON_DEPS,
        visibility = ["PUBLIC"],
        compatible_with = ["ovr_config//cpu:xtensa"],
    )
