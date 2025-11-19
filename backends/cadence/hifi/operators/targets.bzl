load("@fbsource//tools/build_defs:platform_defs.bzl", "CXX")
load("@fbsource//xplat/executorch/build:runtime_wrapper.bzl", "runtime")


def define_operator(name: str, deps: list[str] | None = None) -> None:
    op_name = "op_{}".format(name)

    # Deps used by all operators.
    common_deps = [
        "//executorch/kernels/portable/cpu/util:all_deps",
        "//executorch/kernels/portable/cpu/pattern:all_deps",
        "//executorch/runtime/kernel:kernel_includes",
        "//executorch/kernels/portable/cpu:scalar_utils",
        "//executorch/backends/cadence/hifi/kernels:kernels",
        "//executorch/kernels/portable/cpu/util:dtype_util",
        "//executorch/kernels/portable/cpu/util:elementwise_util",
        "//executorch/kernels/portable/cpu/pattern:bitwise_op",
        "//executorch/backends/cadence/hifi/third-party/nnlib:nnlib-extensions",
        "//executorch/kernels/portable/cpu/pattern:comparison_op",
        "//executorch/backends/cadence/common:xt_macros"
    ]
    if deps == None:
        deps = []

    runtime.cxx_library(
        name = op_name,
        srcs = [op_name + ".cpp"],
        platforms = CXX,
        visibility = [
            "//executorch/backends/cadence/...",
            "@EXECUTORCH_CLIENTS",
        ],
        compatible_with = ["ovr_config//cpu:xtensa"],
        deps = deps + common_deps,
        exported_headers = ["operators.h"],
    )

OPERATORS = [
    "add",
    "atan2",
    "bitwise_and",
    "bitwise_or",
    "bitwise_xor",
    "bmm",
    "cat",
    "clamp",
    "dequantize_per_tensor",
    "dequantize_per_tensor_asym8s",
    "div",
    "embedding",
    "eq",
    "fmod",
    "full",
    "ge",
    "gt",
    "hardtanh",
    "le",
    "lt",
    "masked_fill",
    "maximum",
    "mean",
    "minimum",
    "mm",
    "mul",
    "ne",
    "permute_copy",
    "pow",
    "quantized_conv2d_nchw_out",
    "quantized_conv2d_nchw_asym8sxsym8s_asym8s_per_tensor_out",
    "quantized_conv2d_nchw_asym8uxsym8u_asym8u_per_tensor_out",
    "quantized_conv1d_ncl_asym8sxsym8s_asym8s_per_tensor_out",
    "quantized_conv1d_ncl_asym8uxsym8u_asym8u_per_tensor_out",
    "quantized_conv2d_nchw_depthwise_asym8sxsym8s_asym8s_per_tensor_out",
    "quantized_conv2d_nchw_depthwise_asym8uxsym8u_asym8u_per_tensor_out",
    "quantized_conv2d_nchw_dilated_asym8sxsym8s_asym8s_per_tensor_out",
    "quantized_conv2d_nchw_dilated_asym8uxsym8u_asym8u_per_tensor_out",
    "quantized_conv2d_nhwc_out",
    "quantized_conv2d_nhwc_asym8sxsym8s_asym8s_per_tensor_out",
    "quantized_conv2d_nhwc_asym8uxsym8u_asym8u_per_tensor_out",
    "quantized_conv1d_nlc_asym8sxsym8s_asym8s_per_tensor_out",
    "quantized_conv1d_nlc_asym8uxsym8u_asym8u_per_tensor_out",
    "quantized_conv2d_nhwc_depthwise_asym8sxsym8s_asym8s_per_tensor_out",
    "quantized_conv2d_nhwc_depthwise_asym8uxsym8u_asym8u_per_tensor_out",
    "quantized_conv2d_nhwc_dilated_asym8sxsym8s_asym8s_per_tensor_out",
    "quantized_conv2d_nhwc_dilated_asym8uxsym8u_asym8u_per_tensor_out",
    "quantized_fully_connected_out",
    "quantized_fully_connected_asym8sxasym8s_asym8s_per_tensor_out",
    "quantized_fully_connected_asym8uxasym8u_asym8u_per_tensor_out",
    "quantized_layer_norm",
    "quantized_linear_out",
    "quantized_linear_asym8sxasym8s_asym8s_per_tensor_out",
    "quantized_linear_asym8uxasym8u_asym8u_per_tensor_out",
    "quantized_matmul_out",
    "quantized_matmul_asym8sxasym8s_asym8s_out",
    "quantized_matmul_asym8uxasym8u_asym8u_out",
    "quantized_relu_out",
    "quantized_relu_asym8s_asym8s_per_tensor_out",
    "quantized_relu_asym8u_asym8u_per_tensor_out",
    "quantize_per_tensor",
    "quantize_per_tensor_asym8s",
    "remainder",
    "rsqrt",
    "select_copy",
    "sigmoid",
    "slice_copy",
    "softmax",
    "softmax_f32_f32",
    "split_with_sizes_copy",
    "sub",
    "tanh",
    "view_copy",
    "where"
]

def define_common_targets():
    """Defines targets that should be shared between fbcode and xplat.

    The directory containing this targets.bzl file should also contain both
    TARGETS and BUCK files that call this function.
    """

    # Define build targets for all operators registered in the tables above.
    for op in OPERATORS:
        define_operator(op)
