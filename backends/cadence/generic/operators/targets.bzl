load("@fbsource//tools/build_defs:platform_defs.bzl", "CXX")
load("@fbsource//xplat/executorch/build:runtime_wrapper.bzl", "runtime")

def define_common_targets():
    # Individual operator targets with optimized dependencies

    # Basic operators (need broadcast_util and scalar_utils)
    runtime.cxx_library(
        name = "op_add",
        srcs = ["op_add.cpp"],
        platforms = CXX,
        deps = [
            "//executorch/kernels/portable/cpu/util:broadcast_util",
            "//executorch/runtime/kernel:kernel_includes",
            "//executorch/kernels/portable/cpu:scalar_utils",
        ],
        visibility = [
            "//executorch/backends/cadence/...",
            "@EXECUTORCH_CLIENTS",
        ],
    )

    runtime.cxx_library(
        name = "op_full",
        srcs = ["op_full.cpp"],
        platforms = CXX,
        deps = [
            "//executorch/runtime/kernel:kernel_includes",
            "//executorch/kernels/portable/cpu:scalar_utils",
        ],
        visibility = [
            "//executorch/backends/cadence/...",
            "@EXECUTORCH_CLIENTS",
        ],
    )

    # Simple operators (only need kernel_includes)
    runtime.cxx_library(
        name = "op_embedding",
        srcs = ["op_embedding.cpp"],
        platforms = CXX,
        deps = [
            "//executorch/runtime/kernel:kernel_includes",
        ],
        visibility = [
            "//executorch/backends/cadence/...",
            "@EXECUTORCH_CLIENTS",
        ],
    )

    runtime.cxx_library(
        name = "op_view_copy",
        srcs = ["op_view_copy.cpp"],
        platforms = CXX,
        deps = [
            "//executorch/runtime/kernel:kernel_includes",
        ],
        visibility = [
            "//executorch/backends/cadence/...",
            "@EXECUTORCH_CLIENTS",
        ],
    )

    # Operators that need the operators.h header and basic runtime
    runtime.cxx_library(
        name = "im2row_out",
        srcs = ["im2row_out.cpp"],
        exported_headers = ["operators.h"],
        platforms = CXX,
        deps = [
            "//executorch/runtime/kernel:kernel_includes",
        ],
        visibility = [
            "//executorch/backends/cadence/...",
            "@EXECUTORCH_CLIENTS",
        ],
    )

    runtime.cxx_library(
        name = "op_requantize_out",
        srcs = ["op_requantize_out.cpp"],
        platforms = CXX,
        deps = [
            "//executorch/runtime/kernel:kernel_includes",
            "//executorch/backends/cadence/generic/kernels:cadence_kernels",
        ],
        visibility = [
            "//executorch/backends/cadence/...",
            "@EXECUTORCH_CLIENTS",
        ],
    )

    # Quantized operators that need cadence kernels for quantize/dequantize
    runtime.cxx_library(
        name = "dequantize_per_tensor",
        srcs = ["dequantize_per_tensor.cpp"],
        exported_headers = ["quantized_ops.h"],
        platforms = CXX,
        deps = [
            "//executorch/runtime/kernel:kernel_includes",
            "//executorch/backends/cadence/generic/kernels:cadence_kernels",
        ],
        visibility = [
            "//executorch/backends/cadence/...",
        ],
    )

    runtime.cxx_library(
        name = "quantize_per_tensor",
        srcs = ["quantize_per_tensor.cpp"],
        exported_headers = ["quantized_ops.h"],
        platforms = CXX,
        deps = [
            "//executorch/runtime/kernel:kernel_includes",
            "//executorch/backends/cadence/generic/kernels:cadence_kernels",
        ],
        visibility = [
            "//executorch/backends/cadence/...",
            "@EXECUTORCH_CLIENTS",
        ],
    )

    runtime.cxx_library(
        name = "quantized_add_out",
        srcs = ["quantized_add_out.cpp"],
        exported_headers = ["operators.h", "quantized_ops.h"],
        platforms = CXX,
        deps = [
            "//executorch/runtime/kernel:kernel_includes",
            "//executorch/backends/cadence/generic/kernels:cadence_kernels",
        ],
        visibility = [
            "//executorch/backends/cadence/...",
            "@EXECUTORCH_CLIENTS",
        ],
    )

    runtime.cxx_library(
        name = "quantized_conv2d_nchw_out",
        srcs = ["quantized_conv2d_nchw_out.cpp"],
        exported_headers = ["operators.h", "quantized_ops.h"],
        platforms = CXX,
        deps = [
            "//executorch/runtime/kernel:kernel_includes",
            "//executorch/backends/cadence/generic/kernels:cadence_kernels",
        ],
        visibility = [
            "//executorch/backends/cadence/...",
            "@EXECUTORCH_CLIENTS",
        ],
    )

    runtime.cxx_library(
        name = "quantized_conv2d_nhwc_out",
        srcs = ["quantized_conv2d_nhwc_out.cpp"],
        exported_headers = ["operators.h", "quantized_ops.h"],
        platforms = CXX,
        deps = [
            "//executorch/runtime/kernel:kernel_includes",
            "//executorch/backends/cadence/generic/kernels:cadence_kernels",
        ],
        visibility = [
            "//executorch/backends/cadence/...",
            "@EXECUTORCH_CLIENTS",
        ],
    )

    runtime.cxx_library(
        name = "quantized_fully_connected_out",
        srcs = ["quantized_fully_connected_out.cpp"],
        exported_headers = ["operators.h", "quantized_ops.h"],
        platforms = CXX,
        deps = [
            "//executorch/runtime/kernel:kernel_includes",
            "//executorch/backends/cadence/generic/kernels:cadence_kernels",
        ],
        visibility = [
            "//executorch/backends/cadence/...",
            "@EXECUTORCH_CLIENTS",
        ],
    )

    runtime.cxx_library(
        name = "quantized_layer_norm",
        srcs = ["quantized_layer_norm.cpp"],
        exported_headers = ["operators.h", "quantized_ops.h"],
        platforms = CXX,
        deps = [
            "//executorch/runtime/kernel:kernel_includes",
            "//executorch/backends/cadence/generic/kernels:cadence_kernels",
        ],
        visibility = [
            "//executorch/backends/cadence/...",
            "@EXECUTORCH_CLIENTS",
        ],
    )

    runtime.cxx_library(
        name = "quantized_linear_out",
        srcs = ["quantized_linear_out.cpp"],
        exported_headers = ["operators.h", "quantized_ops.h"],
        platforms = CXX,
        deps = [
            "//executorch/runtime/kernel:kernel_includes",
            "//executorch/backends/cadence/generic/kernels:cadence_kernels",
        ],
        visibility = [
            "//executorch/backends/cadence/...",
            "@EXECUTORCH_CLIENTS",
        ],
    )

    runtime.cxx_library(
        name = "quantized_matmul_out",
        srcs = ["quantized_matmul_out.cpp"],
        exported_headers = ["operators.h", "quantized_ops.h"],
        platforms = CXX,
        deps = [
            "//executorch/runtime/kernel:kernel_includes",
            "//executorch/backends/cadence/generic/kernels:cadence_kernels",
        ],
        visibility = [
            "//executorch/backends/cadence/...",
            "@EXECUTORCH_CLIENTS",
        ],
    )

    runtime.cxx_library(
        name = "quantized_relu_out",
        srcs = ["quantized_relu_out.cpp"],
        exported_headers = ["operators.h", "quantized_ops.h"],
        platforms = CXX,
        deps = [
            "//executorch/runtime/kernel:kernel_includes",
            "//executorch/backends/cadence/generic/kernels:cadence_kernels",
        ],
        visibility = [
            "//executorch/backends/cadence/...",
            "@EXECUTORCH_CLIENTS",
        ],
    )

    # Combined target for backward compatibility
    # NOTE: cadence_aot_lib now uses individual targets directly for better linking
    runtime.cxx_library(
        name = "cadence_generic_ops",
        srcs = glob([
            "*.cpp",
        ]),
        exported_headers = glob([
            "*.h",
        ]),
        platforms = CXX,
        deps = [
            "//executorch/kernels/portable/cpu/util:broadcast_util",
            "//executorch/runtime/kernel:kernel_includes",
            "//executorch/kernels/portable/cpu:scalar_utils",
            "//executorch/backends/cadence/generic/kernels:cadence_kernels",
        ],
        visibility = [
            "//executorch/backends/cadence/...",
            "@EXECUTORCH_CLIENTS",
        ],
    )
