load("@fbsource//tools/build_defs:platform_defs.bzl", "CXX")
load("@fbsource//xplat/executorch/build:runtime_wrapper.bzl", "runtime")

def define_common_targets():
    # Individual operator targets with optimized dependencies

    runtime.cxx_library(
        name = "im2row_out",
        srcs = ["op_im2row.cpp"],
        exported_headers = ["op_im2row.h"],
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
        name = "op_requantize",
        srcs = ["op_requantize.cpp"],
        exported_headers = ["op_requantize.h"],
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
        name = "dequantize_per_tensor",
        srcs = ["op_dequantize_per_tensor.cpp"],
        exported_headers = ["op_dequantize_per_tensor.h"],
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
        name = "quantize_per_tensor",
        srcs = ["op_quantize_per_tensor.cpp"],
        exported_headers = ["op_quantize_per_tensor.h"],
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

    runtime.cxx_library(
        name = "op_where_scalar",
        srcs = ["op_where_scalar.cpp"],
        exported_headers = ["op_where_scalar.h", "operators.h"],
        platforms = CXX,
        deps = [
            "//executorch/runtime/kernel:kernel_includes",
            "//executorch/runtime/core/exec_aten:lib",
            "//executorch/runtime/kernel:kernel_runtime_context",
        ],
        visibility = [
            "//executorch/backends/cadence/...",
            "@EXECUTORCH_CLIENTS",
        ],
    )

    runtime.cxx_library(
        name = "op_rope",
        srcs = ["op_rope.cpp"],
        exported_headers = ["op_rope.h", "operators.h"],
        platforms = CXX,
        deps = [
            "//executorch/runtime/kernel:kernel_includes",
            "//executorch/runtime/core/exec_aten:lib",
            "//executorch/runtime/kernel:kernel_runtime_context",
        ],
        visibility = [
            "//executorch/backends/cadence/...",
            "@EXECUTORCH_CLIENTS",
        ],
    )

    runtime.cxx_library(
        name = "op_linalg_svd",
        srcs = ["op_linalg_svd.cpp"],
        headers = ["op_linalg_svd.h"],
        platforms = CXX,
        deps = [
            "//executorch/runtime/kernel:kernel_includes",
            "//executorch/runtime/core/exec_aten:lib",
            "//executorch/runtime/core/exec_aten/util:tensor_util",
            "//executorch/runtime/kernel:kernel_runtime_context",
        ],
        visibility = [
            "//executorch/backends/cadence/...",
            "@EXECUTORCH_CLIENTS",
        ],
    )

    runtime.cxx_library(
        name = "op_roi_align_box_processor",
        srcs = ["op_roi_align_box_processor.cpp"],
        exported_headers = ["op_roi_align_box_processor.h", "operators.h"],
        platforms = CXX,
        deps = [
            "//executorch/runtime/kernel:kernel_includes",
            "//executorch/runtime/core/exec_aten:lib",
            "//executorch/runtime/kernel:kernel_runtime_context",
        ],
        visibility = [
            "//executorch/backends/cadence/...",
            "@EXECUTORCH_CLIENTS",
        ],
    )

    # Combined target for backward compatibility
    # NOTE: cadence_aot_lib now uses individual targets directly for better linking
    runtime.cxx_library(
        name = "op_where_scalar",
        srcs = ["op_where_scalar.cpp"],
        exported_headers = ["op_where_scalar.h", "operators.h"],
        platforms = CXX,
        deps = [
            "//executorch/runtime/kernel:kernel_includes",
            "//executorch/runtime/core/exec_aten:lib",
            "//executorch/runtime/kernel:kernel_runtime_context",
        ],
        visibility = [
            "//executorch/backends/cadence/...",
            "@EXECUTORCH_CLIENTS",
        ],
    )

    runtime.cxx_library(
        name = "op_rope",
        srcs = ["op_rope.cpp"],
        exported_headers = ["op_rope.h", "operators.h"],
        platforms = CXX,
        deps = [
            "//executorch/runtime/kernel:kernel_includes",
            "//executorch/runtime/core/exec_aten:lib",
            "//executorch/runtime/kernel:kernel_runtime_context",
        ],
        visibility = [
            "//executorch/backends/cadence/...",
            "@EXECUTORCH_CLIENTS",
        ],
    )

    runtime.cxx_library(
        name = "op_linalg_svd",
        srcs = ["op_linalg_svd.cpp"],
        headers = ["op_linalg_svd.h"],
        platforms = CXX,
        deps = [
            "//executorch/runtime/kernel:kernel_includes",
            "//executorch/runtime/core/exec_aten:lib",
            "//executorch/runtime/core/exec_aten/util:tensor_util",
            "//executorch/runtime/kernel:kernel_runtime_context",
        ],
        visibility = [
            "//executorch/backends/cadence/...",
            "@EXECUTORCH_CLIENTS",
        ],
    )

    runtime.cxx_library(
        name = "op_roi_align_box_processor",
        srcs = ["op_roi_align_box_processor.cpp"],
        exported_headers = ["op_roi_align_box_processor.h", "operators.h"],
        platforms = CXX,
        deps = [
            "//executorch/runtime/kernel:kernel_includes",
            "//executorch/runtime/core/exec_aten:lib",
            "//executorch/runtime/kernel:kernel_runtime_context",
        ],
        visibility = [
            "//executorch/backends/cadence/...",
            "@EXECUTORCH_CLIENTS",
        ],
    )

    runtime.cxx_library(
        name = "op_softmax",
        srcs = ["op_softmax.cpp"],
        exported_headers = ["op_softmax.h"],
        platforms = CXX,
        deps = [
            "//executorch/runtime/kernel:kernel_includes",
            "//executorch/kernels/portable/cpu/util:functional_util",
            "//executorch/kernels/portable/cpu/util:reduce_util",
            "//executorch/runtime/core/exec_aten:lib",
            "//executorch/runtime/kernel:kernel_runtime_context",
        ],
        visibility = [
            "//executorch/backends/cadence/...",
            "@EXECUTORCH_CLIENTS",
        ],
    )

    runtime.cxx_library(
        name = "op_conv1d",
        srcs = ["op_conv1d.cpp"],
        exported_headers = ["op_conv1d.h"],
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
        name = "op_conv2d",
        srcs = ["op_conv2d.cpp"],
        exported_headers = ["op_conv2d.h"],
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
        name = "op_conv3d",
        srcs = ["op_conv3d.cpp"],
        exported_headers = ["op_conv3d.h"],
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
        name = "op_avg_pool2d",
        srcs = ["op_avg_pool2d.cpp"],
        exported_headers = ["op_avg_pool2d.h"],
        platforms = CXX,
        deps = [
            "//executorch/runtime/kernel:kernel_includes",
            "//executorch/runtime/core/exec_aten:lib",
            "//executorch/runtime/kernel:kernel_runtime_context",
        ],
        visibility = [
            "//executorch/backends/cadence/...",
            "@EXECUTORCH_CLIENTS",
        ],
    )

    runtime.cxx_library(
        name = "op_fully_connected",
        srcs = ["op_fully_connected.cpp"],
        exported_headers = ["op_fully_connected.h"],
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
        name = "op_idma_copy",
        srcs = ["op_idma_copy.cpp"],
        exported_headers = ["op_idma_copy.h"],
        platforms = CXX,
        deps = [
            "//executorch/runtime/kernel:kernel_includes",
            "//executorch/runtime/core/exec_aten:lib",
            "//executorch/runtime/kernel:kernel_runtime_context",
        ],
        visibility = [
            "//executorch/backends/cadence/...",
            "@EXECUTORCH_CLIENTS",
        ],
    )

    runtime.cxx_library(
        name = "op_idma_wait",
        srcs = ["op_idma_wait.cpp"],
        exported_headers = ["op_idma_wait.h"],
        platforms = CXX,
        deps = [
            "//executorch/runtime/kernel:kernel_includes",
            "//executorch/runtime/core/exec_aten:lib",
            "//executorch/runtime/kernel:kernel_runtime_context",
        ],
        visibility = [
            "//executorch/backends/cadence/...",
            "@EXECUTORCH_CLIENTS",
        ],
    )

    runtime.cxx_library(
        name = "op_im2row",
        srcs = ["op_im2row.cpp"],
        exported_headers = ["op_im2row.h"],
        platforms = CXX,
        deps = [
            "//executorch/backends/cadence/generic/kernels:cadence_kernels",
            "//executorch/runtime/core/exec_aten:lib",
            "//executorch/runtime/kernel:kernel_runtime_context",
        ],
        exported_deps = [
            "//executorch/runtime/kernel:kernel_includes",
        ],
        visibility = [
            "//executorch/backends/cadence/...",
            "@EXECUTORCH_CLIENTS",
        ],
    )

    runtime.cxx_library(
        name = "op_transposed_im2row",
        srcs = ["op_transposed_im2row.cpp"],
        exported_headers = ["op_transposed_im2row.h"],
        platforms = CXX,
        deps = [
            "//executorch/backends/cadence/generic/kernels:cadence_kernels",
            "//executorch/runtime/core/exec_aten:lib",
            "//executorch/runtime/kernel:kernel_runtime_context",
        ],
        exported_deps = [
            "//executorch/runtime/kernel:kernel_includes",
        ],
        visibility = [
            "//executorch/backends/cadence/...",
            "@EXECUTORCH_CLIENTS",
        ],
    )

    runtime.cxx_library(
        name = "op_transposed_convolution",
        srcs = ["op_transposed_convolution.cpp"],
        exported_headers = ["op_transposed_convolution.h"],
        platforms = CXX,
        deps = [
            "//executorch/backends/cadence/generic/kernels:cadence_kernels",
        ],
        exported_deps = [
            "//executorch/runtime/kernel:kernel_includes",
        ],
        visibility = [
            "//executorch/backends/cadence/...",
            "@EXECUTORCH_CLIENTS",
        ],
    )
