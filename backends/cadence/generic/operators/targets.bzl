load("@fbsource//tools/build_defs:platform_defs.bzl", "CXX")
load("@fbsource//xplat/executorch/build:runtime_wrapper.bzl", "runtime")

def define_common_targets():
    # Individual operator targets with optimized dependencies

    # Type utilities for Cadence quantized operators
    runtime.cxx_library(
        name = "cadence_type_util",
        exported_headers = ["cadence_type_util.h"],
    )

    runtime.cxx_library(
        name = "quantized_op_macros",
        exported_headers = ["quantized_op_macros.h"],
        exported_deps = [
            ":cadence_type_util",
            "//executorch/kernels/portable/cpu/util:broadcast_util",
            "//executorch/runtime/kernel:kernel_includes",
        ]
    )

    runtime.cxx_library(
        name = "quantized_linear",
        exported_headers = ["quantized_linear.h"],
        exported_deps = [
            "//executorch/runtime/kernel:kernel_includes",
            "//executorch/backends/cadence/generic/kernels:cadence_kernels",
        ]
    )

    runtime.cxx_library(
        name = "op_dequantize_per_tensor",
        srcs = ["op_dequantize_per_tensor.cpp"],
        exported_headers = ["op_dequantize_per_tensor.h"],
        platforms = CXX,
        deps = [
            "//executorch/runtime/kernel:kernel_includes",
            "//executorch/backends/cadence/generic/kernels:cadence_kernels",
        ],
        visibility = ["PUBLIC"],
    )

    runtime.cxx_library(
        name = "op_quantize_per_tensor",
        srcs = ["op_quantize_per_tensor.cpp"],
        exported_headers = ["op_quantize_per_tensor.h"],
        platforms = CXX,
        deps = [
            "//executorch/runtime/kernel:kernel_includes",
            "//executorch/backends/cadence/generic/kernels:cadence_kernels",
        ],
        visibility = ["PUBLIC"],
    )

    runtime.cxx_library(
        name = "op_where_scalar",
        srcs = ["op_where_scalar.cpp"],
        exported_headers = ["op_where_scalar.h"],
        platforms = CXX,
        deps = [
            "//executorch/runtime/kernel:kernel_includes",
            "//executorch/runtime/core/exec_aten:lib",
            "//executorch/runtime/kernel:kernel_runtime_context",
        ],
        visibility = ["PUBLIC"],
    )

    runtime.cxx_library(
        name = "op_rope",
        srcs = ["op_rope.cpp"],
        exported_headers = ["op_rope.h"],
        platforms = CXX,
        deps = [
            "//executorch/runtime/kernel:kernel_includes",
            "//executorch/runtime/core/exec_aten:lib",
            "//executorch/runtime/kernel:kernel_runtime_context",
        ],
        visibility = ["PUBLIC"],
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
        visibility = ["PUBLIC"],
    )

    runtime.cxx_library(
        name = "op_roi_align_box_processor",
        srcs = ["op_roi_align_box_processor.cpp"],
        exported_headers = ["op_roi_align_box_processor.h"],
        platforms = CXX,
        deps = [
            "//executorch/runtime/kernel:kernel_includes",
            "//executorch/runtime/core/exec_aten:lib",
            "//executorch/runtime/kernel:kernel_runtime_context",
        ],
        visibility = ["PUBLIC"],
    )

    runtime.cxx_library(
        name = "op_quantized_add",
        srcs = ["op_quantized_add.cpp"],
        exported_headers = ["op_quantized_add.h"],
        platforms = CXX,
        deps = [
            "//executorch/backends/cadence/generic/kernels:cadence_kernels",
            "//executorch/kernels/portable/cpu:scalar_utils",
            "//executorch/runtime/kernel:kernel_includes",
            ":quantized_op_macros",
        ],
        visibility = ["PUBLIC"],
    )

    runtime.cxx_library(
        name = "op_quantized_conv1d",
        srcs = ["op_quantized_conv1d.cpp"],
        exported_headers = ["op_quantized_conv1d.h"],
        platforms = CXX,
        deps = [
            ":cadence_type_util",
            "//executorch/backends/cadence/generic/kernels:cadence_kernels",
            "//executorch/runtime/kernel:kernel_includes",
        ],
        visibility = ["PUBLIC"],
    )

    runtime.cxx_library(
        name = "op_quantized_conv2d",
        srcs = ["op_quantized_conv2d.cpp"],
        exported_headers = ["op_quantized_conv2d.h"],
        platforms = CXX,
        deps = [
            ":cadence_type_util",
            "//executorch/backends/cadence/generic/kernels:cadence_kernels",
            "//executorch/runtime/kernel:kernel_includes",
        ],
        visibility = ["PUBLIC"],
    )

    runtime.cxx_library(
        name = "op_quantized_fully_connected",
        srcs = ["op_quantized_fully_connected.cpp"],
        exported_headers = ["op_quantized_fully_connected.h"],
        platforms = CXX,
        deps = [
            "//executorch/backends/cadence/generic/kernels:cadence_kernels",
            "//executorch/runtime/kernel:kernel_includes",
            ":quantized_linear",
            ":quantized_op_macros",
        ],
        visibility = ["PUBLIC"],
    )

    runtime.cxx_library(
        name = "op_quantized_layer_norm",
        srcs = ["op_quantized_layer_norm.cpp"],
        exported_headers = ["op_quantized_layer_norm.h"],
        platforms = CXX,
        deps = [
            "//executorch/backends/cadence/generic/kernels:cadence_kernels",
            "//executorch/runtime/kernel:kernel_includes",
            ":quantized_op_macros",
        ],
        visibility = ["PUBLIC"],
    )

    runtime.cxx_library(
        name = "op_quantized_linear",
        srcs = ["op_quantized_linear.cpp"],
        exported_headers = ["op_quantized_linear.h"],
        platforms = CXX,
        deps = [
            "//executorch/backends/cadence/generic/kernels:cadence_kernels",
            "//executorch/runtime/kernel:kernel_includes",
            ":quantized_op_macros",
            ":quantized_linear",
        ],
        visibility = ["PUBLIC"],
    )

    runtime.cxx_library(
        name = "op_quantized_relu",
        srcs = ["op_quantized_relu.cpp"],
        exported_headers = ["op_quantized_relu.h"],
        platforms = CXX,
        deps = [
            "//executorch/backends/cadence/generic/kernels:cadence_kernels",
            "//executorch/runtime/kernel:kernel_includes",
            ":quantized_op_macros",
        ],
        visibility = ["PUBLIC"],
    )

    runtime.cxx_library(
        name = "op_quantized_matmul",
        srcs = ["op_quantized_matmul.cpp"],
        exported_headers = ["op_quantized_matmul.h"],
        platforms = CXX,
        deps = [
            "//executorch/backends/cadence/generic/kernels:cadence_kernels",
            "//executorch/runtime/kernel:kernel_includes",
            ":quantized_op_macros",
        ],
        visibility = ["PUBLIC"],
    )

    runtime.cxx_library(
        name = "op_quantized_mul",
        srcs = ["op_quantized_mul.cpp"],
        exported_headers = ["op_quantized_mul.h"],
        platforms = CXX,
        deps = [
            "//executorch/backends/cadence/generic/kernels:cadence_kernels",
            "//executorch/runtime/kernel:kernel_includes",
            "//executorch/kernels/portable/cpu:scalar_utils",
            ":quantized_op_macros",
        ],
        visibility = ["PUBLIC"],
    )

    runtime.cxx_library(
        name = "op_quantized_softmax",
        srcs = ["op_quantized_softmax.cpp"],
        exported_headers = ["op_quantized_softmax.h"],
        platforms = CXX,
        deps = [
            "//executorch/backends/cadence/generic/kernels:cadence_kernels",
            "//executorch/kernels/portable/cpu/util:reduce_util",
            "//executorch/kernels/portable/cpu/util:functional_util",
            "//executorch/runtime/kernel:kernel_includes",
            ":quantized_op_macros",
        ],
        visibility = ["PUBLIC"],
    )

    runtime.cxx_library(
        name = "op_quantized_embedding_byte",
        srcs = ["op_quantized_embedding_byte.cpp"],
        exported_headers = ["op_quantized_embedding_byte.h"],
        platforms = CXX,
        deps = [
            "//executorch/runtime/kernel:kernel_includes",
            ":quantized_op_macros",
        ],
        visibility = ["PUBLIC"],
    )

    runtime.cxx_library(
        name = "op_requantize",
        srcs = ["op_requantize.cpp"],
        exported_headers = ["op_requantize.h"],
        platforms = CXX,
        deps = [
            "//executorch/backends/cadence/generic/kernels:cadence_kernels",
            "//executorch/runtime/kernel:kernel_includes",
            ":quantized_op_macros",
        ],
        visibility = ["PUBLIC"],
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
        visibility = ["PUBLIC"],
    )

    runtime.cxx_library(
        name = "op_conv1d",
        srcs = ["op_conv1d.cpp"],
        exported_headers = ["op_conv1d.h"],
        platforms = CXX,
        deps = [
            "//executorch/runtime/kernel:kernel_includes",
        ],
        visibility = ["PUBLIC"],
    )

    runtime.cxx_library(
        name = "op_conv2d",
        srcs = ["op_conv2d.cpp"],
        exported_headers = ["op_conv2d.h"],
        platforms = CXX,
        deps = [
            "//executorch/runtime/kernel:kernel_includes",
        ],
        visibility = ["PUBLIC"],
    )

    runtime.cxx_library(
        name = "op_conv3d",
        srcs = ["op_conv3d.cpp"],
        exported_headers = ["op_conv3d.h"],
        platforms = CXX,
        deps = [
            "//executorch/runtime/kernel:kernel_includes",
        ],
        visibility = ["PUBLIC"],
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
        visibility = ["PUBLIC"],
    )

    runtime.cxx_library(
        name = "op_fully_connected",
        srcs = ["op_fully_connected.cpp"],
        exported_headers = ["op_fully_connected.h"],
        platforms = CXX,
        deps = [
            "//executorch/runtime/kernel:kernel_includes",
        ],
        visibility = ["PUBLIC"],
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
        visibility = ["PUBLIC"],
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
        visibility = ["PUBLIC"],
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
        visibility = ["PUBLIC"],
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
        visibility = ["PUBLIC"],
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
        visibility = ["PUBLIC"],
    )
