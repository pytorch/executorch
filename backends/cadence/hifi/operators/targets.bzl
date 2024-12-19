load("@fbsource//tools/build_defs:platform_defs.bzl", "CXX")
load("@fbsource//xplat/executorch/build:runtime_wrapper.bzl", "runtime")

def define_common_targets():
    """Defines targets that should be shared between fbcode and xplat.

    The directory containing this targets.bzl file should also contain both
    TARGETS and BUCK files that call this function.
    """

    # Define build targets for all operators registered in the tables above.

    runtime.cxx_library(
        name = "quantize_per_tensor",
        srcs = [
            "quantize_per_tensor.cpp"
        ],
        platforms = CXX,
        deps = [
            "//executorch/kernels/portable/cpu/util:all_deps",
            "//executorch/kernels/portable/cpu/pattern:all_deps",
            "//executorch/runtime/kernel:kernel_includes",
            "//executorch/kernels/portable/cpu:scalar_utils",
            "//executorch/backends/cadence/hifi/kernels:kernels",
            "//executorch/backends/cadence/hifi/third-party/nnlib:nnlib-extensions"
        ],
        visibility = [
            "//executorch/backends/cadence/...",
            "@EXECUTORCH_CLIENTS",
        ],
    )

    runtime.cxx_library(
        name = "dequantize_per_tensor",
        srcs = [
            "dequantize_per_tensor.cpp"
        ],
        platforms = CXX,
        deps = [
            "//executorch/kernels/portable/cpu/util:all_deps",
            "//executorch/kernels/portable/cpu/pattern:all_deps",
            "//executorch/runtime/kernel:kernel_includes",
            "//executorch/kernels/portable/cpu:scalar_utils",
            "//executorch/backends/cadence/hifi/kernels:kernels",
            "//executorch/backends/cadence/hifi/third-party/nnlib:nnlib-extensions"
        ],
        visibility = [
            "//executorch/backends/cadence/...",
            "@EXECUTORCH_CLIENTS",
        ],
    )

    runtime.cxx_library(
        name = "quantized_layer_norm",
        srcs = [
            "quantized_layer_norm.cpp"
        ],
        exported_headers = ["operators.h"],
        platforms = CXX,
        deps = [
            "//executorch/kernels/portable/cpu/util:all_deps",
            "//executorch/kernels/portable/cpu/pattern:all_deps",
            "//executorch/runtime/kernel:kernel_includes",
            "//executorch/kernels/portable/cpu:scalar_utils",
            "//executorch/backends/cadence/hifi/kernels:kernels",
            "//executorch/backends/cadence/hifi/third-party/nnlib:nnlib-extensions"
        ],
        visibility = [
            "//executorch/backends/cadence/...",
            "@EXECUTORCH_CLIENTS",
        ],
    )

    runtime.cxx_library(
        name = "quantized_linear_out",
        srcs = [
            "quantized_linear_out.cpp"
        ],
        exported_headers = ["operators.h"],
        platforms = CXX,
        deps = [
            "//executorch/kernels/portable/cpu/util:all_deps",
            "//executorch/kernels/portable/cpu/pattern:all_deps",
            "//executorch/runtime/kernel:kernel_includes",
            "//executorch/kernels/portable/cpu:scalar_utils",
            "//executorch/backends/cadence/hifi/kernels:kernels",
            "//executorch/backends/cadence/hifi/third-party/nnlib:nnlib-extensions"
        ],
        visibility = [
            "//executorch/backends/cadence/...",
            "@EXECUTORCH_CLIENTS",
        ],
    )

    runtime.cxx_library(
        name = "op_add",
        srcs = [
            "op_add.cpp",
        ],
        platforms = CXX,
        deps = [
            "//executorch/kernels/portable/cpu/util:all_deps",
            "//executorch/kernels/portable/cpu/pattern:all_deps",
            "//executorch/runtime/kernel:kernel_includes",
            "//executorch/kernels/portable/cpu:scalar_utils",
            "//executorch/backends/cadence/hifi/kernels:kernels",
            "//executorch/backends/cadence/hifi/third-party/nnlib:nnlib-extensions",
            "//executorch/kernels/portable/cpu/util:dtype_util",
            "//executorch/kernels/portable/cpu/util:elementwise_util",
        ],
        visibility = [
            "//executorch/backends/cadence/...",
            "@EXECUTORCH_CLIENTS",
        ],
    )


    runtime.cxx_library(
        name = "op_mul",
        srcs = [
            "op_mul.cpp",
        ],
        platforms = CXX,
        deps = [
            "//executorch/kernels/portable/cpu/util:all_deps",
            "//executorch/kernels/portable/cpu/pattern:all_deps",
            "//executorch/runtime/kernel:kernel_includes",
            "//executorch/kernels/portable/cpu:scalar_utils",
            "//executorch/backends/cadence/hifi/kernels:kernels",
            "//executorch/kernels/portable/cpu/util:dtype_util",
            "//executorch/kernels/portable/cpu/util:elementwise_util",
            "//executorch/backends/cadence/hifi/third-party/nnlib:nnlib-extensions"
        ],
        visibility = [
            "//executorch/backends/cadence/...",
            "@EXECUTORCH_CLIENTS",
        ],
    )

    runtime.cxx_library(
        name = "op_sub",
        srcs = [
            "op_sub.cpp",
        ],
        platforms = CXX,
        deps = [
            "//executorch/kernels/portable/cpu/util:all_deps",
            "//executorch/kernels/portable/cpu/pattern:all_deps",
            "//executorch/runtime/kernel:kernel_includes",
            "//executorch/kernels/portable/cpu:scalar_utils",
            "//executorch/backends/cadence/hifi/kernels:kernels",
            "//executorch/kernels/portable/cpu/util:dtype_util",
            "//executorch/kernels/portable/cpu/util:elementwise_util",
            "//executorch/backends/cadence/hifi/third-party/nnlib:nnlib-extensions"
        ],
        visibility = [
            "//executorch/backends/cadence/...",
            "@EXECUTORCH_CLIENTS",
        ],
    )

    runtime.cxx_library(
        name = "op_div",
        srcs = [
            "op_div.cpp",
        ],
        platforms = CXX,
        deps = [
            "//executorch/kernels/portable/cpu/util:all_deps",
            "//executorch/kernels/portable/cpu/pattern:all_deps",
            "//executorch/runtime/kernel:kernel_includes",
            "//executorch/kernels/portable/cpu:scalar_utils",
            "//executorch/backends/cadence/hifi/kernels:kernels",
            "//executorch/kernels/portable/cpu/util:dtype_util",
            "//executorch/kernels/portable/cpu/util:elementwise_util",
            "//executorch/backends/cadence/hifi/third-party/nnlib:nnlib-extensions"
        ],
        visibility = [
            "//executorch/backends/cadence/...",
            "@EXECUTORCH_CLIENTS",
        ],
    )

    runtime.cxx_library(
        name = "op_sigmoid",
        srcs = [
            "op_sigmoid.cpp",
        ],
        platforms = CXX,
        deps = [
            "//executorch/kernels/portable/cpu/util:all_deps",
            "//executorch/kernels/portable/cpu/pattern:all_deps",
            "//executorch/runtime/kernel:kernel_includes",
            "//executorch/backends/cadence/hifi/kernels:kernels",
            "//executorch/kernels/portable/cpu/util:dtype_util",
            "//executorch/kernels/portable/cpu/util:elementwise_util",
            "//executorch/backends/cadence/hifi/third-party/nnlib:nnlib-extensions"
        ],
        visibility = [
            "//executorch/backends/cadence/...",
            "@EXECUTORCH_CLIENTS",
        ],
    )

    runtime.cxx_library(
        name = "op_tanh",
        srcs = [
            "op_tanh.cpp",
        ],
        platforms = CXX,
        deps = [
            "//executorch/kernels/portable/cpu/util:all_deps",
            "//executorch/kernels/portable/cpu/pattern:all_deps",
            "//executorch/runtime/kernel:kernel_includes",
            "//executorch/backends/cadence/hifi/kernels:kernels",
            "//executorch/backends/cadence/hifi/third-party/nnlib:nnlib-extensions"
        ],
        visibility = [
            "//executorch/backends/cadence/...",
            "@EXECUTORCH_CLIENTS",
        ],
    )

    
    runtime.cxx_library(
        name = "op_where",
        srcs = [
            "op_where.cpp",
        ],
        platforms = CXX,
        deps = [
            "//executorch/kernels/portable/cpu/util:all_deps",
            "//executorch/kernels/portable/cpu/pattern:all_deps",
            "//executorch/runtime/kernel:kernel_includes",
            "//executorch/backends/cadence/hifi/kernels:kernels",
            "//executorch/kernels/portable/cpu/util:elementwise_util",
            "//executorch/backends/cadence/hifi/third-party/nnlib:nnlib-extensions"
        ],
        visibility = [
            "//executorch/backends/cadence/...",
            "@EXECUTORCH_CLIENTS",
        ],
    )
