load("@fbsource//tools/build_defs:platform_defs.bzl", "CXX")
load("@fbsource//xplat/executorch/build:runtime_wrapper.bzl", "runtime")

def define_common_targets():
    """Defines targets that should be shared between fbcode and xplat.

    The directory containing this targets.bzl file should also contain both
    TARGETS and BUCK files that call this function.
    """

    # Define build targets for all operators registered in the tables above.

    runtime.cxx_library(
        name = "cadence_hifi_ops",
        srcs = glob([
            "*.cpp",
        ]),
        exported_headers = glob(["*.h"]),
        platforms = CXX,
        deps = [
            "//executorch/kernels/portable/cpu/util:all_deps",
            "//executorch/kernels/portable/cpu/pattern:all_deps",
            "//executorch/runtime/kernel:kernel_includes",
            "//executorch/kernels/portable/cpu:scalar_utils",
            "fbsource//third-party/nnlib-hifi4/xa_nnlib:libxa_nnlib",
            "fbsource//third-party/nnlib-hifi4/xa_nnlib:libxa_nnlib_common",
            "//executorch/backends/cadence/hifi/kernels:kernels",
        ],
        visibility = [
            "//executorch/backends/cadence/...",
            "@EXECUTORCH_CLIENTS",
        ],
    )

    runtime.cxx_library(
        name = "op_add",
        srcs = glob([
            "op_add.cpp",
        ]),
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
        name = "op_mul",
        srcs = glob([
            "op_mul.cpp",
        ]),
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
        name = "op_sub",
        srcs = glob([
            "op_sub.cpp",
        ]),
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
        name = "op_div",
        srcs = glob([
            "op_div.cpp",
        ]),
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
        name = "op_sigmoid",
        srcs = glob([
            "op_sigmoid.cpp",
        ]),
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
        name = "op_tanh",
        srcs = glob([
            "op_tanh.cpp",
        ]),
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
