load("@fbsource//tools/build_defs:platform_defs.bzl", "CXX")
load("@fbsource//xplat/executorch/build:runtime_wrapper.bzl", "runtime")

def define_common_targets():
    runtime.cxx_library(
        name = "cadence_cpu_ops",
        srcs = glob([
            "*.cpp",
        ]),
        platforms = CXX,
        deps = [
            "//executorch/kernels/portable/cpu/util:broadcast_util",
            "//executorch/runtime/kernel:kernel_includes",
            "//executorch/kernels/portable/cpu:scalar_utils",
            "//executorch/backends/cadence/reference/kernels:cadence_kernels",
        ],
        visibility = [
            "//executorch/backends/cadence/...",
        ],
    )
