load("@fbsource//tools/build_defs:platform_defs.bzl", "CXX")
load("@fbsource//xplat/executorch/build:runtime_wrapper.bzl", "runtime")

def define_common_targets():
    runtime.cxx_library(
        name = "cadence_kernels",
        srcs = ["kernels.cpp"],
        exported_headers = [
            "kernels.h",
        ],
        visibility = [
            "//executorch/backends/cadence/...",
        ],
        platforms = CXX,
    )
