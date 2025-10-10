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
            "@EXECUTORCH_CLIENTS",
        ],
        platforms = CXX,
        compatible_with = select({
            "DEFAULT": [],
            "ovr_config//cpu:xtensa": ["ovr_config//cpu:xtensa"],
        }),
        define_static_target = True,
        deps = [
            "//executorch/backends/cadence/vision/third-party:vision-nnlib",
            "//executorch/runtime/kernel:kernel_includes",
        ],
    )
