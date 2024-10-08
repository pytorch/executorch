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
        platforms = CXX,
        deps = [
            "//executorch/kernels/portable/cpu/util:broadcast_util",
            "//executorch/runtime/kernel:kernel_includes",
            "//executorch/kernels/portable/cpu:scalar_utils",
            "fbsource//third-party/nnlib-hifi4/xa_nnlib:libxa_nnlib",
            "fbsource//third-party/nnlib-hifi4/xa_nnlib:libxa_nnlib_common",
            "//executorch/backends/cadence/hifi/kernels:kernels",
        ],
        visibility = [
            "//executorch/backends/cadence/...",
        ],
    )
