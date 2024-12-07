load("@fbsource//tools/build_defs:platform_defs.bzl", "CXX")
load("@fbsource//xplat/executorch/build:runtime_wrapper.bzl", "runtime")

def define_common_targets():
    """Defines targets that should be shared between fbcode and xplat.

    The directory containing this targets.bzl file should also contain both
    TARGETS and BUCK files that call this function.
    """

    # Define build targets for all operators registered in the tables above.

    runtime.cxx_library(
        name = "cadence_g3_ops",
        srcs = glob([
            "*.cpp",
        ]),
        exported_headers = glob([
            "*.h",
        ]),
        platforms = CXX,
        deps = [
            "//executorch/kernels/portable/cpu/util:all_deps",
            "//executorch/kernels/portable/cpu/pattern:all_deps",
            "//executorch/runtime/kernel:kernel_includes",
            "//executorch/kernels/portable/cpu:scalar_utils",
            "fbsource//third-party/nnlib-FusionG3/xa_nnlib:libxa_nnlib_common",
            "fbsource//third-party/nnlib-FusionG3/xa_nnlib:libxa_nnlib",
        ],
        visibility = [
            "//executorch/backends/cadence/...",
            "@EXECUTORCH_CLIENTS",
        ],
        exported_deps = [
            "fbsource//third-party/nnlib-FusionG3/xa_nnlib:libxa_nnlib_common",
            "fbsource//third-party/nnlib-FusionG3/xa_nnlib:libxa_nnlib",
        ],
    )
