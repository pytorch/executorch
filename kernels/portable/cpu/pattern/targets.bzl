load("@fbsource//xplat/executorch/build:runtime_wrapper.bzl", "runtime")

def define_common_targets():
    """Defines targets that should be shared between fbcode and xplat.

    The directory containing this targets.bzl file should also contain both
    TARGETS and BUCK files that call this function.
    """

    # Note: add all portable_op dependencies to all_deps. This is used for dtype selective
    # build, where the portable ops are built from source and linked with :all_deps
    runtime.cxx_library(
        name = "all_deps",
        deps = [
            "//executorch/kernels/portable/cpu/pattern:pattern",
            "//executorch/kernels/portable/cpu/pattern:bitwise_op",
            "//executorch/kernels/portable/cpu/pattern:comparison_op",
            "//executorch/kernels/portable/cpu/pattern:logical_op"
        ],
        visibility = ["//executorch/...", "@EXECUTORCH_CLIENTS"],
    )

    runtime.cxx_library(
        name = "bitwise_op",
        exported_headers = [
            "bitwise_op.h",
        ],
        compiler_flags = [],
        visibility = ["//executorch/kernels/portable/cpu/...", "//executorch/kernels/optimized/cpu/..."],
    )

    runtime.cxx_library(
        name = "comparison_op",
        exported_headers = [
            "comparison_op.h",
        ],
        compiler_flags = [],
        visibility = ["//executorch/kernels/portable/cpu/...", "//executorch/kernels/optimized/cpu/..."],
    )

    runtime.cxx_library(
        name = "logical_op",
        exported_headers = [
            "logical_op.h",
        ],
        compiler_flags = [],
        visibility = ["//executorch/kernels/portable/cpu/...", "//executorch/kernels/optimized/cpu/..."],
    )

    runtime.cxx_library(
        name = "pattern",
        srcs = [
            "unary_ufunc_realhb_to_bool.cpp",
            "unary_ufunc_realhbbf16_to_floathbf16.cpp",
            "unary_ufunc_realh.cpp",
        ],
        exported_headers = [
            "pattern.h",
        ],
        compiler_flags = ["-Wno-missing-prototypes"],
        deps = [
            "//executorch/kernels/portable/cpu/util:broadcast_util",
            "//executorch/kernels/portable/cpu/util:functional_util",
            "//executorch/runtime/kernel:kernel_includes",
        ],
        visibility = ["//executorch/kernels/portable/cpu/...", "//executorch/kernels/optimized/cpu/..."],
    )
