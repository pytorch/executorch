load("@fbsource//xplat/executorch/build:runtime_wrapper.bzl", "runtime")

def define_common_targets():
    """Defines targets that should be shared between fbcode and xplat.

    The directory containing this targets.bzl file should also contain both
    TARGETS and BUCK files that call this function.
    """

    runtime.cxx_library(
        name = "pattern",
        srcs = [
            "unary_ufunc_realhb_to_floath.cpp",
            "binary_ufunc_realb_realb_to_realb_logical.cpp",
            "unary_ufunc_realb_to_bool.cpp",
            "unary_ufunc_real.cpp",
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
