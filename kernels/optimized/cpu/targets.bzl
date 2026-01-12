load("@fbsource//xplat/executorch/build:runtime_wrapper.bzl", "runtime")
load("@fbsource//xplat/executorch/kernels/optimized:op_registration_util.bzl", "OPTIMIZED_ATEN_OPS", "define_op_target", "op_target")

def get_sleef_preprocessor_flags():
    if runtime.is_oss:
        return []
    return ["-DAT_BUILD_ARM_VEC256_WITH_SLEEF"]


def define_common_targets():
    """Defines targets that should be shared between fbcode and xplat.

    The directory containing this targets.bzl file should also contain both
    TARGETS and BUCK files that call this function.
    """

    # Define build targets for all operators registered in the tables above.
    for op in OPTIMIZED_ATEN_OPS:
        define_op_target(**op)

    aten_op_targets = [":{}".format(op["name"]) for op in OPTIMIZED_ATEN_OPS]
    all_op_targets = aten_op_targets

    runtime.cxx_library(
        name = "add_sub_impl",
        srcs = [],
        exported_headers = ["op_add_sub_impl.h"],
        visibility = ["PUBLIC"],
        exported_deps = [
            "//executorch/runtime/core:core",
            "//executorch/kernels/portable/cpu/util:broadcast_indexes_range",
            "//executorch/kernels/portable/cpu/util:broadcast_util",
            "//executorch/kernels/portable/cpu/util:dtype_util",
            "//executorch/kernels/portable/cpu/util:elementwise_util",
        ],
    )

    runtime.cxx_library(
        name = "fft_utils",
        srcs = [],
        exported_headers = ["fft_utils.h"],
        visibility = ["PUBLIC"],
        exported_deps = [] if runtime.is_oss else ["fbsource//third-party/pocket_fft:pocketfft"],
    )

    runtime.cxx_library(
        name = "binary_ops",
        srcs = ["binary_ops.cpp"],
        exported_headers = ["binary_ops.h"],
        visibility = ["PUBLIC"],
        exported_deps = [
            "//executorch/runtime/core/exec_aten:lib",
            "//executorch/runtime/kernel:kernel_includes",
            "//executorch/kernels/optimized:libvec",
            "//executorch/kernels/portable/cpu:scalar_utils",
            "//executorch/kernels/portable/cpu/util:broadcast_util",
        ],
    )

    runtime.cxx_library(
        name = "cpu_optimized",
        srcs = [],
        visibility = ["//executorch/kernels/..."],
        exported_deps = all_op_targets,
    )

    runtime.cxx_library(
        name = "moments_utils",
        srcs = [],
        exported_headers = ["moments_utils.h"],
        visibility = ["PUBLIC"],
        exported_deps = [
            "//executorch/runtime/core/portable_type/c10/c10:aten_headers_for_executorch",
            "//executorch/kernels/optimized:libvec",
            "//executorch/kernels/optimized:libutils",
        ],
    )

    # Used for dtype selective build. Collect source and header files.
    runtime.filegroup(
        name = "optimized_source_files",
        srcs = native.glob(["*.cpp"]),
        visibility = ["PUBLIC"],
    )

    runtime.filegroup(
        name = "optimized_header_files",
        srcs = native.glob(["*.h"]),
        visibility = ["PUBLIC"],
    )
