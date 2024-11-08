load("@fbsource//xplat/executorch/build:runtime_wrapper.bzl", "runtime")
load("@fbsource//xplat/executorch/kernels/portable:op_registration_util.bzl", "ATEN_OPS", "CUSTOM_OPS", "define_op_target")

def define_common_targets():
    """Defines targets that should be shared between fbcode and xplat.

    The directory containing this targets.bzl file should also contain both
    TARGETS and BUCK files that call this function.
    """

    # Define build targets for all operators registered in the tables above.
    for op in ATEN_OPS:
        define_op_target(is_aten_op = True, **op)
    for op in CUSTOM_OPS:
        define_op_target(is_aten_op = False, **op)

    custom_op_targets = [":{}".format(op["name"]) for op in CUSTOM_OPS]

    aten_op_targets = [":{}".format(op["name"]) for op in ATEN_OPS]
    all_op_targets = custom_op_targets + aten_op_targets

    runtime.cxx_library(
        name = "cpu",
        srcs = [],
        visibility = [
            "//executorch/kernels/portable/...",
            "//executorch/kernels/test/...",
        ],
        exported_deps = all_op_targets,
    )

    runtime.cxx_library(
        name = "cpu_aten",
        srcs = [],
        visibility = ["//executorch/kernels/portable/..."],
        exported_deps = [t + "_aten" for t in custom_op_targets],
    )

    # Only for use by op targets under //executorch. This API needs to be
    # reevaluated before becoming a public API.
    runtime.cxx_library(
        name = "vec_ops",
        srcs = [],
        exported_headers = ["vec_ops.h"],
        visibility = ["//executorch/kernels/portable/cpu/...", "//executorch/kernels/quantized/..."],
    )

    # Only for use by targets in this directory. Defines constants like M_PI
    # if they arent already defined by the toolchains cmath
    runtime.cxx_library(
        name = "math_constants",
        srcs = [],
        exported_headers = [
            "math_constants.h",
        ],
        visibility = [
            "//executorch/kernels/portable/cpu/...",
        ],
    )

    # Only for use by targets in this directory.
    runtime.cxx_library(
        name = "scalar_utils",
        srcs = [],
        exported_headers = ["scalar_utils.h", "selective_build.h"],
        visibility = [
            "//executorch/kernels/portable/cpu/...",
            "//executorch/kernels/optimized/cpu/...",
            "//executorch/kernels/portable/test/...",
            "@EXECUTORCH_CLIENTS",
        ],
        deps = [
            "//executorch/runtime/core/exec_aten:lib",
            "//executorch/runtime/core/exec_aten/util:scalar_type_util",
        ],
    )

    # Used for dtype selective build. Collect source and header files.
    runtime.filegroup(
        name = "portable_source_files",
        srcs = native.glob(["*.cpp"]),
        visibility = ["//executorch/...", "@EXECUTORCH_CLIENTS"],
    )

    runtime.filegroup(
        name = "portable_header_files",
        srcs = native.glob(["*.h"]),
        visibility = ["//executorch/...", "@EXECUTORCH_CLIENTS"],
    )

    runtime.cxx_library(
        name = "op_div_impl",
        srcs = ["op_div_impl.cpp"],
        exported_headers = ["op_div_impl.h"],
        visibility = [
            "//executorch/kernels/portable/cpu/...",
            "//executorch/kernels/optimized/cpu/...",
            "//executorch/kernels/portable/test/...",
            "@EXECUTORCH_CLIENTS",
        ],
        exported_deps = [
            "//executorch/kernels/portable/cpu/util:broadcast_util",
            "//executorch/kernels/portable/cpu/util:dtype_util",
            "//executorch/kernels/portable/cpu/util:elementwise_util",
            "//executorch/kernels/portable/cpu/util:math_util",
            "//executorch/kernels/portable/cpu:scalar_utils",
            "//executorch/runtime/core/exec_aten:lib",
            "//executorch/runtime/core/exec_aten/util:scalar_type_util",
            "//executorch/runtime/core/exec_aten/util:tensor_util",
            "//executorch/runtime/kernel:kernel_includes",
        ],
    )

    runtime.cxx_library(
        name = "op_mul_impl",
        srcs = ["op_mul_impl.cpp"],
        exported_headers = ["op_mul_impl.h"],
        visibility = [
            "//executorch/kernels/portable/cpu/...",
            "//executorch/kernels/optimized/cpu/...",
            "//executorch/kernels/portable/test/...",
            "@EXECUTORCH_CLIENTS",
        ],
        exported_deps = [
            "//executorch/kernels/portable/cpu/util:broadcast_util",
            "//executorch/kernels/portable/cpu/util:dtype_util",
            "//executorch/kernels/portable/cpu/util:elementwise_util",
            "//executorch/kernels/portable/cpu/util:math_util",
            "//executorch/kernels/portable/cpu:scalar_utils",
            "//executorch/runtime/kernel:kernel_includes",
        ],
    )

    # The following will not participate in dtype selective build because
    # they are refactored such to be used in optimized op implementations as well
    # and we have not enabled selective build for optimized ops.
    # To enable selective build for these ops, they must be copied over by
    # selective build flow, however this results in such files, e.g. op_div_impl.cpp,
    # getting compiled twice, once for selective build and once for optimized, and when
    # put together they result in two copies of op_div_impl.o resulting in duplicate
    # symbols
    runtime.cxx_library(
        name = "all_impl_deps",
        deps = [
            "//executorch/kernels/portable/cpu:op_div_impl",
            "//executorch/kernels/portable/cpu:op_mul_impl",
        ],
        visibility = ["//executorch/...", "@EXECUTORCH_CLIENTS"],
    )
