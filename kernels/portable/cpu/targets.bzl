load("@fbsource//xplat/executorch/build:runtime_wrapper.bzl", "get_aten_mode_options", "runtime")
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

    if True in get_aten_mode_options():
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
        exported_deps = [
            "//executorch/runtime/core/portable_type/c10/c10:c10",
            "//executorch/runtime/platform:compiler",
        ],
        srcs = [],
        exported_headers = ["vec_ops.h"],
        visibility = ["PUBLIC"],
    )

    # Only for use by targets in this directory. Defines constants like M_PI
    # if they arent already defined by the toolchains cmath
    runtime.cxx_library(
        name = "math_constants",
        srcs = [],
        exported_headers = [
            "math_constants.h",
        ],
        visibility = ["PUBLIC"],
    )

    # Only for use by targets in this directory.
    runtime.cxx_library(
        name = "scalar_utils",
        srcs = [],
        exported_headers = ["scalar_utils.h", "selective_build.h"],
        visibility = ["PUBLIC"],
        deps = [
            "//executorch/runtime/core/exec_aten/util:scalar_type_util",
        ],
        exported_deps = [
            "//executorch/runtime/core/exec_aten:lib",
        ],
    )

    # Device copy ops (h2d_copy, d2h_copy) for transferring data between
    # CPU and device memory. Uses DeviceAllocator interface.
    runtime.cxx_library(
        name = "op__device_copy",
        srcs = ["op__device_copy.cpp"],
        visibility = ["PUBLIC"],
        # Constructor needed for op registration.
        compiler_flags = ["-Wno-global-constructors"],
        deps = [
            "//executorch/runtime/core:device_allocator",
            "//executorch/runtime/core/exec_aten:lib",
            "//executorch/runtime/kernel:kernel_includes",
            "//executorch/extension/kernel_util:kernel_util",
        ],
        # @lint-ignore BUCKLINT: Avoid `link_whole=True` (https://fburl.com/avoid-link-whole)
        link_whole = True,
    )

    # Used for dtype selective build. Collect source and header files.
    runtime.filegroup(
        name = "portable_source_files",
        srcs = native.glob(["*.cpp"]),
        visibility = ["PUBLIC"],
    )

    runtime.filegroup(
        name = "portable_header_files",
        srcs = native.glob(["*.h"]),
        visibility = ["PUBLIC"],
    )
