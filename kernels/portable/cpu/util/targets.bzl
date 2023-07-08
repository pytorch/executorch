load("@fbsource//xplat/executorch/build:runtime_wrapper.bzl", "runtime")

def define_common_targets():
    """Defines targets that should be shared between fbcode and xplat.

    The directory containing this targets.bzl file should also contain both
    TARGETS and BUCK files that call this function.
    """

    runtime.cxx_library(
        name = "repeat_util",
        srcs = [
            "repeat_util.cpp",
        ],
        exported_headers = ["repeat_util.h"],
        deps = [
            "//executorch/core/kernel_types/util:scalar_type_util",
            "//executorch/core/kernel_types/util:tensor_util",
        ],
        compiler_flags = ["-Wno-missing-prototypes"],
        visibility = ["//executorch/kernels/portable/cpu/..."],
    )

    runtime.cxx_library(
        name = "broadcast_util",
        srcs = ["broadcast_util.cpp"],
        exported_headers = [
            "broadcast_util.h",
        ],
        compiler_flags = ["-Wno-missing-prototypes"],
        deps = [
            ":repeat_util",
            "//executorch/kernels/portable/cpu:scalar_utils",
            "//executorch/runtime/kernel:kernel_includes",
            "//executorch/core/kernel_types/util:tensor_util",
        ],
        visibility = ["//executorch/kernels/portable/cpu/...", "//executorch/kernels/optimized/cpu/..."],
    )

    runtime.cxx_library(
        name = "transpose_util",
        exported_headers = [
            "transpose_util.h",
        ],
        deps = [
            "//executorch/kernels/portable/cpu:scalar_utils",
            "//executorch/runtime/kernel:kernel_includes",
            "//executorch/core/kernel_types/util:tensor_util",
        ],
        visibility = ["//executorch/kernels/portable/cpu/..."],
    )

    # Utility functions that can be used by operators that perform indexing
    runtime.cxx_library(
        name = "index_util",
        srcs = ["index_util.cpp"],
        exported_headers = ["index_util.h"],
        deps = [
            "//executorch/runtime/kernel:kernel_includes",
            "//executorch/core/kernel_types/util:tensor_util",
        ],
        visibility = ["//executorch/kernels/portable/cpu/...", "//executorch/kernels/quantized/..."],
    )

    # Utility functions that can be used by operators that repeat the same computation for each element in the tensor
    # Note that because this is a header only library, targets must also depend on broadcast_util.
    runtime.cxx_library(
        name = "functional_util",
        srcs = [],
        exported_headers = ["functional_util.h"],
        deps = [
            "//executorch/runtime/kernel:kernel_includes",
            "//executorch/core/kernel_types/util:tensor_util",
            ":broadcast_util",
        ],
        visibility = ["//executorch/kernels/portable/cpu/...", "//executorch/kernels/quantized/..."],
    )

    # Utility functions that can be used by operators that perform reduction
    runtime.cxx_library(
        name = "reduce_util",
        srcs = ["reduce_util.cpp"],
        exported_headers = ["reduce_util.h"],
        deps = [
            "//executorch/runtime/kernel:kernel_includes",
            "//executorch/core/kernel_types/util:tensor_util",
        ],
        visibility = ["//executorch/kernels/portable/cpu/...", "//executorch/kernels/quantized/..."],
    )
