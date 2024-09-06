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
            "//executorch/kernels/portable/cpu/util:functional_util",
            "//executorch/kernels/portable/cpu/util:broadcast_util",
            "//executorch/kernels/portable/cpu/util:kernel_ops_util",
            "//executorch/kernels/portable/cpu:vec_ops",
            "//executorch/kernels/portable/cpu/util:matmul_ops_util",
            "//executorch/kernels/portable/cpu/util:copy_ops_util",
            "//executorch/kernels/portable/cpu/util:transpose_util",
            "//executorch/kernels/portable/cpu/util:index_util",
            "//executorch/kernels/portable/cpu/util:math_util",
            "//executorch/kernels/portable/cpu/util:padding_util",
            "//executorch/kernels/portable/cpu/util:repeat_util",
            "//executorch/kernels/portable/cpu/util:activation_ops_util",
            "//executorch/kernels/portable/cpu/util:reduce_util",
            "//executorch/kernels/portable/cpu/util:normalization_ops_util",
            "//executorch/kernels/portable/cpu/util:distance_util",
            "//executorch/kernels/portable/cpu/util:select_copy_util",
            "//executorch/kernels/portable/cpu/util:advanced_index_util",
            "//executorch/kernels/portable/cpu/util:slice_util",
        ],
        visibility = ["//executorch/...", "@EXECUTORCH_CLIENTS"],
    )

    runtime.cxx_library(
        name = "activation_ops_util",
        srcs = ["activation_ops_util.cpp"],
        exported_headers = [
            "activation_ops_util.h",
        ],
        compiler_flags = ["-Wno-missing-prototypes"],
        deps = [
            "//executorch/runtime/kernel:kernel_includes",
        ],
        visibility = ["//executorch/kernels/portable/cpu/...", "//executorch/kernels/optimized/cpu/..."],
    )

    runtime.cxx_library(
        name = "repeat_util",
        srcs = [
            "repeat_util.cpp",
        ],
        exported_headers = ["repeat_util.h"],
        deps = [
            "//executorch/runtime/kernel:kernel_includes",
            "//executorch/runtime/core/exec_aten/util:scalar_type_util",
            "//executorch/runtime/core/exec_aten/util:tensor_util",
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
            "//executorch/runtime/kernel:kernel_includes",
            "//executorch/runtime/core/exec_aten/util:tensor_util",
        ],
        visibility = ["//executorch/kernels/portable/cpu/...", "//executorch/kernels/optimized/cpu/...", "@EXECUTORCH_CLIENTS"],
    )

    runtime.cxx_library(
        name = "advanced_index_util",
        srcs = ["advanced_index_util.cpp"],
        exported_headers = [
            "advanced_index_util.h",
        ],
        compiler_flags = ["-Wno-missing-prototypes"],
        deps = [
            ":broadcast_util",
            "//executorch/runtime/kernel:kernel_includes",
        ],
        visibility = ["//executorch/kernels/portable/cpu/...", "//executorch/kernels/optimized/cpu/..."],
    )

    runtime.cxx_library(
        name = "copy_ops_util",
        srcs = ["copy_ops_util.cpp"],
        exported_headers = [
            "copy_ops_util.h",
        ],
        compiler_flags = ["-Wno-missing-prototypes"],
        deps = [
            "//executorch/runtime/kernel:kernel_includes",
        ],
        visibility = ["//executorch/kernels/portable/cpu/...", "//executorch/kernels/optimized/cpu/..."],
    )

    runtime.cxx_library(
        name = "distance_util",
        srcs = ["distance_util.cpp"],
        exported_headers = [
            "distance_util.h",
        ],
        compiler_flags = ["-Wno-missing-prototypes"],
        deps = [
            "//executorch/runtime/kernel:kernel_includes",
        ],
        visibility = ["//executorch/kernels/portable/cpu/...", "//executorch/kernels/optimized/cpu/..."],
    )

    runtime.cxx_library(
        name = "kernel_ops_util",
        srcs = ["kernel_ops_util.cpp"],
        exported_headers = [
            "kernel_ops_util.h",
        ],
        compiler_flags = ["-Wno-missing-prototypes"],
        deps = [
            "//executorch/runtime/kernel:kernel_includes",
        ],
        visibility = ["//executorch/kernels/portable/cpu/...", "//executorch/kernels/optimized/cpu/..."],
    )

    runtime.cxx_library(
        name = "matmul_ops_util",
        srcs = ["matmul_ops_util.cpp"],
        exported_headers = [
            "matmul_ops_util.h",
        ],
        compiler_flags = ["-Wno-missing-prototypes"],
        deps = [
            ":broadcast_util",
            "//executorch/runtime/kernel:kernel_includes",
        ],
        visibility = ["//executorch/kernels/portable/cpu/...", "//executorch/kernels/optimized/cpu/..."],
    )

    runtime.cxx_library(
        name = "padding_util",
        srcs = ["padding_util.cpp"],
        exported_headers = [
            "padding_util.h",
        ],
        compiler_flags = ["-Wno-missing-prototypes"],
        deps = [
            "//executorch/runtime/kernel:kernel_includes",
        ],
        visibility = ["//executorch/kernels/portable/cpu/...", "//executorch/kernels/optimized/cpu/..."],
    )

    runtime.cxx_library(
        name = "normalization_ops_util",
        srcs = ["normalization_ops_util.cpp"],
        exported_headers = [
            "normalization_ops_util.h",
        ],
        compiler_flags = ["-Wno-missing-prototypes"],
        deps = [
            "//executorch/runtime/kernel:kernel_includes",
        ],
        visibility = ["//executorch/kernels/portable/cpu/...", "//executorch/kernels/optimized/cpu/..."],
    )

    runtime.cxx_library(
        name = "transpose_util",
        exported_headers = [
            "transpose_util.h",
        ],
        deps = [
            "//executorch/runtime/kernel:kernel_includes",
            "//executorch/runtime/core/exec_aten/util:tensor_util",
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
            "//executorch/runtime/core/exec_aten/util:tensor_util",
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
            "//executorch/runtime/core/exec_aten/util:tensor_util",
            ":broadcast_util",
        ],
        visibility = ["//executorch/kernels/portable/cpu/...", "//executorch/kernels/quantized/..."],
    )

    runtime.cxx_library(
        name = "math_util",
        srcs = [],
        exported_headers = ["math_util.h"],
        visibility = ["//executorch/kernels/portable/cpu/...", "//executorch/kernels/quantized/..."],
    )

    runtime.cxx_library(
        name = "select_copy_util",
        srcs = ["select_copy_util.cpp"],
        exported_headers = ["select_copy_util.h"],
        deps = [
            ":copy_ops_util",
            "//executorch/runtime/kernel:kernel_includes",
            "//executorch/runtime/core/exec_aten/util:tensor_util",
        ],
        visibility = ["//executorch/kernels/portable/cpu/..."],
    )

    runtime.cxx_library(
        name = "slice_util",
        srcs = ["slice_util.cpp"],
        exported_headers = ["slice_util.h"],
        deps = [
            "//executorch/runtime/kernel:kernel_includes",
        ],
        visibility = ["//executorch/kernels/portable/cpu/..."],
    )

    # Utility functions that can be used by operators that perform reduction
    for aten_mode in [True, False]:
        suffix = "_aten" if aten_mode else ""
        runtime.cxx_library(
            name = "reduce_util{}".format(suffix),
            srcs = ["reduce_util.cpp"],
            exported_headers = ["reduce_util.h"],
            deps = [
                "//executorch/runtime/kernel:kernel_includes{}".format(suffix),
                "//executorch/runtime/core/exec_aten/util:tensor_util{}".format(suffix),
            ],
            exported_preprocessor_flags = ["-DUSE_ATEN_LIB"] if aten_mode else [],
            visibility = ["//executorch/kernels/portable/cpu/...", "//executorch/kernels/quantized/..."],
        )
