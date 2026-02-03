load("@fbsource//xplat/executorch/build:runtime_wrapper.bzl", "get_aten_mode_options", "runtime")

def define_common_targets():
    """Defines targets that should be shared between fbcode and xplat.

    The directory containing this targets.bzl file should also contain both
    TARGETS and BUCK files that call this function.
    """

    # Note: add all portable_op dependencies to all_deps. This is used for dtype selective
    # build, where the portable ops are built from source and linked with :all_deps
    runtime.cxx_library(
        name = "all_deps",
        exported_deps = [
            "//executorch/extension/threadpool:threadpool",
            "//executorch/kernels/portable/cpu/util:arange_util",
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
            "//executorch/kernels/portable/cpu/util:stack_util",
            "//executorch/kernels/portable/cpu/util:elementwise_util",
            "//executorch/kernels/portable/cpu/util:upsample_util",
            "//executorch/kernels/portable/cpu/util:vectorized_math",
            "//executorch/kernels/portable/cpu/util:grid_sampler_2d_util",
        ],
        visibility = ["PUBLIC"],
    )

    runtime.cxx_library(
        name = "activation_ops_util",
        srcs = ["activation_ops_util.cpp"],
        exported_headers = [
            "activation_ops_util.h",
        ],
        compiler_flags = ["-Wno-missing-prototypes"],
        deps = [
            "//executorch/runtime/core/exec_aten/util:tensor_shape_to_c_string",
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
        visibility = ["//executorch/kernels/portable/cpu/..."],
    )

    runtime.cxx_library(
        name = "broadcast_util",
        srcs = [
            "broadcast_util.cpp",
            "delinearize_index.cpp",
        ],
        exported_headers = [
            "broadcast_util.h",
            "delinearize_index.h",
        ],
        exported_deps = [
            ":broadcast_indexes_range",
        ],
        deps = [
            ":repeat_util",
            "//executorch/runtime/kernel:kernel_includes",
            "//executorch/runtime/core/exec_aten/util:tensor_shape_to_c_string",
            "//executorch/runtime/core/exec_aten/util:tensor_util",
        ],
        visibility = ["PUBLIC"],
    )

    runtime.cxx_library(
        name = "dtype_util",
        srcs = ["dtype_util.cpp"],
        exported_headers = [
            "dtype_util.h",
        ],
        compiler_flags = ["-Wno-missing-prototypes"],
        deps = [
            "//executorch/runtime/kernel:kernel_includes",
        ],
        visibility = ["PUBLIC"],
    )

    runtime.cxx_library(
        name = "elementwise_util",
        exported_headers = [
            "elementwise_util.h",
        ],
        compiler_flags = ["-Wno-missing-prototypes"],
        exported_deps = [
            ":broadcast_indexes_range",
            ":broadcast_util",
            ":dtype_util",
            ":vectorized_math",
            "//executorch/runtime/core/portable_type/c10/c10:aten_headers_for_executorch",
            "//executorch/runtime/kernel:kernel_runtime_context",
            "//executorch/kernels/portable/cpu:scalar_utils",
            "//executorch/extension/threadpool:threadpool",
            "//executorch/kernels/portable/cpu:scalar_utils",
        ],
        deps = [
            "//executorch/runtime/kernel:kernel_includes",
        ],
        visibility = ["PUBLIC"],
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
            "//executorch/runtime/core/exec_aten/util:tensor_shape_to_c_string",
            "//executorch/runtime/kernel:kernel_includes",
        ],
        visibility = ["PUBLIC"],
    )

    runtime.cxx_library(
        name = "copy_ops_util",
        srcs = ["copy_ops_util.cpp"],
        exported_headers = [
            "copy_ops_util.h",
        ],
        compiler_flags = ["-Wno-missing-prototypes"],
        exported_deps = [
            ":broadcast_util",
        ],
        deps = [
            "//executorch/runtime/kernel:kernel_includes",
        ],
        visibility = ["//executorch/kernels/portable/cpu/...", "//executorch/kernels/optimized/cpu/...", "//executorch/kernels/test/..."],
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
        exported_deps = [
            "//executorch/extension/threadpool:threadpool",
        ],
        deps = [
            "//executorch/runtime/kernel:kernel_includes",
            "//executorch/runtime/core/exec_aten/util:tensor_util",
            ":broadcast_util",
        ],
        visibility = ["PUBLIC"],
    )

    runtime.cxx_library(
        name = "math_util",
        srcs = [],
        exported_headers = ["math_util.h"],
        visibility = ["//executorch/kernels/portable/cpu/...", "//executorch/kernels/quantized/..."],
        exported_deps = [
            "//executorch/runtime/core/portable_type/c10/c10:aten_headers_for_executorch",
        ],
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
            "//executorch/extension/threadpool:threadpool",
        ],
        visibility = ["//executorch/kernels/portable/cpu/..."],
    )

    runtime.cxx_library(
        name = "stack_util",
        srcs = ["stack_util.cpp"],
        exported_headers = ["stack_util.h"],
        deps = [
            "//executorch/kernels/portable/cpu/util:copy_ops_util",
        ],
        exported_deps = [
            "//executorch/runtime/kernel:kernel_includes",
        ],
        visibility = ["PUBLIC"],
    )

    runtime.cxx_library(
        name = "upsample_util",
        srcs = ["upsample_util.cpp"],
        exported_headers = ["upsample_util.h"],
        deps = [
            "//executorch/runtime/kernel:kernel_includes",
        ],
        visibility = ["//executorch/kernels/portable/cpu/..."],
    )

    runtime.cxx_library(
        name = "broadcast_indexes_range",
        exported_headers = ["broadcast_indexes_range.h"],
        deps = [
            "//executorch/runtime/core/exec_aten:lib",
            "//executorch/runtime/core/exec_aten/util:tensor_dimension_limit",
        ],
        visibility = ["PUBLIC"],
    )

    runtime.cxx_library(
        name = "vectorized_math",
        exported_headers = ["vectorized_math.h"],
        visibility = ["//executorch/..."],
        exported_deps = [
            "//executorch/runtime/core/portable_type:portable_type",
            "//executorch/runtime/core/exec_aten/util:scalar_type_util",
        ],
    )

    runtime.cxx_library(
        name = "grid_sampler_2d_util",
        srcs = ["grid_sampler_2d_util.cpp"],
        exported_headers = ["grid_sampler_2d_util.h"],
        deps = [
            "//executorch/runtime/kernel:kernel_includes",
        ],
        visibility = ["//executorch/kernels/portable/cpu/..."],
    )

    # Utility functions that can be used by operators that perform reduction
    for aten_mode in get_aten_mode_options():
        suffix = "_aten" if aten_mode else ""
        runtime.cxx_library(
            name = "reduce_util{}".format(suffix),
            srcs = ["reduce_util.cpp"],
            exported_headers = ["reduce_util.h"],
            deps = [
                "//executorch/runtime/kernel:kernel_includes{}".format(suffix),
                "//executorch/runtime/core/exec_aten/util:tensor_util{}".format(suffix),
            ],
            exported_deps = [
                "//executorch/extension/threadpool:threadpool",
            ],
            exported_preprocessor_flags = ["-DUSE_ATEN_LIB"] if aten_mode else [],
            visibility = ["PUBLIC"],
        )

        runtime.cxx_library(
            name = "arange_util{}".format(suffix),
            srcs = ["arange_util.cpp"],
            exported_headers = ["arange_util.h"],
            exported_deps = [
                "//executorch/runtime/kernel:kernel_includes{}".format(suffix),
            ],
            visibility = [
                "//executorch/kernels/portable/cpu/...",
                "//executorch/extension/llm/...",
            ],
        )
