load("@fbsource//xplat/executorch/build:runtime_wrapper.bzl", "runtime")
load("@fbsource//xplat/executorch/kernels/optimized:op_registration_util.bzl", "define_op_target", "op_target")

_OPTIMIZED_ATEN_OPS = (
    op_target(
        name = "op_add",
        deps = [
            ":binary_ops",
            ":add_sub_impl",
            "//executorch/kernels/portable/cpu:scalar_utils",
            "//executorch/kernels/portable/cpu/util:broadcast_util",
        ],
    ),
    op_target(
        name = "op_bmm",
        deps = [
            "//executorch/kernels/optimized:libblas",
            "//executorch/kernels/portable/cpu/util:matmul_ops_util",
        ],
    ),
    op_target(
        name = "op_div",
        deps = [
            ":binary_ops",
            "//executorch/kernels/portable/cpu:scalar_utils",
            "//executorch/kernels/portable/cpu/util:broadcast_util",
        ],
    ),
    op_target(
        name = "op_elu",
        deps = [
            "//executorch/extension/threadpool:threadpool",
            "//executorch/kernels/portable/cpu:scalar_utils",
            "//executorch/runtime/core/portable_type/c10/c10:aten_headers_for_executorch",
        ],
    ),
    op_target(name = "op_exp"),
    op_target(
        name = "op_fft_r2c",
        compiler_flags = [] if runtime.is_oss else [
            "-Wno-global-constructors",
            "-Wno-shadow",
        ],
        deps = [] if runtime.is_oss else ["fbsource//third-party/pocket_fft:pocketfft"],
    ),
    op_target(name = "op_sigmoid"),
    op_target(
        name = "op_gelu",
        deps = [
            "//executorch/kernels/portable/cpu/util:activation_ops_util",
            "//executorch/runtime/core/portable_type/c10/c10:aten_headers_for_executorch",
        ],
    ),
    op_target(
        name = "op_le",
        deps = [
            "//executorch/kernels/portable/cpu:scalar_utils",
            "//executorch/kernels/portable/cpu/util:broadcast_util",
        ],
    ),
    op_target(
        name = "op_linear",
        deps = [
            "//executorch/kernels/optimized:libblas",
            "//executorch/kernels/portable/cpu/util:matmul_ops_util",
        ],
    ),
    op_target(
        name = "op_log_softmax",
        deps = [
            "//executorch/kernels/portable/cpu/util:activation_ops_util",
            "//executorch/runtime/core/portable_type/c10/c10:aten_headers_for_executorch",
        ],
    ),
    op_target(
        name = "op_mm",
        deps = [
            "//executorch/kernels/optimized:libblas",
            "//executorch/kernels/portable/cpu/util:matmul_ops_util",
        ],
    ),
    op_target(
        name = "op_mul",
        deps = [
            ":binary_ops",
            "//executorch/kernels/portable/cpu:scalar_utils",
            "//executorch/kernels/portable/cpu/util:broadcast_util",
            "//executorch/runtime/core/exec_aten/util:tensor_util",
        ],
    ),
    op_target(
        name = "op_native_layer_norm",
        deps = [
            ":moments_utils",
            "//executorch/kernels/portable/cpu/util:normalization_ops_util",
        ],
    ),
    op_target(name = "op_neg"),
    op_target(
        name = "op_sub",
        deps = [
            ":binary_ops",
            ":add_sub_impl",
            "//executorch/kernels/portable/cpu:scalar_utils",
            "//executorch/kernels/portable/cpu/util:broadcast_util",
        ],
    ),
    op_target(
        name = "op_where",
        deps = [
            "//executorch/extension/threadpool:threadpool",
            "//executorch/kernels/portable/cpu/util:elementwise_util",
        ],
    ),
)


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
    for op in _OPTIMIZED_ATEN_OPS:
        define_op_target(**op)

    aten_op_targets = [":{}".format(op["name"]) for op in _OPTIMIZED_ATEN_OPS]
    all_op_targets = aten_op_targets

    runtime.cxx_library(
        name = "add_sub_impl",
        srcs = [],
        exported_headers = ["op_add_sub_impl.h"],
        visibility = ["//executorch/kernels/optimized/cpu/..."],
        exported_deps = ["//executorch/runtime/core:core"],
    )

    runtime.cxx_library(
        name = "binary_ops",
        exported_headers = ["binary_ops.h"],
        visibility = ["//executorch/kernels/optimized/cpu/..."],
        exported_deps = ["//executorch/runtime/core:core"],
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
        visibility = ["//executorch/kernels/optimized/..."],
        exported_deps = [
            "//executorch/kernels/optimized:libvec",
            "//executorch/kernels/optimized:libutils",
        ],
    )
