load("@fbsource//xplat/executorch/build:runtime_wrapper.bzl", "runtime")
load("@fbsource//xplat/executorch/kernels/optimized:op_registration_util.bzl", "define_op_target", "is_op_disabled", "op_target")

_OPTIMIZED_ATEN_OPS = (
    op_target(
        name = "op_add",
        deps = [
            ":binary_ops",
            "//executorch/kernels/portable/cpu:scalar_utils",
            "//executorch/kernels/portable/cpu/util:broadcast_util",
        ],
    ),
    op_target(
        name = "op_bmm",
        deps = [
            "//executorch/kernels/optimized:libblas",
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
    op_target(name = "op_exp"),
    op_target(name = "op_sigmoid"),
    op_target(
        name = "op_gelu",
        deps = [
            ":aten_headers_for_executorch",
        ],
    ),
    op_target(
        name = "op_le",
        deps = [
            "//executorch/kernels/portable/cpu:scalar_utils",
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
        deps = select({
            "DEFAULT": [
                "//executorch/kernels/portable/cpu/util:activation_ops_util",
            ],
            "ovr_config//cpu:arm64": [
                "//executorch/kernels/portable/cpu/util:activation_ops_util",
                "fbsource//third-party/sleef:sleef_arm",
            ],
        }),
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
            "//executorch/kernels/portable/cpu:scalar_utils",
            "//executorch/kernels/portable/cpu/util:broadcast_util",
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

    enabled_ops = [op for op in _OPTIMIZED_ATEN_OPS if not is_op_disabled(op["name"])]

    # Define build targets for all operators registered in the tables above.
    for op in enabled_ops:
        define_op_target(**op)

    aten_op_targets = [":{}".format(op["name"]) for op in enabled_ops]
    all_op_targets = aten_op_targets

    runtime.cxx_library(
        name = "aten_headers_for_executorch",
        srcs = [],
        visibility = ["//executorch/kernels/optimized/..."],
        exported_deps = select({
            "DEFAULT": [],
            "ovr_config//cpu:arm64": [
                "fbsource//third-party/sleef:sleef_arm",
            ] if not runtime.is_oss else [],
            # fbsource//third-party/sleef:sleef currently fails to
            # link with missing symbols, hence the fbcode-specific dep below.
        }),
        fbcode_exported_deps = [
            "//caffe2:aten-headers-cpu",
            "//caffe2:generated-config-header",
            "//caffe2/c10/core:base",
        ] + select({
            "DEFAULT": [],
            "ovr_config//cpu:x86_64": [
                "third-party//sleef:sleef",
            ]
        }),
        xplat_exported_deps = [
            "//xplat/caffe2:aten_header",
            "//xplat/caffe2:generated_aten_config_header",
            "//xplat/caffe2/c10:c10",
        ],
        exported_preprocessor_flags = select({
            "ovr_config//cpu:x86_64": [
                "-DCPU_CAPABILITY=AVX2",
                "-DCPU_CAPABILITY_AVX2",
                "-DHAVE_AVX2_CPU_DEFINITION",
            ] + get_sleef_preprocessor_flags(),
            "ovr_config//cpu:arm64": get_sleef_preprocessor_flags(),
            "DEFAULT": [],
        }) + ["-DSTANDALONE_TORCH_HEADER"],
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
