load("@fbsource//xplat/executorch/build:runtime_wrapper.bzl", "runtime")
load("@fbsource//xplat/executorch/kernels/optimized:op_registration_util.bzl", "define_op_target", "op_target")

_OPTIMIZED_ATEN_OPS = (
    op_target(
        name = "op_add",
        deps = [
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
            "//executorch/kernels/portable/cpu:scalar_utils",
            "//executorch/kernels/portable/cpu/util:broadcast_util",
        ],
    ),
    op_target(name = "op_exp"),
    op_target(
        name = "op_gelu",
        deps = select({
            "DEFAULT": [],
            "ovr_config//runtime:fbcode-arm64": [
                "fbsource//third-party/sleef:sleef_arm",
            ],
        }),
    ),
    op_target(
        name = "op_le",
        deps = [
            "//executorch/kernels/portable/cpu:scalar_utils",
        ],
    ),
    op_target(
        name = "op_log_softmax",
        deps = select({
            "DEFAULT": [],
            "ovr_config//runtime:fbcode-arm64": [
                "fbsource//third-party/sleef:sleef_arm",
            ],
        }),
    ),
    op_target(
        name = "op_mul",
        deps = [
            "//executorch/kernels/portable/cpu:scalar_utils",
            "//executorch/kernels/portable/cpu/util:broadcast_util",
        ],
    ),
    op_target(
        name = "op_native_layer_norm",
        deps = [
            ":moments_utils",
        ],
    ),
    op_target(name = "op_neg"),
    op_target(
        name = "op_sub",
        deps = [
            "//executorch/kernels/portable/cpu:scalar_utils",
            "//executorch/kernels/portable/cpu/util:broadcast_util",
        ],
    ),
)

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
