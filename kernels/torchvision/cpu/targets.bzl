load("@fbsource//xplat/executorch/build:runtime_wrapper.bzl", "runtime")
load("@fbsource//xplat/executorch/kernels/portable:op_registration_util.bzl", "define_op_target", "op_target")

_TORCHVISION_OPS = (
    op_target(
        name = "op_nms",
        deps = [
            "//executorch/kernels/portable/cpu/util:broadcast_util",
            "//executorch/kernels/portable/cpu/util:select_copy_util",
            "//executorch/kernels/portable/cpu/util:sort_util",
        ],
    ),
)

def define_common_targets():
    for op in _TORCHVISION_OPS:
        define_op_target(is_aten_op = False, **op)

    torchvision_op_targets = [":{}".format(op["name"]) for op in _TORCHVISION_OPS]

    runtime.cxx_library(
        name = "torchvision_cpu",
        srcs = [],
        visibility = ["//executorch/kernels/torchvision/..."],
        exported_deps = torchvision_op_targets,
    )
