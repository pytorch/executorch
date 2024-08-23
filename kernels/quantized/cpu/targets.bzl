load("@fbsource//xplat/executorch/build:runtime_wrapper.bzl", "runtime")
load("@fbsource//xplat/executorch/kernels/portable:op_registration_util.bzl", "define_op_target", "op_target")

_QUANT_OPS = (
    op_target(
        name = "op_add",
    ),
    op_target(
        name = "op_choose_qparams",
        deps = [
            "//executorch/kernels/portable/cpu:vec_ops",
        ],
    ),
    op_target(
        name = "op_dequantize",
        deps = [
            "//executorch/kernels/portable/cpu/util:reduce_util",
        ],
        _aten_mode_deps = [
            "//executorch/kernels/portable/cpu/util:reduce_util_aten",
        ],
    ),
    op_target(
        name = "op_embedding",
    ),
    op_target(
        name = "op_embedding4b",
    ),
    op_target(
        name = "op_mixed_mm",
        deps = [
            "//executorch/kernels/portable/cpu:vec_ops",
        ],
    ),
    op_target(
        name = "op_mixed_linear",
        deps = [
            "//executorch/kernels/portable/cpu:vec_ops",
        ],
    ),
    op_target(
        name = "op_quantize",
        deps = [
            "//executorch/kernels/portable/cpu/util:reduce_util",
        ],
        _aten_mode_deps = [
            "//executorch/kernels/portable/cpu/util:reduce_util_aten",
        ],
    ),
)

def define_common_targets():
    for op in _QUANT_OPS:
        define_op_target(is_aten_op = False, **op)

    quant_op_targets = [":{}".format(op["name"]) for op in _QUANT_OPS]

    runtime.cxx_library(
        name = "quantized_cpu",
        srcs = [],
        visibility = [
            "//executorch/kernels/quantized/...",
            "//executorch/extension/pybindings/test/...",
        ],
        exported_deps = quant_op_targets,
    )

    runtime.cxx_library(
        name = "quantized_cpu_aten",
        srcs = [],
        visibility = ["//executorch/kernels/quantized/..."],
        exported_deps = [t + "_aten" for t in quant_op_targets],
    )
