load("@fbsource//tools/build_defs:platform_defs.bzl", "CXX")
load("@fbsource//xplat/executorch/build:runtime_wrapper.bzl", "runtime")

def define_operator(name: str, deps: list[str] | None = None) -> None:
    op_name = "op_{}".format(name)

    # Deps used by all operators.
    common_deps = [
        "//executorch/kernels/portable/cpu/util:all_deps",
        "//executorch/kernels/portable/cpu/pattern:all_deps",
        "//executorch/runtime/kernel:kernel_includes",
        "//executorch/kernels/portable/cpu:scalar_utils",
        "fbsource//third-party/nnlib-FusionG3/xa_nnlib:libxa_nnlib_common",
        "fbsource//third-party/nnlib-FusionG3/xa_nnlib:libxa_nnlib",
    ]
    if deps == None:
        deps = []

    runtime.cxx_library(
        name = op_name,
        srcs = [op_name + ".cpp"],
        platforms = CXX,
        visibility = [
            "//executorch/backends/cadence/...",
            "@EXECUTORCH_CLIENTS",
        ],
        compatible_with = ["ovr_config//cpu:xtensa"],
        deps = deps + common_deps,
        exported_deps = [
            ":operators_header",
            ":xt_macros",
            ":xt_utils",
        ],
    )

OPERATORS = [
    "add",
    "cat",
    "clamp",
    "lt",
    "rsqrt",
    "sigmoid",
    "sqrt",
    "hardtanh",
    "tanh",
    "transpose_copy",
    "where",
    "dequantize",
    "mul",
    "native_layer_norm",
    "quantize",
    "softmax",
    "sub",
    "div",
    "exp",
    "mean",
    "slice_copy",
    "permute_copy"
]

def define_common_targets():
    """Defines targets that should be shared between fbcode and xplat.

    The directory containing this targets.bzl file should also contain both
    TARGETS and BUCK files that call this function.
    """

    # Define build targets for all operators registered in the tables above.

    runtime.cxx_library(
        name = "operators_header",
        exported_headers = ["operators.h"],
        visibility = [
            "//executorch/backends/cadence/...",
        ],
        exported_deps = [
            "//executorch/runtime/core/exec_aten:lib",
            "//executorch/runtime/kernel:kernel_runtime_context",
        ],
    )

    runtime.cxx_library(
        name = "xt_macros",
        exported_headers = ["xt_macros.h"],
        visibility = [
            "//executorch/backends/cadence/...",
        ],
        exported_deps = [
            "//executorch/runtime/core/exec_aten:lib",
            "//executorch/runtime/kernel:kernel_runtime_context",
        ],
    )

    runtime.cxx_library(
        name = "xt_utils",
        exported_headers = ["xt_utils.h"],
        visibility = [
            "//executorch/backends/cadence/...",
        ],
        exported_deps = [
            "//executorch/runtime/core/exec_aten:lib",
            "//executorch/runtime/kernel:kernel_runtime_context",
        ],
    )

    for op in OPERATORS:
        define_operator(op)
