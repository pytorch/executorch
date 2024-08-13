load("@fbsource//xplat/executorch/build:runtime_wrapper.bzl", "runtime")
load("@fbsource//xplat/executorch/kernels/portable:op_registration_util.bzl", "define_op_target", "op_target")

# Operators that are listed in `functions.yaml`, and are thus compatible with
# the core ATen operators. Every entry here will be backed by a cxx_library
# target with the given name and deps.
#
# Note that a single target (or single .cpp file) can't mix ATen and non-ATen
# ops, and must be split. They can, however, share common code via a library dep
# if necessary.
_EDGE_DIALECT_OPS = (
    op_target(
        name = "op__to_dim_order_copy",
        deps = [
            "//executorch/kernels/aten/cpu/util:copy_ops_util",
        ],
    ),
)

def define_common_targets():
    """Defines targets that should be shared between fbcode and xplat.

    The directory containing this targets.bzl file should also contain both
    TARGETS and BUCK files that call this function.
    """

    # Define build targets for all operators registered in the tables above.
    for op in _EDGE_DIALECT_OPS:
        define_op_target(is_aten_op = False, is_et_op = False, **op)

    all_op_targets = [":{}".format(op["name"]) for op in _EDGE_DIALECT_OPS]

    runtime.cxx_library(
        name = "cpu",
        srcs = [],
        visibility = [
            "//executorch/kernels/aten/...",
            "//executorch/kernels/test/...",
        ],
        exported_deps = [t + "_aten" for t in all_op_targets],
    )
