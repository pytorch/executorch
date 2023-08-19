load("@fbsource//xplat/executorch/build:runtime_wrapper.bzl", "runtime")
load("@fbsource//xplat/executorch/kernels/portable:op_registration_util.bzl", "define_op_target", "op_target")

# Operators that are listed in `functions.yaml`, and are thus compatible with
# the core ATen operators. Every entry here will be backed by a cxx_library
# target with the given name and deps.
#
# Note that a single target (or single .cpp file) can't mix ATen and non-ATen
# ops, and must be split. They can, however, share common code via a library dep
# if necessary.
_ATEN_OPS = (
    op_target(
        name = "op_abs",
        deps = [
            "//executorch/kernels/portable/cpu/util:functional_util",
        ],
    ),
    op_target(
        name = "op_acos",
        deps = [
            "//executorch/kernels/portable/cpu/pattern:pattern",
        ],
    ),
    op_target(
        name = "op_acosh",
        deps = [
            "//executorch/kernels/portable/cpu/pattern:pattern",
        ],
    ),
    op_target(
        name = "op_add",
        deps = [
            "//executorch/kernels/portable/cpu/util:broadcast_util",
            "//executorch/kernels/portable/cpu/util:functional_util",
            ":scalar_utils",
        ],
    ),
    op_target(
        name = "op_addmm",
        deps = [
            "//executorch/kernels/portable/cpu/util:broadcast_util",
            "//executorch/kernels/portable/cpu/util:matmul_ops_util",
            ":scalar_utils",
            ":vec_ops",
        ],
    ),
    op_target(
        name = "op_alias_copy",
    ),
    op_target(
        name = "op_amax",
        deps = [
            "//executorch/runtime/core/exec_aten/util:scalar_type_util",
            "//executorch/runtime/core/exec_aten/util:tensor_util",
            "//executorch/kernels/portable/cpu/util:reduce_util",
        ],
    ),
    op_target(
        name = "op_amin",
        deps = [
            "//executorch/runtime/core/exec_aten/util:scalar_type_util",
            "//executorch/runtime/core/exec_aten/util:tensor_util",
            "//executorch/kernels/portable/cpu/util:reduce_util",
        ],
    ),
    op_target(
        name = "op_any",
        deps = [
            "//executorch/runtime/core/exec_aten/util:tensor_util",
        ],
    ),
    op_target(
        name = "op_arange",
        deps = [
            "//executorch/runtime/core/exec_aten/util:scalar_type_util",
            "//executorch/runtime/core/exec_aten/util:tensor_util",
            ":scalar_utils",
        ],
    ),
    op_target(
        name = "op_argmax",
        deps = [
            "//executorch/kernels/portable/cpu/util:reduce_util",
        ],
    ),
    op_target(
        name = "op_argmin",
        deps = [
            "//executorch/kernels/portable/cpu/util:reduce_util",
        ],
    ),
    op_target(
        name = "op_as_strided_copy",
        deps = [
            ":scalar_utils",
        ],
    ),
    op_target(
        name = "op_asin",
        deps = [
            "//executorch/kernels/portable/cpu/pattern:pattern",
        ],
    ),
    op_target(
        name = "op_asinh",
        deps = [
            "//executorch/kernels/portable/cpu/pattern:pattern",
        ],
    ),
    op_target(
        name = "op_atan",
        deps = [
            "//executorch/kernels/portable/cpu/pattern:pattern",
        ],
    ),
    op_target(
        name = "op_atanh",
        deps = [
            "//executorch/kernels/portable/cpu/pattern:pattern",
        ],
    ),
    op_target(
        name = "op_avg_pool2d",
        deps = [
            "//executorch/kernels/portable/cpu/util:kernel_ops_util",
        ],
    ),
    op_target(
        name = "op_bitwise_and",
        deps = [
            "//executorch/runtime/core/exec_aten/util:scalar_type_util",
            "//executorch/runtime/core/exec_aten/util:tensor_util",
            "//executorch/kernels/portable/cpu/util:broadcast_util",
            "//executorch/kernels/portable/cpu/util:functional_util",
            ":scalar_utils",
        ],
    ),
    op_target(
        name = "op_bitwise_not",
        deps = [
            "//executorch/runtime/core/exec_aten/util:scalar_type_util",
            "//executorch/runtime/core/exec_aten/util:tensor_util",
            "//executorch/kernels/portable/cpu/util:functional_util",
        ],
    ),
    op_target(
        name = "op_bitwise_or",
        deps = [
            "//executorch/runtime/core/exec_aten/util:scalar_type_util",
            "//executorch/runtime/core/exec_aten/util:tensor_util",
            "//executorch/kernels/portable/cpu/util:broadcast_util",
            "//executorch/kernels/portable/cpu/util:functional_util",
            ":scalar_utils",
        ],
    ),
    op_target(
        name = "op_bitwise_xor",
        deps = [
            "//executorch/runtime/core/exec_aten/util:scalar_type_util",
            "//executorch/runtime/core/exec_aten/util:tensor_util",
            "//executorch/kernels/portable/cpu/util:broadcast_util",
            "//executorch/kernels/portable/cpu/util:functional_util",
            ":scalar_utils",
        ],
    ),
    op_target(
        name = "op_pow",
        deps = [
            ":scalar_utils",
            "//executorch/kernels/portable/cpu/util:broadcast_util",
            "//executorch/kernels/portable/cpu/util:functional_util",
        ],
    ),
    op_target(
        name = "op_bmm",
        deps = [
            "//executorch/kernels/portable/cpu/util:matmul_ops_util",
            ":vec_ops",
        ],
    ),
    op_target(
        name = "op_cat",
        deps = [
            "//executorch/kernels/portable/cpu/util:copy_ops_util",
        ],
    ),
    op_target(
        name = "op_ceil",
        deps = [
            "//executorch/kernels/portable/cpu/pattern:pattern",
        ],
    ),
    op_target(
        name = "op_clamp",
        deps = [
            ":scalar_utils",
            "//executorch/kernels/portable/cpu/util:functional_util",
        ],
    ),
    op_target(
        name = "op_clone",
    ),
    op_target(
        name = "op_constant_pad_nd",
        deps = [":scalar_utils"],
    ),
    op_target(
        name = "op_convolution",
        deps = [
            "//executorch/kernels/portable/cpu/util:kernel_ops_util",
            ":vec_ops",
        ],
    ),
    op_target(
        name = "op_copy",
        deps = [
            "//executorch/kernels/portable/cpu/util:broadcast_util",
            "//executorch/runtime/core/exec_aten/util:scalar_type_util",
            "//executorch/runtime/core/exec_aten/util:tensor_util",
            ":scalar_utils",
        ],
    ),
    op_target(
        name = "op_cos",
        deps = [
            "//executorch/kernels/portable/cpu/pattern:pattern",
        ],
    ),
    op_target(
        name = "op_cosh",
        deps = [
            "//executorch/kernels/portable/cpu/pattern:pattern",
        ],
    ),
    op_target(
        name = "op_cumsum",
        deps = [
            "//executorch/runtime/core/exec_aten/util:scalar_type_util",
            "//executorch/runtime/core/exec_aten/util:tensor_util",
        ],
    ),
    op_target(
        name = "op_detach_copy",
        deps = [
            "//executorch/runtime/core/exec_aten/util:tensor_util",
        ],
    ),
    op_target(
        name = "op_div",
        deps = [
            "//executorch/kernels/portable/cpu/util:broadcast_util",
            "//executorch/kernels/portable/cpu/util:functional_util",
            ":scalar_utils",
        ],
    ),
    op_target(
        name = "op_embedding",
    ),
    op_target(
        name = "op_eq",
        deps = [
            ":scalar_utils",
            "//executorch/kernels/portable/cpu/util:broadcast_util",
            "//executorch/kernels/portable/cpu/util:functional_util",
        ],
    ),
    op_target(
        name = "op_empty",
    ),
    op_target(
        name = "op_erf",
        deps = [
            "//executorch/kernels/portable/cpu/pattern:pattern",
        ],
    ),
    op_target(
        name = "op_exp",
        deps = [
            "//executorch/kernels/portable/cpu/pattern:pattern",
        ],
    ),
    op_target(
        name = "op_expand_copy",
        deps = [
            "//executorch/runtime/core/exec_aten/util:scalar_type_util",
            "//executorch/runtime/core/exec_aten/util:tensor_util",
            "//executorch/kernels/portable/cpu/util:repeat_util",
            ":scalar_utils",
        ],
    ),
    op_target(
        name = "op_fill",
        deps = [
            ":scalar_utils",
            "//executorch/kernels/portable/cpu/util:functional_util",
            "//executorch/runtime/core/exec_aten/util:scalar_type_util",
            "//executorch/runtime/core/exec_aten/util:tensor_util",
        ],
    ),
    op_target(
        name = "op_floor",
        deps = [
            "//executorch/kernels/portable/cpu/pattern:pattern",
        ],
    ),
    op_target(
        name = "op_floor_divide",
        deps = [
            "//executorch/kernels/portable/cpu/util:broadcast_util",
        ],
    ),
    op_target(
        name = "op_fmod",
        deps = [
            ":scalar_utils",
            "//executorch/kernels/portable/cpu/util:broadcast_util",
            "//executorch/kernels/portable/cpu/util:functional_util",
        ],
    ),
    op_target(
        name = "op_full",
        deps = [
            "//executorch/runtime/core/exec_aten/util:scalar_type_util",
            "//executorch/runtime/core/exec_aten/util:tensor_util",
            ":scalar_utils",
        ],
    ),
    op_target(
        name = "op_full_like",
        deps = [
            "//executorch/runtime/core/exec_aten/util:scalar_type_util",
            "//executorch/runtime/core/exec_aten/util:tensor_util",
            ":scalar_utils",
        ],
    ),
    op_target(
        name = "op_ge",
        deps = [
            ":scalar_utils",
            "//executorch/kernels/portable/cpu/util:broadcast_util",
            "//executorch/kernels/portable/cpu/util:functional_util",
        ],
    ),
    op_target(
        name = "op_gelu",
        deps = [
            ":math_constants",
            "//executorch/kernels/portable/cpu/util:functional_util",
        ],
    ),
    op_target(
        name = "op_glu",
        deps = [
            "//executorch/runtime/core/exec_aten/util:scalar_type_util",
            "//executorch/runtime/core/exec_aten/util:tensor_util",
        ],
    ),
    op_target(
        name = "op_gt",
        deps = [
            ":scalar_utils",
            "//executorch/kernels/portable/cpu/util:broadcast_util",
            "//executorch/kernels/portable/cpu/util:functional_util",
        ],
    ),
    op_target(
        name = "op_hardtanh",
        deps = [
            "//executorch/kernels/portable/cpu/util:functional_util",
            ":scalar_utils",
        ],
    ),
    op_target(
        name = "op_index",
        deps = [
            "//executorch/kernels/portable/cpu/util:index_util",
        ],
    ),
    op_target(
        name = "op_index_put",
        deps = [
            "//executorch/kernels/portable/cpu/util:index_util",
        ],
    ),
    op_target(
        name = "op_index_select",
        deps = [
            "//executorch/kernels/portable/cpu/util:index_util",
        ],
    ),
    op_target(
        name = "op_isinf",
        deps = [
            "//executorch/kernels/portable/cpu/pattern:pattern",
        ],
    ),
    op_target(
        name = "op_isnan",
        deps = [
            "//executorch/kernels/portable/cpu/pattern:pattern",
        ],
    ),
    op_target(
        name = "op_le",
        deps = [
            ":scalar_utils",
            "//executorch/kernels/portable/cpu/util:broadcast_util",
            "//executorch/kernels/portable/cpu/util:functional_util",
        ],
    ),
    op_target(
        name = "op_leaky_relu",
        deps = [
            "//executorch/kernels/portable/cpu/util:functional_util",
            ":scalar_utils",
        ],
    ),
    op_target(
        name = "op_lift_fresh_copy",
        deps = [
            "//executorch/runtime/core/exec_aten/util:tensor_util",
        ],
    ),
    op_target(
        name = "op_log",
        deps = [
            "//executorch/kernels/portable/cpu/pattern:pattern",
        ],
    ),
    op_target(
        name = "op_log_softmax",
        deps = [
            ":vec_ops",
            "//executorch/kernels/portable/cpu/util:functional_util",
            "//executorch/kernels/portable/cpu/util:reduce_util",
        ],
    ),
    op_target(
        name = "op_logical_and",
        deps = [
            "//executorch/kernels/portable/cpu/pattern:pattern",
        ],
    ),
    op_target(
        name = "op_logical_not",
        deps = [
            "//executorch/runtime/core/exec_aten/util:tensor_util",
            "//executorch/kernels/portable/cpu/util:functional_util",
        ],
    ),
    op_target(
        name = "op_logical_or",
        deps = [
            "//executorch/kernels/portable/cpu/pattern:pattern",
        ],
    ),
    op_target(
        name = "op_logical_xor",
        deps = [
            "//executorch/kernels/portable/cpu/pattern:pattern",
        ],
    ),
    op_target(
        name = "op_logit",
        deps = [
            "//executorch/kernels/portable/cpu/util:functional_util",
        ],
    ),
    op_target(
        name = "op_lt",
        deps = [
            ":scalar_utils",
            "//executorch/kernels/portable/cpu/util:broadcast_util",
            "//executorch/kernels/portable/cpu/util:functional_util",
        ],
    ),
    op_target(
        name = "op_masked_fill",
        deps = [
            "//executorch/kernels/portable/cpu/util:broadcast_util",
            ":scalar_utils",
        ],
    ),
    op_target(
        name = "op_max",
        deps = [
            "//executorch/runtime/core/exec_aten/util:scalar_type_util",
            "//executorch/runtime/core/exec_aten/util:tensor_util",
            "//executorch/kernels/portable/cpu/util:reduce_util",
        ],
    ),
    op_target(
        name = "op_max_pool2d_with_indices",
        deps = [
            "//executorch/kernels/portable/cpu/util:kernel_ops_util",
        ],
    ),
    op_target(
        name = "op_mean",
        deps = [
            "//executorch/runtime/core/exec_aten/util:scalar_type_util",
            "//executorch/runtime/core/exec_aten/util:tensor_util",
            "//executorch/kernels/portable/cpu/util:reduce_util",
        ],
    ),
    op_target(
        name = "op_min",
        deps = [
            "//executorch/runtime/core/exec_aten/util:scalar_type_util",
            "//executorch/runtime/core/exec_aten/util:tensor_util",
            "//executorch/kernels/portable/cpu/util:reduce_util",
        ],
    ),
    op_target(
        name = "op_minimum",
        deps = [
            "//executorch/kernels/portable/cpu/util:broadcast_util",
            ":scalar_utils",
        ],
    ),
    op_target(
        name = "op_mm",
        deps = [
            "//executorch/kernels/portable/cpu/util:matmul_ops_util",
            ":vec_ops",
        ],
    ),
    op_target(
        name = "op_mul",
        deps = [
            "//executorch/kernels/portable/cpu/util:broadcast_util",
            "//executorch/kernels/portable/cpu/util:functional_util",
            ":scalar_utils",
        ],
    ),
    op_target(
        name = "op_native_batch_norm",
        deps = [
            "//executorch/kernels/portable/cpu/util:normalization_ops_util",
        ],
    ),
    op_target(
        name = "op_native_layer_norm",
        deps = [
            ":vec_ops",
            "//executorch/kernels/portable/cpu/util:normalization_ops_util",
        ],
    ),
    op_target(
        name = "op_ne",
        deps = [
            ":scalar_utils",
            "//executorch/kernels/portable/cpu/util:broadcast_util",
            "//executorch/kernels/portable/cpu/util:functional_util",
        ],
    ),
    op_target(
        name = "op_neg",
        deps = [
            "//executorch/kernels/portable/cpu/util:functional_util",
        ],
    ),
    op_target(
        name = "op_nonzero",
        deps = [
            ":scalar_utils",
        ],
    ),
    op_target(
        name = "op_ones",
        deps = [
            "//executorch/runtime/core/exec_aten/util:tensor_util",
        ],
    ),
    op_target(
        name = "op_permute_copy",
        deps = [
            "//executorch/kernels/portable/cpu/util:copy_ops_util",
        ],
    ),
    op_target(
        name = "op_reciprocal",
        deps = [
            "//executorch/kernels/portable/cpu/pattern:pattern",
        ],
    ),
    op_target(
        name = "op_relu",
        deps = [
            "//executorch/kernels/portable/cpu/util:functional_util",
        ],
    ),
    op_target(
        name = "op_remainder",
        deps = [
            ":scalar_utils",
            "//executorch/kernels/portable/cpu/util:broadcast_util",
            "//executorch/kernels/portable/cpu/util:functional_util",
        ],
    ),
    op_target(
        name = "op_repeat",
        deps = [
            "//executorch/kernels/portable/cpu/util:repeat_util",
        ],
    ),
    op_target(
        name = "op_round",
        deps = [
            "//executorch/kernels/portable/cpu/util:functional_util",
        ],
    ),
    op_target(
        name = "op_rsqrt",
        deps = [
            "//executorch/kernels/portable/cpu/pattern:pattern",
        ],
    ),
    op_target(
        name = "op_rsub",
        deps = [
            ":scalar_utils",
            "//executorch/kernels/portable/cpu/util:functional_util",
        ],
    ),
    op_target(
        name = "op_scalar_tensor",
        deps = [":scalar_utils"],
    ),
    op_target(
        name = "op_scatter_add",
        deps = [
            "//executorch/runtime/core/exec_aten/util:scalar_type_util",
            "//executorch/runtime/core/exec_aten/util:tensor_util",
        ],
    ),
    op_target(
        name = "op_select_copy",
    ),
    op_target(
        name = "op_select_scatter",
        deps = [
            "//executorch/runtime/core/exec_aten/util:scalar_type_util",
            "//executorch/runtime/core/exec_aten/util:tensor_util",
        ],
    ),
    op_target(
        name = "op_sigmoid",
        deps = [
            "//executorch/kernels/portable/cpu/util:functional_util",
        ],
    ),
    op_target(
        name = "op_sign",
        deps = [
            "//executorch/kernels/portable/cpu/util:functional_util",
        ],
    ),
    op_target(
        name = "op_sin",
        deps = [
            "//executorch/kernels/portable/cpu/pattern:pattern",
        ],
    ),
    op_target(
        name = "op_sinh",
        deps = [
            "//executorch/kernels/portable/cpu/pattern:pattern",
        ],
    ),
    op_target(
        name = "op_slice_copy",
    ),
    op_target(
        name = "op_slice_scatter",
    ),
    op_target(
        name = "op_softmax",
        deps = [
            ":vec_ops",
            "//executorch/kernels/portable/cpu/util:functional_util",
            "//executorch/kernels/portable/cpu/util:reduce_util",
        ],
    ),
    op_target(
        name = "op_split_copy",
    ),
    op_target(
        name = "op_sqrt",
        deps = [
            "//executorch/kernels/portable/cpu/pattern:pattern",
        ],
    ),
    op_target(
        name = "op_squeeze_copy",
    ),
    op_target(
        name = "op_stack",
        deps = [
            "//executorch/kernels/portable/cpu/util:copy_ops_util",
        ],
    ),
    op_target(
        name = "op_sub",
        deps = [
            ":scalar_utils",
            "//executorch/kernels/portable/cpu/util:broadcast_util",
            "//executorch/kernels/portable/cpu/util:functional_util",
        ],
    ),
    op_target(
        name = "op_sum",
        deps = [
            "//executorch/runtime/core/exec_aten/util:scalar_type_util",
            "//executorch/runtime/core/exec_aten/util:tensor_util",
            "//executorch/kernels/portable/cpu/util:reduce_util",
        ],
    ),
    op_target(
        name = "op_t_copy",
        deps = ["//executorch/kernels/portable/cpu/util:transpose_util"],
    ),
    op_target(
        name = "op_tan",
        deps = [
            "//executorch/kernels/portable/cpu/pattern:pattern",
        ],
    ),
    op_target(
        name = "op_tanh",
        deps = [
            "//executorch/kernels/portable/cpu/pattern:pattern",
        ],
    ),
    op_target(
        name = "op_to_copy",
    ),
    op_target(
        name = "op_transpose_copy",
        deps = ["//executorch/kernels/portable/cpu/util:transpose_util"],
    ),
    op_target(
        name = "op_tril",
        deps = [
            "//executorch/runtime/core/exec_aten/util:tensor_util",
        ],
    ),
    op_target(
        name = "op_unbind_copy",
    ),
    op_target(
        name = "op_unsqueeze_copy",
    ),
    op_target(
        name = "op_var",
        deps = [
            "//executorch/runtime/core/exec_aten/util:scalar_type_util",
            "//executorch/runtime/core/exec_aten/util:tensor_util",
            "//executorch/kernels/portable/cpu/util:reduce_util",
        ],
    ),
    op_target(
        name = "op_view_copy",
    ),
    op_target(
        name = "op_where",
        deps = [
            "//executorch/kernels/portable/cpu/util:broadcast_util",
            "//executorch/kernels/portable/cpu/util:functional_util",
            "//executorch/runtime/core/exec_aten:lib",
        ],
    ),
    op_target(
        name = "op_zeros",
    ),
)

# Operators that are not listed in `functions.yaml` (i.e., operators listed in
# `custom_ops.yaml`), which are not compatible with the core ATen operators.
# Every entry here will be backed by a cxx_library target with the given name
# and deps, as well as a similar `<name>_aten` target that uses at::Tensor and
# related types.
#
# Note that a single target (or single .cpp file) can't mix ATen and non-ATen
# ops, and must be split. They can, however, share common code via a library dep
# if necessary.
_CUSTOM_OPS = (
    op_target(
        name = "op_allclose",
    ),
    op_target(
        name = "op_linear_scratch_example",
    ),
)

def define_common_targets():
    """Defines targets that should be shared between fbcode and xplat.

    The directory containing this targets.bzl file should also contain both
    TARGETS and BUCK files that call this function.
    """

    # Define build targets for all operators registered in the tables above.
    for op in _ATEN_OPS:
        define_op_target(is_aten_op = True, **op)
    for op in _CUSTOM_OPS:
        define_op_target(is_aten_op = False, **op)

    custom_op_targets = [":{}".format(op["name"]) for op in _CUSTOM_OPS]

    aten_op_targets = [":{}".format(op["name"]) for op in _ATEN_OPS]
    all_op_targets = custom_op_targets + aten_op_targets

    runtime.cxx_library(
        name = "cpu",
        srcs = [],
        visibility = [
            "//executorch/kernels/portable/...",
            "//executorch/kernels/test/...",
        ],
        exported_deps = all_op_targets,
    )

    runtime.cxx_library(
        name = "cpu_aten",
        srcs = [],
        visibility = ["//executorch/kernels/portable/..."],
        exported_deps = [t + "_aten" for t in custom_op_targets],
    )

    # Only for use by op targets under //executorch. This API needs to be
    # reevaluated before becoming a public API.
    runtime.cxx_library(
        name = "vec_ops",
        srcs = [],
        exported_headers = ["vec_ops.h"],
        visibility = ["//executorch/kernels/portable/cpu/...", "//executorch/kernels/quantized/..."],
    )

    # Only for use by targets in this directory. Defines constants like M_PI
    # if they arent already defined by the toolchains cmath
    runtime.cxx_library(
        name = "math_constants",
        srcs = [],
        exported_headers = [
            "math_constants.h",
        ],
        visibility = [
            "//executorch/kernels/portable/cpu/...",
        ],
    )

    # Only for use by targets in this directory.
    runtime.cxx_library(
        name = "scalar_utils",
        srcs = [],
        exported_headers = ["scalar_utils.h"],
        visibility = [
            "//executorch/kernels/portable/cpu/...",
            "//executorch/kernels/optimized/cpu/...",
            "@EXECUTORCH_CLIENTS",
        ],
        deps = [
            "//executorch/runtime/core/exec_aten:lib",
            "//executorch/runtime/core/exec_aten/util:scalar_type_util",
        ],
    )
