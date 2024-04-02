load("@fbsource//xplat/executorch/build:runtime_wrapper.bzl", "runtime")
load("@fbsource//xplat/executorch/codegen:codegen.bzl", "et_operator_library", "executorch_generated_lib")
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
            "//executorch/kernels/portable/cpu/util:kernel_ops_util",
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
            "//executorch/kernels/portable/cpu/util:index_util",
            "//executorch/kernels/portable/cpu/util:reduce_util",
        ],
    ),
    op_target(
        name = "op_any",
        deps = [
            "//executorch/kernels/portable/cpu/util:reduce_util",
            "//executorch/runtime/core/exec_aten/util:tensor_util",
        ],
    ),
    op_target(
        name = "op_arange",
        deps = [
            "//executorch/kernels/portable/cpu/util:kernel_ops_util",
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
            "//executorch/kernels/portable/cpu/util:copy_ops_util",
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
        name = "op_atan2",
        deps = [
            "//executorch/kernels/portable/cpu/util:broadcast_util",
            "//executorch/kernels/portable/cpu/util:functional_util",
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
        name = "op_cdist_forward",
        deps = [
            "//executorch/kernels/portable/cpu/util:broadcast_util",
            "//executorch/kernels/portable/cpu/util:distance_util",
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
            "//executorch/kernels/portable/cpu/util:broadcast_util",
            "//executorch/kernels/portable/cpu/util:functional_util",
            "//executorch/kernels/portable/cpu/util:math_util",
        ],
    ),
    op_target(
        name = "op_clone",
    ),
    op_target(
        name = "op_constant_pad_nd",
        deps = [
            ":scalar_utils",
            "//executorch/kernels/portable/cpu/util:kernel_ops_util",
        ],
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
            "//executorch/kernels/portable/cpu/util:kernel_ops_util",
        ],
    ),
    op_target(
        name = "op_detach_copy",
        deps = [
            "//executorch/runtime/core/exec_aten/util:tensor_util",
        ],
    ),
    op_target(
        name = "op_diagonal_copy",
        deps = [
            "//executorch/kernels/portable/cpu/util:copy_ops_util",
        ],
    ),
    op_target(
        name = "op_div",
        deps = [
            "//executorch/kernels/portable/cpu/util:broadcast_util",
            "//executorch/kernels/portable/cpu/util:functional_util",
            "//executorch/kernels/portable/cpu/util:math_util",
            ":scalar_utils",
        ],
    ),
    op_target(
        name = "op_embedding",
        deps = [
            "//executorch/kernels/portable/cpu/util:kernel_ops_util",
        ],
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
            "//executorch/kernels/portable/cpu/util:copy_ops_util",
            "//executorch/kernels/portable/cpu/util:repeat_util",
            ":scalar_utils",
        ],
    ),
    op_target(
        name = "op_expm1",
        deps = [
            "//executorch/kernels/portable/cpu/pattern:pattern",
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
        name = "op_flip",
        deps = [
            "//executorch/kernels/portable/cpu/util:reduce_util",
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
            "//executorch/kernels/portable/cpu/util:math_util",
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
            "//executorch/kernels/portable/cpu/util:activation_ops_util",
            "//executorch/kernels/portable/cpu/util:functional_util",
        ],
    ),
    op_target(
        name = "op_glu",
        deps = [
            "//executorch/kernels/portable/cpu/util:activation_ops_util",
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
            "//executorch/kernels/portable/cpu/util:math_util",
            ":scalar_utils",
        ],
    ),
    op_target(
        name = "op_index",
        deps = [
            "//executorch/kernels/portable/cpu/util:advanced_index_util",
            "//executorch/kernels/portable/cpu/util:broadcast_util",
        ],
    ),
    op_target(
        name = "op_index_put",
        deps = [
            "//executorch/kernels/portable/cpu/util:advanced_index_util",
            "//executorch/kernels/portable/cpu/util:broadcast_util",
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
        name = "op_log10",
        deps = [
            "//executorch/kernels/portable/cpu/pattern:pattern",
        ],
    ),
    op_target(
        name = "op_log1p",
        deps = [
            "//executorch/kernels/portable/cpu/pattern:pattern",
        ],
    ),
    op_target(
        name = "op_log2",
        deps = [
            "//executorch/kernels/portable/cpu/pattern:pattern",
        ],
    ),
    op_target(
        name = "op_log_softmax",
        deps = [
            ":vec_ops",
            "//executorch/kernels/portable/cpu/util:activation_ops_util",
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
            "//executorch/kernels/portable/cpu/util:kernel_ops_util",
            ":scalar_utils",
        ],
    ),
    op_target(
        name = "op_max",
        deps = [
            "//executorch/runtime/core/exec_aten/util:scalar_type_util",
            "//executorch/runtime/core/exec_aten/util:tensor_util",
            "//executorch/kernels/portable/cpu/util:index_util",
            "//executorch/kernels/portable/cpu/util:reduce_util",
        ],
    ),
    op_target(
        name = "op_maximum",
        deps = [
            "//executorch/kernels/portable/cpu/util:broadcast_util",
            ":scalar_utils",
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
            "//executorch/kernels/portable/cpu/util:kernel_ops_util",
            "//executorch/kernels/portable/cpu/util:reduce_util",
        ],
    ),
    op_target(
        name = "op_min",
        deps = [
            "//executorch/runtime/core/exec_aten/util:scalar_type_util",
            "//executorch/runtime/core/exec_aten/util:tensor_util",
            "//executorch/kernels/portable/cpu/util:index_util",
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
        name = "op_native_group_norm",
        deps = [
            ":vec_ops",
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
            "//executorch/kernels/portable/cpu/util:index_util",
            "//executorch/kernels/portable/cpu/util:kernel_ops_util",
        ],
    ),
    op_target(
        name = "op_ones",
        deps = [
            "//executorch/runtime/core/exec_aten/util:tensor_util",
        ],
    ),
    op_target(
        name = "op_pdist_forward",
        deps = [
            "//executorch/kernels/portable/cpu/util:distance_util",
        ],
    ),
    op_target(
        name = "op_permute_copy",
        deps = [
            "//executorch/kernels/portable/cpu/util:copy_ops_util",
        ],
    ),
    op_target(
        name = "op_pixel_shuffle",
        deps = [
            "//executorch/kernels/portable/cpu/util:copy_ops_util",
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
        name = "op_prod",
        deps = [
            "//executorch/runtime/core/exec_aten/util:scalar_type_util",
            "//executorch/runtime/core/exec_aten/util:tensor_util",
            "//executorch/kernels/portable/cpu/util:reduce_util",
        ],
    ),
    op_target(
        name = "op_reciprocal",
        deps = [
            "//executorch/kernels/portable/cpu/pattern:pattern",
        ],
    ),
    op_target(
        name = "op_reflection_pad1d",
        deps = [
            "//executorch/kernels/portable/cpu/util:padding_util",
        ],
    ),
    op_target(
        name = "op_reflection_pad2d",
        deps = [
            "//executorch/kernels/portable/cpu/util:padding_util",
        ],
    ),
    op_target(
        name = "op_reflection_pad3d",
        deps = [
            "//executorch/kernels/portable/cpu/util:padding_util",
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
            "//executorch/kernels/portable/cpu/util:math_util",
        ],
    ),
    op_target(
        name = "op_repeat",
        deps = [
            "//executorch/kernels/portable/cpu/util:repeat_util",
        ],
    ),
    op_target(
        name = "op_replication_pad1d",
        deps = [
            "//executorch/kernels/portable/cpu/util:padding_util",
        ],
    ),
    op_target(
        name = "op_replication_pad2d",
        deps = [
            "//executorch/kernels/portable/cpu/util:padding_util",
        ],
    ),
    op_target(
        name = "op_replication_pad3d",
        deps = [
            "//executorch/kernels/portable/cpu/util:padding_util",
        ],
    ),
    op_target(
        name = "op_roll",
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
            "//executorch/kernels/portable/cpu/util:kernel_ops_util",
        ],
    ),
    op_target(
        name = "op_scalar_tensor",
        deps = [":scalar_utils"],
    ),
    op_target(
        name = "op_scatter_add",
        deps = [
            "//executorch/kernels/portable/cpu/util:index_util",
            "//executorch/kernels/portable/cpu/util:kernel_ops_util",
            "//executorch/runtime/core/exec_aten/util:scalar_type_util",
            "//executorch/runtime/core/exec_aten/util:tensor_util",
        ],
    ),
    op_target(
        name = "op_select_copy",
        deps = [
            "//executorch/kernels/portable/cpu/util:copy_ops_util",
            "//executorch/kernels/portable/cpu/util:select_copy_util",
        ],
    ),
    op_target(
        name = "op_select_scatter",
        deps = [
            "//executorch/kernels/portable/cpu/util:kernel_ops_util",
            "//executorch/runtime/core/exec_aten/util:scalar_type_util",
            "//executorch/runtime/core/exec_aten/util:tensor_util",
            "//executorch/kernels/portable/cpu/util:index_util",
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
        deps = [
            "//executorch/kernels/portable/cpu/util:copy_ops_util",
            "//executorch/kernels/portable/cpu/util:index_util",
        ],
    ),
    op_target(
        name = "op_slice_scatter",
        deps = [
            "//executorch/kernels/portable/cpu/util:index_util",
            "//executorch/kernels/portable/cpu/util:kernel_ops_util",
        ],
    ),
    op_target(
        name = "op_softmax",
        deps = [
            ":vec_ops",
            "//executorch/kernels/portable/cpu/util:activation_ops_util",
            "//executorch/kernels/portable/cpu/util:functional_util",
            "//executorch/kernels/portable/cpu/util:reduce_util",
        ],
    ),
    op_target(
        name = "op_split_copy",
        deps = [
            "//executorch/kernels/portable/cpu/util:copy_ops_util",
        ],
    ),
    op_target(
        name = "op_split_with_sizes_copy",
        deps = [
            "//executorch/kernels/portable/cpu/util:broadcast_util",
            "//executorch/kernels/portable/cpu/util:copy_ops_util",
        ],
    ),
    op_target(
        name = "op_sqrt",
        deps = [
            "//executorch/kernels/portable/cpu/pattern:pattern",
        ],
    ),
    op_target(
        name = "op_squeeze_copy",
        deps = [
            "//executorch/kernels/portable/cpu/util:copy_ops_util",
        ],
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
            "//executorch/kernels/portable/cpu/util:kernel_ops_util",
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
        deps = [
            "//executorch/kernels/portable/cpu/util:copy_ops_util",
        ],
    ),
    op_target(
        name = "op_transpose_copy",
        deps = ["//executorch/kernels/portable/cpu/util:transpose_util"],
    ),
    op_target(
        name = "op_tril",
        deps = [
            "//executorch/kernels/portable/cpu/util:copy_ops_util",
        ],
    ),
    op_target(
        name = "op_trunc",
        deps = [
            "//executorch/kernels/portable/cpu/pattern:pattern",
        ],
    ),
    op_target(
        name = "op_unbind_copy",
        deps = [
            "//executorch/kernels/portable/cpu/util:copy_ops_util",
        ],
    ),
    op_target(
        name = "op_unsqueeze_copy",
        deps = [
            "//executorch/kernels/portable/cpu/util:copy_ops_util",
        ],
    ),
    op_target(
        name = "op_var",
        deps = [
            ":scalar_utils",
            "//executorch/runtime/core/exec_aten/util:scalar_type_util",
            "//executorch/runtime/core/exec_aten/util:tensor_util",
            "//executorch/kernels/portable/cpu/util:reduce_util",
        ],
    ),
    op_target(
        name = "op_view_copy",
        deps = [
            "//executorch/kernels/portable/cpu/util:copy_ops_util",
        ],
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

def define_common_targets(is_fbcode = False):
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

    dtype_selective_build_lib = native.read_config("executorch", "dtype_selective_build_lib", None)
    if dtype_selective_build_lib != None:
        # retrieve selected_op_variants.h from codegen
        genrule_name = dtype_selective_build_lib + "_et_op_dtype_gen[selected_op_variants]"
        runtime.cxx_library(
            name = "dtype_headers",
            srcs = [],
            exported_headers = {
                "selected_op_variants.h": genrule_name,
            },
            visibility = [
                "//executorch/...",
                "@EXECUTORCH_CLIENTS",
            ],
        )

    # Only for use by targets in this directory.
    runtime.cxx_library(
        name = "scalar_utils",
        srcs = [],
        # include dtype selective build flag and header
        exported_preprocessor_flags = ["-DEXECUTORCH_SELECTIVE_BUILD_DTYPE"] if dtype_selective_build_lib != None else [],
        exported_headers = ["scalar_utils.h", "selective_build.h"],
        exported_deps = [":dtype_headers"] if dtype_selective_build_lib != None else [],
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

    # dtype selective build test artifacts
    if is_fbcode:
        et_operator_library(
            name = "add_model",
            model = "fbcode//executorch/test/models:exported_programs[ModuleAdd.pte]",
        )

        executorch_generated_lib(
            name = "add_model_lib",
            functions_yaml_target = "//executorch/kernels/portable:functions.yaml",
            kernel_deps = ["//executorch/kernels/portable:operators"],
            deps = [":add_model"],
            visibility = ["//executorch/kernels/..."],
        )

        runtime.cxx_library(
            name = "dtype_headers_TEST_ONLY",
            srcs = [],
            exported_headers = {
                "selected_op_variants.h": ":add_model_lib_et_op_dtype_gen[selected_op_variants]",
            },
            visibility = [
                "//executorch/...",
                "@EXECUTORCH_CLIENTS",
            ],
        )

        runtime.cxx_library(
            name = "scalar_utils_TEST_ONLY",
            srcs = [],
            exported_preprocessor_flags = ["-DEXECUTORCH_SELECTIVE_BUILD_DTYPE"],
            exported_headers = ["scalar_utils.h", "selective_build.h"],
            exported_deps = [":dtype_headers_TEST_ONLY"],
            visibility = [
                "//executorch/kernels/...",
                "@EXECUTORCH_CLIENTS",
            ],
            deps = [
                "//executorch/runtime/core/exec_aten/util:scalar_type_util",
            ],
        )
