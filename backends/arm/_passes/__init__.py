# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
# Copyright 2025-2026 Arm Limited and/or its affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.


from . import arm_pass_utils  # noqa  # pyrefly: ignore [missing-import]
from .arm_pass import ArmOpTargetedPass, ArmPass  # noqa  # usort: skip  # pyrefly: ignore [missing-import]
from executorch.backends.transforms.fuse_identical_input_transforms_pass import (  # noqa
    NormalizeTransformInputPlaceholdersPass,
)

from .accumulate_index_put_pass import AccumulateIndexPutPass  # noqa  # pyrefly: ignore [missing-import]
from .broadcast_args_pass import BroadcastArgsPass  # noqa  # pyrefly: ignore [missing-import]
from .canonicalize_gather_pass import CanonicalizeGatherPass  # noqa  # pyrefly: ignore [missing-import]
from .canonicalize_view_copy_permute_pass import CanonicalizeViewCopyPermutePass  # noqa  # pyrefly: ignore [missing-import]
from .cast_int64_pass import CastInt64BuffersToInt32Pass  # noqa  # pyrefly: ignore [missing-import]
from .cast_int_comparison_inputs_pass import CastIntComparisonInputsPass  # noqa  # pyrefly: ignore [missing-import]
from .cast_to_int32_pass import CastToInt32Pass  # noqa  # pyrefly: ignore [missing-import]
from .constant_folding_pass import ConstantFoldingPass  # noqa  # pyrefly: ignore [missing-import]
from .convert_bool_sum_pass import ConvertBoolSumPass  # noqa  # pyrefly: ignore [missing-import]
from .convert_elu_params import ConvertELUParamsPass  # noqa  # pyrefly: ignore [missing-import]
from .convert_expand_copy_to_repeat import ConvertExpandCopyToRepeatPass  # noqa  # pyrefly: ignore [missing-import]
from .convert_full_like_to_full_pass import ConvertFullLikeToFullPass  # noqa  # pyrefly: ignore [missing-import]
from .convert_int64_const_ops_to_int32 import ConvertInt64ConstOpsToInt32Pass  # noqa  # pyrefly: ignore [missing-import]
from .convert_int64_output_ops_to_int32 import ConvertInt64OutputOpsToInt32Pass  # noqa  # pyrefly: ignore [missing-import]
from .convert_minmax_pass import ConvertMinMaxPass  # noqa  # pyrefly: ignore [missing-import]
from .convert_permute_singleton_to_view_pass import (  # noqa  # pyrefly: ignore [missing-import]
    ConvertPermuteSingletonToViewPass,
)
from .convert_split_to_slice import ConvertSplitToSlicePass  # noqa  # pyrefly: ignore [missing-import]
from .convert_squeezes_to_view import ConvertSqueezesToViewPass  # noqa  # pyrefly: ignore [missing-import]
from .convert_to_clamp_pass import ConvertToClampPass  # noqa  # pyrefly: ignore [missing-import]
from .decompose_acosh_pass import DecomposeAcoshPass  # noqa  # pyrefly: ignore [missing-import]
from .decompose_adaptive_avg_pool2d_pass import DecomposeAdaptiveAvgPool2dPass  # noqa  # pyrefly: ignore [missing-import]
from .decompose_adaptive_max_pool2d_pass import DecomposeAdaptiveMaxPool2dPass  # noqa  # pyrefly: ignore [missing-import]
from .decompose_add_sub_alpha_pass import DecomposeAddSubAlphaPass  # noqa  # pyrefly: ignore [missing-import]
from .decompose_addmm_pass import DecomposeAddmmPass  # noqa  # pyrefly: ignore [missing-import]
from .decompose_any_pass import DecomposeAnyPass  # noqa  # pyrefly: ignore [missing-import]
from .decompose_as_strided_copy_pass import DecomposeAsStridedCopyPass  # noqa  # pyrefly: ignore [missing-import]
from .decompose_asin_and_acos_pass import DecomposeAsinAndAcosPass  # noqa  # pyrefly: ignore [missing-import]
from .decompose_asinh_pass import DecomposeAsinhPass  # noqa  # pyrefly: ignore [missing-import]
from .decompose_atan_pass import DecomposeAtanPass  # noqa  # pyrefly: ignore [missing-import]
from .decompose_atanh_pass import DecomposeAtanhPass  # noqa  # pyrefly: ignore [missing-import]
from .decompose_avg_pool2d_pass import DecomposeAvgPool2dPass  # noqa  # pyrefly: ignore [missing-import]
from .decompose_batch_norm_no_stats import DecomposeBatchNormNoStatsPass  # noqa  # pyrefly: ignore [missing-import]
from .decompose_choose_qparams_symmetric_pass import (  # noqa  # pyrefly: ignore [missing-import]
    DecomposeChooseQParamsSymmetricPass,
)
from .decompose_cosh_pass import DecomposeCoshPass  # noqa  # pyrefly: ignore [missing-import]
from .decompose_cosine_similarity_pass import DecomposeCosineSimilarityPass  # noqa  # pyrefly: ignore [missing-import]
from .decompose_cumsum_pass import DecomposeCumsumPass  # noqa  # pyrefly: ignore [missing-import]
from .decompose_div_pass import DecomposeDivPass  # noqa  # pyrefly: ignore [missing-import]
from .decompose_div_tensor_mode import DecomposeDivTensorModePass  # noqa  # pyrefly: ignore [missing-import]
from .decompose_dynamic_adaptive_avg_pool2d_pass import (  # noqa  # pyrefly: ignore [missing-import]
    DecomposeDynamicAdaptiveAvgPool2dPass,
)
from .decompose_dynamic_full_pass import DecomposeDynamicFullPass  # noqa  # pyrefly: ignore [missing-import]
from .decompose_einsum_pass import DecomposeEinsumPass  # noqa  # pyrefly: ignore [missing-import]
from .decompose_elu_pass import ConvertEluFamilyToEluPass, DecomposeEluPass  # noqa  # pyrefly: ignore [missing-import]
from .decompose_embedding_pass import DecomposeEmbeddingPass  # noqa  # noqa  # pyrefly: ignore [missing-import]
from .decompose_erfinv_pass import DecomposeErfinvPass  # noqa  # pyrefly: ignore [missing-import]
from .decompose_expm1_pass import DecomposeExpm1Pass  # noqa  # pyrefly: ignore [missing-import]
from .decompose_flip_pass import DecomposeFlipPass  # noqa  # pyrefly: ignore [missing-import]
from .decompose_floor_divide_pass import DecomposeFloorDividePass  # noqa  # pyrefly: ignore [missing-import]
from .decompose_gelu_pass import DecomposeGeluPass  # noqa  # pyrefly: ignore [missing-import]
from .decompose_glu_pass import DecomposeGluPass  # noqa  # pyrefly: ignore [missing-import]
from .decompose_grouped_conv_pass import DecomposeGroupedConvPass  # noqa  # pyrefly: ignore [missing-import]
from .decompose_groupnorm_pass import DecomposeGroupNormPass  # noqa  # pyrefly: ignore [missing-import]
from .decompose_gru_pass import DecomposeGruPass  # noqa  # pyrefly: ignore [missing-import]
from .decompose_index_copy_pass import DecomposeIndexCopyPass  # noqa  # pyrefly: ignore [missing-import]
from .decompose_index_select_to_gather_pass import (  # noqa  # pyrefly: ignore [missing-import]
    DecomposeIndexSelectToGatherPass,
)
from .decompose_index_tensor_to_gather_pass import (  # noqa  # pyrefly: ignore [missing-import]
    DecomposeIndexTensorToGatherPass,
)
from .decompose_int_pow_pass import DecomposeIntPowPass  # noqa  # pyrefly: ignore [missing-import]
from .decompose_isinf_isnan_pass import DecomposeIsInfAndIsNanPass  # noqa  # pyrefly: ignore [missing-import]
from .decompose_large_stride_maxpool2d_pass import (  # noqa  # pyrefly: ignore [missing-import]
    DecomposeLargeStrideMaxPool2dForU55Pass,
)
from .decompose_layernorm_pass import DecomposeLayerNormPass  # noqa  # pyrefly: ignore [missing-import]
from .decompose_leaky_relu_pass import DecomposeLeakyReLUPass  # noqa  # pyrefly: ignore [missing-import]
from .decompose_linalg_vector_norm_pass import DecomposeLinalgVectorNormPass  # noqa  # pyrefly: ignore [missing-import]
from .decompose_linear_pass import DecomposeLinearPass  # noqa  # pyrefly: ignore [missing-import]
from .decompose_log1p_pass import DecomposeLog1pPass  # noqa  # pyrefly: ignore [missing-import]
from .decompose_logit_pass import DecomposeLogitPass  # noqa  # pyrefly: ignore [missing-import]
from .decompose_lstm_pass import DecomposeLstmPass  # noqa  # pyrefly: ignore [missing-import]
from .decompose_masked_fill_pass import DecomposeMaskedFillPass  # noqa  # pyrefly: ignore [missing-import]
from .decompose_matmul import DecomposeMatmulPass  # noqa  # pyrefly: ignore [missing-import]
from .decompose_max_pool1d_pass import DecomposeMaxPool1dPass  # noqa  # pyrefly: ignore [missing-import]
from .decompose_maxpool2d_with_dilation_pass import DecomposeMaxPool2dPass  # noqa  # pyrefly: ignore [missing-import]
from .decompose_meandim_pass import DecomposeMeanDimPass  # noqa  # pyrefly: ignore [missing-import]
from .decompose_ne_pass import DecomposeNotEqualPass  # noqa  # pyrefly: ignore [missing-import]
from .decompose_permute_for_u55_pass import DecomposePermuteForU55Pass  # noqa  # pyrefly: ignore [missing-import]
from .decompose_pow_tensor_tensor_pass import DecomposePowTensorTensorPass  # noqa  # pyrefly: ignore [missing-import]
from .decompose_prelu_pass import DecomposePReLUPass  # noqa  # pyrefly: ignore [missing-import]
from .decompose_prod_pass import DecomposeProdPass  # noqa  # pyrefly: ignore [missing-import]
from .decompose_quant_nodes import DecomposeQuantNodesPass  # noqa  # pyrefly: ignore [missing-import]
from .decompose_remainder_pass import DecomposeRemainderPass  # noqa  # pyrefly: ignore [missing-import]
from .decompose_rnn_pass import DecomposeRnnPass  # noqa  # pyrefly: ignore [missing-import]
from .decompose_roll_pass import DecomposeRollPass  # noqa  # pyrefly: ignore [missing-import]
from .decompose_round_pass import DecomposeRoundPass  # noqa  # pyrefly: ignore [missing-import]
from .decompose_sdpa_pass import DecomposeScaledDotProductAttentionPass  # noqa  # pyrefly: ignore [missing-import]
from .decompose_sdpa_with_regular_softmax_pass import (  # noqa  # pyrefly: ignore [missing-import]
    DecomposeSDPAWithRegularSoftmaxPass,
)
from .decompose_select import DecomposeSelectPass  # noqa  # pyrefly: ignore [missing-import]
from .decompose_select_scatter_pass import DecomposeSelectScatterPass  # noqa  # pyrefly: ignore [missing-import]
from .decompose_sign_pass import DecomposeSignPass  # noqa  # pyrefly: ignore [missing-import]
from .decompose_sinh_pass import DecomposeSinhPass  # noqa  # pyrefly: ignore [missing-import]
from .decompose_slice_scatter_pass import DecomposeSliceScatterPass  # noqa  # pyrefly: ignore [missing-import]
from .decompose_softmax_pass import DecomposeSoftmaxPass  # noqa  # pyrefly: ignore [missing-import]
from .decompose_sqrt_pass import DecomposeSqrtPass  # noqa  # pyrefly: ignore [missing-import]
from .decompose_strided_slice_copy_pass import DecomposeStridedSliceCopyPass  # noqa  # pyrefly: ignore [missing-import]
from .decompose_sum_pass import DecomposeSumPass  # noqa  # pyrefly: ignore [missing-import]
from .decompose_tan_pass import DecomposeTanPass  # noqa  # pyrefly: ignore [missing-import]
from .decompose_topk_pass import DecomposeTopKPass  # noqa  # pyrefly: ignore [missing-import]
from .decompose_tosa_unsupported_clamp_pass import (  # noqa  # pyrefly: ignore [missing-import]
    DecomposeTOSAUnsupportedClampPass,
)
from .decompose_tril_pass import DecomposeTrilPass  # noqa  # pyrefly: ignore [missing-import]
from .decompose_unfold_to_gather_pass import DecomposeUnfoldToGatherPass  # noqa  # pyrefly: ignore [missing-import]
from .decompose_unsupported_bilinear_resize_pass import (  # noqa  # pyrefly: ignore [missing-import]
    DecomposeUnsupportedBilinearResizePass,
)
from .decompose_var_pass import DecomposeVarPass  # noqa  # pyrefly: ignore [missing-import]
from .decompose_where_scalar_other_pass import DecomposeWhereScalarOtherPass  # noqa  # pyrefly: ignore [missing-import]
from .decorate_fp32_to_int32_casting_pass import DecorateFp32toInt32CastingPass  # noqa  # pyrefly: ignore [missing-import]
from .deduplicate_const_shapes_pass import DeduplicateConstShapesPass  # noqa  # pyrefly: ignore [missing-import]
from .deduplicate_get_attr_pass import DeduplicateGetAttrPass  # noqa  # pyrefly: ignore [missing-import]
from .detect_dynamic_w8a8_linear_pass import DetectDynamicW8A8LinearPass  # noqa  # pyrefly: ignore [missing-import]
from .ensure_unique_output_nodes_pass import EnsureUniqueOutputNodesPass  # noqa  # pyrefly: ignore [missing-import]
from .exir_to_tosa_pass import ExirToTosaPass  # noqa  # pyrefly: ignore [missing-import]
from .fold_dyt_affine_into_conv_pass import FoldDyTAffineIntoConvPass  # noqa  # pyrefly: ignore [missing-import]
from .fold_dyt_alpha_into_lut_pass import FoldDyTAlphaIntoLUTPass  # noqa  # pyrefly: ignore [missing-import]
from .fold_qdq_with_annotated_qparams_pass import (  # noqa  # pyrefly: ignore [missing-import]
    FoldAndAnnotateQParamsPass,
    QuantizeClampArgumentsPass,
)
from .fold_scalar_mul_into_conv_pass import FoldScalarMulIntoConvPass  # noqa  # pyrefly: ignore [missing-import]
from .fuse_batch_norm2d_pass import FuseBatchNorm2dPass  # noqa  # pyrefly: ignore [missing-import]
from .fuse_consecutive_clamps_pass import FuseConsecutiveClampsPass  # noqa  # pyrefly: ignore [missing-import]
from .fuse_consecutive_concat_shapes import FuseConsecutiveConcatShapesPass  # noqa  # pyrefly: ignore [missing-import]
from .fuse_consecutive_concats_pass import FuseConsecutiveConcatsPass  # noqa  # pyrefly: ignore [missing-import]
from .fuse_consecutive_rescales_pass import FuseConsecutiveRescalesPass  # noqa  # pyrefly: ignore [missing-import]
from .fuse_consecutive_slices_pass import FuseConsecutiveSlicesPass  # noqa  # pyrefly: ignore [missing-import]
from .fuse_constant_ops_pass import (  # noqa  # pyrefly: ignore [missing-import]
    ComputeConstantOpsAOTPass,
    FuseConstantArgsPass,
)
from .fuse_duplicate_users_pass import FuseDuplicateUsersPass  # noqa  # pyrefly: ignore [missing-import]
from .fuse_equal_placeholders_pass import FuseEqualPlaceholdersPass  # noqa  # pyrefly: ignore [missing-import]
from .fuse_identical_input_transforms_pass import (  # noqa  # pyrefly: ignore [missing-import]
    FuseIdenticalInputTransformsPass,
)
from .fuse_quantized_activation_pass import FuseQuantizedActivationPass  # noqa  # pyrefly: ignore [missing-import]
from .fuse_view_copy_transform_pass import FuseViewCopyTransformPass  # noqa  # pyrefly: ignore [missing-import]
from .insert_const_shapes import InsertConstShapesPass  # noqa  # pyrefly: ignore [missing-import]
from .insert_data_layout_casts_pass import InsertDataLayoutCastsPass  # noqa  # pyrefly: ignore [missing-import]
from .insert_dynamic_padding import InsertDynamicPaddingPass  # noqa  # pyrefly: ignore [missing-import]
from .insert_int32_casts_after_int64_placeholders import (  # noqa  # pyrefly: ignore [missing-import]
    InsertInt32CastsAfterInt64PlaceholdersPass,
)
from .insert_rescales_pass import (  # noqa  # pyrefly: ignore [missing-import]
    InsertControlFlowRescalesPass,
    InsertRescaleInt32Pass,
    InsertRescalePass,
)
from .insert_table_ops import InsertTableOpsPass  # noqa  # pyrefly: ignore [missing-import]
from .lower_dynamic_w8a8_linear_pass import LowerDynamicW8A8LinearPass  # noqa  # pyrefly: ignore [missing-import]
from .match_arg_dtype_pass import MatchArgDtypePass  # noqa  # pyrefly: ignore [missing-import]
from .match_arg_ranks_pass import MatchArgRanksPass  # noqa  # pyrefly: ignore [missing-import]
from .mm_to_bmm_pass import ConvertMmToBmmPass  # noqa  # pyrefly: ignore [missing-import]
from .move_data_movement_ops_to_smaller_dtype_pass import (  # noqa  # pyrefly: ignore [missing-import]
    MoveDataMovementOpsToSmallerDtypePass,
)
from .normalize_delegate_io_layout_pass import NormalizeDelegateIOLayoutPass  # noqa  # pyrefly: ignore [missing-import]
from .normalize_index_put_bool_index_tensor_pass import (  # noqa  # pyrefly: ignore [missing-import]
    NormalizeIndexPutBoolIndexTensorPass,
)
from .normalize_index_put_none_indices_pass import (  # noqa  # pyrefly: ignore [missing-import]
    NormalizeIndexPutNoneIndicesPass,
)
from .normalize_max_pool2d_input_rank_pass import (  # noqa  # pyrefly: ignore [missing-import]
    NormalizeMaxPool2dInputRankPass,
)
from .normalize_while_initial_args_pass import NormalizeWhileInitialArgsPass  # noqa  # pyrefly: ignore [missing-import]
from .prepare_gather_indices_pass import PrepareGatherIndicesPass  # noqa  # pyrefly: ignore [missing-import]
from .promote_bool_operands_pass import PromoteBoolOperandsPass  # noqa  # pyrefly: ignore [missing-import]
from .propagate_view_copy_permute_pass import (  # noqa  # pyrefly: ignore [missing-import]
    PropagateViewCopyPermuteDownPass,
    PropagateViewCopyPermuteUpPass,
)
from .remove_getitem_pass import RemoveGetItemPass  # noqa  # pyrefly: ignore [missing-import]
from .remove_graph_asserts_pass import RemoveGraphAssertsPass  # noqa  # pyrefly: ignore [missing-import]
from .remove_noop_pass import RemoveNoopPass  # noqa  # pyrefly: ignore [missing-import]
from .remove_permutes_around_elementwise_tosa_ops import (  # noqa  # pyrefly: ignore [missing-import]
    RemovePermutesAroundElementwiseTosaOps,
)
from .remove_redundant_type_as_pass import RemoveRedundantTypeAsPass  # noqa  # pyrefly: ignore [missing-import]
from .remove_safe_softmax_guard_pass import RemoveSafeSoftmaxGuardPass  # noqa  # pyrefly: ignore [missing-import]
from .replace_scalar_with_tensor_pass import (  # noqa  # pyrefly: ignore [missing-import]
    ReplaceScalarWithTensorByProfilePass,
)
from .resolve_view_copy_inferred_dim_pass import ResolveViewCopyInferredDimPass  # noqa  # pyrefly: ignore [missing-import]
from .rewrite_adaptive_avg_pool2d import RewriteAdaptiveAvgPool2dPass  # noqa  # pyrefly: ignore [missing-import]
from .rewrite_avg_pool2d_pass import RewriteAvgPool2dPass  # noqa  # pyrefly: ignore [missing-import]
from .rewrite_bool_bitwise_to_logical_pass import (  # noqa  # pyrefly: ignore [missing-import]
    RewriteBoolBitwiseToLogicalPass,
)
from .rewrite_bool_to_fp32_cast_via_int8_pass import (  # noqa  # pyrefly: ignore [missing-import]
    RewriteBoolToFp32CastViaInt8Pass,
)
from .rewrite_cat_slice_pass import RewriteCatSlicePass  # noqa  # pyrefly: ignore [missing-import]
from .rewrite_conv_pass import RewriteConvPass  # noqa  # pyrefly: ignore [missing-import]
from .rewrite_high_rank_singleton_permute_pass import (  # noqa  # pyrefly: ignore [missing-import]
    RewriteHighRankSingletonPermutePass,
)
from .rewrite_index_put_pass import RewriteIndexPutPass  # noqa  # pyrefly: ignore [missing-import]
from .rewrite_inplace_arithmetic_pass import RewriteInplaceArithmeticPass  # noqa  # pyrefly: ignore [missing-import]
from .rewrite_le_lt_to_ge_gt_pass import RewriteLeLtToGeGtPass  # noqa  # pyrefly: ignore [missing-import]
from .rewrite_matmul import RewriteMatmulPass  # noqa  # pyrefly: ignore [missing-import]
from .rewrite_max_pool2d_pass import RewriteMaxPool2dPass  # noqa  # pyrefly: ignore [missing-import]
from .rewrite_mxfp_conv2d import RewriteMXFPConv2dPass  # noqa  # pyrefly: ignore [missing-import]
from .rewrite_mxfp_linear import RewriteMXFPLinearPass  # noqa  # pyrefly: ignore [missing-import]
from .rewrite_pad import RewritePadPass  # noqa  # pyrefly: ignore [missing-import]
from .rewrite_slice import RewriteSlicePass  # noqa  # pyrefly: ignore [missing-import]
from .rewrite_upsample import RewriteUpsamplePass  # noqa  # pyrefly: ignore [missing-import]
from .scalars_to_attribute_pass import ScalarsToAttributePass  # noqa  # pyrefly: ignore [missing-import]
from .size_adjust_input_pass import SizeAdjustInputPass  # noqa  # pyrefly: ignore [missing-import]
from .symbolic_to_tosa_shape_pass import SymbolicToTosaShapesPass  # noqa  # pyrefly: ignore [missing-import]
from .unsqueeze_before_repeat_pass import UnsqueezeBeforeRepeatPass  # noqa  # pyrefly: ignore [missing-import]
from .unsqueeze_scalar_placeholders_pass import UnsqueezeScalarPlaceholdersPass  # noqa  # pyrefly: ignore [missing-import]
from .replace_inf_and_limit_values_pass import (  # noqa  # usort: skip  # pyrefly: ignore [missing-import]
    ReplaceInfAndLimitValuesPass,
)
from .control_flow_const_inline import (  # noqa  # usort: skip  # pyrefly: ignore [missing-import]
    ControlFlowConstInlinePass,
)

# Import all subpackages to allow extensions to patch classes
import importlib  # noqa: E402
import pkgutil  # noqa: E402

for _, _modname, _ispkg in pkgutil.iter_modules(__path__, __name__ + "."):
    if _ispkg:
        importlib.import_module(_modname)

from .arm_pass_manager import ArmPassManager  # noqa  # pyrefly: ignore [missing-import]
