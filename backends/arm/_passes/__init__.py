# Copyright 2025 Arm Limited and/or its affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.


from . import arm_pass_utils  # noqa
from .arm_pass import ArmPass  # noqa  # usort: skip
from .annotate_decomposed_matmul import AnnotateDecomposedMatmulPass  # noqa
from .annotate_output_dim_order_pass import AnnotateOutputDimOrderPass  # noqa
from .broadcast_args_pass import BroadcastArgsPass  # noqa
from .cast_int64_pass import CastInt64BuffersToInt32Pass  # noqa
from .cast_to_int32_pass import CastToInt32Pass  # noqa
from .conv1d_unsqueeze_pass import Conv1dUnsqueezePass  # noqa
from .convert_elu_params import ConvertELUParamsPass  # noqa
from .convert_expand_copy_to_repeat import ConvertExpandCopyToRepeatPass  # noqa
from .convert_full_like_to_full_pass import ConvertFullLikeToFullPass  # noqa
from .convert_int64_const_ops_to_int32 import ConvertInt64ConstOpsToInt32Pass  # noqa
from .convert_int64_output_ops_to_int32 import ConvertInt64OutputOpsToInt32Pass  # noqa
from .convert_minmax_pass import ConvertMinMaxPass  # noqa
from .convert_permute_singleton_to_view_pass import (  # noqa
    ConvertPermuteSingletonToViewPass,
)
from .convert_split_to_slice import ConvertSplitToSlicePass  # noqa
from .convert_squeezes_to_view import ConvertSqueezesToViewPass  # noqa
from .convert_to_clamp_pass import ConvertToClampPass  # noqa
from .decompose_acosh_pass import DecomposeAcoshPass  # noqa
from .decompose_adaptive_avg_pool2d_pass import DecomposeAdaptiveAvgPool2dPass  # noqa
from .decompose_add_sub_alpha_pass import DecomposeAddSubAlphaPass  # noqa
from .decompose_addmm_pass import DecomposeAddmmPass  # noqa
from .decompose_any_pass import DecomposeAnyPass  # noqa
from .decompose_asin_and_acos_pass import DecomposeAsinAndAcosPass  # noqa
from .decompose_asinh_pass import DecomposeAsinhPass  # noqa
from .decompose_atan_pass import DecomposeAtanPass  # noqa
from .decompose_atanh_pass import DecomposeAtanhPass  # noqa
from .decompose_avg_pool2d_pass import DecomposeAvgPool2dPass  # noqa
from .decompose_batch_norm_no_stats import DecomposeBatchNormNoStatsPass  # noqa
from .decompose_cosh_pass import DecomposeCoshPass  # noqa
from .decompose_cosine_similarity_pass import DecomposeCosineSimilarityPass  # noqa
from .decompose_cumsum_pass import DecomposeCumsumPass  # noqa
from .decompose_div_pass import DecomposeDivPass  # noqa
from .decompose_div_tensor_mode import DecomposeDivTensorModePass  # noqa
from .decompose_elu_pass import DecomposeEluPass  # noqa
from .decompose_embedding_pass import DecomposeEmbeddingPass  # noqa  # noqa
from .decompose_expm1_pass import DecomposeExpm1Pass  # noqa
from .decompose_floor_divide_pass import DecomposeFloorDividePass  # noqa
from .decompose_gelu_pass import DecomposeGeluPass  # noqa
from .decompose_glu_pass import DecomposeGluPass  # noqa
from .decompose_grouped_conv_pass import DecomposeGroupedConvPass  # noqa
from .decompose_groupnorm_pass import DecomposeGroupNormPass  # noqa
from .decompose_int16_activation_conv_pass import (  # noqa
    DecomposeConvWithInt16ActivationPass,
)
from .decompose_int32_clamp_pass import DecomposeInt32ClampPass  # noqa
from .decompose_int_pow_pass import DecomposeIntPowPass  # noqa
from .decompose_layernorm_pass import DecomposeLayerNormPass  # noqa
from .decompose_leaky_relu_pass import DecomposeLeakyReLUPass  # noqa
from .decompose_linalg_vector_norm_pass import DecomposeLinalgVectorNormPass  # noqa
from .decompose_linear_pass import DecomposeLinearPass  # noqa
from .decompose_logit_pass import DecomposeLogitPass  # noqa
from .decompose_masked_fill_pass import DecomposeMaskedFillPass  # noqa
from .decompose_maxpool2d_with_dilation_pass import DecomposeMaxPool2dPass  # noqa
from .decompose_meandim_pass import DecomposeMeanDimPass  # noqa
from .decompose_ne_pass import DecomposeNotEqualPass  # noqa
from .decompose_quant_nodes import DecomposeQuantNodesPass  # noqa
from .decompose_remainder_pass import DecomposeRemainderPass  # noqa
from .decompose_round_pass import DecomposeRoundPass  # noqa
from .decompose_sdpa_pass import DecomposeScaledDotProductAttentionPass  # noqa
from .decompose_select import DecomposeSelectPass  # noqa
from .decompose_select_scatter_pass import DecomposeSelectScatterPass  # noqa
from .decompose_sign_pass import DecomposeSignPass  # noqa
from .decompose_silu_pass import DecomposeSiluPass  # noqa
from .decompose_sinh_pass import DecomposeSinhPass  # noqa
from .decompose_softmax_pass import DecomposeSoftmaxPass  # noqa
from .decompose_softmax_unstable_pass import DecomposeSoftmaxUnstablePass  # noqa
from .decompose_sqrt_pass import DecomposeSqrtPass  # noqa
from .decompose_sum_pass import DecomposeSumPass  # noqa
from .decompose_var_pass import DecomposeVarPass  # noqa
from .decorate_fp32_to_int32_casting_pass import DecorateFp32toInt32CastingPass  # noqa
from .fold_qdq_with_annotated_qparams_pass import (  # noqa
    FoldAndAnnotateQParamsPass,
    QuantizeClampArgumentsPass,
)
from .fuse_batch_norm2d_pass import FuseBatchNorm2dPass  # noqa
from .fuse_constant_ops_pass import (  # noqa
    ComputeConstantOpsAOTPass,
    FuseConstantArgsPass,
)
from .fuse_duplicate_users_pass import FuseDuplicateUsersPass  # noqa
from .fuse_equal_placeholders_pass import FuseEqualPlaceholdersPass  # noqa
from .fuse_quantized_activation_pass import FuseQuantizedActivationPass  # noqa
from .fuse_view_copy_transform_pass import FuseViewCopyTransformPass  # noqa
from .insert_int32_casts_after_int64_placeholders import (  # noqa
    InsertInt32CastsAfterInt64PlaceholdersPass,
)
from .insert_rescales_pass import (  # noqa
    InsertControlFlowRescalesPass,
    InsertRescaleInt32Pass,
    InsertRescalePass,
)
from .insert_table_ops import InsertTableOpsPass  # noqa
from .match_arg_dtype_pass import MatchArgDtypePass  # noqa
from .match_arg_ranks_pass import MatchArgRanksPass  # noqa
from .mm_to_bmm_pass import ConvertMmToBmmPass  # noqa
from .normalize_while_initial_args_pass import NormalizeWhileInitialArgsPass  # noqa
from .promote_bool_operands_pass import PromoteBoolOperandsPass  # noqa
from .remove_getitem_pass import RemoveGetItemPass  # noqa
from .remove_graph_asserts_pass import RemoveGraphAssertsPass  # noqa
from .remove_noop_pass import RemoveNoopPass  # noqa
from .replace_scalar_with_tensor_pass import (  # noqa
    ReplaceScalarWithTensorByProfilePass,
)
from .rewrite_conv_pass import RewriteConvPass  # noqa
from .rewrite_matmul import RewriteMatmulPass  # noqa
from .rewrite_upsample import RewriteUpsamplePass  # noqa
from .scalars_to_attribute_pass import ScalarsToAttributePass  # noqa
from .size_adjust_input_pass import SizeAdjustInputPass  # noqa
from .to_tosa_memory_format_pass import ToTosaMemoryFormatPass  # noqa
from .unsqueeze_before_repeat_pass import UnsqueezeBeforeRepeatPass  # noqa
from .unsqueeze_scalar_placeholders_pass import UnsqueezeScalarPlaceholdersPass  # noqa
from .replace_inf_and_limit_values_pass import (  # noqa  # usort: skip
    ReplaceInfAndLimitValuesPass,
)
from .arm_pass_manager import ArmPassManager  # noqa  # usort: skip
