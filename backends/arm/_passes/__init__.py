# Copyright 2025 Arm Limited and/or its affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.


from . import arm_pass_utils  # noqa
from .annotate_channels_last_dim_order_pass import AnnotateChannelsLastDimOrder  # noqa
from .annotate_decomposed_matmul import AnnotateDecomposedMatmulPass  # noqa
from .arm_pass import ArmPass  # noqa
from .broadcast_args_pass import BroadcastArgsPass  # noqa
from .cast_int64_pass import CastInt64BuffersToInt32Pass  # noqa
from .cast_to_int32_pass import CastToInt32Pass  # noqa
from .conv1d_unsqueeze_pass import Conv1dUnsqueezePass  # noqa
from .convert_any_default_dim_dims_pass import ConvertAnyDefaultDimDimsPass  # noqa
from .convert_expand_copy_to_repeat import ConvertExpandCopyToRepeatPass  # noqa
from .convert_full_like_to_full_pass import ConvertFullLikeToFullPass  # noqa
from .convert_int_pow_to_mul import ConvertIntPowToMuls  # noqa
from .convert_minmax_pass import ConvertMinMaxPass  # noqa
from .convert_split_to_slice import ConvertSplitToSlicePass  # noqa
from .convert_squeezes_to_view import ConvertSqueezesToViewPass  # noqa
from .convert_to_clamp import ConvertToClampPass  # noqa
from .decompose_cosine_similarity_pass import DecomposeCosineSimilarityPass  # noqa
from .decompose_div_pass import DecomposeDivPass  # noqa
from .decompose_gelu_pass import DecomposeGeluPass  # noqa
from .decompose_groupnorm_pass import DecomposeGroupNormPass  # noqa
from .decompose_layernorm_pass import DecomposeLayerNormPass  # noqa
from .decompose_leaky_relu_pass import DecomposeLeakyReLUPass  # noqa
from .decompose_linalg_vector_norm_pass import DecomposeLinearVectorNormPass  # noqa
from .decompose_linear_pass import DecomposeLinearPass  # noqa
from .decompose_meandim_pass import DecomposeMeanDimPass  # noqa
from .decompose_ne_pass import DecomposeNotEqualPass  # noqa
from .decompose_select import DecomposeSelectPass  # noqa
from .decompose_silu_pass import DecomposeSiluPass  # noqa
from .decompose_softmax_pass import DecomposeSoftmaxPass  # noqa
from .decompose_softmax_unstable_pass import DecomposeSoftmaxUnstablePass  # noqa
from .decompose_sqrt_pass import DecomposeSqrtPass  # noqa
from .decompose_sum_pass import DecomposeSumPass  # noqa
from .decompose_var_pass import DecomposeVarPass  # noqa
from .fold_qdq_with_annotated_qparams_pass import (  # noqa
    FoldAndAnnotateQParamsPass,
    QuantizeOperatorArguments,
    RetraceFoldedDtypesPass,
)
from .fuse_batchnorm2d_pass import FuseBatchnorm2DPass  # noqa
from .fuse_constant_ops_pass import ComputeConstantOpsAOT, FuseConstantArgsPass  # noqa
from .fuse_equal_placeholders_pass import FuseEqualPlaceholdersPass  # noqa
from .fuse_quantized_activation_pass import FuseQuantizedActivationPass  # noqa
from .insert_rescales_pass import InsertRescalePass  # noqa
from .insert_table_ops import InsertTableOpsPass  # noqa
from .match_arg_ranks_pass import MatchArgRanksPass  # noqa
from .match_where_self_arg_dtype_pass import MatchWhereSelfDtypePass  # noqa
from .mm_to_bmm_pass import ConvertMmToBmmPass  # noqa
from .remove_clone_pass import RemoveClonePass  # noqa
from .replace_scalar_with_tensor_pass import (  # noqa
    ReplaceScalarWithTensorArgPassTOSABI,
    ReplaceScalarWithTensorArgPassTOSAMI,
)
from .scalars_to_attribute_pass import ScalarsToAttributePass  # noqa
from .size_adjust_conv2d_pass import SizeAdjustConv2DPass  # noqa
from .unsqueeze_before_repeat_pass import UnsqueezeBeforeRepeatPass  # noqa
from .unsqueeze_scalar_placeholders_pass import UnsqueezeScalarPlaceholdersPass  # noqa
from .replace_inf_values_pass import ReplaceInfValues  # noqa  # usort: skip
from .arm_pass_manager import ArmPassManager  # noqa  # usort: skip
