# Copyright 2025 Arm Limited and/or its affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.


from . import arm_pass_utils  # noqa
from .annotate_channels_last_dim_order_pass import AnnotateChannelsLastDimOrder  # noqa
from .annotate_decomposed_matmul import AnnotateDecomposedMatmulPass  # noqa
from .cast_int64_pass import CastInt64ToInt32Pass  # noqa
from .conv1d_unsqueeze_pass import Conv1dUnsqueezePass  # noqa
from .convert_expand_copy_to_repeat import ConvertExpandCopyToRepeatPass  # noqa
from .convert_full_like_to_full_pass import ConvertFullLikeToFullPass  # noqa
from .convert_minmax_pass import ConvertMinMaxPass  # noqa
from .convert_split_to_slice import ConvertSplitToSlicePass  # noqa
from .convert_squeezes_to_view import ConvertSqueezesToViewPass  # noqa
from .convert_to_clamp import ConvertToClampPass  # noqa
from .decompose_batchnorm_pass import DecomposeBatchNormPass  # noqa
from .decompose_div_pass import DecomposeDivPass  # noqa
from .decompose_layernorm_pass import DecomposeLayerNormPass  # noqa
from .decompose_linear_pass import DecomposeLinearPass  # noqa
from .decompose_meandim_pass import DecomposeMeanDimPass  # noqa
from .decompose_select import DecomposeSelectPass  # noqa
from .decompose_softmaxes_pass import DecomposeSoftmaxesPass  # noqa
from .decompose_var_pass import DecomposeVarPass  # noqa
from .fold_qdq_with_annotated_qparams_pass import (  # noqa
    FoldAndAnnotateQParamsPass,
    get_input_qparams,
    get_output_qparams,
    QuantizeOperatorArguments,
    RetraceFoldedDtypesPass,
)
from .fuse_batchnorm2d_pass import FuseBatchnorm2DPass  # noqa
from .fuse_constant_ops_pass import FuseConstantOpsPass  # noqa
from .fuse_quantized_activation_pass import FuseQuantizedActivationPass  # noqa
from .insert_rescales_pass import InsertRescalePass  # noqa
from .insert_table_ops import InsertTableOpsPass  # noqa
from .keep_dims_false_to_squeeze_pass import KeepDimsFalseToSqueezePass  # noqa
from .match_arg_ranks_pass import MatchArgRanksPass  # noqa
from .meandim_to_averagepool_pass import ConvertMeanDimToAveragePoolPass  # noqa
from .mm_to_bmm_pass import ConvertMmToBmmPass  # noqa
from .remove_clone_pass import RemoveClonePass  # noqa
from .scalars_to_attribute_pass import ScalarsToAttributePass  # noqa
from .size_adjust_conv2d_pass import SizeAdjustConv2DPass  # noqa
from .unsqueeze_before_repeat_pass import UnsqueezeBeforeRepeatPass  # noqa
from .unsqueeze_scalar_placeholders_pass import UnsqueezeScalarPlaceholdersPass  # noqa
from .arm_pass_manager import ArmPassManager  # noqa  # usort: skip
