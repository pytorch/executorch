# Copyright (c) Qualcomm Innovation Center, Inc.
# All rights reserved
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from .annotate_adaptive_avg_pool1d import AnnotateAdaptiveAvgPool1D
from .annotate_quant_attrs import AnnotateQuantAttrs
from .annotate_stack import AnnotateStack
from .annotate_unbind import AnnotateUnbind
from .convert_bmm_to_matmul import ConvertBmmToMatmul
from .convert_conv1d_to_conv2d import ConvertConv1dToConv2d
from .convert_linear_to_conv2d import ConvertLinearToConv2d
from .convert_square_to_pow import ConvertSquareToPow
from .decompose_any import DecomposeAny
from .decompose_cdist import DecomposeCDist
from .decompose_col_im import DecomposeColIm
from .decompose_einsum import DecomposeEinsum
from .decompose_expm1 import DecomposeExpM1
from .decompose_linalg_vector_norm import DecomposeLinalgVectorNorm
from .decompose_minmaxdim import DecomposeMinMaxDim
from .decompose_roll import DecomposeRoll
from .decompose_silu import DecomposeSilu
from .decompose_wrap_with_autocast import DecomposeWrapWithAutocast
from .expand_broadcast_tensor_shape import ExpandBroadcastTensorShape
from .fixed_linear_keep_dim import FixedLinearKeepDim
from .fold_qdq import FoldQDQ
from .fuse_consecutive_cast import FuseConsecutiveCast
from .fuse_consecutive_transpose import FuseConsecutiveTranspose
from .i64_to_i32 import I64toI32
from .insert_io_qdq import InsertIOQDQ
from .insert_requantize import InsertRequantize
from .layout_transform import LayoutTransform
from .lift_constant_scalar_operands import LiftConstantScalarOperands
from .recompose_pixel_unshuffle import RecomposePixelUnshuffle
from .recompose_rms_norm import RecomposeRmsNorm
from .reduce_dynamic_range import ReduceDynamicRange
from .remove_0d_tensor import Remove0DTensor
from .remove_redundancy import RemoveRedundancy
from .replace_arange_args import ReplaceArangeArgs
from .replace_inf_values import ReplaceInfValues
from .tag_quant_io import TagQuantIO


__all__ = [
    AnnotateAdaptiveAvgPool1D,
    AnnotateQuantAttrs,
    AnnotateStack,
    AnnotateUnbind,
    ConvertBmmToMatmul,
    ConvertConv1dToConv2d,
    ConvertLinearToConv2d,
    ConvertSquareToPow,
    DecomposeAny,
    DecomposeCDist,
    DecomposeColIm,
    DecomposeEinsum,
    DecomposeExpM1,
    DecomposeLinalgVectorNorm,
    DecomposeMinMaxDim,
    DecomposeRoll,
    DecomposeSilu,
    DecomposeWrapWithAutocast,
    ExpandBroadcastTensorShape,
    FixedLinearKeepDim,
    FoldQDQ,
    FuseConsecutiveCast,
    FuseConsecutiveTranspose,
    I64toI32,
    InsertIOQDQ,
    InsertRequantize,
    LayoutTransform,
    LiftConstantScalarOperands,
    RecomposePixelUnshuffle,
    RecomposeRmsNorm,
    ReduceDynamicRange,
    Remove0DTensor,
    RemoveRedundancy,
    ReplaceArangeArgs,
    ReplaceInfValues,
    TagQuantIO,
]
