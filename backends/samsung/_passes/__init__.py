# Copyright (c) Samsung Electronics Co. LTD
# All rights reserved
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from .annotate_qparams import AnnotateQparamsPass
from .annotate_scalar_parameters import AnnotateScalarParametersPass
from .compose_rms_norm import RecomposeRmsNorm
from .compute_const_attrs import ComputeConstAttrs
from .conv1d_to_conv2d import Conv1dToConv2d
from .customized_constant_prop import ConstantPropPass
from .decompose_einsum import DecomposeEinsum
from .decompose_glu import DecomposeGlu
from .decompose_linalg_vector_norm import DecomposeLinalgVectorNorm
from .decompose_roll import DecomposeRoll
from .fold_qdq import FoldQDQPass
from .fuse_activation import FuseActivationPass
from .insert_qdq import InsertQDQPass
from .remove_useless_ops import RemoveUselessOpPass
from .replace_inf_values import ReplaceInfValues
from .replace_scalar_ops import ReplaceOpsWithScalar

__all__ = [
    "AnnotateQparamsPass",
    "AnnotateScalarParametersPass",
    "RecomposeRmsNorm",
    "ComputeConstAttrs",
    "Conv1dToConv2d",
    "ConstantPropPass",
    "DecomposeEinsum",
    "DecomposeGlu",
    "DecomposeLinalgVectorNorm",
    "DecomposeRoll",
    "FoldQDQPass",
    "FuseActivationPass",
    "InsertQDQPass",
    "RemoveUselessOpPass",
    "ReplaceInfValues",
    "ReplaceOpsWithScalar",
]
