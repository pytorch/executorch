# Copyright (c) Qualcomm Innovation Center, Inc.
# All rights reserved
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from executorch.exir.passes import PassManager

from .annotate_and_quant_scalar import AnnotateAndQuantScalar
from .annotate_quant_attrs import AnnotateQuantAttrs
from .convert_addmm_back_to_linear import ConvertAddmmmmWithLinear
from .convert_bmm_to_matmul import ConvertBmmToMatmul
from .convert_hardsigmoid import ConvertHardsigmoid
from .convert_hardswish import ConvertHardswish
from .convert_interpolate_with_upsample2d import ConvertInterpolateWithUpsample2D
from .fold_qdq import FoldQDQ
from .i64_to_i32 import I64toI32
from .insert_io_qdq import InsertIOQDQ
from .layout_transform import LayoutTransform
from .recompose_pixel_shuffle import RecomposePixelShuffle
from .reduce_dynamic_range import ReduceDynamicRange
from .remove_clone import RemoveClone

qnn_partitioner_passes = PassManager(
    passes=[
        RemoveClone(),
        ConvertAddmmmmWithLinear(),
        ConvertHardsigmoid(),
        ConvertHardswish(),
        ConvertBmmToMatmul(),
        ConvertInterpolateWithUpsample2D(),
        I64toI32(),
        LayoutTransform(),
        AnnotateQuantAttrs(),
        AnnotateAndQuantScalar(),
        FoldQDQ(),
    ]
)

qnn_compiler_passes = PassManager(
    passes=[
        ConvertAddmmmmWithLinear(),
        InsertIOQDQ(),
        LayoutTransform(insert_permute=True),
    ]
)

__all__ = [
    qnn_partitioner_passes,
    qnn_compiler_passes,
    AnnotateAndQuantScalar,
    AnnotateQuantAttrs,
    ConvertAddmmmmWithLinear,
    ConvertBmmToMatmul,
    ConvertHardsigmoid,
    ConvertHardswish,
    ConvertInterpolateWithUpsample2D,
    FoldQDQ,
    I64toI32,
    InsertIOQDQ,
    LayoutTransform,
    RecomposePixelShuffle,
    ReduceDynamicRange,
    RemoveClone,
]
