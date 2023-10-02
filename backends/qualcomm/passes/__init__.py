# Copyright (c) Qualcomm Innovation Center, Inc.
# All rights reserved
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from executorch.backends.transforms.addmm_mm_to_linear import AddmmToLinearTransform
from executorch.exir.passes import PassManager

from .annotate_and_quant_scalar import AnnotateAndQuantScalar
from .annotate_quant_attrs import AnnotateQuantAttrs
from .convert_hardsigmoid import ConvertHardsigmoid
from .convert_hardswish import ConvertHardswish
from .convert_interpolate_with_upsample2d import ConvertInterpolateWithUpsample2D
from .fold_qdq import FoldQDQ
from .i64_to_i32 import I64toI32
from .insert_io_qdq import InsertIOQDQ
from .layout_transform import LayoutTransform
from .recompose_pixel_shuffle import RecomposePixelShuffle
from .remove_clone import RemoveClone

qnn_partitioner_passes = PassManager(
    passes=[
        AddmmToLinearTransform(),
        ConvertHardsigmoid(),
        ConvertHardswish(),
        ConvertInterpolateWithUpsample2D(),
        I64toI32(),
        RemoveClone(),
        LayoutTransform(),
        AnnotateQuantAttrs(),
        AnnotateAndQuantScalar(),
        FoldQDQ(),
    ]
)

qnn_compiler_passes = PassManager(
    passes=[
        AddmmToLinearTransform(),
        InsertIOQDQ(),
        LayoutTransform(insert_permute=True),
    ]
)

__all__ = [
    qnn_partitioner_passes,
    qnn_compiler_passes,
    AddmmToLinearTransform,
    AnnotateAndQuantScalar,
    AnnotateQuantAttrs,
    ConvertHardsigmoid,
    ConvertHardswish,
    ConvertInterpolateWithUpsample2D,
    FoldQDQ,
    I64toI32,
    InsertIOQDQ,
    LayoutTransform,
    RecomposePixelShuffle,
    RemoveClone,
]
