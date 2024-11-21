from executorch.backends.transforms.addmm_mm_to_linear import AddmmToLinearTransform
from executorch.exir.passes import PassManager

from .annotate_and_quant_scalar import AnnotateAndQuantScalar
from .annotate_quant_attrs import AnnotateQuantAttrs
from .convert_addmm_back_to_linear import ConvertAddmmmmWithLinear
from .convert_hardsigmoid import ConvertHardsigmoid
from .convert_hardswish import ConvertHardswish
from .convert_interpolate_with_upsample2d import ConvertInterpolateWithUpsample2D
from .fold_qdq import FoldQDQ
from .i64_to_i32 import I64toI32
from .insert_io_qdq import InsertIOQDQ
from .layout_transform import LayoutTransform
from .remove_clone import RemoveClone

qnn_compiler_passes = PassManager(
    passes=[
        AddmmToLinearTransform(),
        ConvertInterpolateWithUpsample2D(),
        AddmmToLinearTransform(),
        ConvertHardsigmoid(),
        ConvertHardswish(),
        ConvertInterpolateWithUpsample2D(),
        RemoveClone(),
    ]
)

qnn_compiler_passes = PassManager(
    passes=[
        AddmmToLinearTransform(),
    ]
)

__all__ = [
    qnn_compiler_passes,
    AddmmToLinearTransform,
    AnnotateAndQuantScalar,
    AnnotateQuantAttrs,
    ConvertHardsigmoid,
    ConvertHardswish,
    ConvertAddmmmmWithLinear,
    ConvertInterpolateWithUpsample2D,
    FoldQDQ,
    I64toI32,
    InsertIOQDQ,
    LayoutTransform,
    RemoveClone,
]
