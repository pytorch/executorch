from typing import Optional, Sequence

import tensorrt as trt
from tensorrt import ITensor as TRTTensor
from torch.fx.node import Target
from torch_tensorrt.dynamo._SourceIR import SourceIR
from torch_tensorrt.dynamo.conversion._ConversionContext import ConversionContext
from torch_tensorrt.dynamo.conversion.converter_utils import (
    has_dynamic_shape,
    set_layer_name,
)
from torch_tensorrt.dynamo.conversion.impl.cat import (
    unify_and_concat_trt_tensors as unify_trt_shape_tensors,
)
from torch_tensorrt.dynamo.conversion.impl.shape import (
    get_shape_with_dynamic_shape,
)


def upsample(
    ctx: ConversionContext,
    target: Target,
    source_ir: Optional[SourceIR],
    name: str,
    input: TRTTensor,
    size: Optional[Sequence[int]],
    scale_factor: Optional[Sequence[float]],
    mode: str,
    align_corners: bool,
) -> TRTTensor:
    layer = ctx.net.add_resize(input)

    if scale_factor is not None:
        layer.scales = [1.0, 1.0] + list(scale_factor)
    else:
        shape = list(input.shape)[:2]
        if size is not None:
            shape += list(size)
        if has_dynamic_shape(shape):
            shape = get_shape_with_dynamic_shape(
                ctx, target, source_ir, name, shape, input
            )
            layer.set_input(1, shape)
        else:
            trt_shape = unify_trt_shape_tensors(
                ctx, target, name, shape, concat_axis=0, force_trt_output=False
            )
            if isinstance(trt_shape, list):
                layer.shape = trt_shape
            else:
                layer.set_input(1, trt_shape)

    if mode == "nearest":
        layer.resize_mode = trt.InterpolationMode.NEAREST
        layer.coordinate_transformation = trt.ResizeCoordinateTransformation.ASYMMETRIC
    elif mode in ("linear", "bilinear", "trilinear"):
        layer.resize_mode = trt.InterpolationMode.LINEAR
        layer.coordinate_transformation = (
            trt.ResizeCoordinateTransformation.ALIGN_CORNERS
            if align_corners
            else trt.ResizeCoordinateTransformation.HALF_PIXEL
        )
    elif mode == "bicubic":
        layer.resize_mode = trt.InterpolationMode.CUBIC
        layer.coordinate_transformation = (
            trt.ResizeCoordinateTransformation.ALIGN_CORNERS
            if align_corners
            else trt.ResizeCoordinateTransformation.HALF_PIXEL
        )

    set_layer_name(layer, target, name, source_ir)
    return layer.get_output(0)
