import logging
from typing import Optional, Union

import numpy as np
import tensorrt as trt
import torch
from tensorrt import DataType as TRTDataType
from tensorrt import ITensor as TRTTensor
from torch.fx.node import Target
from torch_tensorrt import _enums
from torch_tensorrt.dynamo._SourceIR import SourceIR
from torch_tensorrt.dynamo.conversion._ConversionContext import ConversionContext
from torch_tensorrt.dynamo.conversion._ConverterRegistry import ConverterRegistry
from torch_tensorrt.dynamo.conversion.converter_utils import (
    cast_trt_tensor,
    get_trt_tensor,
)

LOGGER: logging.Logger = logging.getLogger(__name__)


def to_copy(
    ctx: ConversionContext,
    target: Target,
    source_ir: Optional[SourceIR],
    name: str,
    input: Union[TRTTensor, torch.Tensor, np.ndarray],
    dtype: Union[TRTDataType, torch.dtype, np.dtype, _enums.dtype],
    force_layer: bool = False,
) -> TRTTensor:
    if not isinstance(input, TRTTensor):
        input = get_trt_tensor(ctx, input, f"{name}_copy_tensor")

    # If cast is forced, insert identity layer regardless of whether the dtype
    # doesn't change
    if force_layer:
        trt_dtype = _enums.dtype._from(dtype).to(trt.DataType)
        source_ir = source_ir if source_ir is not None else SourceIR.UNKNOWN
        target_str = ConverterRegistry.qualified_name_or_str(target)
        target_name = f"{source_ir}_ops{('.' + target_str) if target_str else ''}"

        cast_layer = ctx.net.add_cast(input, trt_dtype)
        cast_layer.name = f"Forced Cast ITensor {input.name} from {input.dtype} to {trt_dtype} - [{target_name}]-[{name}]"
        return cast_layer.get_output(0)
    else:
        casted_tensor = cast_trt_tensor(ctx, input, dtype, name, target, source_ir)
        return casted_tensor
