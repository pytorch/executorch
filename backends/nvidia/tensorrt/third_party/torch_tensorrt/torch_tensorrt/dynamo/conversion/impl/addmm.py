from typing import Optional, Union

import numpy as np
import torch
from tensorrt import ITensor as TRTTensor
from torch.fx.node import Target
from torch_tensorrt.dynamo._SourceIR import SourceIR
from torch_tensorrt.dynamo.conversion import impl
from torch_tensorrt.dynamo.conversion._ConversionContext import ConversionContext


def addmm(
    ctx: ConversionContext,
    target: Target,
    source_ir: Optional[SourceIR],
    name: str,
    input: TRTTensor,
    mat1: Union[TRTTensor, torch.Tensor, np.ndarray],
    mat2: Union[TRTTensor, torch.Tensor, np.ndarray],
    *,
    beta: Union[float, int],
    alpha: Union[float, int],
) -> TRTTensor:
    mm = impl.matmul.matrix_multiply(ctx, target, source_ir, f"{name}_mm", mat1, mat2)
    if alpha != 1:
        mm = impl.elementwise.mul(
            ctx, target, SourceIR.ATEN, f"{name}_mul_alpha", mm, alpha
        )
    if beta != 1:
        input = impl.elementwise.mul(
            ctx, target, SourceIR.ATEN, f"{name}_mul_beta", input, beta
        )

    return impl.elementwise.add(ctx, target, source_ir, f"{name}_add", input, mm)
