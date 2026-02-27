from typing import Optional, Union

import numpy as np
import tensorrt as trt
import torch
from tensorrt import ITensor as TRTTensor
from torch.fx.node import Target
from torch_tensorrt.dynamo._SourceIR import SourceIR
from torch_tensorrt.dynamo.conversion import impl
from torch_tensorrt.dynamo.conversion._ConversionContext import ConversionContext
from torch_tensorrt.dynamo.conversion.converter_utils import get_trt_tensor


def linear(
    ctx: ConversionContext,
    target: Union[Target, str],
    source_ir: Optional[SourceIR],
    name: str,
    input: TRTTensor,
    weight: Union[TRTTensor, torch.Tensor, np.ndarray],
    bias: Optional[Union[TRTTensor, torch.Tensor, np.ndarray]],
) -> TRTTensor:
    # Process weight terms
    if not isinstance(weight, (TRTTensor, torch.Tensor, np.ndarray)):
        raise RuntimeError(
            f"Linear layer {name} has weight of type {type(weight)}, Expect Union[TRTTensor, torch.Tensor, np.ndarray],"
        )
    elif isinstance(weight, (torch.Tensor, np.ndarray)):
        weight = get_trt_tensor(ctx, weight, f"{name}_weight")

    # Process bias terms
    if bias is not None and not isinstance(bias, (TRTTensor, torch.Tensor, np.ndarray)):
        raise RuntimeError(
            f"Linear layer {name} has bias of type {type(bias)}, Expect Union[TRTTensor, torch.Tensor, np.ndarray],"
        )
    elif isinstance(bias, (torch.Tensor, np.ndarray)):
        bias = get_trt_tensor(ctx, bias, f"{name}_bias")

    # add IMatrixMultiplyLayer
    out = impl.matmul.matrix_multiply(
        ctx,
        target,
        source_ir,
        f"{name}_matrix_multiply",
        input,
        weight,
        input_matrix_op=trt.MatrixOperation.NONE,
        other_matrix_op=trt.MatrixOperation.TRANSPOSE,
    )

    if bias is not None:
        # add bias
        out = impl.elementwise.add(
            ctx, target, source_ir, f"{name}_add_bias", out, bias
        )

    return out
