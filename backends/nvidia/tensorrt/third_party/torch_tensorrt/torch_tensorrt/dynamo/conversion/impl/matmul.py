from typing import Optional

import tensorrt as trt
import torch
from tensorrt import ITensor as TRTTensor
from torch.fx.node import Target
from torch_tensorrt import _enums
from torch_tensorrt.dynamo._SourceIR import SourceIR
from torch_tensorrt.dynamo.conversion._ConversionContext import ConversionContext
from torch_tensorrt.dynamo.conversion.converter_utils import (
    broadcast,
    cast_trt_tensor,
    get_trt_tensor,
    set_layer_name,
)


def matrix_multiply(
    ctx: ConversionContext,
    target: Target,
    source_ir: Optional[SourceIR],
    name: str,
    input: TRTTensor,
    other: TRTTensor,
    input_matrix_op: trt.MatrixOperation = trt.MatrixOperation.NONE,
    other_matrix_op: trt.MatrixOperation = trt.MatrixOperation.NONE,
) -> TRTTensor:
    if not isinstance(input, trt.ITensor):
        input = get_trt_tensor(ctx, input, f"{name}_input")
    if not isinstance(other, trt.ITensor):
        other = get_trt_tensor(
            ctx,
            other,
            f"{name}_other",
            dtype=_enums.dtype._from(input.dtype).to(torch.dtype),
        )

    preset_diff = 0

    if len(input.shape) == 1:
        preset_diff -= 1
        input_matrix_op = trt.MatrixOperation.VECTOR

    if len(other.shape) == 1:
        preset_diff += 1
        other_matrix_op = trt.MatrixOperation.VECTOR

    input, other = broadcast(
        ctx, input, other, f"{name}_input", f"{name}_other", preset_diff
    )
    # Get the original input dtype
    input_dtype = _enums.dtype._from(input.dtype).to(torch.dtype)

    if (
        ctx.net.get_flag(trt.NetworkDefinitionCreationFlag.STRONGLY_TYPED)
        and ctx.compilation_settings.use_fp32_acc
        and input_dtype == torch.float16
    ):
        input = cast_trt_tensor(ctx, input, torch.float32, f"{name}_input_casted")
        other = cast_trt_tensor(ctx, other, torch.float32, f"{name}_other_casted")

    matmul_layer = ctx.net.add_matrix_multiply(
        input, input_matrix_op, other, other_matrix_op
    )
    matmul_output = matmul_layer.get_output(0)

    if (
        ctx.net.get_flag(trt.NetworkDefinitionCreationFlag.STRONGLY_TYPED)
        and ctx.compilation_settings.use_fp32_acc
        and input_dtype == torch.float16
    ):
        matmul_output = cast_trt_tensor(
            ctx, matmul_output, input_dtype, f"{name}_output_casted"
        )

    set_layer_name(matmul_layer, target, name, source_ir)
    return matmul_output
