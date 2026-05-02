from typing import Optional, Union

import numpy as np
import tensorrt as trt
import torch
from tensorrt import ITensor as TRTTensor
from torch.fx.experimental.proxy_tensor import unset_fake_temporarily
from torch.fx.node import Target
from torch_tensorrt._utils import is_tensorrt_version_supported
from torch_tensorrt.dynamo._SourceIR import SourceIR
from torch_tensorrt.dynamo.conversion._ConversionContext import ConversionContext
from torch_tensorrt.dynamo.conversion.converter_utils import (
    get_trt_tensor,
    set_layer_name,
)

if is_tensorrt_version_supported("10.8.0"):

    def quantize(
        ctx: ConversionContext,
        target: Target,
        source_ir: Optional[SourceIR],
        name: str,
        input_tensor: TRTTensor,
        block_size: int,
        amax: Union[np.ndarray, torch.Tensor],
        num_bits: int,
        exponent_bits: int,
        scale_num_bits: int,
        scale_exponent_bits: int,
    ) -> TRTTensor:
        """
        Adds quantize and dequantize ops (QDQ) which quantize to FP4 based
        on the output_type set and dequantizes them back.
        """
        if len(input_tensor.shape) not in (2, 3):
            raise ValueError(
                f"dynamic_block_quantize converter received an input of {input_tensor.shape} shape. Supported shapes: 2D or 3D"
            )
        with unset_fake_temporarily():
            axis = -1
            global_scale = _calculate_global_scale(ctx, name, amax)
            if ".weight_quantizer" in name:
                output = _static_double_quantize(
                    ctx,
                    target,
                    source_ir,
                    name,
                    input_tensor,
                    global_scale,
                    axis,
                )
            elif ".input_quantizer" in name:
                output = _dynamic_double_quantize(
                    ctx,
                    target,
                    source_ir,
                    name,
                    input_tensor,
                    global_scale,
                    axis,
                )
            else:
                raise ValueError(
                    f"quantizer received an input of {name}. Supported values: weight_quantizer | input_quantizer"
                )
            return output

    def _dynamic_double_quantize(
        ctx: ConversionContext,
        target: Target,
        source_ir: Optional[SourceIR],
        name: str,
        input_tensor: TRTTensor,
        global_scale: torch.Tensor,
        axis: int = -1,
        block_size: int = 16,
        output_type: trt.DataType = trt.DataType.FP4,
        scale_type: trt.DataType = trt.DataType.FP8,
    ) -> TRTTensor:
        """
        quantize input tensor to fp4
        Parameters:
            ctx: ConversionContext,
            target: Target,
            source_ir: Optional[SourceIR]
            name: str
            input_tensor : TRTTensor (On GPU)
                The input TRTTensor.
            global_scale : Tensor (On GPU)
                The global per-tensor scaling factor. It should contain only 1 element.
            axis : int
                The axis to quantize. Default is -1 (the last axis).
            block_size : int
                The block size for quantization. Default is 16.
            output_type : trt.DataType
                The data type for quantized data. Default is FP4.
            scale_type : trt.DataType
                The data type for block scale. Default is FP8.

        """
        global_scale = get_trt_tensor(ctx, global_scale, name + "_global_scale")

        if input_tensor.dtype not in [
            trt.DataType.HALF,
            trt.DataType.FLOAT,
            trt.DataType.BF16,
        ]:
            raise ValueError(
                f"Currently supported input tensor type is float16 | float32 | bfloat16, got Unsupported dtype: {input_tensor.dtype}"
            )
        # dynamic quantize input tensor to fp4
        dynamic_quantize_layer = ctx.net.add_dynamic_quantize(
            input_tensor,
            axis,
            block_size,
            output_type,
            scale_type,
        )
        dynamic_quantize_layer.set_input(1, global_scale)
        set_layer_name(
            dynamic_quantize_layer, target, name + "_dynamic_quantize", source_ir
        )
        quantized_data_in_fp4 = dynamic_quantize_layer.get_output(0)
        quantized_scale_in_fp8 = dynamic_quantize_layer.get_output(1)

        return _double_dequantize(
            ctx,
            target,
            source_ir,
            name,
            quantized_data_in_fp4,
            quantized_scale_in_fp8,
            global_scale,
            axis,
            input_tensor.dtype,
        )

    def _double_dequantize(
        ctx: ConversionContext,
        target: Target,
        source_ir: Optional[SourceIR],
        name: str,
        quantized_data_in_fp4: TRTTensor,
        quantized_scale_in_fp8: TRTTensor,
        global_scale: torch.Tensor,
        axis: int = -1,
        output_type: trt.DataType = trt.DataType.FLOAT,
    ) -> TRTTensor:
        """
        double dequantize will first dequantize scale from fp8 to orignal dtype(default is float32)
        and then dequantize data from fp4 to orignal dtype(default is float32)
        Parameters:
            ctx: ConversionContext,
            target: Target,
            source_ir: Optional[SourceIR]
            name: str
            quantized_data_in_fp4: TRTTensor
            quantized_scale_in_fp8: TRTTensor
            global_scale: torch.Tensor
            axis: int
            output_type: trt.DataType
        """
        # dequantize scale from fp8 to orignal dtype(default is float32)
        dequantize_scale_layer = ctx.net.add_dequantize(
            quantized_scale_in_fp8, global_scale, output_type
        )
        dequantize_scale_layer.axis = axis
        dequantize_scale_layer.to_type = output_type
        set_layer_name(
            dequantize_scale_layer, target, name + "_dequantize_scale", source_ir
        )
        dequantized_scale = dequantize_scale_layer.get_output(0)

        # dequantize quantized_data_in_fp4 from fp4 to orignal dtype(default is float32)
        dequantize_data_layer = ctx.net.add_dequantize(
            quantized_data_in_fp4, dequantized_scale, output_type
        )
        dequantize_data_layer.axis = axis
        dequantize_data_layer.to_type = output_type
        set_layer_name(
            dequantize_data_layer, target, name + "_dequantize_data", source_ir
        )
        dequantized_data = dequantize_data_layer.get_output(0)
        return dequantized_data

    def _static_double_quantize(
        ctx: ConversionContext,
        target: Target,
        source_ir: Optional[SourceIR],
        name: str,
        weights_tensor: torch.Tensor,
        global_scale: torch.Tensor,
        axis: int,
    ) -> TRTTensor:
        """
        Parameters:
            ctx: ConversionContext,
            target: Target,
            source_ir: Optional[SourceIR],
            name: str,
            weights_tensor : Tensor (On GPU)
                The input tensor for weights.
            global_scale : Tensor (On GPU)
                The global per-tensor scaling factor. It should contain only 1 element.
            axis: int
                The axis to quantize. Default is -1 (the last axis).
        Returns:
            quantized data tensor in fp4
        """

        import modelopt.torch.quantization.qtensor.nvfp4_tensor as nvfp4_tensor

        if weights_tensor.dtype == torch.float16:
            original_dtype = trt.DataType.HALF
        elif weights_tensor.dtype == torch.float32:
            original_dtype = trt.DataType.FLOAT
        elif weights_tensor.dtype == torch.bfloat16:
            original_dtype = trt.DataType.BF16
        else:
            raise ValueError(
                f"Currently supported weights tensor type is float16 | float32 | bfloat16, got Unsupported dtype: {weights_tensor.dtype}"
            )
        block_scale_fp8 = nvfp4_tensor.NVFP4QTensor.get_weights_scaling_factor(
            weights_tensor,
            16,
            global_scale,
        )[0]
        weights_tensor_fp4 = nvfp4_tensor.NVFP4QTensor.quantize(
            weights_tensor,
            16,
            block_scale_fp8,
            global_scale,
        )[0]._quantized_data

        block_scale_fp8 = get_trt_tensor(
            ctx,
            block_scale_fp8,
            name + "_block_scale_fp8",
            target_quantized_type=trt.DataType.FP8,
        )
        global_scale = get_trt_tensor(ctx, global_scale, name + "_global_scale")
        weights_tensor_fp4 = get_trt_tensor(
            ctx,
            weights_tensor_fp4,
            name + "_weights_fp4",
            target_quantized_type=trt.DataType.FP4,
        )

        dequantized_data = _double_dequantize(
            ctx,
            target,
            source_ir,
            name,
            weights_tensor_fp4,
            block_scale_fp8,
            global_scale,
            axis,
            original_dtype,
        )
        return dequantized_data

    def _calculate_global_scale(
        ctx: ConversionContext,
        name: str,
        amax: torch.Tensor,
    ) -> torch.Tensor:
        # calculate global scale (the global per-tensor scaling factor, should only contain 1 element)
        assert len(amax.shape) == 0, "amax should be a scalar"
        global_scale = amax / 6 / 448
        global_scale.masked_fill_(global_scale == 0, 1.0)
        return global_scale
