# Copyright 2023-2025 NXP
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import copy
from typing import List

import numpy as np

from executorch.backends.nxp.backend.ir import logger as logger
from executorch.backends.nxp.backend.ir.tflite_generator import (
    tflite_model as tflite_model,
)


def _validate_or_set_quant_params(
    tensor: tflite_model.Tensor, quant: tflite_model.Quantization
) -> bool:
    """
    Set quantization parameters 'quant' in the tensor. If tensor already has any quantization parameters,
    checks if equals to quant
    :param tensor: tensor where to set the quantization parameters
    :param quant: Quantization parameters
    :return: False if validation failed, True otherwise
    """

    if tensor.quantization is not None:
        return tensor.quantization == quant
    tensor.quantization = copy.copy(quant)

    return True


def propagate_quantization(
    from_tensor: tflite_model.Tensor, to_tensor: tflite_model.Tensor
):
    """
    Propagates quantization parameters from from_tensor to to_tensor. If to_tensor already has the params set
    checks the consistency.
    :raises: logger.Error - INVALID_ONNX_MODEL
    """

    if (
        from_tensor.quantization is not None
        and from_tensor.quantization.is_per_channel()
    ):
        # Note: For simplicity the quantization propagation is allowed only for per tensor quantized tensors.
        # Typically, operator inputs and outputs are per-tensor quantized. Per channel is only for weights.
        logger.e(
            logger.Code.NOT_IMPLEMENTED,
            "Propagation of quantization for PerChannel quantized tensors is not yet supported",
        )

    # noinspection PyTypeChecker
    if not _validate_or_set_quant_params(to_tensor, from_tensor.quantization):
        logger.e(
            logger.Code.INVALID_ONNX_MODEL,
            f'Mismatched quantization parameters between tensors "{from_tensor.name}" and "{to_tensor.name}"',
        )


def set_quantization_parameters_to_tensor(
    tflite_tensor: tflite_model.Tensor,
    scale: np.ndarray,
    zero_point: np.ndarray,
    quantized_dimension: int = 0,
):
    """Create a TFLite QuantizationParameters object, initialize it from given parameters and add it to the
    'tflite_tensor'.
    :param tflite_tensor: The TFLite tensor in the model, to add the quantization to.
    :param scale: The data of the tensor, which is an input of a quantized ONNX operator and represents the
                  quantization scale.
    :param zero_point: The data of the tensor, which is an input of a quantized ONNX operator and represents the
                       quantization zero point.
    :param quantized_dimension: The quantized dimension attribute of TFLite QuantizationParameters.
    """
    if (scale is None) or (zero_point is None):
        logger.e(
            logger.Code.NOT_IMPLEMENTED,
            "Conversion of ONNX quantized operators is only supported when "
            "the quantization parameters are static!",
        )

    if (scale.size == 1) and (zero_point.size == 1):
        # Per tensor quantization
        scale = [scale.item()]
        zero_point = [zero_point.item()]

    elif (scale.size != 1) and (zero_point.size != 1):
        # Per channel quantization

        if scale.size != zero_point.size:
            logger.e(
                logger.Code.INVALID_ONNX_MODEL,
                f"The per channel quantization parameters of ONNX tensor "
                f"'{tflite_tensor.name}' are of different sizes! ('{scale.size}'"
                f" != '{zero_point.size}')",
            )

        quantized_dimension_size = tflite_tensor.shape.get(quantized_dimension)
        if scale.size != quantized_dimension_size:
            logger.e(
                logger.Code.INVALID_ONNX_MODEL,
                f"The ONNX per channel quantization parameter vectors do not "
                f"match the size of the quantized dimension! ('{scale.size}' != "
                f"'{quantized_dimension_size}')",
            )

        scale = scale.tolist()
        zero_point = zero_point.tolist()

    else:
        # Combination of per tensor and per channel quantization parameters
        logger.e(
            logger.Code.INVALID_ONNX_MODEL,
            f"ONNX tensor '{tflite_tensor.name}' uses a combination of per "
            f"tensor and per channel quantization parameters. Conversion to "
            f"TFLite is not possible!",
        )

    quant = tflite_model.Quantization(
        scale=tflite_model.Scale(scale),
        zero_point=tflite_model.ZeroPoint(zero_point),
        quantized_dimension=quantized_dimension,
    )
    if not _validate_or_set_quant_params(tflite_tensor, quant):
        logger.e(
            logger.Code.INVALID_ONNX_MODEL,
            f'Mismatched quantization parameters between tensors: "{tflite_tensor.name}" already '
            f"has the quantization params set",
        )


def quantize_int8(
    data: np.ndarray, scale: List[float], zero_point: List[int]
) -> np.ndarray:
    new_data = np.add(np.round(np.divide(data, scale)), zero_point)
    return np.clip(new_data, -128, 127).astype(np.int8)


def dequantize(
    data: np.ndarray, scale: List[float], zero_point: List[int]
) -> np.ndarray:
    return np.multiply(
        np.subtract(np.array(data, dtype=np.float32), zero_point),
        scale,
        dtype=np.float32,
    )
