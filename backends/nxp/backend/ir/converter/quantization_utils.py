# Copyright 2023 NXP
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import copy
from typing import Iterable, List, Optional

import executorch.backends.nxp.backend.ir.converter.builder.model_builder as model_builder

import numpy as np
from executorch.backends.nxp.backend.ir import logger as logger
from executorch.backends.nxp.backend.ir.converter.conversion.translator import (
    tf_lite_type_to_numpy,
)
from executorch.backends.nxp.backend.ir.lib.tflite import TensorType as tflTensorType
from executorch.backends.nxp.backend.ir.lib.tflite.TensorType import TensorType
from executorch.backends.nxp.backend.ir.tflite_generator import (
    tflite_model as tflite_model,
)


def quantization_is_equal(
    x_scale: np.ndarray,
    x_zp: np.ndarray,
    x_type: TensorType,
    y_scale: np.ndarray,
    y_zp: np.ndarray,
    y_type: TensorType,
) -> bool:
    """Determine if provided quantization parameters of tensors 'x' and 'y' are the same.

    :param x_scale: Scale of the 'x' tensor.
    :param x_zp: Zero point of the 'x' tensor.
    :param x_type: TFLite data type of the 'x' tensor.
    :param y_scale: Scale of the 'y' tensor.
    :param y_zp: Zero point of the 'y' tensor.
    :param y_type: TFLite data type of the 'y' tensor.
    :return: True, if the quantization parameters are equal.
    """
    if x_type != y_type:
        return False

    if not (x_scale.size == x_zp.size == y_scale.size == y_zp.size):
        return False

    x_scale, x_zp = quantization_params_to_lists(x_scale, x_zp)
    y_scale, y_zp = quantization_params_to_lists(y_scale, y_zp)

    return all(
        x_s == y_s and x_z == y_z
        for x_s, y_s, x_z, y_z in zip(x_scale, y_scale, x_zp, y_zp)
    )


def quantization_params_to_lists(
    scale: np.ndarray, zero_point: np.ndarray
) -> (List[float], List[int]):
    if (scale is None) or (zero_point is None):
        logger.e(
            logger.Code.INTERNAL_ERROR,
            "Missing zero_point and/or scale quantization params when converting to list!",
        )

    if (scale.size == 1) and (zero_point.size == 1):
        # Per tensor quantization
        scale = [scale.item()]
        zero_point = [zero_point.item()]
    elif (scale.size != 1) and (zero_point.size != 1):
        # Per channel quantization
        scale = scale.tolist()
        zero_point = zero_point.tolist()
    else:
        logger.e(
            logger.Code.CONVERSION_IMPOSSIBLE,
            "TFLite doesn't support combination of per-channel and per-tensor quantization params.",
        )

    return scale, zero_point


def is_quantization_valid(scale, zero_point):
    return scale.size == zero_point.size


def is_per_tensor_quantized(scale, zero_point):
    return (scale.size == 1) and (zero_point.size == 1)


def is_per_channel_quantized(scale, zero_point):
    return is_quantization_valid(scale, zero_point) and not is_per_tensor_quantized(
        scale, zero_point
    )


def get_symmetric_zero_point_for_type(tensor_type: TensorType):
    match tensor_type:
        case TensorType.INT8:
            return 0
        case TensorType.UINT8:
            return 128
        case _:
            logger.e(
                logger.Code.INTERNAL_ERROR,
                f"Attempt to get zero point definition for type: {tensor_type}",
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


def calculate_uint_to_int_re_quantization_zero_point(
    data_type_byte_size: int, old_zero_point: Iterable[int]
) -> np.ndarray:
    """
        Calculate the new zero points, after a quantized tensor with an unsigned int data type is re-quantized to
        a signed type.
    :param data_type_byte_size: Size of the data type that is used, in Bytes. For example 1 for INT8.
    :param old_zero_point: The zero point quantisation parameter, of the original data, before re-quantization.
    :return: The new zero point quantisation parameter, after re-quantization.
    """
    data_type_bit_size = 8 * data_type_byte_size
    zero_point_shift = 2 ** (data_type_bit_size - 1)
    return np.asarray(np.subtract(np.array(old_zero_point, np.int32), zero_point_shift))


def _re_quantize_uint8_to_int8(tensor_data: np.ndarray) -> np.ndarray:
    """Re-quantize static uint8 data to int8."""
    int16_data = np.asarray(tensor_data, np.int16)
    return np.array(int16_data - 128, np.int8)


def quantize_int8(
    data: np.ndarray, scale: List[float], zero_point: List[int]
) -> np.ndarray:
    new_data = np.add(np.round(np.divide(data, scale)), zero_point)
    return np.clip(new_data, -128, 127).astype(np.int8)


def quantize_uint8(
    data: np.ndarray, scale: List[float], zero_point: List[int]
) -> np.ndarray:
    new_data = np.add(np.round(np.divide(data, scale)), zero_point)
    return np.clip(new_data, 0, 255).astype(np.uint8)


def quantize_int32(
    data: np.ndarray, scale: List[float], zero_point: List[int]
) -> np.ndarray:
    new_data = np.add(np.round(np.divide(data, scale)), zero_point)
    return np.clip(new_data, -2_147_483_648, 2_147_483_648).astype(np.int32)


def dequantize(
    data: np.ndarray, scale: List[float], zero_point: List[int]
) -> np.ndarray:
    return np.multiply(
        np.subtract(np.array(data, dtype=np.float32), zero_point),
        scale,
        dtype=np.float32,
    )


def re_quantize_static_tensor(
    builder: "model_builder.ModelBuilder",
    tflite_tensor: tflite_model.Tensor,
    to_type: tflTensorType.TensorType,
    new_scale: Optional[List[float]] = None,
    new_zero_point: Optional[List[int]] = None,
) -> tflite_model.Tensor:
    """Create a new TFLite Tensor with new quantization parameters, type and data.

    :param builder: A ModelBuilder instance.
    :param tflite_tensor: TFLite tensor to re-quantize.
    :param to_type: The TFLite TensorType, that the tensor will be re-quantized to.
    :param new_scale: New scale quantization parameter. Used only when re-quantizing to the same type.
    :param new_zero_point: New zero point quantization parameter. Used only when re-quantizing to the same type.
    :return: A new re-quantized tensor.
    """
    if tflite_tensor.quantization is None:
        logger.e(
            logger.Code.INTERNAL_ERROR,
            "translator.re_quantize_static_tensor(): Got tensor without quantization!",
        )

    if tflite_tensor.tmp_buffer.data is None:
        logger.e(
            logger.Code.INTERNAL_ERROR,
            "translator.re_quantize_static_tensor(): Got tensor without static data!",
        )

    new_dtype = tf_lite_type_to_numpy(to_type)
    re_quantized_tensor = builder.duplicate_tensor(tflite_tensor)
    tensor_data = re_quantized_tensor.tmp_buffer.data

    if tensor_data.dtype == np.uint8 and new_dtype == np.int8:  # INT8 -> UINT8
        re_quantized_tensor.tmp_buffer.data = _re_quantize_uint8_to_int8(tensor_data)
        re_quantized_tensor.type = tflTensorType.TensorType.INT8
        calculated_zero_point = calculate_uint_to_int_re_quantization_zero_point(
            1, re_quantized_tensor.quantization.zero_point.vector
        )
        re_quantized_tensor.quantization.zero_point = tflite_model.ZeroPoint(
            list(calculated_zero_point)
        )

    elif tensor_data.dtype == np.int32 and new_dtype == np.int8:  # INT32 -> INT8
        if new_zero_point is None or new_scale is None:
            logger.e(
                logger.Code.INTERNAL_ERROR,
                "Missing new zero_point or new scale when re-quantizing tensor.",
            )

        old_zp = re_quantized_tensor.quantization.zero_point.vector
        old_scale = re_quantized_tensor.quantization.scale.vector
        float_data = dequantize(tensor_data, old_scale, old_zp)
        int8_data = quantize_int8(float_data, new_scale, new_zero_point)

        re_quantized_tensor.tmp_buffer.data = int8_data
        re_quantized_tensor.type = tflTensorType.TensorType.INT8
        re_quantized_tensor.quantization.zero_point = tflite_model.ZeroPoint(
            list(new_zero_point)
        )
        re_quantized_tensor.quantization.scale = tflite_model.Scale(list(new_scale))

    elif tensor_data.dtype == np.int32 and new_dtype == np.uint8:  # INT32 -> UINT8
        if new_zero_point is None or new_scale is None:
            logger.e(
                logger.Code.INTERNAL_ERROR,
                "Missing new zero_point or new scale when re-quantizing tensor.",
            )

        old_zp = re_quantized_tensor.quantization.zero_point.vector
        old_scale = re_quantized_tensor.quantization.scale.vector
        float_data = dequantize(tensor_data, old_scale, old_zp)
        uint8_data = quantize_uint8(float_data, new_scale, new_zero_point)

        re_quantized_tensor.tmp_buffer.data = uint8_data
        re_quantized_tensor.type = tflTensorType.TensorType.UINT8
        re_quantized_tensor.quantization.zero_point = tflite_model.ZeroPoint(
            list(new_zero_point)
        )
        re_quantized_tensor.quantization.scale = tflite_model.Scale(list(new_scale))

    elif tensor_data.dtype == np.int8 and new_dtype == np.int8:  # INT8 -> INT8
        # Re-quantizing int8 tensor data with different quantization parameters
        if new_zero_point is None or new_scale is None:
            logger.e(
                logger.Code.INTERNAL_ERROR,
                "Missing new zero_point or new scale when re-quantizing tensor.",
            )

        zero_point_data = re_quantized_tensor.quantization.zero_point.vector
        scale_data = re_quantized_tensor.quantization.scale.vector
        new_tensor_data = dequantize(tensor_data, scale_data, zero_point_data)

        re_quantized_tensor.tmp_buffer.data = quantize_int8(
            new_tensor_data, new_scale, new_zero_point
        )
        re_quantized_tensor.quantization.scale = tflite_model.Scale(new_scale)
        re_quantized_tensor.quantization.zero_point = tflite_model.ZeroPoint(
            new_zero_point
        )

    elif tensor_data.dtype == np.int32 and new_dtype == np.int32:  # INT32 -> INT32
        if new_zero_point is None or new_scale is None:
            logger.e(
                logger.Code.INTERNAL_ERROR,
                "Missing new zero_point or new scale when re-quantizing tensor.",
            )

        old_zp = re_quantized_tensor.quantization.zero_point.vector
        old_scale = re_quantized_tensor.quantization.scale.vector
        float_data = dequantize(tensor_data, old_scale, old_zp)
        int32_data = quantize_int32(float_data, new_scale, new_zero_point)

        re_quantized_tensor.tmp_buffer.data = int32_data
        re_quantized_tensor.quantization.zero_point = tflite_model.ZeroPoint(
            list(new_zero_point)
        )
        re_quantized_tensor.quantization.scale = tflite_model.Scale(list(new_scale))

    else:
        logger.e(
            logger.Code.NOT_IMPLEMENTED,
            f"Re-quantization of static tensors from type '{tensor_data.dtype}' "
            f"to type '{to_type}' is not yet implemented!",
        )

    return re_quantized_tensor


def quantize_static_float_tensor(
    builder: "model_builder.ModelBuilder",
    tflite_tensor: tflite_model.Tensor,
    to_type: tflTensorType.TensorType,
    scale: List[float],
    zero_point: List[int],
    quantized_dimension: int = 0,
) -> tflite_model.Tensor:
    """Quantize tensor 'tflite_tensor' with passed quantization params.

    :param builder: A ModelBuilder instance.
    :param tflite_tensor: TFLite tensor to quantize.
    :param to_type: The TFLite TensorType, that the tensor will be quantized to.
    :param scale: Scale quantization parameter.
    :param zero_point: Zero point quantization parameter.
    :param quantized_dimension: Quantized dimension.
    """
    if tflite_tensor.quantization is not None:
        logger.e(logger.Code.INTERNAL_ERROR, "Got tensor with quantization!")

    if tflite_tensor.tmp_buffer.data is None:
        logger.e(logger.Code.INTERNAL_ERROR, "Got tensor without static data!")

    quantized_tensor = builder.duplicate_tensor(tflite_tensor)
    tensor_data = quantized_tensor.tmp_buffer.data

    if zero_point is None or scale is None:
        logger.e(
            logger.Code.INTERNAL_ERROR,
            "Missing new zero_point or new scale when quantizing tensor.",
        )

    new_dtype = tf_lite_type_to_numpy(to_type)

    if tensor_data.dtype == np.float32 and new_dtype == np.int8:
        int8_data = quantize_int8(tensor_data, scale, zero_point)

        quantized_tensor.tmp_buffer.data = int8_data
        quantized_tensor.type = tflTensorType.TensorType.INT8
        quantized_tensor.quantization = tflite_model.Quantization()
        quantized_tensor.quantization.zero_point = tflite_model.ZeroPoint(
            list(zero_point)
        )
        quantized_tensor.quantization.scale = tflite_model.Scale(list(scale))
        quantized_tensor.quantization.quantized_dimension = quantized_dimension

    elif tensor_data.dtype == np.float32 and new_dtype == np.uint8:
        uint8_data = quantize_uint8(tensor_data, scale, zero_point)

        quantized_tensor.tmp_buffer.data = uint8_data
        quantized_tensor.type = tflTensorType.TensorType.UINT8
        quantized_tensor.quantization = tflite_model.Quantization()
        quantized_tensor.quantization.zero_point = tflite_model.ZeroPoint(
            list(zero_point)
        )
        quantized_tensor.quantization.scale = tflite_model.Scale(list(scale))
        quantized_tensor.quantization.quantized_dimension = quantized_dimension

    elif tensor_data.dtype == np.float32 and new_dtype == np.int32:
        int32_data = quantize_int32(tensor_data, scale, zero_point)

        quantized_tensor.tmp_buffer.data = int32_data
        quantized_tensor.type = tflTensorType.TensorType.INT32
        quantized_tensor.quantization = tflite_model.Quantization()
        quantized_tensor.quantization.zero_point = tflite_model.ZeroPoint(
            list(zero_point)
        )
        quantized_tensor.quantization.scale = tflite_model.Scale(list(scale))
        quantized_tensor.quantization.quantized_dimension = quantized_dimension

    else:
        logger.e(
            logger.Code.NOT_IMPLEMENTED,
            f"Quantization of static tensors from type '{tensor_data.dtype}' "
            f"to type '{to_type}' is not yet implemented!",
        )

    return quantized_tensor
