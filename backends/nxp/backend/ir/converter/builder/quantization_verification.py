# Copyright 2024 NXP
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import abc

import numpy as np
from executorch.backends.nxp.backend.ir import logger

from executorch.backends.nxp.backend.ir.lib.tflite.BuiltinOperator import (
    BuiltinOperator,
)
from executorch.backends.nxp.backend.ir.lib.tflite.TensorType import TensorType
from executorch.backends.nxp.backend.ir.tflite_generator import tflite_model


class IOTensor(abc.ABC):
    idx: int


class Input(IOTensor):
    def __init__(self, idx):
        self.idx = idx

    def __str__(self):
        return f"Input(idx={self.idx})"


class OptionalInput(IOTensor):
    def __init__(self, idx):
        self.idx = idx

    def __str__(self):
        return f"OptionalInput(idx={self.idx})"


class Output(IOTensor):
    def __init__(self, idx):
        self.idx = idx

    def __str__(self):
        return f"Output(idx={self.idx})"


class QuantizationRule(abc.ABC):

    @abc.abstractmethod
    def valid(self, op: tflite_model.Operator) -> bool:
        pass

    @abc.abstractmethod
    def __str__(self):
        pass


class SharedParamsForType(QuantizationRule):

    def __init__(self, tensor_type: TensorType, *tensors: IOTensor):
        self.tensor_type = tensor_type
        self.tensors = tensors

    def valid(self, op: tflite_model.Operator) -> bool:
        shared_tensors = []
        for tensor in self.tensors:
            if isinstance(tensor, Input):
                shared_tensors.append(op.tmp_inputs[tensor.idx])
            elif isinstance(tensor, OptionalInput):
                if tensor.idx < len(op.tmp_inputs):
                    shared_tensors.append(op.tmp_inputs[tensor.idx])
                else:
                    return True
            elif isinstance(tensor, Output):
                shared_tensors.append(op.tmp_outputs[tensor.idx])
            else:
                logger.e(
                    logger.Code.INTERNAL_ERROR, f"Unknown IOTensor type: {type(tensor)}"
                )

        if shared_tensors[0].type != self.tensor_type:
            return True

        if all(tensor.quantization is None for tensor in shared_tensors):
            return True

        first_quantization = shared_tensors[0].quantization

        # Check quantization values (scales & zero-points)
        scales_same = all(
            first_quantization.scale == t.quantization.scale for t in shared_tensors[1:]
        )
        zp_same = all(
            first_quantization.zero_point == t.quantization.zero_point
            for t in shared_tensors[1:]
        )
        return scales_same and zp_same

    def __str__(self):
        return (
            f"Q-params match required for tensors: {', '.join(map(str, self.tensors))}"
        )


class ExactValueForType(QuantizationRule):

    def __init__(
        self,
        tensor_type: TensorType,
        tensor: IOTensor,
        scale: list[float],
        zero_point: list,
    ):
        self.tensor = tensor
        self.tensor_type = tensor_type
        self.scale = scale
        self.zero_point = zero_point

    def valid(self, op: tflite_model.Operator) -> bool:
        if isinstance(self.tensor, Input):
            tflite_tensor = op.tmp_inputs[self.tensor.idx]
        elif isinstance(self.tensor, OptionalInput):
            if self.tensor.idx < len(op.tmp_inputs):
                tflite_tensor = op.tmp_outputs[self.tensor.idx]
            else:
                return True
        elif isinstance(self.tensor, Output):
            tflite_tensor = op.tmp_outputs[self.tensor.idx]
        else:
            logger.e(
                logger.Code.INTERNAL_ERROR,
                f"Unknown IOTensor type: {type(self.tensor)}",
            )

        if tflite_tensor.quantization is None or self.tensor_type != tflite_tensor.type:
            return True

        scale = tflite_tensor.quantization.scale.vector
        zp = tflite_tensor.quantization.zero_point.vector

        # noinspection PyTypeChecker
        return np.allclose(scale, self.scale) and np.allclose(zp, self.zero_point)

    def __str__(self):
        return f"ExactValue(scale={self.scale}, zero_point={self.zero_point}, type={self.tensor_type}, tensor={self.tensor})"


class FullyConnectedWeightZeroPoint(QuantizationRule):
    """LiteRT documentation says that `FullyConnected` must have weight zero point = 0
     (https://ai.google.dev/edge/litert/models/quantization_spec)
    If this condition is not satisfied, LiteRT will not raise any errors but the output will not be correct.

    However, if the `weights` are dynamic the kernels DO in fact support any zero point. Not just 0s.
    """

    def valid(self, op: tflite_model.Operator) -> bool:
        weights = op.tmp_inputs[1]
        if weights.quantization is None:
            return True

        if weights.tmp_buffer is None or weights.tmp_buffer.data is None:
            # The `weights` are dynamic. LiteRT supports any zero point in this case.
            return True

        else:
            # Static `weights`.
            if weights.type == TensorType.INT8:
                zero_point = 0
            elif weights.type == TensorType.UINT8:
                zero_point = 128
            else:
                return True

            return all(zp == zero_point for zp in weights.quantization.zero_point)

    def __str__(self):
        return "FullyConnectedWeightZeroPoint()"


class ValidBiasValues(QuantizationRule):

    def valid(self, op: tflite_model.Operator) -> bool:
        if len(op.tmp_inputs) < 3:
            # Bias tensor not present -> ignore
            return True
        if (bias_quant := op.tmp_inputs[2].quantization) is None:
            # Not quantized -> ignore
            return True

        if (input_1_quant := op.tmp_inputs[0].quantization) is None:
            logger.w(
                "Bias tensor quantized but first input tensor not. This is not supported in TFLite."
            )
            return False
        if (input_2_quant := op.tmp_inputs[1].quantization) is None:
            logger.w(
                "Bias tensor quantized but weight tensor not. This is not supported in TFLite."
            )
            return False

        if op.tmp_inputs[2].type != TensorType.INT32:
            logger.w(
                "Quantized bias tensor's type isn't INT32. This is not supported in TFLite."
            )
            return False

        expected_bias_scale = np.array(input_1_quant.scale.vector) * np.array(
            input_2_quant.scale.vector
        )

        if not np.allclose(
            expected_bias_scale.astype(np.float32),
            np.array(bias_quant.scale.vector, dtype=np.float32),
        ):
            logger.w(
                f"Scale of quantized bias tensor '{op.tmp_inputs[2].name}' is not equal to 'input0_scale * "
                "input1_scale[...]'. This is not supported in TFLite."
            )
            return False

        if bias_quant.zero_point.vector[0] != 0:
            logger.w(
                "Zero point of quantized bias tensor is not equal to '0'. This is not supported in TFLite."
            )
            return False

        return True

    def __str__(self):
        return "ExactBiasValues()"


def verify_quantization_integrity(model: tflite_model.Model):
    rules = {
        BuiltinOperator.AVERAGE_POOL_2D: [
            SharedParamsForType(TensorType.INT8, Input(0), Output(0)),
            SharedParamsForType(TensorType.UINT8, Input(0), Output(0)),
        ],
        BuiltinOperator.BROADCAST_TO: [
            SharedParamsForType(TensorType.INT8, Input(0), Output(0)),
            SharedParamsForType(TensorType.UINT8, Input(0), Output(0)),
        ],
        BuiltinOperator.CONCATENATION: [
            SharedParamsForType(TensorType.INT8, Input(0), Output(0)),
            SharedParamsForType(TensorType.INT8, Input(0), OptionalInput(1)),
            SharedParamsForType(TensorType.INT8, Input(0), OptionalInput(2)),
            SharedParamsForType(TensorType.INT8, Input(0), OptionalInput(3)),
            SharedParamsForType(TensorType.INT8, Input(0), OptionalInput(4)),
        ],
        BuiltinOperator.CONV_2D: [ValidBiasValues()],
        BuiltinOperator.DEPTHWISE_CONV_2D: [ValidBiasValues()],
        BuiltinOperator.FULLY_CONNECTED: [
            ValidBiasValues(),
            FullyConnectedWeightZeroPoint(),
        ],
        BuiltinOperator.GATHER: [
            SharedParamsForType(TensorType.INT8, Input(0), Output(0)),
            SharedParamsForType(TensorType.UINT8, Input(0), Output(0)),
        ],
        BuiltinOperator.GATHER_ND: [
            SharedParamsForType(TensorType.INT8, Input(0), Output(0)),
            SharedParamsForType(TensorType.UINT8, Input(0), Output(0)),
        ],
        BuiltinOperator.L2_NORMALIZATION: [
            ExactValueForType(TensorType.INT8, Output(0), [1.0 / 128.0], [0]),
        ],
        BuiltinOperator.LOG_SOFTMAX: [
            ExactValueForType(TensorType.INT8, Output(0), [16.0 / 256.0], [127]),
            ExactValueForType(TensorType.UINT8, Output(0), [16.0 / 256.0], [255]),
        ],
        BuiltinOperator.LOGISTIC: [
            ExactValueForType(TensorType.INT8, Output(0), [1.0 / 256.0], [-128]),
            ExactValueForType(TensorType.UINT8, Output(0), [1.0 / 256.0], [0]),
        ],
        BuiltinOperator.MAX_POOL_2D: [
            SharedParamsForType(TensorType.INT8, Input(0), Output(0)),
            SharedParamsForType(TensorType.UINT8, Input(0), Output(0)),
        ],
        BuiltinOperator.MAXIMUM: [
            SharedParamsForType(TensorType.INT8, Input(0), Output(0)),
            SharedParamsForType(TensorType.INT8, Input(0), Input(1)),
            SharedParamsForType(TensorType.UINT8, Input(0), Output(0)),
            SharedParamsForType(TensorType.UINT8, Input(0), Input(1)),
        ],
        BuiltinOperator.MINIMUM: [
            SharedParamsForType(TensorType.INT8, Input(0), Output(0)),
            SharedParamsForType(TensorType.INT8, Input(0), Input(1)),
            SharedParamsForType(TensorType.UINT8, Input(0), Output(0)),
            SharedParamsForType(TensorType.UINT8, Input(0), Input(1)),
        ],
        BuiltinOperator.PAD: [
            SharedParamsForType(TensorType.INT8, Input(0), Output(0)),
            SharedParamsForType(TensorType.UINT8, Input(0), Output(0)),
        ],
        BuiltinOperator.PADV2: [
            SharedParamsForType(TensorType.INT8, Input(0), Output(0)),
            SharedParamsForType(TensorType.INT8, Input(0), OptionalInput(2)),
            SharedParamsForType(TensorType.UINT8, Input(0), Output(0)),
            SharedParamsForType(TensorType.UINT8, Input(0), OptionalInput(2)),
        ],
        BuiltinOperator.RESHAPE: [
            SharedParamsForType(TensorType.INT8, Input(0), Output(0)),
            SharedParamsForType(TensorType.UINT8, Input(0), Output(0)),
        ],
        BuiltinOperator.RESIZE_BILINEAR: [
            SharedParamsForType(TensorType.INT8, Input(0), Output(0)),
            SharedParamsForType(TensorType.UINT8, Input(0), Output(0)),
        ],
        BuiltinOperator.RESIZE_NEAREST_NEIGHBOR: [
            SharedParamsForType(TensorType.INT8, Input(0), Output(0)),
            SharedParamsForType(TensorType.UINT8, Input(0), Output(0)),
        ],
        BuiltinOperator.SCATTER_ND: [
            SharedParamsForType(TensorType.INT8, Input(1), Output(0)),
            SharedParamsForType(TensorType.UINT8, Input(1), Output(0)),
        ],
        BuiltinOperator.SELECT_V2: [
            SharedParamsForType(TensorType.INT8, Input(1), Output(0)),
            SharedParamsForType(TensorType.INT8, Input(1), Input(2)),
            SharedParamsForType(TensorType.UINT8, Input(1), Output(0)),
            SharedParamsForType(TensorType.UINT8, Input(1), Input(2)),
        ],
        BuiltinOperator.SLICE: [
            SharedParamsForType(TensorType.INT8, Input(0), Output(0)),
            SharedParamsForType(TensorType.UINT8, Input(0), Output(0)),
        ],
        BuiltinOperator.SOFTMAX: [
            ExactValueForType(TensorType.INT8, Output(0), [1.0 / 256.0], [-128]),
            ExactValueForType(TensorType.UINT8, Output(0), [1.0 / 256.0], [0]),
        ],
        BuiltinOperator.SQUEEZE: [
            SharedParamsForType(TensorType.INT8, Input(0), Output(0)),
            SharedParamsForType(TensorType.UINT8, Input(0), Output(0)),
        ],
        BuiltinOperator.TANH: [
            ExactValueForType(TensorType.INT8, Output(0), [1.0 / 128.0], [0]),
        ],
        BuiltinOperator.TILE: [
            SharedParamsForType(TensorType.INT8, Input(0), Output(0)),
            SharedParamsForType(TensorType.UINT8, Input(0), Output(0)),
        ],
        BuiltinOperator.TRANSPOSE: [
            SharedParamsForType(TensorType.INT8, Input(0), Output(0)),
            SharedParamsForType(TensorType.UINT8, Input(0), Output(0)),
        ],
    }

    ops: list[tflite_model.Operator] = model.sub_graphs.vector[0].operators.vector
    operator_codes = {
        idx: code.builtin_code for idx, code in enumerate(model.operator_codes.vector)
    }
    is_error = False

    for op in ops:
        if op.builtin_options:
            if op.builtin_options.operator_type in rules:
                for rule in rules[op.builtin_options.operator_type]:
                    if not rule.valid(op):
                        logger.w(
                            f"TFLite operator with op_type='{op.builtin_options.operator_type}' wasn't quantized "
                            f"properly. Following TFLite quantization rule was not satisfied: '{rule}'."
                        )
                        is_error = True
        else:
            if operator_codes[op.opcode_index] in rules:
                for rule in rules[operator_codes[op.opcode_index]]:
                    if not rule.valid(op):
                        logger.w(
                            f"TFLite operator with op_type='{operator_codes[op.opcode_index]}' wasn't quantized "
                            f"properly. Following TFLite quantization rule was not satisfied: '{rule}'."
                        )
                        is_error = True

    if is_error:
        logger.e(
            logger.Code.INTERNAL_ERROR,
            "Some ops were not correctly quantized. Refer to previous log messages and please report this issue.",
        )
