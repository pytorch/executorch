# Copyright 2024 NXP
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import executorch.backends.nxp.backend.ir.converter.builder.model_builder as model_builder
from executorch.backends.nxp.backend.ir import logger
from executorch.backends.nxp.backend.ir.lib.tflite.BuiltinOperator import (
    BuiltinOperator,
)
from executorch.backends.nxp.backend.ir.tflite_generator import tflite_model

InputTensorToOpsMap = dict[str, list[tflite_model.Operator]]
OutputTensorToOpMap = dict[str, tflite_model.Operator]
NameToTensorMap = dict[str, tflite_model.Tensor | list[tflite_model.Tensor]]


def create_tensor_to_operator_dictionaries(
    builder: "model_builder.ModelBuilder",
) -> tuple[InputTensorToOpsMap, OutputTensorToOpMap]:
    """Create and return 2 dictionaries, which map a tensor name, to a TFLite operator, which has the tensor as
    input, and output respectively.

    :return: Dictionary mapping a tensor name to a list of operators that use it as an input,
             dictionary mapping a tensor name to the operator, which produces it as its output.
    """
    input_tensor_to_operators: InputTensorToOpsMap = {}
    output_tensor_to_operator: OutputTensorToOpMap = {}

    for op in builder.get_operators().vector:
        for input_tensor in op.tmp_inputs:
            if input_tensor.name not in input_tensor_to_operators.keys():
                input_tensor_to_operators[input_tensor.name] = []

            input_tensor_to_operators[input_tensor.name].append(op)

        for output_tensor in op.tmp_outputs:
            output_tensor_to_operator[output_tensor.name] = op

    return input_tensor_to_operators, output_tensor_to_operator


# Extend this map with operators required for future optimizations.
op_type_to_builtin_operator_map = {
    "Add": BuiltinOperator.ADD,
    "AddN": BuiltinOperator.ADD_N,
    "AveragePool2D": BuiltinOperator.AVERAGE_POOL_2D,
    "BatchMatMul": BuiltinOperator.BATCH_MATMUL,
    "BidirectionalSequenceLSTM": BuiltinOperator.BIDIRECTIONAL_SEQUENCE_LSTM,
    "BidirectionalSequenceRNN": BuiltinOperator.BIDIRECTIONAL_SEQUENCE_RNN,
    "Cast": BuiltinOperator.CAST,
    "Concatenation": BuiltinOperator.CONCATENATION,
    "Conv2D": BuiltinOperator.CONV_2D,
    "Conv3D": BuiltinOperator.CONV_3D,
    "DepthwiseConv2D": BuiltinOperator.DEPTHWISE_CONV_2D,
    "Dequantize": BuiltinOperator.DEQUANTIZE,
    "Div": BuiltinOperator.DIV,
    "FullyConnected": BuiltinOperator.FULLY_CONNECTED,
    "HardSwish": BuiltinOperator.HARD_SWISH,
    "L2Norm": BuiltinOperator.L2_NORMALIZATION,
    "LSTM": BuiltinOperator.LSTM,
    "LeakyRelu": BuiltinOperator.LEAKY_RELU,
    "Logistic": BuiltinOperator.LOGISTIC,
    "MaxPool2D": BuiltinOperator.MAX_POOL_2D,
    "Maximum": BuiltinOperator.MAXIMUM,
    "Mean": BuiltinOperator.MEAN,
    "Minimum": BuiltinOperator.MINIMUM,
    "Mul": BuiltinOperator.MUL,
    "PRelu": BuiltinOperator.PRELU,
    "Quantize": BuiltinOperator.QUANTIZE,
    "RNN": BuiltinOperator.RNN,
    "ReduceProd": BuiltinOperator.REDUCE_PROD,
    "Relu": BuiltinOperator.RELU,
    "Relu6": BuiltinOperator.RELU6,
    "ReluN1To1": BuiltinOperator.RELU_N1_TO_1,
    "Reshape": BuiltinOperator.RESHAPE,
    "SVDF": BuiltinOperator.SVDF,
    "ScatterND": BuiltinOperator.SCATTER_ND,
    "SequenceRNN": BuiltinOperator.UNIDIRECTIONAL_SEQUENCE_RNN,
    "Sign": BuiltinOperator.SIGN,
    "Slice": BuiltinOperator.SLICE,
    "Split": BuiltinOperator.SPLIT,
    "StridedSlice": BuiltinOperator.STRIDED_SLICE,
    "Sub": BuiltinOperator.SUB,
    "Sum": BuiltinOperator.SUM,
    "Tanh": BuiltinOperator.TANH,
    "Transpose": BuiltinOperator.TRANSPOSE,
    "TransposeConv": BuiltinOperator.TRANSPOSE_CONV,
    "UnidirectionalSequenceLSTM": BuiltinOperator.UNIDIRECTIONAL_SEQUENCE_LSTM,
    "Where": BuiltinOperator.WHERE,
}


def builtin_operator_for_op_type(op_type: str) -> BuiltinOperator:
    builtin_op = op_type_to_builtin_operator_map.get(op_type, None)
    if builtin_op is None:
        logger.e(
            logger.Code.INTERNAL_ERROR,
            f"PatternMatcher doesn't support `{op_type}` yet.",
        )

    return builtin_op


def operator_is_type(
    op: tflite_model.Operator, op_type: str, builder: "model_builder.ModelBuilder"
):
    builtin_op = builtin_operator_for_op_type(op_type)

    opcode_indices = builder.op_code_type_index_map.get(builtin_op, None)
    if opcode_indices is None:
        # The operator is not present in the model at all.
        return False

    return op.opcode_index in opcode_indices.values()
