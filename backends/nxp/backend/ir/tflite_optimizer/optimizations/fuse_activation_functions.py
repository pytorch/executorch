# Copyright 2024 NXP
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from executorch.backends.nxp.backend.ir import logger
from executorch.backends.nxp.backend.ir.lib.tflite.ActivationFunctionType import (
    ActivationFunctionType,
)
from executorch.backends.nxp.backend.ir.lib.tflite.BuiltinOperator import (
    BuiltinOperator,
)
from executorch.backends.nxp.backend.ir.tflite_generator import tflite_model
from executorch.backends.nxp.backend.ir.tflite_optimizer.graph_utils import (
    operator_is_type,
)
from executorch.backends.nxp.backend.ir.tflite_optimizer.operator_rules import (
    NoFusedActivationFunction,
)
from executorch.backends.nxp.backend.ir.tflite_optimizer.optimizations.base_optimization import (
    BaseOptimization,
)
from executorch.backends.nxp.backend.ir.tflite_optimizer.pattern_matcher import (
    Op,
    PatternMatcher,
)
from executorch.backends.nxp.backend.ir.tflite_optimizer.tensor_rules import (
    TensorHasOneConsumer,
)


class FuseActivationFunctions(BaseOptimization):
    ops_with_fused_activation_function = [
        "Conv2D",
        "Conv3D",
        "DepthwiseConv2D",
        "TransposeConv",
        "MaxPool2D",
        "AveragePool2D",
        "SVDF",
        "FullyConnected",
        "Add",
        "Mul",
        "Sub",
        "Div",
        # 'Concatenation',  # currently disabled
        # https://github.com/tensorflow/tensorflow/blob/v2.15.0/tensorflow/lite/kernels/concatenation.cc#L139
        # 'L2Norm',  # currently disabled
        # https://github.com/tensorflow/tensorflow/blob/v2.15.0/tensorflow/lite/kernels/l2norm.cc#L72
        # LSTM operators will always already have fused activation functions. They are assigned in `convert_lstm.py`.
        # 'LSTM', 'UnidirectionalSequenceLSTM', 'BidirectionalSequenceLSTM'
        # RNN operators will always already have fused activation functions. They are assigned in `convert_rnn.py`.
        # 'RNN', 'SequenceRNN', 'BidirectionalSequenceRNN',
    ]

    activation_functions = ["Relu", "ReluN1To1", "Relu6", "Tanh", "Sign"]

    supported_activations_for_op: dict[
        BuiltinOperator, list[ActivationFunctionType]
    ] = {
        BuiltinOperator.CONV_2D: [
            ActivationFunctionType.RELU,
            ActivationFunctionType.RELU_N1_TO_1,
            ActivationFunctionType.RELU6,
        ],
        # https://github.com/tensorflow/tensorflow/blob/v2.15.0/tensorflow/lite/kernels/conv.cc#L912
        # https://github.com/tensorflow/tensorflow/blob/v2.15.0/tensorflow/lite/kernels/kernel_util.h#L285-L300
        BuiltinOperator.CONV_3D: [
            ActivationFunctionType.RELU,
            ActivationFunctionType.RELU_N1_TO_1,
            ActivationFunctionType.RELU6,
        ],
        # https://github.com/tensorflow/tensorflow/blob/v2.15.0/tensorflow/lite/kernels/conv3d.cc#L213
        # https://github.com/tensorflow/tensorflow/blob/v2.15.0/tensorflow/lite/kernels/kernel_util.h#L285-L300
        BuiltinOperator.DEPTHWISE_CONV_2D: [
            ActivationFunctionType.RELU,
            ActivationFunctionType.RELU_N1_TO_1,
            ActivationFunctionType.RELU6,
        ],
        # https://github.com/tensorflow/tensorflow/blob/v2.15.0/tensorflow/lite/kernels/depthwise_conv.cc#L307
        # https://github.com/tensorflow/tensorflow/blob/v2.15.0/tensorflow/lite/kernels/kernel_util.h#L285-L300
        BuiltinOperator.TRANSPOSE_CONV: [
            ActivationFunctionType.RELU,
            ActivationFunctionType.RELU_N1_TO_1,
            ActivationFunctionType.RELU6,
        ],
        # https://github.com/tensorflow/tensorflow/blob/v2.15.0/tensorflow/lite/kernels/transpose_conv.cc#L516
        # https://github.com/tensorflow/tensorflow/blob/v2.15.0/tensorflow/lite/kernels/kernel_util.h#L285-L300
        BuiltinOperator.MAX_POOL_2D: [
            ActivationFunctionType.RELU,
            ActivationFunctionType.RELU_N1_TO_1,
            ActivationFunctionType.RELU6,
        ],
        # https://github.com/tensorflow/tensorflow/blob/v2.15.0/tensorflow/lite/kernels/pooling.cc#L247
        # https://github.com/tensorflow/tensorflow/blob/v2.15.0/tensorflow/lite/kernels/kernel_util.h#L285-L300
        BuiltinOperator.AVERAGE_POOL_2D: [
            ActivationFunctionType.RELU,
            ActivationFunctionType.RELU_N1_TO_1,
            ActivationFunctionType.RELU6,
        ],
        # https://github.com/tensorflow/tensorflow/blob/v2.15.0/tensorflow/lite/kernels/pooling.cc#L124
        # https://github.com/tensorflow/tensorflow/blob/v2.15.0/tensorflow/lite/kernels/kernel_util.h#L285-L300
        BuiltinOperator.FULLY_CONNECTED: [
            ActivationFunctionType.RELU,
            ActivationFunctionType.RELU_N1_TO_1,
            ActivationFunctionType.RELU6,
        ],
        # https://github.com/tensorflow/tensorflow/blob/v2.15.0/tensorflow/lite/kernels/fully_connected.cc#L627-L630
        BuiltinOperator.ADD: [
            ActivationFunctionType.RELU,
            ActivationFunctionType.RELU_N1_TO_1,
            ActivationFunctionType.RELU6,
        ],
        # https://github.com/tensorflow/tensorflow/blob/v2.15.0/tensorflow/lite/kernels/add.cc#L246
        # https://github.com/tensorflow/tensorflow/blob/v2.15.0/tensorflow/lite/kernels/kernel_util.h#L285-L300
        BuiltinOperator.MUL: [
            ActivationFunctionType.RELU,
            ActivationFunctionType.RELU_N1_TO_1,
            ActivationFunctionType.RELU6,
        ],
        # https://github.com/tensorflow/tensorflow/blob/v2.15.0/tensorflow/lite/kernels/mul.cc#L159
        # https://github.com/tensorflow/tensorflow/blob/v2.15.0/tensorflow/lite/kernels/kernel_util.h#L285-L300
        BuiltinOperator.SUB: [
            ActivationFunctionType.RELU,
            ActivationFunctionType.RELU_N1_TO_1,
            ActivationFunctionType.RELU6,
        ],
        # https://github.com/tensorflow/tensorflow/blob/v2.15.0/tensorflow/lite/kernels/sub.cc#L306
        # https://github.com/tensorflow/tensorflow/blob/v2.15.0/tensorflow/lite/kernels/kernel_util.h#L285-L300
        BuiltinOperator.DIV: [
            ActivationFunctionType.RELU,
            ActivationFunctionType.RELU_N1_TO_1,
            ActivationFunctionType.RELU6,
        ],
        # https://github.com/tensorflow/tensorflow/blob/v2.15.0/tensorflow/lite/kernels/div.cc#L180
        # https://github.com/tensorflow/tensorflow/blob/v2.15.0/tensorflow/lite/kernels/kernel_util.h#L285-L300
        BuiltinOperator.SVDF: [ActivationFunctionType.RELU],
        # https://github.com/tensorflow/tensorflow/blob/v2.15.0/tensorflow/lite/kernels/svdf.cc#L394
        BuiltinOperator.RNN: [
            ActivationFunctionType.RELU,
            ActivationFunctionType.RELU_N1_TO_1,
            ActivationFunctionType.RELU6,
            ActivationFunctionType.TANH,
            ActivationFunctionType.SIGN_BIT,
        ],
        # https://github.com/tensorflow/tensorflow/blob/v2.15.0/tensorflow/lite/kernels/basic_rnn.cc#L222
        # https://github.com/tensorflow/tensorflow/blob/v2.15.0/tensorflow/lite/kernels/internal/kernel_utils.cc#L71
        # https://github.com/tensorflow/tensorflow/blob/v2.15.0/tensorflow/lite/kernels/internal/tensor_utils.h#L58-L77
        BuiltinOperator.UNIDIRECTIONAL_SEQUENCE_RNN: [
            ActivationFunctionType.RELU,
            ActivationFunctionType.RELU_N1_TO_1,
            ActivationFunctionType.RELU6,
            ActivationFunctionType.TANH,
            ActivationFunctionType.SIGN_BIT,
        ],
        # https://github.com/tensorflow/tensorflow/blob/6887368d6d46223f460358323c4b76d61d1558a8/tensorflow/lite/kernels/unidirectional_sequence_rnn.cc#L239
        # https://github.com/tensorflow/tensorflow/blob/v2.15.0/tensorflow/lite/kernels/internal/kernel_utils.cc#L71
        # https://github.com/tensorflow/tensorflow/blob/v2.15.0/tensorflow/lite/kernels/internal/tensor_utils.h#L58-L77
        BuiltinOperator.BIDIRECTIONAL_SEQUENCE_RNN: [
            ActivationFunctionType.RELU,
            ActivationFunctionType.RELU_N1_TO_1,
            ActivationFunctionType.RELU6,
            ActivationFunctionType.TANH,
            ActivationFunctionType.SIGN_BIT,
        ],
        # https://github.com/tensorflow/tensorflow/blob/6887368d6d46223f460358323c4b76d61d1558a8/tensorflow/lite/kernels/bidirectional_sequence_rnn.cc#L433
        # https://github.com/tensorflow/tensorflow/blob/v2.15.0/tensorflow/lite/kernels/internal/kernel_utils.cc#L71
        # https://github.com/tensorflow/tensorflow/blob/v2.15.0/tensorflow/lite/kernels/internal/tensor_utils.h#L58-L77
    }

    ops_that_need_equal_io_quantization = [
        # Documented restrictions from https://www.tensorflow.org/lite/performance/quantization_spec
        BuiltinOperator.AVERAGE_POOL_2D,
        BuiltinOperator.MAX_POOL_2D,
        BuiltinOperator.CONCATENATION,
    ]

    def _act_fun_type_for_op(self, op: tflite_model.Operator) -> ActivationFunctionType:
        if operator_is_type(op, "Relu", self._builder):
            return ActivationFunctionType.RELU
        elif operator_is_type(op, "ReluN1To1", self._builder):
            return ActivationFunctionType.RELU_N1_TO_1
        elif operator_is_type(op, "Relu6", self._builder):
            return ActivationFunctionType.RELU6
        elif operator_is_type(op, "Tanh", self._builder):
            return ActivationFunctionType.TANH
        elif operator_is_type(op, "Sign", self._builder):
            return ActivationFunctionType.SIGN_BIT

    def __call__(self) -> bool:
        matcher = PatternMatcher(
            self._builder,
            [
                Op(
                    self.ops_with_fused_activation_function,
                    ["x"],
                    ["x1"],
                    [NoFusedActivationFunction()],
                ),
                Op(self.activation_functions, ["x1"], ["y"]),
            ],
            [TensorHasOneConsumer("x1")],
        )

        to_remove = []
        for [leading_op, act_fun_op], tensor_map, _, _ in matcher.match_patterns():
            builtin_leading_op = leading_op.builtin_options.operator_type
            logger.internal_assert(
                builtin_leading_op in self.supported_activations_for_op.keys(),
                f"FuseActivationFunctions: supported activations for operator `{builtin_leading_op}`"
                "are not known.",
            )

            act_fun = self._act_fun_type_for_op(act_fun_op)
            if act_fun not in self.supported_activations_for_op[builtin_leading_op]:
                # The leading op doesn't support this activation function.
                continue

            x, y = tensor_map["x"], tensor_map["y"]
            if (
                x.quantization != y.quantization
                and builtin_leading_op in self.ops_that_need_equal_io_quantization
            ):
                # The fusion would result in different input and output quantization of `leading_op`, which would cause
                #  runtime issues for that particular operator.
                continue

            leading_op.builtin_options.fused_activation_function = act_fun
            leading_op.tmp_outputs[0] = act_fun_op.tmp_outputs[0]
            to_remove.append(act_fun_op)

        for op in to_remove:
            self._builder.get_operators().remove(op)

        return len(to_remove) != 0
