# Copyright 2024 NXP
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from executorch.backends.nxp.backend.ir.lib.tflite.BuiltinOperator import (
    BuiltinOperator,
)
from executorch.backends.nxp.backend.ir.lib.tflite.TensorType import TensorType
from executorch.backends.nxp.backend.ir.tflite_generator import tflite_model
from executorch.backends.nxp.backend.ir.tflite_generator.builtin_options.hard_swish_options import (
    HardSwish,
)
from executorch.backends.nxp.backend.ir.tflite_optimizer.optimizations.base_optimization import (
    BaseOptimization,
)
from executorch.backends.nxp.backend.ir.tflite_optimizer.pattern_matcher import (
    OneOf,
    Op,
    PatternMatcher,
)
from executorch.backends.nxp.backend.ir.tflite_optimizer.tensor_rules import (
    RuleOr,
    TensorHasNConsumers,
    TensorHasStaticValue,
    TensorHasType,
    TensorsAreQuantized,
    TensorsHaveOneConsumer,
    TensorsHaveType,
)


class CombineHardSigmoidAndMulIntoHardSwish(BaseOptimization):

    def __call__(self) -> bool:
        made_changes = self._combine_float_variant()
        made_changes |= self._combine_quantized_variant()

        return made_changes

    def _combine_float_variant(self) -> bool:
        """Fuse some operators in the following pattern. The ops `Mul`, `Add` `Minimum` and `Relu` compute the
        `HardSigmoid` operation, as there is no `HardSigmoid` operator in TFLite.

                      ┌─────┴─────┐  `x`
                   ┌──▼──┐        │
             1/6 ──► Mul │        │
                   └──┬──┘        │
                   ┌──▼──┐        │
             1/2 ──► Add │        │                           │
                   └──┬──┘        │                     ┌─────▼─────┐
                 ┌────▼────┐      │       ─────►        │ HardSwish │
             1 ──► Minimum │      │                     └─────┬─────┘
                 └────┬────┘      │
                   ┌──▼───┐       │
                   │ Relu │       │
                   └──┬───┘       │
                      └───┐   ┌───┘
                         ┌▼───▼┐
                         │ Mul │
                         └──┬──┘
        """

        matcher = PatternMatcher(
            self._builder,
            [
                Op(["Mul"], ["x", "alpha"], ["mul_o"]),
                OneOf(
                    [
                        Op(["Add"], ["mul_o", "beta"], ["add_o"]),
                        Op(["Add"], ["beta", "mul_o"], ["add_o"]),
                    ]
                ),
                OneOf(
                    [
                        Op(["Minimum"], ["add_o", "one"], ["min_o"]),
                        Op(["Minimum"], ["one", "add_o"], ["min_o"]),
                    ]
                ),
                Op(["Relu"], ["min_o"], ["relu_o"]),
                OneOf(
                    [
                        Op(["Mul"], ["x", "relu_o"], ["y"]),
                        Op(["Mul"], ["relu_o", "x"], ["y"]),
                    ]
                ),
            ],
            [
                TensorHasNConsumers("x", 2),
                TensorsHaveOneConsumer(["mul_o", "add_o", "min_o", "relu_o"]),
                TensorHasStaticValue("alpha", 1 / 6),
                TensorHasStaticValue("beta", 0.5),
                TensorHasStaticValue("one", 1),
                # `HardSwishConverter` and `HardSigmoidConverter` both only support float32.
                TensorHasType("x", TensorType.FLOAT32),
            ],
        )

        # The mapped operator (value) will be inserted into the model later, at the position of the `key` operator.
        to_add: dict[tflite_model.Operator, tflite_model.Operator] = {}
        to_remove = []
        for pattern_ops, tensor_map, _, _ in matcher.match_patterns():
            x, y = tensor_map["x"], tensor_map["y"]
            hard_swish = tflite_model.Operator(
                builtin_options=HardSwish(),
                opcode_index=self._builder.op_code_index_for_op_type(
                    BuiltinOperator.HARD_SWISH
                ),
            )
            hard_swish.tmp_inputs = [x]
            hard_swish.tmp_outputs = [y]

            to_add[pattern_ops[0]] = hard_swish

            to_remove.extend(pattern_ops)

        ops = self._builder.get_operators()
        for k, v in to_add.items():
            idx = ops.index(k)
            ops.insert(idx, v)

        for op in to_remove:
            ops.remove(op)

        return len(to_remove) != 0

    def _combine_quantized_variant(self) -> bool:
        """Fuse some operators in the following pattern. The ops `Mul`, `Add` `Minimum` and `Relu` compute the
         `HardSigmoid` operation, as there is no `HardSigmoid` operator in TFLite.

        The following pattern arises from using the `onnx2quant` on a model with `HardSwish`. The quantizer always
         runs a pre-processing step which splits the ONNX `HardSwish` into `HardSigmoid` and `Mul`. It seems like it
         cannot be turned off. Therefore, we cannot add QDQ quantization of `HardSwish`. But since `HardSigmoid`
         gets converted to multiple TFLite operators, we also cannot really add QDQ quantization for that operator.
         This means that `HardSwish` will never get fully quantized by the `onnx2quant`, and the following pattern
         will be created.
        We can, however, convert the entire pattern into a quantized `HardSwish` using this optimization.

                             │  (u)int8    `x`
                       ┌─────▼──────┐
                       │ Dequantize │
                       └─────┬──────┘
                       ┌─────┴─────┐  float32
                    ┌──▼──┐        │
              1/6 ──► Mul │        │
                    └──┬──┘        │
                    ┌──▼──┐        │
              1/2 ──► Add │        │
                    └──┬──┘        │
                  ┌────▼────┐      │
              1 ──► Minimum │      │                           │  (u)int8    `x`
                  └────┬────┘      │                     ┌─────▼─────┐
                    ┌──▼───┐       │       ─────►        │ HardSwish │
                    │ Relu │       │                     └─────┬─────┘
                    └──┬───┘       │                           │  (u)int8    `y`
                  ┌────▼─────┐     │
                  │ Quantize │     │
                  └────┬─────┘     │
                 ┌─────▼──────┐    │
                 │ Dequantize │    │
                 └─────┬──────┘    │
                       └───┐   ┌───┘
                          ┌▼───▼┐
                          │ Mul │
                          └──┬──┘
                             │  float32
                        ┌────▼─────┐
                        │ Quantize │
                        └────┬─────┘
                             │  (u)int8    `y`
        """
        matcher = PatternMatcher(
            self._builder,
            [
                Op(["Dequantize"], ["x"], ["deq1_o"]),
                OneOf(
                    [
                        Op(["Mul"], ["deq1_o", "alpha"], ["mul1_o"]),
                        Op(["Mul"], ["alpha", "deq1_o"], ["mul1_o"]),
                    ]
                ),
                OneOf(
                    [
                        Op(["Add"], ["mul1_o", "beta"], ["add_o"]),
                        Op(["Add"], ["beta", "mul1_o"], ["add_o"]),
                    ]
                ),
                OneOf(
                    [
                        Op(["Minimum"], ["add_o", "one"], ["min_o"]),
                        Op(["Minimum"], ["one", "add_o"], ["min_o"]),
                    ]
                ),
                Op(["Relu"], ["min_o"], ["relu_o"]),
                Op(["Quantize"], ["relu_o"], ["quant1_o"]),
                Op(["Dequantize"], ["quant1_o"], ["deq2_o"]),
                OneOf(
                    [
                        Op(["Mul"], ["deq1_o", "deq2_o"], ["mul2_o"]),
                        Op(["Mul"], ["deq2_o", "deq1_o"], ["mul2_o"]),
                    ]
                ),
                Op(["Quantize"], ["mul2_o"], ["y"]),
            ],
            [
                TensorHasNConsumers("deq1_o", 2),
                TensorsHaveOneConsumer(
                    [
                        "mul1_o",
                        "add_o",
                        "min_o",
                        "relu_o",
                        "quant1_o",
                        "deq2_o",
                        "mul2_o",
                    ]
                ),
                TensorHasStaticValue("alpha", 1 / 6),
                TensorHasStaticValue("beta", 0.5),
                TensorHasStaticValue("one", 1),
                TensorHasType("deq1_o", TensorType.FLOAT32),
                TensorsAreQuantized(["x", "y"]),
                RuleOr(
                    TensorsHaveType(["x", "y"], TensorType.INT8),
                    TensorsHaveType(["x", "y"], TensorType.UINT8),
                ),
            ],
        )

        # The mapped operator (value) will be inserted into the model later, at the position of the `key` operator.
        to_add: dict[tflite_model.Operator, tflite_model.Operator] = {}
        to_remove = []
        for pattern_ops, tensor_map, _, _ in matcher.match_patterns():
            x, y = tensor_map["x"], tensor_map["y"]
            hard_swish = tflite_model.Operator(
                builtin_options=HardSwish(),
                opcode_index=self._builder.op_code_index_for_op_type(
                    BuiltinOperator.HARD_SWISH
                ),
            )
            hard_swish.tmp_inputs = [x]
            hard_swish.tmp_outputs = [y]

            to_add[pattern_ops[0]] = hard_swish

            to_remove.extend(pattern_ops)

        ops = self._builder.get_operators()
        for k, v in to_add.items():
            idx = ops.index(k)
            ops.insert(idx, v)

        for op in to_remove:
            ops.remove(op)

        return len(to_remove) != 0
