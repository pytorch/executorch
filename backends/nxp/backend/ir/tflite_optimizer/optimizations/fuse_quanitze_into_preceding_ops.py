# Copyright 2024 NXP
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from executorch.backends.nxp.backend.ir.tflite_optimizer.operator_rules import WasNotInTheOriginalONNXModel
from executorch.backends.nxp.backend.ir.tflite_optimizer.optimizations.base_optimization import BaseOptimization
from executorch.backends.nxp.backend.ir.tflite_optimizer.pattern_matcher import Op, PatternMatcher
from executorch.backends.nxp.backend.ir.tflite_optimizer.tensor_rules import TensorHasOneConsumer, TensorsArePerTensorQuantized, \
    TensorsHaveSameType


class FuseQuantizeIntoPrecedingOps(BaseOptimization):
    """ Remove some `Quantize` operators in the following pattern.

          │
        ┌─▼──┐
        │ Op │                                                            │
        └─┬──┘                                                          ┌─▼──┐
          │  'x' (same type, quantization params `A`)     ─────►        │ Op │
     ┌────▼─────┐                                                       └─┬──┘
     │ Quantize │                                                         │  (same type, quantization params `B`)
     └────┬─────┘
          │  'y' (same type, quantization params `B`)
    """

    ops_that_can_have_any_output_quantization = [
        # List of operators which don't have restrictions placed on their output quantization and are currently
        #  supported by `onnx2quant`.

        'Add', 'BatchMatMul', 'FullyConnected', 'HardSwish', 'LeakyRelu', 'Mean', 'Mul', 'PRelu', 'ReduceProd',
        'Relu', 'Sub', 'Sum'
    ]

    def __call__(self) -> bool:
        matcher = PatternMatcher(
            self._builder,
            [
                Op(self.ops_that_can_have_any_output_quantization, outputs=[..., 'x', ...]),
                Op(['Quantize'], ['x'], ['y'], [
                    # Restrict this optimization to extra `Quantize` operators which were added during conversion.
                    #  Sometimes the `Quantize` operators which are present in the ONNX model can be essential and
                    #  shouldn't be removed. They can for example perform clipping.
                    WasNotInTheOriginalONNXModel()
                ]),

            ],
            [
                TensorHasOneConsumer('x'),

                # Make sure the `Quantize` is just changing quantization parameters. Otherwise, it couldn't be fused.
                TensorsHaveSameType(['x', 'y']),
                TensorsArePerTensorQuantized(['x', 'y'])
            ])

        to_remove = []
        for [leading_op, quantize], tensor_map, _, _ in matcher.match_patterns():
            x, y = tensor_map['x'], tensor_map['y']

            x_idx = leading_op.tmp_outputs.index(x)
            leading_op.tmp_outputs[x_idx] = y

            to_remove.append(quantize)

        for op in to_remove:
            self._builder.get_operators().remove(op)

        return len(to_remove) != 0
