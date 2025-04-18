# Copyright 2024 NXP
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import numpy as np

from executorch.backends.nxp.backend.ir.lib.tflite.TensorType import TensorType
from executorch.backends.nxp.backend.ir.tflite_generator import tflite_model
from executorch.backends.nxp.backend.ir.tflite_optimizer.optimizations.base_optimization import BaseOptimization
from executorch.backends.nxp.backend.ir.tflite_optimizer.pattern_matcher import MultipleSameOps, Op, PatternMatcher
from executorch.backends.nxp.backend.ir.tflite_optimizer.tensor_rules import TensorIsNotModelOutput, TensorsHaveSameQuantization


class FuseParallelQuantizeOperators(BaseOptimization):
    """ Fuse some `Quantize` operators in the following pattern.

              │  'x'                                               │  'x'
          ┌───┴──── ... ───────┐                              ┌────▼─────┐
     ┌────▼─────┐        ┌────▼─────┐         ─────►          │ Quantize │
     │ Quantize │   ...  │ Quantize │                         └────┬─────┘
     └────┬─────┘        └────┬─────┘                          ┌───┴─ ... ─┐
          │                   │  'y' (same quantization)       │           │  'y'


        The pattern below only has 2 `Quantize` operators. But the `PatternMatcher` will gradually match all parallel
         `Quantize` operators which fit the pattern above, and remove the unnecessary ones.
     """

    def __call__(self) -> bool:
        matcher = PatternMatcher(
            self._builder,
            [
                Op(['Quantize'], ['x'], ['y1']),
                Op(['Quantize'], ['x'], ['y2'])
            ], [
                TensorsHaveSameQuantization(['y1', 'y2']),

                # 'y2' will be removed from the model, so it cannot be a model output. But thanks to the nature of the
                #  `PatternMatcher`, it doesn't matter which `Quantize` produces the model output. The `PatternMatcher`
                #  will first match 1 `Quantize` as the first `Op` and try to optimize. If it doesn't work, it will then
                #  match the second `Quantize` operator with the first `Op` and try to optimize that way. This will
                #  result in a perfectly optimized pattern every time.
                TensorIsNotModelOutput('y2')
            ]
        )

        to_remove = []
        for [_, quant_to_remove], tensor_map, input_to_ops, _ in matcher.match_patterns():
            to_remove.append(quant_to_remove)

            y1 = tensor_map['y1']
            y2 = tensor_map['y2']
            next_ops = input_to_ops.get(y2.name, [])
            for next_op in next_ops:
                while y2 in next_op.tmp_inputs:
                    idx = next_op.tmp_inputs.index(y2)
                    next_op.tmp_inputs[idx] = y1

            quant_to_remove.tmp_inputs = []  # To prevent future matches of this operator.

        ops = self._builder.get_operators()
        for op in to_remove:
            ops.remove(op)

        return ops.len() != 0


# noinspection PyMethodMayBeStatic
class PruneQuantizeOperators(BaseOptimization):
    """ Remove some `Quantize` operators in the following pattern.

              │  'x'
         ┌────▼─────┐
         │ Quantize │
         └────┬─────┘
          ┌───┴──── ... ───────┐  'y'
     ┌────▼─────┐        ┌────▼─────┐
     │ Quantize │   ...  │ Quantize │
     └────┬─────┘        └────┬─────┘
          │                   │  'z'
     """

    def __call__(self) -> bool:
        matcher = PatternMatcher(
            self._builder,
            [
                Op(['Quantize'], ['x'], ['y']),
                MultipleSameOps(['Quantize'], ['y'])  # Nothing other than `Quantize` ops can use `y`.
            ],
            [
                TensorIsNotModelOutput('y')
            ]
        )

        to_remove = []
        for [leading_quantize, following_quantize_ops], tensor_map, input_to_ops, _ in matcher.match_patterns():
            x = tensor_map['x']

            if self._is_quantization_recasting_from_float(x, following_quantize_ops):
                # First Quantize can be skipped because it does only recasting
                to_remove.append(leading_quantize)

                for next_quantize in following_quantize_ops:
                    next_quantize.tmp_inputs[0] = x

            elif self._is_quantization_recasting_from_integer(x, following_quantize_ops):
                # The Quantize ops negate each other -> remove them both
                to_remove.append(leading_quantize)

                graph_outputs = self._builder.get_sub_graph().outputs.tmp_outputs
                for next_quantize in following_quantize_ops:
                    to_remove.append(next_quantize)

                    # Replace the output of the next Quantize with the input of the first Quantize
                    next_quantize_output = next_quantize.tmp_outputs[0]
                    self._bypass_to_next_quantize_ops(input_to_ops, next_quantize_output, x)

                    # If the output of the first Quantize is also the graph output -> replace the graph output too
                    if next_quantize_output in graph_outputs:
                        graph_outputs.remove(next_quantize_output)
                        if x not in graph_outputs:
                            graph_outputs.append(x)

        for op in to_remove:
            self._builder.get_operators().remove(op)

        return len(to_remove) != 0

    def _is_quantization_recasting_from_float(self, quantize_input: tflite_model.Tensor,
                                              next_ops: list[tflite_model.Operator]):
        """
        Check if 'next_ops' just recast from one type to another. Scale + recalculated zp
        must be the same for all nodes. Input of first Quantize op has to be float to match
        criteria.

        float         int8         uint8
        ----> [quant] ---> [quant] ----->
                       zp          zp-128
        OR

        float         uint8          int8
        ----> [quant] ----> [quant] ----->
                       zp           zp+128

        OR (forked variant with similar restrictions as mentioned above)

                           u/int         u/int
        float            / ----> [quant] ---->
        ----> [quant] --|
                         \ ----> [quant] ---->
                           u/int         u/int

        :param quantize_input: Input tensor of first QuantizeLinear node.
        :param next_ops: QuantizeLinear ops that consume output of 'quantize_input'.
        :return: True if pattern with recasting is found.
        """

        if not quantize_input.type == TensorType.FLOAT32:
            return False

        # All 'next_ops' has the same output type and q-params
        next_op_output_match_first = [self._same_type_and_quantization(
            next_ops[0].tmp_outputs[0], next_op.tmp_outputs[0]) for next_op in next_ops]
        if not all(next_op_output_match_first):
            return False

        # All 'next_ops' are the same, do some additional checks on the first one

        next_op_input = next_ops[0].tmp_inputs[0]
        next_op_output = next_ops[0].tmp_outputs[0]

        input_zp = next_op_input.quantization.zero_point.vector
        output_zp = next_op_output.quantization.zero_point.vector

        if next_op_input.quantization.scale != next_op_output.quantization.scale:
            return False

        if next_op_input.type == TensorType.INT8 and next_op_output.type == TensorType.UINT8:
            return np.equal(input_zp, np.array(output_zp) - 128)
        elif next_op_input.type == TensorType.UINT8 and next_op_output.type == TensorType.INT8:
            return np.equal(input_zp, np.array(output_zp) + 128)

        return False

    def _is_quantization_recasting_from_integer(self, quantize_input: tflite_model.Tensor,
                                                next_ops: list[tflite_model.Operator]):
        """
        Check if 'next_ops' just recast from one type to another. Scale + recalculated zp
        must be the same for all nodes. Input of first Quantize op has to be (u)int8 to
        match criteria.

        uint8          int8          uint8
        ----> [quant] -----> [quant] ---->
         zp           zp+128          zp

        OR

        int8         uint8          int8
        ---> [quant] -----> [quant] --->
         zp          zp-128          zp

        OR (forked variant with similar restrictions as mentioned above)

                           u/int         u/int
        u/int            / ----> [quant] ---->
        ----> [quant] --|
                         \ ----> [quant] ---->
                           u/int         u/int

        :param quantize_input: Input tensor of first QuantizeLinear node.
        :param next_ops: QuantizeLinear ops that consume output of 'quantize_input'.
        :return: True if pattern with recasting is found.
        """

        if quantize_input.type not in [TensorType.INT8, TensorType.UINT8]:
            return False

        # All 'next_ops' has the same output type and q-params as input of first Quantize
        next_op_output_match_first = [self._same_type_and_quantization(
            quantize_input, next_op.tmp_outputs[0]) for next_op in next_ops]
        if not all(next_op_output_match_first):
            return False

        # All 'next_ops' are the same, do some additional checks on the first one

        next_op_input = next_ops[0].tmp_inputs[0]
        next_op_output = next_ops[0].tmp_outputs[0]

        input_zp = next_op_input.quantization.zero_point.vector
        output_zp = next_op_output.quantization.zero_point.vector

        if quantize_input.quantization.scale != next_op_input.quantization.scale:
            return False

        if next_op_input.quantization.scale != next_op_output.quantization.scale:
            return False

        if next_op_input.type == TensorType.INT8 and next_op_output.type == TensorType.UINT8:
            return np.equal(input_zp, np.array(output_zp) - 128)
        elif next_op_input.type == TensorType.UINT8 and next_op_output.type == TensorType.INT8:
            return np.equal(input_zp, np.array(output_zp) + 128)

        return False

    def _same_type_and_quantization(self, a: tflite_model.Tensor, b: tflite_model.Tensor):
        same_type = a.type == b.type
        same_quantization = a.quantization == b.quantization

        return same_type and same_quantization

    def _bypass_to_next_quantize_ops(self, input_to_ops, next_quantize_output, quantize_input):
        ops_after_next_quantize = input_to_ops.get(next_quantize_output.name, [])
        for op_after_next_quantize in ops_after_next_quantize:
            for index, input_tensor in enumerate(op_after_next_quantize.tmp_inputs):
                if input_tensor == next_quantize_output:
                    # Replace the input
                    op_after_next_quantize.tmp_inputs[index] = quantize_input
