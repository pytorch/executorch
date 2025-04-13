# Copyright 2024 NXP
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import numpy as np

from executorch.backends.nxp.backend.ir.lib.tflite.BuiltinOperator import BuiltinOperator
from executorch.backends.nxp.backend.ir.lib.tflite.TensorType import TensorType
from executorch.backends.nxp.backend.ir import logger
from executorch.backends.nxp.backend.ir.tflite_generator import tflite_model
from executorch.backends.nxp.backend.ir.tflite_generator.builtin_options.sum_options import Sum
from executorch.backends.nxp.backend.ir.tflite_optimizer.optimizations.base_optimization import BaseOptimization
from executorch.backends.nxp.backend.ir.tflite_optimizer.pattern_matcher import Op, PatternMatcher
from executorch.backends.nxp.backend.ir.tflite_optimizer.tensor_rules import RuleOr, TensorDimensionsMatch, TensorHasData, TensorHasRank, \
    TensorIsChannelsLast, TensorIsFormatless, TensorsAreQuantized, TensorsHaveOneConsumer, TensorsHaveType


class ReplaceAveragePoolBeforeFullyConnectedWithSum(BaseOptimization):
    """ Replace `AveragePool2D` and `Reshape` with `Sum` in the following pattern.
                       │
              ┌────────▼────────┐
              │  AveragePool2D  │  (global kernel)                          │
              └────────┬────────┘                                       ┌───▼───┐
                       │  (4D, channels last)                           │  Sum  │
                 ┌─────▼─────┐                                          └───┬───┘
                 │  Reshape  │                          ─────►              │
                 └─────┬─────┘                                     ┌────────▼─────────┐
                       │  (2D, formatless)                         │  FullyConnected  ◄───── Scaled weights
              ┌────────▼───────┐                                   └────────┬─────────┘
              │ FullyConnected ◄───── Weights  (static)
              └────────┬───────┘
                       │

        This is possible if the `AveragePool2D` is pooling across the entire input (i.e. global AveragePool). In this
         case, it is possible to use a `Sum` operator instead, and then statically divide the `weights` of the
         `FullyConnected`. This will effectively compute the average across the input at runtime.
        This replacement becomes useful when there is a `Reshape` between, which flattens the tensor to 2D. This
         flattening can be done by the `Sum` operator as well (parameter `keep_dims=False`).
        As a result, the `Reshape` must simply remove the `1`s in the spatial dimensions, and keep the `batch size` and
         `channels` unchanged.
    """

    def __call__(self) -> bool:
        matcher = PatternMatcher(
            self._builder,
            [
                Op(['AveragePool2D'], ['x'], ['ap_out']),
                Op(['Reshape'], ['ap_out', ...], ['resh_out']),
                Op(['FullyConnected'], ['resh_out', 'w', ...], ['y'])
            ],
            [
                # Require either float32, or quantized tensors.
                RuleOr(
                    TensorsHaveType(['w', 'resh_out'], TensorType.FLOAT32),
                    TensorsAreQuantized(['w', 'resh_out'])
                ),
                TensorsHaveOneConsumer(['x', 'ap_out', 'resh_out']),
                TensorIsChannelsLast('ap_out'),
                TensorHasRank('resh_out', 2),
                TensorIsFormatless('resh_out'),
                TensorHasRank('w', 2),
                TensorHasData('w'),
                TensorDimensionsMatch('ap_out', 0, 'resh_out', 0),  # Batch size unchanged.
                TensorDimensionsMatch('ap_out', -1, 'resh_out', -1)  # Channels unchanged.
            ])

        # The mapped operator (value) will later be added into the TFLite model, in front of the `key` operator.
        to_add: dict[tflite_model.Operator, tflite_model.Operator] = dict()
        to_remove = []
        for [ap, reshape, fc], tensor_map, _, _ in matcher.match_patterns():
            x, resh_out, w = tensor_map['x'], tensor_map['resh_out'], tensor_map['w']

            kernel_shape = [ap.builtin_options.filter_h, ap.builtin_options.filter_w]
            if kernel_shape != x.shape[1:3]:
                continue  # Not a global average pool.

            # Divide the static FullyConnected weights by the number of kernel elements. This will transform the `sums`
            #  to `averages` at runtime.
            num_kernel_elements = np.prod(kernel_shape).astype('float32')
            new_w = self._builder.duplicate_tensor(w)
            if w.type == TensorType.FLOAT32:
                # Just divide the weights.
                new_w.tmp_buffer.data = np.array(new_w.tmp_buffer.data / num_kernel_elements).astype('float32')

            elif w.quantization is not None:
                # Divide the `scale` quantization parameter instead of the data. Since the `weights` are static,
                #  changing the `scale` will change the actual values represented by the quantized data. This is because
                #  the scale changes, while the raw data remains exactly the same.
                new_w.quantization.scale.vector = [s / num_kernel_elements for s in new_w.quantization.scale.vector]

                # Since the output of the `Sum` will now contain the `sums` of its input and not the `averages`, its
                #  `scale` quantization parameter is not ideal. Multiply the `scale` by the number of elements of the
                #  kernel to maintain the same accuracy.
                resh_out.quantization.scale.vector = [s * num_kernel_elements for s
                                                      in resh_out.quantization.scale.vector]

            else:
                # Should never happen. Raise an exception to notify us just in case.
                logger.e(logger.Code.INTERNAL_ERROR, 'ReplaceAveragePoolBeforeFullyConnectedWithSum: Unexpected type.')

            fc.tmp_inputs[1] = new_w  # Replace the scaled `weights` of the `FullyConnected`.

            # Reduce over the spatial dimensions.
            axes = self._builder.create_tensor_for_data(np.array([1, 2], 'int32'), 'axes')

            sum_op = tflite_model.Operator(
                builtin_options=Sum(keep_dims=False),
                opcode_index=self._builder.op_code_index_for_op_type(BuiltinOperator.SUM)
            )
            sum_op.tmp_inputs = [x, axes]
            sum_op.tmp_outputs = [resh_out]

            to_add[fc] = sum_op
            to_remove.extend([ap, reshape])

        # Add the new `Sum` operators into the model.
        ops = self._builder.get_operators()
        for k, sum_op in to_add.items():
            idx = ops.index(k)
            ops.insert(idx, sum_op)

        # Remove the `AveragePool` and `Reshape` operators from the model.
        for op in to_remove:
            ops.remove(op)

        return len(to_remove) != 0
