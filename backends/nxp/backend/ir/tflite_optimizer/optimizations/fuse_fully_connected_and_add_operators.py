# Copyright 2024 NXP
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from executorch.backends.nxp.backend.ir.lib.tflite.TensorType import TensorType
from executorch.backends.nxp.backend.ir.tflite_optimizer.operator_rules import (
    NoFusedActivationFunction,
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
    RuleAnd,
    RuleIf,
    RuleOr,
    TensorDimensionsMatch,
    TensorHasDimensionOfSize,
    TensorHasOneConsumer,
    TensorHasRank,
    TensorHasType,
    TensorIsQuantized,
)


class FuseFullyConnectedAndAddOperators(BaseOptimization):

    def __call__(self) -> bool:
        """
        FullyConnected -> Add sequence can handle more complicated shapes than just FullyConnected with bias
         (due to shape broadcasting).
        The bias can have shape [N] or [1, N], where N is the first dimension of the FC weights tensor.
         It could also have shape [1, ..., 1, N], but then the TFLite FullyConnected removes the leading ones,
         even if 'keep_num_dims' is True. In ONNX, the output tensor has the leading ones,
         In this case, a Reshape would have to be added, so we do not perform the fusion.

        # https://github.com/tensorflow/tensorflow/blob/v2.15.0/tensorflow/lite/kernels/fully_connected.cc#L398
        """
        matcher = PatternMatcher(
            self._builder,
            [
                # Require exactly 2 inputs.
                Op(
                    ["FullyConnected"], ["x", "w"], ["y"], [NoFusedActivationFunction()]
                ),
                OneOf([Op(["Add"], ["y", "b"]), Op(["Add"], ["b", "y"])]),
            ],
            [
                TensorHasOneConsumer("y"),
                TensorHasRank("w", 2),
                RuleOr(
                    TensorHasRank("b", 1),
                    RuleAnd(TensorHasRank("b", 2), TensorHasDimensionOfSize("b", 0, 1)),
                ),
                TensorDimensionsMatch("w", 0, "b", -1),
                RuleIf(TensorIsQuantized("x"), TensorHasType("b", TensorType.INT32)),
            ],
        )

        to_remove = []
        for (fc, add), tensor_map, _, _ in matcher.match_patterns():
            b = tensor_map["b"]
            fc.tmp_inputs.append(b)

            # Remove the 'Add' operator.
            fc.tmp_outputs[0] = add.tmp_outputs[0]
            fc.builtin_options.fused_activation_function = (
                add.builtin_options.fused_activation_function
            )
            to_remove.append(add)

        for op in to_remove:
            self._builder.get_operators().remove(op)

        return len(to_remove) != 0
