# Copyright 2024 NXP
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from collections import defaultdict
from copy import deepcopy

from executorch.backends.nxp.backend.ir.tflite_generator import tflite_model
from executorch.backends.nxp.backend.ir.tflite_optimizer.operator_rules import (
    AllInputsComeFrom,
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
    TensorsHaveSameQuantization,
)


class MoveActivationBeforeConcatenation(BaseOptimization):
    """
    Move some operators around in the following pattern.
    This is a common pattern that emerges from the conversion of separable convolutions.

          │                │                            │                │
      ┌───▼────┐       ┌───▼────┐                   ┌───▼────┐       ┌───▼────┐
      │ Conv2D │  ...  │ Conv2D │                   │ Conv2D │  ...  │ Conv2D │
      └───┬────┘       └───┬────┘                   └───┬────┘       └───┬────┘
          └──┐          ┌──┘                            │                │
          ┌──▼──────────▼─┐                          ┌──▼───┐         ┌──▼───┐
          │ Concatenation │           ─────►         │ Relu │   ...   │ Relu │
          └───────┬───────┘                          └──┬───┘         └──┬───┘
                  │  'x'                                └──┐          ┌──┘
               ┌──▼───┐                                 ┌──▼──────────▼─┐
               │ Relu │                                 │ Concatenation │
               └──┬───┘                                 └───────┬───────┘
                  │  'y'                                        │
    """

    activations = ["Relu", "ReluN1To1", "Relu6", "Tanh", "Sign"]

    def __call__(self) -> bool:
        matcher = PatternMatcher(
            self._builder,
            [
                Op(["Concatenation"], None, ["x"], [AllInputsComeFrom("Conv2D")]),
                Op(self.activations, ["x"], ["y"]),
            ],
            [
                TensorHasOneConsumer("x"),
                # If the activation function is not changing the quantization parameters, it can be moved without
                #  messing with the quantization elsewhere.
                TensorsHaveSameQuantization(["x", "y"]),
            ],
        )

        to_remove = []

        # Mapping an operator to a list of operators. These operators (value) will later be added into the TFLite
        #  model's `operators` in front of the specified operator (key).
        to_add: dict[tflite_model.Operator, list[tflite_model.Operator]] = defaultdict(
            lambda: []
        )

        for [concat, activation], _, _, _ in matcher.match_patterns():
            new_concat_inputs = []
            for concat_input in concat.tmp_inputs:
                # Create a new operator for the activation function.
                new_activation = deepcopy(activation)
                new_activation.tmp_inputs = [concat_input]
                new_activation_output = self._builder.duplicate_tensor(concat_input)
                new_activation.tmp_outputs = [new_activation_output]

                to_add[concat].append(
                    new_activation
                )  # Insert the new activation into the model later.

                new_concat_inputs.append(
                    new_activation_output
                )  # Connect the activation with the `Concatenation`.

            concat.tmp_inputs = new_concat_inputs

            # Tensor rule ensures that only the activation functions is using the output of the `Concatenation`.
            # It is safe to bypass.
            concat.tmp_outputs[0] = activation.tmp_outputs[0]
            to_remove.append(activation)

        operators = self._builder.get_operators()

        # Add the new activations into the model.
        for concat, activations in to_add.items():
            idx = operators.index(concat)
            for activation in activations:
                operators.insert(idx, activation)

        # Remove the old activations.
        for activation in to_remove:
            operators.remove(activation)

        return len(to_remove) != 0
