# Copyright 2024 NXP
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import numpy as np

from executorch.backends.nxp.backend.ir.converter.conversion.translator import (
    apply_permutation_to,
    combine_permutations,
)
from executorch.backends.nxp.backend.ir.tflite_generator import tflite_model
from executorch.backends.nxp.backend.ir.tflite_optimizer.optimizations.base_optimization import (
    BaseOptimization,
)
from executorch.backends.nxp.backend.ir.tflite_optimizer.pattern_matcher import (
    MultipleSameOps,
    Op,
    PatternMatcher,
)
from executorch.backends.nxp.backend.ir.tflite_optimizer.tensor_rules import (
    RuleOr,
    TensorHasData,
    TensorIsNotModelOutput,
    TensorsHaveData,
)


class FuseTransposeOperators(BaseOptimization):
    """Remove some `Transpose` operators in the following pattern.

              │  'x'
        ┌─────▼─────┐
        │ Transpose │
        └─────┬─────┘                                          │  'x'
          ┌───┴──── ... ────────┐  'y'      ─────►         ┌───┴──── ... ────────┐   ('y' is not in the model anymore)
    ┌─────▼─────┐         ┌─────▼─────┐              ┌─────▼─────┐         ┌─────▼─────┐
    │ Transpose │   ...   │ Transpose │              │ Transpose │   ...   │ Transpose │
    └─────┬─────┘         └─────┬─────┘              └─────┬─────┘         └─────┬─────┘
          │                     │  'z'                     │                     │  'z'
    """

    def __call__(self) -> bool:
        matcher = PatternMatcher(
            self._builder,
            [
                Op(["Transpose"], ["x", "perm1"], ["y"]),
                MultipleSameOps(
                    ["Transpose"], ["y", "perm2"]
                ),  # Nothing other than `Transpose` ops can use `y`.
            ],
            [TensorsHaveData(["perm1", "perm2"]), TensorIsNotModelOutput("y")],
        )

        to_remove = []
        for (
            [leading_transpose, following_transposes],
            tensor_map,
            _,
            _,
        ) in matcher.match_patterns():
            x = tensor_map["x"]
            perm1 = tensor_map["perm1"].tmp_buffer.data

            # Remove the leading transpose.
            for second_transpose in following_transposes:
                # Combine the permutations for a new permutation of the second `Transpose`.
                perm2 = second_transpose.tmp_inputs[1].tmp_buffer.data
                combined_perm = np.array(combine_permutations(perm1, perm2), np.int32)
                second_transpose.tmp_inputs[1] = self._builder.create_tensor_for_data(
                    combined_perm, "perm"
                )

                # Compute the output shape of the second `Transpose`.
                new_output_shape = apply_permutation_to(x.shape.vector, combined_perm)
                second_transpose.tmp_outputs[0].shape = tflite_model.Shape(
                    list(new_output_shape)
                )

                # Bypass the first `Transpose`.
                second_transpose.tmp_inputs[0] = leading_transpose.tmp_inputs[0]

            to_remove.append(leading_transpose)

        for op in to_remove:
            self._builder.get_operators().remove(op)

        return len(to_remove) != 0


class RemoveIdentityTransposeOperators(BaseOptimization):
    """Remove operators that match the following pattern.

          │  'x'
    ┌─────▼─────┐
    │ Transpose ◄───── identity permutation
    └─────┬─────┘
          │  'y'
    """

    def __call__(self) -> bool:
        matcher = PatternMatcher(
            self._builder,
            [Op(["Transpose"], ["x", "perm"], ["y"])],
            [
                TensorHasData(
                    "perm"
                ),  # Note: identity permutation must be checked later.
                RuleOr(
                    TensorIsNotModelOutput("x"),
                    TensorIsNotModelOutput("y"),
                    # If both 'x' and 'y' are model outputs, the `Transpose` cannot be removed. If the op was removed,
                    #  its input and output would be combined into 1 tensor, which would have to represent 2 model
                    #  outputs with 2 different names, which is not possible.
                ),
            ],
        )

        to_remove = []
        for [transpose], tensor_map, input_to_ops, _ in matcher.match_patterns():
            if not self._builder.operator_can_be_skipped(transpose):
                continue

            x = tensor_map["x"]
            y = tensor_map["y"]

            # Check if the `Transpose` is doing nothing.
            permutation = tensor_map["perm"].tmp_buffer.data
            if not np.allclose(permutation, range(x.rank)):
                # Not and identity permutation.
                continue

            model_outputs = self._builder.get_sub_graph().outputs.tmp_outputs

            # Replace `y` with `x` in the inputs of all following operators.
            following_ops = input_to_ops.get(y.name, [])
            for op in following_ops:
                while y in op.tmp_inputs:
                    input_idx = op.tmp_inputs.index(y)
                    op.tmp_inputs[input_idx] = x

            if y in model_outputs:
                # Replace the output as well.
                while y in model_outputs:
                    idx = model_outputs.index(y)
                    model_outputs[idx] = x

                self._builder.swap_tensor_names(x, y)

            to_remove.append(transpose)

        for op in to_remove:
            self._builder.get_operators().remove(op)

        return len(to_remove) != 0
