# Copyright 2024-2026 NXP
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import numpy as np

from executorch.backends.nxp.backend.ir import logger
from executorch.backends.nxp.backend.ir.converter.conversion import translator
from executorch.backends.nxp.backend.ir.neutron_ir_post_processing.optimizations.base_optimization import (
    BaseOptimization,
)
from executorch.backends.nxp.backend.ir.neutron_ir_post_processing.pattern_matcher import (
    Op,
    PatternMatcher,
)
from executorch.backends.nxp.backend.ir.neutron_ir_post_processing.tensor_rules import (
    TensorDimensionsMatch,
    TensorHasRank,
    TensorIsChannelsFirst,
    TensorIsChannelsLast,
    TensorIsFormatless,
    TensorsHaveData,
    TensorsHaveOneConsumer,
)


class PermuteFullyConnectedWeightsAfterReshape(BaseOptimization):

    def __call__(self) -> bool:
        """Search for the pattern:

                       │  (3D / 4D / 5D, channels last)
                ┌──────▼──────┐
                │  Transpose  │
                └──────┬──────┘
                       │  (3D / 4D / 5D, channels first)
                 ┌─────▼─────┐
                 │  Reshape  │
                 └─────┬─────┘
                       │  (2D, formatless)
              ┌────────▼───────┐
              │ FullyConnected ◄───── Weights  (static)
              └────────┬───────┘
                       │  (2D, formatless)
                       ▼

        In this case, it is possible to permute the `weights` of the `FullyConnected`, and remove the `Transpose`.

        How it works:
            - The original model doesn't have the `Transpose`. It just has `Reshape` into `MatMul` (or `Gemm`...).
            - The `Transpose` is added, because the `Reshape` has a channels last input, which was originally
                channels first (in the ExecuTorch model), and so the 2D output of the `Reshape` would have the same data.
                but at different locations. The `Transpose` makes the input channels first, which ensures correct
                output of the `Reshape`.
            - In the scenario in the graph above, it is possible to omit the `Transpose`, which causes the `Reshape`
                output to be "permuted", and then the `weights` of the `FullyConnected` can be statically permuted
                to match. This will result in correct `FullyConnected` output.
            - It is required that the `Reshape` output has shape [N, H * W * ... * C] (if the input was
                [N, H, W, ..., C]). The `weights` will have shape [X, C * H * W * ...] (where X is arbitrary).
                Since we know the values of C, H, W, ..., we can statically reshape the `weights` to
                [X, C, H, W, ...], transpose it to [X, H, W, ..., C], and flatten it back to [X, H * W * ... * C].
        """

        matcher = PatternMatcher(
            self._builder,
            [
                Op(["Transpose"], ["x", "perm"], ["y"]),
                Op(["Reshape"], ["y", ...], ["z"]),
                Op(["FullyConnected"], ["z", "w", ...]),
            ],
            [
                TensorsHaveOneConsumer(["y", "z"]),
                TensorDimensionsMatch("y", 0, "z", 0),
                TensorDimensionsMatch("z", 1, "w", 1),
                TensorIsChannelsLast("x"),
                TensorIsChannelsFirst("y"),
                TensorIsFormatless("z"),
                TensorHasRank("z", 2),
                TensorsHaveData(["perm", "w"]),
            ],
        )

        to_remove = []
        for (transpose, reshape, fc), tensor_map, _, _ in matcher.match_patterns():
            # Make sure the `Transpose` is applying the expected permutation.
            y = tensor_map["y"]
            to_executorch_perm = (
                translator.create_channels_last_to_channels_first_permutation(
                    y.shape.len()
                )
            )
            if not np.allclose(to_executorch_perm, tensor_map["perm"].tmp_buffer.data):
                continue  # The `Transpose` has an unexpected permutation.

            w = tensor_map["w"]
            tmp_shape = [w.shape[0]] + y.shape[1:]  # H, W, C

            data = w.tmp_buffer.data.reshape(tmp_shape)  # Reshape from 2D.
            data = translator.convert_data_to_channels_last(
                data
            )  # Permute to TFLite format.
            data = data.reshape(w.shape.vector)  # Flatten to 2D.

            # Create a new tensor for the data, in case it is used by some other operator as well.
            new_weights = self._builder.duplicate_tensor(w)
            new_weights.tmp_buffer.data = data
            fc.tmp_inputs[1] = new_weights

            # Remove the `Transpose`.
            logger.i(
                f"Permuting the `weights`({w.name}) of a FullyConnected operator and removing an artificial "
                "Transpose operator."
            )
            reshape.tmp_inputs[0] = transpose.tmp_inputs[0]
            to_remove.append(transpose)

        for op in to_remove:
            self._builder.get_operators().remove(op)

        return len(to_remove) != 0
