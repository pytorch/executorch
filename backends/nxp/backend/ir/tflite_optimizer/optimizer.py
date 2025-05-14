#
# Copyright 2023 Martin Pavella
# Copyright 2024 NXP
#
# License: MIT
# See the LICENSE_MIT for more details.
#

from enum import Enum
from typing import Callable

from executorch.backends.nxp.backend.ir import logger
from executorch.backends.nxp.backend.ir.conversion_config import ConversionConfig
from executorch.backends.nxp.backend.ir.tflite_optimizer.optimizations.combine_hard_sigmoid_and_mul_to_hard_swish import (
    CombineHardSigmoidAndMulIntoHardSwish,
)
from executorch.backends.nxp.backend.ir.tflite_optimizer.optimizations.eliminate_dead_branches import (
    EliminateDeadBranches,
)
from executorch.backends.nxp.backend.ir.tflite_optimizer.optimizations.fuse_activation_functions import (
    FuseActivationFunctions,
)
from executorch.backends.nxp.backend.ir.tflite_optimizer.optimizations.fuse_fully_connected_and_add_operators import (
    FuseFullyConnectedAndAddOperators,
)
from executorch.backends.nxp.backend.ir.tflite_optimizer.optimizations.fuse_quanitze_into_preceding_ops import (
    FuseQuantizeIntoPrecedingOps,
)
from executorch.backends.nxp.backend.ir.tflite_optimizer.optimizations.keep_one_empty_buffer import (
    KeepOneEmptyBuffer,
)
from executorch.backends.nxp.backend.ir.tflite_optimizer.optimizations.move_relu_before_concat import (
    MoveActivationBeforeConcatenation,
)
from executorch.backends.nxp.backend.ir.tflite_optimizer.optimizations.permute_fully_connected_weights_after_reshape import (
    PermuteFullyConnectedWeightsAfterReshape,
)
from executorch.backends.nxp.backend.ir.tflite_optimizer.optimizations.prune_cast_operators import (
    FuseCastOperators,
    RemoveCastOperatorsWithNoEffect,
)
from executorch.backends.nxp.backend.ir.tflite_optimizer.optimizations.prune_quantize_operators import (
    FuseParallelQuantizeOperators,
    PruneQuantizeOperators,
)
from executorch.backends.nxp.backend.ir.tflite_optimizer.optimizations.prune_reshape_operators import (
    FuseReshapeOperators,
    RemoveReshapeOperatorsWithNoEffect,
)
from executorch.backends.nxp.backend.ir.tflite_optimizer.optimizations.prune_transpose_operators import (
    FuseTransposeOperators,
    RemoveIdentityTransposeOperators,
)
from executorch.backends.nxp.backend.ir.tflite_optimizer.optimizations.remove_unused_tensors_and_buffers import (
    RemoveUnusedTensorsAndBuffers,
)
from executorch.backends.nxp.backend.ir.tflite_optimizer.optimizations.replace_average_pool_before_fully_connected_with_sum import (
    ReplaceAveragePoolBeforeFullyConnectedWithSum,
)


class Optimization(Enum):
    KEEP_ONE_EMPTY_BUFFER = 0
    FUSE_ACTIVATION_FUNCTIONS = 1
    FUSE_FULLY_CONNECTED_AND_ADD = 2

    FUSE_RESHAPE_OPERATORS = 3
    REMOVE_RESHAPE_OPERATORS_WITH_NO_EFFECT = 4

    FUSE_TRANSPOSE_OPERATORS = 5
    REMOVE_IDENTITY_TRANSPOSE_OPERATORS = 6

    PRUNE_QUANTIZE_OPERATORS = 7
    FUSE_PARALLEL_QUANTIZE_OPERATORS = 8
    FUSE_QUANTIZE_INTO_PRECEDING_OPS = 9

    REMOVE_UNUSED_TENSORS = 10
    ELIMINATE_DEAD_BRANCHES = 11
    PERMUTE_FULLY_CONNECTED_WEIGHTS_AFTER_RESHAPE = 12

    FUSE_CAST_OPERATORS = 13
    REMOVE_CAST_OPERATORS_WITH_NO_EFFECT = 14

    MOVE_ACTIVATION_BEFORE_CONCAT = 15
    COMBINE_HARD_SIGMOID_AND_MUL_INTO_HARD_SWISH = 16
    REPLACE_AVERAGE_POOL_BEFORE_FULLY_CONNECTED_WITH_SUM = 17


class Optimizer:
    """
    Class provides methods to optimize a TFLite model. To do so, it uses a ModelBuilder object, encapsulating
     the TFLite model.

    A lot of these methods were implemented a while ago they are not very efficient. Some of them may also not cover
     all edge cases.
    """

    # avoid circular dependency with importing the model_builder but allow typehints
    _builder: "model_builder.ModelBuilder"  # noqa F821

    # Dictionary which maps optimizations to methods which implement them
    optimization_map: dict[Optimization, Callable]

    # As long as the model is being modified, optimizations will be applied again and again. This variable is the hard
    #  limit to the number of times any single optimization is applied.
    optimization_application_limit = 10  # Empirical value.

    def __init__(
        self,
        builder: "model_builder.ModelBuilder",  # noqa F821
        conversion_config: ConversionConfig,
    ):
        self._builder = builder

        self.optimization_map = {
            Optimization.KEEP_ONE_EMPTY_BUFFER: KeepOneEmptyBuffer(
                builder, conversion_config
            ),
            Optimization.FUSE_ACTIVATION_FUNCTIONS: FuseActivationFunctions(
                builder, conversion_config
            ),
            Optimization.FUSE_FULLY_CONNECTED_AND_ADD: FuseFullyConnectedAndAddOperators(
                builder, conversion_config
            ),
            Optimization.FUSE_RESHAPE_OPERATORS: FuseReshapeOperators(
                builder, conversion_config
            ),
            Optimization.REMOVE_RESHAPE_OPERATORS_WITH_NO_EFFECT: RemoveReshapeOperatorsWithNoEffect(
                builder, conversion_config
            ),
            Optimization.FUSE_TRANSPOSE_OPERATORS: FuseTransposeOperators(
                builder, conversion_config
            ),
            Optimization.REMOVE_IDENTITY_TRANSPOSE_OPERATORS: RemoveIdentityTransposeOperators(
                builder, conversion_config
            ),
            Optimization.PRUNE_QUANTIZE_OPERATORS: PruneQuantizeOperators(
                builder, conversion_config
            ),
            Optimization.FUSE_PARALLEL_QUANTIZE_OPERATORS: FuseParallelQuantizeOperators(
                builder, conversion_config
            ),
            Optimization.FUSE_QUANTIZE_INTO_PRECEDING_OPS: FuseQuantizeIntoPrecedingOps(
                builder, conversion_config
            ),
            Optimization.REMOVE_UNUSED_TENSORS: RemoveUnusedTensorsAndBuffers(
                builder, conversion_config
            ),
            Optimization.ELIMINATE_DEAD_BRANCHES: EliminateDeadBranches(
                builder, conversion_config
            ),
            Optimization.PERMUTE_FULLY_CONNECTED_WEIGHTS_AFTER_RESHAPE: PermuteFullyConnectedWeightsAfterReshape(
                builder, conversion_config
            ),
            Optimization.FUSE_CAST_OPERATORS: FuseCastOperators(
                builder, conversion_config
            ),
            Optimization.REMOVE_CAST_OPERATORS_WITH_NO_EFFECT: RemoveCastOperatorsWithNoEffect(
                builder, conversion_config
            ),
            Optimization.MOVE_ACTIVATION_BEFORE_CONCAT: MoveActivationBeforeConcatenation(
                builder, conversion_config
            ),
            Optimization.COMBINE_HARD_SIGMOID_AND_MUL_INTO_HARD_SWISH: CombineHardSigmoidAndMulIntoHardSwish(
                builder, conversion_config
            ),
            Optimization.REPLACE_AVERAGE_POOL_BEFORE_FULLY_CONNECTED_WITH_SUM: ReplaceAveragePoolBeforeFullyConnectedWithSum(
                builder, conversion_config
            ),
        }

    def optimize(
        self,
        optimization_whitelist: list[Optimization] | None = None,
        optimization_blacklist: list[Optimization] | None = None,
    ):
        """Apply optimizations to the TFLite model encapsulated by 'self._builder'.
        :param optimization_whitelist: A list of optimizations to apply to the model.
        :param optimization_blacklist: A list of optimizations to NOT apply to the model.

        At least one of 'optimization_whitelist' and 'optimization_blacklist' must be 'None'.
        If both are 'None', all optimizations are applied.

        The optimizations will be applied multiple times in a loop, until the model is fully optimized.
        """

        if optimization_whitelist is not None and optimization_blacklist is not None:
            logger.e(
                logger.Code.INVALID_OPTIMIZATION,
                "Optimization whitelist and blacklist cannot both be specified.",
            )

        if optimization_whitelist is not None:
            optimizations = optimization_whitelist
        else:
            # Apply all optimizations
            optimizations = list(Optimization)

        if optimization_blacklist is not None:
            for o in optimization_blacklist:
                try:
                    optimizations.remove(o)
                except ValueError:
                    logger.w(
                        f"Optimization blacklist contains invalid optimization '{o}'."
                    )

        # Execute the optimizations until the model is fully optimized.
        for _i in range(self.optimization_application_limit):
            run_again = False

            for optimization in optimizations:
                if optimization not in self.optimization_map.keys():
                    logger.e(
                        logger.Code.INVALID_OPTIMIZATION,
                        f"The converter doesn't recognise the '{optimization}' optimization.",
                    )

                # Call the optimization
                made_changes = self.optimization_map[optimization]()
                logger.internal_assert(
                    type(made_changes) is bool,
                    f"Optimization `{optimization}` didn't return bool.",
                )
                run_again |= made_changes

            if not run_again:
                # The model is now fully optimized.
                break
