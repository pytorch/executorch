#
# Copyright 2023 Martin Pavella
# Copyright 2024-2026 NXP
#
# License: MIT
# See the LICENSE_MIT for more details.
#

from enum import Enum
from typing import Callable

from executorch.backends.nxp.backend.ir import logger
from executorch.backends.nxp.backend.ir.conversion_config import ConversionConfig
from executorch.backends.nxp.backend.ir.neutron_ir_post_processing.optimizations.permute_fully_connected_weights_after_reshape import (
    PermuteFullyConnectedWeightsAfterReshape,
)
from executorch.backends.nxp.backend.ir.neutron_ir_post_processing.optimizations.prune_transpose_operators import (
    FuseTransposeOperators,
    RemoveIdentityTransposeOperators,
)
from executorch.backends.nxp.backend.neutron_target_spec import NeutronTargetSpec


class Optimization(Enum):
    FUSE_TRANSPOSE_OPERATORS = 5
    REMOVE_IDENTITY_TRANSPOSE_OPERATORS = 6

    PERMUTE_FULLY_CONNECTED_WEIGHTS_AFTER_RESHAPE = 12


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
        neutron_target_spec: NeutronTargetSpec,
    ):
        self._builder = builder

        self.optimization_map = {
            Optimization.FUSE_TRANSPOSE_OPERATORS: FuseTransposeOperators(
                builder, conversion_config, neutron_target_spec
            ),
            Optimization.REMOVE_IDENTITY_TRANSPOSE_OPERATORS: RemoveIdentityTransposeOperators(
                builder, conversion_config, neutron_target_spec
            ),
            Optimization.PERMUTE_FULLY_CONNECTED_WEIGHTS_AFTER_RESHAPE: PermuteFullyConnectedWeightsAfterReshape(
                builder, conversion_config, neutron_target_spec
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
