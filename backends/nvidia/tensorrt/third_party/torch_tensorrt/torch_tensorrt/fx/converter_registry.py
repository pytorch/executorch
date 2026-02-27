import logging
from typing import Any, Callable, Dict

from torch.fx.node import Target

CONVERTERS: Dict[Target, Any] = {}
NO_IMPLICIT_BATCH_DIM_SUPPORT = {}
NO_EXPLICIT_BATCH_DIM_SUPPORT = {}


logger = logging.getLogger(__name__)


def tensorrt_converter(
    key: Target,
    no_implicit_batch_dim: bool = False,
    no_explicit_batch_dim: bool = False,
    enabled: bool = True,
) -> Callable[[Any], Any]:
    def register_converter(converter):
        CONVERTERS[key] = converter
        if no_implicit_batch_dim:
            NO_IMPLICIT_BATCH_DIM_SUPPORT[key] = converter
        if no_explicit_batch_dim:
            NO_EXPLICIT_BATCH_DIM_SUPPORT[key] = converter

        logger.debug(
            f"Converter for {key} added to FX Converter Registry "
            + f"{'without' if no_explicit_batch_dim else 'with'} Explicit Batch Dim Support + "
            + f"{'without' if no_implicit_batch_dim else 'with'} Implicit Batch Dim Support"
        )

        return converter

    def disable_converter(converter):
        return converter

    if enabled:
        return register_converter
    else:
        return disable_converter
