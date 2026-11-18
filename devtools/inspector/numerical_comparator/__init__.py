# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.


# Re-export DebugHandle from _inspector_utils for convenience
from executorch.devtools.inspector._inspector_utils import DebugHandle
from executorch.devtools.inspector.numerical_comparator.l1_numerical_comparator import (
    L1Comparator,
)

from executorch.devtools.inspector.numerical_comparator.mse_numerical_comparator import (
    MSEComparator,
)

from executorch.devtools.inspector.numerical_comparator.numerical_comparator_base import (
    IntermediateOutputMapping,
    NumericalComparatorBase,
)

from executorch.devtools.inspector.numerical_comparator.snr_numerical_comparator import (
    SNRComparator,
)


# The metric names Inspector.calculate_numeric_gap accepts as `distance`.
_COMPARATOR_BY_NAME: dict[str, type[NumericalComparatorBase]] = {
    "L1": L1Comparator,
    "MSE": MSEComparator,
    "SNR": SNRComparator,
}


def comparator_class_for_metric(name: str) -> type[NumericalComparatorBase]:
    """Resolve a built-in metric name to its comparator class."""
    comparator_cls = _COMPARATOR_BY_NAME.get(name.strip().upper())
    if comparator_cls is None:
        raise ValueError(
            f"Unsupported metric {name!r}; expected one of "
            f"{sorted(_COMPARATOR_BY_NAME)}"
        )
    return comparator_cls


__all__ = [
    "comparator_class_for_metric",
    "DebugHandle",
    "IntermediateOutputMapping",
    "L1Comparator",
    "MSEComparator",
    "NumericalComparatorBase",
    "SNRComparator",
]
