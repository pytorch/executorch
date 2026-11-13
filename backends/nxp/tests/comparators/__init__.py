# Copyright 2026 NXP
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# Re-exported so every comparator module is referenced in-tree and shipped in the non-editable wheel.
from executorch.backends.nxp.tests.comparators.all_close_comparator import (
    AllCloseOutputComparator,
)
from executorch.backends.nxp.tests.comparators.base_output_comparator import (
    BaseOutputComparator,
)
from executorch.backends.nxp.tests.comparators.classification_accuracy_comparator import (
    ClassificationAccuracyOutputComparator,
)
from executorch.backends.nxp.tests.comparators.numerical_stats_comparator import (
    NumericalStatsOutputComparator,
)

__all__ = [
    "AllCloseOutputComparator",
    "BaseOutputComparator",
    "ClassificationAccuracyOutputComparator",
    "NumericalStatsOutputComparator",
]
