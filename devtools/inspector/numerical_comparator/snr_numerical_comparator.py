                                                                                                                                                                                                                                                                                                                                                                                                                                                                                       `                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                        # Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.


from typing import Any, Optional, TYPE_CHECKING

import torch
from executorch.devtools.inspector._inspector_utils import convert_to_float_tensor
from executorch.devtools.inspector.numerical_comparator.numerical_comparator_base import (
    NumericalComparatorBase,
)

if TYPE_CHECKING:
    from executorch.devtools.inspector._inspector import Inspector


class SNRComparator(NumericalComparatorBase):
    """Signal-to-Noise Ratio comparator for numerical discrepancy detection."""

    def __init__(self, inspector: Optional["Inspector"] = None) -> None:
        super().__init__(inspector)

    def element_compare(self, a: Any, b: Any) -> float:
        """
        Compare the Signal-to-Noise Ratio (SNR) between two inputs
        Formula: SNR = 10 * log10(original_power / error_power)
        """

        t_a = convert_to_float_tensor(a)
        t_b = convert_to_float_tensor(b)

        # Calculate the signal power and noise power
        original_power = torch.mean(torch.pow(t_a, 2))
        try:
            error = t_a - t_b
            error_power = torch.mean(torch.pow(error, 2))
        except Exception as e:
            raise ValueError(
                f"Error computing SNR difference between tensors: {str(e)}"
            )

        # Calculate SNR
        snr = 10 * torch.log10(original_power / error_power)
        return snr.item()
