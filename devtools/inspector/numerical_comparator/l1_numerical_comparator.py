# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from typing import Any

import torch
from executorch.devtools.inspector._inspector_utils import convert_to_float_tensor
from executorch.devtools.inspector.numerical_comparator.numerical_comparator_base import (
    NumericalComparatorBase,
)


class L1Comparator(NumericalComparatorBase):
    def compare(self, a: Any, b: Any) -> float:
        """Sum up all these element-wise absolute differences between two tensors."""

        t_a = convert_to_float_tensor(a)
        t_b = convert_to_float_tensor(b)

        try:
            res = torch.abs(t_a - t_b).sum().item()
        except Exception as e:
            raise ValueError(f"Error computing L1 difference between tensors: {str(e)}")
        return res
