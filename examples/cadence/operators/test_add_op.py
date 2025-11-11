# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-unsafe

import unittest
from typing import Tuple

from parameterized import parameterized

from executorch.backends.cadence.aot.ops_registrations import *  # noqa

import torch
import torch.nn as nn
from executorch.backends.cadence.aot.export_example import export_and_run_model


class ATenOpTestCases(unittest.TestCase):
    # pyre-fixme[16]: Module `parameterized.parameterized` has no attribute `expand`.
    @parameterized.expand(
        [
            [(7, 5, 6), (7, 5, 6)],
            [(7, 5, 6), (1)],
            [(1), (7, 5, 6)],
            [(1), (7, 5, 6), 2.23],
            [(1), (7, 5, 6), -1.0],
            [(1), (7, 5, 6), -2.23],
            [(7, 5, 6), (7, 5, 6), 1.23],
            [(6, 7), (6, 7)],
            [(6, 7), (6, 7), 2],
            # Broadcast tests (should be optimized on G3)
            [(1, 32, 64), (1, 1, 64)],
            [(1, 32, 64), (64)],
            [(1, 1, 32), (32)],
            [(16, 1, 16), (1, 1, 16)],
            [(16, 1, 16), (16)],
            [(1, 4, 8, 8), (1, 1, 8, 8)],
            [(1, 4, 8, 8), (8, 8)],
            # Broadcast tests (should go to portable ops)
            [(1, 10, 1, 8), (4, 1, 4, 1)],
            [(1, 1, 16), (1, 8, 1), 2.5],
            # # aten.upsample_nearest2d tests
            [(5, 6, 6, 8), (5, 6, 6, 8)],
            [(1, 1, 12, 16), (1, 1, 12, 16)],
        ]
    )
    def test_aten_add_out(
        self, Xshape: Tuple[int], Yshape: Tuple[int], alpha: float = 1
    ) -> None:
        class AddTensor(nn.Module):
            def __init__(self, alpha: float):
                super().__init__()
                self.alpha = alpha

            def forward(self, x: torch.Tensor, y: torch.Tensor):
                return torch.add(x, y, alpha=self.alpha)

        model = AddTensor(alpha)

        X = torch.randn(Xshape)
        Y = torch.randn(Yshape)

        model.eval()
        export_and_run_model(
            model, (X, Y), file_name=self._testMethodName, run_and_compare=False
        )

    # pyre-fixme[16]: Module `parameterized.parameterized` has no attribute `expand`.
    @parameterized.expand(
        [
            [(7, 5, 6), (7, 5, 6)],
            [(7, 5, 6), (1)],
            [(1), (7, 5, 6)],
            [(1), (7, 5, 6), 2.23],
            [(1), (7, 5, 6), -1.0],
            [(1), (7, 5, 6), -2.23],
            [(7, 5, 6), (7, 5, 6), 1.23],
            [(6, 7), (6, 7)],
            [(6, 7), (6, 7), 2],
            # Broadcast tests (should be optimized on G3)
            [(1, 32, 64), (1, 1, 64)],
            [(1, 32, 64), (64)],
            [(1, 1, 32), (32)],
            [(16, 1, 16), (1, 1, 16)],
            [(16, 1, 16), (16)],
            [(1, 4, 8, 8), (1, 1, 8, 8)],
            [(1, 4, 8, 8), (8, 8)],
            # Broadcast tests (should go to portable ops)
            [(1, 10, 1, 8), (4, 1, 4, 1)],
            [(1, 1, 16), (1, 8, 1), 2.5],
            # # aten.upsample_nearest2d tests
            [(5, 6, 6, 8), (5, 6, 6, 8)],
            [(1, 1, 12, 16), (1, 1, 12, 16)],
        ]
    )
    def test_aten_add_scalar_out(
        self, Xshape: Tuple[int], Yshape: Tuple[int], alpha: float = 1
    ) -> None:
        # Tensor-Scalar addition
        class AddScalar(nn.Module):
            def __init__(self, alpha: float):
                super().__init__()
                self.alpha = alpha

            def forward(self, x: torch.Tensor, y: float):
                return torch.add(x, y, alpha=self.alpha)

        model = AddScalar(alpha)

        X = torch.randn(Xshape)
        Y = 2.34

        model.eval()
        export_and_run_model(
            model, (X, Y), file_name=self._testMethodName, run_and_compare=False
        )


if __name__ == "__main__":
    unittest.main()
