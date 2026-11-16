# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import unittest

from executorch.backends.transforms.fused_quant.optimization_passes.fuse_add_into_linear import (
    FuseAddIntoLinear,
)
from executorch.backends.transforms.fused_quant.optimization_passes.fuse_mul_into_linear import (
    FuseMulIntoLinear,
)
from executorch.backends.transforms.fused_quant.optimization_passes.prequantize_embedding import (
    PrequantizeEmbedding,
)
from executorch.backends.transforms.fused_quant.optimization_passes.split_linear_at_slices import (
    SplitLinearAtSlices,
)
from executorch.backends.transforms.fused_quant.optimization_passes.to_channels_last import (
    ToChannelsLast,
)
from executorch.backends.transforms.fused_quant.passes import get_optimization_passes
from parameterized import parameterized

_PASSES_BY_FLAG: dict[str, set[type]] = {
    "channels_last": {ToChannelsLast},
    "fold_into_linear": {FuseAddIntoLinear, FuseMulIntoLinear},
    "prequantize_embedding": {PrequantizeEmbedding},
    "split_linear_at_slices": {SplitLinearAtSlices},
}


def _pass_types(**flags: bool) -> set[type]:
    return {type(pass_) for pass_ in get_optimization_passes(**flags)._passes}


class OptimizationPassesTest(unittest.TestCase):
    def test_flags_are_off_by_default(self) -> None:
        flagged = set().union(*_PASSES_BY_FLAG.values())

        self.assertEqual(_pass_types() & flagged, set())

    @parameterized.expand(list(_PASSES_BY_FLAG.items()))
    def test_each_flag_adds_only_its_passes(
        self, flag: str, expected: set[type]
    ) -> None:
        added = _pass_types(**{flag: True}) - _pass_types()

        self.assertEqual(added, expected)
