# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.


import unittest

import torch
from executorch.export import ExportRecipe, ExportSession
from executorch.export.types import StageType


class Tiny(torch.nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + 1


class TestPrintDelegationInfoGuards(unittest.TestCase):
    def test_print_delegation_info_requires_a_lowering_stage(self) -> None:
        session = ExportSession(
            model=Tiny(),
            example_inputs=[(torch.randn(2, 3),)],
            export_recipe=ExportRecipe(
                name="test",
                pipeline_stages=[StageType.TORCH_EXPORT],
            ),
        )
        with self.assertRaises(RuntimeError) as cm:
            session.print_delegation_info()
        self.assertIn("at least one of the lowering stages", str(cm.exception))

    def test_print_delegation_info_requires_lowering_artifact(self) -> None:
        session = ExportSession(
            model=Tiny(),
            example_inputs=[(torch.randn(2, 3),)],
            export_recipe=ExportRecipe(
                name="test",
                pipeline_stages=[
                    StageType.TORCH_EXPORT,
                    StageType.TO_EDGE_TRANSFORM_AND_LOWER,
                ],
            ),
        )
        with self.assertRaises(RuntimeError) as cm:
            session.print_delegation_info()
        self.assertIn("run the lowering stage first", str(cm.exception))
