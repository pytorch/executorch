# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import unittest

import torch
from executorch.examples.models.mobilenet_v2 import MV2Model
from executorch.examples.models.mobilenet_v3 import MV3Model

from ..utils import _EDGE_COMPILE_CONFIG


class ExportTest(unittest.TestCase):
    def _assert_eager_lowered_same_result(
        self, eager_model: torch.nn.Module, example_inputs
    ):
        import executorch.exir as exir

        capture_config = exir.CaptureConfig(enable_dynamic_shape=False)
        edge_model = exir.capture(eager_model, example_inputs, capture_config).to_edge(
            _EDGE_COMPILE_CONFIG
        )

        executorch_model = edge_model.to_executorch()
        with torch.no_grad():
            eager_output = eager_model(*example_inputs)
        with torch.no_grad():
            executorch_output = executorch_model.graph_module(*example_inputs)
        self.assertTrue(
            torch.allclose(eager_output, executorch_output[0], rtol=1e-5, atol=1e-5)
        )

    def test_mv3_export_to_executorch(self):
        eager_model = MV3Model.get_model().eval()
        example_inputs = MV3Model.get_example_inputs()

        self._assert_eager_lowered_same_result(eager_model, example_inputs)

    def test_mv2_export_to_executorch(self):
        eager_model = MV2Model.get_model().eval()
        example_inputs = MV2Model.get_example_inputs()

        self._assert_eager_lowered_same_result(eager_model, example_inputs)
