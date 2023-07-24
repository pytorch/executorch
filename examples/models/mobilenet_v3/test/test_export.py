import unittest

import torch
from executorch.examples.models.mobilenet_v3 import MV3Model
from executorch.examples.utils import _EDGE_COMPILE_CONFIG


class ExportTest(unittest.TestCase):
    def test_export_to_executorch(self):
        eager_model = MV3Model.get_model().eval()
        import executorch.exir as exir

        capture_config = exir.CaptureConfig(enable_dynamic_shape=False)
        edge_model = exir.capture(
            eager_model, MV3Model.get_example_inputs(), capture_config
        ).to_edge(_EDGE_COMPILE_CONFIG)
        example_inputs = MV3Model.get_example_inputs()
        with torch.no_grad():
            eager_output = eager_model(*example_inputs)
        executorch_model = edge_model.to_executorch()
        with torch.no_grad():
            executorch_output = executorch_model.graph_module(*example_inputs)
        self.assertTrue(
            torch.allclose(eager_output, executorch_output[0], rtol=1e-5, atol=1e-5)
        )
