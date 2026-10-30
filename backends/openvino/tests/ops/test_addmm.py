import torch
from executorch.backends.openvino.tests.ops.base_openvino_op_test import (
    BaseOpenvinoOpTest,
)
from executorch.exir.backend.backend_details import CompileSpec


class TestAddMMOperator(BaseOpenvinoOpTest):

    def create_model(self):
        class AddMM(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.alpha = 1.0
                self.beta = 1.0

            def forward(self, x, y, z):
                # return torch.add(x, y)
                return torch.addmm(x, y, z, alpha=self.alpha, beta=self.beta)

        return AddMM()

    def test_addmm(self):
        module = self.create_model()
        input_x = torch.randn(4, 4, dtype=torch.float32)
        input_y = torch.randn(4, 4, dtype=torch.float32)
        input_z = torch.randn(4, 4, dtype=torch.float32)
        sample_input = (input_x, input_y, input_z)
        self.execute_layer_test(module, sample_input)

    def test_addmm_computes_in_float32(self):
        # On CPUs with bfloat16 units, OpenVINO would otherwise compute this in
        # bfloat16 and miss by about 5e-2.
        self.atol = self.rtol = 1e-4
        sample_input = tuple(torch.randn(256, 256) for _ in range(3))
        self.execute_layer_test(self.create_model(), sample_input)

    def test_addmm_refuses_an_unknown_inference_precision(self):
        sample_input = tuple(torch.randn(4, 4) for _ in range(3))
        with self.assertRaises(RuntimeError):
            self.execute_layer_test(
                self.create_model(),
                sample_input,
                compile_specs=(CompileSpec("inference_precision", b"f7"),),
            )
