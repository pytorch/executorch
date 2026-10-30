import torch
from executorch.backends.openvino.tests.ops.base_openvino_op_test import (
    BaseOpenvinoOpTest,
)

op_params = [
    {"input_shape": [2, 3, 2], "target_shape": [2, 6]},
    {"input_shape": [4], "target_shape": [2, 2]},
]


class TestViewOperator(BaseOpenvinoOpTest):

    def create_model(self, target_shape):

        class View(torch.nn.Module):

            def __init__(self, target_shape) -> None:
                super().__init__()
                self.target_shape = target_shape

            def forward(self, input_tensor):
                return input_tensor.view(self.target_shape)

        return View(target_shape)

    def test_view(self):
        for params in op_params:
            with self.subTest(params=params):

                module = self.create_model(params["target_shape"])

                sample_input = (torch.randn(params["input_shape"]),)

                self.execute_layer_test(module, sample_input)

    def test_view_data_dependent_size(self):
        class View(torch.nn.Module):
            def forward(self, x, n):
                n = n.item()
                torch._check(n >= 1)
                torch._check(n <= x.shape[0])
                return x[:n].view(n, -1).sum(0)

        # The size reaches the OpenVINO partition as a SymInt input.
        self.execute_layer_test(View(), (torch.randn(4, 8), torch.tensor(3)))
