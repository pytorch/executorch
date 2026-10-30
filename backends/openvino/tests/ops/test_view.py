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
            def __init__(self, reduce):
                super().__init__()
                self.reduce = reduce

            def forward(self, x, n):
                n = n.item()
                torch._check(n >= 1)
                torch._check(n <= x.shape[0])
                y = x[:n].view(n, -1)
                return y.sum(0) if self.reduce else y

        # The size reaches the OpenVINO partition as a SymInt input. The second
        # call needs a larger output than the first one returned, unless the
        # sum makes the output static.
        x = torch.randn(4, 8)
        for reduce in (False, True):
            with self.subTest(reduce=reduce):
                self.execute_layer_test(
                    View(reduce),
                    (x, torch.tensor(3)),
                    more_inputs=((x, torch.tensor(4)),),
                )
