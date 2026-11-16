import torch
from executorch.backends.openvino.tests.ops.base_openvino_op_test import (
    BaseOpenvinoOpTest,
)


class TestAddOperator(BaseOpenvinoOpTest):

    def create_model(self):
        class Add(torch.nn.Module):
            def __init__(self):
                super().__init__()

            def forward(self, x, y):
                return torch.add(x, y)

        return Add()

    def test_add(self):
        module = self.create_model()
        sample_input = (torch.randn(2, 5, 1, 3), torch.randn(2, 5, 1, 3))
        self.execute_layer_test(module, sample_input)

    def test_add_0d_inputs(self):
        module = self.create_model()
        for sample_input in (
            (torch.randn(2, 3), torch.tensor(2.5)),
            (torch.tensor(1.5), torch.tensor(2.5)),
        ):
            with self.subTest(shapes=[list(x.shape) for x in sample_input]):
                self.execute_layer_test(module, sample_input)
