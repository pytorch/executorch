from executorch.backends.openvino.tests.ops.base_openvino_op_test import BaseOpenvinoOpTest
import torch

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
