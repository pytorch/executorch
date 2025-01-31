from executorch.backends.openvino.tests.ops.base_openvino_op_test import BaseOpenvinoOpTest
import torch

class TestArangeOperator(BaseOpenvinoOpTest):

    def create_model(self, x):
        class Arange(torch.nn.Module):
            def __init__(self, x):
                super().__init__()
                self.x = x
        
            def forward(self, y):
                return torch.arange(self.x, dtype=torch.float32) + y

        return Arange(5)

    def test_arange(self):
        module = self.create_model(5)
        sample_input = (torch.randn(5),)
        self.execute_layer_test(module, sample_input)
