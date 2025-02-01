from executorch.backends.openvino.tests.ops.base_openvino_op_test import BaseOpenvinoOpTest
import torch

class TestAddMMOperator(BaseOpenvinoOpTest):

    def create_model(self):
        class AddMM(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.alpha = 1.
                self.beta = 1.
        
            def forward(self, x, y, z):
                #return torch.add(x, y)
                return torch.addmm(x, y, z, alpha=self.alpha, beta=self.beta)

        return AddMM()

    def test_addmm(self):
        module = self.create_model()
        input_x = torch.randn(4,4, dtype=torch.float32)
        input_y = torch.randn(4,4, dtype=torch.float32)
        input_z = torch.randn(4,4, dtype=torch.float32)
        sample_input = (input_x, input_y, input_z)
        self.execute_layer_test(module, sample_input)
