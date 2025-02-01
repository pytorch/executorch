from executorch.backends.openvino.tests.ops.base_openvino_op_test import BaseOpenvinoOpTest
import torch

op_params = [{'order': [0, 2, 3, 1]   },
             {'order': [0, 3, 1, 2]   },
             ]

class TestPermuteOperator(BaseOpenvinoOpTest):

    def create_model(self, order):

        class Permute(torch.nn.Module):
            def __init__(self, order):
                super(Permute, self).__init__()
                self.order = order

            def forward(self, x):
                return torch.permute(x, self.order)

        return Permute(order)


    def test_permute(self):
        for params in op_params:
            with self.subTest(params=params):
                module = self.create_model(order=params['order'])

                sample_input = (torch.randn(1, 3, 224, 224),)

                self.execute_layer_test(module, sample_input)
