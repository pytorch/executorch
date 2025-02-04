from executorch.backends.openvino.tests.ops.base_openvino_op_test import BaseOpenvinoOpTest
import torch


OPS = [
    torch.relu,
]


class TestUnaryOperator(BaseOpenvinoOpTest):

    def create_model(self, op, dtype):

        class UnaryOp(torch.nn.Module):
            def __init__(self, op, dtype):
                super().__init__()
                self.dtype = dtype
                self.op = op
        
            def forward(self, x):
                x1 = x.to(self.dtype)
                y = self.op(x1)
                return y, x1

        return UnaryOp(op, dtype)


    def test_unary_op(self):
        for op in OPS:
            with self.subTest(op=OPS):

                module = self.create_model(op, dtype=torch.float32)

                sample_input = (torch.rand(2, 10) * 10 + 1,)

                self.execute_layer_test(module, sample_input)
