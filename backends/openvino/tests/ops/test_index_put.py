import torch
from executorch.backends.openvino.tests.ops.base_openvino_op_test import (
    BaseOpenvinoOpTest,
)


class TestIndexPutOperator(BaseOpenvinoOpTest):

    def create_model(self):
        class IndexPut(torch.nn.Module):
            def forward(self, x, mask, values):
                x = x.clone()
                x[mask] = values[mask]
                return x

        return IndexPut()

    def test_index_put_bool_mask(self):
        mask = torch.rand(2, 8) > 0.5
        for dtype in [torch.float32, torch.bfloat16]:
            with self.subTest(dtype=dtype):
                module = self.create_model()
                sample_input = (
                    torch.zeros(2, 8, dtype=dtype),
                    mask,
                    torch.randn(2, 8).to(dtype),
                )
                self.execute_layer_test(module, sample_input)
