# mypy: disable-error-code=import-not-found

import torch
from executorch.backends.openvino.tests.ops.base_openvino_op_test import (
    BaseOpenvinoOpTest,
)
from openvino.frontend.pytorch.torchdynamo.op_support import (  # type: ignore[import-untyped]
    OperatorSupport,
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

    def test_index_after_index_put(self):
        if "torch.ops.aten.index_put.default" not in OperatorSupport([])._support_dict:
            self.skipTest("this OpenVINO release does not delegate index_put")

        class WriteThenRead(torch.nn.Module):
            def forward(self, x, mask):
                x = x.clone()
                x[mask] = x[mask] * 2
                # The read needs the rank of what the write returns.
                return x + x[mask].sum()

        self.execute_layer_test(
            WriteThenRead(), (torch.randn(2, 8), torch.rand(2, 8) > 0.5)
        )
