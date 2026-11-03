import torch
from executorch.backends.openvino.tests.ops.base_openvino_op_test import (
    BaseOpenvinoOpTest,
)
from executorch.exir import EdgeCompileConfig


class TestCopyOperator(BaseOpenvinoOpTest):

    def test_copy_of_input_as_output(self):
        class CopyAndSin(torch.nn.Module):
            def __init__(self, copy):
                super().__init__()
                self.copy = copy

            def forward(self, x):
                copy = self.copy(x)
                return copy, torch.sin(copy)

        copies = {
            "clone": lambda x: x.clone(),
            "full slice": lambda x: x[:, :],
            "clone of a clone": lambda x: x.clone().clone(),
        }
        inputs = (torch.randn(2, 3, 4, 5),)
        for name, copy in copies.items():
            with self.subTest(copy=name):
                self.execute_layer_test(CopyAndSin(copy), inputs)

        # LLM exports turn dim order off, which makes a clone aten.clone.
        self.execute_layer_test(
            CopyAndSin(lambda x: x.clone()),
            inputs,
            compile_config=EdgeCompileConfig(_skip_dim_order=True),
        )

    def test_copy_of_input_used_inside(self):
        class ChannelsLastMean(torch.nn.Module):
            def forward(self, x):
                return x.clone(memory_format=torch.channels_last).mean(dim=(2, 3))

        # Run outside, the clone would hand OpenVINO a channels-last input,
        # which it reads as if it were contiguous.
        self.execute_layer_test(ChannelsLastMean(), (torch.randn(2, 3, 4, 5),))
