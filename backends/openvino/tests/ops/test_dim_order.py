import torch
from executorch.backends.openvino.tests.ops.base_openvino_op_test import (
    BaseOpenvinoOpTest,
)
from torch.export import Dim


class Model(torch.nn.Module):
    def __init__(self, forward):
        super().__init__()
        self.conv = torch.nn.Conv2d(3, 4, 3)
        self.fn = forward

    def forward(self, x):
        return self.fn(self, x)


def channels_last(x):
    return x.clone(memory_format=torch.channels_last)


# The clone in the input cases runs on CPU, so the partition gets a
# channels-last input.
CASES = {
    "output": lambda m, x: (x * 2).to(memory_format=torch.channels_last),
    "input to mean": lambda m, x: (y := channels_last(x), y.mean((2, 3))),
    "input to flatten": lambda m, x: (y := channels_last(x), y.flatten(1) * 2),
    "input to sin": lambda m, x: (y := channels_last(x), torch.sin(y)),
    "input to conv": lambda m, x: (y := channels_last(x), m.conv(y)),
    "conv output": lambda m, x: m.conv(x).to(memory_format=torch.channels_last),
    "outputs in both orders": lambda m, x: (
        (y := x * 2).sum(1),
        y.to(memory_format=torch.channels_last),
        y.mean(1, keepdim=True),
    ),
    "contiguous": lambda m, x: (x.mean((2, 3)), m.conv(x).flatten(1)),
}


class TestDimOrder(BaseOpenvinoOpTest):

    def test_channels_last_at_partition_edge(self):
        x = torch.randn(2, 3, 6, 5)
        for name, forward in CASES.items():
            with self.subTest(case=name):
                self.execute_layer_test(Model(forward), (x,))

    def test_channels_last_with_dynamic_sizes(self):
        cases = {
            "output": CASES["output"],
            "input to mean": CASES["input to mean"],
        }
        for name, forward in cases.items():
            with self.subTest(case=name):
                self.execute_layer_test(
                    Model(forward),
                    (torch.randn(2, 3, 6, 5),),
                    more_inputs=((torch.randn(2, 3, 3, 5),),),
                    dynamic_shapes=({2: Dim("h", min=2, max=8)},),
                )

    def test_channels_last_with_size_one_dims(self):
        # ExecuTorch keeps a channels-last dim order when only C or only H and W
        # are 1, though the bytes are then laid out as if contiguous.
        for shape in ((2, 1, 4, 5), (2, 3, 1, 1)):
            for name in ("output", "input to mean", "input to flatten"):
                with self.subTest(shape=shape, case=name):
                    self.execute_layer_test(Model(CASES[name]), (torch.randn(shape),))
