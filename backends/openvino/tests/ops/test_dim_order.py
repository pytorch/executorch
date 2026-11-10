import contextlib
import sys
from unittest import mock

import torch
from executorch.backends.openvino.tests.ops.base_openvino_op_test import (
    BaseOpenvinoOpTest,
)
from executorch.exir.backend.backend_details import BackendDetails
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


# In the input cases the clone is returned, so it runs on the CPU and the
# partition gets a channels-last input.
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


@contextlib.contextmanager
def lowered_as_before_the_header():
    """Lower without the header and the transposes, as older ExecuTorch did."""
    with contextlib.ExitStack() as stack:
        # Under pytest the backend can be registered from two module paths.
        for cls in BackendDetails.__subclasses__():
            if cls.__name__ == "OpenvinoBackend":
                module = sys.modules[cls.__module__]
                stack.enter_context(mock.patch.object(module, "_HEADER", b""))
                stack.enter_context(
                    mock.patch.object(
                        module, "_non_default_dim_order", lambda val: None
                    )
                )
        yield


class TestDimOrder(BaseOpenvinoOpTest):

    def check(self, module, inputs, **kwargs):
        program = self.execute_layer_test(module, inputs, **kwargs)
        return [
            op.name for op in program.executorch_program.execution_plan[0].operators
        ]

    def check_case(self, name, inputs, **kwargs):
        operators = self.check(Model(CASES[name]), inputs, **kwargs)
        if name.startswith("input"):
            self.assertIn("dim_order_ops::_clone_dim_order", operators)

    def test_channels_last_at_partition_edge(self):
        x = torch.randn(2, 3, 6, 5)
        for name in CASES:
            with self.subTest(case=name):
                self.check_case(name, (x,))

    def test_channels_last_program_input(self):
        x = torch.randn(2, 3, 6, 5).to(memory_format=torch.channels_last)
        cases = {
            "mean": lambda m, x: x.mean((2, 3)),
            "conv": lambda m, x: m.conv(x),
        }
        for name, forward in cases.items():
            with self.subTest(case=name):
                self.assertEqual(self.check(Model(forward), (x,)), [])

    def test_channels_last_3d_program_input(self):
        # ExecuTorch cannot export a copy into channels_last_3d, so a 5-D
        # channels-last tensor reaches a partition only as a program input.
        x = torch.randn(2, 3, 4, 5, 6).to(memory_format=torch.channels_last_3d)
        module = Model(lambda m, x: x.mean((2, 3, 4)))
        self.assertEqual(self.check(module, (x,)), [])

    def test_channels_last_input_after_int_input(self):
        class IntAndChannelsLast(torch.nn.Module):
            def forward(self, x, n, w):
                n = n.item()
                torch._check(n >= 1)
                torch._check(n <= x.shape[0])
                y = channels_last(w)
                return y, x[:n].sum(0) + y.mean((2, 3))

        # The partition gets x, then the int, then the channels-last clone.
        x, w = torch.randn(4, 3), torch.randn(2, 3, 6, 5)
        self.execute_layer_test(
            IntAndChannelsLast(),
            (x, torch.tensor(3), w),
            more_inputs=((x, torch.tensor(4), w),),
        )

    def test_channels_last_with_dynamic_sizes(self):
        for name in ("output", "input to mean"):
            with self.subTest(case=name):
                self.check_case(
                    name,
                    (torch.randn(2, 3, 6, 5),),
                    more_inputs=((torch.randn(2, 3, 3, 5),),),
                    dynamic_shapes=({2: Dim("h", min=2, max=8)},),
                )

    def test_channels_last_with_size_one_dims(self):
        # ExecuTorch keeps a channels-last dim order when only C or only H and W
        # are 1, though the bytes are then laid out as if contiguous. The
        # lowering and the runtime must pick the same dim order, or the runtime
        # passes sizes the compiled model does not take.
        for shape in ((2, 1, 4, 5), (2, 3, 1, 1)):
            for name in ("output", "input to mean", "input to flatten"):
                with self.subTest(shape=shape, case=name):
                    self.check_case(name, (torch.randn(shape),))

    def test_program_lowered_before_the_header(self):
        # The runtime passes the tensors of such a blob with their sizes in
        # logical order, as before, so the cases that were right stay right.
        cases = (
            ("input to sin", (2, 3, 6, 5)),
            ("input to mean", (2, 1, 4, 5)),
            ("input to mean", (2, 3, 1, 1)),
            ("contiguous", (2, 3, 6, 5)),
        )
        with lowered_as_before_the_header():
            for name, shape in cases:
                with self.subTest(case=name, shape=shape):
                    self.check_case(name, (torch.randn(shape),))
            with self.subTest(case="input to sin", dynamic=True):
                self.check_case(
                    "input to sin",
                    (torch.randn(2, 3, 6, 5),),
                    more_inputs=((torch.randn(2, 3, 3, 5),),),
                    dynamic_shapes=({2: Dim("h", min=2, max=8)},),
                )
