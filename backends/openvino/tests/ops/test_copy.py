import torch
from executorch.backends.openvino.preprocess import OpenvinoBackend
from executorch.backends.openvino.tests.ops.base_openvino_op_test import (
    BaseOpenvinoOpTest,
)
from executorch.exir import EdgeCompileConfig, to_edge
from executorch.exir.backend.backend_api import to_backend
from executorch.exir.backend.backend_details import CompileSpec
from executorch.runtime import Runtime
from torch.export import export


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

    def test_copy_of_constant_as_output(self):
        class CopyOfConstant(torch.nn.Module):
            def __init__(self, kind):
                super().__init__()
                value = torch.randn(2, 3, 4, 5)
                if kind == "parameter":
                    self.constant = torch.nn.Parameter(value)
                elif kind == "buffer":
                    self.register_buffer("constant", value)
                else:
                    self.constant = value

            def forward(self, x):
                return self.constant.clone(), x + self.constant

        for kind in ("parameter", "buffer", "lifted tensor"):
            with self.subTest(kind=kind):
                program = self.execute_layer_test(
                    CopyOfConstant(kind), (torch.randn(2, 3, 4, 5),)
                )
                # OpenVINO gets a constant as its own data, not as an input, so
                # the copy stays. Run outside, it would store the constant twice.
                plan = program.executorch_program.execution_plan[0]
                self.assertEqual(plan.operators, [])

    def test_copy_of_mutated_buffer_as_output(self):
        class CopyOfMutatedBuffer(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.register_buffer("buffer", torch.randn(2, 3, 4, 5))

            def forward(self, x):
                copy = self.buffer.clone()
                self.buffer.add_(x)
                return copy, torch.sin(copy)

        # The program does not keep the starting value of a mutated buffer, so
        # there is no reference output to compare with.
        program = self.execute_layer_test(
            CopyOfMutatedBuffer(),
            (torch.randn(2, 3, 4, 5),),
            assert_output_equal=False,
        )
        # A mutated buffer is passed to the partition as an input, so its copy
        # must run outside.
        operators = program.executorch_program.execution_plan[0].operators
        self.assertIn("dim_order_ops::_clone_dim_order", [op.name for op in operators])

    def test_whole_program_returning_copy_of_input(self):
        class CopyAndSin(torch.nn.Module):
            def forward(self, x):
                copy = x.clone()
                return copy, torch.sin(copy)

        x = torch.randn(2, 3, 4, 5)
        # Without a partitioner the copy cannot run on the CPU, so OpenVINO
        # has to make it.
        lowered = to_backend(
            OpenvinoBackend.__name__,
            to_edge(export(CopyAndSin(), (x,))).exported_program(),
            [CompileSpec("device", self.device.encode())],
        )
        program = Runtime.get().load_program(lowered.buffer())
        method = program.load_method("forward")
        assert method is not None
        torch.testing.assert_close(
            method.execute((x,)), [x, torch.sin(x)], atol=self.atol, rtol=self.rtol
        )
