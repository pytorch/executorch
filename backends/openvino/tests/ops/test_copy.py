import itertools

import torch
from executorch.backends.openvino.preprocess import OpenvinoBackend
from executorch.backends.openvino.tests.ops.base_openvino_op_test import (
    BaseOpenvinoOpTest,
)
from executorch.exir import EdgeCompileConfig, to_edge
from executorch.exir.backend.backend_api import to_backend
from executorch.exir.backend.backend_details import CompileSpec
from torch.export import Dim, export


class TestCopyOperator(BaseOpenvinoOpTest):

    def test_copy_of_input_as_output(self):
        class CopyAndSin(torch.nn.Module):
            def __init__(self, copy):
                super().__init__()
                self.copy = copy

            def forward(self, x):
                copy = self.copy(x)
                return copy, torch.sin(copy)

        clone = "dim_order_ops::_clone_dim_order"
        copies = {
            "clone": (lambda x: x.clone(), clone, None),
            "full slice": (lambda x: x[:, :], "aten::alias_copy", None),
            "clone of a clone": (lambda x: x.clone().clone(), clone, None),
            # LLM exports turn dim order off, which makes a clone aten.clone.
            "clone, dim order off": (
                lambda x: x.clone(),
                "aten::clone",
                EdgeCompileConfig(_skip_dim_order=True),
            ),
        }
        inputs = (torch.randn(2, 3, 4, 5),)
        for name, (copy, operator, config) in copies.items():
            with self.subTest(copy=name):
                program = self.execute_layer_test(
                    CopyAndSin(copy), inputs, compile_config=config
                )
                # On the CPU, preprocess would also give right values, so check
                # where the copy runs.
                operators = program.executorch_program.execution_plan[0].operators
                self.assertIn(operator, [op.name for op in operators])

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
                    CopyOfConstant(kind),
                    (torch.randn(2, 3, 4, 5),),
                    more_inputs=((torch.randn(2, 3, 4, 5),),),
                )
                # OpenVINO gets a constant as its own data, not as an input, so
                # the copy stays. Run outside, it would store the constant twice.
                plan = program.executorch_program.execution_plan[0]
                self.assertEqual(plan.operators, [])

    def test_copy_of_constant_that_runs_outside(self):
        class Buffered(torch.nn.Module):
            def __init__(self, body, buffer):
                super().__init__()
                self.body = body
                self.register_buffer("buffer", buffer)

            def forward(self, x):
                return self.body(self.buffer, x)

        def copy_of_copy_and_sin(buffer, x):
            copy = buffer.clone().clone(memory_format=torch.channels_last)
            return copy, torch.sin(copy)

        floats = torch.randn(2, 3, 4, 5)
        cases = {
            # OpenVINO writes every output in the default dim order.
            "channels last": (
                lambda b, x: (b.clone(memory_format=torch.channels_last), b + x),
                floats,
            ),
            # The inner copy must leave too, or export finds a cycle.
            "copy of a copy, channels last": (copy_of_copy_and_sin, floats),
            # The CPU plugin computes 64-bit types in 32 bits.
            "int64": (
                lambda b, x: (b.clone(), x + 1),
                torch.tensor([2**40, -(2**40)]),
            ),
            # The partitioner does not take atan, so the CPU stores the buffer
            # anyway, and a copy made there keeps it stored once.
            "also read outside": (lambda b, x: (b.clone(), torch.atan(b) + x), floats),
        }
        for name, (body, buffer) in cases.items():
            with self.subTest(case=name):
                program = self.execute_layer_test(
                    Buffered(body, buffer), (torch.randn(2, 3, 4, 5),)
                )
                operators = program.executorch_program.execution_plan[0].operators
                names = [op.name for op in operators]
                self.assertIn("dim_order_ops::_clone_dim_order", names)
                if name == "also read outside":
                    self.assertIn("aten::atan", names)

    def test_copy_of_mutated_buffer_as_output(self):
        class CopyOfMutatedBuffer(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.register_buffer("buffer", torch.randn(2, 3, 4, 5))

            def forward(self, x):
                copy = self.buffer.clone()
                self.buffer.add_(x)
                return copy, torch.sin(copy)

        program = self.execute_layer_test(
            CopyOfMutatedBuffer(),
            (torch.randn(2, 3, 4, 5),),
            assert_output_equal=False,
        )
        # A mutated buffer is passed to the partition as an input, so its copy
        # must run outside.
        operators = program.executorch_program.execution_plan[0].operators
        self.assertIn("dim_order_ops::_clone_dim_order", [op.name for op in operators])

        # The program does not keep the starting value of a mutated buffer, so
        # check each call against the one before it.
        x1, x2 = torch.randn(2, 3, 4, 5), torch.randn(2, 3, 4, 5)
        first, second = self.execute_program(program, ((x1,), (x2,)))
        tolerance = {"atol": self.atol, "rtol": self.rtol}
        torch.testing.assert_close(second[0], first[0] + x1, **tolerance)
        for copy, sine in (first, second):
            torch.testing.assert_close(sine, torch.sin(copy), **tolerance)

    def test_whole_program_returning_copy_of_input(self):
        class CopyAndReshape(torch.nn.Module):
            def forward(self, x):
                return x.clone(), x.reshape(-1)

        class Lowered(torch.nn.Module):
            def __init__(self, lowered):
                super().__init__()
                self.lowered = lowered

            def forward(self, x):
                return self.lowered(x)

        inputs = {
            "float": lambda n: torch.randn(n, 3),
            "int64": lambda n: torch.randint(-100, 100, (n, 3)),
            "bool": lambda n: torch.randn(n, 3) > 0,
        }
        for (name, make), dynamic in itertools.product(inputs.items(), (False, True)):
            dynamic_shapes = ({0: Dim("rows", min=1, max=4)},) if dynamic else None
            with self.subTest(dtype=name, dynamic=dynamic):
                x = make(4)
                # Without a partitioner the copy cannot run on the CPU, so
                # OpenVINO has to make it.
                lowered = to_backend(
                    OpenvinoBackend.__name__,
                    to_edge(
                        export(CopyAndReshape(), (x,), dynamic_shapes=dynamic_shapes)
                    ).exported_program(),
                    [CompileSpec("device", self.device.encode())],
                )
                program = to_edge(
                    export(Lowered(lowered), (x,), dynamic_shapes=dynamic_shapes)
                ).to_executorch()
                values = (x, make(2 if dynamic else 4))
                outputs = self.execute_program(program, [(v,) for v in values])
                for value, (copy, flat) in zip(values, outputs):
                    torch.testing.assert_close(copy, value)
                    torch.testing.assert_close(flat, value.reshape(-1))
