import os
import tempfile

import torch
from executorch.backends.openvino.preprocess import OpenvinoBackend
from executorch.backends.openvino.tests.ops.base_openvino_op_test import (
    BaseOpenvinoOpTest,
)
from executorch.exir import EdgeCompileConfig, to_edge
from executorch.exir.backend.backend_api import to_backend
from executorch.exir.backend.backend_details import CompileSpec
from executorch.runtime import Runtime
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
                    CopyOfConstant(kind),
                    (torch.randn(2, 3, 4, 5),),
                    more_inputs=((torch.randn(2, 3, 4, 5),),),
                )
                # OpenVINO gets a constant as its own data, not as an input, so
                # the copy stays. Run outside, it would store the constant twice.
                plan = program.executorch_program.execution_plan[0]
                self.assertEqual(plan.operators, [])

    def test_copy_of_constant_that_runs_outside(self):
        class CopyOfBuffer(torch.nn.Module):
            def __init__(self, copy, read):
                super().__init__()
                self.copy = copy
                self.read = read
                self.register_buffer("buffer", torch.randn(2, 3, 4, 5))

            def forward(self, x):
                return self.copy(self.buffer), self.read(self.buffer) + x

        cases = {
            # OpenVINO writes every output in the default dim order.
            "channels last": (
                lambda b: b.clone(memory_format=torch.channels_last),
                lambda b: b,
            ),
            # The partitioner does not take atan, so the CPU stores the buffer
            # anyway, and a copy made there keeps it stored once.
            "also read outside": (lambda b: b.clone(), torch.atan),
        }
        for name, (copy, read) in cases.items():
            with self.subTest(case=name):
                program = self.execute_layer_test(
                    CopyOfBuffer(copy, read), (torch.randn(2, 3, 4, 5),)
                )
                operators = program.executorch_program.execution_plan[0].operators
                self.assertIn(
                    "dim_order_ops::_clone_dim_order", [op.name for op in operators]
                )

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
        with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmp_dir:
            pte_path = os.path.join(tmp_dir, "model.pte")
            with open(pte_path, "wb") as f:
                program.write_to_file(f)
            loaded = Runtime.get().load_program(pte_path)
            method = loaded.load_method("forward")
            assert method is not None
            first = method.execute((x1,))
            second = method.execute((x2,))
            del method, loaded
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
        dynamic_shapes = ({0: Dim("rows", min=1, max=4)},)
        for name, make in inputs.items():
            with self.subTest(dtype=name):
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
                with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmp:
                    pte_path = os.path.join(tmp, "model.pte")
                    with open(pte_path, "wb") as f:
                        program.write_to_file(f)
                    loaded = Runtime.get().load_program(pte_path)
                    method = loaded.load_method("forward")
                    assert method is not None
                    for value in (x, make(2)):
                        copy, flat = method.execute((value,))
                        torch.testing.assert_close(copy, value)
                        torch.testing.assert_close(flat, value.reshape(-1))
                    del method, loaded
