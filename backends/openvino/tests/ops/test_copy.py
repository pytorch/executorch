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

        # The clone stays in the partition, since only the mean reads it.
        program = self.execute_layer_test(
            ChannelsLastMean(), (torch.randn(2, 3, 4, 5),)
        )
        self.assertEqual(program.executorch_program.execution_plan[0].operators, [])

    def test_copy_of_constant_as_output(self):
        class CopyOfConstant(torch.nn.Module):
            def __init__(self, kind, memory_format=None, dtype=torch.float32):
                super().__init__()
                self.memory_format = memory_format
                if dtype.is_floating_point:
                    value = torch.randn(2, 3, 4, 5).to(dtype)
                elif dtype == torch.bool:
                    value = torch.randn(2, 3, 4, 5) > 0
                else:
                    # float16 holds every integer only up to 2048, so these
                    # show a copy rounded through it, and x + constant stays
                    # in its range.
                    info = torch.iinfo(dtype)
                    low, high = max(info.min, -(2**15)), min(info.max, 2**15)
                    value = torch.randint(low, high, (2, 3, 4, 5), dtype=dtype)
                if kind == "parameter":
                    self.constant = torch.nn.Parameter(value)
                elif kind == "buffer":
                    self.register_buffer("constant", value)
                else:
                    self.constant = value

            def forward(self, x):
                copy = self.constant.clone(memory_format=self.memory_format)
                return copy, x + self.constant

        cases = {
            "parameter": (CopyOfConstant("parameter"), None),
            "buffer": (CopyOfConstant("buffer"), None),
            "lifted tensor": (CopyOfConstant("lifted tensor"), None),
            # OpenVINO transposes an output in another dim order.
            "buffer, channels last": (
                CopyOfConstant("buffer", torch.channels_last),
                None,
            ),
            # With dim order off this stays aten.clone, and the portable clone
            # kernel rejects preserve_format.
            "buffer, dim order off": (
                CopyOfConstant("buffer", torch.preserve_format),
                EdgeCompileConfig(_skip_dim_order=True),
            ),
            # The precision changes only how OpenVINO computes floats.
            "int32 buffer, f16 precision": (
                CopyOfConstant("buffer", dtype=torch.int32),
                None,
                CompileSpec("inference_precision", b"f16"),
            ),
        }
        for dtype in (
            torch.bool,
            torch.uint8,
            torch.int8,
            torch.int16,
            torch.int32,
            torch.float16,
            torch.bfloat16,
        ):
            cases[f"{dtype} buffer"] = (CopyOfConstant("buffer", dtype=dtype), None)
        on_cpu = self.device.startswith("CPU")
        for name, (module, config, *compile_specs) in cases.items():
            with self.subTest(case=name):
                if module.constant.is_floating_point() and not on_cpu:
                    # Off the CPU, a float copy stays only when f32 is set.
                    if self.device.startswith("NPU"):
                        self.skipTest("the NPU does not accept f32")
                    compile_specs.append(CompileSpec("inference_precision", b"f32"))
                inputs = (torch.randn(2, 3, 4, 5),)
                program = self.execute_layer_test(
                    module,
                    inputs,
                    more_inputs=((torch.randn(2, 3, 4, 5),),),
                    compile_specs=tuple(compile_specs),
                    compile_config=config,
                )
                # OpenVINO gets a constant as its own data, not as an input, so
                # the copy stays. Run outside, it would store the constant twice.
                plan = program.executorch_program.execution_plan[0]
                self.assertEqual(plan.operators, [])
                # The tolerance above lets a copy rounded through float16 pass.
                copy = self.execute_program(program, [inputs])[0][0]
                torch.testing.assert_close(copy, module.constant, rtol=0, atol=0)

    def test_copy_of_constant_that_runs_outside(self):
        class Buffered(torch.nn.Module):
            def __init__(self, body, buffer):
                super().__init__()
                self.body = body
                self.register_buffer("buffer", buffer)

            def forward(self, x):
                return self.body(self.buffer, x)

        def copy_of_copy_and_add(buffer, x):
            copy = buffer.clone().clone()
            return copy, copy + 1

        def copy_and_copy_of_it(buffer, x):
            copy = buffer.clone()
            return copy, copy.clone(), buffer + x

        floats = torch.randn(2, 3, 4, 5)
        cases = {
            # The inner copy must leave too, or export finds a cycle.
            "copy of a copy, int64": (copy_of_copy_and_add, torch.tensor([2, -3])),
            # The CPU plugin computes 64-bit types in 32 bits.
            "int64": (
                lambda b, x: (b.clone(), x + 1),
                torch.tensor([2**40, -(2**40)]),
            ),
            # The runtime cannot bind a uint16 output.
            "uint16": (
                lambda b, x: (b.clone(), x + 1),
                torch.tensor([0, 40000, 65535], dtype=torch.uint16),
            ),
            # OpenVINO's PyTorch frontend cannot return one constant twice.
            "two copies": (copy_and_copy_of_it, floats),
            # In a lower precision OpenVINO can round a float copy.
            "float16 precision": (
                lambda b, x: (b.clone(), x + 1),
                floats,
                CompileSpec("inference_precision", b"f16"),
            ),
            # The partitioner does not take atan, so the CPU stores the buffer
            # anyway, and a copy made there keeps it stored once.
            "also read outside": (lambda b, x: (b.clone(), torch.atan(b) + x), floats),
        }
        for name, (body, buffer, *compile_specs) in cases.items():
            with self.subTest(case=name):
                program = self.execute_layer_test(
                    Buffered(body, buffer),
                    (torch.randn(2, 3, 4, 5),),
                    compile_specs=tuple(compile_specs),
                )
                operators = program.executorch_program.execution_plan[0].operators
                names = [op.name for op in operators]
                if name == "also read outside":
                    self.assertIn("aten::atan", names)
                self.assertIn("dim_order_ops::_clone_dim_order", names)

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
        if not self.device.startswith("CPU"):
            self.skipTest("preprocess adds the copy only for the CPU")

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
