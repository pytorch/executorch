# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import unittest
from typing import Any, Callable, Tuple

import torch
from executorch.exir import ExecutorchProgramManager, to_edge
from torch.export import export


def make_test(  # noqa: C901
    tester: unittest.TestCase,
    load_fn: Callable,
) -> Callable[[unittest.TestCase], None]:
    """
    Returns a function that operates as a test case within a unittest.TestCase class.

    Used to allow the test code for pybindings to be shared across different pybinding libs
    which will all have different load functions. In this case each individual test case is a
    subfunction of wrapper.
    """

    def wrapper(tester: unittest.TestCase) -> None:
        class ModuleAdd(torch.nn.Module):
            """The module to serialize and execute."""

            def __init__(self):
                super(ModuleAdd, self).__init__()

            def forward(self, x, y):
                return x + y

            def get_methods_to_export(self):
                return ("forward",)

            def get_inputs(self):
                return (torch.ones(2, 2), torch.ones(2, 2))

        class ModuleMulti(torch.nn.Module):
            """The module to serialize and execute."""

            def __init__(self):
                super(ModuleMulti, self).__init__()

            def forward(self, x, y):
                return x + y

            def forward2(self, x, y):
                return x + y + 1

            def get_methods_to_export(self):
                return ("forward", "forward2")

            def get_inputs(self):
                return (torch.ones(2, 2), torch.ones(2, 2))

        class ModuleAddSingleInput(torch.nn.Module):
            """The module to serialize and execute."""

            def __init__(self):
                super(ModuleAddSingleInput, self).__init__()

            def forward(self, x):
                return x + x

            def get_methods_to_export(self):
                return ("forward",)

            def get_inputs(self):
                return (torch.ones(2, 2),)

        def create_program(
            eager_module: torch.nn.Module,
        ) -> Tuple[ExecutorchProgramManager, Tuple[Any, ...]]:
            """Returns an executorch program based on ModuleAdd, along with inputs."""

            # Trace the test module and create a serialized ExecuTorch program.
            inputs = eager_module.get_inputs()
            input_map = {}
            for method in eager_module.get_methods_to_export():
                input_map[method] = inputs

            class WrapperModule(torch.nn.Module):
                def __init__(self, fn):
                    super().__init__()
                    self.fn = fn

                def forward(self, *args, **kwargs):
                    return self.fn(*args, **kwargs)

            exported_methods = {}
            # These cleanup passes are required to convert the `add` op to its out
            # variant, along with some other transformations.
            for method_name, method_input in input_map.items():
                wrapped_mod = WrapperModule(  # pyre-ignore[16]
                    getattr(eager_module, method_name)
                )
                exported_methods[method_name] = export(wrapped_mod, method_input)

            exec_prog = to_edge(exported_methods).to_executorch()

            # Create the ExecuTorch program from the graph.
            exec_prog.dump_executorch_program(verbose=True)
            return (exec_prog, inputs)

        ######### TEST CASES #########

        def test_e2e(tester):
            # Create an ExecuTorch program from ModuleAdd.
            exported_program, inputs = create_program(ModuleAdd())

            # Use pybindings to load and execute the program.
            executorch_module = load_fn(exported_program.buffer)
            executorch_output = executorch_module.forward(inputs)[0]

            # The test module adds the two inputs, so its output should be the same
            # as adding them directly.
            expected = inputs[0] + inputs[1]

            tester.assertEqual(str(expected), str(executorch_output))

        def test_multiple_entry(tester):

            program, inputs = create_program(ModuleMulti())
            executorch_module = load_fn(program.buffer)

            executorch_output = executorch_module.forward(inputs)[0]
            tester.assertTrue(torch.allclose(executorch_output, torch.ones(2, 2) * 2))

            executorch_output2 = executorch_module.run_method("forward2", inputs)[0]
            tester.assertTrue(torch.allclose(executorch_output2, torch.ones(2, 2) * 3))

        def test_output_lifespan(tester):
            def lower_function_call():
                program, inputs = create_program(ModuleMulti())
                executorch_module = load_fn(program.buffer)

                return executorch_module.forward(inputs)
                # executorch_module is destructed here and all of its memory is freed

            outputs = lower_function_call()
            tester.assertTrue(torch.allclose(outputs[0], torch.ones(2, 2) * 2))

        def test_module_callable(tester):
            # Create an ExecuTorch program from ModuleAdd.
            exported_program, inputs = create_program(ModuleAdd())

            # Use pybindings to load and execute the program.
            executorch_module = load_fn(exported_program.buffer)
            # Invoke the callable on executorch_module instead of calling module.forward.
            executorch_output = executorch_module(inputs)[0]

            # The test module adds the two inputs, so its output should be the same
            # as adding them directly.
            expected = inputs[0] + inputs[1]
            tester.assertEqual(str(expected), str(executorch_output))

        def test_module_single_input(tester):
            # Create an ExecuTorch program from ModuleAdd.
            exported_program, inputs = create_program(ModuleAddSingleInput())

            # Use pybindings to load and execute the program.
            executorch_module = load_fn(exported_program.buffer)
            # Inovke the callable on executorch_module instead of calling module.forward.
            # Use only one input to test this case.
            executorch_output = executorch_module(inputs[0])[0]

            # The test module adds the two inputs, so its output should be the same
            # as adding them directly.
            expected = inputs[0] + inputs[0]
            tester.assertEqual(str(expected), str(executorch_output))

        def test_stderr_redirect(tester):
            import sys
            from io import StringIO

            class RedirectedStderr:
                def __init__(self):
                    self._stderr = None
                    self._string_io = None

                def __enter__(self):
                    self._stderr = sys.stderr
                    sys.stderr = self._string_io = StringIO()
                    return self

                def __exit__(self, type, value, traceback):
                    sys.stderr = self._stderr

                def __str__(self):
                    return self._string_io.getvalue()

            with RedirectedStderr() as out:
                try:
                    # Create an ExecuTorch program from ModuleAdd.
                    exported_program, inputs = create_program(ModuleAdd())

                    # Use pybindings to load and execute the program.
                    executorch_module = load_fn(exported_program.buffer)

                    # add an extra input to trigger error
                    inputs = (*inputs, 1)

                    # Invoke the callable on executorch_module instead of calling module.forward.
                    executorch_output = executorch_module(inputs)[0]  # noqa
                    tester.assertFalse(True)  # should be unreachable
                except Exception:
                    tester.assertTrue(str(out).find("The length of given input array"))

        test_e2e(tester)
        test_multiple_entry(tester)
        test_output_lifespan(tester)
        test_module_callable(tester)
        test_module_single_input(tester)
        test_stderr_redirect(tester)

    return wrapper
