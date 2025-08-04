# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-unsafe

import unittest
from types import ModuleType
from typing import Any, Callable, Optional, Tuple

import torch
from executorch.exir import ExecutorchBackendConfig, ExecutorchProgramManager, to_edge
from executorch.exir.passes import MemoryPlanningPass
from torch.export import export


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


class ModuleChannelsLast(torch.nn.Module):
    """The module to serialize and execute."""

    def forward(self, x):
        return torch.nn.functional.interpolate(
            x,
            scale_factor=2,
            mode="nearest",
        )

    def get_methods_to_export(self):
        return ("forward",)

    def get_inputs(self):
        return (torch.ones(1, 2, 3, 4).to(memory_format=torch.channels_last),)


class ModuleChannelsLastInDefaultOut(torch.nn.Module):
    """The module to serialize and execute."""

    def forward(self, x):
        return torch.nn.functional.interpolate(
            x,
            scale_factor=2,
            mode="nearest",
        ).to(memory_format=torch.contiguous_format)

    def get_methods_to_export(self):
        return ("forward",)

    def get_inputs(self):
        return (torch.ones(1, 2, 3, 4).to(memory_format=torch.channels_last),)


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


class ModuleAddConstReturn(torch.nn.Module):
    """The module to serialize and execute."""

    def __init__(self):
        super(ModuleAddConstReturn, self).__init__()
        self.state = torch.ones(2, 2)

    def forward(self, x):
        return x + self.state, self.state

    def get_methods_to_export(self):
        return ("forward",)

    def get_inputs(self):
        return (torch.ones(2, 2),)


class ModuleAddWithAttributes(torch.nn.Module):
    """The module to serialize and execute."""

    def __init__(self):
        super(ModuleAddWithAttributes, self).__init__()
        self.register_buffer("state", torch.zeros(2, 2))

    def forward(self, x, y):
        self.state.add_(1)
        return x + y + self.state

    def get_methods_to_export(self):
        return ("forward",)

    def get_inputs(self):
        return (torch.ones(2, 2), torch.ones(2, 2))


def create_program(
    eager_module: torch.nn.Module,
    et_config: Optional[ExecutorchBackendConfig] = None,
) -> Tuple[ExecutorchProgramManager, Tuple[Any, ...]]:
    """Returns an executorch program based on ModuleAdd, along with inputs."""

    # Trace the test module and create a serialized ExecuTorch program.
    # pyre-fixme[29]: `Union[torch._tensor.Tensor, torch.nn.modules.module.Module]`
    #  is not a function.
    inputs = eager_module.get_inputs()
    input_map = {}
    # pyre-fixme[29]: `Union[torch._tensor.Tensor, torch.nn.modules.module.Module]`
    #  is not a function.
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
        wrapped_mod = WrapperModule(getattr(eager_module, method_name))
        exported_methods[method_name] = export(wrapped_mod, method_input, strict=True)

    exec_prog = to_edge(exported_methods).to_executorch(config=et_config)

    # Create the ExecuTorch program from the graph.
    exec_prog.dump_executorch_program(verbose=True)
    return (exec_prog, inputs)


def make_test(  # noqa: C901
    tester: unittest.TestCase,
    runtime: ModuleType,
) -> Callable[[unittest.TestCase], None]:
    """
    Returns a function that operates as a test case within a unittest.TestCase class.

    Used to allow the test code for pybindings to be shared across different pybinding libs
    which will all have different load functions. In this case each individual test case is a
    subfunction of wrapper.
    """
    load_fn: Callable = runtime._load_for_executorch_from_buffer
    load_prog_fn: Callable = runtime._load_program_from_buffer

    def wrapper(tester: unittest.TestCase) -> None:
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

        def test_quantized_ops(tester):
            eager_module = ModuleAdd()

            from executorch.exir import EdgeCompileConfig
            from executorch.exir.passes.quant_fusion_pass import QuantFusionPass
            from torch.ao.quantization import get_default_qconfig_mapping
            from torch.ao.quantization.backend_config.executorch import (
                get_executorch_backend_config,
            )
            from torch.ao.quantization.quantize_fx import (
                _convert_to_reference_decomposed_fx,
                prepare_fx,
            )

            qconfig_mapping = get_default_qconfig_mapping("qnnpack")
            example_inputs = (
                torch.ones(1, 5, dtype=torch.float32),
                torch.ones(1, 5, dtype=torch.float32),
            )
            m = prepare_fx(
                eager_module,
                qconfig_mapping,
                example_inputs,
                backend_config=get_executorch_backend_config(),
            )
            m = _convert_to_reference_decomposed_fx(m)
            config = EdgeCompileConfig(_check_ir_validity=False)
            m = to_edge(export(m, example_inputs, strict=True), compile_config=config)
            m = m.transform([QuantFusionPass(_fix_node_meta_val=True)])

            exec_prog = m.to_executorch()

            executorch_module = load_fn(exec_prog.buffer)
            executorch_output = executorch_module.forward(example_inputs)[0]

            expected = example_inputs[0] + example_inputs[1]
            tester.assertEqual(str(expected), str(executorch_output))

        def test_constant_output_not_memory_planned(tester):
            # Create an ExecuTorch program from ModuleAdd.
            exported_program, inputs = create_program(
                ModuleAddConstReturn(),
                et_config=ExecutorchBackendConfig(
                    memory_planning_pass=MemoryPlanningPass(alloc_graph_output=False)
                ),
            )

            exported_program.dump_executorch_program(verbose=True)

            # Use pybindings to load and execute the program.
            executorch_module = load_fn(exported_program.buffer)
            # Invoke the callable on executorch_module instead of calling module.forward.
            # Use only one input to test this case.
            executorch_output = executorch_module((torch.ones(2, 2),))
            print(executorch_output)

            # The test module adds the input to torch.ones(2,2), so its output should be the same
            # as adding them directly.
            expected = torch.ones(2, 2) + torch.ones(2, 2)
            tester.assertTrue(torch.allclose(expected, executorch_output[0]))

            # The test module returns the state. Check that its value is correct.
            tester.assertEqual(str(torch.ones(2, 2)), str(executorch_output[1]))

        def test_channels_last(tester) -> None:
            # Create an ExecuTorch program from ModuleChannelsLast.
            model = ModuleChannelsLast()
            exported_program, inputs = create_program(model)

            # Use pybindings to load and execute the program.
            executorch_module = load_fn(exported_program.buffer)
            # Inovke the callable on executorch_module instead of calling module.forward.
            # Use only one input to test this case.
            executorch_output = executorch_module(inputs[0])[0]

            # The test module adds the two inputs, so its output should be the same
            # as adding them directly.
            expected = model(inputs[0])
            tester.assertTrue(torch.allclose(expected, executorch_output))

        def test_unsupported_dim_order(tester) -> None:
            """
            Verify that the pybind layer rejects unsupported dim orders.
            """

            # Create an ExecuTorch program from ModuleChannelsLast.
            model = ModuleChannelsLast()
            exported_program, inputs = create_program(model)
            inputs = (
                torch.randn(1, 2, 3, 4, 5).to(memory_format=torch.channels_last_3d),
            )

            # Use pybindings to load and execute the program.
            executorch_module = load_fn(exported_program.buffer)

            # We expect execution to error because of the invalid input dim order.
            tester.assertRaises(RuntimeError, executorch_module, inputs[0])

        def test_channels_last_in_default_out(tester) -> None:
            # Create an ExecuTorch program from ModuleChannelsLastInDefaultOut.
            model = ModuleChannelsLastInDefaultOut()
            exported_program, inputs = create_program(model)

            # Use pybindings to load and execute the program.
            executorch_module = load_fn(exported_program.buffer)
            # Inovke the callable on executorch_module instead of calling module.forward.
            # Use only one input to test this case.
            executorch_output = executorch_module(inputs[0])[0]

            # The test module adds the two inputs, so its output should be the same
            # as adding them directly.
            expected = model(inputs[0])
            tester.assertTrue(torch.allclose(expected, executorch_output))

        def test_method_meta(tester) -> None:
            exported_program, inputs = create_program(ModuleAdd())

            # Use pybindings to load the program and query its metadata.
            executorch_module = load_fn(exported_program.buffer)
            meta = executorch_module.method_meta("forward")

            # Ensure that all these APIs work even if the module object is destroyed.
            del executorch_module
            tester.assertEqual(meta.name(), "forward")
            tester.assertEqual(meta.num_inputs(), 2)
            tester.assertEqual(meta.num_outputs(), 1)
            # Common string for all these tensors.
            tensor_info = "TensorInfo(sizes=[2, 2], dtype=Float, is_memory_planned=True, nbytes=16)"
            float_dtype = 6
            tester.assertEqual(
                str(meta),
                "MethodMeta(name='forward', num_inputs=2, "
                f"input_tensor_meta=['{tensor_info}', '{tensor_info}'], "
                f"num_outputs=1, output_tensor_meta=['{tensor_info}'])",
            )

            input_tensors = [meta.input_tensor_meta(i) for i in range(2)]
            output_tensor = meta.output_tensor_meta(0)
            # Check that accessing out of bounds raises IndexError.
            with tester.assertRaises(IndexError):
                meta.input_tensor_meta(2)
            # Test that tensor metadata can outlive method metadata.
            del meta
            tester.assertEqual([t.sizes() for t in input_tensors], [(2, 2), (2, 2)])
            tester.assertEqual(
                [t.dtype() for t in input_tensors], [float_dtype, float_dtype]
            )
            tester.assertEqual(
                [t.is_memory_planned() for t in input_tensors], [True, True]
            )
            tester.assertEqual([t.nbytes() for t in input_tensors], [16, 16])
            tester.assertEqual(str(input_tensors), f"[{tensor_info}, {tensor_info}]")

            tester.assertEqual(output_tensor.sizes(), (2, 2))
            tester.assertEqual(output_tensor.dtype(), float_dtype)
            tester.assertEqual(output_tensor.is_memory_planned(), True)
            tester.assertEqual(output_tensor.nbytes(), 16)
            tester.assertEqual(str(output_tensor), tensor_info)

        def test_bad_name(tester) -> None:
            # Create an ExecuTorch program from ModuleAdd.
            exported_program, inputs = create_program(ModuleAdd())

            # Use pybindings to load and execute the program.
            executorch_module = load_fn(exported_program.buffer)
            # Invoke the callable on executorch_module instead of calling module.forward.
            with tester.assertRaises(RuntimeError):
                executorch_module.run_method("not_a_real_method", inputs)

        def test_verification_config(tester) -> None:
            # Create an ExecuTorch program from ModuleAdd.
            exported_program, inputs = create_program(ModuleAdd())
            Verification = runtime.Verification

            # Use pybindings to load and execute the program.
            for config in [Verification.Minimal, Verification.InternalConsistency]:
                executorch_module = load_fn(
                    exported_program.buffer,
                    enable_etdump=False,
                    debug_buffer_size=0,
                    program_verification=config,
                )

                executorch_output = executorch_module.forward(inputs)[0]

                # The test module adds the two inputs, so its output should be the same
                # as adding them directly.
                expected = inputs[0] + inputs[1]

                tester.assertEqual(str(expected), str(executorch_output))

        def test_unsupported_input_type(tester):
            exported_program, inputs = create_program(ModuleAdd())
            executorch_module = load_fn(exported_program.buffer)

            # Pass an unsupported input type to the module.
            inputs = ([*inputs],)

            # This should raise a Python error, not hit a fatal assert in the C++ code.
            tester.assertRaises(RuntimeError, executorch_module, inputs)

        def test_program_methods_one(tester):
            # Create an ExecuTorch program from ModuleAdd.
            exported_program, _ = create_program(ModuleAdd())

            # Use pybindings to load the program.
            executorch_program = load_prog_fn(exported_program.buffer)

            tester.assertEqual(executorch_program.num_methods(), 1)
            tester.assertEqual(executorch_program.get_method_name(0), "forward")

        def test_program_methods_multi(tester):
            # Create an ExecuTorch program from ModuleMulti.
            exported_program, _ = create_program(ModuleMulti())

            # Use pybindings to load the program.
            executorch_program = load_prog_fn(exported_program.buffer)

            tester.assertEqual(executorch_program.num_methods(), 2)
            tester.assertEqual(executorch_program.get_method_name(0), "forward")
            tester.assertEqual(executorch_program.get_method_name(1), "forward2")

        def test_program_method_index_out_of_bounds(tester):
            # Create an ExecuTorch program from ModuleMulti.
            exported_program, _ = create_program(ModuleMulti())

            # Use pybindings to load the program.
            executorch_program = load_prog_fn(exported_program.buffer)

            tester.assertRaises(RuntimeError, executorch_program.get_method_name, 2)

        def test_method_e2e(tester):
            # Create an ExecuTorch program from ModuleAdd.
            exported_program, inputs = create_program(ModuleAdd())

            # Use pybindings to load the program.
            executorch_program = load_prog_fn(exported_program.buffer)

            # Use pybindings to load and execute the method.
            executorch_method = executorch_program.load_method("forward")
            executorch_output = executorch_method.call(inputs)[0]

            # The test module adds the two inputs, so its output should be the same
            # as adding them directly.
            expected = inputs[0] + inputs[1]

            tester.assertEqual(str(expected), str(executorch_output))

        def test_method_output_lifespan(tester):
            def lower_function_call():
                program, inputs = create_program(ModuleMulti())
                executorch_program = load_prog_fn(program.buffer)

                executorch_method = executorch_program.load_method("forward")
                return executorch_method.call(inputs)
                # executorch_program is destructed here and all of its memory is freed

            outputs = lower_function_call()
            tester.assertTrue(torch.allclose(outputs[0], torch.ones(2, 2) * 2))

        def test_method_multiple_entry(tester):
            program, inputs = create_program(ModuleMulti())
            executorch_program = load_prog_fn(program.buffer)

            executorch_method = executorch_program.load_method("forward")
            executorch_output = executorch_method.call(inputs)[0]
            tester.assertTrue(torch.allclose(executorch_output, torch.ones(2, 2) * 2))

            executorch_method2 = executorch_program.load_method("forward2")
            executorch_output2 = executorch_method2.call(inputs)[0]
            tester.assertTrue(torch.allclose(executorch_output2, torch.ones(2, 2) * 3))

        def test_method_by_parts(tester):
            # Create an ExecuTorch program from ModuleAdd.
            exported_program, inputs = create_program(ModuleAdd())

            # Use pybindings to load the program.
            executorch_program = load_prog_fn(exported_program.buffer)

            # Use pybindings to load and the method.
            executorch_method = executorch_program.load_method("forward")

            # Call each part separately.
            executorch_method.set_inputs(inputs)
            executorch_method.execute()
            executorch_output = executorch_method.get_outputs()[0]

            # The test module adds the two inputs, so its output should be the same
            # as adding them directly.
            expected = inputs[0] + inputs[1]

            tester.assertEqual(str(expected), str(executorch_output))

        def test_method_callable(tester):
            # Create an ExecuTorch program from ModuleAdd.
            exported_program, inputs = create_program(ModuleAdd())

            # Use pybindings to load and execute the program.
            executorch_program = load_prog_fn(exported_program.buffer)
            # Invoke the callable on executorch_method instead of calling module.forward.
            executorch_method = executorch_program.load_method("forward")
            executorch_output = executorch_method(inputs)[0]

            # The test module adds the two inputs, so its output should be the same
            # as adding them directly.
            expected = inputs[0] + inputs[1]
            tester.assertEqual(str(expected), str(executorch_output))

        def test_method_single_input(tester):
            # Create an ExecuTorch program from ModuleAdd.
            exported_program, inputs = create_program(ModuleAddSingleInput())

            # Use pybindings to load and execute the program.
            executorch_program = load_prog_fn(exported_program.buffer)
            # Inovke the callable on executorch_method instead of calling module.forward.
            # Use only one input to test this case.
            executorch_method = executorch_program.load_method("forward")
            executorch_output = executorch_method(inputs[0])[0]

            # The test module adds the two inputs, so its output should be the same
            # as adding them directly.
            expected = inputs[0] + inputs[0]
            tester.assertEqual(str(expected), str(executorch_output))

        def test_method_stderr_redirect(tester):
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
                    program, inputs = create_program(ModuleAdd())

                    # Use pybindings to load the program.
                    executorch_program = load_prog_fn(program.buffer)

                    # Use pybindings to load and execute the method.
                    executorch_method = executorch_program.load_method("forward")

                    # add an extra input to trigger error
                    inputs = (*inputs, 1)

                    # Invoke the callable on executorch_module instead of calling module.forward.
                    executorch_output = executorch_method(inputs)[0]  # noqa
                    tester.assertFalse(True)  # should be unreachable
                except Exception:
                    tester.assertTrue(str(out).find("The length of given input array"))

        def test_method_quantized_ops(tester):
            eager_module = ModuleAdd()

            from executorch.exir import EdgeCompileConfig
            from executorch.exir.passes.quant_fusion_pass import QuantFusionPass
            from torch.ao.quantization import get_default_qconfig_mapping
            from torch.ao.quantization.backend_config.executorch import (
                get_executorch_backend_config,
            )
            from torch.ao.quantization.quantize_fx import (
                _convert_to_reference_decomposed_fx,
                prepare_fx,
            )

            qconfig_mapping = get_default_qconfig_mapping("qnnpack")
            example_inputs = (
                torch.ones(1, 5, dtype=torch.float32),
                torch.ones(1, 5, dtype=torch.float32),
            )
            m = prepare_fx(
                eager_module,
                qconfig_mapping,
                example_inputs,
                backend_config=get_executorch_backend_config(),
            )
            m = _convert_to_reference_decomposed_fx(m)
            config = EdgeCompileConfig(_check_ir_validity=False)
            m = to_edge(export(m, example_inputs, strict=True), compile_config=config)
            m = m.transform([QuantFusionPass(_fix_node_meta_val=True)])

            exec_prog = m.to_executorch()

            executorch_program = load_prog_fn(exec_prog.buffer)
            executorch_method = executorch_program.load_method("forward")
            executorch_output = executorch_method(example_inputs)[0]

            expected = example_inputs[0] + example_inputs[1]
            tester.assertEqual(str(expected), str(executorch_output))

        def test_method_constant_output_not_memory_planned(tester):
            # Create an ExecuTorch program from ModuleAdd.
            exported_program, _ = create_program(
                ModuleAddConstReturn(),
                et_config=ExecutorchBackendConfig(
                    memory_planning_pass=MemoryPlanningPass(alloc_graph_output=False)
                ),
            )

            # Use pybindings to load and execute the program.
            executorch_program = load_prog_fn(exported_program.buffer)
            # Invoke the callable on executorch_module instead of calling module.forward.
            # Use only one input to test this case.
            executorch_method = executorch_program.load_method("forward")
            executorch_output = executorch_method((torch.ones(2, 2),))

            # The test module adds the input to torch.ones(2,2), so its output should be the same
            # as adding them directly.
            expected = torch.ones(2, 2) + torch.ones(2, 2)
            tester.assertTrue(torch.allclose(expected, executorch_output[0]))

            # The test module returns the state. Check that its value is correct.
            tester.assertEqual(str(torch.ones(2, 2)), str(executorch_output[1]))

        def test_method_channels_last(tester) -> None:
            # Create an ExecuTorch program from ModuleChannelsLast.
            model = ModuleChannelsLast()
            exported_program, inputs = create_program(model)

            # Use pybindings to load and execute the program.
            executorch_program = load_prog_fn(exported_program.buffer)
            # Inovke the callable on executorch_module instead of calling module.forward.
            # Use only one input to test this case.
            executorch_method = executorch_program.load_method("forward")
            executorch_output = executorch_method(inputs[0])[0]

            # The test module adds the two inputs, so its output should be the same
            # as adding them directly.
            expected = model(inputs[0])
            tester.assertTrue(torch.allclose(expected, executorch_output))

        def test_method_unsupported_dim_order(tester) -> None:
            """
            Verify that the pybind layer rejects unsupported dim orders.
            """

            # Create an ExecuTorch program from ModuleChannelsLast.
            model = ModuleChannelsLast()
            exported_program, inputs = create_program(model)
            inputs = (
                torch.randn(1, 2, 3, 4, 5).to(memory_format=torch.channels_last_3d),
            )

            # Use pybindings to load and execute the program.
            executorch_program = load_prog_fn(exported_program.buffer)
            executorch_method = executorch_program.load_method("forward")

            # We expect execution to error because of the invalid input dim order.
            tester.assertRaises(RuntimeError, executorch_method, inputs[0])

        def test_method_channels_last_in_default_out(tester) -> None:
            # Create an ExecuTorch program from ModuleChannelsLastInDefaultOut.
            model = ModuleChannelsLastInDefaultOut()
            exported_program, inputs = create_program(model)

            # Use pybindings to load and execute the program.
            executorch_program = load_prog_fn(exported_program.buffer)
            # Inovke the callable on executorch_module instead of calling module.forward.
            # Use only one input to test this case.
            executorch_method = executorch_program.load_method("forward")
            executorch_output = executorch_method(inputs[0])[0]

            # The test module adds the two inputs, so its output should be the same
            # as adding them directly.
            expected = model(inputs[0])
            tester.assertTrue(torch.allclose(expected, executorch_output))

        def test_method_bad_name(tester) -> None:
            # Create an ExecuTorch program from ModuleAdd.
            exported_program, inputs = create_program(ModuleAdd())

            # Use pybindings to load and execute the program.
            executorch_program = load_prog_fn(exported_program.buffer)
            # Invoke the callable on executorch_module instead of calling module.forward.
            with tester.assertRaises(RuntimeError):
                executorch_program.load_method("not_a_real_method")

        def test_program_verification_config(tester) -> None:
            # Create an ExecuTorch program from ModuleAdd.
            exported_program, inputs = create_program(ModuleAdd())
            Verification = runtime.Verification

            # Use pybindings to load and execute the program.
            for config in [Verification.Minimal, Verification.InternalConsistency]:
                executorch_program = load_prog_fn(
                    exported_program.buffer,
                    enable_etdump=False,
                    debug_buffer_size=0,
                    program_verification=config,
                )

                executorch_method = executorch_program.load_method("forward")
                executorch_output = executorch_method(inputs)[0]

                # The test module adds the two inputs, so its output should be the same
                # as adding them directly.
                expected = inputs[0] + inputs[1]

                tester.assertEqual(str(expected), str(executorch_output))

        def test_method_unsupported_input_type(tester):
            exported_program, inputs = create_program(ModuleAdd())
            executorch_program = load_prog_fn(exported_program.buffer)

            # Pass an unsupported input type to the module.
            inputs = ([*inputs],)

            # This should raise a Python error, not hit a fatal assert in the C++ code.
            executorch_method = executorch_program.load_method("forward")
            tester.assertRaises(RuntimeError, executorch_method, inputs)

        def test_method_attribute(tester):
            eager_module = ModuleAddWithAttributes()

            # Trace the test module and create a serialized ExecuTorch program.
            inputs = eager_module.get_inputs()

            exported_program = export(eager_module, inputs, strict=True)
            exec_prog = to_edge(exported_program).to_executorch(
                config=ExecutorchBackendConfig(
                    emit_mutable_buffer_names=True,
                )
            )

            # Create the ExecuTorch program from the graph.
            exec_prog.dump_executorch_program(verbose=True)

            # Use pybindings to load the program.
            executorch_program = load_prog_fn(exec_prog.buffer)

            # Use pybindings to load and execute the method.
            executorch_method = executorch_program.load_method("forward")
            executorch_method(inputs)
            tester.assertEqual(
                str(executorch_method.get_attribute("state")), str(torch.ones(2, 2))
            )

        def test_program_method_meta(tester) -> None:
            exported_program, inputs = create_program(ModuleAdd())

            # Use pybindings to load the program and query its metadata.
            executorch_program = load_prog_fn(exported_program.buffer)
            meta = executorch_program.method_meta("forward")

            # Ensure that all these APIs work even if the module object is destroyed.
            del executorch_program
            tester.assertEqual(meta.name(), "forward")
            tester.assertEqual(meta.num_inputs(), 2)
            tester.assertEqual(meta.num_outputs(), 1)
            # Common string for all these tensors.
            tensor_info = "TensorInfo(sizes=[2, 2], dtype=Float, is_memory_planned=True, nbytes=16)"
            float_dtype = 6
            tester.assertEqual(
                str(meta),
                "MethodMeta(name='forward', num_inputs=2, "
                f"input_tensor_meta=['{tensor_info}', '{tensor_info}'], "
                f"num_outputs=1, output_tensor_meta=['{tensor_info}'])",
            )

            input_tensors = [meta.input_tensor_meta(i) for i in range(2)]
            output_tensor = meta.output_tensor_meta(0)
            # Check that accessing out of bounds raises IndexError.
            with tester.assertRaises(IndexError):
                meta.input_tensor_meta(2)
            # Test that tensor metadata can outlive method metadata.
            del meta
            tester.assertEqual([t.sizes() for t in input_tensors], [(2, 2), (2, 2)])
            tester.assertEqual(
                [t.dtype() for t in input_tensors], [float_dtype, float_dtype]
            )
            tester.assertEqual(
                [t.is_memory_planned() for t in input_tensors], [True, True]
            )
            tester.assertEqual([t.nbytes() for t in input_tensors], [16, 16])
            tester.assertEqual(str(input_tensors), f"[{tensor_info}, {tensor_info}]")

            tester.assertEqual(output_tensor.sizes(), (2, 2))
            tester.assertEqual(output_tensor.dtype(), float_dtype)
            tester.assertEqual(output_tensor.is_memory_planned(), True)
            tester.assertEqual(output_tensor.nbytes(), 16)
            tester.assertEqual(str(output_tensor), tensor_info)

        def test_method_method_meta(tester) -> None:
            exported_program, inputs = create_program(ModuleAdd())

            # Use pybindings to load the program and query its metadata.
            executorch_program = load_prog_fn(exported_program.buffer)
            executorch_method = executorch_program.load_method("forward")
            meta = executorch_method.method_meta()

            # Ensure that all these APIs work even if the module object is destroyed.
            del executorch_program
            del executorch_method
            tester.assertEqual(meta.name(), "forward")
            tester.assertEqual(meta.num_inputs(), 2)
            tester.assertEqual(meta.num_outputs(), 1)
            # Common string for all these tensors.
            tensor_info = "TensorInfo(sizes=[2, 2], dtype=Float, is_memory_planned=True, nbytes=16)"
            float_dtype = 6
            tester.assertEqual(
                str(meta),
                "MethodMeta(name='forward', num_inputs=2, "
                f"input_tensor_meta=['{tensor_info}', '{tensor_info}'], "
                f"num_outputs=1, output_tensor_meta=['{tensor_info}'])",
            )

            input_tensors = [meta.input_tensor_meta(i) for i in range(2)]
            output_tensor = meta.output_tensor_meta(0)
            # Check that accessing out of bounds raises IndexError.
            with tester.assertRaises(IndexError):
                meta.input_tensor_meta(2)
            # Test that tensor metadata can outlive method metadata.
            del meta
            tester.assertEqual([t.sizes() for t in input_tensors], [(2, 2), (2, 2)])
            tester.assertEqual(
                [t.dtype() for t in input_tensors], [float_dtype, float_dtype]
            )
            tester.assertEqual(
                [t.is_memory_planned() for t in input_tensors], [True, True]
            )
            tester.assertEqual([t.nbytes() for t in input_tensors], [16, 16])
            tester.assertEqual(str(input_tensors), f"[{tensor_info}, {tensor_info}]")

            tester.assertEqual(output_tensor.sizes(), (2, 2))
            tester.assertEqual(output_tensor.dtype(), float_dtype)
            tester.assertEqual(output_tensor.is_memory_planned(), True)
            tester.assertEqual(output_tensor.nbytes(), 16)
            tester.assertEqual(str(output_tensor), tensor_info)

        ######### RUN TEST CASES #########
        test_e2e(tester)
        test_multiple_entry(tester)
        test_output_lifespan(tester)
        test_module_callable(tester)
        test_module_single_input(tester)
        test_stderr_redirect(tester)
        test_quantized_ops(tester)
        test_channels_last(tester)
        test_channels_last_in_default_out(tester)
        test_unsupported_dim_order(tester)
        test_constant_output_not_memory_planned(tester)
        test_method_meta(tester)
        test_bad_name(tester)
        test_verification_config(tester)
        test_unsupported_input_type(tester)
        test_program_methods_one(tester)
        test_program_methods_multi(tester)
        test_program_method_index_out_of_bounds(tester)
        test_method_e2e(tester)
        test_method_output_lifespan(tester)
        test_method_multiple_entry(tester)
        test_method_by_parts(tester)
        test_method_callable(tester)
        test_method_single_input(tester)
        test_method_stderr_redirect(tester)
        test_method_quantized_ops(tester)
        test_method_constant_output_not_memory_planned(tester)
        test_method_channels_last(tester)
        test_method_unsupported_dim_order(tester)
        test_method_channels_last_in_default_out(tester)
        test_method_bad_name(tester)
        test_program_verification_config(tester)
        test_method_unsupported_input_type(tester)
        test_method_attribute(tester)
        test_program_method_meta(tester)
        test_method_method_meta(tester)

    return wrapper
