# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-unsafe

import sys
import unittest
from io import StringIO

import torch

from executorch.exir import ExecutorchBackendConfig, to_edge
from executorch.exir.passes import MemoryPlanningPass
from executorch.extension.pybindings.test.make_test import (
    create_program,
    ModuleAdd,
    ModuleAddConstReturn,
    ModuleAddSingleInput,
    ModuleAddWithAttributes,
    ModuleChannelsLast,
    ModuleChannelsLastInDefaultOut,
    ModuleLinear,
    ModuleMulti,
)
from torch.export import export


class PybindingsTest(unittest.TestCase):
    def setUp(self):
        # Will test both portable and aten
        kernel_mode = None
        try:
            from executorch.extension.pybindings import portable_lib as runtime

            kernel_mode = "portable"
        except Exception:
            print("can't load portable lib")

        if kernel_mode is None:
            try:
                from executorch.extension.pybindings import (  # noqa: F811
                    aten_lib as runtime,
                )

                kernel_mode = "aten"
            except Exception:
                print("can't load aten lib")

        assert kernel_mode is not None
        self.load_fn = runtime._load_for_executorch_from_buffer
        self.load_prog_fn = runtime._load_program_from_buffer
        self.runtime = runtime

    def test_e2e(self):
        exported_program, inputs = create_program(ModuleAdd())
        executorch_module = self.load_fn(exported_program.buffer)
        executorch_output = executorch_module.forward(inputs)[0]
        expected = inputs[0] + inputs[1]
        self.assertEqual(str(expected), str(executorch_output))

    def test_multiple_entry(self):
        program, inputs = create_program(ModuleMulti())
        executorch_module = self.load_fn(program.buffer)

        executorch_output = executorch_module.forward(inputs)[0]
        self.assertTrue(torch.allclose(executorch_output, torch.ones(2, 2) * 2))

        executorch_output2 = executorch_module.run_method("forward2", inputs)[0]
        self.assertTrue(torch.allclose(executorch_output2, torch.ones(2, 2) * 3))

    def test_output_lifespan(self):
        def lower_function_call():
            program, inputs = create_program(ModuleMulti())
            executorch_module = self.load_fn(program.buffer)
            return executorch_module.forward(inputs)

        outputs = lower_function_call()
        self.assertTrue(torch.allclose(outputs[0], torch.ones(2, 2) * 2))

    def test_module_callable(self):
        exported_program, inputs = create_program(ModuleAdd())
        executorch_module = self.load_fn(exported_program.buffer)
        executorch_output = executorch_module(inputs)[0]
        expected = inputs[0] + inputs[1]
        self.assertEqual(str(expected), str(executorch_output))

    def test_module_single_input(self):
        exported_program, inputs = create_program(ModuleAddSingleInput())
        executorch_module = self.load_fn(exported_program.buffer)
        executorch_output = executorch_module(inputs[0])[0]
        expected = inputs[0] + inputs[0]
        self.assertEqual(str(expected), str(executorch_output))

    def test_stderr_redirect(self):
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
                exported_program, inputs = create_program(ModuleAdd())
                executorch_module = self.load_fn(exported_program.buffer)
                inputs = (*inputs, 1)
                executorch_output = executorch_module(inputs)[0]  # noqa
                self.assertFalse(True)  # should be unreachable
            except Exception:
                self.assertTrue(str(out).find("The length of given input array"))

    def test_quantized_ops(self):
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

        executorch_module = self.load_fn(exec_prog.buffer)
        executorch_output = executorch_module.forward(example_inputs)[0]

        expected = example_inputs[0] + example_inputs[1]
        self.assertEqual(str(expected), str(executorch_output))

    def test_constant_output_not_memory_planned(self):
        exported_program, inputs = create_program(
            ModuleAddConstReturn(),
            et_config=ExecutorchBackendConfig(
                memory_planning_pass=MemoryPlanningPass(alloc_graph_output=False)
            ),
        )

        exported_program.dump_executorch_program(verbose=True)

        executorch_module = self.load_fn(exported_program.buffer)
        executorch_output = executorch_module((torch.ones(2, 2),))

        expected = torch.ones(2, 2) + torch.ones(2, 2)
        self.assertTrue(torch.allclose(expected, executorch_output[0]))
        self.assertEqual(str(torch.ones(2, 2)), str(executorch_output[1]))

    def test_channels_last(self) -> None:
        model = ModuleChannelsLast()
        exported_program, inputs = create_program(model)

        executorch_module = self.load_fn(exported_program.buffer)
        executorch_output = executorch_module(inputs[0])[0]

        expected = model(inputs[0])
        self.assertTrue(torch.allclose(expected, executorch_output))

    def test_unsupported_dim_order(self) -> None:
        model = ModuleChannelsLast()
        exported_program, inputs = create_program(model)
        inputs = (torch.randn(1, 2, 3, 4, 5).to(memory_format=torch.channels_last_3d),)

        executorch_module = self.load_fn(exported_program.buffer)
        self.assertRaises(RuntimeError, executorch_module, inputs[0])

    def test_channels_last_in_default_out(self) -> None:
        model = ModuleChannelsLastInDefaultOut()
        exported_program, inputs = create_program(model)

        executorch_module = self.load_fn(exported_program.buffer)
        executorch_output = executorch_module(inputs[0])[0]

        expected = model(inputs[0])
        self.assertTrue(torch.allclose(expected, executorch_output))

    def test_method_meta(self) -> None:
        exported_program, inputs = create_program(ModuleAdd())

        executorch_module = self.load_fn(exported_program.buffer)
        meta = executorch_module.method_meta("forward")

        del executorch_module
        self.assertEqual(meta.name(), "forward")
        self.assertEqual(meta.num_inputs(), 2)
        self.assertEqual(meta.num_outputs(), 1)

        tensor_info = (
            "TensorInfo(sizes=[2, 2], dtype=Float, is_memory_planned=True, nbytes=16)"
        )
        float_dtype = 6
        self.assertEqual(
            str(meta),
            "MethodMeta(name='forward', num_inputs=2, "
            f"input_tensor_meta=['{tensor_info}', '{tensor_info}'], "
            f"num_outputs=1, output_tensor_meta=['{tensor_info}'])",
        )

        input_tensors = [meta.input_tensor_meta(i) for i in range(2)]
        output_tensor = meta.output_tensor_meta(0)

        with self.assertRaises(IndexError):
            meta.input_tensor_meta(2)

        del meta
        self.assertEqual([t.sizes() for t in input_tensors], [(2, 2), (2, 2)])
        self.assertEqual([t.dtype() for t in input_tensors], [float_dtype, float_dtype])
        self.assertEqual([t.is_memory_planned() for t in input_tensors], [True, True])
        self.assertEqual([t.nbytes() for t in input_tensors], [16, 16])
        self.assertEqual(str(input_tensors), f"[{tensor_info}, {tensor_info}]")

        self.assertEqual(output_tensor.sizes(), (2, 2))
        self.assertEqual(output_tensor.dtype(), float_dtype)
        self.assertEqual(output_tensor.is_memory_planned(), True)
        self.assertEqual(output_tensor.nbytes(), 16)
        self.assertEqual(str(output_tensor), tensor_info)

    def test_bad_name(self) -> None:
        exported_program, inputs = create_program(ModuleAdd())
        executorch_module = self.load_fn(exported_program.buffer)

        with self.assertRaises(RuntimeError):
            executorch_module.run_method("not_a_real_method", inputs)

    def test_verification_config(self) -> None:
        exported_program, inputs = create_program(ModuleAdd())
        Verification = self.runtime.Verification

        for config in [Verification.Minimal, Verification.InternalConsistency]:
            executorch_module = self.load_fn(
                exported_program.buffer,
                enable_etdump=False,
                debug_buffer_size=0,
                program_verification=config,
            )

            executorch_output = executorch_module.forward(inputs)[0]
            expected = inputs[0] + inputs[1]
            self.assertEqual(str(expected), str(executorch_output))

    def test_unsupported_input_type(self):
        exported_program, inputs = create_program(ModuleAdd())
        executorch_module = self.load_fn(exported_program.buffer)
        inputs = ([*inputs],)
        self.assertRaises(RuntimeError, executorch_module, inputs)

    def test_program_methods_one(self):
        exported_program, _ = create_program(ModuleAdd())
        executorch_program = self.load_prog_fn(exported_program.buffer)

        self.assertEqual(executorch_program.num_methods(), 1)
        self.assertEqual(executorch_program.get_method_name(0), "forward")

    def test_program_methods_multi(self):
        exported_program, _ = create_program(ModuleMulti())
        executorch_program = self.load_prog_fn(exported_program.buffer)

        self.assertEqual(executorch_program.num_methods(), 2)
        self.assertEqual(executorch_program.get_method_name(0), "forward")
        self.assertEqual(executorch_program.get_method_name(1), "forward2")

    def test_program_method_index_out_of_bounds(self):
        exported_program, _ = create_program(ModuleMulti())
        executorch_program = self.load_prog_fn(exported_program.buffer)
        self.assertRaises(RuntimeError, executorch_program.get_method_name, 2)

    def test_method_e2e(self):
        exported_program, inputs = create_program(ModuleAdd())
        executorch_program = self.load_prog_fn(exported_program.buffer)
        executorch_method = executorch_program.load_method("forward")
        executorch_output = executorch_method.call(inputs)[0]
        expected = inputs[0] + inputs[1]
        self.assertEqual(str(expected), str(executorch_output))

    def test_method_output_lifespan(self):
        def lower_function_call():
            program, inputs = create_program(ModuleMulti())
            executorch_program = self.load_prog_fn(program.buffer)
            executorch_method = executorch_program.load_method("forward")
            return executorch_method.call(inputs)

        outputs = lower_function_call()
        self.assertTrue(torch.allclose(outputs[0], torch.ones(2, 2) * 2))

    def test_method_multiple_entry(self):
        program, inputs = create_program(ModuleMulti())
        executorch_program = self.load_prog_fn(program.buffer)

        executorch_method = executorch_program.load_method("forward")
        executorch_output = executorch_method.call(inputs)[0]
        self.assertTrue(torch.allclose(executorch_output, torch.ones(2, 2) * 2))

        executorch_method2 = executorch_program.load_method("forward2")
        executorch_output2 = executorch_method2.call(inputs)[0]
        self.assertTrue(torch.allclose(executorch_output2, torch.ones(2, 2) * 3))

    def test_method_by_parts(self):
        exported_program, inputs = create_program(ModuleAdd())
        executorch_program = self.load_prog_fn(exported_program.buffer)
        executorch_method = executorch_program.load_method("forward")

        executorch_method.set_inputs(inputs)
        executorch_method.execute()
        executorch_output = executorch_method.get_outputs()[0]

        expected = inputs[0] + inputs[1]
        self.assertEqual(str(expected), str(executorch_output))

    def test_method_callable(self):
        exported_program, inputs = create_program(ModuleAdd())
        executorch_program = self.load_prog_fn(exported_program.buffer)
        executorch_method = executorch_program.load_method("forward")
        executorch_output = executorch_method(inputs)[0]
        expected = inputs[0] + inputs[1]
        self.assertEqual(str(expected), str(executorch_output))

    def test_method_single_input(self):
        exported_program, inputs = create_program(ModuleAddSingleInput())
        executorch_program = self.load_prog_fn(exported_program.buffer)
        executorch_method = executorch_program.load_method("forward")
        executorch_output = executorch_method(inputs[0])[0]
        expected = inputs[0] + inputs[0]
        self.assertEqual(str(expected), str(executorch_output))

    def test_method_stderr_redirect(self):
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
                program, inputs = create_program(ModuleAdd())
                executorch_program = self.load_prog_fn(program.buffer)
                executorch_method = executorch_program.load_method("forward")
                inputs = (*inputs, 1)
                executorch_output = executorch_method(inputs)[0]  # noqa
                self.assertFalse(True)  # should be unreachable
            except Exception:
                self.assertTrue(str(out).find("The length of given input array"))

    def test_method_quantized_ops(self):
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

        executorch_program = self.load_prog_fn(exec_prog.buffer)
        executorch_method = executorch_program.load_method("forward")
        executorch_output = executorch_method(example_inputs)[0]

        expected = example_inputs[0] + example_inputs[1]
        self.assertEqual(str(expected), str(executorch_output))

    def test_method_constant_output_not_memory_planned(self):
        exported_program, _ = create_program(
            ModuleAddConstReturn(),
            et_config=ExecutorchBackendConfig(
                memory_planning_pass=MemoryPlanningPass(alloc_graph_output=False)
            ),
        )

        executorch_program = self.load_prog_fn(exported_program.buffer)
        executorch_method = executorch_program.load_method("forward")
        executorch_output = executorch_method((torch.ones(2, 2),))

        expected = torch.ones(2, 2) + torch.ones(2, 2)
        self.assertTrue(torch.allclose(expected, executorch_output[0]))
        self.assertEqual(str(torch.ones(2, 2)), str(executorch_output[1]))

    def test_method_channels_last(self) -> None:
        model = ModuleChannelsLast()
        exported_program, inputs = create_program(model)

        executorch_program = self.load_prog_fn(exported_program.buffer)
        executorch_method = executorch_program.load_method("forward")
        executorch_output = executorch_method(inputs[0])[0]

        expected = model(inputs[0])
        self.assertTrue(torch.allclose(expected, executorch_output))

    def test_method_unsupported_dim_order(self) -> None:
        model = ModuleChannelsLast()
        exported_program, inputs = create_program(model)
        inputs = (torch.randn(1, 2, 3, 4, 5).to(memory_format=torch.channels_last_3d),)

        executorch_program = self.load_prog_fn(exported_program.buffer)
        executorch_method = executorch_program.load_method("forward")
        self.assertRaises(RuntimeError, executorch_method, inputs[0])

    def test_method_channels_last_in_default_out(self) -> None:
        model = ModuleChannelsLastInDefaultOut()
        exported_program, inputs = create_program(model)

        executorch_program = self.load_prog_fn(exported_program.buffer)
        executorch_method = executorch_program.load_method("forward")
        executorch_output = executorch_method(inputs[0])[0]

        expected = model(inputs[0])
        self.assertTrue(torch.allclose(expected, executorch_output))

    def test_method_bad_name(self) -> None:
        exported_program, inputs = create_program(ModuleAdd())
        executorch_program = self.load_prog_fn(exported_program.buffer)

        with self.assertRaises(RuntimeError):
            executorch_program.load_method("not_a_real_method")

    def test_program_verification_config(self) -> None:
        exported_program, inputs = create_program(ModuleAdd())
        Verification = self.runtime.Verification

        for config in [Verification.Minimal, Verification.InternalConsistency]:
            executorch_program = self.load_prog_fn(
                exported_program.buffer,
                enable_etdump=False,
                debug_buffer_size=0,
                program_verification=config,
            )

            executorch_method = executorch_program.load_method("forward")
            executorch_output = executorch_method(inputs)[0]

            expected = inputs[0] + inputs[1]
            self.assertEqual(str(expected), str(executorch_output))

    def test_method_unsupported_input_type(self):
        exported_program, inputs = create_program(ModuleAdd())
        executorch_program = self.load_prog_fn(exported_program.buffer)
        inputs = ([*inputs],)
        executorch_method = executorch_program.load_method("forward")
        self.assertRaises(RuntimeError, executorch_method, inputs)

    def test_method_attribute(self):
        eager_module = ModuleAddWithAttributes()
        inputs = eager_module.get_inputs()

        exported_program = export(eager_module, inputs, strict=True)
        exec_prog = to_edge(exported_program).to_executorch(
            config=ExecutorchBackendConfig(
                emit_mutable_buffer_names=True,
            )
        )

        exec_prog.dump_executorch_program(verbose=True)

        executorch_program = self.load_prog_fn(exec_prog.buffer)
        executorch_method = executorch_program.load_method("forward")
        executorch_method(inputs)
        self.assertEqual(
            str(executorch_method.get_attribute("state")), str(torch.ones(2, 2))
        )

    def test_program_method_meta(self) -> None:
        eager_module = ModuleAddWithAttributes()
        inputs = eager_module.get_inputs()

        exported_program = export(eager_module, inputs, strict=True)
        exec_prog = to_edge(exported_program).to_executorch(
            config=ExecutorchBackendConfig(
                emit_mutable_buffer_names=True,
            )
        )

        exec_prog.dump_executorch_program(verbose=True)

        executorch_program = self.load_prog_fn(exec_prog.buffer)

        meta = executorch_program.method_meta("forward")

        del executorch_program
        self.assertEqual(meta.name(), "forward")
        self.assertEqual(meta.num_inputs(), 2)
        self.assertEqual(meta.num_outputs(), 1)
        self.assertEqual(meta.num_attributes(), 1)

        tensor_info = (
            "TensorInfo(sizes=[2, 2], dtype=Float, is_memory_planned=True, nbytes=16)"
        )

        float_dtype = 6
        self.assertEqual(
            str(meta),
            "MethodMeta(name='forward', num_inputs=2, "
            f"input_tensor_meta=['{tensor_info}', '{tensor_info}'], "
            f"num_outputs=1, output_tensor_meta=['{tensor_info}'])",
        )

        input_tensors = [meta.input_tensor_meta(i) for i in range(2)]
        output_tensor = meta.output_tensor_meta(0)
        attribute_tensor = meta.attribute_tensor_meta(0)

        with self.assertRaises(IndexError):
            meta.input_tensor_meta(2)

        with self.assertRaises(IndexError):
            meta.attribute_tensor_meta(1)

        del meta
        self.assertEqual([t.sizes() for t in input_tensors], [(2, 2), (2, 2)])
        self.assertEqual([t.dtype() for t in input_tensors], [float_dtype, float_dtype])
        self.assertEqual([t.is_memory_planned() for t in input_tensors], [True, True])
        self.assertEqual([t.nbytes() for t in input_tensors], [16, 16])
        self.assertEqual(str(input_tensors), f"[{tensor_info}, {tensor_info}]")

        self.assertEqual(output_tensor.sizes(), (2, 2))
        self.assertEqual(output_tensor.dtype(), float_dtype)
        self.assertEqual(output_tensor.is_memory_planned(), True)
        self.assertEqual(output_tensor.nbytes(), 16)
        self.assertEqual(str(output_tensor), tensor_info)

        self.assertEqual(attribute_tensor.sizes(), (2, 2))
        self.assertEqual(attribute_tensor.dtype(), float_dtype)
        self.assertEqual(attribute_tensor.is_memory_planned(), True)
        self.assertEqual(attribute_tensor.nbytes(), 16)
        self.assertEqual(str(attribute_tensor), tensor_info)

    def test_method_method_meta(self) -> None:
        exported_program, inputs = create_program(ModuleAdd())

        executorch_program = self.load_prog_fn(exported_program.buffer)
        executorch_method = executorch_program.load_method("forward")
        meta = executorch_method.method_meta()

        del executorch_program
        del executorch_method
        self.assertEqual(meta.name(), "forward")
        self.assertEqual(meta.num_inputs(), 2)
        self.assertEqual(meta.num_outputs(), 1)

        tensor_info = (
            "TensorInfo(sizes=[2, 2], dtype=Float, is_memory_planned=True, nbytes=16)"
        )
        float_dtype = 6
        self.assertEqual(
            str(meta),
            "MethodMeta(name='forward', num_inputs=2, "
            f"input_tensor_meta=['{tensor_info}', '{tensor_info}'], "
            f"num_outputs=1, output_tensor_meta=['{tensor_info}'])",
        )

        input_tensors = [meta.input_tensor_meta(i) for i in range(2)]
        output_tensor = meta.output_tensor_meta(0)

        with self.assertRaises(IndexError):
            meta.input_tensor_meta(2)

        del meta
        self.assertEqual([t.sizes() for t in input_tensors], [(2, 2), (2, 2)])
        self.assertEqual([t.dtype() for t in input_tensors], [float_dtype, float_dtype])
        self.assertEqual([t.is_memory_planned() for t in input_tensors], [True, True])
        self.assertEqual([t.nbytes() for t in input_tensors], [16, 16])
        self.assertEqual(str(input_tensors), f"[{tensor_info}, {tensor_info}]")

        self.assertEqual(output_tensor.sizes(), (2, 2))
        self.assertEqual(output_tensor.dtype(), float_dtype)
        self.assertEqual(output_tensor.is_memory_planned(), True)
        self.assertEqual(output_tensor.nbytes(), 16)
        self.assertEqual(str(output_tensor), tensor_info)

    def test_program_data_separation(self) -> None:
        eager_module = ModuleLinear()
        inputs = eager_module.get_inputs()
        exported_program = export(eager_module, inputs, strict=True)
        exec_program = to_edge(exported_program).to_executorch(
            config=ExecutorchBackendConfig(
                # Move all tensor data to '_default_external_constant' file.
                external_constants=True,
            )
        )
        program_buffer = exec_program.buffer
        assert len(exec_program._tensor_data) == 1
        data_buffer = bytes(exec_program._tensor_data.pop("_default_external_constant"))

        import os
        import tempfile

        with tempfile.TemporaryDirectory() as tmpdir:
            pte_file = os.path.join(tmpdir, "linear.pte")
            with open(pte_file, "wb") as f:
                f.write(program_buffer)
            ptd_file = os.path.join(tmpdir, "linear.ptd")
            with open(ptd_file, "wb") as ptd:
                ptd.write(data_buffer)
            expected = eager_module(inputs[0])
            # Test 1: File-based loading with external data file
            executorch_module_file = self.runtime._load_for_executorch(
                pte_file, ptd_file
            )
            executorch_output_file = executorch_module_file.forward(inputs)[0]
            self.assertTrue(torch.allclose(expected, executorch_output_file))

        # Test 2: Buffer-based loading with external data buffer
        executorch_module_buffer = self.load_fn(program_buffer, data_buffer)
        executorch_output_buffer = executorch_module_buffer.forward(inputs)[0]
        self.assertTrue(torch.allclose(expected, executorch_output_buffer))

        # Test 3: Buffer-based loading without external data file (should fail or work differently)
        # This should fail because the program expects external data
        executorch_module_no_data = self.load_fn(program_buffer)
        with self.assertRaises(RuntimeError):
            executorch_module_no_data.forward(inputs)

        # Test 4: Test with invalid data buffer (should fail)
        invalid_bytes = b"invalid bytes"
        executorch_module_invalid_data = self.load_fn(program_buffer, invalid_bytes)
        with self.assertRaises(RuntimeError):
            executorch_module_invalid_data.forward(inputs)

        # Test 5: Test bundled program loading with external data
        # First create a bundled program with external constants
        from executorch.devtools.bundled_program.config import (
            MethodTestCase,
            MethodTestSuite,
        )
        from executorch.devtools.bundled_program.core import BundledProgram
        from executorch.devtools.bundled_program.serialize import (
            serialize_from_bundled_program_to_flatbuffer,
        )

        method_test_suites = [
            MethodTestSuite(
                method_name="forward",
                test_cases=[
                    MethodTestCase(
                        inputs=input,
                        expected_outputs=expected,
                    )
                    for input in inputs
                ],
            ),
        ]
        bundled_program = BundledProgram(exec_program, method_test_suites)
        bundled_buffer = serialize_from_bundled_program_to_flatbuffer(bundled_program)
        bundled_module = self.runtime._load_bundled_program_from_buffer(bundled_buffer)

        # Load module from bundled program with external data buffer
        executorch_module_bundled = (
            self.runtime._load_for_executorch_from_bundled_program(
                bundled_module, data_buffer
            )
        )
        executorch_output_bundled = executorch_module_bundled.forward(inputs)[0]
        self.assertTrue(torch.allclose(expected, executorch_output_bundled))

        # Load module from bundled program with external data file
        with tempfile.TemporaryDirectory() as tmpdir:
            ptd_file = os.path.join(tmpdir, "linear.ptd")
            with open(ptd_file, "wb") as ptd:
                ptd.write(data_buffer)
            executorch_module_bundled_data_file = (
                self.runtime._load_for_executorch_from_bundled_program(
                    bundled_module, ptd_file
                )
            )
            executorch_output_bundled_data_file = (
                executorch_module_bundled_data_file.forward(inputs)[0]
            )
            self.assertTrue(
                torch.allclose(expected, executorch_output_bundled_data_file)
            )

        # Test 6: Bundled program without external data should fail
        executorch_module_bundled_no_data = (
            self.runtime._load_for_executorch_from_bundled_program(bundled_module)
        )
        with self.assertRaises(RuntimeError):
            executorch_module_bundled_no_data.forward(inputs)

    def test_flat_tensor_data_map(self) -> None:
        eager_module = ModuleLinear()
        inputs = eager_module.get_inputs()
        expected = eager_module(inputs[0])
        exported_program = export(eager_module, inputs, strict=True)
        exec_program = to_edge(exported_program).to_executorch(
            config=ExecutorchBackendConfig(
                # Move all tensor data to '_default_external_constant' file.
                external_constants=True,
            )
        )
        program_buffer = exec_program.buffer
        assert len(exec_program._tensor_data) == 1
        data_buffer = bytes(exec_program._tensor_data.pop("_default_external_constant"))

        # Test 1: Load FlatTensorDataMap from buffer.
        program_from_buffer = self.load_prog_fn(program_buffer)
        data_map_from_buffer = self.runtime._load_flat_tensor_data_map_from_buffer(
            data_buffer
        )
        method = program_from_buffer.load_method(
            "forward", data_map_from_buffer.get_named_data_map()
        )
        executorch_output = method(inputs)[0]
        self.assertTrue(torch.allclose(expected, executorch_output))

        # Test 2: Load FlatTensorDataMap from file.
        import os
        import tempfile

        with tempfile.TemporaryDirectory() as tmpdir:
            pte_file = os.path.join(tmpdir, "linear.pte")
            with open(pte_file, "wb") as f:
                f.write(program_buffer)
            ptd_file = os.path.join(tmpdir, "linear.ptd")
            with open(ptd_file, "wb") as ptd:
                ptd.write(data_buffer)

            program_from_file = self.runtime._load_program(pte_file)
            data_map_from_file = self.runtime._load_flat_tensor_data_map(ptd_file)
            method_1 = program_from_file.load_method(
                "forward", data_map_from_file.get_named_data_map()
            )
            executorch_output1 = method_1(inputs)[0]
            self.assertTrue(torch.allclose(expected, executorch_output1))
