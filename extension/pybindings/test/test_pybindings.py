# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-unsafe

import os
import sys
import tempfile
import unittest
from io import StringIO

import torch

from executorch.exir import ExecutorchBackendConfig, to_edge
from executorch.exir.backend.test.device_util import DeviceAwarePartitioner
from executorch.exir.passes import MemoryPlanningPass
from executorch.exir.schema import DeviceType
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
        # Only the portable build converts an incoming tensor, so a test for
        # that conversion has to skip under the aten build.
        self.kernel_mode = kernel_mode
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

    def test_method_rejects_input_on_unrepresentable_device(self):
        # The conversion used to label every input CPU, so an input whose memory
        # the host cannot read was described as host memory, and the failure
        # surfaced later as a crash instead of an error naming the input.
        if self.kernel_mode != "portable":
            self.skipTest("only the portable build converts the input tensor")

        exported_program, inputs = create_program(ModuleAdd())
        executorch_module = self.load_fn(exported_program.buffer)

        with self.assertRaises(RuntimeError) as caught:
            executorch_module.forward([inputs[0].to("meta"), inputs[1]])
        message = str(caught.exception)
        # Asserts the device is named as Python spells it, since an uppercased or
        # index-less name would not match what the caller passed.
        self.assertIn("is on device meta", message)
        self.assertIn("only CPU and CUDA tensors", message)

    def test_method_accepts_a_cpu_input_after_the_device_check(self):
        # The rejection tests above pass for a change that throws on every input, so this
        # asserts the other half: an ordinary CPU input still converts and runs. Without it
        # the pair does not distinguish "rejects what it cannot represent" from "rejects
        # everything".
        exported_program, inputs = create_program(ModuleAdd())
        executorch_module = self.load_fn(exported_program.buffer)

        executorch_output = executorch_module.forward(inputs)[0]

        self.assertTrue(
            torch.allclose(executorch_output, inputs[0] + inputs[1]),
            "a CPU input must still reach the method and produce the same result",
        )

    def test_program_method_rejects_input_on_unrepresentable_device(self):
        # The other conversion site, reached through a loaded program rather than a
        # module. It got the same treatment, and without this it had no test: before
        # the change this path aborted the process rather than raising.
        if self.kernel_mode != "portable":
            self.skipTest("only the portable build converts the input tensor")

        exported_program, inputs = create_program(ModuleAdd())
        program = self.load_prog_fn(exported_program.buffer)
        method = program.load_method("forward")

        with self.assertRaises(RuntimeError) as caught:
            method.set_inputs([inputs[0].to("meta"), inputs[1]])
        message = str(caught.exception)
        self.assertIn("is on device meta", message)
        self.assertIn("only CPU and CUDA tensors", message)

    def test_program_loads_when_one_method_is_device_planned(self):
        # Linking the CUDA backend registers a CUDA allocator at static init, and the
        # registry has no way to drop one, so there the device load would succeed with
        # or without the fix and this test could not tell them apart.
        if "CudaBackend" in self.runtime._get_registered_backend_names():
            self.skipTest("a registered CUDA allocator satisfies the device load")

        exported_program, inputs = create_program(
            ModuleMulti(),
            et_config=ExecutorchBackendConfig(enable_non_cpu_memory_planning=True),
            partitioner={"forward2": DeviceAwarePartitioner()},
        )

        # Without this the test would quietly become a second copy of
        # test_method_multiple_entry if the planner stopped tagging devices.
        planned_devices = {
            plan.name: [
                buffer.device_type for buffer in (plan.non_const_buffer_device or [])
            ]
            for plan in exported_program.executorch_program.execution_plan
        }
        self.assertIn(DeviceType.CUDA, planned_devices["forward2"])
        self.assertNotIn(DeviceType.CUDA, planned_devices["forward"])

        program = self.load_prog_fn(exported_program.buffer)
        self.assertEqual(program.num_methods(), 2)

        method = program.load_method("forward")
        self.assertTrue(torch.allclose(method.call(inputs)[0], torch.ones(2, 2) * 2))

        # Asking for device memory at all is what this asserts. Before the fix the
        # loader put every planned buffer in host memory, so the device-planned method
        # never asked and simply loaded. Now it asks, and with no device allocator
        # registered the request is refused.
        with self.assertRaises(RuntimeError) as caught:
            program.load_method("forward2")
        self.assertIn("on device", str(caught.exception))

        # A failed load must leave the method that already loaded usable.
        self.assertTrue(torch.allclose(method.call(inputs)[0], torch.ones(2, 2) * 2))

    def test_device_planned_method_allocates_on_the_device(self):
        # The other device test covers the refusal. This one covers what the
        # refusal is protecting: on a build that does have a device allocator,
        # the arena has to come off the device, not out of host memory. It needs
        # a real accelerator, so it only runs where one is present.
        if "CudaBackend" not in self.runtime._get_registered_backend_names():
            self.skipTest("needs a build with the CUDA backend linked in")
        if not torch.cuda.is_available():
            self.skipTest("needs a visible CUDA device")

        from executorch.backends.cuda.cuda_partitioner import CudaPartitioner
        from executorch.exir import to_edge_transform_and_lower
        from executorch.exir.backend.compile_spec_schema import CompileSpec

        # Large enough that the arena is far bigger than the noise in
        # mem_get_info, which moves by a few MiB as contexts are created.
        side = 2048
        inputs = (torch.ones(side, side), torch.ones(side, side))

        class HostOnly(torch.nn.Module):
            def forward(self, x, y):
                return x + y

        class Delegated(torch.nn.Module):
            def forward(self, x, y):
                return (x + y) * 2.0

        edge = to_edge_transform_and_lower(
            {
                "forward": export(HostOnly(), inputs, strict=True),
                "forward2": export(Delegated(), inputs, strict=True),
            },
            partitioner={
                "forward2": [CudaPartitioner([CompileSpec("method_name", b"forward2")])]
            },
        )
        exported_program = edge.to_executorch(
            config=ExecutorchBackendConfig(enable_non_cpu_memory_planning=True)
        )

        plans = {
            plan.name: plan
            for plan in exported_program.executorch_program.execution_plan
        }
        device_buffers = {
            buffer.buffer_idx
            for buffer in (plans["forward2"].non_const_buffer_device or [])
        }
        self.assertTrue(device_buffers, "the planner tagged nothing for the device")
        self.assertFalse(plans["forward"].non_const_buffer_device or [])
        device_bytes = sum(
            size
            for index, size in enumerate(plans["forward2"].non_const_buffer_sizes)
            if index in device_buffers
        )

        # The CUDA backend keeps its weights in a separate file, so the program
        # has to be loaded from disk with that file alongside it.
        with tempfile.TemporaryDirectory() as directory:
            pte_path = os.path.join(directory, "program.pte")
            with open(pte_path, "wb") as pte_file:
                exported_program.write_to_file(pte_file)
            data_names = sorted(exported_program._tensor_data or {})
            exported_program.write_tensor_data_to_file(directory)
            data_path = (
                os.path.join(directory, data_names[0] + ".ptd") if data_names else None
            )

            torch.cuda.init()
            program = self.runtime._load_program(pte_path, data_path=data_path)

            torch.cuda.synchronize()
            free_before, _ = torch.cuda.mem_get_info()

            # The host-only method must not touch the device at all.
            host_method = program.load_method("forward")
            torch.cuda.synchronize()
            free_after_host, _ = torch.cuda.mem_get_info()
            self.assertLess(free_before - free_after_host, device_bytes // 2)

            device_method = program.load_method("forward2")
            torch.cuda.synchronize()
            free_after_device, _ = torch.cuda.mem_get_info()
            self.assertGreaterEqual(
                free_after_host - free_after_device, device_bytes * 0.9
            )

            # Before the fix this ran on a host pointer and the backend rejected
            # it, so getting the right answer back is itself part of the check.
            expected = (inputs[0] + inputs[1]) * 2.0
            self.assertTrue(
                torch.allclose(device_method.call(inputs)[0].cpu(), expected)
            )
            self.assertTrue(
                torch.allclose(host_method.call(inputs)[0].cpu(), inputs[0] + inputs[1])
            )
            # The device arenas are private, so running the host method in
            # between must not disturb them.
            self.assertTrue(
                torch.allclose(device_method.call(inputs)[0].cpu(), expected)
            )

            del device_method
            del host_method
            del program
            torch.cuda.synchronize()
            free_at_end, _ = torch.cuda.mem_get_info()
            self.assertGreaterEqual(free_at_end - free_after_device, device_bytes * 0.9)
