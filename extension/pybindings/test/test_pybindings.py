# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-unsafe

import gc
import os
import subprocess
import sys
import tempfile
import unittest
import weakref
from io import StringIO

import numpy as np
import torch

from executorch.exir import ExecutorchBackendConfig, to_edge
from executorch.exir.backend.test.device_util import DeviceAwarePartitioner
from executorch.exir.passes import MemoryPlanningPass
from executorch.exir.schema import DeviceType
from executorch.extension.pybindings.test.make_test import (
    create_program,
    ModuleAdd,
    ModuleAddConstReturn,
    ModuleAddEmpty,
    ModuleAddHalf,
    ModuleAddScalar,
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

    def test_method_load_error_preserves_runtime_code(self):
        from executorch.exir._serialize._program import PTEFile, serialize_pte_binary
        from executorch.exir.schema import (
            BackendDelegate,
            BackendDelegateDataReference,
            BackendDelegateInlineData,
            DataLocation,
        )

        exported_program, inputs = create_program(ModuleAdd())
        schema = exported_program.executorch_program
        schema.execution_plan[0].delegates.append(
            BackendDelegate(
                id="MissingBackendForErrorTest",
                processed=BackendDelegateDataReference(
                    location=DataLocation.INLINE,
                    index=len(schema.backend_delegate_data),
                ),
                compile_specs=[],
            )
        )
        schema.backend_delegate_data.append(BackendDelegateInlineData(data=b""))
        data = bytes(serialize_pte_binary(PTEFile(schema)))
        for protocol, values in (
            ("torch", inputs),
            ("numpy", [value.numpy() for value in inputs]),
        ):
            with self.subTest(protocol=protocol):
                module = self.load_fn(data)
                with self.assertRaisesRegex(RuntimeError, r"\b0x:?20\b"):
                    module.run_method("forward", values)

    def test_numpy_buffer_inputs(self):
        exported_program, inputs = create_program(ModuleAdd())
        executorch_module = self.load_fn(exported_program.buffer)

        output = executorch_module.forward([value.numpy() for value in inputs])[0]

        self.assertTrue(torch.allclose(output, inputs[0] + inputs[1]))

    def test_numpy_float16_buffer_input(self):
        exported_program, inputs = create_program(ModuleAddHalf())
        executorch_module = self.load_fn(exported_program.buffer)

        output = executorch_module(inputs[0].numpy())[0]

        self.assertTrue(torch.equal(output, inputs[0] + inputs[0]))

    def test_buffer_input_does_not_import_torch(self):
        exported_program, inputs = create_program(ModuleAddSingleInput())
        executorch_module = self.load_fn(exported_program.buffer)
        torch_module = sys.modules.pop("torch")
        try:
            output = executorch_module(inputs[0].numpy())[0]
            self.assertNotIn("torch", sys.modules)
        finally:
            sys.modules["torch"] = torch_module

        self.assertTrue(torch.equal(output, inputs[0] + inputs[0]))

    def test_empty_numpy_buffer_input(self):
        exported_program, inputs = create_program(ModuleAddEmpty())
        array = np.zeros((3, 0), dtype=np.float32)

        module = self.load_fn(exported_program.buffer)
        self.assertTrue(torch.equal(module(array)[0], inputs[0] + inputs[0]))

        method = self.load_prog_fn(exported_program.buffer).load_method("forward")
        self.assertTrue(torch.equal(method(array)[0], inputs[0] + inputs[0]))

    def test_empty_torch_output_preserves_layout_and_device(self):
        if self.kernel_mode != "portable":
            self.skipTest("only portable bindings convert outputs through Python")

        class Identity(torch.nn.Module):
            def forward(self, x):
                return x

        input_tensor = torch.empty(0, 2, 3, 4).to(memory_format=torch.channels_last)
        program = to_edge(export(Identity(), (input_tensor,))).to_executorch(
            ExecutorchBackendConfig(
                memory_planning_pass=MemoryPlanningPass(alloc_graph_input=False)
            )
        )
        module = self.load_fn(program.buffer)
        method = self.load_prog_fn(program.buffer).load_method("forward")
        for executor in (module, method):
            for clone_outputs in (True, False):
                with self.subTest(executor=type(executor), clone=clone_outputs):
                    with torch.device("meta"):
                        output = executor(input_tensor, clone_outputs=clone_outputs)[0]
                    self.assertEqual(output.device, input_tensor.device)
                    self.assertEqual(output.stride(), input_tensor.stride())
                    self.assertEqual(
                        executor(output)[0].stride(), input_tensor.stride()
                    )

    def test_torch_inputs_are_safe_across_threads(self):
        if self.kernel_mode != "portable":
            self.skipTest("only portable bindings use the Python torch adapter")

        script = """
import threading

import torch
from executorch.exir import to_edge
from executorch.extension.pybindings import portable_lib


class Add(torch.nn.Module):
    def forward(self, x, y):
        return x + y


inputs = (torch.ones(2, 2), torch.ones(2, 2))
program = to_edge(torch.export.export(Add(), inputs)).to_executorch()
data = program.buffer
loaded = portable_lib._load_program_from_buffer(data)


def run(method):
    for _ in range(300):
        method.set_inputs([torch.ones(2, 2), torch.ones(2, 2)])
        method.execute()
        method.get_outputs()


threads = [
    threading.Thread(target=run, args=(loaded.load_method("forward"),))
    for _ in range(4)
]
for thread in threads:
    thread.start()
for thread in threads:
    thread.join()
print("all calls finished")
"""
        completed = subprocess.run(
            [sys.executable, "-c", script],
            capture_output=True,
            text=True,
            timeout=60,
        )

        self.assertEqual(
            completed.returncode,
            0,
            f"stdout:\n{completed.stdout}\nstderr:\n{completed.stderr}",
        )
        self.assertIn("all calls finished", completed.stdout)

    def test_bfloat16_output_preserves_dtype(self):
        class ModuleAddBFloat16(ModuleAdd):
            def get_inputs(self):
                return (
                    torch.ones(2, dtype=torch.bfloat16),
                    torch.ones(2, dtype=torch.bfloat16),
                )

        exported_program, inputs = create_program(ModuleAddBFloat16())
        output = self.load_fn(exported_program.buffer)(inputs)[0]

        self.assertEqual(output.dtype, torch.bfloat16)
        torch.testing.assert_close(output, inputs[0] + inputs[1])

    def test_default_numpy_integer_dtype_error_is_descriptive(self):
        exported_program, _ = create_program(ModuleAddSingleInput())
        executorch_module = self.load_fn(exported_program.buffer)

        with self.assertRaisesRegex(ValueError, "dtype Long.*expects Float"):
            executorch_module(np.ones((2, 2), dtype=int))

    def test_numpy_float_scalar_input(self):
        exported_program, inputs = create_program(ModuleAddScalar())
        executorch_module = self.load_fn(exported_program.buffer)

        output = executorch_module.forward([inputs[0], np.float64(inputs[1])])[0]

        self.assertTrue(torch.equal(output, inputs[0] + inputs[1]))

    def test_numpy_float_scalar_set_inputs(self):
        exported_program, inputs = create_program(ModuleAddScalar())
        program = self.load_prog_fn(exported_program.buffer)
        method = program.load_method("forward")

        method.set_inputs([inputs[0], np.float64(inputs[1])])
        method.execute()

        self.assertTrue(torch.equal(method.get_outputs()[0], inputs[0] + inputs[1]))

    def test_method_rejects_tensor_for_scalar_input(self):
        exported_program, inputs = create_program(ModuleAddScalar())
        method = self.load_prog_fn(exported_program.buffer).load_method("forward")

        with self.assertRaises(RuntimeError):
            method.set_inputs([inputs[0], torch.ones(1)])

    def test_numpy_array_is_a_single_input(self):
        exported_program, inputs = create_program(ModuleAddSingleInput())
        executorch_module = self.load_fn(exported_program.buffer)

        output = executorch_module(inputs[0].numpy())[0]

        self.assertTrue(torch.allclose(output, inputs[0] + inputs[0]))

    def test_mixed_tensor_protocols_are_rejected(self):
        exported_program, inputs = create_program(ModuleAdd())
        executorch_module = self.load_fn(exported_program.buffer)

        with self.assertRaisesRegex(TypeError, "cannot mix buffer and torch"):
            executorch_module.forward([inputs[0], inputs[1].numpy()])

    def test_non_dense_numpy_input_is_rejected(self):
        exported_program, inputs = create_program(ModuleAddSingleInput())
        executorch_module = self.load_fn(exported_program.buffer)
        non_dense = np.zeros((4, 4), dtype=np.float32)[::2, ::2]

        with self.assertRaisesRegex(BufferError, "dense, non-overlapping"):
            executorch_module(non_dense)

    def test_aten_accepts_nonstandard_torch_strides(self):
        if self.kernel_mode != "aten":
            self.skipTest("ATen-only input behavior")
        exported_program, inputs = create_program(
            ModuleAddSingleInput(),
            et_config=ExecutorchBackendConfig(
                memory_planning_pass=MemoryPlanningPass(alloc_graph_input=False)
            ),
        )
        input_tensor = torch.arange(8, dtype=inputs[0].dtype).reshape(2, 4)[:, ::2]

        module = self.load_fn(exported_program.buffer)
        self.assertTrue(torch.equal(module(input_tensor)[0], input_tensor * 2))

        method = self.load_prog_fn(exported_program.buffer).load_method("forward")
        method.set_inputs(input_tensor)
        method.execute()
        self.assertTrue(torch.equal(method.get_outputs()[0], input_tensor * 2))

    def test_complex_numpy_input_is_rejected(self):
        exported_program, _ = create_program(ModuleAddSingleInput())
        executorch_module = self.load_fn(exported_program.buffer)

        with self.assertRaisesRegex(BufferError, "Unsupported buffer format"):
            executorch_module(np.ones((2, 2), dtype=np.complex64))

    def test_readonly_buffer_input_is_copied(self):
        exported_program, inputs = create_program(
            ModuleAddSingleInput(),
            et_config=ExecutorchBackendConfig(
                memory_planning_pass=MemoryPlanningPass(alloc_graph_input=False)
            ),
        )
        program = self.load_prog_fn(exported_program.buffer)
        method = program.load_method("forward")
        array = inputs[0].numpy()
        method.set_inputs(memoryview(array).toreadonly())
        array.fill(3)

        method.execute()

        self.assertTrue(torch.equal(method.get_outputs()[0], torch.full((2, 2), 2.0)))

    def test_numpy_layout_must_match_exported_layout(self):
        model = ModuleChannelsLast()
        exported_program, inputs = create_program(model)
        executorch_module = self.load_fn(exported_program.buffer)
        contiguous = inputs[0].contiguous().numpy()

        with self.assertRaisesRegex(ValueError, "method expects stride"):
            executorch_module(contiguous)

    def test_tensor_subclass_without_storage_is_rejected(self):
        class Wrapper(torch.Tensor):
            @staticmethod
            def __new__(cls, elem):
                return torch.Tensor._make_wrapper_subclass(
                    cls, elem.shape, dtype=elem.dtype
                )

            @classmethod
            def __torch_dispatch__(cls, func, types, args=(), kwargs=None):
                raise NotImplementedError(func)

        exported_program, inputs = create_program(ModuleAddSingleInput())
        executorch_module = self.load_fn(exported_program.buffer)

        with self.assertRaisesRegex(ValueError, "data is not allocated"):
            executorch_module(Wrapper(inputs[0]))

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

    def test_clone_outputs_false_returns_a_view(self):
        eager_module = ModuleAddWithAttributes()
        inputs = eager_module.get_inputs()
        exported_program = export(eager_module, inputs, strict=True)
        exec_prog = to_edge(exported_program).to_executorch(
            config=ExecutorchBackendConfig(emit_mutable_buffer_names=True)
        )
        executorch_module = self.load_fn(exec_prog.buffer)

        output = executorch_module(inputs, clone_outputs=False)[0]
        snapshot = output.clone()
        executorch_module(inputs)

        self.assertFalse(torch.equal(output, snapshot))

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
                self.assertIn("must be less than the number of inputs", str(out))

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

    def test_torch_input_layout_matches_exported_dim_order(self):
        class AddInput(ModuleAddSingleInput):
            def __init__(self, input_tensor):
                super().__init__()
                self.input_tensor = input_tensor

            def get_inputs(self):
                return (self.input_tensor,)

        cases = (
            ((2, 3, 4, 5), torch.channels_last),
            ((2, 1, 3, 4), torch.channels_last),
            ((2, 3, 1, 1), torch.channels_last),
            ((2, 3, 4, 5, 6), torch.channels_last_3d),
        )
        for shape, memory_format in cases:
            input_tensor = torch.arange(torch.Size(shape).numel(), dtype=torch.float32)
            input_tensor = input_tensor.reshape(shape).to(memory_format=memory_format)
            # Singleton dimensions can make channels last storage compatible
            # with a model exported using the default dimension order.
            export_input = (
                input_tensor.clone(memory_format=torch.contiguous_format)
                if input_tensor.is_contiguous()
                else input_tensor
            )
            for planned in (True, False):
                with self.subTest(shape=shape, planned=planned):
                    program, _ = create_program(
                        AddInput(export_input),
                        et_config=ExecutorchBackendConfig(
                            memory_planning_pass=MemoryPlanningPass(
                                alloc_graph_input=planned
                            )
                        ),
                    )
                    expected = input_tensor * 2
                    module = self.load_fn(program.buffer)
                    torch.testing.assert_close(module(input_tensor)[0], expected)

                    method = self.load_prog_fn(program.buffer).load_method("forward")
                    method.set_inputs(input_tensor)
                    method.execute()
                    torch.testing.assert_close(method.get_outputs()[0], expected)

                    if shape == (2, 1, 3, 4):
                        strides = list(input_tensor.stride())
                        strides[1] = 77
                        equivalent = input_tensor.as_strided(shape, strides)
                        torch.testing.assert_close(module(equivalent)[0], expected)
                        method.set_inputs(equivalent)
                        method.execute()
                        torch.testing.assert_close(method.get_outputs()[0], expected)

    def test_torch_input_layout_mismatch_is_rejected(self):
        if self.kernel_mode != "portable":
            self.skipTest("portable bindings validate strides against exported layout")
        program, inputs = create_program(ModuleChannelsLast())
        input_tensor = inputs[0].contiguous()
        module = self.load_fn(program.buffer)
        with self.assertRaisesRegex(ValueError, "method expects stride"):
            module(input_tensor)
        method = self.load_prog_fn(program.buffer).load_method("forward")
        with self.assertRaisesRegex(ValueError, "method expects stride"):
            method.set_inputs(input_tensor)

    def test_unsupported_dim_order(self) -> None:
        model = ModuleChannelsLast()
        exported_program, inputs = create_program(model)
        inputs = (torch.randn(1, 2, 3, 4, 5).to(memory_format=torch.channels_last_3d),)

        executorch_module = self.load_fn(exported_program.buffer)
        self.assertRaisesRegex(
            ValueError, "rank 5.*expects rank 4", executorch_module, inputs[0]
        )

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

    def test_program_keeps_buffer_alive(self):
        exported_program, inputs = create_program(ModuleAdd())
        program_data = bytes(exported_program.buffer)
        initial_refcount = sys.getrefcount(program_data)

        program = self.load_prog_fn(program_data)

        self.assertGreater(sys.getrefcount(program_data), initial_refcount)
        del program_data
        output = program.load_method("forward")(inputs)[0]
        self.assertTrue(torch.equal(output, inputs[0] + inputs[1]))

    def test_method_preserves_unplanned_input_that_is_also_an_output(self):
        class ReturnInputAndSum(torch.nn.Module):
            def forward(self, x, y):
                return x, x + y

            def get_methods_to_export(self):
                return ("forward",)

            def get_inputs(self):
                return (torch.ones(2, 2), torch.full((2, 2), 2.0))

        exported_program, inputs = create_program(
            ReturnInputAndSum(),
            et_config=ExecutorchBackendConfig(
                memory_planning_pass=MemoryPlanningPass(alloc_graph_input=False)
            ),
        )
        method = self.load_prog_fn(exported_program.buffer).load_method("forward")

        outputs = method(inputs)

        self.assertTrue(torch.equal(outputs[0], inputs[0]))
        self.assertTrue(torch.equal(outputs[1], inputs[0] + inputs[1]))

    def test_borrowed_output_does_not_create_method_cycle(self):
        if self.kernel_mode != "portable":
            self.skipTest("only portable bindings retain Python tensor owners")
        for planned_input in (True, False):
            for planned_output in (True, False):
                with self.subTest(input=planned_input, output=planned_output):
                    program, inputs = create_program(
                        ModuleAddSingleInput(),
                        et_config=ExecutorchBackendConfig(
                            memory_planning_pass=MemoryPlanningPass(
                                alloc_graph_input=planned_input,
                                alloc_graph_output=planned_output,
                            )
                        ),
                    )
                    method = self.load_prog_fn(program.buffer).load_method("forward")
                    output = method(inputs, clone_outputs=False)[0]
                    method.set_inputs(output)
                    expected = inputs[0] * 2
                    if planned_output:
                        previous_ref = weakref.ref(output)
                        next_output = method(output, clone_outputs=False)[0]
                        method.set_inputs(next_output)
                        del output
                        gc.collect()
                        self.assertIsNone(previous_ref())
                        output = next_output
                        expected *= 2
                    method_ref = weakref.ref(method)
                    del method
                    gc.collect()
                    self.assertIsNone(method_ref())
                    torch.testing.assert_close(output, expected)

    def test_borrowed_output_preserves_replaced_input_storage(self):
        if self.kernel_mode != "portable":
            self.skipTest("only portable bindings retain Python tensor owners")

        class Identity(torch.nn.Module):
            def forward(self, x):
                return x

        program = to_edge(export(Identity(), (torch.ones(2, 2),))).to_executorch(
            ExecutorchBackendConfig(
                memory_planning_pass=MemoryPlanningPass(alloc_graph_input=False)
            )
        )
        for buffer_input in (True, False):
            with self.subTest(buffer_input=buffer_input):
                value = (
                    np.full((2, 2), 3.0, dtype=np.float32)
                    if buffer_input
                    else torch.full((2, 2), 3.0)
                )
                value_ref = weakref.ref(value)
                method = self.load_prog_fn(program.buffer).load_method("forward")
                output = method(value, clone_outputs=False)[0]
                method.set_inputs(np.zeros((2, 2), dtype=np.float32))
                method_ref = weakref.ref(method)
                del value, method
                gc.collect()
                self.assertIsNone(method_ref())
                if buffer_input:
                    self.assertIsNotNone(value_ref())
                torch.testing.assert_close(output, torch.full((2, 2), 3.0))
                del output
                gc.collect()
                self.assertIsNone(value_ref())

    def test_borrowed_external_output_does_not_retain_previous_outputs(self):
        if self.kernel_mode != "portable":
            self.skipTest("only portable bindings retain Python tensor owners")

        class Identity(torch.nn.Module):
            def forward(self, x):
                return x

        for channels_last in (False, True):
            with self.subTest(channels_last=channels_last):
                shape = (2, 3, 4, 5) if channels_last else (2, 2)
                value = torch.arange(torch.Size(shape).numel() + 4, dtype=torch.float32)
                value = value[4:].reshape(shape)
                if channels_last:
                    value = value.to(memory_format=torch.channels_last)
                value.requires_grad_()
                expected = value.detach().clone()
                program = to_edge(export(Identity(), (value,))).to_executorch(
                    ExecutorchBackendConfig(
                        memory_planning_pass=MemoryPlanningPass(
                            alloc_graph_input=False, alloc_graph_output=False
                        )
                    )
                )
                method = self.load_prog_fn(program.buffer).load_method("forward")
                output = method(value, clone_outputs=False)[0]
                address = value.data_ptr()
                strides = value.stride()
                offset = value.storage_offset()
                for _ in range(32):
                    previous = weakref.ref(output)
                    output = method(output, clone_outputs=False)[0]
                    method.set_inputs(output)
                    gc.collect()
                    self.assertIsNone(previous())
                    self.assertEqual(output.data_ptr(), address)
                    self.assertEqual(output.stride(), strides)
                    self.assertEqual(output.storage_offset(), offset)
                    self.assertFalse(output.requires_grad)
                del value, method
                torch.testing.assert_close(output, expected)

    def test_borrowed_result_reuses_backing_owner_through_buffer_views(self):
        if self.kernel_mode != "portable":
            self.skipTest("only portable bindings retain Python buffer owners")

        class Identity(torch.nn.Module):
            def forward(self, x):
                return x

        program = to_edge(export(Identity(), (torch.ones(2, 2),))).to_executorch(
            ExecutorchBackendConfig(
                memory_planning_pass=MemoryPlanningPass(
                    alloc_graph_input=False, alloc_graph_output=False
                )
            )
        )
        script = """
import gc
import sys
import weakref
import numpy as np

from executorch.extension.pybindings import portable_lib as runtime

with open(sys.argv[1], "rb") as program_file:
    data = program_file.read()
def readonly_array(value):
    array = np.asarray(value)
    array.setflags(write=False)
    return array
for copied_root in (False, True):
    for wrap in (lambda value: value, np.asarray, memoryview,
                 readonly_array, lambda value: memoryview(value).toreadonly()):
        method = runtime._load_program_from_buffer(data).load_method("forward")
        output = method(np.full((2, 2), 3.0, dtype=np.float32),
                        clone_outputs=copied_root)[0]
        output = method(wrap(output), clone_outputs=False)[0]
        for _ in range(32):
            previous = weakref.ref(output)
            output = method(wrap(output), clone_outputs=False)[0]
            method.set_inputs(wrap(output))
            gc.collect()
            assert previous() is None
        del method
        gc.collect()
        np.testing.assert_array_equal(np.asarray(output), np.full((2, 2), 3.0))
"""
        generic_buffer_script = """
import array
import gc
import importlib.abc
import sys
import weakref

class BlockTensorPackages(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path, target=None):
        if fullname.split(".")[0] in ("numpy", "torch"):
            raise ModuleNotFoundError(fullname, name=fullname)
sys.meta_path.insert(0, BlockTensorPackages())

from executorch.extension.pybindings import portable_lib as runtime

with open(sys.argv[1], "rb") as program_file:
    data = program_file.read()
method = runtime._load_program_from_buffer(data).load_method("forward")
value = memoryview(array.array("f", [3.0] * 4)).cast("B").cast("f", (2, 2))
output = method(value, clone_outputs=False)[0]
for _ in range(32):
    previous = weakref.ref(output)
    output = method(output, clone_outputs=False)[0]
    method.set_inputs(output)
    gc.collect()
    assert previous() is None
del method, value
assert memoryview(output).tolist() == [[3.0, 3.0], [3.0, 3.0]]
assert "numpy" not in sys.modules and "torch" not in sys.modules
"""
        with tempfile.TemporaryDirectory() as directory:
            path = os.path.join(directory, "identity.pte")
            with open(path, "wb") as program_file:
                program_file.write(program.buffer)
            for code in (script, generic_buffer_script):
                completed = subprocess.run(
                    [
                        sys.executable,
                        "-c",
                        code,
                        path,
                    ],
                    env={
                        **os.environ,
                        "EXECUTORCH_PYBINDINGS_TENSOR_OUTPUT": "executorch",
                    },
                    capture_output=True,
                    text=True,
                    timeout=60,
                )
                self.assertEqual(completed.returncode, 0, completed.stderr)

    def test_failed_set_inputs_keeps_buffer_storage_alive(self):
        exported_program, inputs = create_program(
            ModuleAdd(),
            et_config=ExecutorchBackendConfig(
                memory_planning_pass=MemoryPlanningPass(alloc_graph_input=False)
            ),
        )
        program = self.load_prog_fn(exported_program.buffer)
        method = program.load_method("forward")
        method.set_inputs([value.numpy() for value in inputs])

        with self.assertRaises(ValueError):
            method.set_inputs([inputs[0].numpy(), inputs[1].to(torch.int32).numpy()])

        method.execute()
        output = method.get_outputs()[0]
        self.assertTrue(torch.allclose(output, inputs[0] + inputs[1]))

    def test_method_rejects_resized_borrowed_input(self):
        exported_program, inputs = create_program(
            ModuleAdd(),
            et_config=ExecutorchBackendConfig(
                memory_planning_pass=MemoryPlanningPass(alloc_graph_input=False)
            ),
        )
        protocol_inputs = {
            "torch": [value.clone() for value in inputs],
            "numpy": [value.numpy().copy() for value in inputs],
        }

        for protocol, values in protocol_inputs.items():
            with self.subTest(protocol=protocol):
                method = self.load_prog_fn(exported_program.buffer).load_method(
                    "forward"
                )
                method.set_inputs(values)
                if protocol == "torch":
                    values[0].resize_(64, 64)
                else:
                    values[0].resize((64, 64), refcheck=False)

                with self.assertRaisesRegex(
                    ValueError, "Input 0 storage changed.*call set_inputs"
                ):
                    method.execute()

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
                self.assertIn("Invalid number of inputs provided", str(out))

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
        self.assertRaisesRegex(
            ValueError, "rank 5.*expects rank 4", executorch_method, inputs[0]
        )

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

    def test_method_attribute_is_a_view(self):
        eager_module = ModuleAddWithAttributes()
        inputs = eager_module.get_inputs()
        exported_program = export(eager_module, inputs, strict=True)
        exec_prog = to_edge(exported_program).to_executorch(
            config=ExecutorchBackendConfig(emit_mutable_buffer_names=True)
        )
        executorch_method = self.load_prog_fn(exec_prog.buffer).load_method("forward")

        state = executorch_method.get_attribute("state")
        executorch_method(inputs)

        self.assertTrue(torch.equal(state, torch.ones(2, 2)))

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

    def test_rejects_noncanonical_empty_strides(self):
        exported_program, inputs = create_program(ModuleAdd())
        executorch_module = self.load_fn(exported_program.buffer)
        empty_view = torch.empty(5, 7)[0:0, 0:2]

        with self.assertRaisesRegex(ValueError, "stride"):
            executorch_module.forward([empty_view, inputs[1]])

    def test_empty_input_does_not_require_device_memory(self):
        if self.kernel_mode != "portable":
            self.skipTest("only the portable build converts the input tensor")

        exported_program, inputs = create_program(ModuleAdd())
        executorch_module = self.load_fn(exported_program.buffer)

        with self.assertRaises(RuntimeError) as caught:
            executorch_module.forward([torch.empty(0, 2, device="meta"), inputs[1]])

        self.assertNotIn("is on device meta", str(caught.exception))

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

    def test_unplanned_device_output_is_device_memory(self):
        # A program exported with alloc_graph_output=False leaves its outputs to the caller, and a
        # device-resident one tags them for the device, so the bindings supply that memory. They
        # used to supply host memory, which the CUDA backend rejects (InvalidArgument) and which a
        # delegate without that check writes into, failing later wherever the GPU cannot read
        # pageable host memory. This checks that the output buffer is real device memory.
        if "CudaBackend" not in self.runtime._get_registered_backend_names():
            self.skipTest("needs a build with the CUDA backend linked in")
        if not torch.cuda.is_available():
            self.skipTest("needs a visible CUDA device")

        import ctypes

        from executorch.backends.cuda.cuda_partitioner import CudaPartitioner
        from executorch.exir import to_edge_transform_and_lower
        from executorch.exir.backend.compile_spec_schema import CompileSpec
        from executorch.exir.passes.memory_planning_pass import MemoryPlanningPass
        from executorch.exir.passes.propagate_device_pass import PropagateDeviceConfig
        from executorch.runtime import Runtime

        class Delegated(torch.nn.Module):
            def forward(self, x, y):
                return (x + y) * 2.0

        inputs = (torch.ones(4, 4), torch.ones(4, 4))
        edge = to_edge_transform_and_lower(
            {"forward": export(Delegated(), inputs, strict=True)},
            partitioner={
                "forward": [CudaPartitioner([CompileSpec("method_name", b"forward")])]
            },
        )
        exported_program = edge.to_executorch(
            config=ExecutorchBackendConfig(
                propagate_device_config=PropagateDeviceConfig(
                    skip_h2d_for_method_inputs=True, skip_d2h_for_method_outputs=True
                ),
                enable_non_cpu_memory_planning=True,
                memory_planning_pass=MemoryPlanningPass(
                    alloc_graph_input=False, alloc_graph_output=False
                ),
            )
        )
        # Without this the test would quietly stop covering the unplanned path if the planner
        # started placing these outputs: a planned device arena is device memory too, so the
        # pointer check below would still pass.
        plan = exported_program.executorch_program.execution_plan[0]
        for index in plan.outputs:
            output_tensor = plan.values[index].val
            self.assertIsNone(output_tensor.allocation_info)
            self.assertEqual(
                output_tensor.extra_tensor_info.device_type, DeviceType.CUDA
            )
        with tempfile.TemporaryDirectory() as directory:
            pte_path = os.path.join(directory, "program.pte")
            with open(pte_path, "wb") as pte_file:
                exported_program.write_to_file(pte_file)
            data_names = sorted(exported_program._tensor_data or {})
            exported_program.write_tensor_data_to_file(directory)
            data_path = (
                os.path.join(directory, data_names[0] + ".ptd") if data_names else None
            )
            method = (
                Runtime.get()
                .load_program(pte_path, data_path=data_path)
                .load_method("forward")
            )
            cuda_inputs = [tensor.cuda() for tensor in inputs]
            output = method._method(cuda_inputs, clone_outputs=False)[0]
            # Ask the driver what the uncloned output points at, not the tensor's device tag,
            # which says cuda either way. CU_POINTER_ATTRIBUTE_MEMORY_TYPE is 2 and
            # CU_MEMORYTYPE_DEVICE is 2. The driver does not know a pageable host pointer at
            # all, and returns an error for it.
            memory_type = ctypes.c_uint(0)
            status = ctypes.CDLL("libcuda.so.1").cuPointerGetAttribute(
                ctypes.byref(memory_type), 2, ctypes.c_uint64(output.data_ptr())
            )
            self.assertEqual((status, memory_type.value), (0, 2))
            torch.testing.assert_close(
                method.execute(cuda_inputs)[0].cpu(), torch.full((4, 4), 4.0)
            )

    def test_cuda_torch_tensor_interchange_without_aten(self):
        if self.kernel_mode != "portable":
            self.skipTest("only portable bindings use the Python Torch adapter")
        if not torch.cuda.is_available():
            self.skipTest("needs a visible CUDA device")

        from executorch.exir._serialize._program import PTEFile, serialize_pte_binary
        from executorch.exir.schema import ExtraTensorInfo, Tensor

        class Identity(torch.nn.Module):
            def forward(self, x):
                return x

        device_indices = sorted({0, torch.cuda.device_count() - 1})
        for device_index in device_indices:
            for dtype in (torch.float32, torch.float16, torch.bfloat16):
                for empty in (True, False):
                    shape = (0, 2, 3, 4) if empty else (2, 3, 4, 5)
                    input_tensor = torch.arange(
                        torch.Size(shape).numel() + 5,
                        device=f"cuda:{device_index}",
                        dtype=dtype,
                    )[5:].reshape(shape)
                    for memory_format in (torch.contiguous_format, torch.channels_last):
                        with self.subTest(
                            device=device_index,
                            dtype=dtype,
                            empty=empty,
                            memory_format=memory_format,
                        ):
                            value = (
                                input_tensor
                                if memory_format == torch.contiguous_format
                                else input_tensor.to(memory_format=memory_format)
                            )
                            program = to_edge(
                                export(Identity(), (value.cpu(),))
                            ).to_executorch(
                                ExecutorchBackendConfig(
                                    memory_planning_pass=MemoryPlanningPass(
                                        alloc_graph_input=False,
                                        alloc_graph_output=False,
                                    )
                                )
                            )
                            # This identity has no kernels or device arenas, so
                            # interchange does not require a linked CUDA backend.
                            schema = program.executorch_program
                            for entry in schema.execution_plan[0].values:
                                if isinstance(entry.val, Tensor):
                                    entry.val.extra_tensor_info = ExtraTensorInfo(
                                        device_type=DeviceType.CUDA,
                                        device_index=device_index,
                                    )
                            data = bytes(serialize_pte_binary(PTEFile(schema)))
                            module = self.load_fn(data)
                            method = self.load_prog_fn(data).load_method("forward")
                            for executor in (module, method):
                                borrowed = executor(value, clone_outputs=False)[0]
                                self.assertEqual(borrowed.device, value.device)
                                self.assertEqual(borrowed.dtype, value.dtype)
                                self.assertEqual(borrowed.stride(), value.stride())
                                self.assertEqual(borrowed.data_ptr(), value.data_ptr())
                                cloned = executor(value)[0]
                                self.assertEqual(cloned.device, value.device)
                                self.assertEqual(cloned.stride(), value.stride())
                                torch.testing.assert_close(cloned.cpu(), value.cpu())
                                if not empty:
                                    self.assertNotEqual(
                                        cloned.data_ptr(), value.data_ptr()
                                    )

    def test_torch_input_device_must_match_method(self):
        if self.kernel_mode != "portable":
            self.skipTest("only portable bindings inspect Torch device metadata")

        from executorch.exir._serialize._program import PTEFile, serialize_pte_binary
        from executorch.exir.schema import ExtraTensorInfo, Tensor

        class Identity(torch.nn.Module):
            def forward(self, x):
                return x

        value = torch.ones(2, 2)
        program = to_edge(export(Identity(), (value,))).to_executorch(
            ExecutorchBackendConfig(
                memory_planning_pass=MemoryPlanningPass(alloc_graph_input=False)
            )
        )
        schema = program.executorch_program
        for entry in schema.execution_plan[0].values:
            if isinstance(entry.val, Tensor):
                entry.val.extra_tensor_info = ExtraTensorInfo(
                    device_type=DeviceType.CUDA, device_index=3
                )
        data = bytes(serialize_pte_binary(PTEFile(schema)))
        module = self.load_fn(data)
        method = self.load_prog_fn(data).load_method("forward")
        for executor in (module, method):
            with self.assertRaisesRegex(ValueError, "cpu.*expects cuda:3"):
                executor(value)

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
