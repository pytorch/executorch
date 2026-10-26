# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-unsafe

import ctypes
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
    ModuleAddBFloat16,
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


class DLPackOnly:
    def __init__(self, array) -> None:
        self.array = array

    def __dlpack__(self, stream=None):
        return self.array.__dlpack__(stream=stream)

    def __dlpack_device__(self):
        return self.array.__dlpack_device__()


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
            ("dlpack", [DLPackOnly(value.numpy()) for value in inputs]),
        ):
            with self.subTest(protocol=protocol):
                module = self.load_fn(data)
                with self.assertRaisesRegex(
                    RuntimeError, "Failed to execute method forward, error: 0x20"
                ):
                    module.run_method("forward", values)
        module = self.load_fn(data)
        with self.assertRaisesRegex(
            RuntimeError, "Failed to execute method forward, error: 0x20"
        ):
            module.run_method("forward")

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

    def test_tensor_output_import_policy(self):
        if self.kernel_mode != "portable":
            self.skipTest("only portable bindings configure Python tensor outputs")
        program, _ = create_program(ModuleAddSingleInput())
        script = """
import importlib
import importlib.abc
import os
import sys
from pathlib import Path

class BlockTorch(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname == "torch" or fullname.startswith("torch."):
            raise ImportError("torch is unavailable")

entrypoint, native_name, policy, available, program_path, native_path = sys.argv[1:]
if sys.platform == "win32":
    native_directory = Path(native_path).parent
    os.add_dll_directory(str(native_directory))
    library_directory = native_directory.parent.parent / "lib"
    if library_directory.is_dir():
        os.add_dll_directory(str(library_directory))
blocker = BlockTorch()
if available == "no":
    sys.meta_path.insert(0, blocker)
try:
    bindings = importlib.import_module(entrypoint)
except (ImportError, SystemError) as error:
    # Built-in extension loaders may wrap initialization errors.
    assert policy == "torch" and available == "no"
    while error.__cause__ is not None:
        error = error.__cause__
    assert isinstance(error, ImportError)
    assert "torch is unavailable" in str(error)
else:
    assert policy != "torch" or available == "yes"
    native = sys.modules[native_name]
    expected = "torch" if policy != "executorch" and available == "yes" else "executorch"
    assert native._tensor_output == expected
    assert ("torch" in sys.modules) == (expected == "torch")
    import numpy as np
    program_data = Path(program_path).read_bytes()
    module = bindings._load_for_executorch_from_buffer(program_data)
    output = module(np.ones((2, 2), dtype=np.float32))[0]
    if expected == "executorch":
        assert isinstance(output, bindings.ExecuTorchResult)
        np.testing.assert_array_equal(np.asarray(output), np.full((2, 2), 2.0))
        if available == "no":
            sys.meta_path.remove(blocker)
            import torch
            assert isinstance(module(torch.ones(2, 2))[0], bindings.ExecuTorchResult)
    else:
        import torch
        assert isinstance(output, torch.Tensor)
        torch.testing.assert_close(output, torch.full((2, 2), 2.0))
"""
        native_name = self.load_fn.__module__
        native_path = sys.modules[native_name].__spec__.origin
        with tempfile.TemporaryDirectory() as directory:
            program_path = os.path.join(directory, "add.pte")
            with open(program_path, "wb") as handle:
                handle.write(program.buffer)
            for entrypoint in dict.fromkeys((self.runtime.__name__, native_name)):
                for policy in (None, "torch", "executorch"):
                    for available in ("yes", "no"):
                        with self.subTest(
                            entrypoint=entrypoint, policy=policy, available=available
                        ):
                            environment = os.environ.copy()
                            environment.pop("EXECUTORCH_PYBINDINGS_TENSOR_OUTPUT", None)
                            if policy is not None:
                                environment["EXECUTORCH_PYBINDINGS_TENSOR_OUTPUT"] = (
                                    policy
                                )
                            completed = subprocess.run(
                                [
                                    sys.executable,
                                    "-c",
                                    script,
                                    entrypoint,
                                    native_name,
                                    str(policy),
                                    available,
                                    program_path,
                                    native_path,
                                ],
                                env=environment,
                                capture_output=True,
                                text=True,
                                timeout=60,
                            )
                            self.assertEqual(
                                completed.returncode,
                                0,
                                completed.stdout + completed.stderr,
                            )

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
        input_tensor = input_tensor.as_strided(input_tensor.shape, (1000, 1, 100, 2))
        module = self.load_fn(program.buffer)
        method = self.load_prog_fn(program.buffer).load_method("forward")
        for executor in (module, method):
            for clone_outputs in (True, False):
                with self.subTest(executor=type(executor), clone=clone_outputs):
                    with torch.device("meta"):
                        output = executor(input_tensor, clone_outputs=clone_outputs)[0]
                    self.assertEqual(output.device, input_tensor.device)
                    self.assertEqual(output.stride(), (24, 1, 8, 2))
                    self.assertEqual(executor(output)[0].stride(), (24, 1, 8, 2))

    def test_torch_inputs_are_safe_across_threads(self):
        if self.kernel_mode != "portable":
            self.skipTest("only portable bindings use the Python torch adapter")

        script = """
import threading
from concurrent.futures import ThreadPoolExecutor

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


locks = {}


def run(executor, separate_calls, locked=True):
    for _ in range(300):
        inputs = [torch.ones(2, 2), torch.ones(2, 2)]
        if separate_calls and locked:
            # Separate calls on a shared method need the caller's lock, or
            # another thread's set_inputs() lands between them.
            with locks.setdefault(id(executor), threading.Lock()):
                executor.set_inputs(inputs)
                executor.execute()
                outputs = executor.get_outputs()
        elif separate_calls:
            # Without the lock, another thread's set_inputs() may hide the
            # outputs, which must raise rather than crash or return garbage.
            try:
                executor.set_inputs(inputs)
                executor.execute()
                outputs = executor.get_outputs()
            except RuntimeError as error:
                assert "no outputs" in str(error), error
                continue
        else:
            outputs = executor(inputs)
        output = outputs[0]
        if not isinstance(output, torch.Tensor):
            output = torch.from_numpy(output.__array__())
        torch.testing.assert_close(output, inputs[0] + inputs[1])


shared_method = loaded.load_method("forward")
shared_module = portable_lib._load_for_executorch_from_buffer(data)
for executors, separate_calls, locked in (
    ([loaded.load_method("forward") for _ in range(8)], True, True),
    ([shared_method] * 8, True, True),
    ([shared_method] * 8, True, False),
    ([shared_method] * 8, False, True),
    ([shared_module] * 8, False, True),
):
    with ThreadPoolExecutor(max_workers=8) as pool:
        futures = [
            pool.submit(run, executor, separate_calls, locked)
            for executor in executors
        ]
        for future in futures:
            future.result()
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

    def test_refused_input_errors_say_why(self):
        exported_program, inputs = create_program(ModuleAddScalar())
        executorch_module = self.load_fn(exported_program.buffer)
        with self.assertRaises(RuntimeError) as error:
            executorch_module.forward([None, inputs[1]])
        self.assertIn("type Tensor but was None", str(error.exception))
        # Only the reason is reported, not lines about other inputs.
        self.assertNotIn("is not Tensor", str(error.exception))

        # The portable runtime logs a static shape mismatch over two lines.
        if self.kernel_mode == "portable":
            exported_program, inputs = create_program(ModuleAdd())
            with self.assertRaisesRegex(
                RuntimeError, r"received \(3, 2\)\.; Error resizing tensor at input 0"
            ):
                self.load_fn(exported_program.buffer).forward(
                    [torch.ones(3, 2), inputs[1]]
                )

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
        module = self.load_fn(exported_program.buffer)
        method = self.load_prog_fn(exported_program.buffer).load_method("forward")

        for run in (module.forward, method.set_inputs):
            with self.subTest(run=run):
                with self.assertRaisesRegex(
                    RuntimeError,
                    r"^Input 1 of method 'forward': the method expects Double, "
                    r"but got <class 'torch\.Tensor'>$",
                ):
                    run([inputs[0], torch.ones(1)])

    def test_method_rejects_aliased_input_whose_storage_changed(self):
        # An output that is the input itself is returned as a view of the
        # caller's tensor, which must still be the storage the method read.
        if self.kernel_mode != "portable":
            self.skipTest("only portable bindings return a view of the input")
        from executorch.extension.pybindings import _C

        if _C._tensor_output != "torch":
            self.skipTest("only torch output mode returns a view of the input")

        class Identity(torch.nn.Module):
            def forward(self, x):
                return x

        value = torch.ones(2, 2)
        program = to_edge(export(Identity(), (value,))).to_executorch(
            ExecutorchBackendConfig(
                memory_planning_pass=MemoryPlanningPass(
                    alloc_graph_input=False, alloc_graph_output=False
                )
            )
        )
        method = self.load_prog_fn(program.buffer).load_method("forward")
        method.set_inputs([value])
        method.execute()
        value.set_(torch.zeros(2, 2))
        with self.assertRaisesRegex(
            ValueError,
            r"^Input 0 of method 'forward': its storage changed before the output "
            r"was read; pass it again$",
        ):
            method.get_outputs(clone_outputs=False)

    def test_dlpack_inputs(self):
        exported_program, inputs = create_program(ModuleAdd())
        module = self.load_fn(exported_program.buffer)
        method = self.load_prog_fn(exported_program.buffer).load_method("forward")
        for executor in (module, method):
            with self.subTest(executor=type(executor)):
                output = executor([DLPackOnly(value.numpy()) for value in inputs])[0]
                torch.testing.assert_close(output, inputs[0] + inputs[1])

    def test_dlpack_input_in_another_layout_is_converted(self):
        if self.kernel_mode != "portable":
            self.skipTest("portable input layout compatibility")
        exported_program, _ = create_program(ModuleAddSingleInput())
        module = self.load_fn(exported_program.buffer)
        method = self.load_prog_fn(exported_program.buffer).load_method("forward")
        # Distinct values, so reading the transposed data in the exported
        # layout would permute them.
        values = np.arange(4, dtype=np.float32).reshape(2, 2).T
        for executor in (module, method):
            with self.subTest(executor=type(executor)):
                output = executor(DLPackOnly(values))[0]
                torch.testing.assert_close(output, torch.from_numpy(values * 2))

    def test_borrowed_result_feedback_has_bounded_ownership(self):
        if self.kernel_mode != "portable":
            self.skipTest("only portable bindings implement ExecuTorchResult outputs")

        class Identity(torch.nn.Module):
            def forward(self, value):
                return value

        class ToChannelsLast(torch.nn.Module):
            def forward(self, value):
                return value.to(memory_format=torch.channels_last)

        config = ExecutorchBackendConfig(
            memory_planning_pass=MemoryPlanningPass(
                alloc_graph_input=False, alloc_graph_output=False
            )
        )
        program = to_edge(export(Identity(), (torch.ones(2, 2),))).to_executorch(config)
        # A channels-last result, read by a contiguous identity method, is
        # copied into the contiguous layout first.
        channels_last_program = to_edge(
            export(ToChannelsLast(), (torch.ones(1, 2, 2, 2),))
        ).to_executorch(config)
        contiguous_program = to_edge(
            export(Identity(), (torch.ones(1, 2, 2, 2),))
        ).to_executorch(config)
        script = """
import gc
import importlib
import sys
import weakref
import numpy as np

runtime = importlib.import_module(sys.argv[4])

native = sys.modules[sys.argv[3]]
assert native.__spec__.origin == sys.argv[2]

class DLPackOnly:
    def __init__(self, value):
        self.value = value
    def __dlpack__(self, stream=None):
        return self.value.__dlpack__(stream=stream)
    def __dlpack_device__(self):
        return self.value.__dlpack_device__()

assert not runtime._uses_aten
assert native._tensor_output == "executorch"
with open(sys.argv[1], "rb") as source:
    data = source.read()
with open(sys.argv[5], "rb") as source:
    channels_last_data = source.read()
with open(sys.argv[6], "rb") as source:
    contiguous_data = source.read()

def live_lists():
    gc.collect()
    return sum(isinstance(value, list) for value in gc.get_objects())

def check_feedback(executor):
    source = np.arange(4, dtype=np.float32).reshape(2, 2)
    source_ref = weakref.ref(source)
    executor_ref = weakref.ref(executor)
    pointer = source.ctypes.data
    result = executor(source, clone_outputs=False)[0]
    def advance(result, count):
        for index in range(count):
            producer = result if index % 2 == 0 else DLPackOnly(result)
            result = executor(producer, clone_outputs=False)[0]
            del producer
            view = np.from_dlpack(result)
            assert view.ctypes.data == pointer
            np.testing.assert_array_equal(view, source)
            del view
        return result
    result = advance(result, 16)
    before = live_lists()
    result = advance(result, 128)
    after = live_lists()
    # Repeated feedback must release old ownership containers, even while
    # the current result still shares the original allocation.
    assert after < before + 32, (type(executor), before, after)
    del executor, source
    gc.collect()
    np.testing.assert_array_equal(np.from_dlpack(result), np.arange(4).reshape(2, 2))
    del result, advance
    gc.collect()
    assert executor_ref() is None
    assert source_ref() is None

check_feedback(runtime._load_for_executorch_from_buffer(data))
check_feedback(runtime._load_program_from_buffer(data).load_method("forward"))

# A result in another layout is copied into the exported one, and a borrowed
# result over it reads that copy. It must keep the copy alive, not the
# producer's result, whose memory holds the other layout.
method = runtime._load_program_from_buffer(contiguous_data).load_method("forward")
converter = runtime._load_program_from_buffer(channels_last_data).load_method(
    "forward"
)
values = np.arange(8, dtype=np.float32).reshape(1, 2, 2, 2)
source = converter(values)[0]
assert np.from_dlpack(source).strides != values.strides
method.set_inputs([DLPackOnly(source)])
method.execute()
borrowed = method.get_outputs(clone_outputs=False)[0]
method.set_inputs([np.full((1, 2, 2, 2), -7.0, dtype=np.float32)])
gc.collect()
np.testing.assert_array_equal(np.from_dlpack(borrowed), values)
print("Borrowed result feedback retained bounded ownership")
"""
        with tempfile.TemporaryDirectory() as directory:
            path = os.path.join(directory, "identity.pte")
            with open(path, "wb") as source:
                source.write(program.buffer)
            channels_last_path = os.path.join(directory, "channels_last.pte")
            with open(channels_last_path, "wb") as source:
                source.write(channels_last_program.buffer)
            contiguous_path = os.path.join(directory, "contiguous.pte")
            with open(contiguous_path, "wb") as source:
                source.write(contiguous_program.buffer)
            environment = os.environ.copy()
            environment["EXECUTORCH_PYBINDINGS_TENSOR_OUTPUT"] = "executorch"
            native_name = self.runtime.ExecuTorchResult.__module__
            completed = subprocess.run(
                [
                    sys.executable,
                    "-c",
                    script,
                    path,
                    sys.modules[native_name].__spec__.origin,
                    native_name,
                    self.runtime.__name__,
                    channels_last_path,
                    contiguous_path,
                ],
                env=environment,
                capture_output=True,
                text=True,
                timeout=60,
            )
        self.assertEqual(
            completed.returncode,
            0,
            f"stdout:\n{completed.stdout}\nstderr:\n{completed.stderr}",
        )
        self.assertIn(
            "Borrowed result feedback retained bounded ownership", completed.stdout
        )

    def test_module_retains_dlpack_inputs_for_reexecution(self):
        if self.kernel_mode != "portable":
            self.skipTest("only portable bindings implement generic DLPack inputs")
        program, _ = create_program(
            ModuleAddSingleInput(),
            et_config=ExecutorchBackendConfig(
                memory_planning_pass=MemoryPlanningPass(alloc_graph_input=False)
            ),
        )
        module = self.load_fn(program.buffer)
        source = np.ones((2, 2), dtype=np.float32)
        producer = DLPackOnly(source)
        producer_ref = weakref.ref(producer)
        source_ref = weakref.ref(source)
        output = module(producer, clone_outputs=False)[0]
        del producer, source
        gc.collect()
        self.assertIsNotNone(producer_ref())
        self.assertIsNotNone(source_ref())

        source_ref().fill(3.0)
        repeated = module()[0]
        planned = module.plan_execute("forward", clone_outputs=False)[0]
        torch.testing.assert_close(repeated, torch.full((2, 2), 6.0))
        torch.testing.assert_close(planned, torch.full((2, 2), 6.0))
        del output, repeated, planned

        replacement = DLPackOnly(np.full((2, 2), 5.0, dtype=np.float32))
        replacement_ref = weakref.ref(replacement)
        module(replacement)
        del replacement
        gc.collect()
        self.assertIsNone(producer_ref())
        self.assertIsNone(source_ref())
        self.assertIsNotNone(replacement_ref())
        output = module.plan_execute("forward", clone_outputs=False)[0]
        del module
        gc.collect()
        self.assertIsNone(replacement_ref())
        torch.testing.assert_close(output, torch.full((2, 2), 10.0))
        del output
        gc.collect()
        self.assertIsNone(replacement_ref())

    def test_cuda_dlpack_producer_and_executorch_result(self):
        if self.kernel_mode != "portable":
            self.skipTest("only portable bindings implement ExecuTorchResult outputs")
        if not torch.cuda.is_available():
            self.skipTest("needs a visible CUDA device")

        from executorch.exir._serialize._program import PTEFile, serialize_pte_binary
        from executorch.exir.schema import ExtraTensorInfo, Tensor

        class Identity(torch.nn.Module):
            def forward(self, x):
                return x

        value = torch.ones(2, 3, 4, 5).to(memory_format=torch.channels_last)
        program = to_edge(export(Identity(), (value,))).to_executorch(
            ExecutorchBackendConfig(
                memory_planning_pass=MemoryPlanningPass(
                    alloc_graph_input=False, alloc_graph_output=False
                )
            )
        )
        schema = program.executorch_program
        device_index = torch.cuda.device_count() - 1
        for entry in schema.execution_plan[0].values:
            if isinstance(entry.val, Tensor):
                entry.val.extra_tensor_info = ExtraTensorInfo(
                    device_type=DeviceType.CUDA, device_index=device_index
                )
        data = bytes(serialize_pte_binary(PTEFile(schema)))

        # No delegate or device arena is needed for this identity. Both APIs
        # must consume the generic producer rather than the Torch input path.
        value = value.to(f"cuda:{device_index}")
        module = self.load_fn(data)
        method = self.load_prog_fn(data).load_method("forward")
        for executor in (module, method):
            for clone_outputs in (False, True):
                with self.subTest(executor=type(executor), clone=clone_outputs):
                    output = executor(DLPackOnly(value), clone_outputs=clone_outputs)[0]
                    self.assertEqual(output.device, value.device)
                    self.assertEqual(output.dtype, value.dtype)
                    self.assertEqual(output.stride(), value.stride())
                    torch.testing.assert_close(output, value)
                    if clone_outputs:
                        self.assertNotEqual(output.data_ptr(), value.data_ptr())
                    else:
                        self.assertEqual(output.data_ptr(), value.data_ptr())

        script = """
import sys
import torch
from executorch.extension.pybindings import portable_lib as runtime

class DLPackOnly:
    def __init__(self, tensor):
        self.tensor = tensor
    def __dlpack__(self, stream=None):
        return self.tensor.__dlpack__(stream=stream)
    def __dlpack_device__(self):
        return self.tensor.__dlpack_device__()

assert not runtime._uses_aten
with open(sys.argv[1], "rb") as source:
    data = source.read()
device_index = int(sys.argv[2])
value = torch.ones(2, 3, 4, 5, device=f"cuda:{device_index}").to(
    memory_format=torch.channels_last
)
module = runtime._load_for_executorch_from_buffer(data)
method = runtime._load_program_from_buffer(data).load_method("forward")
for executor in (module, method):
    output = executor(DLPackOnly(value), clone_outputs=False)[0]
    assert isinstance(output, runtime.ExecuTorchResult)
    assert output.__dlpack_device__() == (2, device_index)
    tensor = torch.from_dlpack(output)
    assert tensor.data_ptr() == value.data_ptr()
    assert tensor.stride() == value.stride()
    torch.testing.assert_close(tensor, value)
    roundtrip = executor(output, clone_outputs=False)[0]
    assert isinstance(roundtrip, runtime.ExecuTorchResult)
    assert roundtrip.__dlpack_device__() == (2, device_index)
    torch.testing.assert_close(torch.from_dlpack(roundtrip), value)
print("CUDA DLPack roundtrip preserved device storage")
"""
        with tempfile.TemporaryDirectory() as directory:
            path = os.path.join(directory, "cuda_identity.pte")
            with open(path, "wb") as source:
                source.write(data)
            environment = os.environ.copy()
            environment["EXECUTORCH_PYBINDINGS_TENSOR_OUTPUT"] = "executorch"
            completed = subprocess.run(
                [sys.executable, "-c", script, path, str(device_index)],
                env=environment,
                capture_output=True,
                text=True,
                timeout=60,
            )
        self.assertEqual(
            completed.returncode,
            0,
            f"stdout:\n{completed.stdout}\nstderr:\n{completed.stderr}",
        )
        self.assertIn(
            "CUDA DLPack roundtrip preserved device storage", completed.stdout
        )

    def test_bfloat16_raw_buffer_inputs(self):
        exported_program, inputs = create_program(ModuleAddBFloat16())
        executorch_module = self.load_fn(exported_program.buffer)
        method = self.load_prog_fn(exported_program.buffer).load_method("forward")
        raw_inputs = [value.view(torch.uint8).numpy().tobytes() for value in inputs]

        for executor in (executorch_module, method):
            with self.subTest(executor=type(executor)):
                output = executor(raw_inputs)[0]
                self.assertTrue(torch.equal(output, inputs[0] + inputs[1]))
                for noncontiguous in (
                    np.arange(16, dtype=np.uint8)[::2],
                    np.arange(8, dtype=np.uint8)[::-1],
                ):
                    with self.assertRaisesRegex(BufferError, "contiguous byte buffers"):
                        executor([noncontiguous, raw_inputs[1]])
                with self.assertRaisesRegex(
                    ValueError,
                    "^Input 0 of method 'forward': raw bfloat16 input has "
                    f"{len(raw_inputs[0]) - 2} bytes, but the method expects "
                    f"{len(raw_inputs[0])}$",
                ):
                    executor([raw_inputs[0][:-2], raw_inputs[1]])

    def test_numpy_array_is_a_single_input(self):
        exported_program, inputs = create_program(ModuleAddSingleInput())
        executorch_module = self.load_fn(exported_program.buffer)

        output = executorch_module(inputs[0].numpy())[0]

        self.assertTrue(torch.allclose(output, inputs[0] + inputs[0]))

    def test_mixed_tensor_protocols(self):
        exported_program, inputs = create_program(ModuleAdd())
        executorch_module = self.load_fn(exported_program.buffer)
        method = self.load_prog_fn(exported_program.buffer).load_method("forward")

        for executor in (executorch_module, method):
            for mixed_inputs in (
                [inputs[0], inputs[1].numpy()],
                [inputs[0], DLPackOnly(inputs[1].numpy())],
                [inputs[0].numpy(), DLPackOnly(inputs[1].numpy())],
            ):
                with self.subTest(
                    executor=type(executor),
                    protocols=[type(value) for value in mixed_inputs],
                ):
                    output = executor(mixed_inputs)[0]
                    self.assertTrue(torch.allclose(output, inputs[0] + inputs[1]))

    def test_raw_dlpack_capsule_is_a_single_input(self):
        exported_program, inputs = create_program(ModuleAddSingleInput())
        module = self.load_fn(exported_program.buffer)
        method = self.load_prog_fn(exported_program.buffer).load_method("forward")
        for executor in (module, method):
            with self.subTest(executor=type(executor)):
                output = executor(inputs[0].numpy().__dlpack__())[0]
                torch.testing.assert_close(
                    torch.from_dlpack(output), inputs[0] + inputs[0]
                )

    def test_non_dense_numpy_input_is_rejected(self):
        exported_program, inputs = create_program(ModuleAddSingleInput())
        executorch_module = self.load_fn(exported_program.buffer)
        non_dense = np.zeros((4, 4), dtype=np.float32)[::2, ::2]

        with self.assertRaisesRegex(BufferError, "dense, non-overlapping"):
            executorch_module(non_dense)

    def test_complex_numpy_input(self):
        class Identity(torch.nn.Module):
            def forward(self, x):
                return x.clone()

        exported_program = to_edge(
            export(Identity(), (torch.ones(4, dtype=torch.complex64),), strict=True)
        ).to_executorch()
        executorch_module = self.load_fn(exported_program.buffer)
        values = np.array([1 + 2j, 3 - 4j, 0, 1j], dtype=np.complex64)

        output = executorch_module(values)[0]

        np.testing.assert_array_equal(np.asarray(output), values)

    def test_complex_half_dlpack_input(self):
        if self.kernel_mode != "portable":
            self.skipTest("only portable bindings read DLPack inputs themselves")

        class Identity(torch.nn.Module):
            def forward(self, x):
                return x

        value = torch.tensor([1 + 2j, 3 - 4j], dtype=torch.complex32)
        program = to_edge(export(Identity(), (value,))).to_executorch(
            ExecutorchBackendConfig(
                memory_planning_pass=MemoryPlanningPass(
                    alloc_graph_input=False, alloc_graph_output=False
                )
            )
        )
        method = self.load_prog_fn(program.buffer).load_method("forward")
        output = torch.from_dlpack(method(DLPackOnly(value))[0])
        self.assertEqual(output.dtype, torch.complex32)
        self.assertTrue(
            torch.equal(output.view(torch.float16), value.view(torch.float16))
        )

    def test_unsupported_buffer_format_is_rejected(self):
        exported_program, _ = create_program(ModuleAddSingleInput())
        executorch_module = self.load_fn(exported_program.buffer)

        for value in (
            np.zeros((2, 2), dtype=[("a", np.float32)]),
            memoryview(b"\x00" * 4).cast("c"),
        ):
            with self.subTest(format=memoryview(value).format):
                with self.assertRaisesRegex(
                    BufferError,
                    "Input 0 of method 'forward': Unsupported buffer format",
                ):
                    executorch_module(value)

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

    def test_numpy_input_in_another_layout_is_converted(self):
        model = ModuleChannelsLast()
        exported_program, inputs = create_program(model)
        executorch_module = self.load_fn(exported_program.buffer)
        values = torch.arange(inputs[0].numel(), dtype=torch.float32).reshape(
            inputs[0].shape
        )

        output = executorch_module(values.numpy())[0]
        torch.testing.assert_close(output, model(values))

    def test_transposed_inputs_of_every_item_size_are_converted(self):
        if self.kernel_mode != "portable":
            self.skipTest("portable input layout compatibility")

        class Double(torch.nn.Module):
            def forward(self, x):
                return x + x

        for dtype in (np.int8, np.int16, np.float32, np.int64, np.float64):
            with self.subTest(dtype=np.dtype(dtype).name):
                example = torch.from_numpy(np.ones((2, 3), dtype=dtype))
                exported_program = to_edge(
                    export(Double(), (example,), strict=True)
                ).to_executorch()
                method = self.load_prog_fn(exported_program.buffer).load_method(
                    "forward"
                )
                values = np.arange(6, dtype=dtype).reshape(3, 2).T
                output = method([values])[0]
                np.testing.assert_array_equal(np.asarray(output), values + values)

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
                executorch_module.run_method("missing", inputs)
                self.assertFalse(True)  # should be unreachable
            except Exception:
                # The runtime's own log, routed to Python's stderr.
                self.assertIn("No method named 'missing' in program", str(out))

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
        if self.kernel_mode != "portable":
            self.skipTest("ATen nonstandard input layouts are deferred to a follow-up")

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

    def test_torch_input_in_another_layout_is_converted(self):
        if self.kernel_mode != "portable":
            self.skipTest("portable input layout compatibility")
        program, inputs = create_program(ModuleChannelsLast())
        input_tensor = torch.arange(inputs[0].numel(), dtype=torch.float32).reshape(
            inputs[0].shape
        )
        expected = ModuleChannelsLast()(input_tensor)
        module = self.load_fn(program.buffer)
        torch.testing.assert_close(module(input_tensor)[0], expected)
        method = self.load_prog_fn(program.buffer).load_method("forward")
        method.set_inputs(input_tensor)
        method.execute()
        torch.testing.assert_close(method.get_outputs()[0], expected)

    def test_channels_last_input_into_a_contiguous_method_is_converted(self):
        if self.kernel_mode != "portable":
            self.skipTest("portable input layout compatibility")
        values = torch.arange(2 * 3 * 4 * 4, dtype=torch.float32).reshape(2, 3, 4, 4)
        method_program = to_edge(
            export(ModuleAddSingleInput(), (values,), strict=True)
        ).to_executorch()
        executorch_module = self.load_fn(method_program.buffer)

        channels_last = values.to(memory_format=torch.channels_last)
        for value in (channels_last, channels_last.numpy(), DLPackOnly(channels_last)):
            with self.subTest(kind=type(value).__name__):
                output = executorch_module(value)[0]
                torch.testing.assert_close(
                    torch.as_tensor(np.asarray(output)), values * 2
                )

    def test_converted_input_is_a_copy_made_when_passed(self):
        if self.kernel_mode != "portable":
            self.skipTest("portable input layout compatibility")
        exported_program, _ = create_program(
            ModuleAddSingleInput(),
            et_config=ExecutorchBackendConfig(
                memory_planning_pass=MemoryPlanningPass(alloc_graph_input=False)
            ),
        )
        method = self.load_prog_fn(exported_program.buffer).load_method("forward")
        transposed = torch.arange(4, dtype=torch.float32).reshape(2, 2).t()
        expected = transposed * 2
        method.set_inputs([transposed])
        # The method reads the converted copy, so the caller can reuse or
        # resize its own tensor without invalidating the inputs.
        transposed.resize_(64, 64).fill_(9.0)
        method.execute()
        torch.testing.assert_close(method.get_outputs()[0], expected)

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

        tensor_info = "TensorInfo(sizes=[2, 2], dtype=Float, is_memory_planned=True, nbytes=16, device=cpu)"
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
                    ValueError,
                    "Input 0 of method 'forward': its storage changed.*pass it again",
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
                program, _ = create_program(ModuleAdd())
                executorch_program = self.load_prog_fn(program.buffer)
                executorch_program.load_method("missing")
                self.assertFalse(True)  # should be unreachable
            except Exception:
                self.assertIn("No method named 'missing' in program", str(out))

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

        tensor_info = "TensorInfo(sizes=[2, 2], dtype=Float, is_memory_planned=True, nbytes=16, device=cpu)"

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

        tensor_info = "TensorInfo(sizes=[2, 2], dtype=Float, is_memory_planned=True, nbytes=16, device=cpu)"
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

    def test_constant_attribute_is_never_writable(self):
        # A constant from the data file can sit in read-only pages, so a write
        # through a borrowed view would crash the process.
        if self.kernel_mode != "portable":
            self.skipTest("only portable bindings return borrowed attributes")
        eager_module = ModuleLinear()
        program = to_edge(
            export(eager_module, eager_module.get_inputs(), strict=True)
        ).to_executorch(ExecutorchBackendConfig(external_constants=True))
        with tempfile.TemporaryDirectory() as directory:
            pte_path = os.path.join(directory, "linear.pte")
            ptd_path = os.path.join(directory, "linear.ptd")
            with open(pte_path, "wb") as file:
                file.write(program.buffer)
            with open(ptd_path, "wb") as file:
                file.write(bytes(program._tensor_data["_default_external_constant"]))
            method = self.runtime._load_program(
                pte_path, data_path=ptd_path
            ).load_method("forward")
            weight = method.get_attribute("linear.weight")
            expected = eager_module.linear.weight.detach()
            if isinstance(weight, torch.Tensor):
                # A torch tensor cannot be read-only, so it is a copy.
                self.assertNotEqual(
                    weight.data_ptr(), method.get_attribute("linear.weight").data_ptr()
                )
                torch.testing.assert_close(weight, expected)
            else:
                self.assertTrue(memoryview(weight).readonly)
                self.assertFalse(np.asarray(weight).flags.writeable)
                np.testing.assert_array_equal(np.asarray(weight), expected.numpy())
            # The method maps the data file, and Windows cannot delete a mapped
            # file, so drop it before the directory goes away.
            del weight, method
            gc.collect()
            # torch ignores the DLPack read-only flag, so a DLPack export must be
            # a copy that it can write. Checked in ExecuTorchResult output mode.
            script = """
import sys

import numpy as np
import torch
from executorch.extension.pybindings import portable_lib as runtime

method = runtime._load_program(sys.argv[1], data_path=sys.argv[2]).load_method(
    "forward"
)
weight = method.get_attribute("linear.weight")
assert isinstance(weight, runtime.ExecuTorchResult)
before = np.asarray(weight).copy()
torch.from_dlpack(weight).add_(1)
assert (np.asarray(weight) == before).all()
print("the constant was not written")
"""
            completed = subprocess.run(
                [sys.executable, "-c", script, pte_path, ptd_path],
                env=dict(os.environ, EXECUTORCH_PYBINDINGS_TENSOR_OUTPUT="executorch"),
                capture_output=True,
                text=True,
                timeout=120,
            )
        self.assertEqual(
            completed.returncode,
            0,
            f"stdout:\n{completed.stdout}\nstderr:\n{completed.stderr}",
        )
        self.assertIn("the constant was not written", completed.stdout)

    def test_method_keeps_its_event_tracer_alive(self):
        # The program is dropped right away, while its method keeps running.
        # glibc's MALLOC_PERTURB_ overwrites freed memory, so a method still
        # pointing at a freed tracer crashes instead of passing by luck.
        if self.kernel_mode != "portable" or not sys.platform.startswith("linux"):
            self.skipTest("needs the portable bindings and glibc")
        script = """
import gc
import sys

import torch
from executorch.extension.pybindings import portable_lib as runtime

with open(sys.argv[1], "rb") as source:
    data = source.read()
method = runtime._load_program_from_buffer(
    data, enable_etdump=True, debug_buffer_size=1 << 16
).load_method("forward")
gc.collect()
junk = [bytearray(256) for _ in range(20000)]
for _ in range(5):
    method([torch.ones(2, 2), torch.ones(2, 2)])
print("ran after its program was dropped")
"""
        exported_program, _ = create_program(ModuleAdd())
        with tempfile.TemporaryDirectory() as directory:
            path = os.path.join(directory, "add.pte")
            with open(path, "wb") as file:
                file.write(exported_program.buffer)
            completed = subprocess.run(
                [sys.executable, "-c", script, path],
                env=dict(os.environ, MALLOC_PERTURB_="165"),
                capture_output=True,
                text=True,
                timeout=120,
            )
        self.assertEqual(
            completed.returncode,
            0,
            f"stdout:\n{completed.stdout}\nstderr:\n{completed.stderr}",
        )
        self.assertIn("ran after its program was dropped", completed.stdout)

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
        self.assertIn("Input 0 of method 'forward'", message)
        # Asserts the device is named as Python spells it, since an uppercased or
        # index-less name would not match what the caller passed.
        self.assertIn("is on device meta", message)
        self.assertIn("only CPU and CUDA tensors", message)

    def test_rejects_unexpected_empty_shape(self):
        exported_program, inputs = create_program(ModuleAdd())
        executorch_module = self.load_fn(exported_program.buffer)
        empty_view = torch.empty(5, 7)[0:0, 0:2]

        with self.assertRaises(RuntimeError):
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

    def test_method_refuses_inputs_left_by_a_failed_set_inputs(self):
        exported_program, inputs = create_program(ModuleAdd())
        method = self.load_prog_fn(exported_program.buffer).load_method("forward")
        method.set_inputs(inputs)
        method.execute()
        # Input 0 installs, then a number where input 1 takes a tensor fails,
        # leaving a mix of two calls.
        with self.assertRaises(RuntimeError):
            method.set_inputs([torch.full((2, 2), 5.0), 1.0])

        with self.assertRaisesRegex(RuntimeError, "set_inputs.*again"):
            method.execute()
        # Outputs can alias inputs, so the last results are no longer readable.
        with self.assertRaisesRegex(RuntimeError, "no outputs"):
            method.get_outputs()

        method.set_inputs([torch.full((2, 2), 5.0), torch.full((2, 2), 5.0)])
        method.execute()
        torch.testing.assert_close(method.get_outputs()[0], torch.full((2, 2), 10.0))

    def test_failed_calls_keep_none_of_their_inputs(self):
        exported_program, inputs = create_program(
            ModuleAdd(),
            et_config=ExecutorchBackendConfig(
                memory_planning_pass=MemoryPlanningPass(alloc_graph_input=False)
            ),
        )
        method = self.load_prog_fn(exported_program.buffer).load_method("forward")
        executorch_module = self.load_fn(exported_program.buffer)
        method.set_inputs(inputs)
        executorch_module.forward(inputs)

        # A failed call may leave its first input installed, but nothing reads
        # it before every input is passed again, so none of it is kept.
        for name, call in (
            ("method", lambda values: method.set_inputs(values)),
            ("module", lambda values: executorch_module.forward(values)),
        ):
            with self.subTest(executor=name):
                references = []
                for _ in range(10):
                    installed = torch.full((2, 2), 5.0)
                    references.append(weakref.ref(installed))
                    with self.assertRaises(RuntimeError):
                        call([installed, 1.0])
                    del installed
                gc.collect()
                self.assertEqual(
                    sum(reference() is not None for reference in references), 0
                )

    def test_method_keeps_its_inputs_after_a_wrong_input_count(self):
        exported_program, inputs = create_program(
            ModuleAdd(),
            et_config=ExecutorchBackendConfig(
                memory_planning_pass=MemoryPlanningPass(alloc_graph_input=False)
            ),
        )
        method = self.load_prog_fn(exported_program.buffer).load_method("forward")
        method.set_inputs(inputs)
        rejected = torch.full((2, 2), 5.0)
        rejected_ref = weakref.ref(rejected)
        with self.assertRaisesRegex(RuntimeError, "takes 2 inputs"):
            method.set_inputs([rejected])
        # A rejected call installs nothing, so it must not keep its tensor or
        # tie the method to that tensor's storage.
        rejected.resize_(1 << 16)
        del rejected
        gc.collect()
        self.assertIsNone(rejected_ref())

        method.execute()
        torch.testing.assert_close(method.get_outputs()[0], inputs[0] + inputs[1])

    def test_method_has_no_outputs_without_a_successful_execute(self):
        exported_program, inputs = create_program(
            ModuleAdd(),
            et_config=ExecutorchBackendConfig(
                memory_planning_pass=MemoryPlanningPass(alloc_graph_input=False)
            ),
        )
        method = self.load_prog_fn(exported_program.buffer).load_method("forward")
        with self.assertRaisesRegex(RuntimeError, "no outputs"):
            method.get_outputs()

        method.set_inputs(inputs)
        method.execute()
        torch.testing.assert_close(method.get_outputs()[0], inputs[0] + inputs[1])

        moved = inputs[0].clone()
        method.set_inputs([moved, inputs[1]])
        moved.resize_(64, 64)
        with self.assertRaises(ValueError):
            method.execute()
        with self.assertRaisesRegex(RuntimeError, "no outputs"):
            method.get_outputs()

    def test_execute_before_any_inputs_is_refused(self):
        exported_program, inputs = create_program(
            ModuleAdd(),
            et_config=ExecutorchBackendConfig(
                memory_planning_pass=MemoryPlanningPass(
                    alloc_graph_input=False, alloc_graph_output=False
                )
            ),
        )
        method = self.load_prog_fn(exported_program.buffer).load_method("forward")
        # Without the check this reads inputs that were never set.
        with self.assertRaisesRegex(RuntimeError, "no inputs yet"):
            method.execute()
        executorch_module = self.load_fn(exported_program.buffer)
        with self.assertRaisesRegex(RuntimeError, "Pass every input"):
            executorch_module.plan_execute("forward")
        with self.assertRaisesRegex(RuntimeError, "Pass every input"):
            executorch_module.forward([inputs[0]])

        method.set_inputs(inputs)
        method.execute()
        torch.testing.assert_close(method.get_outputs()[0], inputs[0] + inputs[1])

    def test_method_without_inputs_executes_without_set_inputs(self):
        class Constant(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.register_buffer("value", torch.full((2,), 3.0))

            def forward(self):
                return self.value + 1

        exported_program = to_edge(export(Constant(), (), strict=True)).to_executorch()
        method = self.load_prog_fn(exported_program.buffer).load_method("forward")
        method.execute()
        torch.testing.assert_close(
            torch.as_tensor(np.asarray(method.get_outputs()[0])),
            torch.full((2,), 4.0),
        )

    def test_new_inputs_hide_outputs_that_alias_an_input(self):
        class Identity(torch.nn.Module):
            def forward(self, x):
                return x

        exported_program = to_edge(
            export(Identity(), (torch.ones(2, 2),), strict=True)
        ).to_executorch(
            config=ExecutorchBackendConfig(
                memory_planning_pass=MemoryPlanningPass(
                    alloc_graph_input=False, alloc_graph_output=False
                )
            )
        )
        method = self.load_prog_fn(exported_program.buffer).load_method("forward")
        method.set_inputs([torch.full((2, 2), 3.0)])
        method.execute()
        torch.testing.assert_close(method.get_outputs()[0], torch.full((2, 2), 3.0))

        # The output is the input itself, so it would now show 9 without any
        # execute().
        method.set_inputs([torch.full((2, 2), 9.0)])
        with self.assertRaisesRegex(RuntimeError, "no outputs"):
            method.get_outputs()

    def test_new_inputs_hide_outputs_planned_over_an_input(self):
        class ReluTwice(torch.nn.Module):
            def forward(self, x):
                return x.relu() * 2

        # Default memory planning puts this output in the input's bytes.
        exported_program = to_edge(
            export(ReluTwice(), (torch.ones(4),), strict=True)
        ).to_executorch()
        method = self.load_prog_fn(exported_program.buffer).load_method("forward")
        method.set_inputs([torch.ones(4)])
        method.execute()
        torch.testing.assert_close(method.get_outputs()[0], torch.full((4,), 2.0))

        method.set_inputs([torch.full((4,), 5.0)])
        with self.assertRaisesRegex(RuntimeError, "no outputs"):
            method.get_outputs()

    def test_module_refuses_inputs_left_by_a_failed_call(self):
        exported_program, inputs = create_program(ModuleAdd())
        executorch_module = self.load_fn(exported_program.buffer)
        executorch_module.forward(inputs)
        with self.assertRaises(RuntimeError):
            executorch_module.forward([torch.full((2, 2), 5.0), 1.0])

        with self.assertRaisesRegex(RuntimeError, "Pass every input"):
            executorch_module.plan_execute("forward")

        executorch_module.forward([torch.full((2, 2), 5.0), torch.full((2, 2), 5.0)])
        torch.testing.assert_close(
            executorch_module.plan_execute("forward")[0], torch.full((2, 2), 10.0)
        )

    def test_module_refuses_extra_inputs_before_installing_any(self):
        exported_program, inputs = create_program(ModuleAdd())
        executorch_module = self.load_fn(exported_program.buffer)
        executorch_module.forward(inputs)
        fives = torch.full((2, 2), 5.0)
        with self.assertRaisesRegex(RuntimeError, "takes 2 inputs, but got 3"):
            executorch_module.forward([fives, fives, 1.0])

        # Nothing was installed, so the previous inputs are still whole.
        torch.testing.assert_close(
            executorch_module.plan_execute("forward")[0], inputs[0] + inputs[1]
        )

    def test_module_reuses_inputs_after_a_failure_while_running(self):
        class Take(torch.nn.Module):
            def forward(self, x, index):
                return x.index_select(0, index)

        exported_program = to_edge(
            export(Take(), (torch.ones(4, 2), torch.tensor([1])), strict=True)
        ).to_executorch()
        executorch_module = self.load_fn(exported_program.buffer)
        values = torch.arange(8, dtype=torch.float32).reshape(4, 2)
        # Every input installs, then the out-of-range index fails the run. The
        # installed inputs are one call's, so reusing them is not refused.
        with self.assertRaisesRegex(RuntimeError, "Failed to execute"):
            executorch_module.forward([values, torch.tensor([9])])
        with self.assertRaisesRegex(RuntimeError, "Failed to execute"):
            executorch_module.plan_execute("forward")
        output = executorch_module.forward([values, torch.tensor([3])])
        torch.testing.assert_close(torch.as_tensor(np.asarray(output[0])), values[3:4])

    def test_planned_inputs_are_installed_again_at_each_run(self):
        class Accumulate(torch.nn.Module):
            def forward(self, x, state):
                state.add_(x)
                return state * 1

        x = torch.arange(4, dtype=torch.float32) + 1
        state = torch.zeros(4)
        # Each run writes the planned copy of state, so the next run has to
        # install the caller's tensor again.
        program = to_edge(export(Accumulate(), (x, state), strict=True)).to_executorch()

        method = self.load_prog_fn(program.buffer).load_method("forward")
        method.set_inputs([x, state])
        for _ in range(2):
            method.execute()
            torch.testing.assert_close(method.get_outputs()[0], x)

        module = self.load_fn(program.buffer)
        torch.testing.assert_close(module.forward([x, state])[0], x)
        torch.testing.assert_close(module.forward([x])[0], x)

        # The inputs are read again, so a change to their storage is refused.
        state.resize_(64, 64)
        with self.assertRaisesRegex(ValueError, "storage changed"):
            method.execute()
        with self.assertRaisesRegex(ValueError, "storage changed"):
            module.forward([x])

    def test_module_keeps_a_temporary_program_buffer_alive(self):
        exported_program, inputs = create_program(ModuleAdd())
        data = bytes(bytearray(exported_program.buffer))
        references = sys.getrefcount(data)

        executorch_module = self.load_fn(data)

        # The loader reads the program lazily from these bytes, so the module
        # must own them rather than rely on the caller keeping them alive.
        self.assertGreater(sys.getrefcount(data), references)
        del data
        gc.collect()
        executorch_output = executorch_module.forward(inputs)[0]
        torch.testing.assert_close(executorch_output, inputs[0] + inputs[1])

    def test_bundled_program_module_keeps_its_bytes_alive(self):
        from executorch.devtools.bundled_program.config import (
            MethodTestCase,
            MethodTestSuite,
        )
        from executorch.devtools.bundled_program.core import BundledProgram
        from executorch.devtools.bundled_program.serialize import (
            serialize_from_bundled_program_to_flatbuffer,
        )

        model = ModuleLinear()
        inputs = (torch.ones(2, 3),)
        program = to_edge(export(model, inputs, strict=True)).to_executorch(
            config=ExecutorchBackendConfig(external_constants=True)
        )
        data = bytes(program._tensor_data.pop("_default_external_constant"))
        suite = MethodTestSuite(
            method_name="forward",
            test_cases=[
                MethodTestCase(inputs=inputs, expected_outputs=(model(*inputs),))
            ],
        )
        bundle = serialize_from_bundled_program_to_flatbuffer(
            BundledProgram(program, [suite])
        )
        bundle_bytes = bytes(bytearray(bundle))
        bundle_references = sys.getrefcount(bundle_bytes)
        bundled_module = self.runtime._load_bundled_program_from_buffer(bundle_bytes)
        data_copy = bytes(bytearray(data))
        references = sys.getrefcount(data_copy)
        bundle_references_with_module = sys.getrefcount(bundle_bytes)

        executorch_module = self.runtime._load_for_executorch_from_bundled_program(
            bundled_module, data_copy
        )

        # Both the program, inside the bundle, and the data map are read
        # lazily, so the module must own them.
        self.assertGreater(sys.getrefcount(data_copy), references)
        self.assertGreater(sys.getrefcount(bundle_bytes), bundle_references_with_module)
        self.assertGreater(bundle_references_with_module, bundle_references)
        del bundled_module, data_copy, bundle_bytes
        gc.collect()
        torch.testing.assert_close(executorch_module.forward(inputs)[0], model(*inputs))

        # The overload that reads the data map from a file owns the bundle too.
        with tempfile.TemporaryDirectory() as directory:
            data_path = os.path.join(directory, "model.ptd")
            with open(data_path, "wb") as file:
                file.write(data)
            bundle_bytes = bytes(bytearray(bundle))
            bundled_module = self.runtime._load_bundled_program_from_buffer(
                bundle_bytes
            )
            references = sys.getrefcount(bundle_bytes)
            executorch_module = self.runtime._load_for_executorch_from_bundled_program(
                bundled_module, data_path
            )
            self.assertGreater(sys.getrefcount(bundle_bytes), references)
            del bundled_module, bundle_bytes
            gc.collect()
            torch.testing.assert_close(
                executorch_module.forward(inputs)[0], model(*inputs)
            )
            # Windows cannot delete the data file while the module maps it.
            del executorch_module
            gc.collect()

    def test_numbers_for_number_inputs(self):
        class AddInt(torch.nn.Module):
            def forward(self, x, n: int):
                return x + n

        class Choose(torch.nn.Module):
            def forward(self, x, flag: bool):
                return x + 1 if flag else x - 1

        class NotNumpy:
            ndim = 0

            def item(self):
                return 1.0

        # Each case: model, example, accepted values, expected, wrong kinds.
        cases = (
            (
                ModuleAddScalar(),
                1.0,
                (1, np.int64(1), np.float32(1.0), np.array(1.0)),
                2.0,
                (True, np.bool_(True)),
            ),
            (
                AddInt(),
                1,
                (np.int64(1), np.array(1)),
                2.0,
                (True, np.bool_(True), 1.5, np.float32(1.0)),
            ),
            (Choose(), True, (np.bool_(True), np.array(True)), 2.0, (1, 1.0)),
        )
        for model, example, values, expected, wrong_kinds in cases:
            exported_program = to_edge(
                export(model, (torch.ones(2), example), strict=True)
            ).to_executorch()
            executorch_module = self.load_fn(exported_program.buffer)
            method = self.load_prog_fn(exported_program.buffer).load_method("forward")
            for executor in (executorch_module, method):
                for value in values:
                    with self.subTest(
                        model=type(model).__name__,
                        executor=type(executor).__name__,
                        value=repr(value),
                    ):
                        output = executor([torch.ones(2), value])[0]
                        torch.testing.assert_close(
                            torch.as_tensor(np.asarray(output)),
                            torch.full((2,), expected),
                        )
                for wrong in (
                    np.array([1.0, 2.0]),
                    NotNumpy(),
                    torch.tensor(1.0),
                    *wrong_kinds,
                ):
                    with self.subTest(
                        model=type(model).__name__,
                        executor=type(executor).__name__,
                        wrong=repr(wrong),
                    ):
                        with self.assertRaisesRegex(
                            RuntimeError, "of method 'forward'"
                        ):
                            executor([torch.ones(2), wrong])

    def test_buffer_formats_with_native_byte_order_prefix(self):
        exported_program, _ = create_program(ModuleAddSingleInput())
        executorch_module = self.load_fn(exported_program.buffer)
        values = np.arange(4, dtype=np.float32)
        foreign = ">" if sys.byteorder == "little" else "<"

        # Pass each object itself: converting through NumPy would drop the
        # explicit prefix under test.
        for value in (
            (ctypes.c_float * 2 * 2)((0.0, 1.0), (2.0, 3.0)),
            memoryview(values.tobytes()).cast("B").cast("@f", (2, 2)),
        ):
            with self.subTest(format=memoryview(value).format):
                output = executorch_module(value)[0]
                np.testing.assert_array_equal(
                    np.asarray(output).reshape(-1), values * 2
                )

        with self.assertRaisesRegex(
            BufferError, "Input 0 of method 'forward': .*byte order"
        ):
            executorch_module(np.ones((2, 2), dtype=f"{foreign}f4"))

    def test_ctypes_integer_buffers(self):
        class Double(torch.nn.Module):
            def forward(self, x):
                return x * 2

        exported_program = to_edge(
            export(Double(), (torch.ones(4, dtype=torch.int64),), strict=True)
        ).to_executorch()
        executorch_module = self.load_fn(exported_program.buffer)

        # ctypes spells the byte order out, for example "<q".
        output = executorch_module((ctypes.c_int64 * 4)(1, 2, 3, 4))[0]

        np.testing.assert_array_equal(np.asarray(output), np.array([2, 4, 6, 8]))

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
        self.assertIn("Input 0 of method 'forward'", message)
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

    def test_method_meta_reports_tensor_devices(self):
        from executorch.exir._serialize._program import PTEFile, serialize_pte_binary
        from executorch.exir.schema import ExtraTensorInfo, Tensor

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
        host_meta = self.load_prog_fn(program.buffer).method_meta("forward")
        self.assertEqual(host_meta.input_tensor_meta(0).device(), "cpu")
        self.assertEqual(host_meta.output_tensor_meta(0).device(), "cpu")
        schema = program.executorch_program
        for entry in schema.execution_plan[0].values:
            if isinstance(entry.val, Tensor):
                entry.val.extra_tensor_info = ExtraTensorInfo(
                    device_type=DeviceType.CUDA, device_index=3
                )
        data = bytes(serialize_pte_binary(PTEFile(schema)))
        for meta in (
            self.load_prog_fn(data).method_meta("forward"),
            self.load_fn(data).method_meta("forward"),
        ):
            self.assertEqual(meta.input_tensor_meta(0).device(), "cuda:3")
            self.assertEqual(meta.output_tensor_meta(0).device(), "cuda:3")
            self.assertIn("device=cuda:3", repr(meta.input_tensor_meta(0)))

        # A named buffer reports its device too.
        eager_module = ModuleAddWithAttributes()
        program = to_edge(
            export(eager_module, eager_module.get_inputs(), strict=True)
        ).to_executorch(ExecutorchBackendConfig(emit_mutable_buffer_names=True))
        schema = program.executorch_program
        for entry in schema.execution_plan[0].values:
            if isinstance(entry.val, Tensor):
                entry.val.extra_tensor_info = ExtraTensorInfo(
                    device_type=DeviceType.CUDA,
                    device_index=3,
                    fully_qualified_name=(
                        entry.val.extra_tensor_info.fully_qualified_name
                        if entry.val.extra_tensor_info
                        else None
                    ),
                )
        meta = self.load_prog_fn(
            bytes(serialize_pte_binary(PTEFile(schema)))
        ).method_meta("forward")
        self.assertEqual(meta.num_attributes(), 1)
        self.assertEqual(meta.attribute_tensor_meta(0).device(), "cuda:3")

    def test_input_device_must_match_method(self):
        if self.kernel_mode != "portable":
            self.skipTest("only portable bindings inspect input device metadata")

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
        # A host buffer would otherwise be read as device memory.
        for executor in (module, method):
            for host_input in (
                value,
                DLPackOnly(value),
                value.numpy(),
                memoryview(value.numpy()),
            ):
                with self.subTest(executor=type(executor), input=type(host_input)):
                    with self.assertRaisesRegex(
                        ValueError,
                        r"^Input 0 of method 'forward' is on device cpu, but the "
                        r"method expects cuda:3\. Pass device memory instead, such "
                        r"as ExecuTorchResult\.from_host\(\.\.\.\) or a CUDA tensor$",
                    ):
                        executor(host_input)

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
