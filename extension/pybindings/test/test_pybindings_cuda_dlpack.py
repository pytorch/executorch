# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-unsafe

"""CUDA DLPack integration tests, run in both output modes by cuda.yml."""

import array
import ctypes
import functools
import gc
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import torch


class CudaDLPackOnly:
    """A generic producer whose writes belong to a particular CUDA stream."""

    def __init__(self, tensor, producer_stream):
        self.tensor = tensor
        self.producer_stream = producer_stream
        self.requested_streams = []

    def __dlpack_device__(self):
        return self.tensor.__dlpack_device__()

    def __dlpack__(self, stream=None):
        self.requested_streams.append(stream)
        with torch.cuda.stream(self.producer_stream):
            return self.tensor.__dlpack__(stream=stream)


class CudaDLPackTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        required = os.environ.get("EXECUTORCH_PYBIND_TEST_REQUIRE_CUDA") == "1"
        if not torch.cuda.is_available():
            if required:
                raise AssertionError("CUDA DLPack CI requires a visible GPU")
            raise unittest.SkipTest("requires a visible CUDA GPU")

        from executorch.extension.pybindings import portable_lib

        if portable_lib._uses_aten:
            raise AssertionError(
                "CUDA DLPack CI must exercise bindings without C++ ATen"
            )
        if "CudaBackend" not in portable_lib._get_registered_backend_names():
            if required:
                raise AssertionError("CUDA DLPack CI requires the CUDA backend")
            raise unittest.SkipTest("requires the CUDA backend")

        cls.bindings = portable_lib
        cls.executorch_output = (
            os.environ.get("EXECUTORCH_PYBINDINGS_TENSOR_OUTPUT") == "executorch"
        )
        cls.directory = tempfile.TemporaryDirectory()
        cls.addClassCleanup(cls.directory.cleanup)
        cls.device = torch.device("cuda:0")
        cls.consumer_streams = {0: torch.cuda.Stream(device=cls.device)}
        cls.driver = ctypes.CDLL("libcuda.so.1")
        cls.driver.cuPointerGetAttribute.argtypes = [
            ctypes.c_void_p,
            ctypes.c_int,
            ctypes.c_uint64,
        ]
        cls.driver.cuPointerGetAttribute.restype = ctypes.c_int
        cls._export_delegate()

    @classmethod
    def _export_delegate(cls):
        from executorch.backends.cuda.cuda_partitioner import CudaPartitioner
        from executorch.exir import ExecutorchBackendConfig, to_edge_transform_and_lower
        from executorch.exir.backend.compile_spec_schema import CompileSpec
        from executorch.exir.passes import MemoryPlanningPass
        from executorch.exir.passes.propagate_device_pass import PropagateDeviceConfig
        from executorch.exir.schema import DeviceType

        class Add(torch.nn.Module):
            def forward(self, value):
                return (value + 1.0) * 2.0

        cls.shape = (256, 256)
        edge = to_edge_transform_and_lower(
            {
                "forward": torch.export.export(
                    Add(), (torch.ones(cls.shape),), strict=True
                )
            },
            partitioner={
                "forward": [CudaPartitioner([CompileSpec("method_name", b"forward")])]
            },
        )
        program = edge.to_executorch(
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
        plan = program.executorch_program.execution_plan[0]
        if not any(delegate.id == "CudaBackend" for delegate in plan.delegates):
            raise AssertionError("test model must execute a CUDA delegate")
        for index in (*plan.inputs, *plan.outputs):
            tensor = plan.values[index].val
            if tensor.allocation_info is not None:
                raise AssertionError(
                    "test must cover borrowed inputs and unplanned outputs"
                )
            if tensor.extra_tensor_info.device_type != DeviceType.CUDA:
                raise AssertionError("test must retain CUDA inputs and outputs")

        cls.program_path = Path(cls.directory.name) / "cuda_dlpack.pte"
        with cls.program_path.open("wb") as source:
            program.write_to_file(source)
        data_names = sorted(program._tensor_data or {})
        program.write_tensor_data_to_file(cls.directory.name)
        cls.data_path = (
            Path(cls.directory.name) / (data_names[0] + ".ptd") if data_names else None
        )

    def _load_method(self):
        from executorch.runtime import Runtime

        return (
            Runtime.get()
            .load_program(self.program_path, data_path=self.data_path)
            .load_method("forward")
            ._method
        )

    def _consumer_stream(self, device_index):
        if device_index not in self.consumer_streams:
            self.consumer_streams[device_index] = torch.cuda.Stream(
                device=torch.device("cuda", device_index)
            )
        return self.consumer_streams[device_index]

    def _tensor(self, output, device_index=0):
        if self.executorch_output:
            self.assertIsInstance(output, self.bindings.ExecuTorchResult)
        else:
            self.assertIsInstance(output, torch.Tensor)
        self.assertEqual(output.__dlpack_device__(), (2, device_index))
        # Import on a separate stream, then ask the driver about the original
        # allocation. A CUDA device tag alone also describes a bad host pointer.
        with torch.cuda.stream(self._consumer_stream(device_index)):
            tensor = torch.from_dlpack(output)
        memory_type = ctypes.c_uint()
        device_ordinal = ctypes.c_int()
        for attribute, storage, expected in (
            (2, memory_type, 2),  # CU_POINTER_ATTRIBUTE_MEMORY_TYPE: DEVICE
            (9, device_ordinal, device_index),  # DEVICE_ORDINAL
        ):
            status = self.driver.cuPointerGetAttribute(
                ctypes.byref(storage), attribute, tensor.data_ptr()
            )
            self.assertEqual((status, storage.value), (0, expected))
        self.assertEqual(tensor.device, torch.device("cuda", device_index))
        return tensor

    def _assert_ready_value(self, output, expected):
        tensor = self._tensor(output)
        with torch.cuda.stream(self._consumer_stream(0)):
            snapshot = tensor.clone()
            ready = torch.cuda.Event()
            ready.record()
        # Only wait for the consumer. A global wait before the read would hide
        # missing ordering between the producer, delegate, and consumer streams.
        ready.synchronize()
        torch.testing.assert_close(snapshot.cpu(), torch.full(self.shape, expected))
        return tensor

    def test_delegate_cross_stream_input_and_output(self):
        method = self._load_method()
        producer_stream = torch.cuda.Stream(device=self.device)
        for clone_outputs, protocol in (
            (clone, protocol)
            for clone in (False, True)
            for protocol in ("dlpack", "torch")
        ):
            with self.subTest(clone_outputs=clone_outputs, protocol=protocol):
                # Include a nonzero storage offset. Initialize before starting
                # the delayed producer so an unordered read is observably wrong.
                storage = torch.zeros(
                    torch.Size(self.shape).numel() + 7, device=self.device
                )
                value = storage[7:].view(self.shape)
                torch.cuda.current_stream(self.device).synchronize()
                with torch.cuda.stream(producer_stream):
                    torch.cuda._sleep(50_000_000)
                    value.fill_(41.0)
                producer = CudaDLPackOnly(value, producer_stream)
                if protocol == "dlpack":
                    output = method(producer, clone_outputs=clone_outputs)[0]
                    self.assertTrue(producer.requested_streams)
                    self.assertNotIn(None, producer.requested_streams)
                    self.assertNotIn(-1, producer.requested_streams)
                else:
                    # Torch's DLPack exporter discovers its producer through
                    # the current stream, so keep that stream selected here.
                    with torch.cuda.stream(producer_stream):
                        output = method(value, clone_outputs=clone_outputs)[0]
                tensor = self._assert_ready_value(output, 84.0)
                self.assertNotEqual(tensor.data_ptr(), value.data_ptr())

    def test_clone_roundtrip_and_borrowed_output_lifetime(self):
        method = self._load_method()
        stream = torch.cuda.current_stream(self.device)
        value = torch.full(self.shape, 3.0, device=self.device)
        borrowed = method(CudaDLPackOnly(value, stream), clone_outputs=False)[0]
        borrowed_tensor = self._assert_ready_value(borrowed, 8.0)
        cloned = method(CudaDLPackOnly(value, stream), clone_outputs=True)[0]
        cloned_tensor = self._assert_ready_value(cloned, 8.0)
        self.assertNotEqual(cloned_tensor.data_ptr(), borrowed_tensor.data_ptr())

        # Feed the public result itself back in, including ExecuTorchResult in
        # the executorch output mode. Use another method to avoid aliasing its
        # output arena with the input under test.
        second_method = self._load_method()
        roundtrip = second_method(borrowed, clone_outputs=False)[0]
        self._assert_ready_value(roundtrip, 18.0)
        replacement = torch.full(self.shape, 10.0, device=self.device)
        changed = method(CudaDLPackOnly(replacement, stream), clone_outputs=False)[0]
        self._assert_ready_value(changed, 22.0)
        self._assert_ready_value(cloned, 8.0)

        del method, second_method, value, replacement
        del borrowed, borrowed_tensor, cloned_tensor, roundtrip
        gc.collect()
        self._assert_ready_value(changed, 22.0)
        self._assert_ready_value(cloned, 8.0)

    def _device_copies(self, call):
        from torch.profiler import profile, ProfilerActivity

        source = torch.ones(16, device=self.device)
        target = torch.empty_like(source)
        with profile(activities=[ProfilerActivity.CUDA]) as trace:
            output = call()
            # One known copy, so a profiler that records nothing cannot pass.
            target.copy_(source)
            torch.cuda.synchronize(self.device)
        copies = sum(
            event.count for event in trace.key_averages() if "Memcpy DtoD" in event.key
        )
        self.assertGreaterEqual(copies, 1)
        return output, copies - 1

    def test_device_outputs_are_handed_over_without_a_copy(self):
        method = self._load_method()
        stream = torch.cuda.current_stream(self.device)

        def run(value, clone_outputs=True):
            return method(CudaDLPackOnly(value, stream), clone_outputs=clone_outputs)[0]

        run(torch.full(self.shape, 1.0, device=self.device))
        borrowed, borrowed_copies = self._device_copies(
            lambda: run(torch.full(self.shape, 1.0, device=self.device), False)
        )
        # A live borrowed result shares the buffer, so a default result made
        # while it lives is a copy. Without it, nothing is copied.
        del borrowed
        first, handed_copies = self._device_copies(
            lambda: run(torch.full(self.shape, 3.0, device=self.device))
        )
        # The delegate may copy into its output; returning it adds no copy.
        self.assertEqual(handed_copies, borrowed_copies)
        self._assert_ready_value(first, 8.0)
        self.assertIs(method.get_outputs()[0], first)
        self.assertIs(method.get_outputs(clone_outputs=False)[0], first)

        # Feeding the result back must not write the next output into it.
        second = method(first)[0]
        self._assert_ready_value(second, 18.0)
        self._assert_ready_value(first, 8.0)
        third = method(second)[0]
        self._assert_ready_value(third, 38.0)
        self._assert_ready_value(first, 8.0)
        self._assert_ready_value(second, 18.0)

        # Once the caller drops the results, the next call reuses the memory.
        address = self._tensor(third).data_ptr()
        del first, second, third
        gc.collect()
        again = method(
            CudaDLPackOnly(torch.ones(self.shape, device=self.device), stream)
        )[0]
        self.assertEqual(self._tensor(again).data_ptr(), address)

    def test_module_hands_device_outputs_over_without_copies(self):
        program = self.program_path.read_bytes()
        data = self.data_path.read_bytes() if self.data_path else None
        module = self.bindings._load_for_executorch_from_buffer(program, data)
        stream = torch.cuda.current_stream(self.device)
        inputs = [
            [CudaDLPackOnly(torch.full(self.shape, value, device=self.device), stream)]
            for value in (1.0, 1.0, 3.0)
        ]
        module.forward(inputs[0])
        borrowed, borrowed_copies = self._device_copies(
            functools.partial(module.forward, inputs[1], clone_outputs=False)
        )
        outputs, handed_copies = self._device_copies(
            functools.partial(module.forward, inputs[2])
        )
        self.assertEqual(handed_copies, borrowed_copies)
        first = outputs[0]
        second = module.forward([first])[0]
        self._assert_ready_value(first, 8.0)
        self._assert_ready_value(second, 18.0)

        # The native module owns the program bytes, so their count shows
        # whether a result still keeps the module alive.
        references = sys.getrefcount(program)
        del module, outputs, borrowed, first
        gc.collect()
        self.assertLess(sys.getrefcount(program), references)
        self._assert_ready_value(second, 18.0)

    def _device_identity(self, device_index, example):
        from executorch.exir import ExecutorchBackendConfig, to_edge
        from executorch.exir._serialize._program import PTEFile, serialize_pte_binary
        from executorch.exir.passes import MemoryPlanningPass
        from executorch.exir.schema import DeviceType, ExtraTensorInfo, Tensor

        class Identity(torch.nn.Module):
            def forward(self, value):
                return value

        program = to_edge(torch.export.export(Identity(), (example,))).to_executorch(
            ExecutorchBackendConfig(
                memory_planning_pass=MemoryPlanningPass(
                    alloc_graph_input=False, alloc_graph_output=False
                )
            )
        )
        schema = program.executorch_program
        for entry in schema.execution_plan[0].values:
            if isinstance(entry.val, Tensor):
                entry.val.extra_tensor_info = ExtraTensorInfo(
                    device_type=DeviceType.CUDA, device_index=device_index
                )
        return bytes(serialize_pte_binary(PTEFile(schema)))

    def test_cuda_input_in_another_layout(self):
        data = self._device_identity(0, torch.ones(2, 3, 4, 5))
        method = self.bindings._load_program_from_buffer(data).load_method("forward")
        value = (
            torch.arange(120, dtype=torch.float32, device=self.device)
            .reshape(2, 3, 4, 5)
            .to(memory_format=torch.channels_last)
        )
        # Torch converts its tensor on the device into the exported layout.
        output = self._tensor(method(value)[0])
        self.assertTrue(output.is_contiguous())
        torch.testing.assert_close(output, value.contiguous())
        # Other producers have no way to convert device memory here.
        stream = torch.cuda.current_stream(self.device)
        with self.assertRaisesRegex(ValueError, "laid out differently"):
            method(CudaDLPackOnly(value, stream))

    def test_dlpack_input_on_a_non_current_device(self):
        if torch.cuda.device_count() < 2:
            self.skipTest("needs two CUDA devices")

        class PlainProducer:
            def __init__(self, tensor):
                self.tensor = tensor

            def __dlpack_device__(self):
                return self.tensor.__dlpack_device__()

            def __dlpack__(self, stream=None):
                return self.tensor.__dlpack__(stream=stream)

        data = self._device_identity(1, torch.ones(2, 3, 4, 5))
        method = self.bindings._load_program_from_buffer(data).load_method("forward")
        value = torch.arange(120, dtype=torch.float32, device="cuda:1").reshape(
            2, 3, 4, 5
        )
        self.assertEqual(torch.cuda.current_device(), 0)
        output = method(PlainProducer(value))[0]
        torch.testing.assert_close(self._tensor(output, 1).cpu(), value.cpu())

    def test_several_device_outputs_stay_valid(self):
        from executorch.backends.cuda.cuda_partitioner import CudaPartitioner
        from executorch.exir import ExecutorchBackendConfig, to_edge_transform_and_lower
        from executorch.exir.backend.compile_spec_schema import CompileSpec
        from executorch.exir.passes import MemoryPlanningPass
        from executorch.exir.passes.propagate_device_pass import PropagateDeviceConfig

        class Outputs(torch.nn.Module):
            def forward(self, value):
                result = (value + 1.0) * 2.0
                return result, result, value * 3.0

        edge = to_edge_transform_and_lower(
            {
                "forward": torch.export.export(
                    Outputs(), (torch.ones(self.shape),), strict=True
                )
            },
            partitioner={
                "forward": [CudaPartitioner([CompileSpec("method_name", b"forward")])]
            },
        )
        program = edge.to_executorch(
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
        directory = Path(self.directory.name) / "outputs"
        directory.mkdir(exist_ok=True)
        program_path = directory / "outputs.pte"
        with program_path.open("wb") as source:
            program.write_to_file(source)
        data_names = sorted(program._tensor_data or {})
        program.write_tensor_data_to_file(str(directory))
        data_path = directory / (data_names[0] + ".ptd") if data_names else None
        stream = torch.cuda.current_stream(self.device)
        load = functools.partial(
            self.bindings._load_program,
            str(program_path),
            data_path=str(data_path) if data_path else None,
        )
        value = torch.full(self.shape, 3.0, device=self.device)
        for clone_outputs in (False, True):
            with self.subTest(clone_outputs=clone_outputs):
                method = load().load_method("forward")
                first, second, third = method(
                    CudaDLPackOnly(value, stream), clone_outputs=clone_outputs
                )
                if clone_outputs:
                    # Each default result is independent, even of a duplicate.
                    self.assertNotEqual(
                        self._tensor(first).data_ptr(),
                        self._tensor(second).data_ptr(),
                    )
                self.assertNotEqual(
                    self._tensor(first).data_ptr(), self._tensor(third).data_ptr()
                )
                del second
                gc.collect()
                if clone_outputs:
                    method(
                        CudaDLPackOnly(
                            torch.full(self.shape, 10.0, device=self.device), stream
                        )
                    )
                else:
                    # A borrowed result keeps the memory it reads alive.
                    del method
                    gc.collect()
                self._assert_ready_value(first, 8.0)
                self._assert_ready_value(third, 9.0)

    def test_reused_output_memory_waits_for_pending_readers(self):
        method = self._load_method()
        value = torch.full(self.shape, 1.0, device=self.device)
        stream = torch.cuda.current_stream(self.device)
        reader = torch.cuda.Stream(self.device)
        method(CudaDLPackOnly(value, stream))
        result = method.get_outputs()[0]
        # Queue a slow read of the result, then drop it before the read runs.
        with torch.cuda.stream(reader):
            torch.cuda._sleep(50_000_000)
            snapshot = self._tensor(result).clone()
        del result
        gc.collect()
        value.fill_(5.0)
        method(CudaDLPackOnly(value, stream))
        reader.synchronize()
        torch.testing.assert_close(snapshot.cpu(), torch.full(self.shape, 4.0))

    def test_host_upload_and_download_without_torch_tensors(self):
        method = self._load_method()
        count = torch.Size(self.shape).numel()
        host = (
            memoryview(array.array("f", [3.0]) * count).cast("B").cast("f", self.shape)
        )

        uploaded = self.bindings.ExecuTorchResult.from_host(host, self.device.index)
        self.assertEqual(uploaded.__dlpack_device__(), (2, self.device.index))
        output = method(uploaded, clone_outputs=True)[0]
        if self.executorch_output:
            downloaded = torch.from_dlpack(output.to_host())
        else:
            downloaded = output.cpu()

        torch.testing.assert_close(downloaded, torch.full(self.shape, 8.0))

    def test_bfloat16_host_result_moves_to_the_device(self):
        if not self.executorch_output:
            self.skipTest("ExecuTorchResult output was not selected")
        value = torch.arange(6, dtype=torch.bfloat16, device=self.device).reshape(2, 3)
        data = self._device_identity(self.device.index, value.cpu())
        method = self.bindings._load_program_from_buffer(data).load_method("forward")
        on_device = method(value)[0]
        host = on_device.to_host()
        self.assertEqual(host.dtype_name, "bfloat16")
        # The host export of bfloat16 is raw bytes, so passing the result itself
        # is what keeps its dtype and shape.
        back = self.bindings.ExecuTorchResult.from_host(host, self.device.index)
        self.assertEqual(back.dtype_name, "bfloat16")
        self.assertEqual(back.shape, (2, 3))
        torch.testing.assert_close(torch.from_dlpack(back), value)
        with self.assertRaisesRegex(ValueError, "already on a device"):
            self.bindings.ExecuTorchResult.from_host(on_device, self.device.index)

    def _require_libgomp_without_torch(self):
        # The torch-free chain checks loaded libraries through /proc.
        if not sys.platform.startswith("linux"):
            self.skipTest("needs Linux")
        # The CUDA delegate's compiled kernels link libgomp, which a host may
        # provide only inside the torch package.
        probe = subprocess.run(
            [sys.executable, "-c", "import ctypes; ctypes.CDLL('libgomp.so.1')"],
            capture_output=True,
        )
        if probe.returncode != 0:
            message = "libgomp.so.1 is not loadable without torch"
            if os.environ.get("EXECUTORCH_PYBIND_TEST_REQUIRE_CUDA") == "1":
                raise AssertionError(message)
            self.skipTest(message)

    def test_device_chain_runs_with_torch_blocked(self):
        # The device-resident chain path, in a process where `import torch`
        # fails: upload host data, run, feed the output straight into a second
        # run on the GPU, then read the result back.
        self._require_libgomp_without_torch()
        script = """
import importlib.abc
import re
import sys
from pathlib import Path

class BlockTorch(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname == "torch" or fullname.startswith("torch."):
            raise ImportError("torch is unavailable")

sys.meta_path.insert(0, BlockTorch())
import numpy as np
from executorch.extension.pybindings import _C as native, portable_lib as runtime

assert not runtime._uses_aten
assert native._tensor_output == "executorch"
program_path, data_path, rows, columns = sys.argv[1:5]
shape = (int(rows), int(columns))
data = data_path or None
module = runtime._load_for_executorch(program_path, data)
method = runtime._load_program(program_path, data_path=data).load_method("forward")
for executor in (module, method):
    host = np.full(shape, 3.0, dtype=np.float32)
    uploaded = runtime.ExecuTorchResult.from_host(host, 0)
    first = executor([uploaded])[0]
    assert first.__dlpack_device__() == (2, 0)
    second = executor([first])[0]
    assert second.__dlpack_device__() == (2, 0)
    np.testing.assert_array_equal(
        np.asarray(second.to_host()), np.full(shape, 18.0, dtype=np.float32)
    )
assert "torch" not in sys.modules
assert re.search(
    r"lib(?:torch|aten|c10)", Path("/proc/self/maps").read_text().lower()
) is None
print("Device chain ran without torch")
"""
        environment = os.environ.copy()
        environment["EXECUTORCH_PYBINDINGS_TENSOR_OUTPUT"] = "executorch"
        completed = subprocess.run(
            [
                sys.executable,
                "-c",
                script,
                str(self.program_path),
                str(self.data_path) if self.data_path else "",
                *map(str, self.shape),
            ],
            env=environment,
            capture_output=True,
            text=True,
            timeout=120,
        )
        self.assertEqual(
            completed.returncode,
            0,
            f"stdout:\n{completed.stdout}\nstderr:\n{completed.stderr}",
        )
        self.assertIn("Device chain ran without torch", completed.stdout)

    def test_device_input_works_beside_a_torch_without_cuda(self):
        # A torch built without CUDA cannot select a CUDA device. Inputs from
        # other producers must not depend on it.
        self._require_libgomp_without_torch()
        script = """
import sys
import types

import numpy as np

class CpuOnlyCuda:
    @staticmethod
    def is_available():
        return False

    @staticmethod
    def device(index):
        raise AssertionError("Torch not compiled with CUDA enabled")

class Tensor:
    pass

sys.modules["torch"] = types.SimpleNamespace(cuda=CpuOnlyCuda(), Tensor=Tensor)
from executorch.extension.pybindings import portable_lib as runtime

program_path, data_path, rows, columns = sys.argv[1:5]
shape = (int(rows), int(columns))
module = runtime._load_for_executorch(program_path, data_path or None)
uploaded = runtime.ExecuTorchResult.from_host(np.full(shape, 3.0, np.float32), 0)
second = module.forward([module.forward([uploaded])[0]])[0]
np.testing.assert_array_equal(
    np.asarray(second.to_host()), np.full(shape, 18.0, dtype=np.float32)
)
print("Device input worked beside a torch without CUDA")
"""
        environment = os.environ.copy()
        environment["EXECUTORCH_PYBINDINGS_TENSOR_OUTPUT"] = "executorch"
        completed = subprocess.run(
            [
                sys.executable,
                "-c",
                script,
                str(self.program_path),
                str(self.data_path) if self.data_path else "",
                *map(str, self.shape),
            ],
            env=environment,
            capture_output=True,
            text=True,
            timeout=120,
        )
        self.assertEqual(
            completed.returncode,
            0,
            f"stdout:\n{completed.stdout}\nstderr:\n{completed.stderr}",
        )
        self.assertIn(
            "Device input worked beside a torch without CUDA", completed.stdout
        )

    def test_host_upload_to_a_missing_device_leaves_cuda_usable(self):
        missing = torch.cuda.device_count()
        for count in (0, 4):
            with self.subTest(elements=count):
                with self.assertRaisesRegex(ValueError, "not available"):
                    self.bindings.ExecuTorchResult.from_host(
                        memoryview(array.array("f", [1.0] * count)), missing
                    )
                # A failed device selection would surface in the next CUDA call.
                torch.ones(1, device=self.device).add_(1)
                torch.cuda.synchronize(self.device)

    def test_identity_preserves_channels_last_offset_and_device_index(self):
        from executorch.exir import ExecutorchBackendConfig, to_edge
        from executorch.exir._serialize._program import PTEFile, serialize_pte_binary
        from executorch.exir.passes import MemoryPlanningPass
        from executorch.exir.schema import DeviceType, ExtraTensorInfo, Tensor

        class Identity(torch.nn.Module):
            def forward(self, value):
                return value

        shape = (2, 3, 4, 5)
        reference = torch.ones(shape).to(memory_format=torch.channels_last)
        program = to_edge(torch.export.export(Identity(), (reference,))).to_executorch(
            ExecutorchBackendConfig(
                memory_planning_pass=MemoryPlanningPass(
                    alloc_graph_input=False, alloc_graph_output=False
                )
            )
        )
        device_index = torch.cuda.device_count() - 1
        schema = program.executorch_program
        for entry in schema.execution_plan[0].values:
            if isinstance(entry.val, Tensor):
                entry.val.extra_tensor_info = ExtraTensorInfo(
                    device_type=DeviceType.CUDA, device_index=device_index
                )
        data = bytes(serialize_pte_binary(PTEFile(schema)))
        storage = torch.arange(
            reference.numel() + 11, device=f"cuda:{device_index}", dtype=reference.dtype
        )
        value = torch.as_strided(storage, shape, reference.stride(), storage_offset=11)
        stream = torch.cuda.current_stream(device_index)
        module = self.bindings._load_for_executorch_from_buffer(data)
        method = self.bindings._load_program_from_buffer(data).load_method("forward")
        for input_value in (value, CudaDLPackOnly(value, stream)):
            for executor in (module, method):
                for clone_outputs in (False, True):
                    with self.subTest(
                        protocol=type(input_value),
                        executor=type(executor),
                        clone_outputs=clone_outputs,
                    ):
                        output = executor(input_value, clone_outputs=clone_outputs)[0]
                        tensor = self._tensor(output, device_index)
                        self.assertEqual(tensor.stride(), value.stride())
                        with torch.cuda.stream(self._consumer_stream(device_index)):
                            torch.testing.assert_close(tensor.cpu(), value.cpu())
                        if clone_outputs:
                            self.assertNotEqual(tensor.data_ptr(), value.data_ptr())
                        else:
                            self.assertEqual(tensor.data_ptr(), value.data_ptr())
                            roundtrip = executor(output, clone_outputs=False)[0]
                            self.assertEqual(
                                self._tensor(roundtrip, device_index).data_ptr(),
                                value.data_ptr(),
                            )


if __name__ == "__main__":
    unittest.main()
