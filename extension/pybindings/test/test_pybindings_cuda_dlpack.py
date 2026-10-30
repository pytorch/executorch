# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-unsafe

"""CUDA DLPack integration tests, run in both output modes by cuda.yml."""

import array
import contextlib
import copy
import ctypes
import functools
import gc
import os
import subprocess
import sys
import tempfile
import types
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
        cls.program_path, cls.data_path = cls._export_delegate("cuda_dlpack")
        cls.host_input_program = cls._export_delegate(
            "host_inputs", inputs_on_device=False
        )

    @classmethod
    def _export_delegate(
        cls, name, inputs_on_device=True, outputs_on_device=True, plan_outputs=False
    ):
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
                    skip_h2d_for_method_inputs=inputs_on_device,
                    skip_d2h_for_method_outputs=outputs_on_device,
                ),
                enable_non_cpu_memory_planning=True,
                memory_planning_pass=MemoryPlanningPass(
                    alloc_graph_input=False, alloc_graph_output=plan_outputs
                ),
            )
        )
        plan = program.executorch_program.execution_plan[0]
        if not any(delegate.id == "CudaBackend" for delegate in plan.delegates):
            raise AssertionError("test model must execute a CUDA delegate")
        for index in (*plan.inputs, *plan.outputs):
            tensor = plan.values[index].val
            is_output = index in plan.outputs
            on_device = outputs_on_device if is_output else inputs_on_device
            info = tensor.extra_tensor_info
            if on_device != (info is not None and info.device_type == DeviceType.CUDA):
                raise AssertionError("test program has a wrong device")
            planned = tensor.allocation_info is not None
            if planned != (plan_outputs and is_output):
                raise AssertionError("test program is planned differently")

        directory = Path(cls.directory.name) / name
        directory.mkdir()
        program_path = directory / f"{name}.pte"
        with program_path.open("wb") as source:
            program.write_to_file(source)
        data_names = sorted(program._tensor_data or {})
        program.write_tensor_data_to_file(str(directory))
        return program_path, (
            directory / (data_names[0] + ".ptd") if data_names else None
        )

    def _load_method(self, program=None):
        from executorch.runtime import Runtime

        program_path, data_path = program or (self.program_path, self.data_path)
        return (
            Runtime.get()
            .load_program(program_path, data_path=data_path)
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
        # An ExecuTorchResult waits for the device in __dlpack__ already, so in
        # that output mode this catches a wrong value, not a missing wait.
        ready.synchronize()
        torch.testing.assert_close(snapshot.cpu(), torch.full(self.shape, expected))
        return tensor

    def _export_to_directory(self, name, module, example):
        from executorch.backends.cuda.cuda_partitioner import CudaPartitioner
        from executorch.exir import to_edge_transform_and_lower
        from executorch.exir.backend.compile_spec_schema import CompileSpec

        directory = os.path.join(self.directory.name, name)
        os.makedirs(directory, exist_ok=True)
        program = to_edge_transform_and_lower(
            torch.export.export(module, example, strict=True),
            partitioner=[CudaPartitioner([CompileSpec("method_name", b"forward")])],
        ).to_executorch()
        plan = program.executorch_program.execution_plan[0]
        if not any(delegate.id == "CudaBackend" for delegate in plan.delegates):
            raise AssertionError("test model must execute a CUDA delegate")
        path = os.path.join(directory, "model.pte")
        with open(path, "wb") as file:
            file.write(program.buffer)
        program.write_tensor_data_to_file(directory)
        data = [f for f in os.listdir(directory) if f.endswith(".ptd")]
        return path, os.path.join(directory, data[0]) if data else None

    def test_delegate_runs_on_the_caller_stream(self):
        from executorch.runtime import cuda, Runtime

        class MatmulChain(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.weight = torch.nn.Parameter(torch.randn(1024, 1024) / 32)

            def forward(self, value):
                value = value.repeat(64, 64)
                for _ in range(16):
                    value = torch.tanh(value @ self.weight)
                return value.sum()

        path, data = self._export_to_directory(
            "matmul_chain", MatmulChain().eval(), (torch.randn(16, 16),)
        )
        method = Runtime.get().load_program(path, data_path=data).load_method("forward")
        value = torch.randn(16, 16)
        stream = cuda.Stream()
        caller = torch.cuda.ExternalStream(stream.cuda_stream, device=self.device)

        def work_on_stream_ms(use_stream):
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            with torch.cuda.stream(caller):
                # Keeps the stream busy while the call is queued, so the two
                # events bracket only work queued on the stream in between.
                torch.cuda._sleep(100_000_000)
                start.record()
            if use_stream:
                with cuda.stream(stream):
                    method.execute([value])
            else:
                method.execute([value])
            with torch.cuda.stream(caller):
                end.record()
            end.synchronize()
            return start.elapsed_time(end)

        work_on_stream_ms(False)
        outside = work_on_stream_ms(False)
        inside = work_on_stream_ms(True)
        self.assertGreater(inside, 4 * outside + 0.5)

        foreign = torch.cuda.Stream(device=self.device)
        with cuda.stream(foreign):
            method.execute([value])
        # The legacy default stream, as 0 or cudaStreamLegacy, is refused.
        for default in (
            cuda.ExternalStream(0),
            cuda.ExternalStream(1),
            torch.cuda.default_stream(self.device),
        ):
            with self.assertRaises(ValueError):
                with cuda.stream(default):
                    pass
        with self.assertRaises(TypeError):
            with cuda.stream(stream.cuda_stream):
                pass
        for device in (1, "cuda:1"):
            with self.assertRaises(RuntimeError):
                with cuda.stream(cuda.ExternalStream(foreign.cuda_stream, device)):
                    pass

        class Protocol:  # a cuda.core-style stream: device.device_id
            def __init__(self, version, device_id):
                self.version = version
                self.device = types.SimpleNamespace(device_id=device_id)

            def __cuda_stream__(self):
                return (self.version, foreign.cuda_stream)

        with cuda.stream(Protocol(0, 0)):
            pass
        with self.assertRaises(RuntimeError):
            with cuda.stream(Protocol(0, 1)):
                pass
        with self.assertRaises(ValueError):
            with cuda.stream(Protocol(1, 0)):
                pass
        cupy_style = types.SimpleNamespace(
            device_id=1, __cuda_stream__=lambda: (0, foreign.cuda_stream)
        )
        with self.assertRaises(RuntimeError):
            with cuda.stream(cupy_style):
                pass
        # A Stream's handle selected through any wrapper keeps it open.
        for selected in (stream, cuda.ExternalStream(stream.cuda_stream)):
            with cuda.stream(selected), self.assertRaises(RuntimeError):
                stream.close()
        second = cuda.Stream()
        outer, inner = cuda.stream(stream), cuda.stream(second)

        def exit_if_active(scope):
            with contextlib.suppress(RuntimeError):
                scope.__exit__(None, None, None)

        outer.__enter__()
        self.addCleanup(exit_if_active, outer)
        inner.__enter__()
        self.addCleanup(exit_if_active, inner)
        with self.assertRaises(RuntimeError):
            outer.__exit__(None, None, None)
        with self.assertRaises(RuntimeError):
            stream.close()
        inner.__exit__(None, None, None)
        # The refused scope ended with the inner one, so it can be used again.
        with outer:
            pass
        stream.close()
        second.close()
        stream = cuda.Stream()
        with self.assertRaises(TypeError):
            copy.copy(stream)
        made_before_close = cuda.stream(stream)
        stream.close()
        with self.assertRaises(RuntimeError):
            with made_before_close:
                pass

    def test_stream_on_another_device_is_refused(self):
        if torch.cuda.device_count() < 2:
            self.skipTest("needs two CUDA devices")
        from executorch.runtime import cuda

        owned = cuda.Stream(device=1)
        self.addCleanup(owned.close)
        refused = [owned, cuda.ExternalStream(owned.cuda_stream)]
        # These wrappers report no device or the current one, so only CUDA can
        # tell that the stream is on device 1.
        if cuda._bindings()._CallerStreamScope.can_check_device():
            foreign = torch.cuda.Stream(device=1)
            refused += [
                cuda.ExternalStream(foreign.cuda_stream),
                torch.cuda.ExternalStream(foreign.cuda_stream),
            ]
        for selected in refused:
            with self.assertRaises(RuntimeError):
                with cuda.stream(selected):
                    pass

    @staticmethod
    def _close_after_teardown(stream):
        import threading

        # Native selections of ended threads end at their teardown.
        for _ in range(100):
            try:
                stream.close()
                return
            except RuntimeError:
                threading.Event().wait(0.05)
        stream.close()

    def test_stream_scopes_across_threads(self):
        import threading

        from executorch.runtime import cuda

        stream = cuda.Stream()
        shared = cuda.stream(stream)
        errors = []

        def enter_and_exit():
            try:
                for _ in range(200):
                    with shared:
                        pass
            except Exception as error:
                errors.append(error)

        # Switch threads often, so a race in the shared object shows up.
        interval = sys.getswitchinterval()
        sys.setswitchinterval(1e-6)
        try:
            workers = [threading.Thread(target=enter_and_exit) for _ in range(2)]
            for worker in workers:
                worker.start()
            for worker in workers:
                worker.join()
        finally:
            sys.setswitchinterval(interval)
        self.assertEqual(errors, [])

        # A thread that ends inside a block does not block later threads.
        ended = threading.Thread(target=shared.__enter__)
        ended.start()
        ended.join()
        later = threading.Thread(target=enter_and_exit)
        later.start()
        later.join()
        self.assertEqual(errors, [])
        # An exit from a thread that is not inside the block is refused.
        with self.assertRaises(RuntimeError):
            shared.__exit__(None, None, None)
        with shared:
            pass
        self._close_after_teardown(stream)

    def test_stream_scopes_on_threads_outside_threading(self):
        import _thread
        import threading

        from executorch.runtime import cuda

        # Threads started outside the threading module reuse ids, and their
        # slots must still be freed when they end inside a block.
        stream = cuda.Stream()
        shared = cuda.stream(stream)
        errors = []

        def enter_and_end(done):
            try:
                shared.__enter__()
            except Exception as error:
                errors.append(error)
            done.set()

        for _ in range(2):
            done = threading.Event()
            _thread.start_new_thread(enter_and_end, (done,))
            done.wait()
        self.assertEqual(errors, [])
        with shared:
            pass
        self._close_after_teardown(stream)

    def test_stream_scope_ends_on_its_own_thread(self):
        import threading

        from executorch.runtime import cuda

        stream = cuda.Stream()
        entered = threading.Event()
        leave = threading.Event()
        scope = cuda.stream(stream)

        def hold():
            with scope:
                entered.set()
                leave.wait()

        holder = threading.Thread(target=hold)
        holder.start()
        entered.wait()
        try:
            with self.assertRaises(RuntimeError):
                scope.__exit__(None, None, None)
            with self.assertRaises(RuntimeError):
                stream.close()
        finally:
            leave.set()
            holder.join()
        # A scope object dropped without an exit ends its block.
        dropped = cuda.stream(stream)
        dropped.__enter__()
        del dropped
        gc.collect()
        # The native scope keeps the same rules for callers that use it directly.
        native = cuda._bindings()._CallerStreamScope(stream._native)
        native.__enter__()
        try:
            with self.assertRaises(RuntimeError):
                native.__enter__()
            refused = []

            def exit_elsewhere():
                try:
                    native.__exit__(None, None, None)
                except RuntimeError:
                    refused.append(True)

            exiter = threading.Thread(target=exit_elsewhere)
            exiter.start()
            exiter.join()
            self.assertEqual(refused, [True])
        finally:
            native.__exit__(None, None, None)
        self._close_after_teardown(stream)

    def test_torch_free_caller_shares_one_stream_across_programs(self):
        self._require_libgomp_without_torch()
        programs = [
            self._export_to_directory(
                f"torch_free_{index}", module, (torch.ones(4, 8),)
            )
            for index, module in enumerate((torch.nn.ReLU(), torch.nn.Tanh()))
        ]
        child = """
import sys
import numpy as np
from executorch.runtime import Runtime, cuda

value = np.linspace(-2, 2, 32, dtype=np.float32).reshape(4, 8)
stream = cuda.Stream()
with cuda.stream(stream):
    outputs = [
        np.asarray(
            Runtime.get().load_program(path, data_path=data or None)
            .load_method("forward").execute([value])[0]
        )
        for path, data in zip(sys.argv[1::2], sys.argv[2::2])
    ]
# A delegate that would replay a CUDA graph refuses a caller stream, so the
# refusal shows that the selection reached both programs.
Runtime.get().backend_registry.set_option(
    "CudaBackend", {"enable_cuda_graph_for_method": "forward"}
)
graphed = [
    Runtime.get().load_program(path, data_path=data or None).load_method("forward")
    for path, data in zip(sys.argv[1::2], sys.argv[2::2])
]
with cuda.stream(stream):
    for method in graphed:
        try:
            method.execute([value])
        except RuntimeError:
            continue
        raise AssertionError("A delegate did not see the selected stream")
stream.close()
np.testing.assert_allclose(outputs[0], np.maximum(value, 0), atol=1e-6)
np.testing.assert_allclose(outputs[1], np.tanh(value), atol=1e-6)
assert "torch" not in sys.modules
"""
        arguments = [item or "" for program in programs for item in program]
        result = subprocess.run(
            [sys.executable, "-c", child, *arguments],
            env=dict(os.environ, EXECUTORCH_PYBINDINGS_TENSOR_OUTPUT="executorch"),
            capture_output=True,
            text=True,
            timeout=600,
        )
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)

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

    def test_result_handed_out_after_set_inputs_is_waited_for(self):
        # A result set as an input can still be handed to another library
        # before execute(). The library may then write it on its own stream, so
        # execute() must wait for that write.
        method = self._load_method()
        first = method(
            CudaDLPackOnly(
                torch.full(self.shape, 1.0, device=self.device),
                torch.cuda.current_stream(self.device),
            )
        )[0]
        method.set_inputs([first])
        producer_stream = torch.cuda.Stream(device=self.device)
        with torch.cuda.stream(producer_stream):
            shared = torch.from_dlpack(first)
            torch.cuda._sleep(50_000_000)
            shared.fill_(41.0)
        method.execute()
        self._assert_ready_value(method.get_outputs()[0], 84.0)

    def test_borrowed_result_written_by_another_library_is_waited_for(self):
        # Two clone_outputs=False reads share the same memory. Another library
        # writes it through one of them, so passing the other back in must
        # wait for that write.
        method = self._load_method()
        stream = torch.cuda.current_stream(self.device)
        # The reader's outputs are on the host, so only the wait for its input
        # can order the read. Its first run sets it up, allocates its outputs,
        # and waits for the whole device, so it runs once before the write.
        reader = self._load_method(
            self._export_delegate(
                "host_outputs", outputs_on_device=False, plan_outputs=True
            )
        )
        reader(torch.zeros(self.shape, device=self.device))
        value = torch.full(self.shape, 1.0, device=self.device)
        written = method(CudaDLPackOnly(value, stream), clone_outputs=False)[0]
        passed = method.get_outputs(clone_outputs=False)[0]
        writer = torch.cuda.Stream(self.device)
        with torch.cuda.stream(writer):
            shared = torch.from_dlpack(written)
            torch.cuda._sleep(1_000_000_000)
            shared.fill_(41.0)
        output = reader(passed)[0]
        writer.synchronize()
        torch.testing.assert_close(
            torch.from_dlpack(output), torch.full(self.shape, 84.0)
        )

    def test_device_inputs_with_unplanned_host_outputs(self):
        # The delegate writes a device tensor that the method then copies into
        # the caller's host output, so that device tensor must still be planned.
        method = self._load_method(
            self._export_delegate("unplanned_host_outputs", outputs_on_device=False)
        )
        output = method(torch.full(self.shape, 1.0, device=self.device))[0]
        torch.testing.assert_close(
            torch.from_dlpack(output), torch.full(self.shape, 4.0)
        )

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
        self.assertIsNot(method.get_outputs(clone_outputs=False)[0], first)

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
        with self.assertRaisesRegex(
            ValueError, "Input 0 of method 'forward': .*laid out differently"
        ):
            method(CudaDLPackOnly(value, stream))

    def test_dlpack_input_on_a_non_current_device(self):
        if torch.cuda.device_count() < 2:
            self.skipTest("needs two CUDA devices")
        # A fresh process, so no earlier CUDA work has made cuda:1 usable from
        # cuda:0. torch exports a CUDA tensor only while its device is current,
        # so this fails unless the bindings switch to the input's device.
        script = """
import sys
import torch
from executorch.extension.pybindings import portable_lib as runtime

class PlainProducer:
    def __init__(self, tensor):
        self.tensor = tensor
    def __dlpack_device__(self):
        return self.tensor.__dlpack_device__()
    def __dlpack__(self, stream=None):
        return self.tensor.__dlpack__(stream=stream)

with open(sys.argv[1], "rb") as source:
    data = source.read()
method = runtime._load_program_from_buffer(data).load_method("forward")
value = torch.arange(120, dtype=torch.float32, device="cuda:1").reshape(2, 3, 4, 5)
assert torch.cuda.current_device() == 0
output = method(PlainProducer(value))[0]
with torch.cuda.device(1):
    torch.testing.assert_close(torch.from_dlpack(output).cpu(), value.cpu())
print("Input on a non-current device was read")
"""
        with tempfile.TemporaryDirectory() as directory:
            path = os.path.join(directory, "identity.pte")
            with open(path, "wb") as program:
                program.write(self._device_identity(1, torch.ones(2, 3, 4, 5)))
            completed = subprocess.run(
                [sys.executable, "-c", script, path],
                capture_output=True,
                text=True,
                timeout=120,
            )
        self.assertEqual(
            completed.returncode,
            0,
            f"stdout:\n{completed.stdout}\nstderr:\n{completed.stderr}",
        )
        self.assertIn("Input on a non-current device was read", completed.stdout)

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
        # Once the result is dropped, the next run writes into its memory, so it
        # must wait for a read another library still has queued there, whether
        # its own input is on the device or on the host.
        count = torch.Size(self.shape).numel()

        def on_device(value):
            host = memoryview(array.array("f", [value]) * count).cast("B")
            return self.bindings.ExecuTorchResult.from_host(
                host.cast("f", self.shape), self.device.index
            )

        def on_host(value):
            return torch.full(self.shape, value)

        for host_inputs, make_input in ((False, on_device), (True, on_host)):
            with self.subTest(host_inputs=host_inputs):
                method = self._load_method(
                    self.host_input_program if host_inputs else None
                )
                first, second = make_input(1.0), make_input(5.0)
                torch.cuda.synchronize(self.device)
                result = method(first)[0]
                # Imported before the read is queued: an ExecuTorchResult waits
                # for the device in __dlpack__, which would finish the read.
                view = self._tensor(result)
                reader = torch.cuda.Stream(self.device)
                with torch.cuda.stream(reader):
                    torch.cuda._sleep(1_000_000_000)
                    snapshot = view.clone()
                del result, view
                gc.collect()
                method(second)
                reader.synchronize()
                torch.testing.assert_close(snapshot.cpu(), torch.full(self.shape, 4.0))

    def test_planned_output_memory_waits_for_pending_readers(self):
        # Each run writes a memory-planned output again, so it must wait for a
        # read another library still has queued there. The inputs are on the
        # host, so only the wait for the output's device can order it.
        program_path, data_path = self._export_delegate(
            "planned_outputs", inputs_on_device=False, plan_outputs=True
        )
        module = self.bindings._load_for_executorch_from_buffer(
            program_path.read_bytes(), data_path.read_bytes() if data_path else None
        )
        method = self._load_method((program_path, data_path))
        for name, run in (("method", method), ("module", module.forward)):
            with self.subTest(executor=name):
                result = run(torch.full(self.shape, 1.0), clone_outputs=False)[0]
                # Imported before the read is queued: an ExecuTorchResult waits
                # for the device in __dlpack__, which would finish the read.
                view = self._tensor(result)
                snapshot = torch.empty_like(view)
                reader = torch.cuda.Stream(self.device)
                with torch.cuda.stream(reader):
                    torch.cuda._sleep(1_000_000_000)
                    snapshot.copy_(view)
                run(torch.full(self.shape, 5.0), clone_outputs=False)
                reader.synchronize()
                torch.testing.assert_close(snapshot.cpu(), torch.full(self.shape, 4.0))

    def test_method_outputs_take_turns_between_two_buffers(self):
        method = self._load_method()
        value = torch.full(self.shape, 1.0, device=self.device)
        addresses = []
        out = None
        for _ in range(6):
            out = method(value)[0]
            addresses.append(self._tensor(out).data_ptr())
        self.assertEqual(len(set(addresses)), 2, addresses)
        self.assertEqual(addresses[0::2], [addresses[0]] * 3)
        self.assertEqual(addresses[1::2], [addresses[1]] * 3)
        self._assert_ready_value(out, 4.0)

    def test_converted_torch_input_waits_for_pending_writes(self):
        method = self._load_method()
        value = torch.zeros(self.shape, device=self.device)
        # Make the same copy once first. The first one loads its kernel and
        # allocates new memory, and both wait for the whole device, which
        # would hide a missing wait.
        value.t().contiguous()
        torch.cuda.synchronize(self.device)
        writer = torch.cuda.Stream(self.device)
        with torch.cuda.stream(writer):
            torch.cuda._sleep(1_000_000_000)
            value.fill_(1.0)
        # The transposed view is copied into the exported layout first, and
        # that copy must see the write still queued on the writer stream.
        output = method(value.t())[0]
        writer.synchronize()
        self._assert_ready_value(output, 4.0)

    def test_cloned_output_is_not_changed_by_later_writes(self):
        # Torch output mode clones through torch, which waits; this covers the
        # bindings' own copy.
        if not self.executorch_output:
            self.skipTest("clones through the bindings' own copy")
        data = self._device_identity(0, torch.ones(self.shape))
        method = self.bindings._load_program_from_buffer(data).load_method("forward")
        value = torch.full(self.shape, 3.0, device=self.device)
        method.set_inputs([value])
        method.execute()
        # Work queued ahead of the clone delays it, so a clone that returns
        # before it has finished would copy the write below.
        torch.cuda._sleep(1_000_000_000)
        output = method.get_outputs()[0]
        writer = torch.cuda.Stream(self.device)
        with torch.cuda.stream(writer):
            value.fill_(99.0)
        writer.synchronize()
        torch.cuda.synchronize(self.device)
        torch.testing.assert_close(
            self._tensor(output).cpu(), torch.full(self.shape, 3.0)
        )

    def test_to_host_waits_for_writes_by_another_library(self):
        if not self.executorch_output:
            self.skipTest("to_host() is an ExecuTorchResult method")
        method = self._load_method()
        result = method(torch.full(self.shape, 1.0, device=self.device))[0]
        shared = torch.from_dlpack(result)
        writer = torch.cuda.Stream(self.device)
        with torch.cuda.stream(writer):
            torch.cuda._sleep(1_000_000_000)
            shared.fill_(42.0)
        host = result.to_host()
        writer.synchronize()
        torch.testing.assert_close(
            torch.from_dlpack(host), torch.full(self.shape, 42.0)
        )

    def test_module_waits_for_inputs_kept_from_earlier_calls(self):
        program = self.program_path.read_bytes()
        data = self.data_path.read_bytes() if self.data_path else None
        module = self.bindings._load_for_executorch_from_buffer(program, data)
        value = torch.zeros(self.shape, device=self.device)
        stream = torch.cuda.current_stream(self.device)
        kept = module.forward([CudaDLPackOnly(value, stream)])
        writer = torch.cuda.Stream(self.device)
        with torch.cuda.stream(writer):
            torch.cuda._sleep(1_000_000_000)
            value.fill_(1.0)
        # plan_execute() reads the input kept from the call above.
        output = module.plan_execute("forward")[0]
        writer.synchronize()
        self._assert_ready_value(output, 4.0)
        del kept

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
        self.assertEqual(host.dtype, "bfloat16")
        # The host export of bfloat16 is unsigned integers, so passing the
        # result itself is what keeps its dtype.
        back = self.bindings.ExecuTorchResult.from_host(host, self.device.index)
        self.assertEqual(back.dtype, "bfloat16")
        self.assertEqual(back.shape, (2, 3))
        torch.testing.assert_close(torch.from_dlpack(back), value)
        with self.assertRaisesRegex(ValueError, "already on a device"):
            self.bindings.ExecuTorchResult.from_host(on_device, self.device.index)

    def _device_waits(self, body):
        from torch.profiler import profile, ProfilerActivity

        with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as trace:
            body()
        return sum(
            event.count
            for event in trace.key_averages()
            if event.key == "cudaDeviceSynchronize"
        )

    def test_dlpack_export_keywords(self):
        if not self.executorch_output:
            self.skipTest("__dlpack__ keywords are an ExecuTorchResult feature")
        method = self._load_method()
        result = method(torch.full(self.shape, 1.0, device=self.device))[0]
        # The profiler waits for the device itself when it stops.
        idle = self._device_waits(lambda: None)
        # A consumer that passes stream=-1 orders the work itself.
        self.assertEqual(self._device_waits(lambda: result.__dlpack__(stream=-1)), idle)
        self.assertEqual(
            self._device_waits(lambda: result.__dlpack__(stream=1)), idle + 1
        )
        copied = torch.from_dlpack(result.__dlpack__(max_version=(1, 0), copy=True))
        self.assertNotEqual(copied.data_ptr(), torch.from_dlpack(result).data_ptr())
        torch.testing.assert_close(copied.cpu(), torch.full(self.shape, 4.0))
        host = torch.from_dlpack(
            result.__dlpack__(max_version=(1, 0), dl_device=(1, 0), copy=True)
        )
        self.assertEqual(host.device, torch.device("cpu"))
        torch.testing.assert_close(host, torch.full(self.shape, 4.0))

    def test_device_chain_waits_for_the_device_twice_per_call(self):
        # Feeding each output into the next call, with a host upload per call.
        # Each call waits for the device twice: once before it runs, in case
        # another library still uses its input or output memory, and once after,
        # so its output is ready on any stream. The count is fixed per call, not
        # per tensor.
        from torch.profiler import profile, ProfilerActivity

        method = self._load_method()
        count = torch.Size(self.shape).numel()
        host = (
            memoryview(array.array("f", [3.0]) * count).cast("B").cast("f", self.shape)
        )
        calls = 8
        value = self.bindings.ExecuTorchResult.from_host(host, self.device.index)
        value = method(value, clone_outputs=True)[0]

        def waits(body):
            with profile(
                activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]
            ) as trace:
                body()
            return sum(
                event.count
                for event in trace.key_averages()
                if event.key == "cudaDeviceSynchronize"
            )

        def chain():
            nonlocal value
            for _ in range(calls):
                self.bindings.ExecuTorchResult.from_host(host, self.device.index)
                value = method(value, clone_outputs=True)[0]

        # The profiler waits for the device itself when it stops.
        self.assertEqual(waits(chain) - waits(lambda: None), 2 * calls)
        expected = 8.0
        for _ in range(calls):
            expected = (expected + 1.0) * 2.0
        final = value.to_host() if self.executorch_output else value.cpu()
        torch.testing.assert_close(
            torch.from_dlpack(final), torch.full(self.shape, expected)
        )

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
class OtherProducer:
    # Results of these bindings are read without asking a producer, so wrap one
    # to reach the path that asks.
    def __init__(self, source):
        self.source = source

    def __dlpack_device__(self):
        return self.source.__dlpack_device__()

    def __dlpack__(self, stream=None):
        return self.source.__dlpack__(stream=stream)

uploaded = runtime.ExecuTorchResult.from_host(np.full(shape, 3.0, np.float32), 0)
first = module.forward([OtherProducer(uploaded)])[0]
second = module.forward([OtherProducer(first)])[0]
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
