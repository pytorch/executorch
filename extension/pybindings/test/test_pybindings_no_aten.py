# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import gc
import importlib
import importlib.util
import os
import subprocess
import sys
import textwrap
import unittest
import weakref
from pathlib import Path

import numpy as np

if importlib.util.find_spec("_C") is None:
    raise unittest.SkipTest("the standalone no-ATen extension is not built")

runtime = importlib.import_module("_C")
TENSOR_OUTPUT_ENV = "EXECUTORCH_PYBINDINGS_TENSOR_OUTPUT"


class DLPackOnly:
    def __init__(self, array: np.ndarray) -> None:
        self.array = array

    def __dlpack__(self, stream=None):
        return self.array.__dlpack__(stream=stream)

    def __dlpack_device__(self):
        return self.array.__dlpack_device__()


class PybindingsNoAtenTest(unittest.TestCase):
    def _run_in_fresh_process(self, script: str) -> None:
        runtime_path = Path(runtime.__file__).resolve()
        environment = os.environ.copy()
        environment[TENSOR_OUTPUT_ENV] = "executorch"
        environment["PYTHONPATH"] = os.pathsep.join(
            filter(
                None,
                (str(runtime_path.parent), environment.get("PYTHONPATH")),
            )
        )
        prelude = """
import sys
from pathlib import Path

import _C as runtime

assert Path(runtime.__file__).resolve() == Path(sys.argv[1]).resolve()
assert not runtime._uses_aten
assert runtime._tensor_output == "executorch"
"""
        completed = subprocess.run(
            [
                sys.executable,
                "-c",
                prelude + textwrap.dedent(script),
                str(runtime_path),
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

    def test_import_does_not_load_torch(self) -> None:
        self._run_in_fresh_process(
            """
            assert "torch" not in sys.modules
            assert "executorch.exir" not in sys.modules
            """
        )

    def test_executorch_result_is_not_constructible(self) -> None:
        with self.assertRaises(TypeError):
            runtime.ExecuTorchResult()

    def test_executes_program_without_torch(self) -> None:
        self.assertEqual(os.environ.get(TENSOR_OUTPUT_ENV), "executorch")
        self.assertEqual(runtime._tensor_output, "executorch")
        self._run_in_fresh_process(
            """
            import os

            import numpy as np

            assert "torch" not in sys.modules
            with open(os.environ["EXECUTORCH_PYBIND_TEST_PTE"], "rb") as program_file:
                program_data = program_file.read()
            module = runtime._load_for_executorch_from_buffer(program_data)
            output = module(
                (np.array([1.0], dtype=np.float32), np.array([2.0], dtype=np.float32))
            )[0]

            assert isinstance(output, runtime.ExecuTorchResult)
            assert output.shape == (1,)
            assert output.dtype == np.dtype("float32")
            np.testing.assert_array_equal(np.asarray(output), np.array([3.0]))
            assert "torch" not in sys.modules
            assert "executorch.exir" not in sys.modules
            """
        )

    @unittest.skipIf(
        importlib.util.find_spec("torch") is not None, "torch is installed"
    )
    def test_output_policy_without_torch(self) -> None:
        runtime_path = Path(runtime.__file__).resolve()
        script = """
import sys
from pathlib import Path

if sys.argv[2] == "torch":
    try:
        import _C
    except ImportError:
        pass
    else:
        raise AssertionError("An explicit Torch output request must fail without Torch")
else:
    import _C as runtime
    import numpy as np
    import os

    assert Path(runtime.__file__).resolve() == Path(sys.argv[1]).resolve()
    assert runtime._tensor_output == "executorch"
    with open(os.environ["EXECUTORCH_PYBIND_TEST_PTE"], "rb") as source:
        data = source.read()
    module = runtime._load_for_executorch_from_buffer(data)
    output = module((np.array([1.0], dtype=np.float32), np.array([2.0], dtype=np.float32)))[0]
    assert isinstance(output, runtime.ExecuTorchResult)
    np.testing.assert_array_equal(np.asarray(output), [3.0])
    assert "torch" not in sys.modules
"""
        for policy in (None, "torch"):
            with self.subTest(policy=policy):
                environment = os.environ.copy()
                environment.pop(TENSOR_OUTPUT_ENV, None)
                if policy is not None:
                    environment[TENSOR_OUTPUT_ENV] = policy
                environment["PYTHONPATH"] = os.pathsep.join(
                    filter(
                        None, (str(runtime_path.parent), environment.get("PYTHONPATH"))
                    )
                )
                completed = subprocess.run(
                    [sys.executable, "-c", script, str(runtime_path), str(policy)],
                    env=environment,
                    capture_output=True,
                    text=True,
                    timeout=60,
                )
                self.assertEqual(
                    completed.returncode, 0, completed.stdout + completed.stderr
                )

    def test_executorch_result_array_copy_contract(self) -> None:
        if runtime._tensor_output != "executorch":
            self.skipTest("ExecuTorchResult output was not selected")
        with open(os.environ["EXECUTORCH_PYBIND_TEST_PTE"], "rb") as program_file:
            program_data = program_file.read()
        module = runtime._load_for_executorch_from_buffer(program_data)
        output = module(
            (np.array([1.0], dtype=np.float32), np.array([2.0], dtype=np.float32))
        )[0]
        view = np.asarray(output)
        for copy in (None, False):
            for dtype in (None, np.dtype("float32")):
                with self.subTest(copy=copy, dtype=dtype):
                    array = output.__array__(dtype=dtype, copy=copy)
                    self.assertTrue(np.shares_memory(array, view))
        copied = output.__array__(copy=True)
        self.assertFalse(np.shares_memory(copied, view))
        copied[0] = 99.0
        np.testing.assert_array_equal(view, np.array([3.0], dtype=np.float32))
        for copy in (None, True):
            converted = output.__array__(dtype=np.float64, copy=copy)
            self.assertEqual(converted.dtype, np.dtype("float64"))
            self.assertFalse(np.shares_memory(converted, view))
        with self.assertRaisesRegex(ValueError, "requires a copy"):
            output.__array__(dtype=np.float64, copy=False)

    def test_executorch_result_to_host_and_repr(self) -> None:
        if runtime._tensor_output != "executorch":
            self.skipTest("ExecuTorchResult output was not selected")
        with open(os.environ["EXECUTORCH_PYBIND_TEST_PTE"], "rb") as program_file:
            program_data = program_file.read()
        module = runtime._load_for_executorch_from_buffer(program_data)
        output = module(
            (np.array([1.0], dtype=np.float32), np.array([2.0], dtype=np.float32))
        )[0]

        copy = output.to_host()

        self.assertIsInstance(output.dtype, str)
        self.assertEqual(output.dtype, "float32")
        self.assertEqual(
            repr(copy), "ExecuTorchResult(shape=(1,), dtype=float32, device=cpu)"
        )
        self.assertFalse(np.shares_memory(np.asarray(copy), np.asarray(output)))
        np.testing.assert_array_equal(np.asarray(copy), np.array([3.0]))

    def test_executorch_result_from_host_requires_cuda(self) -> None:
        # A build with CUDA support uploads instead; the CUDA suite covers it.
        if "CudaBackend" in runtime._get_registered_backend_names():
            self.skipTest("bindings are built with CUDA support")
        for shape in ((4,), (0,)):
            with self.subTest(shape=shape):
                with self.assertRaisesRegex(RuntimeError, "CUDA support"):
                    runtime.ExecuTorchResult.from_host(np.ones(shape, dtype=np.float32))

    def test_borrowed_result_does_not_create_method_cycle(self) -> None:
        if runtime._tensor_output != "executorch":
            self.skipTest("ExecuTorchResult output was not selected")
        with open(os.environ["EXECUTORCH_PYBIND_TEST_PTE"], "rb") as program_file:
            program_data = program_file.read()
        method = runtime._load_program_from_buffer(program_data).load_method("forward")
        output = method(
            (np.array([1.0], dtype=np.float32), np.array([2.0], dtype=np.float32)),
            clone_outputs=False,
        )[0]
        method.set_inputs((output, output))
        method_ref = weakref.ref(method)
        del method
        gc.collect()
        self.assertIsNone(method_ref())
        np.testing.assert_array_equal(np.asarray(output), np.array([3.0]))

    @unittest.skipUnless(
        importlib.util.find_spec("torch") is not None, "torch is not installed"
    )
    def test_executorch_output_is_sticky_after_torch_import(self) -> None:
        if runtime._tensor_output != "executorch":
            self.skipTest("ExecuTorchResult output was not selected")

        import torch  # noqa: F401

        with open(os.environ["EXECUTORCH_PYBIND_TEST_PTE"], "rb") as program_file:
            program_data = program_file.read()
        module = runtime._load_for_executorch_from_buffer(program_data)

        output = module(
            (np.array([1.0], dtype=np.float32), np.array([2.0], dtype=np.float32))
        )[0]

        self.assertIsInstance(output, runtime.ExecuTorchResult)

    def test_dlpack_input_and_output(self) -> None:
        with open(os.environ["EXECUTORCH_PYBIND_TEST_PTE"], "rb") as program_file:
            program_data = program_file.read()
        module = runtime._load_for_executorch_from_buffer(program_data)
        method = runtime._load_program_from_buffer(program_data).load_method("forward")
        for executor in (module, method):
            with self.subTest(executor=type(executor)):
                output = executor(
                    (
                        DLPackOnly(np.array([1.0], dtype=np.float32)),
                        DLPackOnly(np.array([2.0], dtype=np.float32)),
                    )
                )[0]
                self.assertEqual(output.__dlpack_device__(), (1, 0))
                np.testing.assert_array_equal(np.from_dlpack(output), np.array([3.0]))

    def test_dlpack_export_keywords(self) -> None:
        with open(os.environ["EXECUTORCH_PYBIND_TEST_PTE"], "rb") as program_file:
            program_data = program_file.read()
        method = runtime._load_program_from_buffer(program_data).load_method("forward")
        values = np.array([1.0], dtype=np.float32)
        result = method((values, values))[0]
        if not isinstance(result, runtime.ExecuTorchResult):
            self.skipTest("torch output mode returns torch tensors")

        self.assertEqual(result.device, "cpu")
        shared = np.from_dlpack(result, copy=False)
        self.assertTrue(np.shares_memory(shared, np.from_dlpack(result)))
        copied = np.from_dlpack(result, copy=True)
        self.assertFalse(np.shares_memory(copied, shared))
        np.testing.assert_array_equal(copied, shared)
        np.testing.assert_array_equal(np.from_dlpack(result, device="cpu"), shared)
        self.assertIn(
            '"dltensor_versioned"', repr(result.__dlpack__(max_version=(1, 0)))
        )
        self.assertIn('"dltensor"', repr(result.__dlpack__()))
        with self.assertRaises(BufferError):
            result.__dlpack__(dl_device=(2, 0), copy=False)
        with self.assertRaises(TypeError):
            result.__dlpack__(stream="default")

    def test_raw_dlpack_capsule_inputs(self) -> None:
        with open(os.environ["EXECUTORCH_PYBIND_TEST_PTE"], "rb") as program_file:
            program_data = program_file.read()
        module = runtime._load_for_executorch_from_buffer(program_data)
        method = runtime._load_program_from_buffer(program_data).load_method("forward")
        first = np.array([1.0], dtype=np.float32)
        second = np.array([2.0], dtype=np.float32)
        read_only = second.copy()
        read_only.flags.writeable = False
        for executor in (module, method):
            for capsules in (
                (first.__dlpack__(), second.__dlpack__()),
                (
                    first.__dlpack__(max_version=(1, 0)),
                    read_only.__dlpack__(max_version=(1, 0)),
                ),
            ):
                with self.subTest(executor=type(executor), capsules=capsules):
                    output = executor(capsules)[0]
                    np.testing.assert_array_equal(np.from_dlpack(output), [3.0])

        # A producer that knows DLPack 1.0 is asked for a versioned capsule.
        class VersionedProducer:
            def __init__(self, array: np.ndarray) -> None:
                self.array = array
                self.requests = []

            def __dlpack_device__(self):
                return self.array.__dlpack_device__()

            def __dlpack__(self, stream=None, max_version=None):
                self.requests.append(max_version)
                return self.array.__dlpack__(stream=stream, max_version=max_version)

        producer = VersionedProducer(read_only)
        np.testing.assert_array_equal(
            np.from_dlpack(method((first, producer))[0]), [3.0]
        )
        self.assertEqual(producer.requests, [(1, 0)])
        # A capsule gives its tensor away once.
        used = first.__dlpack__()
        method((used, second))
        with self.assertRaisesRegex(ValueError, "already consumed"):
            method((used, second))

    def test_bfloat16_raw_buffer_input_and_output(self) -> None:
        with open(
            os.environ["EXECUTORCH_PYBIND_TEST_BFLOAT16_PTE"], "rb"
        ) as program_file:
            program_data = program_file.read()
        module = runtime._load_for_executorch_from_buffer(program_data)
        method = runtime._load_program_from_buffer(program_data).load_method("forward")
        one = bytes((0x80, 0x3F)) * 4

        for executor in (module, method):
            with self.subTest(executor=type(executor)):
                output = executor((one, one))[0]
                # NumPy has no bfloat16, so the buffer shows unsigned integers
                # of the same width in the real shape.
                raw = memoryview(output)
                self.assertEqual(raw.format, "H")
                self.assertEqual(raw.shape, (2, 2))
                self.assertEqual(raw.tobytes(), bytes((0x00, 0x40)) * 4)
                self.assertEqual(output.shape, (2, 2))
                self.assertEqual(output.__dlpack_device__(), (1, 0))
                if isinstance(output, runtime.ExecuTorchResult):
                    self.assertEqual(output.dtype, "bfloat16")
                    # NumPy reads the buffer first, so call __array__ directly.
                    self.assertEqual(output.__array__().dtype, np.uint16)
                for noncontiguous in (
                    np.arange(16, dtype=np.uint8)[::2],
                    np.arange(8, dtype=np.uint8)[::-1],
                ):
                    with self.assertRaisesRegex(BufferError, "contiguous byte buffers"):
                        executor((noncontiguous, one))
                if importlib.util.find_spec("torch") is not None:
                    import torch

                    tensor = torch.from_dlpack(output)
                    self.assertEqual(tensor.dtype, torch.bfloat16)
                    torch.testing.assert_close(
                        tensor, torch.full((2, 2), 2.0, dtype=torch.bfloat16)
                    )

    def test_bfloat16_result_in_another_layout_feeds_back_correctly(self) -> None:
        # The method takes contiguous inputs and returns a channels-last result.
        # Fed back, it must be read in its own layout, or its values come back
        # permuted.
        if runtime._tensor_output != "executorch":
            self.skipTest("ExecuTorchResult output was not selected")
        with open(
            os.environ["EXECUTORCH_PYBIND_TEST_BFLOAT16_CHANNELS_LAST_PTE"], "rb"
        ) as program_file:
            program_data = program_file.read()
        method = runtime._load_program_from_buffer(program_data).load_method("forward")
        # bfloat16 is the upper half of float32, so these are exactly 0 to 7.
        values = (np.arange(8, dtype=np.float32).view(np.uint32) >> 16).astype(
            np.uint16
        )
        zeros = bytes(values.nbytes)

        result = method((values.tobytes(), zeros))[0]
        self.assertEqual(result.strides, (16, 2, 8, 4))
        expected = values.reshape(1, 2, 2, 2)
        np.testing.assert_array_equal(np.asarray(memoryview(result)), expected)
        fed_back = method((result, zeros))[0]
        np.testing.assert_array_equal(np.asarray(memoryview(fed_back)), expected)

    def test_input_errors_name_the_input(self) -> None:
        class FailingProducer:
            def __dlpack_device__(self):
                return (1, 0)

            def __dlpack__(self, stream=None):
                raise UnicodeDecodeError("utf-8", b"\xff", 0, 1, "invalid start byte")

        class FailingDevice(np.ndarray):
            def __dlpack_device__(self):
                raise UnicodeDecodeError("utf-8", b"\xff", 0, 1, "invalid start byte")

        with open(os.environ["EXECUTORCH_PYBIND_TEST_PTE"], "rb") as program_file:
            program_data = program_file.read()
        module = runtime._load_for_executorch_from_buffer(program_data)
        method = runtime._load_program_from_buffer(program_data).load_method("forward")
        value = np.array([1.0], dtype=np.float32)
        for executor in (module, method):
            # A Python error keeps its type, even one whose constructor takes
            # more than a message, and carries the name as a note. Choosing how
            # to read an input can raise too.
            for failing in (FailingProducer(), value.view(FailingDevice)):
                with self.subTest(executor=type(executor), input=type(failing)):
                    with self.assertRaises(UnicodeDecodeError) as caught:
                        executor((failing, value))
                    self.assertIn(
                        "Input 0 of method 'forward'",
                        getattr(caught.exception, "__notes__", []),
                    )
            with self.subTest(executor=type(executor)):
                with self.assertRaisesRegex(
                    RuntimeError,
                    "^Input 0 of method 'forward': unsupported python type",
                ):
                    executor((object(), value))
        # The count is checked before any input is read.
        with self.assertRaisesRegex(
            RuntimeError, r"^Method 'forward' takes 2 inputs, but got 3$"
        ):
            method((value, value, object()))

    def test_mixed_dlpack_and_buffer_inputs(self) -> None:
        with open(os.environ["EXECUTORCH_PYBIND_TEST_PTE"], "rb") as program_file:
            program_data = program_file.read()
        module = runtime._load_for_executorch_from_buffer(program_data)
        method = runtime._load_program_from_buffer(program_data).load_method("forward")

        for executor in (module, method):
            with self.subTest(executor=type(executor)):
                output = executor(
                    (
                        DLPackOnly(np.array([1.0], dtype=np.float32)),
                        np.array([2.0], dtype=np.float32),
                    )
                )[0]
                np.testing.assert_array_equal(np.asarray(output), np.array([3.0]))

    @unittest.skipUnless(
        importlib.util.find_spec("torch") is not None, "torch is not installed"
    )
    def test_z_torch_tensor_uses_python_api(self) -> None:
        import torch

        with open(os.environ["EXECUTORCH_PYBIND_TEST_PTE"], "rb") as program_file:
            program_data = program_file.read()
        module = runtime._load_for_executorch_from_buffer(program_data)

        output = module((torch.tensor([1.0]), torch.tensor([2.0])))[0]

        expected_type = (
            torch.Tensor
            if runtime._tensor_output == "torch"
            else runtime.ExecuTorchResult
        )
        self.assertIsInstance(output, expected_type)
        np.testing.assert_array_equal(np.asarray(output), np.array([3.0]))
        torch.testing.assert_close(torch.from_dlpack(output), torch.tensor([3.0]))
        with self.assertRaisesRegex(ValueError, "must be resolved"):
            module((torch._neg_view(torch.tensor([1.0])), torch.tensor([2.0])))

    def test_process_has_no_aten_libraries(self) -> None:
        process_maps = Path("/proc/self/maps")
        if not process_maps.exists():
            self.skipTest("mapped-library inspection is only available on Linux")

        self._run_in_fresh_process(
            """
            import re

            assert re.search(
                r"lib(?:torch|aten|c10)", Path("/proc/self/maps").read_text().lower()
            ) is None
            """
        )


if __name__ == "__main__":
    unittest.main()
