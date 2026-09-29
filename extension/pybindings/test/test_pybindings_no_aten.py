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
                raw = memoryview(output)
                self.assertEqual(raw.format, "B")
                self.assertEqual(raw.shape, (8,))
                self.assertEqual(raw.tobytes(), bytes((0x00, 0x40)) * 4)
                self.assertEqual(output.shape, (2, 2))
                self.assertEqual(output.__dlpack_device__(), (1, 0))
                if importlib.util.find_spec("torch") is not None:
                    import torch

                    tensor = torch.from_dlpack(output)
                    self.assertEqual(tensor.dtype, torch.bfloat16)
                    torch.testing.assert_close(
                        tensor, torch.full((2, 2), 2.0, dtype=torch.bfloat16)
                    )

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
