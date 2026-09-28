# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import importlib
import importlib.util
import os
import re
import sys
import unittest
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
    def test_import_does_not_load_torch(self) -> None:
        self.assertFalse(runtime._uses_aten)
        self.assertNotIn("torch", sys.modules)
        self.assertNotIn("executorch.exir", sys.modules)

    def test_executorch_result_is_not_constructible(self) -> None:
        with self.assertRaises(TypeError):
            runtime.ExecuTorchResult()

    def test_executes_program_without_torch(self) -> None:
        self.assertEqual(os.environ.get(TENSOR_OUTPUT_ENV), "executorch")
        self.assertEqual(runtime._tensor_output, "executorch")
        with open(os.environ["EXECUTORCH_PYBIND_TEST_PTE"], "rb") as program_file:
            program_data = program_file.read()
        module = runtime._load_for_executorch_from_buffer(program_data)

        output = module(
            (np.array([1.0], dtype=np.float32), np.array([2.0], dtype=np.float32))
        )[0]

        self.assertIsInstance(output, runtime.ExecuTorchResult)
        self.assertEqual(output.shape, (1,))
        self.assertEqual(output.dtype, np.dtype("float32"))
        np.testing.assert_array_equal(np.asarray(output), np.array([3.0]))

    def test_torch_output_error_mentions_executorch_opt_out(self) -> None:
        if importlib.util.find_spec("torch") is not None:
            self.skipTest("torch is installed")
        if runtime._tensor_output != "torch":
            self.skipTest("ExecuTorchResult output was explicitly selected")

        with open(os.environ["EXECUTORCH_PYBIND_TEST_PTE"], "rb") as program_file:
            program_data = program_file.read()
        module = runtime._load_for_executorch_from_buffer(program_data)

        with self.assertRaisesRegex(
            RuntimeError,
            rf"{TENSOR_OUTPUT_ENV}=executorch",
        ):
            module(
                (
                    np.array([1.0], dtype=np.float32),
                    np.array([2.0], dtype=np.float32),
                )
            )

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

        output = module(
            (
                DLPackOnly(np.array([1.0], dtype=np.float32)),
                DLPackOnly(np.array([2.0], dtype=np.float32)),
            )
        )[0]

        np.testing.assert_array_equal(np.from_dlpack(output), np.array([3.0]))

    def test_mixed_dlpack_and_buffer_inputs_are_rejected(self) -> None:
        with open(os.environ["EXECUTORCH_PYBIND_TEST_PTE"], "rb") as program_file:
            program_data = program_file.read()
        module = runtime._load_for_executorch_from_buffer(program_data)

        with self.assertRaisesRegex(TypeError, "cannot mix tensor input protocols"):
            module(
                (
                    DLPackOnly(np.array([1.0], dtype=np.float32)),
                    np.array([2.0], dtype=np.float32),
                )
            )

    @unittest.skipUnless(
        importlib.util.find_spec("torch") is not None, "torch is not installed"
    )
    def test_z_torch_tensor_uses_python_api(self) -> None:
        import torch

        with open(os.environ["EXECUTORCH_PYBIND_TEST_PTE"], "rb") as program_file:
            program_data = program_file.read()
        module = runtime._load_for_executorch_from_buffer(program_data)

        output = module((torch.tensor([1.0]), torch.tensor([2.0])))[0]

        self.assertIsInstance(output, torch.Tensor)
        np.testing.assert_array_equal(np.asarray(output), np.array([3.0]))
        torch.testing.assert_close(torch.from_dlpack(output), torch.tensor([3.0]))
        with self.assertRaisesRegex(ValueError, "must be resolved"):
            module((torch._neg_view(torch.tensor([1.0])), torch.tensor([2.0])))

    def test_process_has_no_aten_libraries(self) -> None:
        process_maps = Path("/proc/self/maps")
        if not process_maps.exists():
            self.skipTest("mapped-library inspection is only available on Linux")

        self.assertIsNone(
            re.search(r"lib(?:torch|aten|c10)", process_maps.read_text().lower())
        )


if __name__ == "__main__":
    unittest.main()
