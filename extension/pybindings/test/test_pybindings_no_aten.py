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


class PybindingsNoAtenTest(unittest.TestCase):
    def test_import_does_not_load_torch(self) -> None:
        self.assertFalse(runtime._uses_aten)
        self.assertNotIn("torch", sys.modules)
        self.assertNotIn("executorch.exir", sys.modules)

    def test_result_memory_is_not_constructible(self) -> None:
        with self.assertRaises(TypeError):
            runtime.ResultMemory()

    def test_executes_program_without_torch(self) -> None:
        with open(os.environ["EXECUTORCH_PYBIND_TEST_PTE"], "rb") as program_file:
            program_data = program_file.read()
        module = runtime._load_for_executorch_from_buffer(program_data)

        output = module(
            (np.array([1.0], dtype=np.float32), np.array([2.0], dtype=np.float32))
        )[0]

        self.assertIsInstance(output, runtime.ResultMemory)
        self.assertEqual(output.shape, (1,))
        self.assertEqual(output.dtype, np.dtype("float32"))
        np.testing.assert_array_equal(np.asarray(output), np.array([3.0]))

    def test_z_torch_tensor_uses_python_api(self) -> None:
        import torch

        with open(os.environ["EXECUTORCH_PYBIND_TEST_PTE"], "rb") as program_file:
            program_data = program_file.read()
        module = runtime._load_for_executorch_from_buffer(program_data)

        output = module((torch.tensor([1.0]), torch.tensor([2.0])))[0]

        np.testing.assert_array_equal(np.asarray(output), np.array([3.0]))
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
