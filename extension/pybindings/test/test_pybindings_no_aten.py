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

    def test_tensor_from_numpy_and_list(self) -> None:
        array = np.arange(6, dtype=np.float32).reshape(2, 3)
        np.testing.assert_array_equal(runtime.Tensor(array).numpy(), array)
        np.testing.assert_array_equal(
            runtime.Tensor([[1, 2], [3, 4]], dtype=np.int32).numpy(),
            np.array([[1, 2], [3, 4]], dtype=np.int32),
        )

    def test_executes_program_without_torch(self) -> None:
        with open(os.environ["EXECUTORCH_PYBIND_TEST_PTE"], "rb") as program_file:
            program_data = program_file.read()
        module = runtime._load_for_executorch_from_buffer(program_data)

        output = module(
            (
                runtime.Tensor([1.0], dtype=np.float32),
                runtime.Tensor([2.0], dtype=np.float32),
            )
        )[0]

        np.testing.assert_array_equal(output.numpy(), np.array([3.0]))

    def test_process_has_no_aten_libraries(self) -> None:
        process_maps = Path("/proc/self/maps")
        if not process_maps.exists():
            self.skipTest("mapped-library inspection is only available on Linux")

        self.assertIsNone(
            re.search(r"lib(?:torch|aten|c10)", process_maps.read_text().lower())
        )

    def test_missing_device_allocator(self) -> None:
        with self.assertRaisesRegex(RuntimeError, "No allocator is registered"):
            runtime.Tensor([1, 2], device=runtime.Device(runtime.DeviceType.CUDA))


if __name__ == "__main__":
    unittest.main()
