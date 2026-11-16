# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import gc
import os
import tempfile
import unittest

import torch

from executorch.backends.qualcomm.serialization.qc_schema import QcomChipset
from executorch.backends.qualcomm.utils.utils import (
    generate_htp_compiler_spec,
    generate_qnn_executorch_compiler_spec,
    get_qnn_context_binary_alignment,
    to_edge_transform_and_lower_to_qnn,
)
from executorch.exir import ExecutorchBackendConfig
from executorch.extension.pybindings.portable_lib import (
    _load_for_executorch,
    _load_program,
)


class ScaleShift(torch.nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * 2.0 + 1.0


class TestQnnSharedDelegate(unittest.TestCase):
    """Modules that load the same program must each keep running, whichever of
    them is destroyed first."""

    use_multi_contexts = False

    @classmethod
    def setUpClass(cls) -> None:
        cls.sample_input = torch.randn(1, 3, 8, 8)
        model = ScaleShift()
        cls.expected = model(cls.sample_input)
        backend_options = generate_htp_compiler_spec(
            use_fp16=True, use_multi_contexts=cls.use_multi_contexts
        )
        if cls.use_multi_contexts:
            backend_options.htp_options.max_sf_buf_size = 1024 * 1024
        compiler_specs = generate_qnn_executorch_compiler_spec(
            soc_model=QcomChipset.SM8650,
            backend_options=backend_options,
        )
        program = to_edge_transform_and_lower_to_qnn(
            {"forward": model, "shift": model},
            {
                "forward": (cls.sample_input,),
                "shift": (cls.sample_input,),
            },
            {"forward": compiler_specs, "shift": compiler_specs},
        ).to_executorch(
            ExecutorchBackendConfig(
                segment_alignment=get_qnn_context_binary_alignment()
            )
        )
        cls.tmp_dir = tempfile.TemporaryDirectory()
        cls.pte_path = os.path.join(cls.tmp_dir.name, "scale_shift.pte")
        with open(cls.pte_path, "wb") as f:
            f.write(program.buffer)

    @classmethod
    def tearDownClass(cls) -> None:
        cls.tmp_dir.cleanup()

    def assert_forward_matches(self, module) -> None:
        (output,) = module.forward([self.sample_input])
        torch.testing.assert_close(output, self.expected, atol=1e-1, rtol=1e-1)

    def load_two_modules(self):
        first = _load_for_executorch(self.pte_path)
        self.assert_forward_matches(first)
        second = _load_for_executorch(self.pte_path)
        self.assert_forward_matches(second)
        return first, second

    def test_methods_in_one_module_execute(self) -> None:
        module = _load_for_executorch(self.pte_path)
        self.assert_forward_matches(module)
        (output,) = module.run_method("shift", [self.sample_input])
        torch.testing.assert_close(output, self.expected, atol=1e-1, rtol=1e-1)

    def test_second_module_runs_after_first_is_destroyed(self) -> None:
        first, second = self.load_two_modules()
        del first
        gc.collect()

        for _ in range(3):
            self.assert_forward_matches(second)

    def test_first_module_runs_after_second_is_destroyed(self) -> None:
        first, second = self.load_two_modules()
        del second
        gc.collect()

        for _ in range(3):
            self.assert_forward_matches(first)

    def test_new_module_runs_after_previous_ones_are_destroyed(self) -> None:
        first, second = self.load_two_modules()
        del first, second
        gc.collect()

        self.assert_forward_matches(_load_for_executorch(self.pte_path))

    def test_method_survives_sibling_unload_and_reload(self) -> None:
        for first_name, second_name in (("forward", "shift"), ("shift", "forward")):
            with self.subTest(first=first_name):
                program = _load_program(self.pte_path)
                first = program.load_method(first_name)
                second = program.load_method(second_name)
                del first
                gc.collect()
                (output,) = second([self.sample_input])
                torch.testing.assert_close(output, self.expected, atol=1e-1, rtol=1e-1)
                first = program.load_method(first_name)
                del second
                gc.collect()
                (output,) = first([self.sample_input])
                torch.testing.assert_close(output, self.expected, atol=1e-1, rtol=1e-1)
                del first
                gc.collect()
                reloaded = program.load_method(second_name)
                (output,) = reloaded([self.sample_input])
                torch.testing.assert_close(output, self.expected, atol=1e-1, rtol=1e-1)
                del reloaded, program


class TestQnnSpillFillSharedDelegate(TestQnnSharedDelegate):
    use_multi_contexts = True


if __name__ == "__main__":
    unittest.main()
