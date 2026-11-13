# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import os
import unittest
from unittest import mock

import torch
from executorch.backends.cadence.runtime import executor, runtime
from executorch.devtools.bundled_program.serialize import (
    deserialize_from_flatbuffer_to_bundled_program,
)
from executorch.exir import ExecutorchProgramManager, to_edge


def _linear_program(out_features: int) -> ExecutorchProgramManager:
    return to_edge(
        torch.export.export(torch.nn.Linear(4, out_features), (torch.randn(2, 4),))
    ).to_executorch()


class RunBundledTest(unittest.TestCase):
    def _run_with_mocked_runner(
        self, program: ExecutorchProgramManager, inputs: tuple[torch.Tensor, ...]
    ) -> tuple[mock.MagicMock, dict[str, bytes]]:
        """Returns the mocked Executor class and the files it found staged."""
        staged_files: dict[str, bytes] = {}
        with mock.patch.object(
            runtime.os.path, "isfile", return_value=True
        ), mock.patch.object(runtime, "Executor") as executor_cls, mock.patch.object(
            runtime, "CadenceETDump"
        ) as etdump:

            def read_working_dir() -> None:
                working_dir = executor_cls.call_args.args[0]
                for name in os.listdir(working_dir):
                    with open(os.path.join(working_dir, name), "rb") as f:
                        staged_files[name] = f.read()

            executor_cls.return_value.side_effect = read_working_dir
            output = torch.zeros(1)
            etdump.return_value.get_outputs.return_value = [output]

            result = runtime.run_bundled(program, inputs)

        self.assertIs(result, output)
        return executor_cls, staged_files

    def test_stages_a_bundled_program_with_the_program_and_inputs(self) -> None:
        program = _linear_program(4)
        inputs = (torch.randn(2, 4),)

        _, staged_files = self._run_with_mocked_runner(program, inputs)

        self.assertEqual(list(staged_files), ["program.bpte"])
        bundled = deserialize_from_flatbuffer_to_bundled_program(
            staged_files["program.bpte"]
        )
        self.assertEqual(bundled.program, program.buffer)
        (suite,) = bundled.method_test_suites
        self.assertEqual(suite.method_name, "forward")
        (case,) = suite.test_cases
        (value,) = case.inputs
        self.assertEqual(bytes(value.val.data), inputs[0].numpy().tobytes())

    def test_keeps_the_runner_default_buffer_for_small_outputs(self) -> None:
        executor_cls, _ = self._run_with_mocked_runner(
            _linear_program(4), (torch.randn(2, 4),)
        )

        self.assertEqual(executor_cls.call_args.kwargs["debug_buffer_size"], 256 * 1024)

    def test_sizes_the_buffer_for_large_outputs(self) -> None:
        executor_cls, _ = self._run_with_mocked_runner(
            _linear_program(32768), (torch.randn(2, 4),)
        )

        # 2 x 32768 float32 values, starting at a 64-byte boundary.
        self.assertEqual(
            executor_cls.call_args.kwargs["debug_buffer_size"], 2 * 32768 * 4 + 64
        )

    def test_missing_runner_raises_without_building(self) -> None:
        with mock.patch.object(
            runtime.os.path, "isfile", return_value=False
        ), mock.patch.object(executor, "execute") as run:
            with self.assertRaises(FileNotFoundError):
                runtime.run_bundled(_linear_program(4), (torch.randn(2, 4),))

        run.assert_not_called()
