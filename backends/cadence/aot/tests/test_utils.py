# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-unsafe

import os
import tempfile
import unittest
from unittest import mock

from executorch.backends.cadence.aot.utils import save_pte_program
from executorch.exir import ExecutorchProgramManager


class TestSavePteProgram(unittest.TestCase):
    def test_saves_through_the_program_manager(self) -> None:
        prog = mock.create_autospec(ExecutorchProgramManager, instance=True)
        save_pte_program(prog, "model", "out")
        prog.save.assert_called_once_with(os.path.join("out", "model.pte"))

    def test_raises_when_the_write_fails(self) -> None:
        prog = mock.create_autospec(ExecutorchProgramManager, instance=True)
        prog.save.side_effect = OSError("disk full")
        with self.assertRaises(OSError):
            save_pte_program(prog, "model.pte")

    def test_still_writes_a_legacy_program(self) -> None:
        class Legacy:  # Like the deprecated ExecutorchProgram, which has no save.
            def write_to_file(self, f) -> None:
                f.write(b"program")

        with tempfile.TemporaryDirectory() as directory:
            save_pte_program(Legacy(), "model", directory)
            with open(os.path.join(directory, "model.pte"), "rb") as f:
                self.assertEqual(b"program", f.read())
