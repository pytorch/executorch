# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Tests for the MLX op test harness itself."""

import tempfile
import unittest
from pathlib import Path

import torch
from executorch.backends.mlx.test.run_all_tests import _run_test_in_subprocess
from executorch.backends.mlx.test.test_utils import (
    clean_test_outputs,
    print_mlx_graph_summary,
)
from executorch.exir import to_edge


class RunAllTestsTest(unittest.TestCase):
    def test_isolated_run_reports_result(self):
        self.addCleanup(clean_test_outputs, ["sub_10"])
        self.assertTrue(_run_test_in_subprocess("sub_10"))
        self.assertFalse(_run_test_in_subprocess("no_such_config"))

    def test_summary_of_program_without_mlx_delegate_returns(self):
        program = to_edge(
            torch.export.export(torch.nn.ReLU(), (torch.randn(4),))
        ).to_executorch()
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "model.pte"
            path.write_bytes(program.buffer)
            # A test that lowers to no MLX delegate fails on its delegate count
            # check after this, so printing must not end the process.
            print_mlx_graph_summary(path)


if __name__ == "__main__":
    unittest.main()
