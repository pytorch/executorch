# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Runs a CUDA-delegated program through executorch_pybindings(use_cuda = True).

LibTorch defines the same aoti_torch_* symbols as the ExecuTorch CUDA shims, so
cuda_lib cannot link into a binary with LibTorch. This test exports with
LibTorch and runs the program in a separate LibTorch-free binary.
"""

import os
import subprocess
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch
from executorch.backends.cuda.cuda_backend import CudaBackend
from executorch.backends.cuda.cuda_partitioner import CudaPartitioner
from executorch.exir import EdgeCompileConfig, to_edge_transform_and_lower


class ScaleShiftReLU(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.scale = torch.nn.Parameter(torch.randn(8))
        self.shift = torch.nn.Parameter(torch.randn(8))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.relu(x * self.scale + self.shift)


class CudaLibTest(unittest.TestCase):
    def test_runs_a_cuda_delegated_program_with_external_weights(self) -> None:
        self.assertTrue(torch.cuda.is_available(), "requires an NVIDIA GPU")
        torch.manual_seed(0)
        model = ScaleShiftReLU().eval()
        x = torch.randn(2, 8)
        with torch.no_grad():
            expected = model(x).numpy()
            program = to_edge_transform_and_lower(
                torch.export.export(model, (x,)),
                partitioner=[
                    CudaPartitioner(
                        [CudaBackend.generate_method_name_compile_spec("forward")]
                    )
                ],
                compile_config=EdgeCompileConfig(_check_ir_validity=False),
            ).to_executorch()
        delegates = program.executorch_program.execution_plan[0].delegates
        self.assertEqual([d.id for d in delegates], ["CudaBackend"])

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "model.pte").write_bytes(program.buffer)
            program.write_tensor_data_to_file(directory)
            np.save(root / "input.npy", x.numpy())
            subprocess.run([os.environ["RUN_CUDA_PROGRAM"], directory], check=True)
            actual = np.load(root / "output.npy")

        np.testing.assert_allclose(actual, expected, rtol=1e-5, atol=1e-6)
