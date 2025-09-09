# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-unsafe

import argparse
import os

import torch
from executorch.exir import ExecutorchBackendConfig, to_edge

from torch.export import ExportedProgram
from torch.export.pt2_archive._package import package_pt2


class ModuleLinear(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(3, 3)

    def forward(self, x: torch.Tensor):
        return self.linear(x)

    def get_random_inputs(self):
        return (torch.randn(3),)


def main() -> None:
    torch.manual_seed(0)
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--outdir",
        type=str,
        required=True,
        help="Path to the directory to write model.pt2 files to",
    )
    args = parser.parse_args()

    m = ModuleLinear()
    sample_inputs = m.get_random_inputs()
    ep = torch.export.export(m, sample_inputs)

    # Lower to ExecuTorch
    exec_prog = to_edge(ep).to_executorch(
        ExecutorchBackendConfig(external_constants=True)
    )

    if not isinstance(ep, ExportedProgram):
        raise TypeError(
            f"The 'ep' parameter must be an instance of 'ExportedProgram', got '{type(ep).__name__}' instead."
        )

    # Create PT2 archive file
    os.makedirs(args.outdir, exist_ok=True)
    filename = os.path.join(args.outdir, "model.pt2")
    package_pt2(
        filename,
        exported_programs={"model": ep},
        executorch_files={"model.pte": exec_prog.buffer},
    )


if __name__ == "__main__":
    main()
