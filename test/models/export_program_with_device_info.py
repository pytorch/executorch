# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-unsafe

"""Exports a simple model with device-annotated tensors for C++ testing.

Uses DeviceAwarePartitioner (BackendWithCompilerDemo + target_device=cuda:0)
so that delegate output tensors are annotated with CUDA device in the .pte.
"""

import argparse
import os

import torch
from executorch.exir import EdgeCompileConfig, ExecutorchBackendConfig, to_edge
from executorch.exir.backend.test.device_util import DeviceAwarePartitioner
from torch import nn
from torch.export import export


class ModuleAddWithDevice(nn.Module):
    """Simple add model — the add op will be delegated with CUDA device annotation."""

    def forward(self, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        return torch.add(a, b)

    def get_random_inputs(self):
        return (torch.randn(2, 2), torch.randn(2, 2))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--outdir", type=str, required=True)
    args = parser.parse_args()

    torch.manual_seed(0)
    model = ModuleAddWithDevice()
    inputs = model.get_random_inputs()

    edge = to_edge(
        export(model, inputs),
        compile_config=EdgeCompileConfig(_check_ir_validity=False),
    )
    lowered = edge.to_backend(DeviceAwarePartitioner())
    et_prog = lowered.to_executorch(
        ExecutorchBackendConfig(  # type: ignore[call-arg]
            emit_stacktrace=False,
            enable_non_cpu_memory_planning=True,
        )
    )

    os.makedirs(args.outdir, exist_ok=True)
    outfile = os.path.join(args.outdir, "ModuleAddWithDevice.pte")

    with open(outfile, "wb") as fp:
        fp.write(et_prog.buffer)
    print(f"Exported ModuleAddWithDevice to {outfile}")


if __name__ == "__main__":
    main()
