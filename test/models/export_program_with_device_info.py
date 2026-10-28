# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-unsafe

"""Exports simple models with device-annotated tensors for C++ testing.

Uses DeviceAwarePartitioner (BackendWithCompilerDemo + target_device=cuda:0)
so that delegate output tensors are annotated with CUDA device in the .pte.
ModuleAddConstantWithDevice is exported with skip_h2d_for_constants and
target_device=cuda:1, so its constant is planned in cuda:1 memory with its data
as the initial state, once in the program and once in a separate .ptd file.
"""

import argparse
import os

import torch
from executorch.exir import EdgeCompileConfig, ExecutorchBackendConfig, to_edge
from executorch.exir.backend.test.device_util import DeviceAwarePartitioner
from executorch.exir.passes import MemoryPlanningPass
from executorch.exir.passes.propagate_device_config import PropagateDeviceConfig
from torch import nn
from torch.export import export


class ModuleAddWithDevice(nn.Module):
    """Simple add model — the add op will be delegated with CUDA device annotation."""

    def forward(self, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        return torch.add(a, b)

    def get_random_inputs(self):
        return (torch.randn(2, 2), torch.randn(2, 2))


class ModuleAddConstantWithDevice(nn.Module):
    """Adds a constant buffer to the input in a delegate on CUDA."""

    offset: torch.Tensor

    def __init__(self) -> None:
        super().__init__()
        self.register_buffer("offset", torch.arange(4.0).reshape(2, 2))

    def forward(self, a: torch.Tensor) -> torch.Tensor:
        return torch.add(a, self.offset)

    def get_random_inputs(self):
        return (torch.randn(2, 2),)


def export_constant_on_device(outdir: str, external_constants: bool) -> None:
    model = ModuleAddConstantWithDevice()
    lowered = to_edge(
        export(model, model.get_random_inputs()),
        compile_config=EdgeCompileConfig(_check_ir_validity=False),
    ).to_backend(DeviceAwarePartitioner("cuda:1"))
    et_prog = lowered.to_executorch(
        ExecutorchBackendConfig(  # type: ignore[call-arg]
            emit_stacktrace=False,
            enable_non_cpu_memory_planning=True,
            propagate_device_config=PropagateDeviceConfig(
                skip_h2d_for_method_inputs=True, skip_h2d_for_constants=True
            ),
            memory_planning_pass=MemoryPlanningPass(alloc_graph_input=False),
            external_constants=external_constants,
        )
    )
    name = "ModuleAddConstantWithDevice" + ("Program" if external_constants else "")
    with open(os.path.join(outdir, f"{name}.pte"), "wb") as fp:
        fp.write(et_prog.buffer)
    if external_constants:
        et_prog._tensor_data[name] = et_prog._tensor_data.pop(
            "_default_external_constant"
        )
        et_prog.write_tensor_data_to_file(outdir)


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

    for external_constants in (False, True):
        export_constant_on_device(args.outdir, external_constants)


if __name__ == "__main__":
    main()
