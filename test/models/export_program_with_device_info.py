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
from typing import Dict, final

import torch
from executorch.exir import EdgeCompileConfig, ExecutorchBackendConfig, to_edge
from executorch.exir.backend.canonical_partitioners.pattern_op_partitioner import (
    generate_pattern_op_partitions,
)
from executorch.exir.backend.compile_spec_schema import CompileSpec
from executorch.exir.backend.partitioner import (
    DelegationSpec,
    Partitioner,
    PartitionResult,
)
from executorch.exir.backend.test.backend_with_compiler_demo import (
    BackendWithCompilerDemo,
)
from executorch.exir.dialects._ops import ops as exir_ops
from executorch.exir.passes.propagate_device_pass import TARGET_DEVICE_COMPILE_SPEC_KEY
from torch import nn
from torch.export import export
from torch.fx.passes.operator_support import any_chain, OperatorSupportBase


class _AddOperatorSupport(OperatorSupportBase):
    def is_node_supported(self, submodules, node: torch.fx.Node) -> bool:
        return node.op == "call_function" and node.target in [
            exir_ops.edge.aten.add.Tensor,
        ]


@final
class _DeviceAwarePartitioner(Partitioner):
    """Partitioner that tags add ops for delegation with target_device=cuda:0."""

    def __init__(self) -> None:
        super().__init__()
        self.delegation_spec = DelegationSpec(
            BackendWithCompilerDemo.__name__,
            [
                CompileSpec("max_value", bytes([4])),
                CompileSpec(TARGET_DEVICE_COMPILE_SPEC_KEY, b"cuda:0"),
            ],
        )

    def partition(self, exported_program) -> PartitionResult:
        partition_tags: Dict[str, DelegationSpec] = {}
        partition_list = generate_pattern_op_partitions(
            exported_program.graph_module,
            op_support=any_chain(_AddOperatorSupport()),
        )
        for partition in partition_list:
            for node in partition.nodes:
                tag = f"tag{partition.id}"
                node.meta["delegation_tag"] = tag
                partition_tags[tag] = self.delegation_spec
        return PartitionResult(
            tagged_exported_program=exported_program,
            partition_tags=partition_tags,
        )


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
    lowered = edge.to_backend(_DeviceAwarePartitioner())
    et_prog = lowered.to_executorch(
        ExecutorchBackendConfig(
            emit_stacktrace=False,
            enable_non_cpu_memory_planning=True,
        )
    )

    os.makedirs(args.outdir, exist_ok=True)
    outfile = os.path.join(args.outdir, "ModuleAddWithDevice.pte")

    # Verify device annotations are present in the serialized program
    from executorch.exir.schema import DeviceType, Tensor as SchemaTensor

    program = et_prog._emitter_output.program
    plan = program.execution_plan[0]
    print(f"Delegates: {len(plan.delegates)}")
    cuda_count = 0
    for i, v in enumerate(plan.values):
        if isinstance(v.val, SchemaTensor):
            t = v.val
            eti = t.extra_tensor_info
            dev = eti.device_type if eti else "no_eti"
            print(f"  Tensor[{i}]: sizes={list(t.sizes)}, device={dev}")
            if eti and eti.device_type == DeviceType.CUDA:
                cuda_count += 1
    print(f"CUDA tensors: {cuda_count}")

    # Also check graph module specs
    from executorch.exir.delegate import executorch_call_delegate
    from executorch.exir.tensor import TensorSpec

    gm = et_prog.exported_program().graph_module
    for node in gm.graph.nodes:
        if node.op == "call_function" and node.target == executorch_call_delegate:
            specs = node.meta.get("spec")
            print(
                f"  Delegate node '{node.name}' spec.device = {specs.device if isinstance(specs, TensorSpec) else [s.device for s in specs if isinstance(s, TensorSpec)]}"
            )

    with open(outfile, "wb") as fp:
        fp.write(et_prog.buffer)
    print(f"Exported ModuleAddWithDevice to {outfile}")


if __name__ == "__main__":
    main()
