# Copyright 2026 Arm Limited and/or its affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.
"""Export and assert a deterministic VGF -> Vulkan -> VGF execution plan."""

import argparse
import json
from pathlib import Path
from typing import Mapping

import torch
import torch.nn.functional as F
from executorch.backends.arm.vgf import VgfCompileSpec
from executorch.backends.arm.vgf.partitioner import VgfPartitioner
from executorch.backends.vulkan.partitioner.vulkan_partitioner import VulkanPartitioner
from executorch.exir import EdgeCompileConfig, to_edge_transform_and_lower
from executorch.exir.dialects._ops import ops as exir_ops
from executorch.exir.schema import DelegateCall, KernelCall
from torch.fx.passes.operator_support import OperatorSupportBase

EXPECTED_SEQUENCE = ["VgfBackend", "VulkanBackend", "VgfBackend"]


class MixedModel(torch.nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.avg_pool2d(x, kernel_size=2, stride=2)  # VGF prefix
        x = torch.sigmoid(x)  # Deliberately assigned to Vulkan, not a support claim.
        return x * x + 0.25  # VGF suffix


class KeepSigmoidForVulkan(OperatorSupportBase):
    """Placement policy for this test only; do not change backend support tables."""

    def is_node_supported(
        self, submodules: Mapping[str, torch.nn.Module], node: torch.fx.Node
    ) -> bool:
        del submodules
        return node.target not in {
            torch.ops.aten.sigmoid.default,
            exir_ops.edge.aten.sigmoid.default,
        }


def assert_execution_plan(program) -> list[str]:
    plans = [
        p for p in program.executorch_program.execution_plan if p.name == "forward"
    ]
    if len(plans) != 1:
        raise RuntimeError(f"Expected one forward execution plan, got {len(plans)}")
    plan = plans[0]
    if len(plan.chains) != 1:
        raise RuntimeError("This linear example must emit exactly one execution chain")
    sequence: list[str] = []
    fallback: list[str] = []
    for instruction in plan.chains[0].instructions:
        args = instruction.instr_args
        if isinstance(args, DelegateCall):
            sequence.append(plan.delegates[args.delegate_index].id)
        elif isinstance(args, KernelCall):
            operator = plan.operators[args.op_index]
            fallback.append(f"{operator.name}.{operator.overload}")
    if sequence != EXPECTED_SEQUENCE or fallback:
        raise RuntimeError(
            "Mixed placement changed; refusing to publish a misleading .pte. "
            f"Expected {EXPECTED_SEQUENCE}, got {sequence}; CPU kernels={fallback}"
        )
    return sequence


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir", type=Path, default=Path("vgf_vulkan_shared_out")
    )
    parser.add_argument("--compiler-flag", action="append", default=[])
    args = parser.parse_args()
    torch.manual_seed(0)
    model = MixedModel().eval()
    x = torch.linspace(-2.0, 2.0, 1 * 3 * 8 * 8, dtype=torch.float32).reshape(
        1, 3, 8, 8
    )
    with torch.no_grad():
        expected = model(x).contiguous()
        exported = torch.export.export(model, (x,), strict=True)
    spec = VgfCompileSpec(tosa_spec="TOSA-1.0+FP", compiler_flags=args.compiler_flag)
    spec.validate_environment()
    edge = to_edge_transform_and_lower(
        exported,
        partitioner=[
            VgfPartitioner(spec, additional_checks=[KeepSigmoidForVulkan()]),
            VulkanPartitioner(),
        ],
        compile_config=EdgeCompileConfig(
            _check_ir_validity=False, _skip_dim_order=True
        ),
    )
    program = edge.to_executorch()
    sequence = assert_execution_plan(program)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    pte = args.output_dir / "vgf_vulkan_vgf.pte"
    with pte.open("wb") as stream:
        program.write_to_file(stream)
    x.numpy().astype("<f4").tofile(args.output_dir / "input0.bin")
    expected.numpy().astype("<f4").tofile(args.output_dir / "expected0.bin")
    (args.output_dir / "edge_graph.txt").write_text(
        str(edge.exported_program().graph_module.graph) + "\n", encoding="utf-8"
    )
    (args.output_dir / "manifest.json").write_text(
        json.dumps(
            {
                "model": pte.name,
                "input_shape": list(x.shape),
                "output_shape": list(expected.shape),
                "dtype": "float32-little-endian",
                "delegate_call_order": sequence,
                "cpu_kernel_calls": [],
                "rtol": 1e-4,
                "atol": 1e-5,
                "placement": "sigmoid explicitly excluded from VGF for this example",
                "runtime_options_serialized": False,
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    print(" -> ".join(sequence))
    print(f"Wrote {pte}")


if __name__ == "__main__":
    main()
