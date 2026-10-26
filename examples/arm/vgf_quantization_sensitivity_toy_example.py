# Copyright 2026 Arm Limited and/or its affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.
"""Small end-to-end example for VGF quantization sensitivity analysis.

The example does not compile or execute a VGF binary. It uses the VGF PT2E
quantizer to create an INT8 graph, measures FP32-vs-INT8 graph error, and then
rebuilds the model with one requested module left in floating point.
"""

from __future__ import annotations

import json

import torch

from executorch.backends.arm.quantizer import (
    get_symmetric_quantization_config,
    VgfQuantizer,
)
from executorch.backends.arm.vgf import VgfCompileSpec
from executorch.backends.arm.vgf.quantization_assessment import (
    assess_vgf_quantization_across_graph,
)
from executorch.backends.arm.vgf.quantization_sensitivity import (
    analyze_vgf_quantization_sensitivity,
)
from torch.export import ExportedProgram
from torch.fx import GraphModule
from torchao.quantization.pt2e.quantize_pt2e import convert_pt2e, prepare_pt2e


class ToyModel(torch.nn.Module):
    """Tiny MLP used to demonstrate one-module FP32 rescue."""

    def __init__(self) -> None:
        super().__init__()
        self.fc1 = torch.nn.Linear(8, 8)
        self.relu = torch.nn.ReLU()
        self.fc2 = torch.nn.Linear(8, 4)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Run the toy classifier."""
        return self.fc2(self.relu(self.fc1(x)))


def _export(model: torch.nn.Module, example_input: torch.Tensor) -> ExportedProgram:
    return torch.export.export(model.eval(), (example_input,), strict=True)


def _module(exported_program: ExportedProgram) -> GraphModule:
    try:
        return exported_program.module(check_guards=False)
    except TypeError:
        return exported_program.module()


def _build_quantized(
    model: torch.nn.Module,
    example_input: torch.Tensor,
    calibration_inputs: list[torch.Tensor],
    float_module_fqns: tuple[str, ...] = (),
) -> GraphModule:
    graph_module = _module(_export(model, example_input))
    quantizer = VgfQuantizer(VgfCompileSpec("TOSA-1.0+INT"))
    quantizer.set_global(get_symmetric_quantization_config(is_per_channel=False))
    for module_fqn in float_module_fqns:
        quantizer.set_module_name(module_fqn, None)

    prepared = prepare_pt2e(graph_module, quantizer)
    with torch.no_grad():
        for sample in calibration_inputs:
            prepared(sample)
    return convert_pt2e(prepared)


def main() -> None:
    """Run the graph assessment and two FP32 rescue experiments."""
    torch.manual_seed(0)
    model = ToyModel().eval()
    calibration_inputs = [torch.randn(4, 8) for _ in range(8)]
    assessment_inputs = [torch.randn(4, 8) for _ in range(4)]
    example_input = calibration_inputs[0]

    fp32 = _export(model, example_input)
    baseline_int8 = _build_quantized(
        model,
        example_input,
        calibration_inputs,
    )
    assessment = assess_vgf_quantization_across_graph(
        fp32,
        baseline_int8,
        assessment_inputs,
        strict=False,
    )

    def build_mixed(module_fqn: str) -> GraphModule:
        return _build_quantized(
            model,
            example_input,
            calibration_inputs,
            float_module_fqns=(module_fqn,),
        )

    report = analyze_vgf_quantization_sensitivity(
        source_model=model,
        fp32_model=fp32,
        baseline_quantized_model=baseline_int8,
        baseline_assessment=assessment,
        representative_inputs=assessment_inputs,
        build_mixed_quantized_model=build_mixed,
        # Explicit candidates make the example deterministic. Remove this argument
        # to let the assessment select the lowest-SNR modules automatically.
        candidate_module_fqns=("fc1", "fc2"),
    )

    print(json.dumps(report.to_dict(), indent=2))


if __name__ == "__main__":
    main()
