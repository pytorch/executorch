# Copyright 2025 Arm Limited and/or its affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from collections.abc import Iterable

import torch
import torch.fx

from executorch.backends.arm.constants import DISALLOW_TFA_META_KEY
from executorch.backends.arm.test.tester.test_pipeline import TosaPipelineINT
from executorch.backends.test.harness.stages import StageType


def _collect_disallow_flags(graph: torch.fx.Graph) -> dict[str, bool | None]:
    flags: dict[str, bool | None] = {}
    for node in graph.nodes:
        if DISALLOW_TFA_META_KEY in node.meta:
            flags[node.name] = node.meta[DISALLOW_TFA_META_KEY]
    return flags


def _run_quantization_pipeline(
    module: torch.nn.Module,
    unquantized_submodules: Iterable[type[torch.nn.Module]],
) -> torch.fx.Graph:
    """Run the Arm TOSA quantization pipeline for ``module`` while keeping the
    specified submodules in floating-point.
    """
    pipeline = TosaPipelineINT[tuple[torch.Tensor]](
        module,
        module.example_inputs(),  # type: ignore[operator]
        [],
        [],
    )

    quant_stage = pipeline._stages[0].args[0]
    for mod in unquantized_submodules:
        quant_stage.quantizer.set_module_type(mod, None)

    pipeline.pop_stage("check_count.exir")
    pipeline.pop_stage("run_method_and_compare_outputs")
    pipeline.run()

    return pipeline.tester.get_artifact(StageType.QUANTIZE).graph


def test_disallow_tfa_for_skipped_module():
    """Ensure a softmax skipped for quantization is not decomposed and that its
    node has `disallow_tfa` set."""

    class TwoOpModel(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.softmax = torch.nn.Softmax(dim=1)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            return self.softmax(x) + 1

        def example_inputs(self) -> tuple[torch.Tensor, ...]:
            return (torch.randn(2, 3),)

    graph_after_quant_stage = _run_quantization_pipeline(
        TwoOpModel(), [torch.nn.Softmax]
    )

    flags = _collect_disallow_flags(graph_after_quant_stage)

    assert flags.get("x") is False, "'x' should not be disallowed for TFA"
    assert flags.get("softmax") is True, "'softmax' should be disallowed for TFA"
    assert flags.get("add") is False, "'add' should not be disallowed for TFA"
    assert flags.get("output") is False, "'output' should not be disallowed for TFA"


def test_disallow_tfa_for_two_skipped_modules():
    class LinearSoftmaxModel(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.linear = torch.nn.Linear(10, 10)
            self.softmax = torch.nn.Softmax(dim=1)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            x = self.linear(x)
            x = torch.abs(x)
            x = self.softmax(x)
            return x + 1

        def example_inputs(self) -> tuple[torch.Tensor, ...]:
            return (torch.randn(1, 10),)

    graph_after_quant_stage = _run_quantization_pipeline(
        LinearSoftmaxModel(),
        [torch.nn.Linear, torch.nn.Softmax],
    )

    flags = _collect_disallow_flags(graph_after_quant_stage)

    assert flags.get("x") is False, "'x' should not be disallowed for TFA"
    assert flags.get("linear") is True, "'linear' should be disallowed for TFA"
    assert flags.get("softmax") is True, "'softmax' should be disallowed for TFA"
    assert flags.get("add") is False, "'add' should not be disallowed for TFA"
    assert flags.get("abs_1") is False, "'abs_1' should not be disallowed for TFA"
    assert flags.get("output") is False, "'output' should not be disallowed for TFA"
