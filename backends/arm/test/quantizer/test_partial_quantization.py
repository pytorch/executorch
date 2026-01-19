# Copyright 2025-2026 Arm Limited and/or its affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from collections.abc import Iterable

import torch
import torch.fx

from executorch.backends.arm.constants import DISALLOW_TFA_META_KEY
from executorch.backends.arm.quantizer.arm_quantizer import (
    get_symmetric_quantization_config,
    QuantizationConfig,
)
from executorch.backends.arm.test.tester.test_pipeline import TosaPipelineINT
from executorch.backends.test.harness.stages import StageType


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


class TwoSigmoidsModel(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.sigmoid0 = torch.nn.Sigmoid()
        self.sigmoid1 = torch.nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.sigmoid0(x - 1) + self.sigmoid1(x + 1)

    def example_inputs(self) -> tuple[torch.Tensor, ...]:
        return (torch.randn(2, 3),)


def _collect_disallow_flags(graph: torch.fx.Graph) -> dict[str, bool]:
    flags: dict[str, bool] = {}
    for node in graph.nodes:
        if DISALLOW_TFA_META_KEY in node.meta:
            flags[node.name] = node.meta[DISALLOW_TFA_META_KEY]
    return flags


def _run_quantization_pipeline(
    module: torch.nn.Module,
    global_config: QuantizationConfig | None = None,
    module_type_configs: (
        Iterable[tuple[type[torch.nn.Module], QuantizationConfig | None]] | None
    ) = None,
    module_name_configs: Iterable[tuple[str, QuantizationConfig]] | None = None,
) -> torch.fx.Graph:
    if module_type_configs is None:
        module_type_configs = []
    if module_name_configs is None:
        module_name_configs = []

    pipeline = TosaPipelineINT[tuple[torch.Tensor]](
        module,
        module.example_inputs(),  # type: ignore[operator]
        [],
        [],
    )

    # Set configs for the quantizer
    pipeline.quantizer.set_global(global_config)
    for mod, config in module_type_configs:
        pipeline.quantizer.set_module_type(mod, config)
    for name, config in module_name_configs:
        pipeline.quantizer.set_module_name(name, config)

    pipeline.pop_stage("check_count.exir")
    pipeline.pop_stage("run_method_and_compare_outputs")

    pipeline.run()

    return pipeline.tester.get_artifact(StageType.QUANTIZE).graph


def _assert_disallow_flags(
    graph: torch.fx.Graph,
    expected_flags: dict[str, bool],
) -> None:
    actual_flags = _collect_disallow_flags(graph)
    for node_name, expected_flag in expected_flags.items():
        actual_flag = actual_flags.get(node_name)
        assert (
            actual_flag == expected_flag
        ), f"Node '{node_name}': expected DISALLOW_TFA_META_KEY={expected_flag}, got {actual_flag}"


def test_disallow_tfa_for_skipped_module():
    """Ensure a softmax skipped for quantization is not decomposed and that its
    node has `disallow_tfa` set.
    """

    class TwoOpModel(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.softmax = torch.nn.Softmax(dim=1)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            return self.softmax(x) + 1

        def example_inputs(self) -> tuple[torch.Tensor, ...]:
            return (torch.randn(2, 3),)

    graph_after_quant_stage = _run_quantization_pipeline(
        TwoOpModel(),
        global_config=get_symmetric_quantization_config(),
        module_type_configs=[(torch.nn.Softmax, None)],
    )
    _assert_disallow_flags(
        graph_after_quant_stage,
        {
            "x": False,
            "softmax": True,
            "add": False,
            "output": False,
        },
    )


def test_disallow_tfa_for_two_skipped_modules():
    """Ensure a softmax and linear skipped for quantization are not decomposed
    and have their `disallow_tfa` set.
    """

    graph_after_quant_stage = _run_quantization_pipeline(
        LinearSoftmaxModel(),
        global_config=get_symmetric_quantization_config(),
        module_type_configs=[(torch.nn.Linear, None), (torch.nn.Softmax, None)],
    )
    _assert_disallow_flags(
        graph_after_quant_stage,
        {
            "x": False,
            "linear": True,
            "softmax": True,
            "add": False,
            "abs_1": False,
            "output": False,
        },
    )


def test_disallow_tfa_with_global_none_and_one_quantized_module():
    """Ensure that with a global None quantization config, only the linear
    module (with its own quantization config) is quantized, and that the
    other nodes have `disallow_tfa` set.
    """

    graph_after_quant_stage = _run_quantization_pipeline(
        LinearSoftmaxModel(),
        global_config=None,
        module_type_configs=[(torch.nn.Linear, get_symmetric_quantization_config())],
    )
    _assert_disallow_flags(
        graph_after_quant_stage,
        {
            "x": True,
            "linear": False,
            "softmax": True,
            "add": True,
            "abs_1": True,
            "output": True,
        },
    )


def test_disallow_tfa_for_submodule_by_name():
    """Ensure submodules can be skipped for quantization by name and have their
    nodes marked as disallowed for TFA.
    """

    # Quantize the entire model except self.sigmoid0
    graph_after_quant_stage = _run_quantization_pipeline(
        TwoSigmoidsModel(),
        global_config=get_symmetric_quantization_config(),
        module_name_configs=[("sigmoid0", None)],
    )
    _assert_disallow_flags(
        graph_after_quant_stage,
        {
            "x": False,
            "sigmoid": True,
            "sigmoid_1": False,
            "add": False,
            "output": False,
        },
    )


def test_disallow_tfa_name_config_contradicts_type_config():
    """Ensure that module name configs take precedence over module type configs
    when they contradict each other.
    """

    graph_after_quant_stage = _run_quantization_pipeline(
        TwoSigmoidsModel(),
        global_config=None,
        module_type_configs=[(torch.nn.Sigmoid, get_symmetric_quantization_config())],
        module_name_configs=[("sigmoid0", None)],
    )
    _assert_disallow_flags(
        graph_after_quant_stage,
        {
            "x": True,
            "sigmoid": True,
            "sigmoid_1": False,
            "add": True,
            "output": True,
        },
    )
