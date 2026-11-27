# Copyright 2026 Arm Limited and/or its affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import copy

import pytest
import torch

from executorch.backends.arm.vgf.quantization_assessment import (
    assess_vgf_quantization_across_graph,
)
from executorch.backends.arm.vgf.quantization_quality import (
    compute_vgf_quantization_metrics,
)


class ToyAttention(torch.nn.Module):
    def __init__(self, width: int) -> None:
        super().__init__()
        self.q = torch.nn.Linear(width, width)
        self.k = torch.nn.Linear(width, width)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.relu(self.q(x) + self.k(x))


class ToyMlp(torch.nn.Module):
    def __init__(self, width: int) -> None:
        super().__init__()
        self.fc1 = torch.nn.Linear(width, width * 2)
        self.fc2 = torch.nn.Linear(width * 2, width)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc2(torch.relu(self.fc1(x)))


class ToyBlock(torch.nn.Module):
    def __init__(self, width: int) -> None:
        super().__init__()
        self.attn = ToyAttention(width)
        self.mlp = ToyMlp(width)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.mlp(self.attn(x))


class ToyModel(torch.nn.Module):
    def __init__(self, width: int = 4, blocks: int = 2) -> None:
        super().__init__()
        self.blocks = torch.nn.ModuleList([ToyBlock(width) for _ in range(blocks)])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for block in self.blocks:
            x = block(x)
        return x


def _rounded_copy(model: ToyModel, step: float = 0.05) -> ToyModel:
    rounded = copy.deepcopy(model)
    with torch.no_grad():
        for parameter in rounded.parameters():
            parameter.copy_(torch.round(parameter / step) * step)
    return rounded


def _export(model: torch.nn.Module, example: torch.Tensor):
    return torch.export.export(model.eval(), (example,), strict=True)


def _set_debug_handle(exported_program, module_fqn: str, handle: int) -> None:
    for node in exported_program.graph_module.graph.nodes:
        stack = node.meta.get("nn_module_stack", {})
        fqns = [
            value[0]
            for value in stack.values()
            if isinstance(value, (tuple, list)) and value and value[0]
        ]
        if fqns and fqns[-1] == module_fqn and node.op == "call_function":
            node.meta["debug_handle"] = handle


def test_assessment_uses_leaf_module_fqn_as_identity() -> None:
    torch.manual_seed(0)
    model = ToyModel()
    quantized_model = _rounded_copy(model)
    sample = torch.randn(2, 4)

    result = assess_vgf_quantization_across_graph(
        _export(model, sample),
        _export(quantized_model, sample),
        [(sample,)],
    )

    assert set(result.modules) == {
        "blocks.0.attn.q",
        "blocks.0.attn.k",
        "blocks.0.mlp.fc1",
        "blocks.0.mlp.fc2",
        "blocks.1.attn.q",
        "blocks.1.attn.k",
        "blocks.1.mlp.fc1",
        "blocks.1.mlp.fc2",
    }
    assert "blocks.0" not in result.modules
    assert "blocks.0.attn" not in result.modules
    assert result.fp32_only_module_fqns == ()
    assert result.quantized_only_module_fqns == ()
    assert result.skipped_module_fqns == {}


def test_assessment_aggregates_representative_inputs() -> None:
    torch.manual_seed(1)
    model = ToyModel(blocks=1)
    quantized_model = _rounded_copy(model, step=0.1)
    samples = [torch.randn(2, 4), torch.randn(2, 4)]

    result = assess_vgf_quantization_across_graph(
        _export(model, samples[0]),
        _export(quantized_model, samples[0]),
        [(sample,) for sample in samples],
    )

    reference_block = model.blocks[0]
    quantized_block = quantized_model.blocks[0]
    assert isinstance(reference_block, ToyBlock)
    assert isinstance(quantized_block, ToyBlock)

    reference = torch.cat(
        [reference_block.attn.q(sample).reshape(-1) for sample in samples]
    )
    quantized = torch.cat(
        [quantized_block.attn.q(sample).reshape(-1) for sample in samples]
    )
    expected = compute_vgf_quantization_metrics(reference, quantized)
    assessment = result.modules["blocks.0.attn.q"]

    assert assessment.metrics.mse == pytest.approx(expected.mse)
    assert assessment.metrics.snr_db == pytest.approx(expected.snr_db)
    assert assessment.metrics.cosine_similarity == pytest.approx(
        expected.cosine_similarity
    )
    assert assessment.samples == 2
    assert assessment.compared_numel == reference.numel()


def test_compact_output_can_be_sorted_worst_quality_first() -> None:
    torch.manual_seed(4)
    model = ToyModel(blocks=1)
    quantized_model = _rounded_copy(model, step=0.1)
    sample = torch.randn(2, 4)

    result = assess_vgf_quantization_across_graph(
        _export(model, sample),
        _export(quantized_model, sample),
        [sample],
    )
    payload = result.to_dict(sort_by_quality=True)

    snr_values = [metrics["snr_db"] for metrics in payload.values()]
    assert snr_values == sorted(snr_values)
    assert set(payload) == set(result.modules)


def test_compact_output_matches_requested_vgf_shape() -> None:
    torch.manual_seed(2)
    model = ToyModel(blocks=1)
    quantized_model = _rounded_copy(model)
    sample = torch.randn(2, 4)

    payload = assess_vgf_quantization_across_graph(
        _export(model, sample),
        _export(quantized_model, sample),
        [sample],
    ).to_dict()

    assert set(payload["blocks.0.attn.q"]) == {"mse", "snr_db", "cosine"}


def test_detailed_output_retains_fx_and_debug_handle_metadata() -> None:
    torch.manual_seed(3)
    model = ToyModel(blocks=1)
    quantized_model = _rounded_copy(model)
    sample = torch.randn(2, 4)
    fp32_export = _export(model, sample)
    quantized_export = _export(quantized_model, sample)

    _set_debug_handle(fp32_export, "blocks.0.attn.q", 101)
    _set_debug_handle(quantized_export, "blocks.0.attn.q", 202)

    detailed = assess_vgf_quantization_across_graph(
        fp32_export,
        quantized_export,
        [sample],
    ).to_dict(include_metadata=True)

    q = detailed["blocks.0.attn.q"]
    assert q["metadata"]["occurrences"] == 1
    assert q["metadata"]["nodes"][0]["fp32_debug_handles"] == [101]
    assert q["metadata"]["nodes"][0]["quantized_debug_handles"] == [202]
    assert q["metadata"]["nodes"][0]["fp32_fx_node"]
    assert q["metadata"]["nodes"][0]["quantized_fx_node"]


def test_reports_module_fqns_present_in_only_one_graph() -> None:
    class WithTail(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.q = torch.nn.Linear(4, 4)
            self.tail = torch.nn.Linear(4, 4)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            return self.tail(self.q(x))

    class WithoutTail(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.q = torch.nn.Linear(4, 4)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            return self.q(x)

    sample = torch.randn(2, 4)
    result = assess_vgf_quantization_across_graph(
        _export(WithTail(), sample),
        _export(WithoutTail(), sample),
        [sample],
    )

    assert result.fp32_only_module_fqns == ("tail",)
    assert "q" in result.modules

    with pytest.raises(ValueError, match="module FQN mismatch"):
        assess_vgf_quantization_across_graph(
            _export(WithTail(), sample),
            _export(WithoutTail(), sample),
            [sample],
            strict=True,
        )


def test_requires_representative_inputs() -> None:
    sample = torch.randn(2, 4)
    exported = _export(ToyModel(blocks=1), sample)

    with pytest.raises(ValueError, match="at least one sample"):
        assess_vgf_quantization_across_graph(exported, exported, [])
