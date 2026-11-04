# Copyright 2026 Arm Limited and/or its affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import copy
import json
import math

import re
from dataclasses import replace

import pytest

import torch

from executorch.backends.arm.vgf.quantization_assessment import (
    assess_vgf_quantization_across_graph,
    VgfGraphQuantizationAssessment,
)
from executorch.backends.arm.vgf.quantization_quality import VgfQuantizationMetrics
from executorch.backends.arm.vgf.quantization_sensitivity import (
    _module_graph_signature,
    _module_intervention_verified,
    _module_qdq_boundary_signature,
    _snr_rescue,
    analyze_vgf_quantization_sensitivity,
    expand_module_fqns_with_parents,
    resolve_source_module_fqn,
    select_vgf_quantization_sensitivity_candidates,
    VgfQuantizationSensitivityCandidate,
    VgfQuantizationSensitivityIntervention,
    VgfQuantizationSensitivityOutputResult,
    VgfQuantizationSensitivityReport,
)
from torch.fx import Graph, GraphModule


class ToyBlock(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.linear = torch.nn.Linear(2, 2, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear(x)


class ToyModel(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.a = torch.nn.Linear(2, 2, bias=False)
        self.b = torch.nn.Linear(2, 2, bias=False)
        self.block = ToyBlock()
        with torch.no_grad():
            self.a.weight.copy_(torch.eye(2))
            self.b.weight.copy_(torch.tensor([[0.3, -0.2], [0.7, 0.4]]))
            self.block.linear.weight.copy_(torch.eye(2))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(self.b(self.a(x)))


def _export(model: torch.nn.Module, sample: torch.Tensor):
    return torch.export.export(model.eval(), (sample,), strict=True)


def _rounded_copy(
    model: ToyModel,
    float_module_fqn: str | None = None,
) -> ToyModel:
    rounded = copy.deepcopy(model)
    float_prefix = None if float_module_fqn is None else f"{float_module_fqn}."
    with torch.no_grad():
        for module_name, module in rounded.named_modules():
            if not isinstance(module, torch.nn.Linear):
                continue
            if float_module_fqn is not None and (
                module_name == float_module_fqn
                or (float_prefix is not None and module_name.startswith(float_prefix))
            ):
                continue
            for parameter in module.parameters(recurse=False):
                parameter.copy_(torch.round(parameter * 2.0) / 2.0)
    return rounded


def test_resolves_exporter_synthetic_fqn_to_source_block() -> None:
    model = ToyModel()
    assert (
        resolve_source_module_fqn(
            "block.forward.__self__.linear",
            model,
        )
        == "block"
    )
    assert resolve_source_module_fqn("b", model) == "b"


def test_selects_unique_worst_source_modules() -> None:
    model = ToyModel()
    sample = torch.tensor([[1.0, -2.0]])
    baseline = _rounded_copy(model)
    assessment = assess_vgf_quantization_across_graph(
        _export(model, sample),
        _export(baseline, sample),
        [sample],
    )

    candidates = select_vgf_quantization_sensitivity_candidates(
        assessment,
        model,
        limit=2,
    )

    assert candidates
    assert len({candidate.module_fqn for candidate in candidates}) == len(candidates)


def test_report_to_dict_normalizes_non_finite_metrics() -> None:
    """Report dictionaries must serialize with strict JSON semantics."""
    baseline_metrics = VgfQuantizationMetrics(
        mse=math.nan,
        snr_db=-math.inf,
        cosine_similarity=math.inf,
        max_abs_error=0.0,
        relative_error=0.0,
        reference_min=0.0,
        reference_max=1.0,
        quantized_min=0.0,
        quantized_max=1.0,
        saturation_ratio=None,
        clipping_ratio=None,
    )
    mixed_metrics = replace(
        baseline_metrics,
        mse=0.25,
        snr_db=math.inf,
        cosine_similarity=0.9,
    )
    candidate = VgfQuantizationSensitivityCandidate(
        module_fqn="b",
        assessment_fqn="b",
        local_snr_db=-math.inf,
        local_mse=math.nan,
        local_cosine=math.inf,
    )
    output_result = VgfQuantizationSensitivityOutputResult(
        name="final",
        baseline_metrics=baseline_metrics,
        mixed_metrics=mixed_metrics,
        snr_rescue_db=math.inf,
        mse_reduction_ratio=-math.inf,
        cosine_delta=math.nan,
    )
    intervention = VgfQuantizationSensitivityIntervention(
        candidate=candidate,
        mixed_local_metrics=mixed_metrics,
        local_snr_rescue_db=math.inf,
        local_mse_reduction_ratio=-math.inf,
        local_cosine_delta=math.nan,
        intervention_verified=True,
        baseline_qdq_input_edges=1,
        mixed_qdq_input_edges=0,
        output_results=(output_result,),
        primary_output_name="final",
    )
    report = VgfQuantizationSensitivityReport(
        baseline_output_metrics={"final": baseline_metrics},
        interventions=(intervention,),
        primary_output_name="final",
    )

    payload = report.to_dict()

    # This fails if any nested report field still contains NaN or +/-Infinity.
    encoded = json.dumps(payload, allow_nan=False)
    decoded = json.loads(encoded)

    assert decoded["baseline_final"]["snr_db"] is None
    assert decoded["baseline_final"]["mse"] is None
    assert decoded["baseline_final"]["cosine"] is None
    assert decoded["interventions"][0]["local_snr_db"] is None
    assert decoded["interventions"][0]["outputs"]["final"]["snr_rescue_db"] is None
    assert (
        decoded["interventions"][0]["outputs"]["final"]["mse_reduction_ratio"] is None
    )
    assert decoded["interventions"][0]["outputs"]["final"]["cosine_delta"] is None


def test_automatic_selection_keeps_negative_infinite_snr() -> None:
    """Keep -inf as the worst candidate while skipping NaN and +inf."""
    model = ToyModel()
    sample = torch.tensor([[1.0, -2.0]])
    baseline = _rounded_copy(model)
    assessment = assess_vgf_quantization_across_graph(
        _export(model, sample),
        _export(baseline, sample),
        [sample],
    )

    # ToyModel produces these leaf-module assessment entries. Override only
    # their SNR values so this test isolates candidate-selection semantics.
    assert {"a", "b", "block.linear"} <= set(assessment.modules)
    modules = dict(assessment.modules)
    modules["a"] = replace(
        modules["a"],
        metrics=replace(modules["a"].metrics, snr_db=-math.inf),
    )
    modules["b"] = replace(
        modules["b"],
        metrics=replace(modules["b"].metrics, snr_db=math.inf),
    )
    modules["block.linear"] = replace(
        modules["block.linear"],
        metrics=replace(modules["block.linear"].metrics, snr_db=math.nan),
    )
    assessment = replace(assessment, modules=modules)

    candidates = select_vgf_quantization_sensitivity_candidates(
        assessment,
        model,
        limit=3,
    )

    # +inf and NaN are not useful worst-case candidates, but -inf represents
    # maximal quantization error and must not be discarded.
    assert [candidate.module_fqn for candidate in candidates] == ["a"]
    assert candidates[0].local_snr_db == -math.inf


@pytest.mark.parametrize(
    ("mixed_snr_db", "baseline_snr_db", "expected"),
    [
        (3.0, -math.inf, math.inf),
        (math.inf, -math.inf, math.inf),
        (-math.inf, -math.inf, math.nan),
        (math.inf, math.inf, math.nan),
        (math.inf, 3.0, math.inf),
        (3.0, math.inf, -math.inf),
        (-math.inf, 3.0, -math.inf),
        (7.5, 2.0, 5.5),
        (math.nan, 2.0, math.nan),
        (2.0, math.nan, math.nan),
    ],
)
def test_snr_rescue_handles_infinite_baseline(
    mixed_snr_db: float,
    baseline_snr_db: float,
    expected: float,
) -> None:
    """Report ordered infinite improvements without hiding them as NaN."""
    rescue = _snr_rescue(mixed_snr_db, baseline_snr_db)
    if math.isnan(expected):
        assert math.isnan(rescue)
    else:
        assert rescue == expected


def test_expands_leaf_candidate_to_parent_block() -> None:
    model = ToyModel()
    assert expand_module_fqns_with_parents(
        ("block.linear",),
        model,
        parent_levels=1,
    ) == ("block.linear", "block")


def _fake_quantize_per_tensor(x: torch.Tensor) -> torch.Tensor:
    return x


def _fake_dequantize_per_tensor(x: torch.Tensor) -> torch.Tensor:
    return x


def _graph_with_output_qdq(*, output_qdq: bool) -> GraphModule:
    graph = Graph()
    x = graph.placeholder("x")
    module_node = graph.call_function(torch.relu, args=(x,))
    module_node.meta["nn_module_stack"] = {
        "block": ("block", ToyBlock),
    }

    output = module_node
    if output_qdq:
        quantized = graph.call_function(
            _fake_quantize_per_tensor,
            args=(module_node,),
        )
        output = graph.call_function(
            _fake_dequantize_per_tensor,
            args=(quantized,),
        )
    graph.output(output)
    return GraphModule({}, graph)


def test_intervention_verification_detects_output_qdq_change() -> None:
    """Removing only output Q/DQ must verify the FP32 intervention."""
    baseline = _graph_with_output_qdq(output_qdq=True)
    mixed = _graph_with_output_qdq(output_qdq=False)

    assert _module_graph_signature(baseline, "block") == _module_graph_signature(
        mixed,
        "block",
    )
    assert _module_qdq_boundary_signature(
        baseline,
        "block",
    ) != _module_qdq_boundary_signature(
        mixed,
        "block",
    )
    assert _module_intervention_verified(baseline, mixed, "block")
    assert not _module_intervention_verified(mixed, mixed, "block")


def test_float_rescue_reports_multiple_output_views() -> None:
    model = ToyModel()
    samples = [
        torch.tensor([[1.0, -2.0]]),
        torch.tensor([[0.5, 1.5]]),
    ]
    fp32_export = _export(model, samples[0])
    baseline_model = _rounded_copy(model)
    baseline_export = _export(baseline_model, samples[0])
    assessment = assess_vgf_quantization_across_graph(
        fp32_export,
        baseline_export,
        samples,
    )

    def build_mixed(module_fqn: str):
        return _export(_rounded_copy(model, module_fqn), samples[0])

    report = analyze_vgf_quantization_sensitivity(
        source_model=model,
        fp32_model=fp32_export,
        baseline_quantized_model=baseline_export,
        baseline_assessment=assessment,
        representative_inputs=samples,
        build_mixed_quantized_model=build_mixed,
        candidate_module_fqns=("b",),
        output_selectors={
            "final": lambda value: value,
            "scaled": lambda value: value * 2.0,
        },
        primary_output_name="final",
    )

    assert set(report.baseline_output_metrics) == {"final", "scaled"}
    assert len(report.interventions) == 1
    intervention = report.interventions[0]
    assert intervention.candidate.module_fqn == "b"
    assert intervention.output_result("final").mixed_metrics.mse < (
        report.baseline_output_metrics["final"].mse
    )
    assert intervention.output_result("final").snr_rescue_db > 0.0
    assert intervention.output_result("scaled").snr_rescue_db > 0.0


def test_parent_block_can_be_intervened_explicitly() -> None:
    model = ToyModel()
    samples = [torch.tensor([[1.0, -2.0]])]
    fp32_export = _export(model, samples[0])
    baseline_export = _export(_rounded_copy(model), samples[0])
    assessment = assess_vgf_quantization_across_graph(
        fp32_export,
        baseline_export,
        samples,
    )

    def build_mixed(module_fqn: str):
        return _export(_rounded_copy(model, module_fqn), samples[0])

    report = analyze_vgf_quantization_sensitivity(
        source_model=model,
        fp32_model=fp32_export,
        baseline_quantized_model=baseline_export,
        baseline_assessment=assessment,
        representative_inputs=samples,
        build_mixed_quantized_model=build_mixed,
        candidate_module_fqns=("block",),
    )

    assert report.interventions[0].candidate.module_fqn == "block"


# Regression test for explicit-candidate validation.
@pytest.mark.parametrize(
    ("candidate_fqn", "parent_levels"),
    [
        ("missing", 0),
        ("block.missing", 1),
    ],
)
def test_explicit_unknown_candidate_is_rejected_before_parent_expansion(
    candidate_fqn: str,
    parent_levels: int,
) -> None:
    class CandidateValidationModel(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.block = torch.nn.Sequential(
                torch.nn.Linear(4, 4),
                torch.nn.ReLU(),
            )

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            return self.block(x)

    model = CandidateValidationModel().eval()
    sample = torch.randn(2, 4)
    exported = torch.export.export(model, (sample,), strict=True)
    assessment = VgfGraphQuantizationAssessment(
        modules={},
        fp32_only_module_fqns=(),
        quantized_only_module_fqns=(),
        skipped_module_fqns={},
    )
    built_modules: list[str] = []

    def build_mixed(module_fqn: str) -> torch.export.ExportedProgram:
        built_modules.append(module_fqn)
        return exported

    with pytest.raises(
        ValueError,
        match=rf"unknown source module FQN: {re.escape(candidate_fqn)}",
    ):
        analyze_vgf_quantization_sensitivity(
            source_model=model,
            fp32_model=exported,
            baseline_quantized_model=exported,
            baseline_assessment=assessment,
            representative_inputs=[sample],
            build_mixed_quantized_model=build_mixed,
            candidate_module_fqns=(candidate_fqn,),
            parent_levels=parent_levels,
        )

    assert built_modules == []
