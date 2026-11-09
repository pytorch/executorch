# Copyright 2026 Arm Limited and/or its affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.
"""Sensitivity analysis for VGF quantization.

The graph-wide quantization assessment answers where FP32 and quantized
intermediates disagree. This module answers a different question: whether
quantizing a suspicious source module materially contributes to downstream
numerical error.

For each selected source module, a caller-provided builder rebuilds the
quantized model with that module left in floats. The mixed model is compared
with the baseline using the same representative inputs. The analyzer also re-
runs the graph assessment for the mixed model and checks that the FX subgraph
belonging to the requested source module actually changed.

The first version intentionally performs independent one-module-at-a-time
changes. It does not search combinations or claim task-level accuracy
improvements.

"""

from __future__ import annotations

import math
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import torch

from executorch.backends.arm.vgf.quantization_assessment import (
    assess_vgf_quantization_across_graph,
    GraphInput,
    VgfGraphQuantizationAssessment,
)
from executorch.backends.arm.vgf.quantization_quality import (
    compute_vgf_quantization_metrics,
    VgfQuantizationMetrics,
)
from torch.export import ExportedProgram
from torch.fx import GraphModule, Node


MixedQuantizedModelBuilder = Callable[[str], GraphInput]
OutputSelector = Callable[[Any], Any]


def _json_safe_float(value: float | None) -> float | None:
    """Return None for non-finite values so report dictionaries are valid JSON.

    Analysis keeps NaN and infinite values internally because they carry useful
    metric semantics. JSON-facing dictionaries use null for those values so
    standards-compliant JSON encoders can consume the report.

    """
    if value is None or not math.isfinite(value):
        return None
    return value


@dataclass(frozen=True)
class VgfQuantizationSensitivityCandidate:
    """One source module selected for a float-rescue experiment."""

    module_fqn: str
    assessment_fqn: str
    local_snr_db: float
    local_mse: float
    local_cosine: float

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable representation of this candidate."""
        return {
            "module_fqn": self.module_fqn,
            "assessment_fqn": self.assessment_fqn,
            "local_snr_db": _json_safe_float(self.local_snr_db),
            "local_mse": _json_safe_float(self.local_mse),
            "local_cosine": _json_safe_float(self.local_cosine),
        }


@dataclass(frozen=True)
class VgfQuantizationSensitivityOutputResult:
    """Rescue metrics for one stable final-output view."""

    name: str
    baseline_metrics: VgfQuantizationMetrics
    mixed_metrics: VgfQuantizationMetrics
    snr_rescue_db: float
    mse_reduction_ratio: float
    cosine_delta: float

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable representation of this output result."""
        return {
            "baseline": _metrics_to_dict(self.baseline_metrics),
            "mixed": _metrics_to_dict(self.mixed_metrics),
            "snr_rescue_db": _json_safe_float(self.snr_rescue_db),
            "mse_reduction_ratio": _json_safe_float(self.mse_reduction_ratio),
            "cosine_delta": _json_safe_float(self.cosine_delta),
        }


@dataclass(frozen=True)
class VgfQuantizationSensitivityIntervention:
    """Result after keeping one source module in floating point."""

    candidate: VgfQuantizationSensitivityCandidate
    mixed_local_metrics: VgfQuantizationMetrics | None
    local_snr_rescue_db: float | None
    local_mse_reduction_ratio: float | None
    local_cosine_delta: float | None
    intervention_verified: bool
    baseline_qdq_input_edges: int
    mixed_qdq_input_edges: int
    output_results: tuple[VgfQuantizationSensitivityOutputResult, ...]
    primary_output_name: str

    def output_result(self, name: str) -> VgfQuantizationSensitivityOutputResult:
        """Return rescue metrics for the named output view."""
        for result in self.output_results:
            if result.name == name:
                return result
        raise KeyError(name)

    @property
    def final_metrics(self) -> VgfQuantizationMetrics:
        """Backward-compatible primary mixed-output metrics."""
        return self.output_result(self.primary_output_name).mixed_metrics

    @property
    def snr_rescue_db(self) -> float:
        """Backward-compatible primary SNR rescue."""
        return self.output_result(self.primary_output_name).snr_rescue_db

    @property
    def mse_reduction_ratio(self) -> float:
        """Backward-compatible primary MSE reduction."""
        return self.output_result(self.primary_output_name).mse_reduction_ratio

    @property
    def cosine_delta(self) -> float:
        """Backward-compatible primary cosine change."""
        return self.output_result(self.primary_output_name).cosine_delta

    def best_output_rescue_db(self) -> float:
        """Return the largest finite SNR rescue across output views."""
        finite = [
            result.snr_rescue_db
            for result in self.output_results
            if not math.isnan(result.snr_rescue_db)
        ]
        return max(finite, default=-math.inf)

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable representation of this intervention."""
        result = {
            **self.candidate.to_dict(),
            "intervention_verified": self.intervention_verified,
            "baseline_qdq_input_edges": self.baseline_qdq_input_edges,
            "mixed_qdq_input_edges": self.mixed_qdq_input_edges,
            "target": {
                "baseline": {
                    "snr_db": _json_safe_float(self.candidate.local_snr_db),
                    "mse": _json_safe_float(self.candidate.local_mse),
                    "cosine": _json_safe_float(self.candidate.local_cosine),
                },
                "mixed": (
                    None
                    if self.mixed_local_metrics is None
                    else _metrics_to_dict(self.mixed_local_metrics)
                ),
                "snr_rescue_db": _json_safe_float(self.local_snr_rescue_db),
                "mse_reduction_ratio": _json_safe_float(self.local_mse_reduction_ratio),
                "cosine_delta": _json_safe_float(self.local_cosine_delta),
            },
            "outputs": {
                output.name: output.to_dict() for output in self.output_results
            },
        }

        # Preserve the v1 primary-output keys for existing consumers.
        result.update(
            {
                "final_snr_db": _json_safe_float(self.final_metrics.snr_db),
                "final_mse": _json_safe_float(self.final_metrics.mse),
                "final_cosine": _json_safe_float(self.final_metrics.cosine_similarity),
                "snr_rescue_db": _json_safe_float(self.snr_rescue_db),
                "mse_reduction_ratio": _json_safe_float(self.mse_reduction_ratio),
                "cosine_delta": _json_safe_float(self.cosine_delta),
            }
        )
        return result


@dataclass(frozen=True)
class VgfQuantizationSensitivityReport:
    """Baseline and one-module-at-a-time float-rescue results."""

    baseline_output_metrics: Mapping[str, VgfQuantizationMetrics]
    interventions: tuple[VgfQuantizationSensitivityIntervention, ...]
    primary_output_name: str

    @property
    def baseline_final_metrics(self) -> VgfQuantizationMetrics:
        """Backward-compatible primary baseline-output metrics."""
        return self.baseline_output_metrics[self.primary_output_name]

    def ranked_by_rescue(self) -> tuple[VgfQuantizationSensitivityIntervention, ...]:
        """Return interventions from largest to smallest output rescue."""

        def key(item: VgfQuantizationSensitivityIntervention) -> tuple[float, str]:
            rescue = item.best_output_rescue_db()
            sortable = -math.inf if math.isnan(rescue) else rescue
            return (-sortable, item.candidate.module_fqn)

        return tuple(sorted(self.interventions, key=key))

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable representation of this report."""
        return {
            "primary_output": self.primary_output_name,
            "baseline_outputs": {
                name: _metrics_to_dict(metrics)
                for name, metrics in self.baseline_output_metrics.items()
            },
            # Preserve the v1 field for existing report readers.
            "baseline_final": _metrics_to_dict(self.baseline_final_metrics),
            "interventions": [
                intervention.to_dict() for intervention in self.ranked_by_rescue()
            ],
        }


def _metrics_to_dict(
    metrics: VgfQuantizationMetrics,
) -> dict[str, float | None]:
    return {
        "snr_db": _json_safe_float(metrics.snr_db),
        "mse": _json_safe_float(metrics.mse),
        "cosine": _json_safe_float(metrics.cosine_similarity),
    }


def resolve_source_module_fqn(
    assessment_fqn: str,
    source_model: torch.nn.Module,
) -> str | None:
    """Map an exported assessment FQN to a quantizer-selectable source module.

    Some exported models, notably Ultralytics modules, carry synthetic stack
    names such as ``foo.forward.__self__.act``. For a sensitivity intervention
    we want the semantic source module (``foo``), so that Conv/BN/activation
    fusion remains coherent. Ordinary module FQNs are returned unchanged when
    they exist in ``source_model.named_modules()``.

    """
    module_names = {name for name, _ in source_model.named_modules() if name}

    synthetic_marker = ".forward.__self__."
    if synthetic_marker in assessment_fqn:
        source_prefix = assessment_fqn.split(synthetic_marker, maxsplit=1)[0]
        if source_prefix in module_names:
            return source_prefix

    candidate = assessment_fqn
    while candidate:
        if candidate in module_names:
            return candidate
        if "." not in candidate:
            break
        candidate = candidate.rsplit(".", maxsplit=1)[0]
    return None


def select_vgf_quantization_sensitivity_candidates(
    assessment: VgfGraphQuantizationAssessment,
    source_model: torch.nn.Module,
    *,
    limit: int = 5,
    max_snr_db: float | None = None,
) -> tuple[VgfQuantizationSensitivityCandidate, ...]:
    """Select worst unique source modules from a graph assessment."""
    if limit <= 0:
        raise ValueError("limit must be positive")

    ordered = sorted(
        assessment.modules.items(),
        key=lambda item: (
            (
                item[1].metrics.snr_db
                if not math.isnan(item[1].metrics.snr_db)
                else math.inf
            ),
            item[0],
        ),
    )
    selected: list[VgfQuantizationSensitivityCandidate] = []
    seen_modules: set[str] = set()

    for assessment_fqn, module_assessment in ordered:
        metrics = module_assessment.metrics
        # ``-inf`` is the worst possible SNR and is therefore a valid
        # sensitivity candidate. Only undefined or perfect-SNR entries are
        # unsuitable for automatic worst-first selection.
        if math.isnan(metrics.snr_db) or metrics.snr_db == math.inf:
            continue
        if max_snr_db is not None and metrics.snr_db > max_snr_db:
            continue

        module_fqn = resolve_source_module_fqn(assessment_fqn, source_model)
        if module_fqn is None or module_fqn in seen_modules:
            continue

        seen_modules.add(module_fqn)
        selected.append(
            VgfQuantizationSensitivityCandidate(
                module_fqn=module_fqn,
                assessment_fqn=assessment_fqn,
                local_snr_db=metrics.snr_db,
                local_mse=metrics.mse,
                local_cosine=metrics.cosine_similarity,
            )
        )
        if len(selected) == limit:
            break

    return tuple(selected)


def _candidate_for_module(
    module_fqn: str,
    assessment: VgfGraphQuantizationAssessment,
    source_model: torch.nn.Module,
) -> VgfQuantizationSensitivityCandidate:
    module_names = {name for name, _ in source_model.named_modules() if name}
    if module_fqn not in module_names:
        raise ValueError(f"unknown source module FQN: {module_fqn}")

    matches: list[tuple[str, Any]] = []
    for assessment_fqn, module_assessment in assessment.modules.items():
        resolved = resolve_source_module_fqn(assessment_fqn, source_model)
        if resolved is None:
            continue
        if resolved == module_fqn or resolved.startswith(f"{module_fqn}."):
            matches.append((assessment_fqn, module_assessment))

    if not matches:
        raise ValueError(
            f"source module {module_fqn!r} has no entry in the baseline assessment"
        )

    assessment_fqn, module_assessment = min(
        matches,
        key=lambda item: (
            (
                item[1].metrics.snr_db
                if not math.isnan(item[1].metrics.snr_db)
                else math.inf
            ),
            item[0],
        ),
    )
    metrics = module_assessment.metrics
    return VgfQuantizationSensitivityCandidate(
        module_fqn=module_fqn,
        assessment_fqn=assessment_fqn,
        local_snr_db=metrics.snr_db,
        local_mse=metrics.mse,
        local_cosine=metrics.cosine_similarity,
    )


def _validate_explicit_candidate_module_fqns(
    module_fqns: Sequence[str],
    source_model: torch.nn.Module,
) -> None:
    """Reject requested module FQNs that are not in the model."""
    module_names = {name for name, _ in source_model.named_modules() if name}
    for module_fqn in module_fqns:
        if module_fqn not in module_names:
            raise ValueError(f"unknown source module FQN: {module_fqn}")


def expand_module_fqns_with_parents(
    module_fqns: Sequence[str],
    source_model: torch.nn.Module,
    *,
    parent_levels: int,
) -> tuple[str, ...]:
    """Add source-module parents for hierarchical rescue experiments."""
    if parent_levels < 0:
        raise ValueError("parent_levels must be non-negative")
    module_names = {name for name, _ in source_model.named_modules() if name}
    expanded: list[str] = []
    seen: set[str] = set()

    for module_fqn in module_fqns:
        current = module_fqn
        for _ in range(parent_levels + 1):
            if current in module_names and current not in seen:
                seen.add(current)
                expanded.append(current)
            if "." not in current:
                break
            current = current.rsplit(".", maxsplit=1)[0]
    return tuple(expanded)


def _normalize_inputs(
    representative_inputs: Sequence[tuple[Any, ...] | torch.Tensor],
) -> tuple[tuple[Any, ...], ...]:
    if not representative_inputs:
        raise ValueError("representative_inputs must contain at least one sample")

    normalized: list[tuple[Any, ...]] = []
    for index, sample in enumerate(representative_inputs):
        if isinstance(sample, torch.Tensor):
            normalized.append((sample,))
        elif isinstance(sample, tuple):
            normalized.append(sample)
        else:
            raise TypeError(
                "each representative input sample must be a Tensor or tuple of "
                f"positional arguments; sample {index} is {type(sample).__name__}"
            )
    return tuple(normalized)


def _as_graph_module(model: GraphInput) -> GraphModule:
    if isinstance(model, ExportedProgram):
        return model.module()
    if isinstance(model, GraphModule):
        return model
    raise TypeError(
        "expected torch.export.ExportedProgram or torch.fx.GraphModule, "
        f"got {type(model).__name__}"
    )


def _flatten_tensor_output(value: Any) -> torch.Tensor:
    tensors: list[torch.Tensor] = []

    def visit(item: Any) -> None:
        if isinstance(item, torch.Tensor):
            tensors.append(item.detach().cpu().reshape(-1))
        elif isinstance(item, (tuple, list)):
            for child in item:
                visit(child)
        elif isinstance(item, Mapping):
            for child in item.values():
                visit(child)

    visit(value)
    if not tensors:
        raise TypeError("model output does not contain a tensor")
    if len(tensors) == 1:
        return tensors[0]
    return torch.cat(tensors)


def _normalize_output_selectors(
    output_selector: OutputSelector | None,
    output_selectors: Mapping[str, OutputSelector] | None,
) -> dict[str, OutputSelector | None]:
    if output_selector is not None and output_selectors is not None:
        raise ValueError("pass either output_selector or output_selectors, not both")
    if output_selectors is not None:
        if not output_selectors:
            raise ValueError("output_selectors must not be empty")
        return dict(output_selectors)
    return {"final": output_selector}


def _collect_output_views(
    model: GraphInput,
    samples: tuple[tuple[Any, ...], ...],
    selectors: Mapping[str, OutputSelector | None],
) -> dict[str, torch.Tensor]:
    graph_module = _as_graph_module(model)
    outputs: dict[str, list[torch.Tensor]] = {name: [] for name in selectors}
    with torch.no_grad():
        for args in samples:
            value = graph_module(*args)
            for name, selector in selectors.items():
                selected = value if selector is None else selector(value)
                outputs[name].append(_flatten_tensor_output(selected))
    return {name: torch.cat(values) for name, values in outputs.items()}


def compare_vgf_model_output_views(
    fp32_model: GraphInput,
    candidate_model: GraphInput,
    representative_inputs: Sequence[tuple[Any, ...] | torch.Tensor],
    *,
    output_selector: OutputSelector | None = None,
    output_selectors: Mapping[str, OutputSelector] | None = None,
) -> dict[str, VgfQuantizationMetrics]:
    """Compare one or more stable final-output views."""
    samples = _normalize_inputs(representative_inputs)
    selectors = _normalize_output_selectors(output_selector, output_selectors)
    reference = _collect_output_views(fp32_model, samples, selectors)
    candidate = _collect_output_views(candidate_model, samples, selectors)
    return {
        name: compute_vgf_quantization_metrics(reference[name], candidate[name])
        for name in selectors
    }


def compare_vgf_model_outputs(
    fp32_model: GraphInput,
    candidate_model: GraphInput,
    representative_inputs: Sequence[tuple[Any, ...] | torch.Tensor],
    *,
    output_selector: OutputSelector | None = None,
) -> VgfQuantizationMetrics:
    """Backward-compatible comparison of one selected final output."""
    return compare_vgf_model_output_views(
        fp32_model,
        candidate_model,
        representative_inputs,
        output_selector=output_selector,
    )["final"]


def _node_module_fqns(node: Node) -> tuple[str, ...]:
    stack = node.meta.get("nn_module_stack", {})
    if not isinstance(stack, Mapping):
        return ()
    result: list[str] = []
    for value in stack.values():
        if (
            isinstance(value, (tuple, list))
            and value
            and isinstance(value[0], str)
            and value[0]
        ):
            result.append(value[0])
    return tuple(result)


def _node_belongs_to_module(node: Node, module_fqn: str) -> bool:
    prefix = f"{module_fqn}."
    return any(
        fqn == module_fqn or fqn.startswith(prefix) for fqn in _node_module_fqns(node)
    )


def _module_graph_signature(
    model: GraphInput,
    module_fqn: str,
) -> tuple[tuple[str, tuple[tuple[str, str], ...]], ...]:
    """Return a structure-only signature for call nodes."""
    graph_module = _as_graph_module(model)
    signature: list[tuple[str, tuple[tuple[str, str], ...]]] = []
    for node in graph_module.graph.nodes:
        if node.op != "call_function" or not _node_belongs_to_module(node, module_fqn):
            continue
        inputs = tuple(
            (input_node.op, str(input_node.target))
            for input_node in node.all_input_nodes
        )
        signature.append((str(node.target), inputs))
    return tuple(signature)


def _is_qdq_node(node: Node) -> bool:
    if node.op != "call_function":
        return False
    target = str(node.target)
    return (
        "quantize_per_" in target
        or "dequantize_per_" in target
        or "quantize_affine" in target
        or "dequantize_affine" in target
    )


def _module_qdq_boundary_signature(
    model: GraphInput,
    module_fqn: str,
) -> tuple[tuple[str, tuple[str, ...], tuple[str, ...]], ...]:
    """Return direct Q/DQ boundaries on module inputs and outputs."""
    graph_module = _as_graph_module(model)
    signature: list[tuple[str, tuple[str, ...], tuple[str, ...]]] = []
    for node in graph_module.graph.nodes:
        if node.op != "call_function" or not _node_belongs_to_module(node, module_fqn):
            continue

        input_qdq = tuple(
            sorted(
                str(input_node.target)
                for input_node in node.all_input_nodes
                if _is_qdq_node(input_node)
            )
        )
        output_qdq = tuple(
            sorted(str(user.target) for user in node.users if _is_qdq_node(user))
        )
        signature.append((str(node.target), input_qdq, output_qdq))
    return tuple(signature)


def _module_intervention_verified(
    baseline_model: GraphInput,
    mixed_model: GraphInput,
    module_fqn: str,
) -> bool:
    """Return whether the requested module's quantization graph changed."""
    baseline_structure = _module_graph_signature(baseline_model, module_fqn)
    if not baseline_structure:
        return False

    mixed_structure = _module_graph_signature(mixed_model, module_fqn)
    if baseline_structure != mixed_structure:
        return True

    return _module_qdq_boundary_signature(
        baseline_model,
        module_fqn,
    ) != _module_qdq_boundary_signature(
        mixed_model,
        module_fqn,
    )


def _module_qdq_input_edges(model: GraphInput, module_fqn: str) -> int:
    graph_module = _as_graph_module(model)
    count = 0
    for node in graph_module.graph.nodes:
        if node.op != "call_function" or not _node_belongs_to_module(node, module_fqn):
            continue
        for input_node in node.all_input_nodes:
            target = str(input_node.target)
            if "dequantize_per_" in target or "dequantize_affine" in target:
                count += 1
    return count


def _snr_rescue(mixed_snr_db: float, baseline_snr_db: float) -> float:
    """Return the SNR improvement from baseline to mixed precision.

    ``-inf`` is a valid worst-case SNR. Improving it to any finite value or
    ``+inf`` therefore has ``+inf`` rescue. Matching infinities remain
    indeterminate, as do inputs containing NaN.

    """
    if math.isnan(mixed_snr_db) or math.isnan(baseline_snr_db):
        return math.nan
    if math.isinf(mixed_snr_db) and mixed_snr_db == baseline_snr_db:
        return math.nan
    return mixed_snr_db - baseline_snr_db


def _mse_reduction(mixed_mse: float, baseline_mse: float) -> float:
    if baseline_mse == 0.0:
        return 0.0 if mixed_mse == 0.0 else -math.inf
    return (baseline_mse - mixed_mse) / baseline_mse


def _find_mixed_local_metrics(
    candidate: VgfQuantizationSensitivityCandidate,
    mixed_assessment: VgfGraphQuantizationAssessment,
    source_model: torch.nn.Module,
) -> VgfQuantizationMetrics | None:
    exact = mixed_assessment.modules.get(candidate.assessment_fqn)
    if exact is not None:
        return exact.metrics

    matches: list[Any] = []
    for assessment_fqn, module_assessment in mixed_assessment.modules.items():
        resolved = resolve_source_module_fqn(assessment_fqn, source_model)
        if resolved is None:
            continue
        if resolved == candidate.module_fqn or resolved.startswith(
            f"{candidate.module_fqn}."
        ):
            matches.append(module_assessment)
    if not matches:
        return None
    return min(matches, key=lambda item: item.metrics.snr_db).metrics


def analyze_vgf_quantization_sensitivity(
    *,
    source_model: torch.nn.Module,
    fp32_model: GraphInput,
    baseline_quantized_model: GraphInput,
    baseline_assessment: VgfGraphQuantizationAssessment,
    representative_inputs: Sequence[tuple[Any, ...] | torch.Tensor],
    build_mixed_quantized_model: MixedQuantizedModelBuilder,
    candidate_module_fqns: Sequence[str] | None = None,
    candidate_limit: int = 5,
    candidate_max_snr_db: float | None = None,
    parent_levels: int = 0,
    output_selector: OutputSelector | None = None,
    output_selectors: Mapping[str, OutputSelector] | None = None,
    primary_output_name: str | None = None,
) -> VgfQuantizationSensitivityReport:
    """Run independent floating-point rescue experiments.

    ``build_mixed_quantized_model`` must rebuild and calibrate the same quantized
    model while applying ``set_module_name(candidate_fqn, None)`` (or an
    equivalent policy) to the requested source module.

    ``output_selectors`` should select stable, index-aligned outputs. Detection
    models should prefer raw head tensors over post-top-k/NMS detections.

    Explicit ``candidate_module_fqns`` are validated against ``source_model``
    before any parent expansion is applied.

    """
    samples = _normalize_inputs(representative_inputs)
    selectors = _normalize_output_selectors(output_selector, output_selectors)
    if primary_output_name is None:
        primary_output_name = next(iter(selectors))
    if primary_output_name not in selectors:
        raise ValueError(
            f"primary_output_name {primary_output_name!r} is not in output selectors"
        )

    baseline_reference = _collect_output_views(fp32_model, samples, selectors)
    baseline_candidate = _collect_output_views(
        baseline_quantized_model,
        samples,
        selectors,
    )
    baseline_outputs = {
        name: compute_vgf_quantization_metrics(
            baseline_reference[name],
            baseline_candidate[name],
        )
        for name in selectors
    }

    if candidate_module_fqns is None:
        base_candidates = select_vgf_quantization_sensitivity_candidates(
            baseline_assessment,
            source_model,
            limit=candidate_limit,
            max_snr_db=candidate_max_snr_db,
        )
        module_fqns = tuple(candidate.module_fqn for candidate in base_candidates)
    else:
        module_fqns = tuple(candidate_module_fqns)
        _validate_explicit_candidate_module_fqns(module_fqns, source_model)

    expanded_fqns = expand_module_fqns_with_parents(
        module_fqns,
        source_model,
        parent_levels=parent_levels,
    )
    candidates = tuple(
        _candidate_for_module(module_fqn, baseline_assessment, source_model)
        for module_fqn in expanded_fqns
    )

    interventions: list[VgfQuantizationSensitivityIntervention] = []
    for candidate in candidates:
        mixed_model = build_mixed_quantized_model(candidate.module_fqn)
        mixed_outputs = _collect_output_views(mixed_model, samples, selectors)
        output_results: list[VgfQuantizationSensitivityOutputResult] = []
        for name in selectors:
            mixed_metrics = compute_vgf_quantization_metrics(
                baseline_reference[name],
                mixed_outputs[name],
            )
            baseline_metrics = baseline_outputs[name]
            output_results.append(
                VgfQuantizationSensitivityOutputResult(
                    name=name,
                    baseline_metrics=baseline_metrics,
                    mixed_metrics=mixed_metrics,
                    snr_rescue_db=_snr_rescue(
                        mixed_metrics.snr_db,
                        baseline_metrics.snr_db,
                    ),
                    mse_reduction_ratio=_mse_reduction(
                        mixed_metrics.mse,
                        baseline_metrics.mse,
                    ),
                    cosine_delta=(
                        mixed_metrics.cosine_similarity
                        - baseline_metrics.cosine_similarity
                    ),
                )
            )

        mixed_assessment = assess_vgf_quantization_across_graph(
            fp32_model,
            mixed_model,
            samples,
            strict=False,
        )
        mixed_local = _find_mixed_local_metrics(
            candidate,
            mixed_assessment,
            source_model,
        )
        if mixed_local is None:
            local_snr_rescue = None
            local_mse_reduction = None
            local_cosine_delta = None
        else:
            local_snr_rescue = _snr_rescue(
                mixed_local.snr_db,
                candidate.local_snr_db,
            )
            local_mse_reduction = _mse_reduction(
                mixed_local.mse,
                candidate.local_mse,
            )
            local_cosine_delta = mixed_local.cosine_similarity - candidate.local_cosine

        baseline_qdq = _module_qdq_input_edges(
            baseline_quantized_model,
            candidate.module_fqn,
        )
        mixed_qdq = _module_qdq_input_edges(mixed_model, candidate.module_fqn)
        intervention_verified = _module_intervention_verified(
            baseline_quantized_model,
            mixed_model,
            candidate.module_fqn,
        )

        interventions.append(
            VgfQuantizationSensitivityIntervention(
                candidate=candidate,
                mixed_local_metrics=mixed_local,
                local_snr_rescue_db=local_snr_rescue,
                local_mse_reduction_ratio=local_mse_reduction,
                local_cosine_delta=local_cosine_delta,
                intervention_verified=intervention_verified,
                baseline_qdq_input_edges=baseline_qdq,
                mixed_qdq_input_edges=mixed_qdq,
                output_results=tuple(output_results),
                primary_output_name=primary_output_name,
            )
        )

    return VgfQuantizationSensitivityReport(
        baseline_output_metrics=baseline_outputs,
        interventions=tuple(interventions),
        primary_output_name=primary_output_name,
    )
