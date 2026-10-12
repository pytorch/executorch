# Copyright 2026 Arm Limited and/or its affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.
"""Assess VGF quantization quality across an exported FX graph.

VGF v1 uses the most specific (innermost) module FQN carried by
``node.meta["nn_module_stack"]`` as its preferred identity. The assessment
compares leaf-module outputs between an FP32 exported graph and a quantized
exported graph over representative inputs.

This module deliberately does not try to establish universal provenance across
backends or arbitrary graph rewrites. FX node names/targets and debug handles
are retained only as additional metadata.

"""

from __future__ import annotations

import math
from collections import defaultdict, deque
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import torch

from executorch.backends.arm.vgf.quantization_quality import (
    compute_vgf_quantization_metrics,
    VgfQuantizationMetrics,
)
from torch.export import ExportedProgram
from torch.fx import GraphModule, Interpreter, Node


GraphInput = ExportedProgram | GraphModule


@dataclass(frozen=True)
class VgfGraphNodeMetadata:
    """FX provenance retained for one aligned module occurrence."""

    fp32_fx_node: str
    fp32_target: str
    fp32_capture_node: str
    fp32_debug_handles: tuple[int, ...]
    quantized_fx_node: str
    quantized_target: str
    quantized_capture_node: str
    quantized_debug_handles: tuple[int, ...]


@dataclass(frozen=True)
class VgfModuleQuantizationAssessment:
    """Aggregated quantization assessment for one module FQN."""

    module_fqn: str
    metrics: VgfQuantizationMetrics
    occurrences: int
    samples: int
    compared_numel: int
    nodes: tuple[VgfGraphNodeMetadata, ...]
    precision: str | None = None

    def to_dict(self, *, include_metadata: bool = False) -> dict[str, Any]:
        """Return the public JSON-friendly representation.

        The compact form intentionally matches the VGF-v1 user-facing output:
        MSE, SNR and cosine similarity keyed by module FQN. More detailed error
        metrics plus FX/debug metadata can be requested for diagnostics.

        """
        result: dict[str, Any] = {
            "mse": self.metrics.mse,
            "snr_db": self.metrics.snr_db,
            "cosine": self.metrics.cosine_similarity,
        }
        if include_metadata:
            result.update(
                {
                    "max_abs_error": self.metrics.max_abs_error,
                    "relative_error": self.metrics.relative_error,
                    "reference_min": self.metrics.reference_min,
                    "reference_max": self.metrics.reference_max,
                    "quantized_min": self.metrics.quantized_min,
                    "quantized_max": self.metrics.quantized_max,
                    "precision": self.precision,
                    "saturation_ratio": self.metrics.saturation_ratio,
                    "clipping_ratio": self.metrics.clipping_ratio,
                    "metadata": {
                        "occurrences": self.occurrences,
                        "samples": self.samples,
                        "compared_numel": self.compared_numel,
                        "nodes": [
                            {
                                "fp32_fx_node": node.fp32_fx_node,
                                "fp32_target": node.fp32_target,
                                "fp32_capture_node": node.fp32_capture_node,
                                "fp32_debug_handles": list(node.fp32_debug_handles),
                                "quantized_fx_node": node.quantized_fx_node,
                                "quantized_target": node.quantized_target,
                                "quantized_capture_node": node.quantized_capture_node,
                                "quantized_debug_handles": list(
                                    node.quantized_debug_handles
                                ),
                            }
                            for node in self.nodes
                        ],
                    },
                }
            )
        return result


def _quantization_quality_sort_key(
    item: tuple[str, VgfModuleQuantizationAssessment],
) -> tuple[float, float, float, str]:
    """Return a deterministic worst-to-best sort key for one module assessment.

    Lower SNR and cosine indicate worse agreement. Higher MSE breaks remaining
    ties. NaN values are treated as worst so invalid/degenerate measurements are
    surfaced rather than hidden at the end of the report.

    """
    module_fqn, assessment = item
    metrics = assessment.metrics
    snr_db = -math.inf if math.isnan(metrics.snr_db) else metrics.snr_db
    cosine = (
        -math.inf
        if math.isnan(metrics.cosine_similarity)
        else metrics.cosine_similarity
    )
    mse = math.inf if math.isnan(metrics.mse) else metrics.mse
    return (snr_db, cosine, -mse, module_fqn)


@dataclass(frozen=True)
class VgfGraphQuantizationAssessment:
    """Graph-wide VGF quantization assessment keyed by module FQN."""

    modules: dict[str, VgfModuleQuantizationAssessment]
    fp32_only_module_fqns: tuple[str, ...]
    quantized_only_module_fqns: tuple[str, ...]
    skipped_module_fqns: dict[str, str]

    def to_dict(
        self,
        *,
        include_metadata: bool = False,
        sort_by_quality: bool = False,
    ) -> dict[str, dict[str, Any]]:
        """Return module-FQN keyed assessment results.

        Args:
            include_metadata: Include extended metrics and FX/debug provenance.
            sort_by_quality: If true, order modules from worst to best numerical
                agreement using SNR, cosine similarity, then MSE. The default
                preserves graph/insertion order for backward compatibility.

        """
        module_items = (
            sorted(self.modules.items(), key=_quantization_quality_sort_key)
            if sort_by_quality
            else self.modules.items()
        )
        return {
            module_fqn: assessment.to_dict(include_metadata=include_metadata)
            for module_fqn, assessment in module_items
        }


@dataclass(frozen=True)
class _AlignedNode:
    fp32_semantic: Node
    fp32_capture: Node
    quantized_semantic: Node
    quantized_capture: Node


class _TensorCaptureInterpreter(Interpreter):
    """Execute an FX graph while retaining selected intermediate values."""

    def __init__(self, module: GraphModule, capture_node_names: set[str]) -> None:
        super().__init__(module)
        self._capture_node_names = capture_node_names
        self.captured: dict[str, Any] = {}

    def run_node(self, node: Node) -> Any:
        value = super().run_node(node)
        if node.name in self._capture_node_names:
            self.captured[node.name] = value
        return value


def _as_graph_module(model: GraphInput) -> GraphModule:
    if isinstance(model, ExportedProgram):
        return model.module()
    if isinstance(model, GraphModule):
        return model
    raise TypeError(
        "expected torch.export.ExportedProgram or torch.fx.GraphModule, "
        f"got {type(model).__name__}"
    )


def _stack_fqns(node: Node) -> tuple[str, ...]:
    stack = node.meta.get("nn_module_stack")
    if not isinstance(stack, Mapping):
        return ()

    fqns: list[str] = []
    for module_meta in stack.values():
        if (
            isinstance(module_meta, (tuple, list))
            and len(module_meta) > 0
            and isinstance(module_meta[0], str)
            and module_meta[0]
        ):
            fqns.append(module_meta[0])
    return tuple(fqns)


def _module_fqn(node: Node) -> str | None:
    fqns = _stack_fqns(node)
    return fqns[-1] if fqns else None


def _ordered_leaf_module_fqns(graph_module: GraphModule) -> tuple[str, ...]:
    ordered: list[str] = []
    seen: set[str] = set()
    all_fqns: set[str] = set()

    for node in graph_module.graph.nodes:
        for fqn in _stack_fqns(node):
            all_fqns.add(fqn)
            if fqn not in seen:
                seen.add(fqn)
                ordered.append(fqn)

    return tuple(
        fqn
        for fqn in ordered
        if not any(other.startswith(f"{fqn}.") for other in all_fqns if other != fqn)
    )


def _is_semantic_node(node: Node) -> bool:
    # Exported programs are normally call_function graphs. call_method is kept
    # for robustness, while call_module is deliberately excluded because
    # ExportedProgram.module() may add helper modules such as _guards_fn that
    # inherit misleading nn_module_stack metadata.
    return node.op in ("call_function", "call_method") and _module_fqn(node) is not None


def _is_quantize_node(node: Node) -> bool:
    return "quantized_decomposed.quantize_" in str(node.target)


def _is_dequantize_node(node: Node) -> bool:
    return "quantized_decomposed.dequantize_" in str(node.target)


def _is_quantization_plumbing(node: Node) -> bool:
    return _module_fqn(node) is None and (
        _is_quantize_node(node) or _is_dequantize_node(node)
    )


def _has_same_fqn_descendant_through_quantization(node: Node, fqn: str) -> bool:
    """Return whether this node feeds another semantic node for the same FQN.

    PT2E may place unattributed Q/DQ nodes between two semantic nodes that still
    belong to one leaf module. Traversing only this known quantization plumbing
    prevents an internal operator from being mistaken for a module output.

    """
    queue: deque[Node] = deque(node.users)
    visited: set[Node] = set()
    while queue:
        user = queue.popleft()
        if user in visited:
            continue
        visited.add(user)

        user_fqn = _module_fqn(user)
        if user_fqn == fqn and _is_semantic_node(user):
            return True
        if _is_quantization_plumbing(user):
            queue.extend(user.users)
    return False


def _terminal_semantic_nodes(
    graph_module: GraphModule,
    module_fqn: str,
) -> tuple[Node, ...]:
    candidates = [
        node
        for node in graph_module.graph.nodes
        if _is_semantic_node(node) and _module_fqn(node) == module_fqn
    ]
    return tuple(
        node
        for node in candidates
        if not _has_same_fqn_descendant_through_quantization(node, module_fqn)
    )


@dataclass(frozen=True)
class _QuantizationBoundaryInfo:
    """Static quantization metadata for one semantic output boundary."""

    precision: str | None
    clip_min: float | None
    clip_max: float | None


def _output_qdq_nodes(node: Node) -> tuple[Node | None, Node | None]:
    """Return a direct quantize/dequantize pair after ``node`` when unique."""
    users = list(node.users)
    if len(users) != 1 or not _is_quantize_node(users[0]):
        return None, None

    quantize = users[0]
    quantize_users = list(quantize.users)
    if len(quantize_users) != 1 or not _is_dequantize_node(quantize_users[0]):
        return None, None
    return quantize, quantize_users[0]


def _precision_label(dtype: Any) -> str | None:
    """Return a short precision label such as INT8 for a torch dtype."""
    if dtype is None:
        return None
    label = str(dtype).removeprefix("torch.").upper()
    return label if label else None


def _resolve_scalar_arg(value: Any, graph_module: GraphModule) -> float | None:
    """Resolve a scalar literal or get_attr node used by a quantize op."""
    resolved = value
    if isinstance(value, Node) and value.op == "get_attr":
        try:
            resolved = graph_module
            for part in str(value.target).split("."):
                resolved = getattr(resolved, part)
        except AttributeError:
            return None

    if isinstance(resolved, (int, float)):
        return float(resolved)
    if isinstance(resolved, torch.Tensor) and resolved.numel() == 1:
        return float(resolved.detach().cpu().item())
    return None


def _quantization_boundary_info(
    node: Node,
    graph_module: GraphModule,
) -> _QuantizationBoundaryInfo:
    """Read precision and dequantized clip bounds from an output Q/DQ pair.

    PT2E activation quantization normally emits
    ``quantize_per_tensor(input, scale, zero_point, qmin, qmax, dtype)``.
    Those arguments provide the representable floating-point range needed by
    the existing saturation/clipping metric implementation. For other Q/DQ
    forms, precision is still reported but scalar clipping bounds are omitted.

    """
    quantize, _ = _output_qdq_nodes(node)
    if quantize is None:
        return _QuantizationBoundaryInfo(None, None, None)

    args = quantize.args
    precision = _precision_label(args[-1] if args else None)
    if "quantize_per_tensor" not in str(quantize.target) or len(args) < 6:
        return _QuantizationBoundaryInfo(precision, None, None)

    scale = _resolve_scalar_arg(args[1], graph_module)
    zero_point = _resolve_scalar_arg(args[2], graph_module)
    quant_min = _resolve_scalar_arg(args[3], graph_module)
    quant_max = _resolve_scalar_arg(args[4], graph_module)
    if None in (scale, zero_point, quant_min, quant_max):
        return _QuantizationBoundaryInfo(precision, None, None)

    assert scale is not None
    assert zero_point is not None
    assert quant_min is not None
    assert quant_max is not None

    # Match the float32 arithmetic used by the Q/DQ graph so the existing
    # exact-bound saturation check remains stable after promotion to float64.
    clip_min = torch.tensor(
        (quant_min - zero_point) * scale,
        dtype=torch.float32,
    ).item()
    clip_max = torch.tensor(
        (quant_max - zero_point) * scale,
        dtype=torch.float32,
    ).item()
    return _QuantizationBoundaryInfo(precision, clip_min, clip_max)


def _common_quantization_boundary_info(
    aligned_nodes: tuple[_AlignedNode, ...],
    quantized_graph: GraphModule,
) -> _QuantizationBoundaryInfo:
    """Return common module-level precision and clip bounds when unambiguous."""
    infos = [
        _quantization_boundary_info(aligned.quantized_semantic, quantized_graph)
        for aligned in aligned_nodes
    ]
    precisions = {info.precision for info in infos if info.precision is not None}
    if len(precisions) == 1:
        precision = next(iter(precisions))
    elif len(precisions) > 1:
        precision = "MIXED"
    else:
        precision = None

    bounds = [
        (info.clip_min, info.clip_max)
        for info in infos
        if info.clip_min is not None and info.clip_max is not None
    ]
    if len(bounds) != len(infos) or not bounds:
        return _QuantizationBoundaryInfo(precision, None, None)

    first_min, first_max = bounds[0]
    assert first_min is not None and first_max is not None
    for clip_min, clip_max in bounds[1:]:
        assert clip_min is not None and clip_max is not None
        if not (
            math.isclose(clip_min, first_min, rel_tol=0.0, abs_tol=1.0e-12)
            and math.isclose(clip_max, first_max, rel_tol=0.0, abs_tol=1.0e-12)
        ):
            return _QuantizationBoundaryInfo(precision, None, None)

    return _QuantizationBoundaryInfo(precision, first_min, first_max)


def _capture_node_after_output_qdq(node: Node) -> Node:
    """Include a direct output Q/DQ boundary when it is unambiguous.

    The semantic FX node remains the provenance identity. When PT2E inserts a
    unique quantize -> dequantize pair immediately after it, comparison uses the
    dequantized value because that is the activation consumed by the rest of the
    quantized graph.

    """
    _, dequantize = _output_qdq_nodes(node)
    return dequantize if dequantize is not None else node


def _debug_handles(node: Node) -> tuple[int, ...]:
    value = node.meta.get("debug_handle")
    if isinstance(value, int):
        return (value,)
    if isinstance(value, (tuple, list)) and all(
        isinstance(item, int) for item in value
    ):
        return tuple(value)
    return ()


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
        raise TypeError("captured FX value does not contain a tensor")
    if len(tensors) == 1:
        return tensors[0]
    return torch.cat(tensors)


def _normalize_representative_inputs(
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


def _align_module_nodes(
    fp32_graph: GraphModule,
    quantized_graph: GraphModule,
    module_fqn: str,
) -> tuple[tuple[_AlignedNode, ...] | None, str | None]:
    fp32_nodes = _terminal_semantic_nodes(fp32_graph, module_fqn)
    quantized_nodes = _terminal_semantic_nodes(quantized_graph, module_fqn)

    if not fp32_nodes:
        return None, "no tensor-producing FP32 semantic node found"
    if not quantized_nodes:
        return None, "no tensor-producing quantized semantic node found"
    if len(fp32_nodes) != len(quantized_nodes):
        return None, (
            "module FQN maps to different numbers of terminal FX nodes: "
            f"FP32={len(fp32_nodes)}, quantized={len(quantized_nodes)}"
        )

    return (
        tuple(
            _AlignedNode(
                fp32_semantic=fp32_node,
                fp32_capture=_capture_node_after_output_qdq(fp32_node),
                quantized_semantic=quantized_node,
                quantized_capture=_capture_node_after_output_qdq(quantized_node),
            )
            for fp32_node, quantized_node in zip(fp32_nodes, quantized_nodes)
        ),
        None,
    )


def _run_capture(
    graph_module: GraphModule,
    capture_names: set[str],
    args: tuple[Any, ...],
) -> dict[str, Any]:
    interpreter = _TensorCaptureInterpreter(graph_module, capture_names)
    with torch.no_grad():
        interpreter.run(*args)
    return interpreter.captured


def _compare_leaf_module_fqns(
    fp32_graph: GraphModule,
    quantized_graph: GraphModule,
    *,
    strict: bool,
) -> tuple[tuple[str, ...], tuple[str, ...], tuple[str, ...]]:
    """Return shared, FP32-only, and quantized-only leaf module FQNs."""
    fp32_leaf_fqns = _ordered_leaf_module_fqns(fp32_graph)
    quantized_leaf_fqns = _ordered_leaf_module_fqns(quantized_graph)
    fp32_leaf_set = set(fp32_leaf_fqns)
    quantized_leaf_set = set(quantized_leaf_fqns)

    fp32_only = tuple(fqn for fqn in fp32_leaf_fqns if fqn not in quantized_leaf_set)
    quantized_only = tuple(
        fqn for fqn in quantized_leaf_fqns if fqn not in fp32_leaf_set
    )
    shared_fqns = tuple(fqn for fqn in fp32_leaf_fqns if fqn in quantized_leaf_set)

    if strict and (fp32_only or quantized_only):
        raise ValueError(
            "module FQN mismatch between FP32 and quantized graphs: "
            f"fp32_only={fp32_only}, quantized_only={quantized_only}"
        )
    if not shared_fqns:
        raise ValueError(
            "FP32 and quantized graphs have no shared leaf module FQNs in "
            'node.meta["nn_module_stack"]'
        )

    return shared_fqns, fp32_only, quantized_only


def _align_shared_module_fqns(
    fp32_graph: GraphModule,
    quantized_graph: GraphModule,
    shared_fqns: tuple[str, ...],
    *,
    strict: bool,
) -> tuple[dict[str, tuple[_AlignedNode, ...]], dict[str, str]]:
    """Align terminal FX nodes for each shared module FQN."""
    aligned_by_fqn: dict[str, tuple[_AlignedNode, ...]] = {}
    skipped: dict[str, str] = {}

    for fqn in shared_fqns:
        aligned, reason = _align_module_nodes(fp32_graph, quantized_graph, fqn)
        if aligned is None:
            skipped[fqn] = reason or "unable to align module FQN"
        else:
            aligned_by_fqn[fqn] = aligned

    if strict and skipped:
        raise ValueError(f"could not align module FQNs: {skipped}")
    if not aligned_by_fqn:
        raise ValueError("no shared module FQNs could be aligned to FX outputs")

    return aligned_by_fqn, skipped


def _capture_names(
    aligned_by_fqn: dict[str, tuple[_AlignedNode, ...]],
    *,
    fp32: bool,
) -> set[str]:
    """Collect the FX node names whose runtime values should be captured."""
    if fp32:
        return {
            aligned.fp32_capture.name
            for aligned_nodes in aligned_by_fqn.values()
            for aligned in aligned_nodes
        }
    return {
        aligned.quantized_capture.name
        for aligned_nodes in aligned_by_fqn.values()
        for aligned in aligned_nodes
    }


def _capture_aligned_values(
    fp32_graph: GraphModule,
    quantized_graph: GraphModule,
    aligned_by_fqn: dict[str, tuple[_AlignedNode, ...]],
    samples: tuple[tuple[Any, ...], ...],
) -> tuple[
    dict[str, list[torch.Tensor]],
    dict[str, list[torch.Tensor]],
    dict[str, str],
]:
    """Run representative samples and collect aligned intermediate tensors."""
    fp32_capture_names = _capture_names(aligned_by_fqn, fp32=True)
    quantized_capture_names = _capture_names(aligned_by_fqn, fp32=False)
    fp32_values: dict[str, list[torch.Tensor]] = defaultdict(list)
    quantized_values: dict[str, list[torch.Tensor]] = defaultdict(list)
    runtime_skip: dict[str, str] = {}

    for args in samples:
        fp32_captured = _run_capture(fp32_graph, fp32_capture_names, args)
        quantized_captured = _run_capture(
            quantized_graph, quantized_capture_names, args
        )
        _append_captured_sample(
            aligned_by_fqn,
            fp32_captured,
            quantized_captured,
            fp32_values,
            quantized_values,
            runtime_skip,
        )

    return fp32_values, quantized_values, runtime_skip


def _append_captured_sample(
    aligned_by_fqn: dict[str, tuple[_AlignedNode, ...]],
    fp32_captured: dict[str, Any],
    quantized_captured: dict[str, Any],
    fp32_values: dict[str, list[torch.Tensor]],
    quantized_values: dict[str, list[torch.Tensor]],
    runtime_skip: dict[str, str],
) -> None:
    """Append one sample's aligned values."""
    for fqn, aligned_nodes in aligned_by_fqn.items():
        if fqn in runtime_skip:
            continue
        try:
            for aligned in aligned_nodes:
                fp32_tensor = _flatten_tensor_output(
                    fp32_captured[aligned.fp32_capture.name]
                )
                quantized_tensor = _flatten_tensor_output(
                    quantized_captured[aligned.quantized_capture.name]
                )
                if fp32_tensor.shape != quantized_tensor.shape:
                    raise ValueError(
                        "captured output shapes differ: "
                        f"FP32={tuple(fp32_tensor.shape)}, "
                        f"quantized={tuple(quantized_tensor.shape)}"
                    )
                fp32_values[fqn].append(fp32_tensor)
                quantized_values[fqn].append(quantized_tensor)
        except (KeyError, TypeError, ValueError) as exc:
            runtime_skip[fqn] = str(exc)
            fp32_values.pop(fqn, None)
            quantized_values.pop(fqn, None)


def _node_metadata(
    aligned_nodes: tuple[_AlignedNode, ...],
) -> tuple[VgfGraphNodeMetadata, ...]:
    """Build diagnostic provenance for aligned FX nodes."""
    return tuple(
        VgfGraphNodeMetadata(
            fp32_fx_node=aligned.fp32_semantic.name,
            fp32_target=str(aligned.fp32_semantic.target),
            fp32_capture_node=aligned.fp32_capture.name,
            fp32_debug_handles=_debug_handles(aligned.fp32_semantic),
            quantized_fx_node=aligned.quantized_semantic.name,
            quantized_target=str(aligned.quantized_semantic.target),
            quantized_capture_node=aligned.quantized_capture.name,
            quantized_debug_handles=_debug_handles(aligned.quantized_semantic),
        )
        for aligned in aligned_nodes
    )


def _build_module_assessments(
    shared_fqns: tuple[str, ...],
    aligned_by_fqn: dict[str, tuple[_AlignedNode, ...]],
    skipped: dict[str, str],
    fp32_values: dict[str, list[torch.Tensor]],
    quantized_values: dict[str, list[torch.Tensor]],
    quantized_graph: GraphModule,
    *,
    sample_count: int,
) -> dict[str, VgfModuleQuantizationAssessment]:
    """Aggregate representative-sample tensors into per-module metrics."""
    modules: dict[str, VgfModuleQuantizationAssessment] = {}

    for fqn in shared_fqns:
        aligned_nodes = aligned_by_fqn.get(fqn)
        if aligned_nodes is None or fqn in skipped:
            continue

        reference = torch.cat(fp32_values[fqn])
        quantized = torch.cat(quantized_values[fqn])
        boundary = _common_quantization_boundary_info(
            aligned_nodes,
            quantized_graph,
        )
        modules[fqn] = VgfModuleQuantizationAssessment(
            module_fqn=fqn,
            metrics=compute_vgf_quantization_metrics(
                reference,
                quantized,
                clip_min=boundary.clip_min,
                clip_max=boundary.clip_max,
            ),
            occurrences=len(aligned_nodes),
            samples=sample_count,
            compared_numel=reference.numel(),
            nodes=_node_metadata(aligned_nodes),
            precision=boundary.precision,
        )

    return modules


def assess_vgf_quantization_across_graph(
    fp32_model: GraphInput,
    quantized_model: GraphInput,
    representative_inputs: Sequence[tuple[Any, ...] | torch.Tensor],
    *,
    strict: bool = False,
) -> VgfGraphQuantizationAssessment:
    """Compare FP32 and quantized exported graphs using module FQN identity.

    Args:
        fp32_model: FP32 ExportedProgram or runnable exported GraphModule.
        quantized_model: Quantized ExportedProgram or runnable converted GraphModule.
        representative_inputs: Positional input samples. A single-input model may
            provide tensors directly; multi-input models should provide tuples.
        strict: If true, fail on unmatched/skipped module FQNs instead of
            returning diagnostics for the comparable intersection.

    Returns:
        VgfGraphQuantizationAssessment keyed by leaf module FQN.

    Raises:
        ValueError: If no representative samples are supplied, no comparable
            module FQNs exist, or strict mode encounters provenance mismatch.

    """
    fp32_graph = _as_graph_module(fp32_model)
    quantized_graph = _as_graph_module(quantized_model)
    samples = _normalize_representative_inputs(representative_inputs)

    shared_fqns, fp32_only, quantized_only = _compare_leaf_module_fqns(
        fp32_graph,
        quantized_graph,
        strict=strict,
    )
    aligned_by_fqn, skipped = _align_shared_module_fqns(
        fp32_graph,
        quantized_graph,
        shared_fqns,
        strict=strict,
    )
    fp32_values, quantized_values, runtime_skip = _capture_aligned_values(
        fp32_graph,
        quantized_graph,
        aligned_by_fqn,
        samples,
    )

    skipped.update(runtime_skip)
    if strict and runtime_skip:
        raise ValueError(f"could not compare aligned module FQNs: {runtime_skip}")

    modules = _build_module_assessments(
        shared_fqns,
        aligned_by_fqn,
        skipped,
        fp32_values,
        quantized_values,
        quantized_graph,
        sample_count=len(samples),
    )
    if not modules:
        raise ValueError(f"all comparable module FQNs were skipped: {skipped}")

    return VgfGraphQuantizationAssessment(
        modules=modules,
        fp32_only_module_fqns=fp32_only,
        quantized_only_module_fqns=quantized_only,
        skipped_module_fqns=skipped,
    )
