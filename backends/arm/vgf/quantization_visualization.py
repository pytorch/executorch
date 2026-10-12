# Copyright 2026 Arm Limited and/or its affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.
"""Visualize VGF quantization assessment results in Model Explorer.

The visual structure is the FP32 ``ExportedProgram`` used by the assessment.
Module FQNs are used as Model Explorer namespaces, and assessment provenance
maps metrics back to the corresponding FP32 FX nodes. This keeps the first
visualization version independent of TOSA/VGF runtime provenance.

Model Explorer is an optional Arm example dependency. Imports are intentionally
lazy so importing this module does not require Model Explorer to be installed.

"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

from executorch.backends.arm.vgf.quantization_assessment import (
    VgfGraphQuantizationAssessment,
)

from torch.export import ExportedProgram
from torch.fx import Node


@dataclass(frozen=True)
class VgfQuantizationVisualizationEntry:
    """Model Explorer-ready data for one assessed module FQN."""

    module_fqn: str
    fp32_node_ids: tuple[str, ...]
    precision: str
    mse: float
    snr_db: float
    cosine: float
    saturation_percent: float | None


def build_vgf_quantization_visualization_entries(
    assessment: VgfGraphQuantizationAssessment,
) -> tuple[VgfQuantizationVisualizationEntry, ...]:
    """Convert graph-assessment results into visualization entries."""
    entries: list[VgfQuantizationVisualizationEntry] = []
    for module_fqn, module in assessment.modules.items():
        saturation = module.metrics.saturation_ratio
        entries.append(
            VgfQuantizationVisualizationEntry(
                module_fqn=module_fqn,
                fp32_node_ids=tuple(node.fp32_fx_node for node in module.nodes),
                precision=module.precision or "UNKNOWN",
                mse=module.metrics.mse,
                snr_db=module.metrics.snr_db,
                cosine=module.metrics.cosine_similarity,
                saturation_percent=(None if saturation is None else 100.0 * saturation),
            )
        )
    return tuple(entries)


def _namespace_from_fqn(module_fqn: str) -> str:
    return module_fqn.replace(".", "/")


def _fx_node_module_fqn(node: Node) -> str | None:
    stack = node.meta.get("nn_module_stack")
    if not isinstance(stack, dict):
        return None

    fqns = [
        value[0]
        for value in stack.values()
        if isinstance(value, (tuple, list))
        and value
        and isinstance(value[0], str)
        and value[0]
    ]
    return fqns[-1] if fqns else None


def _format_number(value: float, digits: int) -> str:
    if math.isnan(value):
        return "nan"
    if value == math.inf:
        return "inf"
    if value == -math.inf:
        return "-inf"
    return f"{value:.{digits}g}"


def _entry_attributes(
    entry: VgfQuantizationVisualizationEntry,
) -> dict[str, str]:
    saturation = (
        "N/A"
        if entry.saturation_percent is None
        else f"{entry.saturation_percent:.2f}%"
    )
    return {
        "Module FQN": entry.module_fqn,
        "Precision": entry.precision,
        "MSE": _format_number(entry.mse, 6),
        "SNR": f"{_format_number(entry.snr_db, 5)} dB",
        "Cosine": _format_number(entry.cosine, 6),
        "Saturation": saturation,
    }


def _model_explorer_imports() -> tuple[Any, Any, Any, Any, Any]:
    try:
        from model_explorer import (  # type: ignore[import-not-found]
            config as model_explorer_config,
            consts,
            node_data_builder as ndb,
        )
        from model_explorer.graph_builder import (  # type: ignore[import-not-found]
            KeyValue,
        )
        from model_explorer.pytorch_exported_program_adater_impl import (  # type: ignore
            PytorchExportedProgramAdapterImpl,
        )
    except ImportError as exc:
        raise ImportError(
            "Model Explorer is required for VGF quantization visualization. "
            "Install the existing Arm Model Explorer dependencies with "
            "examples/arm/setup.sh --enable-model-explorer and make its "
            "model-explorer target directory available on PYTHONPATH."
        ) from exc

    return (
        model_explorer_config,
        consts,
        ndb,
        KeyValue,
        PytorchExportedProgramAdapterImpl,
    )


def _gradient(ndb: Any, *, low_is_bad: bool) -> list[Any]:
    if low_is_bad:
        colors = ("#d73027", "#fee08b", "#1a9850")
    else:
        colors = ("#1a9850", "#fee08b", "#d73027")
    return [
        ndb.GradientItem(stop=0.0, bgColor=colors[0]),
        ndb.GradientItem(stop=0.5, bgColor=colors[1]),
        ndb.GradientItem(stop=1.0, bgColor=colors[2]),
    ]


def _node_values(
    entries: tuple[VgfQuantizationVisualizationEntry, ...],
    metric: str,
) -> dict[str, float]:
    values: dict[str, float] = {}
    for entry in entries:
        value = getattr(entry, metric)
        if value is None or not math.isfinite(float(value)):
            continue
        for node_id in entry.fp32_node_ids:
            values[node_id] = float(value)
    return values


def _build_node_data(
    ndb: Any,
    graph_id: str,
    values: dict[str, float],
    *,
    low_is_bad: bool,
) -> Any:
    return ndb.ModelNodeData(
        graphsData={
            graph_id: ndb.GraphNodeData(
                results={
                    node_id: ndb.NodeDataResult(value=value)
                    for node_id, value in values.items()
                },
                gradient=_gradient(ndb, low_is_bad=low_is_bad),
            )
        }
    )


def _decorate_graph(
    exported_program: ExportedProgram,
    graph: Any,
    entries: tuple[VgfQuantizationVisualizationEntry, ...],
    key_value_type: Any,
) -> None:
    """Apply module-FQN hierarchy and metrics to graph."""
    fx_nodes = list(exported_program.graph_module.graph.nodes)
    if len(fx_nodes) != len(graph.nodes):
        raise ValueError(
            "Model Explorer adapter node count does not match the ExportedProgram "
            f"graph: fx={len(fx_nodes)}, explorer={len(graph.nodes)}"
        )

    for explorer_node, fx_node in zip(graph.nodes, fx_nodes, strict=True):
        module_fqn = _fx_node_module_fqn(fx_node)
        if module_fqn is not None:
            explorer_node.namespace = _namespace_from_fqn(module_fqn)

    entries_by_node: dict[str, VgfQuantizationVisualizationEntry] = {}
    for entry in entries:
        for node_id in entry.fp32_node_ids:
            entries_by_node[node_id] = entry

    for explorer_node in graph.nodes:
        node_entry = entries_by_node.get(explorer_node.id)
        if node_entry is None:
            continue
        existing_keys = {
            (
                getattr(attr, "key", None)
                if not isinstance(attr, dict)
                else attr.get("key")
            )
            for attr in explorer_node.attrs
        }
        for key, value in _entry_attributes(node_entry).items():
            if key not in existing_keys:
                explorer_node.attrs.append(key_value_type(key=key, value=value))

    group_attributes = dict(getattr(graph, "groupNodeAttributes", None) or {})
    for entry in entries:
        group_attributes[_namespace_from_fqn(entry.module_fqn)] = _entry_attributes(
            entry
        )
    graph.groupNodeAttributes = group_attributes


def build_vgf_quantization_model_explorer_config(
    fp32_exported_program: ExportedProgram,
    assessment: VgfGraphQuantizationAssessment,
) -> Any:
    """Build a Model Explorer config with MSE, SNR, and saturation views."""
    (
        model_explorer_config,
        consts,
        ndb,
        key_value_type,
        adapter_type,
    ) = _model_explorer_imports()

    config = model_explorer_config()
    adapter = adapter_type(fp32_exported_program, consts.DEFAULT_SETTINGS)
    graphs = adapter.convert()
    graph = graphs["graphs"][0]
    entries = build_vgf_quantization_visualization_entries(assessment)
    _decorate_graph(fp32_exported_program, graph, entries, key_value_type)

    graphs_index = len(config.graphs_list)
    config.graphs_list.append(graphs)
    config.model_sources.append({"url": f"graphs://vgf_quantization/{graphs_index}"})

    error_values = _node_values(entries, "mse")
    snr_values = _node_values(entries, "snr_db")
    saturation_values = _node_values(entries, "saturation_percent")

    if error_values:
        config.add_node_data(
            "Error (MSE)",
            _build_node_data(
                ndb,
                graph.id,
                error_values,
                low_is_bad=False,
            ),
        )
    if snr_values:
        config.add_node_data(
            "SNR (dB)",
            _build_node_data(
                ndb,
                graph.id,
                snr_values,
                low_is_bad=True,
            ),
        )
    if saturation_values:
        config.add_node_data(
            "Saturation (%)",
            _build_node_data(
                ndb,
                graph.id,
                saturation_values,
                low_is_bad=False,
            ),
        )

    return config


def visualize_vgf_quantization_assessment(
    fp32_exported_program: ExportedProgram,
    assessment: VgfGraphQuantizationAssessment,
    *,
    no_open_in_browser: bool = False,
    reuse_server: bool = True,
) -> None:
    """Launch the existing ExecuTorch Model Explorer integration."""
    from executorch.devtools.visualization.visualization_utils import (
        visualize_model_explorer,
    )

    config = build_vgf_quantization_model_explorer_config(
        fp32_exported_program,
        assessment,
    )
    if reuse_server:
        config.set_reuse_server()
    visualize_model_explorer(
        config=config,
        no_open_in_browser=no_open_in_browser,
    )
