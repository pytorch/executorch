# Copyright (c) Qualcomm Innovation Center, Inc.
# All rights reserved
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Per-layer accuracy lens with sparse from_node-root matching.

Design constraints:
1. Sparse correspondence only: each match key maps to one node per graph.
2. Key priority: from_node_root first, then node-id fallback when root missing.
3. Last topological node wins for duplicate keys in each graph.
4. Per-layer metrics are computed on one sample index, reusing AccuracyLens
   worst-index selection when available.
"""

from __future__ import annotations

import html
import math
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional, Tuple

import torch

from executorch.devtools.fx_viewer.color_rules import ColorRule
from executorch.devtools.fx_viewer.extension import GraphExtension

from ..interfaces import (
    AnalysisResult,
    Frontend,
    GraphCompareSpec,
    GraphView,
    HtmlBlock,
    HtmlRecordSpec,
    Lens,
    ObservationContext,
    RecordAnalysis,
    RecordDigest,
    TableBlock,
    TableRecordSpec,
    ViewList,
)
from .accuracy import AbsErr, AccuracyLens, CosineSimilarity, MSE, PSNR
from .pipeline_graph_collector import PipelineGraphCollectorLens


@dataclass
class _SparseNodeRef:
    node_id: str
    key_kind: str
    from_node_root: Optional[str]
    topo_index: int


class _NodeOutputCapturer(torch.fx.Interpreter):
    """Capture intermediate outputs by node id."""

    def __init__(self, module: torch.fx.GraphModule):
        super().__init__(module)
        self.outputs: Dict[str, Any] = {}

    def run_node(self, n: torch.fx.Node) -> Any:
        out = super().run_node(n)
        if n.op not in ("placeholder", "output"):
            self.outputs[n.name] = out
        return out


class _MetricNumericColorRule(ColorRule):
    """Numeric color rule with optional inverse severity direction.

    When ``fixed_range`` is supplied, the given ``(vmin, vmax)`` is used for
    normalization instead of computing it from ``nodes_data``. This lets the
    caller share a single color scale across multiple records.
    """

    def __init__(
        self,
        attribute: str,
        *,
        low_rgb: Tuple[int, int, int],
        high_rgb: Tuple[int, int, int],
        inverse: bool = False,
        fixed_range: Optional[Tuple[float, float]] = None,
    ) -> None:
        super().__init__(attribute)
        self.low_rgb = low_rgb
        self.high_rgb = high_rgb
        self.inverse = inverse
        self.fixed_range = fixed_range

    @staticmethod
    def _interp(low: int, high: int, ratio: float) -> int:
        return int(low + (high - low) * ratio)

    def _color(self, ratio: float) -> str:
        ratio = max(0.0, min(1.0, ratio))
        lr, lg, lb = self.low_rgb
        hr, hg, hb = self.high_rgb
        r = self._interp(lr, hr, ratio)
        g = self._interp(lg, hg, ratio)
        b = self._interp(lb, hb, ratio)
        return f"#{r:02x}{g:02x}{b:02x}"

    def _resolve_range(self, vals: List[float]) -> Optional[Tuple[float, float]]:
        if self.fixed_range is not None:
            rmin, rmax = self.fixed_range
            if math.isfinite(rmin) and math.isfinite(rmax):
                vmin, vmax = float(rmin), float(rmax)
                if vmin == vmax:
                    vmax = vmin + 1e-12
                return vmin, vmax
        if not vals:
            return None
        vmin, vmax = min(vals), max(vals)
        if vmin == vmax:
            vmax = vmin + 1e-12
        return vmin, vmax

    def apply(self, nodes_data: dict) -> tuple[dict, list]:
        vals = []
        for data in nodes_data.values():
            v = data.get(self.attribute)
            if isinstance(v, (int, float)):
                fv = float(v)
                if math.isfinite(fv):
                    vals.append(fv)

        resolved = self._resolve_range(vals)
        if resolved is None:
            return {}, []
        vmin, vmax = resolved

        node_colors = {}
        for node_id, data in nodes_data.items():
            v = data.get(self.attribute)
            if not isinstance(v, (int, float)):
                continue
            fv = float(v)
            if not math.isfinite(fv):
                continue
            ratio = (fv - vmin) / (vmax - vmin)
            if self.inverse:
                ratio = 1.0 - ratio
            node_colors[node_id] = self._color(ratio)

        legend = []
        for i in range(5):
            t = i / 4.0
            ratio = 1.0 - t if self.inverse else t
            color = self._color(ratio)
            val = vmin + t * (vmax - vmin)
            legend.append({"label": f"{val:.3f}", "color": color})
        return node_colors, legend


class PerLayerAccuracyLens(Lens):
    _anchor_graph_module: Optional[torch.fx.GraphModule] = None
    _anchor_record_name: Optional[str] = None
    _anchor_sparse_index: Dict[str, _SparseNodeRef] = {}
    _anchor_outputs_cache: Dict[int, Dict[str, Any]] = {}

    @classmethod
    def get_name(cls) -> str:
        return "per_layer_accuracy"

    @classmethod
    def clear(cls) -> None:
        cls._anchor_graph_module = None
        cls._anchor_record_name = None
        cls._anchor_sparse_index = {}
        cls._anchor_outputs_cache = {}

    @classmethod
    def on_session_end(cls, context: ObservationContext) -> None:
        cls.clear()

    @classmethod
    def _lens_config(cls, context: ObservationContext) -> Dict[str, Any]:
        cfg = context.config.get("per_layer_accuracy", {})
        return cfg if isinstance(cfg, dict) else {}

    @classmethod
    def _to_graph_module(cls, artifact: Any) -> Optional[torch.fx.GraphModule]:
        try:
            from torch.export import ExportedProgram

            if isinstance(artifact, ExportedProgram):
                # Use executable GraphModule with bound params/buffers.
                # Raw ExportedProgram.graph_module has lifted placeholders and
                # cannot be executed with dataset sample inputs alone.
                gm = artifact.module()
                return gm if isinstance(gm, torch.fx.GraphModule) else None

            exported_program = getattr(artifact, "exported_program", None)
            if isinstance(exported_program, ExportedProgram):
                gm = exported_program.module()
                return gm if isinstance(gm, torch.fx.GraphModule) else None
        except Exception:
            pass

        if isinstance(artifact, torch.fx.GraphModule):
            return artifact

        graph_module = getattr(artifact, "graph_module", None)
        if isinstance(graph_module, torch.fx.GraphModule):
            return graph_module

        return None

    @staticmethod
    def _extract_from_node_root(node: torch.fx.Node) -> Optional[str]:
        root_name = node.meta.get("from_node_root")
        if isinstance(root_name, str) and root_name:
            return root_name

        from_node = node.meta.get("from_node")
        if not isinstance(from_node, list) or not from_node:
            return None

        try:
            ns = from_node[-1]
            while getattr(ns, "from_node", None):
                parent = ns.from_node
                if not isinstance(parent, list) or not parent:
                    break
                ns = parent[-1]
            name = getattr(ns, "name", None)
            return str(name) if name else None
        except Exception:
            return None

    @classmethod
    def _build_sparse_node_index(
        cls, graph_module: torch.fx.GraphModule
    ) -> Dict[str, _SparseNodeRef]:
        """Sparse index with last-topological node selection per key."""
        sparse: Dict[str, _SparseNodeRef] = {}
        for topo, node in enumerate(graph_module.graph.nodes):
            if node.op in ("placeholder", "output"):
                continue
            root = cls._extract_from_node_root(node)
            if root:
                key = f"root:{root}"
                kind = "root"
            else:
                key = f"id:{node.name}"
                kind = "id_fallback"
            sparse[key] = _SparseNodeRef(
                node_id=node.name,
                key_kind=kind,
                from_node_root=root,
                topo_index=topo,
            )
        return sparse

    @staticmethod
    def _resolve_dataset() -> Optional[List[Any]]:
        if (
            isinstance(AccuracyLens._captured_dataset, list)
            and AccuracyLens._captured_dataset
        ):
            return AccuracyLens._captured_dataset
        if (
            isinstance(PipelineGraphCollectorLens._last_calibration_dataset, list)
            and PipelineGraphCollectorLens._last_calibration_dataset
        ):
            return PipelineGraphCollectorLens._last_calibration_dataset
        return None

    @staticmethod
    def _pick_sample_index(
        dataset: Optional[List[Any]],
        config: Dict[str, Any],
    ) -> Tuple[int, str]:
        if not dataset:
            return 0, "default(0)"

        explicit = config.get("sample_index")
        if isinstance(explicit, int):
            idx = min(max(explicit, 0), len(dataset) - 1)
            return idx, "config.sample_index"

        priority = config.get(
            "worst_metric_priority",
            ["psnr", "cosine_sim", "mse", "abs_err", "top_1", "top_5"],
        )
        if not isinstance(priority, list):
            priority = ["psnr", "cosine_sim", "mse", "abs_err", "top_1", "top_5"]

        for metric_name in priority:
            idx = AccuracyLens._worst_indices.get(str(metric_name))
            if isinstance(idx, int):
                return (
                    min(max(idx, 0), len(dataset) - 1),
                    f"accuracy.worst[{metric_name}]",
                )

        return 0, "default(0)"

    @staticmethod
    def _safe_float(value: Any, default: float = 0.0) -> float:
        try:
            parsed = float(value)
        except (TypeError, ValueError):
            return default
        return parsed if math.isfinite(parsed) else default

    @staticmethod
    def _normalize_sample(sample: Any) -> Tuple[Any, ...]:
        if isinstance(sample, tuple):
            return sample
        if isinstance(sample, list):
            return tuple(sample)
        return (sample,)

    @classmethod
    def _capture_outputs(
        cls,
        graph_module: torch.fx.GraphModule,
        sample: Tuple[Any, ...],
    ) -> Dict[str, Any]:
        capturer = _NodeOutputCapturer(graph_module)
        with torch.no_grad():
            capturer.run(*sample)
        return capturer.outputs

    @staticmethod
    def _flatten_for_metric(value: Any) -> Tuple[Optional[torch.Tensor], str]:
        if isinstance(value, torch.Tensor):
            return value.detach().cpu().to(torch.float64).reshape(-1), str(
                tuple(value.shape)
            )

        if isinstance(value, (tuple, list)):
            tensors = [
                v.detach().cpu().to(torch.float64).reshape(-1)
                for v in value
                if isinstance(v, torch.Tensor)
            ]
            if tensors:
                shape = (
                    "["
                    + ", ".join(
                        str(tuple(v.shape))
                        for v in value
                        if isinstance(v, torch.Tensor)
                    )
                    + "]"
                )
                return torch.cat(tensors), shape
            scalars = [float(v) for v in value if isinstance(v, (int, float, bool))]
            if scalars:
                return (
                    torch.tensor(scalars, dtype=torch.float64),
                    f"list(len={len(scalars)})",
                )
            return None, "unsupported_sequence"

        if isinstance(value, (int, float, bool)):
            return torch.tensor([float(value)], dtype=torch.float64), "scalar"

        return None, f"unsupported:{type(value).__name__}"

    @classmethod
    def _compute_pair_metrics(
        cls,
        anchor_value: Any,
        target_value: Any,
    ) -> Optional[Dict[str, Any]]:
        anchor_flat, anchor_shape = cls._flatten_for_metric(anchor_value)
        target_flat, target_shape = cls._flatten_for_metric(target_value)
        if anchor_flat is None or target_flat is None:
            return None

        compared = min(anchor_flat.numel(), target_flat.numel())
        if compared <= 0:
            return None

        anchor_vec = torch.nan_to_num(
            anchor_flat[:compared], nan=0.0, posinf=0.0, neginf=0.0
        )
        target_vec = torch.nan_to_num(
            target_flat[:compared], nan=0.0, posinf=0.0, neginf=0.0
        )

        predictions = [target_vec]
        golden = [anchor_vec]

        psnr = cls._safe_float(PSNR(golden).calculate(predictions))
        cosine = cls._safe_float(CosineSimilarity(golden).calculate(predictions))
        mse = cls._safe_float(MSE(golden).calculate(predictions))
        abs_err = cls._safe_float(AbsErr(golden).calculate(predictions))

        return {
            "numel_compared": int(compared),
            "anchor_shape": anchor_shape,
            "target_shape": target_shape,
            "psnr": float(psnr),
            "cosine_sim": float(cosine),
            "mse": float(mse),
            "abs_err": float(abs_err),
        }

    @staticmethod
    def _summarize_rows(rows: List[Dict[str, Any]]) -> Dict[str, float]:
        if not rows:
            return {
                "psnr_mean": 0.0,
                "psnr_min": 0.0,
                "psnr_max": 0.0,
                "cosine_sim_mean": 0.0,
                "mse_mean": 0.0,
                "abs_err_mean": 0.0,
            }

        def _values(k: str) -> List[float]:
            return [PerLayerAccuracyLens._safe_float(r.get(k, 0.0)) for r in rows]

        psnr_vals = _values("psnr")
        return {
            "psnr_mean": f"{float(sum(psnr_vals) / len(psnr_vals)):.4f}",
            "psnr_min": f"{float(min(psnr_vals)):.4f}",
            "psnr_max": f"{float(max(psnr_vals)):.4f}",
            "cosine_sim_mean": f"{float(sum(_values('cosine_sim')) / len(rows)):.4f}",
            "mse_mean": f"{float(sum(_values('mse')) / len(rows)):.4f}",
            "abs_err_mean": f"{float(sum(_values('abs_err')) / len(rows)):.4f}",
        }

    @classmethod
    def observe(cls, artifact: Any, context: ObservationContext) -> Any:
        acc_config = context.config.get("accuracy", {})
        if not acc_config.get("enabled", True):
            return None

        graph_module = cls._to_graph_module(artifact)
        if graph_module is None:
            return None

        cfg = cls._lens_config(context)
        anchor_name = str(cfg.get("anchor_record_name", "Exported Float"))
        record_name = str(context.shared_state.get("record_name", "no_record_name!"))

        sparse_index = cls._build_sparse_node_index(graph_module)
        if record_name == anchor_name:
            cls._anchor_graph_module = graph_module
            cls._anchor_record_name = record_name
            cls._anchor_sparse_index = sparse_index
            cls._anchor_outputs_cache = {}

        if cls._anchor_graph_module is None or not cls._anchor_sparse_index:
            return {
                "graph_ref": record_name,
                "anchor_record": anchor_name,
                "sample_index": 0,
                "sample_source": "no_anchor",
                "rows": [],
                "summary": {},
                "match_count": 0,
                "anchor_sparse_count": 0,
                "target_sparse_count": len(sparse_index),
            }

        dataset = cls._resolve_dataset()
        if not dataset:
            return {
                "graph_ref": record_name,
                "anchor_record": cls._anchor_record_name or anchor_name,
                "sample_index": 0,
                "sample_source": "no_dataset",
                "rows": [],
                "summary": {},
                "match_count": 0,
                "anchor_sparse_count": len(cls._anchor_sparse_index),
                "target_sparse_count": len(sparse_index),
            }

        sample_index, sample_source = cls._pick_sample_index(dataset, cfg)
        sample = cls._normalize_sample(dataset[sample_index])

        if sample_index not in cls._anchor_outputs_cache:
            cls._anchor_outputs_cache[sample_index] = cls._capture_outputs(
                cls._anchor_graph_module, sample
            )
        anchor_outputs = cls._anchor_outputs_cache[sample_index]

        if graph_module is cls._anchor_graph_module:
            target_outputs = anchor_outputs
        else:
            target_outputs = cls._capture_outputs(graph_module, sample)

        matched_keys = sorted(
            set(cls._anchor_sparse_index.keys()) & set(sparse_index.keys())
        )
        rows: List[Dict[str, Any]] = []
        for key in matched_keys:
            anchor_ref = cls._anchor_sparse_index[key]
            target_ref = sparse_index[key]
            if (
                anchor_ref.node_id not in anchor_outputs
                or target_ref.node_id not in target_outputs
            ):
                continue

            metrics = cls._compute_pair_metrics(
                anchor_outputs[anchor_ref.node_id],
                target_outputs[target_ref.node_id],
            )
            if metrics is None:
                continue

            rows.append(
                {
                    "match_key": key,
                    "key_kind": target_ref.key_kind,
                    "from_node_root": target_ref.from_node_root,
                    "anchor_node": anchor_ref.node_id,
                    "target_node": target_ref.node_id,
                    "anchor_topo_index": anchor_ref.topo_index,
                    "target_topo_index": target_ref.topo_index,
                    **metrics,
                }
            )

        return {
            "graph_ref": record_name,
            "anchor_record": cls._anchor_record_name or anchor_name,
            "sample_index": sample_index,
            "sample_source": sample_source,
            "rows": rows,
            "summary": cls._summarize_rows(rows),
            "match_count": len(rows),
            "anchor_sparse_count": len(cls._anchor_sparse_index),
            "target_sparse_count": len(sparse_index),
        }

    @classmethod
    def digest(cls, observation: Any, context: ObservationContext) -> Any:
        return observation

    @staticmethod
    def _metric_specs() -> Dict[str, Dict[str, Any]]:
        return {
            # Lower value is worse for PSNR/Cosine.
            "psnr": {
                "name": "Per-layer PSNR",
                "label": "PSNR",
                "inverse": True,
            },
            "cosine_sim": {
                "name": "Per-layer Cosine Similarity",
                "label": "Cosine",
                "inverse": True,
            },
            # Higher value is worse for error metrics.
            "mse": {
                "name": "Per-layer MSE",
                "label": "MSE",
                "inverse": False,
            },
            "abs_err": {
                "name": "Per-layer AbsErr",
                "label": "AbsErr",
                "inverse": False,
            },
        }

    @classmethod
    def _build_metric_extension(
        cls,
        rows: List[Dict[str, Any]],
        metric_name: str,
        *,
        fixed_range: Optional[Tuple[float, float]] = None,
    ) -> GraphExtension:
        spec = cls._metric_specs().get(metric_name)
        if not spec:
            raise ValueError(f"Unsupported per-layer metric extension: {metric_name}")

        ext = GraphExtension(id=metric_name, name=str(spec["name"]))
        for row in rows:
            node_id = str(row["target_node"])
            info = {
                "sparse_match_key": row.get("match_key", ""),
                "key_kind": row.get("key_kind", ""),
                "from_node_root": row.get("from_node_root", ""),
                "anchor_node": row.get("anchor_node", ""),
                "target_node": row.get("target_node", ""),
                "anchor_topo_index": row.get("anchor_topo_index", -1),
                "target_topo_index": row.get("target_topo_index", -1),
                "numel_compared": row.get("numel_compared", 0),
                "anchor_shape": row.get("anchor_shape", "n/a"),
                "target_shape": row.get("target_shape", "n/a"),
                "psnr": cls._safe_float(row.get("psnr", 0.0)),
                "cosine_sim": cls._safe_float(row.get("cosine_sim", 0.0)),
                "mse": cls._safe_float(row.get("mse", 0.0)),
                "abs_err": cls._safe_float(row.get("abs_err", 0.0)),
            }
            ext.add_node_data(node_id, info)

        ext.set_sync_key("sparse_match_key")

        def _format_metric_value(value: float, *, tooltip: bool = False) -> str:
            if metric_name in ("mse", "abs_err"):
                return f"{value:.6e}" if tooltip else f"{value:.3e}"
            return f"{value:.6f}" if tooltip else f"{value:.4f}"

        def _label_formatter(d: Dict[str, Any]) -> List[str]:
            primary = cls._safe_float(d.get(metric_name, 0.0))
            primary_label = str(spec["label"])
            return [f"{primary_label}={_format_metric_value(primary)}"]

        ext.set_label_formatter(_label_formatter)

        def _tooltip_formatter(d: Dict[str, Any]) -> List[str]:
            primary = cls._safe_float(d.get(metric_name, 0.0))
            primary_label = str(spec["label"])
            return [
                f"target_node={d.get('target_node', 'n/a')}",
                f"match_key={d.get('sparse_match_key', '')}",
                f"{primary_label}={_format_metric_value(primary, tooltip=True)}",
            ]

        ext.set_tooltip_formatter(_tooltip_formatter)
        ext.set_color_rule(
            _MetricNumericColorRule(
                attribute=metric_name,
                # Severe values map to darker red.
                low_rgb=(254, 224, 210),
                high_rgb=(165, 15, 21),
                inverse=bool(spec["inverse"]),
                fixed_range=fixed_range,
            )
        )
        return ext

    @staticmethod
    def _aggregate_metric_ranges(
        records: List[RecordDigest],
    ) -> Dict[str, List[float]]:
        """Union (min, max) per metric across every record's rows.

        Returned as list-of-floats for clean JSON round-tripping through
        ``AnalysisResult.global_data``.
        """
        pools: Dict[str, List[float]] = {
            metric: [] for metric in PerLayerAccuracyLens._metric_specs().keys()
        }
        for record in records:
            digest = record.data.get("per_layer_accuracy")
            if not isinstance(digest, dict):
                continue
            rows = digest.get("rows")
            if not isinstance(rows, list):
                continue
            for row in rows:
                for metric in pools:
                    v = row.get(metric)
                    if isinstance(v, (int, float)):
                        fv = float(v)
                        if math.isfinite(fv):
                            pools[metric].append(fv)

        ranges: Dict[str, List[float]] = {}
        for metric, vals in pools.items():
            if vals:
                ranges[metric] = [min(vals), max(vals)]
        return ranges

    @staticmethod
    def analyze(records: List[RecordDigest], config: Dict[str, Any]) -> AnalysisResult:
        result = AnalysisResult()
        metric_ranges = PerLayerAccuracyLens._aggregate_metric_ranges(records)
        if metric_ranges:
            result.global_data["metric_ranges"] = metric_ranges

        for record in records:
            digest = record.data.get("per_layer_accuracy")
            if not isinstance(digest, dict):
                continue
            rows = digest.get("rows")
            if not isinstance(rows, list) or not rows:
                continue

            analysis = RecordAnalysis(
                data={
                    "match_count": digest.get("match_count", 0),
                    "sample_index": digest.get("sample_index", 0),
                    "sample_source": digest.get("sample_source", "n/a"),
                }
            )

            for metric_name in ("cosine_sim",):
                # TODO other options "psnr"  "mse", "abs_err"
                r = metric_ranges.get(metric_name)
                fixed_range = (r[0], r[1]) if r else None
                metric_ext = PerLayerAccuracyLens._build_metric_extension(
                    rows, metric_name, fixed_range=fixed_range
                )
                analysis.add_graph_layer(metric_name, metric_ext)

            result.per_record_data[record.name] = analysis

        return result

    class _PerLayerAccuracyFrontend(Frontend):
        @staticmethod
        def _interp_color(
            ratio: float,
            low_rgb: Tuple[int, int, int],
            high_rgb: Tuple[int, int, int],
        ) -> str:
            ratio = max(0.0, min(1.0, ratio))
            r = int(low_rgb[0] + (high_rgb[0] - low_rgb[0]) * ratio)
            g = int(low_rgb[1] + (high_rgb[1] - low_rgb[1]) * ratio)
            b = int(low_rgb[2] + (high_rgb[2] - low_rgb[2]) * ratio)
            return f"#{r:02x}{g:02x}{b:02x}"

        @staticmethod
        def _hex_to_rgb(hex_color: str) -> Tuple[int, int, int]:
            h = hex_color.lstrip("#")
            if len(h) != 6:
                return (255, 255, 255)
            return (int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16))

        @classmethod
        def _text_color_for_bg(cls, hex_color: str) -> str:
            r, g, b = cls._hex_to_rgb(hex_color)
            # WCAG-ish luma heuristic.
            luma = 0.299 * r + 0.587 * g + 0.114 * b
            return "#111111" if luma > 150 else "#f8f8f8"

        @classmethod
        def _metric_cell_style(
            cls,
            value: float,
            vmin: float,
            vmax: float,
            *,
            low_rgb: Tuple[int, int, int],
            high_rgb: Tuple[int, int, int],
            inverse: bool,
        ) -> str:
            value = PerLayerAccuracyLens._safe_float(value)
            vmin = PerLayerAccuracyLens._safe_float(vmin)
            vmax = PerLayerAccuracyLens._safe_float(vmax)
            if vmax <= vmin:
                ratio = 0.0
            else:
                ratio = (value - vmin) / (vmax - vmin)
            if not math.isfinite(ratio):
                ratio = 0.0
            if inverse:
                ratio = 1.0 - ratio
            bg = cls._interp_color(ratio, low_rgb, high_rgb)
            fg = cls._text_color_for_bg(bg)
            return f"background:{bg};color:{fg};"

        @classmethod
        def _merged_metrics_table_html(
            cls,
            rows: Iterable[Dict[str, Any]],
            metric_ranges: Optional[Dict[str, List[float]]] = None,
        ) -> str:
            row_list = list(rows)
            if not row_list:
                return "<div class='pla-empty'>No matched nodes.</div>"

            # Worst -> best ranking uses PSNR primarily (lower is worse).
            row_list.sort(
                key=lambda r: (
                    PerLayerAccuracyLens._safe_float(r.get("psnr", 0.0)),
                    -PerLayerAccuracyLens._safe_float(r.get("mse", 0.0)),
                    -PerLayerAccuracyLens._safe_float(r.get("abs_err", 0.0)),
                    PerLayerAccuracyLens._safe_float(r.get("cosine_sim", 0.0)),
                )
            )

            def _range(k: str) -> Tuple[float, float]:
                if metric_ranges and k in metric_ranges:
                    r = metric_ranges[k]
                    if len(r) == 2:
                        return (float(r[0]), float(r[1]))
                vals = [
                    PerLayerAccuracyLens._safe_float(r.get(k, 0.0)) for r in row_list
                ]
                return (min(vals), max(vals)) if vals else (0.0, 0.0)

            psnr_min, psnr_max = _range("psnr")
            cos_min, cos_max = _range("cosine_sim")
            mse_min, mse_max = _range("mse")
            abs_min, abs_max = _range("abs_err")

            parts = [
                "<table class='pla-metric-table'>",
                "<thead><tr>",
                "<th>#</th><th>Target Node</th><th>Anchor Node</th><th>Root</th>",
                "<th>PSNR</th><th>Cosine</th><th>MSE</th><th>AbsErr</th>",
                "</tr></thead><tbody>",
            ]

            for rank, row in enumerate(row_list, start=1):
                node = html.escape(str(row.get("target_node", "")))
                anchor = html.escape(str(row.get("anchor_node", "")))
                root = html.escape(str(row.get("from_node_root") or "n/a"))
                psnr = PerLayerAccuracyLens._safe_float(row.get("psnr", 0.0))
                cosine = PerLayerAccuracyLens._safe_float(row.get("cosine_sim", 0.0))
                mse = PerLayerAccuracyLens._safe_float(row.get("mse", 0.0))
                abs_err = PerLayerAccuracyLens._safe_float(row.get("abs_err", 0.0))

                psnr_style = cls._metric_cell_style(
                    psnr,
                    psnr_min,
                    psnr_max,
                    low_rgb=(254, 224, 210),
                    high_rgb=(165, 15, 21),
                    inverse=True,
                )
                cos_style = cls._metric_cell_style(
                    cosine,
                    cos_min,
                    cos_max,
                    low_rgb=(254, 237, 222),
                    high_rgb=(217, 72, 1),
                    inverse=True,
                )
                mse_style = cls._metric_cell_style(
                    mse,
                    mse_min,
                    mse_max,
                    low_rgb=(239, 243, 255),
                    high_rgb=(8, 81, 156),
                    inverse=False,
                )
                abs_style = cls._metric_cell_style(
                    abs_err,
                    abs_min,
                    abs_max,
                    low_rgb=(242, 240, 247),
                    high_rgb=(84, 39, 143),
                    inverse=False,
                )

                parts.append(
                    "<tr>"
                    f"<td>{rank}</td>"
                    f"<td>{node}</td>"
                    f"<td>{anchor}</td>"
                    f"<td>{root}</td>"
                    f"<td style='{psnr_style}'>{psnr:.4f}</td>"
                    f"<td style='{cos_style}'>{cosine:.6f}</td>"
                    f"<td style='{mse_style}'>{mse:.6e}</td>"
                    f"<td style='{abs_style}'>{abs_err:.6e}</td>"
                    "</tr>"
                )

            parts.append("</tbody></table>")
            return "".join(parts)

        def resources(self) -> Dict[str, str]:
            return {
                "css": """
.pla-metric-table {
  width: 100%;
  border-collapse: collapse;
  font-size: 12px;
}
.pla-metric-table th, .pla-metric-table td {
  border: 1px solid var(--border-color);
  padding: 4px 6px;
  text-align: left;
}
.pla-metric-table th {
  background: var(--bg-tertiary);
  color: var(--text-primary);
}
.pla-empty {
  color: var(--text-secondary);
  font-size: 12px;
}
"""
            }

        def record(
            self, digest: Any, analysis: Dict[str, Any], context: Dict[str, Any]
        ) -> Optional[ViewList]:
            if not isinstance(digest, dict):
                return None

            rows = digest.get("rows") if isinstance(digest.get("rows"), list) else []
            summary = (
                digest.get("summary", {})
                if isinstance(digest.get("summary"), dict)
                else {}
            )
            graph_ref = str(digest.get("graph_ref") or context.get("name") or "")
            lens_name = PerLayerAccuracyLens.get_name()
            metric_ranges = None
            if isinstance(analysis, dict):
                raw = analysis.get("global")
                if isinstance(raw, dict):
                    mr = raw.get("metric_ranges")
                    if isinstance(mr, dict):
                        metric_ranges = mr

            summary_data = {
                "anchor_record": digest.get("anchor_record", "n/a"),
                "sample_index": digest.get("sample_index", 0),
                "sample_source": digest.get("sample_source", "n/a"),
                "match_count": digest.get("match_count", 0),
                "anchor_sparse_count": digest.get("anchor_sparse_count", 0),
                "target_sparse_count": digest.get("target_sparse_count", 0),
                **summary,
            }

            blocks = [
                TableBlock(
                    id="per_layer_accuracy_summary",
                    title="Per-layer Accuracy Summary",
                    record=TableRecordSpec(data=summary_data),
                    order=20,
                ),
                GraphView(
                    id="per_layer_accuracy_graph",
                    title="Per-layer Accuracy Graph",
                    graph_ref=graph_ref,
                    default_layers=[f"{lens_name}/cosine_sim"],
                    default_color_by=f"{lens_name}/cosine_sim",
                    compare=GraphCompareSpec(
                        default_sync={
                            "mode": "layer",
                            "layer": f"{lens_name}/cosine_sim",
                            "field": "sparse_match_key",
                        }
                    ),
                    order=21,
                ).as_block(),
                HtmlBlock(
                    id="per_layer_accuracy_metrics_table",
                    title="Per-layer Metrics (Worst → Best)",
                    record=HtmlRecordSpec(
                        content=self._merged_metrics_table_html(rows, metric_ranges)
                    ),
                    order=22,
                ),
            ]
            return ViewList(blocks=blocks)

        def check_badges(
            self, digest: Any, analysis: Dict[str, Any]
        ) -> List[Dict[str, str]]:
            if isinstance(digest, dict) and int(digest.get("match_count", 0)) > 0:
                return [
                    {"label": "PLA", "class": "badge", "title": "Per-layer accuracy"}
                ]
            return []

    @staticmethod
    def get_frontend_spec() -> Frontend:
        return PerLayerAccuracyLens._PerLayerAccuracyFrontend()
