# Copyright (c) Qualcomm Innovation Center, Inc.
# All rights reserved
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Observatory runtime core.

Lifecycle summary:
1. Runtime capture: `observe -> digest`.
2. Analysis: per-lens `analyze(records, config)`.
3. Assembly: merge frontend blocks + graph assets/layers.
4. Rendering/export: JSON and HTML reports.

The runtime enforces strict typed interfaces from `interfaces.py`.
"""

from __future__ import annotations

import base64
import copy
import gzip
import json
import logging
import os
import time
import traceback
from contextlib import contextmanager
from dataclasses import asdict
from datetime import datetime
from typing import Any, ContextManager, Dict, List, Optional, Set, Type

from executorch.backends.qualcomm.utils.fx_viewer.exporter import FXGraphExporter

from .graph_hub import GraphHub
from .interfaces import (
    AnalysisResult,
    Frontend,
    Lens,
    ObservationContext,
    RecordAnalysis,
    RecordDigest,
    SessionResult,
    ViewList,
    validate_view_list,
)


class Observatory:
    """Global registry for collecting and rendering observability artifacts."""

    _records: Dict[str, RecordDigest] = {}
    _ignored_graphs: Set[str] = set()
    _session_result: SessionResult = SessionResult()
    _lens_registry: List[Type[Lens]] = []
    _lenses_initialized: bool = False
    _config_stack: List[Dict[str, Any]] = []

    @classmethod
    def register_lens(cls, lens_cls: Type[Lens]) -> None:
        """Register lens class and run one-time setup."""

        if lens_cls in cls._lens_registry:
            return
        cls._lens_registry.append(lens_cls)
        try:
            lens_cls.setup()
        except Exception as exc:
            logging.error("[Observatory] Failed to setup lens %s: %s", lens_cls, exc)

    @classmethod
    def _ensure_default_lenses(cls) -> None:
        """Lazy-register built-in lenses for the minimal observatory runtime."""

        if cls._lenses_initialized:
            return

        from .lenses.graph import GraphLens
        from .lenses.metadata import MetadataLens
        from .lenses.stack_trace import StackTraceLens

        cls.register_lens(GraphLens)
        cls.register_lens(MetadataLens)
        cls.register_lens(StackTraceLens)
        cls._lenses_initialized = True

    @classmethod
    @contextmanager
    def enable_context(cls, config: Optional[Dict[str, Any]] = None) -> ContextManager[None]:
        """Enable observation context with nested config overrides.

        Session hooks run once per outermost context:
        1. On first enter, auto-collection patches are installed and
           `on_session_start` hooks are called.
        2. On last exit, `on_session_end` hooks are called and patches removed.
        """

        cls._ensure_default_lenses()

        def merge_config_dict(base: Dict[str, Any], new: Dict[str, Any]) -> Dict[str, Any]:
            result = copy.copy(base)
            result.update({k: copy.copy(v) for k, v in base.items() if isinstance(v, dict)})
            for key, value in new.items():
                if isinstance(value, dict) and isinstance(result.get(key), dict):
                    result[key].update(value)
                else:
                    result[key] = value
            return result

        parent_config = cls._config_stack[-1] if cls._config_stack else {}
        context_config = merge_config_dict(parent_config, config or {})

        is_outermost_start = len(cls._config_stack) == 0
        cls._config_stack.append(context_config)
        hook_ctx = ObservationContext(config=context_config)

        if is_outermost_start:
            for lens in cls._lens_registry:
                try:
                    data = lens.on_session_start(hook_ctx)
                    if data:
                        cls._session_result.start_data[lens.get_name()] = data
                except Exception as exc:
                    logging.error("[Observatory] Lens %s failed on_session_start: %s", lens, exc)

        try:
            yield
        finally:
            is_outermost_end = len(cls._config_stack) == 1

            if is_outermost_end:
                for lens in cls._lens_registry:
                    try:
                        data = lens.on_session_end(hook_ctx)
                        if data:
                            cls._session_result.end_data[lens.get_name()] = data
                    except Exception as exc:
                        logging.error("[Observatory] Lens %s failed on_session_end: %s", lens, exc)

            cls._config_stack.pop()

    @classmethod
    def _get_current_context(cls) -> Optional[ObservationContext]:
        """Return current observation context or None if disabled."""

        if not cls._config_stack:
            return None
        return ObservationContext(config=cls._config_stack[-1])

    @classmethod
    def ignore_graphs(cls, names: List[str]) -> None:
        """Ignore future collect calls with matching names and drop existing records."""

        for name in names:
            cls._ignored_graphs.add(name)
            if name in cls._records:
                del cls._records[name]

    @classmethod
    def collect(cls, name: str, artifact: Any) -> None:
        """Capture one record across all registered lenses.

        Notes:
        1. No-op when context is disabled.
        2. Record name is exposed via `context.shared_state['record_name']`.
        """

        if any(ignored in name for ignored in cls._ignored_graphs):
            return

        if not cls._config_stack:
            return

        active_config = cls._config_stack[-1]
        ctx = ObservationContext(config=active_config)
        ctx.shared_state["record_name"] = name

        record = RecordDigest(name=name, timestamp=datetime.now().timestamp())
        t_start = time.perf_counter()

        for lens in cls._lens_registry:
            try:
                lens_name = lens.get_name()
                observation = lens.observe(artifact, ctx)
                if observation is None:
                    continue
                digest = lens.digest(observation, ctx)
                if digest is not None:
                    record.data[lens_name] = digest
            except Exception as exc:
                logging.error("[Observatory] Lens %s failed collection for %s: %s", lens, name, exc)

        cls._records[name] = record
        elapsed_ms = (time.perf_counter() - t_start) * 1000.0
        logging.info("[Observatory] Collected %s in %.1f ms", name, elapsed_ms)

    @classmethod
    def list_collected(cls) -> List[str]:
        return list(cls._records.keys())

    @classmethod
    def get(cls, name: str) -> Optional[RecordDigest]:
        return cls._records.get(name)

    @classmethod
    def clear(cls) -> None:
        """Clear records/session state and reset lens runtime state."""

        cls._records.clear()
        cls._session_result = SessionResult()

        for lens in cls._lens_registry:
            try:
                lens.clear()
            except Exception as exc:
                logging.error("[Observatory] Lens %s failed clear: %s", lens, exc)

    @staticmethod
    def _serialize_view_list(result: Any) -> Optional[Dict[str, Any]]:
        """Validate and serialize frontend return values."""

        if result is None:
            return None

        if not isinstance(result, ViewList):
            raise TypeError(f"Frontend must return ViewList, got {type(result)}")

        validate_view_list(result)
        return {"blocks": [asdict(block) for block in result.blocks]}

    @classmethod
    def _safe_frontend_call(cls, lens_name: str, method: Any, *args: Any, **kwargs: Any) -> Optional[Dict[str, Any]]:
        """Run frontend method with error isolation and fallback error block."""

        try:
            result = method(*args, **kwargs)
            return cls._serialize_view_list(result)
        except Exception as exc:
            logging.error(
                "[Observatory] Frontend %s.%s failed: %s\n%s",
                lens_name,
                getattr(method, "__name__", "<unknown>"),
                exc,
                traceback.format_exc(),
            )
            return {
                "blocks": [
                    {
                        "id": "frontend_error",
                        "title": "Frontend Error",
                        "type": "html",
                        "record": {
                            "content": (
                                '<div style="color:var(--error-color);padding:1rem;border:1px solid var(--error-color);'
                                'border-radius:4px;background:var(--error-bg);">'
                                f"<strong>Error:</strong> {str(exc)}</div>"
                            )
                        },
                        "compare": {"mode": "disabled"},
                        "order": 999,
                        "collapsible": True,
                    }
                ]
            }

    @staticmethod
    def _encode_html_blocks(serialized_records: list, dashboard: dict) -> None:
        """Base64-encode HtmlBlock.content strings in-place to prevent JSON corruption."""

        def _encode_blocks(blocks: list) -> None:
            for block in blocks:
                if block.get("type") == "html":
                    content = (block.get("record") or {}).get("content", "")
                    if content:
                        block["record"]["content"] = base64.b64encode(
                            content.encode("utf-8")
                        ).decode("ascii")

        for record in serialized_records:
            for view in (record.get("views") or {}).values():
                _encode_blocks(view.get("blocks") or [])
        for view in dashboard.values():
            _encode_blocks(view.get("blocks") or [])

    @staticmethod
    def _compress_payload(json_data: str, threshold: int = 8192) -> tuple:
        """Gzip+base64 compress JSON payload if above threshold bytes."""

        raw = json_data.encode("utf-8")
        if len(raw) >= threshold:
            compressed = gzip.compress(raw, compresslevel=6)
            return base64.b64encode(compressed).decode("ascii"), True
        return json_data, False

    @classmethod
    def _generate_report_payload(
        cls,
        records: List[RecordDigest],
        session: SessionResult,
        config: Dict[str, Any],
        lens_registry: List[Type[Lens]],
    ) -> Dict[str, Any]:
        """Build full report payload including graph assets/layers and views."""

        analysis_results: Dict[str, AnalysisResult] = {
            lens.get_name(): lens.analyze(records, config) for lens in lens_registry
        }

        resources: Dict[str, List[str]] = {"js": [], "css": []}
        try:
            resources["js"].append(FXGraphExporter._load_viewer_js_bundle())
        except Exception as exc:
            logging.warning("[Observatory] Failed loading fx_viewer runtime bundle: %s", exc)

        for lens in lens_registry:
            frontend = lens.get_frontend_spec()
            res = frontend.resources() if isinstance(frontend, Frontend) else {}
            if res.get("js"):
                resources["js"].append(res["js"])
            if res.get("css"):
                resources["css"].append(res["css"])

        resources["js"] = [
            base64.b64encode(s.encode("utf-8")).decode("ascii") for s in resources["js"]
        ]
        resources["css"] = [
            base64.b64encode(s.encode("utf-8")).decode("ascii") for s in resources["css"]
        ]

        graph_hub = GraphHub()
        serialized_records = []

        for i, record in enumerate(records):
            serialized = {
                "name": record.name,
                "timestamp": datetime.fromtimestamp(record.timestamp).strftime("%Y-%m-%d %H:%M:%S"),
                "views": {},
                "badges": [],
                "diff_index": {},
                "digests": record.data,
            }

            for lens in lens_registry:
                lens_name = lens.get_name()
                digest = record.data.get(lens_name)
                if digest is None:
                    continue

                analysis = analysis_results.get(lens_name, AnalysisResult())
                record_analysis: RecordAnalysis | None = analysis.per_record_data.get(record.name)
                analysis_ctx = {
                    "global": analysis.global_data,
                    "record": (record_analysis.data if record_analysis else {}),
                }

                graph_ref = record.name
                # extract graph data from graph Lense
                if lens_name == "graph":
                    assert isinstance(digest, dict) and isinstance(digest.get("base"), dict), "[Observatory] error validating graph lense output."
                    assert digest["graph_ref"] == graph_ref, "[Observatory] graph ref should be consistant with record name"
                    graph_hub.register_asset(
                        graph_ref,
                        digest["base"],
                        digest.get("meta", {}),
                    )
                graph_hub.add_analysis_layers(graph_ref, lens_name, record_analysis)

                frontend = lens.get_frontend_spec()
                try:
                    serialized["badges"].extend(frontend.check_badges(digest, analysis.global_data))
                except Exception as exc:
                    logging.error("[Observatory] check_badges failed for %s: %s", lens_name, exc)

                if i > 0:
                    prev_digest = records[i - 1].data.get(lens_name)
                    if prev_digest is not None:
                        try:
                            serialized["diff_index"].update(
                                frontend.check_index_diffs(prev_digest, digest, analysis.global_data)
                            )
                        except Exception as exc:
                            logging.error("[Observatory] check_index_diffs failed for %s: %s", lens_name, exc)

                serialized_view = cls._safe_frontend_call(
                    lens_name,
                    frontend.record,
                    digest,
                    analysis_ctx,
                    {"index": i, "name": record.name},
                )
                if serialized_view:
                    serialized["views"][lens_name] = serialized_view

            serialized_records.append(serialized)

        dashboard_views = {}
        for lens in lens_registry:
            lens_name = lens.get_name()
            frontend = lens.get_frontend_spec()
            dashboard_view = cls._safe_frontend_call(
                lens_name,
                frontend.dashboard,
                session.start_data.get(lens_name, {}),
                session.end_data.get(lens_name, {}),
                analysis_results.get(lens_name, AnalysisResult()).global_data,
                records,
            )
            if dashboard_view:
                dashboard_views[lens_name] = dashboard_view

        graph_payload = graph_hub.build_payload()

        payload = {
            "resources": resources,
            "records": serialized_records,
            "dashboard": dashboard_views,
            "analysis_results": {
                key: {
                    "global_data": value.global_data,
                    "per_record_data": {
                        rec_name: {"data": rec_analysis.data}
                        for rec_name, rec_analysis in value.per_record_data.items()
                    },
                }
                for key, value in analysis_results.items()
            },
            "session": {
                "start_data": session.start_data,
                "end_data": session.end_data,
            },
            "graph_assets": graph_payload["graph_assets"],
            "graph_layers": graph_payload["graph_layers"],
        }

        Observatory._encode_html_blocks(serialized_records, dashboard_views)
        return payload

    @classmethod
    def export_html_report(
        cls,
        output_path: str,
        title: str = "Observatory Report",
        config: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Export collected records to HTML report."""

        if not cls._records:
            logging.warning("[Observatory] No records collected, skipping HTML export")
            return

        cls._ensure_default_lenses()
        payload = cls._generate_report_payload(
            list(cls._records.values()),
            cls._session_result,
            config or {},
            cls._lens_registry,
        )
        payload["title"] = title
        payload["generated_at"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        from .html_template import get_html_template

        json_data = json.dumps(payload, default=str).replace("</", "<\\/")
        payload_str, is_compressed = Observatory._compress_payload(json_data)
        html_content = get_html_template(title, payload_str, is_compressed=is_compressed)

        os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as f:
            f.write(html_content)

        logging.info("[Observatory] Exported HTML report to %s", output_path)

    @classmethod
    def export_json(cls, output_path: str) -> None:
        """Export raw records/session data as JSON."""

        if not cls._records:
            logging.warning("[Observatory] No records collected, skipping JSON export")
            return

        data = {
            "records": [asdict(r) for r in cls._records.values()],
            "session": asdict(cls._session_result),
        }

        os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)

        logging.info("[Observatory] Exported raw data to %s", output_path)

    @staticmethod
    def generate_html_from_json(
        json_path: str,
        html_path: str,
        title: str = "Observatory Report",
        config: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Generate HTML report from previously exported raw JSON."""

        with open(json_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        records = [RecordDigest(**r) for r in data["records"]]
        session = SessionResult(**data["session"])

        Observatory._ensure_default_lenses()
        payload = Observatory._generate_report_payload(
            records,
            session,
            config or {},
            Observatory._lens_registry,
        )
        payload["title"] = title
        payload["generated_at"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        from .html_template import get_html_template

        json_data = json.dumps(payload, default=str).replace("</", "<\\/")
        payload_str, is_compressed = Observatory._compress_payload(json_data)
        html_content = get_html_template(title, payload_str, is_compressed=is_compressed)

        os.makedirs(os.path.dirname(html_path) or ".", exist_ok=True)
        with open(html_path, "w", encoding="utf-8") as f:
            f.write(html_content)

        logging.info("[Observatory] Generated HTML report at %s from %s", html_path, json_path)
