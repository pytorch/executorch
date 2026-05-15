# Copyright (c) Qualcomm Innovation Center, Inc.
# All rights reserved
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

import os
import platform
import sys
from datetime import datetime
from typing import Any, Dict, List, Optional

import torch

from ..interfaces import (
    AnalysisResult,
    Frontend,
    Lens,
    ObservationContext,
    RecordAnalysis,
    RecordDigest,
    TableBlock,
    TableRecordSpec,
    ViewList,
)


class MetadataLens(Lens):
    """Collects basic metadata about artifacts and runtime environment."""

    @classmethod
    def get_name(cls) -> str:
        return "metadata"

    @classmethod
    def observe(cls, artifact: Any, context: ObservationContext) -> Any:
        metadata_config = context.config.get("metadata", {})
        if not metadata_config.get("enabled", True):
            return None

        artifact_type = str(type(artifact).__name__)
        node_count: Any = "N/A"

        try:
            from torch.export import ExportedProgram

            if isinstance(artifact, torch.fx.GraphModule):
                artifact_type = "GM"
                node_count = len(list(artifact.graph.nodes))
            elif isinstance(artifact, ExportedProgram):
                artifact_type = "EP"
                node_count = len(list(artifact.graph_module.graph.nodes))
            elif isinstance(artifact, torch.nn.Module):
                artifact_type = "NN"
        except Exception:
            pass

        context.shared_state["artifact_type"] = artifact_type
        return {
            "artifact_type": artifact_type,
            "node_count": node_count,
        }

    @classmethod
    def digest(cls, observation: Any, context: ObservationContext) -> Any:
        return observation

    @classmethod
    def on_session_start(cls, context: ObservationContext) -> Optional[Dict[str, Any]]:
        return {
            "command_line": " ".join(sys.orig_argv),
            "python_version": sys.version.split("\n")[0],
            "platform_system": platform.system(),
            "platform_release": platform.release(),
            "platform_machine": platform.machine(),
            "working_directory": os.getcwd(),
            "start_time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        }

    @staticmethod
    def analyze(records: List[RecordDigest], config: Dict[str, Any]) -> AnalysisResult:
        per_record: Dict[str, RecordAnalysis] = {}

        for i in range(len(records) - 1):
            def _count(rec: RecordDigest) -> int:
                data = rec.data.get("metadata")
                if not data:
                    return 0
                value = data.get("node_count")
                return int(value) if isinstance(value, (int, float)) else 0

            before = _count(records[i])
            after = _count(records[i + 1])
            per_record[records[i + 1].name] = RecordAnalysis(
                data={"node_diff": after - before}
            )

        return AnalysisResult(per_record_data=per_record)

    class MetadataFrontend(Frontend):
        def dashboard(self, start, end, analysis, records) -> Optional[ViewList]:
            return ViewList(
                blocks=[
                    TableBlock(
                        id="metadata_dashboard",
                        title="Session Metadata",
                        record=TableRecordSpec(data=start or {}),
                        order=0,
                    )
                ]
            )

        def record(self, digest, analysis, context) -> Optional[ViewList]:
            data = digest.copy() if isinstance(digest, dict) else {}
            record_analysis = (analysis or {}).get("record") or {}
            node_diff = record_analysis.get("node_diff", 0)
            if node_diff:
                data["nodes_change"] = f"{node_diff:+d}"

            return ViewList(
                blocks=[
                    TableBlock(
                        id="metadata_record",
                        title="Metadata",
                        record=TableRecordSpec(data=data),
                        order=0,
                    )
                ]
            )

        def check_index_diffs(self, prev_digest, curr_digest, analysis):
            try:
                before = int(prev_digest.get("node_count", 0))
                after = int(curr_digest.get("node_count", 0))
                diff = after - before
                if diff:
                    return {"nodes": f"{diff:+d}"}
            except Exception:
                return {}
            return {}

        def check_badges(self, digest, analysis):
            badges = []
            if digest and "artifact_type" in digest:
                badges.append(
                    {
                        "label": str(digest["artifact_type"]),
                        "class": "badge",
                        "title": "Artifact Type",
                    }
                )
            return badges

    @staticmethod
    def get_frontend_spec() -> Frontend:
        return MetadataLens.MetadataFrontend()
