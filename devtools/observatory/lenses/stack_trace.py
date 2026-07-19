# Copyright (c) Qualcomm Innovation Center, Inc.
# All rights reserved
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

import inspect
import logging
import os
from typing import Any, Dict, List

from ..interfaces import Frontend, HtmlBlock, HtmlRecordSpec, Lens, ObservationContext, ViewList
from ..utils import get_git_info, get_repo_root, is_in_repo


class StackTraceLens(Lens):
    """Collects repository-local stack trace frames."""

    @classmethod
    def get_name(cls) -> str:
        return "stack_trace"

    @classmethod
    def _get_stack_trace(cls) -> List[Dict[str, Any]]:
        repo_root = get_repo_root()
        git_info = get_git_info()

        frames = []
        for frame_info in inspect.stack():
            if not is_in_repo(frame_info.filename):
                continue
            if "/observatory/observatory.py" in frame_info.filename.replace("\\", "/"):
                continue

            github_link = None
            link_root = git_info.commit_blob_url or git_info.branch_blob_url or git_info.github_link
            if link_root and repo_root:
                try:
                    rel_path = os.path.relpath(frame_info.filename, repo_root)
                    github_link = f"{link_root}/{rel_path}#L{frame_info.lineno}"
                except Exception:
                    pass

            rel_path = frame_info.filename
            if repo_root and frame_info.filename.startswith(repo_root):
                rel_path = os.path.relpath(frame_info.filename, repo_root)

            frames.append(
                {
                    "function": frame_info.function,
                    "filename": os.path.basename(rel_path),
                    "dir": os.path.dirname(rel_path),
                    "line": frame_info.lineno,
                    "context": frame_info.code_context[0].strip() if frame_info.code_context else None,
                    "link": github_link,
                }
            )

        frames.reverse()
        return frames

    @classmethod
    def observe(cls, artifact: Any, context: ObservationContext) -> Any:
        stacktrace_config = context.config.get("stacktrace", {})
        if not stacktrace_config.get("enabled", True):
            return None

        try:
            return cls._get_stack_trace()
        except Exception as exc:
            logging.warning("[Observatory] Failed to collect stack trace: %s", exc)
            return []

    @classmethod
    def digest(cls, observation: Any, context: ObservationContext) -> Any:
        return observation

    class StackTraceFrontend(Frontend):
        def record(self, digest, analysis, context):
            if not digest:
                return ViewList(
                    blocks=[
                        HtmlBlock(
                            id="stack_trace_record",
                            title="Stack Trace",
                            record=HtmlRecordSpec(content="<div>No stack trace available</div>"),
                            order=40,
                        )
                    ]
                )

            html = ["<div style=\"font-family:monospace;font-size:0.9rem;\">"]
            for frame in digest:
                link_prefix = (
                    f'<a href="{frame["link"]}" target="_blank" style="color:var(--link-color);">'
                    if frame.get("link")
                    else ""
                )
                link_suffix = "</a>" if frame.get("link") else ""
                snippet = ""
                if frame.get("context"):
                    snippet = (
                        "<div style=\"background:var(--bg-tertiary);color:var(--text-primary);"
                        "padding:0.2rem;margin-top:0.2rem;white-space:nowrap;border-radius:3px;\">"
                        f"{frame['context']}"
                        "</div>"
                    )

                html.append(
                    "<div style=\"margin-bottom:0.8rem;padding-left:0.8rem;border-left:3px solid var(--border-color);\">"
                    f"<div style=\"font-weight:bold;color:var(--accent-color);white-space:nowrap;\">{frame['function']}</div>"
                    f"<div style=\"color:var(--text-secondary);white-space:nowrap;\">{link_prefix}{frame['dir']}/{frame['filename']}:{frame['line']}{link_suffix}</div>"
                    f"{snippet}</div>"
                )
            html.append("</div>")

            return ViewList(
                blocks=[
                    HtmlBlock(
                        id="stack_trace_record",
                        title="Stack Trace",
                        record=HtmlRecordSpec(content="".join(html)),
                        order=40,
                    )
                ]
            )

    @staticmethod
    def get_frontend_spec() -> Frontend:
        return StackTraceLens.StackTraceFrontend()
