# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Compare-view HTML exporter for fx_viewer.

Orchestrates N `FXGraphExporter` instances into one interactive HTML page
that mounts each viewer and wires `FXGraphCompare` for cross-graph selection
sync. Column order follows the insertion order of `viewers`.
"""

from __future__ import annotations

import json
from collections import OrderedDict
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

from .exporter import FXGraphExporter

_ViewerArg = Union[
    "OrderedDict[str, FXGraphExporter]",
    Dict[str, FXGraphExporter],
    Sequence[Tuple[str, FXGraphExporter]],
]

_KNOWN_SYNC_MODES = frozenset({"auto", "id", "layer", "none"})


class FXGraphCompareExporter:
    """Export N FX graphs as one interactive HTML compare view.

    Example:
        >>> from executorch.devtools.fx_viewer import (
        ...     FXGraphCompareExporter, FXGraphExporter,
        ... )
        >>> from collections import OrderedDict
        >>> viewers = OrderedDict(
        ...     [("Aten", FXGraphExporter(aten_gm)),
        ...      ("Edge", FXGraphExporter(edge_gm))]
        ... )
        >>> FXGraphCompareExporter(viewers).export_html("compare.html")

    Sync modes:
        * ``"auto"`` (default) — try ``from_node_root`` first, then
          ``debug_handle`` set-intersection, then node id.
        * ``"id"`` — node id match only.
        * ``"layer"`` — match by ``extensions[sync_layer].nodes[nodeId].info[sync_field]``.
        * ``"none"`` — no propagation.

    Args:
        viewers: ordered mapping/pairs of ``{column_name: FXGraphExporter}``.
        title: browser tab and page title.
        sync_mode: one of ``"auto" | "id" | "layer" | "none"``.
        sync_layer, sync_field: required iff ``sync_mode == "layer"``.
        active_extensions: extension ids to enable at mount time; applied
            uniformly to every viewer.
        color_by: extension id used for color-by at mount time; applied
            uniformly to every viewer.
    """

    def __init__(
        self,
        viewers: _ViewerArg,
        *,
        title: str = "FX Graph Compare",
        sync_mode: str = "auto",
        sync_layer: Optional[str] = None,
        sync_field: Optional[str] = None,
        active_extensions: Optional[List[str]] = None,
        color_by: Optional[str] = None,
    ) -> None:
        pairs: List[Tuple[str, FXGraphExporter]]
        if isinstance(viewers, dict):
            pairs = list(viewers.items())
        else:
            pairs = [(name, exp) for name, exp in viewers]

        if not pairs:
            raise ValueError("FXGraphCompareExporter requires at least one viewer")
        seen = set()
        for name, exp in pairs:
            if not isinstance(name, str) or not name:
                raise ValueError(
                    f"viewer name must be a non-empty string, got {name!r}"
                )
            if name in seen:
                raise ValueError(f"duplicate viewer name: {name!r}")
            seen.add(name)
            if not isinstance(exp, FXGraphExporter):
                raise TypeError(
                    f"viewer {name!r} must be FXGraphExporter, got {type(exp).__name__}"
                )

        if sync_mode not in _KNOWN_SYNC_MODES:
            raise ValueError(
                f"sync_mode must be one of {sorted(_KNOWN_SYNC_MODES)}, got {sync_mode!r}"
            )
        if sync_mode == "layer" and (not sync_layer or not sync_field):
            raise ValueError(
                "sync_mode='layer' requires both sync_layer and sync_field"
            )

        self._viewers: List[Tuple[str, FXGraphExporter]] = pairs
        self._title = title
        self._sync_mode = sync_mode
        self._sync_layer = sync_layer
        self._sync_field = sync_field
        self._active_extensions: List[str] = (
            list(active_extensions) if active_extensions else []
        )
        self._color_by = color_by

    def generate_json_payload(self) -> Dict[str, Any]:
        """Return ``{title, viewers[{name, payload}], sync, state}``."""
        sync: Dict[str, Any] = {"mode": self._sync_mode}
        if self._sync_mode == "layer":
            sync["layer"] = self._sync_layer
            sync["field"] = self._sync_field
        return {
            "title": self._title,
            "viewers": [
                {"name": name, "payload": exp.generate_json_payload()}
                for name, exp in self._viewers
            ],
            "sync": sync,
            "state": {
                "activeExtensions": list(self._active_extensions),
                "colorBy": self._color_by,
            },
        }

    def export_json(self, output_path: str) -> None:
        with open(output_path, "w") as f:
            json.dump(self.generate_json_payload(), f, indent=2)

    def export_html(self, output_html: str = "compare.html") -> None:
        payload_json = json.dumps(self.generate_json_payload())
        js_bundle = FXGraphExporter._load_viewer_js_bundle()
        html = _COMPARE_HTML_TEMPLATE.format(
            title=self._title,
            payload_json=payload_json,
            js_bundle=js_bundle,
        )
        Path(output_html).write_text(html)
        print(f"Wrote compare HTML to {output_html}")


_COMPARE_HTML_TEMPLATE = r"""<!DOCTYPE html>
<html>
<head>
  <meta charset="UTF-8">
  <title>{title}</title>
  <style>
    html, body {{ margin: 0; padding: 0; width: 100%; height: 100%; font-family: sans-serif; background: #f3f4f6; }}
    .topbar {{ min-height: 40px; display: flex; align-items: center; gap: 12px; padding: 8px 14px; border-bottom: 1px solid #d1d5db; background: #ffffff; }}
    .title {{ font-weight: 600; font-size: 15px; }}
    .main {{ height: calc(100% - 48px); padding: 8px; box-sizing: border-box; }}
    #compare_root {{ width: 100%; height: 100%; }}
  </style>
</head>
<body>
  <div class="topbar"><div class="title">{title}</div></div>
  <div class="main"><div id="compare_root"></div></div>
  <script>
    const compareData = {payload_json};
  </script>
  <script>
    {js_bundle}
  </script>
  <script>
    window.onload = function() {{
      const state = compareData.state || {{}};
      const activeExtensions = Array.isArray(state.activeExtensions) ? state.activeExtensions : [];
      const colorBy = state.colorBy || null;
      const viewerMap = new Map();
      for (const entry of compareData.viewers) {{
        const viewer = FXGraphViewer.create({{
          payload: entry.payload,
          mount: {{ root: document.createElement('div') }},
          layout: {{ preset: 'split' }},
          state: {{ activeExtensions: activeExtensions.slice(), colorBy: colorBy }},
        }});
        viewer.init();
        viewerMap.set(entry.name, viewer);
      }}
      window.fxCompare = FXGraphCompare.create({{
        viewers: viewerMap,
        layout: {{ container: '#compare_root' }},
        sync: compareData.sync || {{ mode: 'auto' }},
      }});
      window.fxViewers = viewerMap;
    }};
  </script>
</body>
</html>
"""
