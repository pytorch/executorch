#!/usr/bin/env python3
# Copyright (c) Qualcomm Innovation Center, Inc.
# All rights reserved
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Generate an interactive observatory UI harness for JS/HTML test cases."""

from __future__ import annotations

import argparse
import base64
import json
import os

from executorch.backends.qualcomm.debugger.observatory.html_template import get_html_template
from executorch.backends.qualcomm.utils.fx_viewer.exporter import FXGraphExporter


def _sample_graph_assets() -> dict:
    nodes = [
        {
            "id": "n0",
            "label": "input",
            "x": 0,
            "y": 0,
            "width": 110,
            "height": 36,
            "info": {"name": "n0", "op": "placeholder", "target": "x"},
            "tooltip": ["Name: n0", "Op: placeholder", "Target: x"],
        },
        {
            "id": "n1",
            "label": "linear",
            "x": 200,
            "y": 0,
            "width": 120,
            "height": 36,
            "info": {"name": "n1", "op": "call_module", "target": "linear"},
            "tooltip": ["Name: n1", "Op: call_module", "Target: linear"],
        },
    ]
    edges = [{"v": "n0", "w": "n1", "points": []}]

    return {
        "graph_assets": {
            "record_0": {
                "base": {"legend": [], "nodes": nodes, "edges": edges},
                "meta": {"record_name": "record_0", "node_count": 2},
            },
            "record_1": {
                "base": {"legend": [], "nodes": nodes, "edges": edges},
                "meta": {"record_name": "record_1", "node_count": 2},
            },
        },
        "graph_layers": {
            "record_0": {
                "accuracy/error": {
                    "name": "Accuracy Error",
                    "legend": [
                        {"label": "Low", "color": "#93c5fd"},
                        {"label": "High", "color": "#b91c1c"},
                    ],
                    "nodes": {
                        "n0": {"info": {"mse": 0.0}, "label_append": ["mse=0.0"], "fill_color": "#93c5fd"},
                        "n1": {"info": {"mse": 0.2}, "label_append": ["mse=0.2"], "fill_color": "#b91c1c"},
                    },
                }
            },
            "record_1": {
                "accuracy/error": {
                    "name": "Accuracy Error",
                    "legend": [
                        {"label": "Low", "color": "#93c5fd"},
                        {"label": "High", "color": "#b91c1c"},
                    ],
                    "nodes": {
                        "n0": {"info": {"mse": 0.0}, "label_append": ["mse=0.0"], "fill_color": "#93c5fd"},
                        "n1": {"info": {"mse": 0.6}, "label_append": ["mse=0.6"], "fill_color": "#b91c1c"},
                    },
                }
            },
        },
    }


def _custom_js() -> str:
    return """
window.renderHarnessCustom = function(container, args, context, analysis) {
  const p = document.createElement('p');
  p.textContent = 'Custom block says: ' + (args.message || 'hello');
  container.appendChild(p);
  const btn = document.createElement('button');
  btn.textContent = 'Go Compare 0,1';
  btn.setAttribute('data-ob-action', 'open-compare');
  btn.setAttribute('data-ob-indices', '0,1');
  container.appendChild(btn);
};
"""


def build_payload() -> dict:
    graph_data = _sample_graph_assets()

    blocks_record_0 = {
        "blocks": [
            {
                "id": "meta",
                "title": "Metadata",
                "type": "table",
                "record": {"data": {"artifact_type": "GM", "node_count": 2}},
                "compare": {"mode": "auto"},
                "order": 0,
                "collapsible": True,
            },
            {
                "id": "custom",
                "title": "Custom Action",
                "type": "custom",
                "record": {"js_func": "renderHarnessCustom", "args": {"message": "record_0"}},
                "compare": {"mode": "disabled"},
                "order": 5,
                "collapsible": True,
            },
            {
                "id": "graph",
                "title": "Graph",
                "type": "graph",
                "record": {
                    "graph_ref": "record_0",
                    "default_layers": ["accuracy/error"],
                    "default_color_by": "accuracy/error",
                    "viewer_options": {"layout_mode": "full"},
                },
                "compare": {
                    "mode": "auto",
                    "max_parallel": 2,
                    "sync_toggle": True,
                    "viewer_options_compare": {"layout_mode": "compare_compact"},
                },
                "order": 10,
                "collapsible": True,
            },
        ]
    }

    blocks_record_1 = json.loads(json.dumps(blocks_record_0))
    blocks_record_1["blocks"][0]["record"]["data"]["node_count"] = 3
    blocks_record_1["blocks"][1]["record"]["args"]["message"] = "record_1"
    blocks_record_1["blocks"][2]["record"]["graph_ref"] = "record_1"

    resources = {
        "js": [
            base64.b64encode(s.encode("utf-8")).decode("ascii")
            for s in [FXGraphExporter._load_viewer_js_bundle(), _custom_js()]
        ],
        "css": [],
    }

    payload = {
        "title": "Observatory UI Harness",
        "generated_at": "N/A",
        "resources": resources,
        "records": [
            {
                "name": "Record 0",
                "timestamp": "N/A",
                "views": {"tutorial": blocks_record_0},
                "badges": [{"label": "GM", "class": "badge", "title": "GraphModule"}],
                "diff_index": {},
                "digests": {},
            },
            {
                "name": "Record 1",
                "timestamp": "N/A",
                "views": {"tutorial": blocks_record_1},
                "badges": [{"label": "GM", "class": "badge", "title": "GraphModule"}],
                "diff_index": {"nodes": "+1"},
                "digests": {},
            },
        ],
        "dashboard": {
            "tutorial": {
                "blocks": [
                    {
                        "id": "dashboard",
                        "title": "Harness Dashboard",
                        "type": "html",
                        "record": {
                            "content": "<p>This dashboard validates block rendering and action delegation.</p>"
                        },
                        "compare": {"mode": "disabled"},
                        "order": 0,
                        "collapsible": True,
                    }
                ]
            }
        },
        "analysis_results": {},
        "session": {"start_data": {}, "end_data": {}},
        "graph_assets": graph_data["graph_assets"],
        "graph_layers": graph_data["graph_layers"],
    }

    return payload


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate observatory UI harness HTML")
    default_output = os.path.join(os.path.dirname(__file__), "observatory_ui_harness.html")
    parser.add_argument(
        "--output",
        default=default_output,
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    payload = build_payload()
    html = get_html_template(payload["title"], json.dumps(payload))

    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as f:
        f.write(html)

    print(args.output)


if __name__ == "__main__":
    main()
