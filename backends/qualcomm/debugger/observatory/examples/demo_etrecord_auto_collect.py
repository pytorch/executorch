#!/usr/bin/env python3
# Copyright (c) Qualcomm Innovation Center, Inc.
# All rights reserved
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Demonstrate ETRecord monkey-patch auto collection for Observatory.

This script intentionally avoids manual `Observatory.collect(...)` for graph capture.
When context is enabled, ETRecord API calls are patched and collected automatically.

Run from repository root:

    source ~/executorch/.venv/bin/activate
    source ~/executorch/qairt/2.37.0.250724/bin/envsetup.sh
    export PYTHONPATH=~/

    python backends/qualcomm/debugger/observatory/examples/demo_etrecord_auto_collect.py \
        --output-dir /tmp/observatory_etrecord_demo
"""

from __future__ import annotations

import argparse
import json
import os

import torch

from executorch.devtools.observatory import Observatory
from executorch.devtools.observatory.lenses.pipeline_graph_collector import (
    PipelineGraphCollectorLens,
)
from executorch.devtools.etrecord import ETRecord


class _DemoModel(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.fc1 = torch.nn.Linear(16, 16)
        self.fc2 = torch.nn.Linear(16, 4)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = torch.relu(self.fc1(x))
        return self.fc2(x)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="ETRecord auto-collection demo")
    parser.add_argument("--output-dir", default="/tmp/observatory_etrecord_demo")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    model = _DemoModel().eval()
    sample_inputs = (torch.rand(1, 16),)
    exported_program = torch.export.export(model, sample_inputs, strict=False)

    Observatory.clear()
    Observatory.register_lens(PipelineGraphCollectorLens)

    with Observatory.enable_context():
        # No manual Observatory.collect call here.
        etrecord = ETRecord()
        etrecord.add_exported_program(exported_program)

    html_path = os.path.join(args.output_dir, "etrecord_auto_collect_report.html")
    json_path = os.path.join(args.output_dir, "etrecord_auto_collect_report.json")

    Observatory.export_html_report(html_path, title="ETRecord Auto Collect Demo")
    Observatory.export_json(json_path)

    print(
        json.dumps(
            {
                "report_html": html_path,
                "report_json": json_path,
                "collected_records": Observatory.list_collected(),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
