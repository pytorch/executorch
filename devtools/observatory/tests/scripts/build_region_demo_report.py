# Copyright (c) Qualcomm Innovation Center, Inc.
# All rights reserved
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Generate a small region-aware HTML report for Playwright verification.

Mirrors the typical pipeline_graph_collector + AdbLens region structure
without running an actual AOT compile / on-device inference, by
manually opening the regions and collecting torch GraphModule artifacts.

Usage:
    PYTHONPATH=~ python ~/executorch/devtools/observatory/tests/scripts/build_region_demo_report.py \
        /tmp/obs_demo_regions.html
"""

from __future__ import annotations

import sys
from pathlib import Path

import torch

from executorch.devtools.observatory import Observatory


class _Tiny(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.fc = torch.nn.Linear(4, 4)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc(x)


def main(out_path: str) -> None:
    Observatory.clear()

    sample = torch.randn(1, 4)
    model = _Tiny().eval()
    ep = torch.export.export(model, (sample,))
    gm = ep.module()

    # Mirror the lens structure: top-level Session opens and three top-level
    # regions get exercised — quantization, edge (with nested etrecord),
    # device.
    with Observatory.enter_context("aot_compiler"):
        with Observatory.enter_context("quantization"):
            Observatory.collect("Annotated Model", gm)
            Observatory.collect("Calibrated Model", gm)
            Observatory.collect("Quantized Model", gm)
        with Observatory.enter_context("edge"):
            Observatory.collect("Pre-EdgeTransform/forward", gm)
            Observatory.collect("EdgeProgramManager EP", gm)
            with Observatory.enter_context("etrecord"):
                Observatory.collect("ETRecord Exported/forward", gm)
                Observatory.collect("ETRecord Edge/forward", gm)
        with Observatory.enter_context("device"):
            Observatory.collect("adb.execute #1", gm)
            Observatory.collect("adb.execute #2", gm)

    Observatory.export_html_report(out_path)
    print(f"Wrote demo report to {out_path}")


if __name__ == "__main__":
    out = sys.argv[1] if len(sys.argv) > 1 else "/tmp/obs_demo_regions.html"
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    main(out)
