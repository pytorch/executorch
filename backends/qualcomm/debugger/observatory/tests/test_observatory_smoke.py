# Copyright (c) Qualcomm Innovation Center, Inc.
# All rights reserved
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import torch

from executorch.backends.qualcomm.debugger.observatory import Observatory


class _SmokeModel(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.fc = torch.nn.Linear(4, 4)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc(x)


def test_observatory_collect_and_export_html(tmp_path) -> None:
    Observatory.clear()

    model = _SmokeModel().eval()
    graph_module = torch.fx.symbolic_trace(model)

    with Observatory.enable_context():
        Observatory.collect("smoke", graph_module)

    out = tmp_path / "report.html"
    Observatory.export_html_report(str(out), title="Smoke")

    assert out.exists()
    assert out.stat().st_size > 0

    Observatory.clear()
