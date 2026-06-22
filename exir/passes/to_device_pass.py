# Copyright 2026 Arm Limited and/or its affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from typing import Set, Type

import torch
from executorch.exir.pass_base import ExportPass, PassResult


class ToDevicePass(ExportPass):
    """Call .to(device) and rewrite explicit `device=` kwargs on call_function
    nodes to given device.
    """

    _passes_required_after: Set[Type[ExportPass]] = set()

    def __init__(self, device: str | torch.device, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.device = torch.device(device)

    def call(self, graph_module: torch.fx.GraphModule) -> PassResult:
        graph_module = graph_module.to(self.device)
        modified = False

        for node in graph_module.graph.nodes:
            if node.op != "call_function" or "device" not in node.kwargs:
                continue

            current_device = node.kwargs["device"]
            if current_device == self.device:
                continue

            node.update_kwarg("device", self.device)
            modified = True

        if modified:
            graph_module.recompile()

        return PassResult(graph_module, True)

    def __call__(self, graph_module: torch.fx.GraphModule) -> PassResult:
        """Reimplement __call__ to avoid Optional[PassResult] type hint."""
        return self.call(graph_module)
