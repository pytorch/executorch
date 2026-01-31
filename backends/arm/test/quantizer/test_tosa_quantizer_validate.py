# Copyright 2026 Arm Limited and/or its affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import pytest
import torch

from executorch.backends.arm.quantizer import TOSAQuantizer
from executorch.backends.arm.tosa import TosaSpecification
from torch.fx import symbolic_trace


def _annotate_placeholders_with_devices(gm, device_map):
    for node in gm.graph.nodes:
        if node.op == "placeholder":
            device = device_map[node.target]
            node.meta["val"] = torch.empty(1, device=device)


def _get_quantizer():
    return TOSAQuantizer(TosaSpecification.create_from_string("TOSA-1.0+INT"))


class TwoIndependentAdds(torch.nn.Module):

    def forward(self, x: torch.Tensor, y: torch.Tensor):
        return x + 1, y + 1


class CrossDeviceAdd(torch.nn.Module):

    def forward(self, x: torch.Tensor, y: torch.Tensor):
        return x + y


def test_validate_allows_different_devices_across_operators():
    gm = symbolic_trace(TwoIndependentAdds())
    _annotate_placeholders_with_devices(
        gm, {"x": torch.device("cpu"), "y": torch.device("meta")}
    )

    quantizer = _get_quantizer()
    quantizer.validate(gm)


def test_validate_rejects_mixed_devices_within_operator():
    gm = symbolic_trace(CrossDeviceAdd())
    _annotate_placeholders_with_devices(
        gm, {"x": torch.device("cpu"), "y": torch.device("meta")}
    )

    quantizer = _get_quantizer()
    with pytest.raises(ValueError, match="Quantizer detected operator"):
        quantizer.validate(gm)
