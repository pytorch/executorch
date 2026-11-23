# Copyright 2026 Arm Limited and/or its affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import pytest
import torch
from examples.arm.QAT_example.qat_loop import make_quantizer
from torchao.quantization.pt2e.fake_quantize import (
    disable_fake_quant,
    disable_observer,
    enable_fake_quant,
)
from torchao.quantization.pt2e.quantize_pt2e import (
    convert_pt2e,
    prepare_pt2e,
    prepare_qat_pt2e,
)


class _ImageEncoder(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(6, 4, 1)

    def forward(self, first, second):
        return self.conv(torch.cat((first, second), dim=1))


@pytest.mark.parametrize("qat", [False, True])
def test_uint8_image_inputs_keep_fixed_qparams_and_signed_internal_boundary(qat):
    inputs = (
        torch.full((1, 3, 4, 4), 0.2),
        torch.full((1, 3, 4, 4), 0.8),
    )
    model = torch.export.export(_ImageEncoder().eval(), inputs).module()
    quantizer = make_quantizer(is_qat=qat, io_quantization="uint8", accuracy=True)
    prepare = prepare_qat_pt2e if qat else prepare_pt2e
    prepared = prepare(model, quantizer)
    if qat:
        prepared.apply(disable_fake_quant)
    prepared(*inputs)
    if qat:
        prepared.apply(disable_observer)
        prepared.apply(enable_fake_quant)
        prepared(*inputs).sum().backward()
    converted = convert_pt2e(prepared)
    quantize = torch.ops.quantized_decomposed.quantize_per_tensor.default
    dequantize = torch.ops.quantized_decomposed.dequantize_per_tensor.default
    placeholders = [n for n in converted.graph.nodes if n.op == "placeholder"]
    assert len(placeholders) == 2
    for image in placeholders:
        (external,) = image.users
        assert external.target == quantize
        assert external.args[1] == pytest.approx(1 / 255)
        assert external.args[2:] == (0, 0, 255, torch.uint8)
        (decoded,) = external.users
        assert decoded.target == dequantize
        (internal,) = decoded.users
        assert internal.target == quantize
        assert internal.args[1] == pytest.approx(1 / 254)
        assert internal.args[2:] == (-127, -127, 127, torch.int8)
    assert torch.isfinite(converted(*inputs)).all()
