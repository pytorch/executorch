# Copyright 2026 Mwaniki Njagi.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.
#
# SPDX-License-Identifier: BSD-3-Clause

"""
Host tests for the XTS3 v6 additions: data movement (COPY), upsampling,
transposed and dilated convolution, broadcasting, padding, any-axis concat,
split, and multiple graph inputs and outputs.

    pytest backends/espressif/test/test_int8_ops.py
"""

from __future__ import annotations

import pytest

np = pytest.importorskip("numpy")
pytest.importorskip("torchao")

import torch
import torch.nn as nn
import torch.nn.functional as F
from executorch.backends.espressif.quantizer.espnn_quantizer import EspNNQuantizer

from executorch.backends.espressif.reference import esp_nn_reference as reference
from executorch.backends.espressif.serialization.xts3 import (
    build_graph,
    parse_graph,
    serialize_graph,
    XTS3_VERSION,
    XTS3OpCode,
)
from torchao.quantization.pt2e.quantize_pt2e import convert_pt2e, prepare_pt2e


@pytest.fixture(autouse=True)
def disable_grad():
    with torch.no_grad():
        yield


def quantize(model: nn.Module, shapes, samples: int = 24, seed: int = 0):
    torch.manual_seed(seed)
    model = model.eval()
    example = tuple(torch.randn(shape) for shape in shapes)
    prepared = prepare_pt2e(
        torch.export.export(model, example, strict=True).module(), EspNNQuantizer()
    )
    for _ in range(samples):
        prepared(*(torch.randn(shape) for shape in shapes))
    converted = convert_pt2e(prepared)
    return converted, torch.export.export(converted, example, strict=True)


def lower(program):
    graph = parse_graph(serialize_graph(build_graph(program)))
    assert graph.version == XTS3_VERSION
    return graph


def run_reference(graph, inputs):
    interpreter = reference.ReferenceInterpreter(graph)
    device_inputs = []
    for tensor_id, value in zip(interpreter.inputs, inputs):
        tensor = graph.tensors[tensor_id]
        device_inputs.append(
            reference.to_device_layout(
                tensor, reference.quantize(tensor, value.numpy())
            )
        )
    outputs = interpreter.run_outputs(device_inputs)
    return [
        (graph.tensors[tensor_id], output)
        for tensor_id, output in zip(interpreter.outputs, outputs)
    ]


def max_int8_error(converted, graph, shapes, trials: int = 4) -> int:
    worst = 0
    for trial in range(trials):
        torch.manual_seed(1000 + trial)
        inputs = [torch.randn(shape) for shape in shapes]
        expected = converted(*inputs)
        if isinstance(expected, torch.Tensor):
            expected = (expected,)
        results = run_reference(graph, inputs)
        assert len(results) == len(expected)
        for (tensor, device_output), want in zip(results, expected):
            logical = reference.to_logical_layout(tensor, device_output).astype(
                np.int64
            )
            assert logical.shape == tuple(want.shape)
            want_q = np.clip(
                np.rint(want.numpy() / tensor.quant_scale) + tensor.quant_zero_point,
                -128,
                127,
            )
            worst = max(worst, int(np.max(np.abs(want_q - logical))))
    return worst


class DetectionHead(nn.Module):
    def __init__(self, anchors=3, classes=5):
        super().__init__()
        self.anchors, self.classes = anchors, classes
        self.backbone = nn.Conv2d(3, 16, 3, stride=2, padding=1)
        self.p4 = nn.Conv2d(16, 16, 3, stride=2, padding=1)
        self.head3 = nn.Conv2d(16, anchors * (classes + 4), 1)
        self.head4 = nn.Conv2d(16, anchors * (classes + 4), 1)

    def _flatten(self, x):
        n, _, h, w = x.shape
        x = x.view(n, self.anchors, self.classes + 4, h, w)
        return x.permute(0, 1, 3, 4, 2).reshape(n, -1, self.classes + 4)

    def forward(self, x):
        f3 = F.relu(self.backbone(x))
        f4 = F.relu(self.p4(f3))
        boxes = torch.cat(
            [self._flatten(self.head3(f3)), self._flatten(self.head4(f4))], dim=1
        )
        return boxes[..., :4], boxes[..., 4:]


class SegmentationDecoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = nn.Conv2d(3, 8, 3, stride=2, padding=1)
        self.dilated = nn.Conv2d(8, 8, 3, padding=2, dilation=2)
        self.dilated_dw = nn.Conv2d(8, 8, 3, padding=3, dilation=3, groups=8)
        self.up = nn.ConvTranspose2d(8, 6, 3, stride=2, padding=1, output_padding=1)
        self.skip = nn.Conv2d(3, 6, 1)

    def forward(self, x):
        y = F.relu(self.encoder(x))
        y = F.relu(self.dilated(y))
        y = F.relu(self.dilated_dw(y))
        y = self.up(y)
        return F.relu(y + self.skip(x))


class UpsampleModel(nn.Module):
    def __init__(self, mode: str, align_corners=None, scale=2.0, size=None):
        super().__init__()
        self.mode, self.align_corners, self.scale, self.size = (
            mode,
            align_corners,
            scale,
            size,
        )
        self.conv = nn.Conv2d(3, 6, 3, stride=2, padding=1)
        self.out = nn.Conv2d(6, 4, 1)

    def forward(self, x):
        y = F.relu(self.conv(x))
        if self.size is not None:
            y = F.interpolate(
                y, size=self.size, mode=self.mode, align_corners=self.align_corners
            )
        else:
            y = F.interpolate(
                y,
                scale_factor=self.scale,
                mode=self.mode,
                align_corners=self.align_corners,
            )
        return self.out(y)


class BroadcastPadSplit(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(4, 8, 3, padding=1)
        self.row_gate = nn.Conv2d(4, 8, (3, 1), padding=(1, 0))
        self.register_buffer(
            "bias_map", torch.linspace(-1, 1, 8 * 6).reshape(1, 8, 1, 6)
        )
        self.mix = nn.Conv2d(4, 4, 1)

    def forward(self, x):
        y = self.conv(x)
        gate = F.adaptive_avg_pool2d(self.row_gate(x), (1, 6))
        y = F.relu(y + gate)
        y = y * torch.sigmoid(gate)
        y = y + self.bias_map
        a, b = torch.split(y, [4, 4], dim=1)
        a = F.pad(a, (1, 1, 2, 0))
        b = F.pad(b, (1, 1, 2, 0))
        return self.mix(torch.cat([a, b], dim=2))


class TwoInputTwoOutput(nn.Module):
    def __init__(self):
        super().__init__()
        self.image = nn.Conv2d(3, 8, 3, stride=2, padding=1)
        self.extra = nn.Linear(6, 8)
        self.classifier = nn.Linear(8, 4)
        self.mask = nn.Conv2d(8, 1, 1)

    def forward(self, image, features):
        x = F.relu(self.image(image))
        y = F.relu(self.extra(features))
        pooled = torch.flatten(F.adaptive_avg_pool2d(x, 1), 1)
        logits = self.classifier(pooled + y)
        mask = torch.sigmoid(self.mask(x))
        return mask, logits


class TransposeSelect(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(3, 6, 3, padding=1)
        self.fc = nn.Linear(6, 5)

    def forward(self, x):
        y = F.relu(self.conv(x))
        tokens = y.flatten(2).transpose(1, 2)
        first = tokens[:, 0, :]
        last = tokens.select(1, -1)
        mixed = torch.stack([first, last], dim=1)
        return self.fc(mixed).unsqueeze(-1)


torch.manual_seed(20261014)


class GroupedConvs(nn.Module):
    """Grouped (not depthwise) convolutions: plain, strided and dilated."""

    def __init__(self):
        super().__init__()
        self.plain = nn.Conv2d(16, 16, 3, padding=1, groups=4)
        self.strided = nn.Conv2d(16, 24, 3, stride=2, padding=1, groups=8)
        self.dilated = nn.Conv2d(24, 24, 3, padding=2, dilation=2, groups=3)

    def forward(self, x):
        x = F.relu(self.plain(x))
        x = F.relu(self.strided(x))
        return self.dilated(x)


CASES = [
    ("grouped-conv", GroupedConvs(), [(1, 16, 12, 12)], 2),
    ("detection-head-multi-output", DetectionHead(), [(1, 3, 16, 16)], 2),
    ("segmentation-dilated-transposed", SegmentationDecoder(), [(1, 3, 16, 16)], 3),
    ("upsample-nearest-x2", UpsampleModel("nearest"), [(1, 3, 16, 16)], 2),
    (
        "upsample-bilinear-x2",
        UpsampleModel("bilinear", align_corners=False),
        [(1, 3, 16, 16)],
        2,
    ),
    (
        "upsample-bilinear-align",
        UpsampleModel("bilinear", align_corners=True, size=(13, 11)),
        [(1, 3, 16, 16)],
        2,
    ),
    ("broadcast-pad-split-cat", BroadcastPadSplit(), [(1, 4, 6, 6)], 2),
    ("two-inputs-two-outputs", TwoInputTwoOutput(), [(1, 3, 12, 12), (1, 6)], 2),
    ("transpose-select-stack", TransposeSelect(), [(1, 3, 4, 5)], 2),
]


@pytest.mark.parametrize(
    "name, model, shapes, tolerance", CASES, ids=[c[0] for c in CASES]
)
def test_v6_lowering_tracks_pytorch_int8(name, model, shapes, tolerance):
    converted, program = quantize(model, shapes)
    graph = lower(program)
    assert len(graph.inputs) == len(shapes)
    error = max_int8_error(converted, graph, shapes)
    assert error <= tolerance, f"{name}: worst int8 difference {error}"


def test_detection_head_uses_copy_and_no_rank5_tensors():
    _, program = quantize(DetectionHead(), [(1, 3, 16, 16)])
    graph = lower(program)
    assert any(op.opcode == XTS3OpCode.COPY for op in graph.operations)
    assert all(len(t.shape) <= 4 for t in graph.tensors)
    assert len(graph.outputs) == 2


def test_io_table_round_trips_order():
    _, program = quantize(TwoInputTwoOutput(), [(1, 3, 12, 12), (1, 6)])
    graph = build_graph(program)
    parsed = parse_graph(serialize_graph(graph))
    assert parsed.inputs == graph.inputs and parsed.outputs == graph.outputs
    assert [parsed.tensors[i].shape for i in parsed.inputs] == [(1, 3, 12, 12), (1, 6)]
    assert [parsed.tensors[i].shape for i in parsed.outputs] == [(1, 1, 6, 6), (1, 4)]


def test_constants_are_16_byte_aligned():
    _, program = quantize(BroadcastPadSplit(), [(1, 4, 6, 6)])
    blob = serialize_graph(build_graph(program))
    parsed = parse_graph(blob)
    constant_base = len(blob) - len(parsed.constant_data)
    assert constant_base % 16 == 0


def test_constant_records_match_their_device_layout():
    """Regression: transposed-conv weights were stored (Cout,kh,kw,Cin) but recorded as (Cin,Cout,kh,kw)."""
    _, program = quantize(SegmentationDecoder(), [(1, 3, 16, 16)])
    graph = lower(program)
    transposed = [
        op for op in graph.operations if op.opcode == XTS3OpCode.TRANSPOSE_CONV2D
    ]
    assert transposed
    for op in transposed:
        weight = graph.tensors[op.inputs[1]]
        source = graph.tensors[op.inputs[0]]
        result = graph.tensors[op.outputs[0]]
        assert weight.shape[0] == result.shape[1] and weight.shape[3] == source.shape[1]
    for tensor in graph.tensors:
        if tensor.is_constant:
            assert tensor.data_size == int(np.prod(tensor.shape)) * (
                4 if tensor.dtype == 4 else 1
            )
