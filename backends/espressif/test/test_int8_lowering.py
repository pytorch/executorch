# Copyright 2026 Mwaniki Njagi.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.
#
# SPDX-License-Identifier: BSD-3-Clause

"""
Host tests for the ESP32 int8 quantize-and-lower pipeline.

    pytest backends/espressif/test/test_int8_lowering.py
"""

from __future__ import annotations

import random
import struct

import pytest

np = pytest.importorskip("numpy")
pytest.importorskip("torchao")

import torch
import torch.nn as nn
import torch.nn.functional as F
from executorch.backends.espressif.quantizer.espnn_quantizer import EspNNQuantizer

from executorch.backends.espressif.reference import esp_nn_reference as reference
from executorch.backends.espressif.serialization.xts3 import (
    _quantize_multiplier,
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


INT32_MIN = -(1 << 31)
INT32_MAX = (1 << 31) - 1


def c_srdhm(a: int, b: int) -> int:
    if a == b == INT32_MIN:
        return INT32_MAX
    nudge = (1 << 30) if (a >= 0) == (b >= 0) else 1 - (1 << 30)
    product = a * b + nudge
    quotient = abs(product) >> 31
    return quotient if product >= 0 else -quotient


def c_rounding_divide(x: int, exponent: int) -> int:
    mask = (1 << exponent) - 1
    remainder = x & mask
    result = x >> exponent
    threshold = (mask >> 1) + (1 if result < 0 else 0)
    return result + (1 if remainder > threshold else 0)


def c_requantize(x: int, multiplier: int, shift: int) -> int:
    left = max(shift, 0)
    right = max(-shift, 0)
    shifted = ((x << left) + (1 << 31)) % (1 << 32) - (1 << 31)
    return c_rounding_divide(c_srdhm(shifted, multiplier), right)


def test_requantization_matches_c_semantics():
    rng = random.Random(0)
    xs, mults, shifts = [], [], []
    for _ in range(20000):
        xs.append(rng.randint(-(1 << 24), 1 << 24))
        mults.append(rng.randint(1 << 30, INT32_MAX))
        shifts.append(rng.randint(-20, 2))
    xs += [INT32_MIN, INT32_MAX, 0, -1, 1]
    mults += [INT32_MAX, 1 << 30, 1 << 30, INT32_MAX, INT32_MAX]
    shifts += [0, -1, -31, -5, 0]
    expected = [c_requantize(x, m, s) for x, m, s in zip(xs, mults, shifts)]
    actual = reference.multiply_by_quantized_multiplier(
        np.array(xs, dtype=np.int64), np.array(mults), np.array(shifts)
    )
    assert actual.tolist() == expected


def test_srdhm_overflow_case():
    assert (
        int(reference.saturating_rounding_doubling_high_mul(INT32_MIN, INT32_MIN))
        == INT32_MAX
    )


@pytest.mark.parametrize("value", [1e-6, 0.0123, 0.5, 0.999999, 1.0, 3.75, 1e3])
def test_quantize_multiplier_reconstructs_value(value):
    multiplier, shift = _quantize_multiplier(value)
    assert (1 << 30) <= multiplier <= INT32_MAX
    assert abs(multiplier * 2.0 ** (shift - 31) - value) <= value * 2**-30


class ConvActModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(3, 16, 3, stride=2, padding=1)
        self.dw = nn.Conv2d(16, 16, 3, padding=1, groups=16)
        self.grouped = nn.Conv2d(16, 16, 3, padding=1, groups=4)
        self.pw = nn.Conv2d(16, 8, 1)
        self.fc = nn.Linear(8 * 2 * 2, 10)

    def forward(self, x):
        x = F.relu6(self.conv(x))
        y = F.relu(self.dw(x))
        x = F.relu(self.grouped(y) + x)
        x = F.max_pool2d(x, 3, 2, 1)
        x = self.pw(x)
        x = F.avg_pool2d(F.relu(x), 2)
        return self.fc(torch.flatten(x, 1))


class ConcatModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.a = nn.Conv2d(4, 6, 1)
        self.b = nn.Conv2d(4, 10, 3, padding=1)
        self.head = nn.Linear(16, 5)

    def forward(self, x):
        x = torch.cat([F.relu(self.a(x)), F.relu(self.b(x))], dim=1)
        x = F.adaptive_avg_pool2d(x, 1)
        return self.head(torch.flatten(x, 1))


class SEBlockModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(3, 16, 3, stride=2, padding=1)
        self.act = nn.Hardswish()
        self.fc1 = nn.Conv2d(16, 8, 1)
        self.fc2 = nn.Conv2d(8, 16, 1)
        self.gate = nn.Hardsigmoid()
        self.dw = nn.Conv2d(16, 16, 3, padding=1, groups=16)
        self.head = nn.Linear(16, 6)

    def forward(self, x):
        x = self.act(self.conv(x))
        scale = self.gate(self.fc2(F.relu(self.fc1(F.adaptive_avg_pool2d(x, 1)))))
        x = x * scale
        x = F.silu(self.dw(x))
        x = torch.sigmoid(x) * x + x * 0.5 + 1.0
        x = torch.flatten(F.adaptive_avg_pool2d(x, 1), 1)
        return torch.softmax(self.head(x), dim=1)


class UnsupportedModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(3, 4, 3)

    def forward(self, x):
        return torch.cumsum(self.conv(x), dim=1)


def quantize(model: nn.Module, shape, samples: int = 16, seed: int = 0):
    torch.manual_seed(seed)
    model = model.eval()
    example = (torch.randn(shape),)
    prepared = prepare_pt2e(
        torch.export.export(model, example, strict=True).module(), EspNNQuantizer()
    )
    for _ in range(samples):
        prepared(torch.randn(shape))
    converted = convert_pt2e(prepared)
    return converted, torch.export.export(converted, example, strict=True)


def run_device_model(graph, x: torch.Tensor):
    interpreter = reference.ReferenceInterpreter(graph)
    source = graph.tensors[interpreter.inputs[0]]
    result = graph.tensors[interpreter.outputs[0]]
    quantized = reference.to_device_layout(
        source, reference.quantize(source, x.numpy())
    )
    output = interpreter.run({source.tensor_id: quantized})[result.tensor_id]
    return result, output


@pytest.mark.parametrize(
    "model, shape",
    [
        (ConvActModel(), (1, 3, 16, 16)),
        (ConcatModel(), (1, 4, 6, 6)),
        (SEBlockModel(), (1, 3, 16, 16)),
    ],
    ids=["conv-act-pool-linear", "concat-mean-linear", "se-lut-mul-softmax"],
)
def test_lowering_tracks_pytorch_int8(model, shape):
    converted, program = quantize(model, shape)
    graph = parse_graph(serialize_graph(build_graph(program)))
    assert graph.version == XTS3_VERSION
    for seed in range(4):
        torch.manual_seed(100 + seed)
        x = torch.randn(shape)
        result, output = run_device_model(graph, x)
        expected = converted(x).reshape(-1).numpy()
        expected_q = np.clip(
            np.rint(expected / result.quant_scale) + result.quant_zero_point, -128, 127
        )
        assert np.max(np.abs(expected_q - output.reshape(-1).astype(np.int64))) <= 2


def test_relu6_is_fused_into_convolution():
    _, program = quantize(ConvActModel(), (1, 3, 16, 16))
    graph = build_graph(program)
    opcodes = [op.opcode for op in graph.operations]
    assert XTS3OpCode.RELU6 not in opcodes
    conv = graph.operations[0]
    output = graph.tensors[conv.outputs[0]]
    act_min, act_max = struct.unpack_from("<ii", conv.params, 12)
    assert act_min == max(-128, output.quant_zero_point)
    assert act_max == min(
        127, output.quant_zero_point + int(6.0 / output.quant_scale + 0.5)
    )


def test_serialization_round_trip_and_alignment():
    _, program = quantize(ConcatModel(), (1, 4, 6, 6))
    graph = build_graph(program)
    blob = serialize_graph(graph)
    parsed = parse_graph(blob)
    assert parsed.constant_data == graph.constant_data
    assert (len(blob) - len(parsed.constant_data)) % 16 == 0
    assert [t.shape for t in parsed.tensors] == [t.shape for t in graph.tensors]
    assert [op.params for op in parsed.operations] == [
        op.params for op in graph.operations
    ]
    assert serialize_graph(parsed) == blob


def test_unsupported_operator_is_reported():
    _, program = quantize(UnsupportedModel(), (1, 3, 8, 8))
    with pytest.raises(ValueError, match="cumsum"):
        build_graph(program)


def test_se_block_uses_lut_broadcast_mul_and_softmax():
    _, program = quantize(SEBlockModel(), (1, 3, 16, 16))
    graph = build_graph(program)
    counts = {}
    for op in graph.operations:
        counts[op.opcode] = counts.get(op.opcode, 0) + 1
    assert counts.get(XTS3OpCode.LUT, 0) >= 5
    assert counts.get(XTS3OpCode.SOFTMAX) == 1
    modes = [
        struct.unpack("<5i", op.params)[4]
        for op in graph.operations
        if op.opcode == XTS3OpCode.MUL
    ]
    assert 1 in modes and 0 in modes
    softmax = [op for op in graph.operations if op.opcode == XTS3OpCode.SOFTMAX][0]
    output = graph.tensors[softmax.outputs[0]]
    assert output.quant_zero_point == -128 and abs(output.quant_scale - 1 / 256) < 1e-12


def test_lut_matches_pytorch_quantized_function():
    from executorch.backends.espressif.serialization.xts3 import (
        build_lut,
        XTS3DType,
        XTS3Tensor,
    )

    source = XTS3Tensor(
        0, (1, 4), XTS3DType.INT8, quant_scale=0.05, quant_zero_point=-3
    )
    result = XTS3Tensor(
        1, (1, 4), XTS3DType.INT8, quant_scale=0.03, quant_zero_point=-100
    )
    table = np.frombuffer(build_lut(F.hardswish, source, result), dtype=np.int8)
    codes = torch.arange(-128, 128).to(torch.int8)
    expected = torch.ops.quantized_decomposed.quantize_per_tensor(
        F.hardswish(
            torch.ops.quantized_decomposed.dequantize_per_tensor(
                codes, 0.05, -3, -128, 127, torch.int8
            )
        ),
        0.03,
        -100,
        -128,
        127,
        torch.int8,
    )
    assert table.tolist() == expected.tolist()


def test_float_model_cannot_be_lowered_to_int8():
    program = torch.export.export(
        ConcatModel().eval(), (torch.randn(1, 4, 6, 6),), strict=True
    )
    with pytest.raises(ValueError):
        build_graph(program)


def test_executorch_delegate_matches_standalone_blob():
    from executorch.backends.espressif.partition.espnn_partitioner import (
        EspNNPartitioner,
    )
    from executorch.exir import EdgeCompileConfig, to_edge_transform_and_lower
    from executorch.exir.lowered_backend_module import LoweredBackendModule

    _, program = quantize(SEBlockModel(), (1, 3, 16, 16))
    standalone = serialize_graph(build_graph(program))
    edge = to_edge_transform_and_lower(
        program,
        partitioner=[EspNNPartitioner(allow_cpu_fallback=False)],
        compile_config=EdgeCompileConfig(_check_ir_validity=False),
    )
    delegates = [
        module
        for _, module in edge.exported_program().graph_module.named_children()
        if isinstance(module, LoweredBackendModule)
    ]
    assert len(delegates) == 1
    assert bytes(delegates[0].processed_bytes) == standalone
    assert len(edge.to_executorch().buffer) > len(standalone)


def test_memory_plan_reuses_buffers():
    _, program = quantize(ConvActModel(), (1, 3, 16, 16))
    graph = build_graph(program)
    arena, offsets = reference.plan_activation_arena(graph)
    total = sum(
        (int(np.prod(t.shape)) + 15) // 16 * 16
        for t in graph.tensors
        if not t.is_constant
    )
    assert 0 < arena < total
    assert all(offset % 16 == 0 for offset in offsets.values())
