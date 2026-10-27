# Copyright 2026 Mwaniki Njagi.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.
#
# SPDX-License-Identifier: BSD-3-Clause

"""
Host tests for partial delegation: int8 regions run on the ESP-NN delegate,
unsupported operators stay on the ExecuTorch CPU (portable) kernels.

    pytest backends/espressif/test/test_cpu_fallback.py
"""

from __future__ import annotations

import math

import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F
from executorch.backends.espressif.lowering import to_espnn_edge

from executorch.backends.espressif.reference.program_emulator import ProgramEmulator

from executorch.backends.espressif.test.test_int8_ops import quantize
from executorch.exir.lowered_backend_module import LoweredBackendModule


@pytest.fixture(autouse=True)
def disable_grad():
    with torch.no_grad():
        yield


def sqnr_db(reference: torch.Tensor, actual: torch.Tensor) -> float:
    reference = reference.to(torch.float64)
    noise = (reference - actual.to(torch.float64)).pow(2).sum().item()
    signal = reference.pow(2).sum().item()
    if noise == 0:
        return math.inf
    return 10 * math.log10(signal / noise)


class CumsumHead(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(3, 8, 3, padding=1)
        self.out = nn.Conv2d(8, 4, 1)

    def forward(self, x):
        y = F.relu(self.conv(x))
        y = torch.cumsum(y, dim=3)
        return self.out(y)


class TinyTransformer(nn.Module):
    def __init__(self, dim=16, heads=2, classes=5):
        super().__init__()
        self.heads = heads
        self.stem = nn.Conv2d(3, dim, 4, stride=4)
        self.norm1 = nn.LayerNorm(dim)
        self.qkv = nn.Linear(dim, 3 * dim)
        self.proj = nn.Linear(dim, dim)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp1 = nn.Linear(dim, 2 * dim)
        self.mlp2 = nn.Linear(2 * dim, dim)
        self.head = nn.Linear(dim, classes)

    def forward(self, x):
        tokens = self.stem(x).flatten(2).transpose(1, 2)
        n, t, d = tokens.shape
        h = self.norm1(tokens)
        q, k, v = self.qkv(h).chunk(3, dim=-1)
        q = q.reshape(n, t, self.heads, d // self.heads).transpose(1, 2)
        k = k.reshape(n, t, self.heads, d // self.heads).transpose(1, 2)
        v = v.reshape(n, t, self.heads, d // self.heads).transpose(1, 2)
        attention = torch.softmax(
            q @ k.transpose(-2, -1) / math.sqrt(d // self.heads), dim=-1
        )
        mixed = (attention @ v).transpose(1, 2).reshape(n, t, d)
        tokens = tokens + self.proj(mixed)
        tokens = tokens + self.mlp2(F.gelu(self.mlp1(self.norm2(tokens))))
        return self.head(tokens.mean(dim=1))


class KeywordLSTM(nn.Module):
    def __init__(self):
        super().__init__()
        self.frontend = nn.Linear(10, 16)
        self.lstm = nn.LSTM(16, 24, batch_first=True)
        self.classifier = nn.Linear(24, 6)

    def forward(self, features):
        x = F.relu(self.frontend(features))
        sequence, _ = self.lstm(x)
        return self.classifier(sequence[:, -1, :])


class CpuIsland(nn.Module):
    def __init__(self):
        super().__init__()
        self.a = nn.Conv2d(3, 8, 3, padding=1)
        self.b = nn.Conv2d(8, 8, 3, padding=1)
        self.c = nn.Conv2d(8, 4, 1)

    def forward(self, x):
        y = F.relu(self.a(x))
        z = torch.nn.functional.normalize(y, dim=1)
        z = F.relu(self.b(z)) + y
        return self.c(z)


torch.manual_seed(31)
CASES = [
    ("cumsum-on-cpu", CumsumHead(), [(1, 3, 8, 8)], 25.0),
    ("tiny-transformer", TinyTransformer(), [(1, 3, 16, 16)], 15.0),
    ("lstm-keywords", KeywordLSTM(), [(1, 12, 10)], 15.0),
    ("cpu-island-between-delegates", CpuIsland(), [(1, 3, 8, 8)], 20.0),
]


def lower(model, shapes):
    converted, program = quantize(model, shapes)
    edge, report = to_espnn_edge(program)
    return converted, edge, report


@pytest.mark.parametrize(
    "name, model, shapes, min_sqnr", CASES, ids=[c[0] for c in CASES]
)
def test_partial_delegation_tracks_pytorch(name, model, shapes, min_sqnr):
    converted, edge, report = lower(model, shapes)
    program = edge.exported_program()
    delegates = [
        m
        for _, m in program.graph_module.named_children()
        if isinstance(m, LoweredBackendModule)
    ]
    assert delegates, f"{name}: nothing was delegated"
    assert report["cpu_ops"], f"{name}: expected some operators to fall back to the CPU"
    emulator = ProgramEmulator(program)
    for trial in range(3):
        torch.manual_seed(500 + trial)
        inputs = [torch.randn(shape) for shape in shapes]
        expected = converted(*inputs)
        actual = emulator(*inputs)
        expected = expected if isinstance(expected, (tuple, list)) else (expected,)
        actual = actual if isinstance(actual, (tuple, list)) else (actual,)
        for want, got in zip(expected, actual):
            assert got.shape == want.shape
            assert (
                sqnr_db(want, got) >= min_sqnr
            ), f"{name}: SQNR {sqnr_db(want, got):.1f} dB"
    assert len(edge.to_executorch().buffer) > 0


def test_fully_supported_model_has_no_cpu_ops():
    from executorch.backends.espressif.test.test_int8_lowering import SEBlockModel

    _, edge, report = lower(SEBlockModel(), [(1, 3, 16, 16)])
    assert report["partitions"] == 1
    assert report["cpu_ops"] == []


def test_strict_mode_still_rejects_unsupported_operators():
    _, program = quantize(CumsumHead(), [(1, 3, 8, 8)])
    with pytest.raises(Exception, match="cumsum"):
        to_espnn_edge(program, allow_cpu_fallback=False)
