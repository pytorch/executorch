# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Backward of 4-bit quantized linear (`et_vk.linear_q4gsw_backward`) export + fp64 golden.

Mirrors test_quantized_linear.py. The backward computes `d_x = d_out @ dequant(W)` so
gradients flow through a frozen 4-bit base into a LoRA/DiReFT adapter. CONFIGS reuse the
real Llama-3.2-1B linear shapes (the backward's d_out is [M, N] and its output d_x is
[M, K]). The golden is the fp64 dequant-matmul truth; the native test
(test_webgpu_native.cpp) reconstructs the identical deterministic ramp bit-for-bit.
"""

import os
import unittest
from dataclasses import dataclass

import numpy as np
import torch

from executorch.backends.vulkan import VulkanPartitioner
from executorch.exir import to_edge_transform_and_lower
from torchao.quantization.granularity import PerGroup
from torchao.quantization.quant_api import IntxWeightOnlyConfig, quantize_


@dataclass(frozen=True)
class BwdConfig:
    name: str
    m: int  # rows (tokens)
    k: int  # in_features (== d_x cols)
    n: int  # out_features (== d_out cols)
    group_size: int = 32  # K % group_size == 0, K % 8 == 0, N % 8 == 0
    heavy: bool = False


# Mirrored by the C++ kQ4gswBackwardConfigs table (Llama-3.2-1B shapes).
CONFIGS = [
    BwdConfig("q_proj", 1, 2048, 2048),  # also o_proj
    BwdConfig("kv_proj", 1, 2048, 512),
    BwdConfig("gate_proj", 1, 2048, 8192),
    BwdConfig("down_proj", 1, 8192, 2048),
    BwdConfig("q_proj_112", 112, 2048, 2048),  # S=112 multi-row training window
]


def _make_quantized_model(k: int, n: int, group_size: int) -> torch.nn.Module:
    torch.manual_seed(0)  # load-bearing: fixes the weights the golden uses
    m = torch.nn.Linear(k, n, bias=False).eval()
    quantize_(
        m,
        IntxWeightOnlyConfig(weight_dtype=torch.int4, granularity=PerGroup(group_size)),
    )
    return m


def _ramp(m_rows: int, cols: int) -> torch.Tensor:
    """Deterministic fp32 [rows, cols]; the C++ side reconstructs it bit-for-bit.

    v[flat] = ((flat % 17) - 8) / 16 -- exact in fp32 (small modulus, po2 denominator).
    """
    flat = np.arange(m_rows * cols, dtype=np.int64)
    v = ((flat % 17) - 8).astype(np.float32) / np.float32(16.0)
    return torch.from_numpy(v).reshape(m_rows, cols)


def _packed_qweights(m: torch.nn.Module):
    """The int4 packed weights + per-group scales `et_vk.linear_q4gsw` consumes.

    Recover them the same way the forward op does: `dequant(W)` is [N, K]; the backward op
    takes the same (weights, weight_scales, group_size) triple the partitioner extracts.
    """
    aqt = m.weight  # AffineQuantizedTensor
    return aqt


def _fp64_golden(m: torch.nn.Module, d_out: torch.Tensor) -> np.ndarray:
    """fp64 truth: d_x = d_out @ dequant(W), dequant(W) is [N, K] -> [M, N]@[N, K] = [M, K]."""
    wq = m.weight.dequantize()  # [N, K]
    d_x = d_out.double() @ wq.double()  # [M, K] in fp64
    return d_x.to(torch.float32).numpy().astype("<f4")


class _BackwardModule(torch.nn.Module):
    """Wraps the linear so autograd through it exercises the registered backward op."""

    def __init__(self, lin: torch.nn.Module) -> None:
        super().__init__()
        self.lin = lin

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.lin(x)


def _export_backward(m: torch.nn.Module, x: torch.Tensor):
    # Training-style export: forward + backward makes the backward reachable.
    mod = _BackwardModule(m)
    ep = torch.export.export(mod, (x,))
    return to_edge_transform_and_lower(
        ep, partitioner=[VulkanPartitioner()]
    ).to_executorch()


class TestQuantizedLinearBackward(unittest.TestCase):
    def test_op_matches_fp64_golden(self) -> None:
        # Op impl (d_out @ dequant(W)) vs fp64 truth: guards backward formula.
        for cfg in CONFIGS:
            if cfg.heavy:
                continue
            with self.subTest(config=cfg.name):
                m = _make_quantized_model(cfg.k, cfg.n, cfg.group_size)
                d_out = _ramp(cfg.m, cfg.n)
                got = torch.ops.et_vk.linear_q4gsw_backward(
                    d_out, _packed_qweights(m), None, cfg.group_size
                )
                golden = torch.from_numpy(_fp64_golden(m, d_out))
                torch.testing.assert_close(got, golden, atol=5e-4, rtol=1e-3)

    def test_autograd_backward_matches_golden(self) -> None:
        # autograd through linear_q4gsw uses the registered backward op.
        for cfg in CONFIGS:
            if cfg.heavy:
                continue
            with self.subTest(config=cfg.name):
                m = _make_quantized_model(cfg.k, cfg.n, cfg.group_size)
                x = _ramp(cfg.m, cfg.k).requires_grad_(True)
                d_out = _ramp(cfg.m, cfg.n)
                y = m(x)
                y.backward(d_out)
                golden = torch.from_numpy(_fp64_golden(m, d_out))
                torch.testing.assert_close(x.grad, golden, atol=5e-4, rtol=1e-3)


def export_backward_model(cfg: BwdConfig, pte_path: str, golden_path: str) -> None:
    """Export one config's backward .pte + its fp64 golden (raw LE fp32)."""
    m = _make_quantized_model(cfg.k, cfg.n, cfg.group_size)
    x = _ramp(cfg.m, cfg.k)
    et = _export_backward(m, x)
    with open(pte_path, "wb") as f:
        f.write(et.buffer)
    _fp64_golden(m, _ramp(cfg.m, cfg.n)).tofile(golden_path)
    print(f"Exported {pte_path}; golden {golden_path} ({cfg.m * cfg.k} floats)")


def export_all_backward_models(out_dir: str, include_heavy: bool = False) -> None:
    for cfg in CONFIGS:
        if cfg.heavy and not include_heavy:
            continue
        pte = os.path.join(out_dir, f"q4gsw_backward_{cfg.name}.pte")
        golden = os.path.join(out_dir, f"q4gsw_backward_{cfg.name}.golden.bin")
        export_backward_model(cfg, pte, golden)


if __name__ == "__main__":
    unittest.main()
