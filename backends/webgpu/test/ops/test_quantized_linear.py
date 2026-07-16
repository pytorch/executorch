# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""4-bit weight-only quantized linear (`et_vk.linear_q4gsw`) export + fp64 golden.

Mirrors test_sdpa.py: a named CONFIGS sweep over real Llama-3.2-1B linear shapes
(q/o/k/v/gate/up/down proj + lm_head) plus large-M (4k/8k) prefill stress, each
exported through VulkanPartitioner (which fuses dq+linear into
`et_vk.linear_q4gsw.default`). The golden is the fp64 dequant-matmul truth
(x @ dequant(W).T), so the GPU's fp32 error is measured against truth, not another
fp32 approximation. The native test (test_webgpu_native.cpp) mirrors the same
CONFIGS table and reconstructs the identical deterministic ramp input bit-for-bit.
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
class Q4gswConfig:
    name: str
    m: int  # rows (tokens)
    k: int  # in_features (reduction dim)
    n: int  # out_features
    group_size: int = 32  # K % group_size == 0, K % 8 == 0, N % 8 == 0
    # heavy = huge fixture / slow on a CPU rasterizer; export_all skips unless asked.
    heavy: bool = False


# Single source of truth, mirrored by the C++ kQ4gswConfigs table. Llama-3.2-1B:
# hidden=2048, n_heads=32 head_dim=64 (q/o=2048->2048), n_kv=8 (k/v=2048->512),
# FFN=8192 (gate/up=2048->8192), down=8192->2048, vocab=128256 (lm_head).
CONFIGS = [
    # name              M     K       N
    Q4gswConfig("q_proj", 1, 2048, 2048),  # also covers o_proj (same shape)
    Q4gswConfig("kv_proj", 1, 2048, 512),  # k_proj / v_proj
    Q4gswConfig("gate_proj", 1, 2048, 8192),  # gate_proj / up_proj
    Q4gswConfig("down_proj", 1, 8192, 2048),  # big reduction K
    Q4gswConfig("lm_head", 1, 2048, 128256, heavy=True),  # 131MB packed .pte
    Q4gswConfig("q_proj_4k", 4096, 2048, 2048),  # 4k-token prefill
    Q4gswConfig("kv_proj_4k", 4096, 2048, 512),
    Q4gswConfig("q_proj_8k", 8192, 2048, 2048, heavy=True),  # 67MB golden
    Q4gswConfig("kv_proj_8k", 8192, 2048, 512, heavy=True),
    # The M==1 decode configs above (q/kv/gate/down_proj) exercise the bicol 2-col
    # decode GEMV: the handler routes M==1 -> bicol, so each reads its own per-
    # column scale (col0/col1) across many K-groups (down_proj: 256 groups). q4gsw
    # requires N % 8 == 0 (torchao pads N for the scale layout), so odd-N / N=1 are
    # not exportable -- bicol's has1 odd-N guard is defensive (mirrors coop4's
    # general-N robustness) and unreachable through this op.
    # M>1 prefill: prefer the steel GEMM (K%16==0) on a >=256-invocation device
    # (e.g. lvp); else shmem (K>=4096 or N>=2048) or register-tiled (SwiftShader
    # caps at 128). Same fp64 golden regardless of which kernel runs.
    Q4gswConfig("steel", 96, 2048, 256),  # steel-isolating (K<4096, N<2048)
    # Same shape as "steel"; the .pte is dtype-independent, so this fixture feeds
    # the f16-multiply steel kernel (selected at runtime when the device reports
    # shader-f16; goldened at a looser f16 tol in the native test).
    Q4gswConfig("steel_f16", 96, 2048, 256),  # f16-multiply steel (shader-f16)
    # Partial M and N steel tiles under the f16 kernel; exercises f16 boundary
    # masking (the exact-N "steel_f16" shape does not). N%8==0, steel-isolating.
    Q4gswConfig("steel_f16_edge", 70, 1024, 136),  # f16 partial-tile
    # pwdq (packed-word dequant) backs the f16 steel path at group_size % BK(16)
    # == 0 (bit-exact to steel_half; steel_f16 above runs it at gs=32). These lock
    # the gs gate at group sizes those omit: gs=64 stays on pwdq; gs=8 (< BK) falls
    # back to the per-nibble steel_half kernel (its hoisted-per-BK scale is invalid
    # there). Same fp64 golden regardless of which kernel runs.
    Q4gswConfig("pwdq_gs64", 96, 2048, 256, group_size=64),  # pwdq, non-32 group
    Q4gswConfig("pwdq_gs8", 96, 2048, 256, group_size=8),  # steel_half fallback
    # pwdqf16acc (f16-accumulate) runs when the enable_f16_accumulate_gemm runtime
    # spec is set and gs % BK == 0 (perplexity-gated; see the kernel diff). Same
    # .pte as the f32 configs -- only the accumulator dtype differs -- goldened at a
    # looser f16-accumulate tol in the native test; deep-K stresses the worst case.
    Q4gswConfig("pwdqf16acc", 96, 2048, 256),  # f16-accumulate steel (runtime)
    Q4gswConfig("pwdqf16acc_down", 128, 8192, 2048),  # deep-K f16-accum worst case
    Q4gswConfig("gate_proj_pf", 128, 2048, 8192),  # gate/up prefill (shmem via N)
    Q4gswConfig("down_proj_pf", 128, 8192, 2048),  # down prefill (shmem via K)
    Q4gswConfig("shmem_edge", 130, 4096, 2056),  # partial 32-tile bounds
]


def _make_quantized_model(k: int, n: int, group_size: int) -> torch.nn.Module:
    torch.manual_seed(0)  # load-bearing: fixes the weights the golden derives from
    m = torch.nn.Linear(k, n, bias=False).eval()
    quantize_(
        m,
        IntxWeightOnlyConfig(weight_dtype=torch.int4, granularity=PerGroup(group_size)),
    )
    return m


def _ramp_input(m_rows: int, k: int) -> torch.Tensor:
    """Deterministic fp32 input [M,K]; C++ q4gsw_ramp reconstructs it bit-for-bit.

    x[flat] = ((flat % 17) - 8) / 16 over the flat row-major index -- exact in fp32
    (small modulus, power-of-two denominator).
    """
    flat = np.arange(m_rows * k, dtype=np.int64)
    x = ((flat % 17) - 8).astype(np.float32) / np.float32(16.0)
    return torch.from_numpy(x).reshape(m_rows, k)


def _fp64_golden(m: torch.nn.Module, x: torch.Tensor) -> np.ndarray:
    """fp64 truth: x @ dequant(W).T. The kernel computes the same dequant-matmul, so
    fp64 makes this the true answer -- GPU fp32 error is measured vs truth, not vs a
    second fp32 approximation. torchao handles the signed-nibble recovery in dequantize().
    """
    wq = m.weight.dequantize()  # AffineQuantizedTensor -> dequantized weight [N,K]
    golden = x.double() @ wq.double().t()  # [M,N] in fp64
    return golden.to(torch.float32).numpy().astype("<f4")


def _export(m: torch.nn.Module, x: torch.Tensor):
    ep = torch.export.export(m, (x,))
    return to_edge_transform_and_lower(
        ep, partitioner=[VulkanPartitioner()]
    ).to_executorch()


class TestQuantizedLinear(unittest.TestCase):
    def test_export_delegates(self) -> None:
        # Each (non-heavy) config must fuse to a VulkanBackend delegate (q4gsw);
        # fusion is shape-independent, so skipping the heavy 131MB+ fixtures is free.
        for cfg in CONFIGS:
            if cfg.heavy:
                continue
            with self.subTest(config=cfg.name):
                m = _make_quantized_model(cfg.k, cfg.n, cfg.group_size)
                et = _export(m, _ramp_input(1, cfg.k))
                found = any(
                    d.id == "VulkanBackend"
                    for plan in et.executorch_program.execution_plan
                    for d in plan.delegates
                )
                self.assertTrue(found, f"no VulkanBackend delegate in {cfg.name}")

    def test_golden_matches_eager(self) -> None:
        # Dual oracle (mirrors SDPA test_golden_matches_eager_op): the fp64 dequant-
        # matmul truth and torchao's own fp32 quantized forward are independent refs
        # that must agree -- guards a bug in the fp64 oracle / dequantize() accessor.
        # M=1 non-heavy shapes (cheap; the math is shape-independent).
        for cfg in CONFIGS:
            if cfg.m != 1 or cfg.heavy:
                continue
            with self.subTest(config=cfg.name):
                m = _make_quantized_model(cfg.k, cfg.n, cfg.group_size)
                x = _ramp_input(1, cfg.k)
                golden = torch.from_numpy(_fp64_golden(m, x))
                torch.testing.assert_close(m(x), golden, atol=5e-4, rtol=1e-3)


def export_quantized_linear_model(
    cfg: Q4gswConfig, pte_path: str, golden_path: str
) -> None:
    """Export one config's q4gsw .pte + its fp64 golden (raw LE fp32)."""
    m = _make_quantized_model(cfg.k, cfg.n, cfg.group_size)
    x = _ramp_input(cfg.m, cfg.k)
    et = _export(m, x)
    with open(pte_path, "wb") as f:
        f.write(et.buffer)
    _fp64_golden(m, x).tofile(golden_path)
    print(f"Exported {pte_path}; golden {golden_path} ({cfg.m * cfg.n} floats)")


def export_all_quantized_linear_models(
    out_dir: str, include_heavy: bool = False
) -> None:
    """Write q4gsw_<name>.pte + q4gsw_<name>.golden.bin for each config.

    Heavy configs (lm_head 131MB .pte; M=8k 67MB goldens) are skipped unless
    include_heavy -- plain CI never writes them; a real-GPU run opts in.
    """
    for cfg in CONFIGS:
        if cfg.heavy and not include_heavy:
            print(f"(skipping heavy config {cfg.name}; set include_heavy=True)")
            continue
        pte = os.path.join(out_dir, f"q4gsw_{cfg.name}.pte")
        golden = os.path.join(out_dir, f"q4gsw_{cfg.name}.golden.bin")
        export_quantized_linear_model(cfg, pte, golden)


if __name__ == "__main__":
    unittest.main()
