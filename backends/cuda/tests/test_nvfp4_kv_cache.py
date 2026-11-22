# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""NVFP4 KV cache kernels (fused write, exact FP16 attention) vs PyTorch references."""

import math
import unittest

import torch

from executorch.backends.cuda.triton.kernels.nvfp4_kv import (
    nvfp4_kv_dequantize_reference as deq,
)

D, HKV, HQ = 128, 2, 32
SCALE = (43.7840518911 / math.sqrt(D)) / math.sqrt(D)


def ref_quant(x):
    """Reference NVFP4 quantization of fp32 rows [..., D] (unit global scale)."""
    lut = torch.tensor([0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0], device=x.device)
    xb = x.reshape(*x.shape[:-1], D // 16, 16)
    sf = (xb.abs().amax(-1) / 6).clamp(max=448).to(torch.float8_e4m3fn).float()
    den = torch.where(sf > 0, sf, torch.ones_like(sf))
    v = xb / den[..., None]
    a = v.abs().clamp(max=6)
    edges = torch.tensor([0.25, 0.75, 1.25, 1.75, 2.5, 3.5, 5.0], device=x.device)
    idx = torch.bucketize(a, edges, right=False)
    # Ties go to the even code (0.75 -> 1.0, 1.75 -> 2.0, 3.5 -> 4.0).
    idx = idx + torch.isin(a, torch.tensor([0.75, 1.75, 3.5], device=x.device)).long()
    return (lut[idx] * torch.sign(v) * den[..., None]).reshape(x.shape)


def rope_ref(x, pos, theta=500000.0):
    inv = 1.0 / (
        theta ** (torch.arange(0, D, 2, device=x.device, dtype=torch.float32) / D)
    )
    f = torch.outer(pos.float(), inv)
    c, s = torch.cos(f)[None, :, None, :], torch.sin(f)[None, :, None, :]
    x1, x2 = x[..., 0::2], x[..., 1::2]
    return torch.stack([x1 * c - x2 * s, x1 * s + x2 * c], -1).flatten(-2), inv


def new_cache(length):
    return [
        torch.zeros(1, HKV, length, D // 2, dtype=torch.uint8, device="cuda"),
        torch.zeros(1, HKV, length, D // 16, dtype=torch.uint8, device="cuda"),
    ]


def rel(a, b):
    return ((a.float() - b.float()).norm() / b.float().norm()).item()


def ref_attn(q, kd, vd, mask):
    """FP32 GQA attention, one query head at a time to bound memory."""
    out = torch.empty(q.shape, device=q.device, dtype=torch.float32)
    for h in range(HQ):
        kh = kd[:, h // (HQ // HKV)].float()
        vh = vd[:, h // (HQ // HKV)].float()
        s = (q[:, h].float() @ kh.transpose(-1, -2)) * SCALE
        p = torch.softmax(s.masked_fill(~mask[:, 0], float("-inf")), -1)
        out[:, h] = p @ vh
    return out


def sliding_mask(input_pos, buf, window):
    """The model's dense ring mask (MuseGlimmerModel._build_masks)."""
    total = input_pos[0] + input_pos.shape[0]
    j = torch.arange(buf, device=input_pos.device)
    ring_pos = j + ((total - 1 - j) // buf) * buf
    delta = input_pos[:, None] - ring_pos[None, :]
    return ((ring_pos >= 0) & (delta >= 0) & (delta < window))[None, None]


def unit_rows(x):
    xf = x.float()
    return (xf * torch.rsqrt(xf.pow(2).mean(-1, keepdim=True) + 1e-5)).to(x.dtype)


class TestNVFP4KVCache(unittest.TestCase):
    def setUp(self):
        if not torch.cuda.is_available() or torch.version.hip is not None:
            self.skipTest("CUDA required")
        torch.manual_seed(0)

    def test_write(self):
        T, L = 300, 4096
        raw = torch.randn(1, T, 8448, device="cuda", dtype=torch.bfloat16) * 3
        k = raw[..., 4096:4352].view(1, T, HKV, D)  # strided, like the qko output
        v = (
            torch.randn(1, T, HKV, D, device="cuda")
            * torch.logspace(-2, 1, D, device="cuda")
        ).to(torch.bfloat16)
        pos = torch.arange(5000, 5000 + T, device="cuda")
        kc, vc = new_cache(L), new_cache(L)
        kf = k.float()
        kr, inv = rope_ref(
            kf * torch.rsqrt(kf.pow(2).mean(-1, keepdim=True) + 1e-5), pos
        )
        torch.ops.triton.nvfp4_kv_write(k, v, pos, *kc, *vc, inv, True, 1e-5)
        slots = pos % L
        kd = deq(*kc)[:, :, slots].transpose(1, 2)
        vd = deq(*vc)[:, :, slots].transpose(1, 2)
        self.assertLess(rel(kd, ref_quant(kr)), 2e-3)
        self.assertLess(rel(vd, ref_quant(v.float())), 1e-2)
        self.assertLess(rel(kd, kr), 0.12)
        untouched = torch.ones(L, dtype=torch.bool, device="cuda")
        untouched[slots] = False
        self.assertEqual(kc[0][:, :, untouched].count_nonzero().item(), 0)
        self.assertEqual(vc[1][:, :, untouched].count_nonzero().item(), 0)

    def test_write_out_of_range_dropped(self):
        L = 64
        k = torch.randn(1, 4, HKV, D, device="cuda", dtype=torch.bfloat16)
        v = torch.randn(1, 4, HKV, D, device="cuda", dtype=torch.bfloat16)
        for ring, pos in ((False, [-1, 3, L, 2 * L - 1]), (True, [-1, 3, -L, -5])):
            # Each cache tensor sits between two zero guards of its own size, so
            # a store outside the cache lands in a guard.
            flat = [
                torch.zeros(3 * n, dtype=torch.uint8, device="cuda")
                for n in (HKV * L * D // 2, HKV * L * D // 16) * 2
            ]
            cache = [f[f.numel() // 3 : 2 * f.numel() // 3] for f in flat]
            cache = [c.view(1, HKV, L, -1) for c in cache]
            pos_t = torch.tensor(pos, device="cuda")
            torch.ops.triton.nvfp4_kv_write(k, v, pos_t, *cache, None, ring, -1.0)
            expected = new_cache(L) + new_cache(L)
            torch.ops.triton.nvfp4_kv_write(
                k[:, 1:2], v[:, 1:2], pos_t[1:2], *expected, None, ring, -1.0
            )
            for c, e in zip(cache, expected):
                self.assertTrue(torch.equal(c, e), f"ring={ring}")
            for f in flat:
                n = f.numel() // 3
                self.assertEqual(f[:n].count_nonzero().item(), 0, f"ring={ring}")
                self.assertEqual(f[2 * n :].count_nonzero().item(), 0, f"ring={ring}")

    def test_write_saturates(self):
        sign = torch.where(torch.rand(1, 2, HKV, D, device="cuda") < 0.5, -1.0, 1.0)
        mag = 2700.0 * 10 ** (torch.rand(1, 2, HKV, D, device="cuda") * 26)
        mag[0, 1, :, : D // 2] = 2688.0
        x = (sign * mag).to(torch.bfloat16)
        kc, vc = new_cache(16), new_cache(16)
        pos = torch.arange(2, device="cuda")
        torch.ops.triton.nvfp4_kv_write(x, x, pos, *kc, *vc, None, False, -1.0)
        for c in (kc, vc):
            self.assertTrue(
                torch.equal(deq(*c)[:, :, :2].transpose(1, 2), 2688.0 * sign)
            )

    def _global(self, length, n_total, lq):
        kc, vc = new_cache(length), new_cache(length)
        k = unit_rows(
            torch.randn(1, n_total, HKV, D, device="cuda", dtype=torch.bfloat16)
        )
        v = (torch.randn(1, n_total, HKV, D, device="cuda") * 2).to(torch.bfloat16)
        pos = torch.arange(n_total, device="cuda")
        for s in range(0, n_total, 4096):
            e = min(n_total, s + 4096)
            torch.ops.triton.nvfp4_kv_write(
                k[:, s:e], v[:, s:e], pos[s:e], *kc, *vc, None, False, -1.0
            )
        q = unit_rows(torch.randn(1, HQ, lq, D, device="cuda", dtype=torch.bfloat16))
        kv_len = torch.tensor([n_total], device="cuda")
        out = torch.ops.triton.nvfp4_sdpa(q, *kc, *vc, None, True, SCALE, kv_len)
        qpos = torch.arange(n_total - lq, n_total, device="cuda")
        mask = (torch.arange(n_total, device="cuda")[None, :] <= qpos[:, None])[
            None, None
        ]
        ref = ref_attn(q, deq(*kc)[:, :, :n_total], deq(*vc)[:, :, :n_total], mask)
        self.assertLess(rel(out, ref), 1e-2, f"global kv={n_total} Lq={lq}")

    def test_global_decode(self):
        for n in (1, 777, 70001):
            self._global(131072, n, 1)

    def test_global_small_query(self):
        self._global(131072, 5000, 4)
        self._global(131072, 33333, 16)

    def test_global_prefill(self):
        self._global(131072, 512, 512)
        self._global(131072, 4096 * 3 + 100, 4096)
        self._global(131072, 3072 + 4096 * 2, 3072)

    def _sliding(self, n_before, lq, buf=4096, window=2048, chunk=4096, keep=1.0):
        n_total = n_before + lq
        kc, vc = new_cache(buf), new_cache(buf)
        k = torch.randn(1, n_total, HKV, D, device="cuda", dtype=torch.bfloat16)
        v = (torch.randn(1, n_total, HKV, D, device="cuda") * 2).to(torch.bfloat16)
        pos = torch.arange(n_total, device="cuda")
        inv = 1.0 / (
            500000.0 ** (torch.arange(0, D, 2, device="cuda", dtype=torch.float32) / D)
        )
        bounds = list(range(0, n_before, chunk)) + [n_before, n_total]
        for s, e in zip(bounds[:-1], bounds[1:]):
            if e > s:
                torch.ops.triton.nvfp4_kv_write(
                    k[:, s:e], v[:, s:e], pos[s:e], *kc, *vc, inv, True, 1e-5
                )
        q = torch.randn(1, HQ, lq, D, device="cuda", dtype=torch.bfloat16)
        mask = sliding_mask(pos[n_before:], buf, window)
        if keep < 1.0:
            mask = mask & (torch.rand(mask.shape, device="cuda") < keep)
        ref = ref_attn(q, deq(*kc), deq(*vc), mask)
        tag = f"sliding ring={buf} pos={n_before} Lq={lq} keep={keep}"
        out = torch.ops.triton.nvfp4_sdpa(q, *kc, *vc, mask, False, SCALE, None)
        self.assertLess(rel(out, ref), 1e-2, f"{tag} dense mask")
        kv_len = torch.tensor([n_total], device="cuda")
        if keep == 1.0:
            out = torch.ops.triton.nvfp4_sdpa(
                q, *kc, *vc, None, False, SCALE, kv_len, window
            )
            self.assertLess(rel(out, ref), 1e-2, f"{tag} in-kernel window")
        out = torch.ops.triton.nvfp4_sdpa(
            q, *kc, *vc, mask, False, SCALE, kv_len, window
        )
        self.assertLess(rel(out, ref), 1e-2, f"{tag} window + slot-indexed mask")
        return mask

    def test_sliding_decode(self):
        self._sliding(10000, 1)
        self._sliding(100, 1)

    def test_sliding_small_query(self):
        self._sliding(9000, 8)
        self._sliding(3000, 4)

    def test_sliding_prefill(self):
        for n_before, lq in (
            (0, 4096),
            (4096 * 2 + 17, 4096),
            (2048, 512),
            (4096 * 7 + 5, 4096),
            (1500, 600),
        ):
            self._sliding(n_before, lq)

    def test_sliding_window_with_mask(self):
        for n_before, lq in ((10000, 1), (9000, 8), (4096 * 2 + 17, 512)):
            self._sliding(n_before, lq, keep=0.5)

    def test_sliding_ring_capacity(self):
        # A ring of window + chunk - 1 slots keeps every row's whole window.
        mask = self._sliding(4096 * 2, 4096, buf=2048 + 4096 - 1)
        self.assertTrue(bool((mask.sum(-1) == 2048).all()))


class _WriteAndAttend(torch.nn.Module):
    def __init__(self, length, ring):
        super().__init__()
        self.ring = ring
        for kind in ("k", "v"):
            for name, t in zip(("q", "s"), new_cache(length)):
                self.register_buffer(f"{kind}_{name}", t, persistent=False)
        self.register_buffer(
            "inv_freq",
            1.0 / (10000.0 ** (torch.arange(0, D, 2, device="cuda") / D)),
            persistent=False,
        )

    def forward(self, q, k, v, input_pos):
        cache = (self.k_q, self.k_s, self.v_q, self.v_s)
        torch.ops.triton.nvfp4_kv_write(
            k, v, input_pos, *cache, self.inv_freq, self.ring, 1e-5
        )
        kv_len = input_pos[-1:] + 1
        return torch.ops.triton.nvfp4_sdpa(
            q, *cache, None, not self.ring, SCALE, kv_len, 64 if self.ring else 0
        )


class TestLowerNVFP4KVWrite(unittest.TestCase):
    def setUp(self):
        if not torch.cuda.is_available() or torch.version.hip is not None:
            self.skipTest("CUDA required")
        torch.manual_seed(0)

    def test_pass_inlines_write_kernel(self):
        from executorch.backends.cuda.passes.lower_nvfp4_kv import LowerNVFP4KVWritePass

        T = 8
        args = (
            torch.randn(1, HQ, T, D, device="cuda", dtype=torch.bfloat16),
            torch.randn(1, T, HKV, D, device="cuda", dtype=torch.bfloat16),
            torch.randn(1, T, HKV, D, device="cuda", dtype=torch.bfloat16),
            torch.arange(100, 100 + T, device="cuda"),
        )
        for ring in (False, True):
            expected = _WriteAndAttend(256, ring)(*args)
            # Older exporters emit auto_functionalized, newer ones _v2.
            for v2 in (False, True):
                with torch._export.config.patch(
                    enable_auto_functionalized_v2_for_export=v2
                ):
                    ep = torch.export.export(
                        _WriteAndAttend(256, ring), args, strict=True
                    )
                ep = ep.run_decompositions({})
                self.assertIn("auto_functionalized", str(ep.graph))
                ep = LowerNVFP4KVWritePass()(ep)
                graph = str(ep.graph)
                self.assertNotIn("auto_functionalized", graph)
                self.assertIn("triton_kernel_wrapper_functional", graph)
                out = ep.module()(*args)
                self.assertTrue(torch.equal(out, expected), f"ring={ring} v2={v2}")
                mutated = [
                    spec
                    for spec in ep.graph_signature.output_specs
                    if spec.kind.name == "BUFFER_MUTATION"
                ]
                self.assertEqual(len(mutated), 4)


if __name__ == "__main__":
    unittest.main()
