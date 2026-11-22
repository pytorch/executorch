# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""NVFP4 KV cache kernels (fused write, exact FP16 attention) vs PyTorch references."""

import itertools
import math
import tempfile
import unittest
from pathlib import Path

import executorch.backends.cuda.triton.kernels.nvfp4_kv as K
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


def row_err(a, b):
    """Worst query row's max error relative to that row's magnitude, so a few
    wrong rows (tail tiles, mask edges, split-K merge) cannot hide."""
    err = (a.float() - b.float()).abs().amax(-1)
    return (err / b.float().abs().amax(-1).clamp_min(1e-6)).max().item()


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

    def test_write_kernel_never_stores_out_of_range(self):
        # The op raises for a flat cache past its end; the kernel still masks.
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
            K._nvfp4_kv_write_impl(k, v, pos_t, *cache, None, ring, None)
            expected = new_cache(L) + new_cache(L)
            torch.ops.triton.nvfp4_kv_write(
                k[:, 1:2], v[:, 1:2], pos_t[1:2], *expected, None, ring, None
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
        torch.ops.triton.nvfp4_kv_write(x, x, pos, *kc, *vc, None, False, None)
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
                k[:, s:e], v[:, s:e], pos[s:e], *kc, *vc, None, False, None
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
        self.assertLess(row_err(out, ref), 2e-2, f"global kv={n_total} Lq={lq}")
        return kc, vc, q, ref

    def test_global_decode(self):
        for n in (1, 777, 70001):
            self._global(131072, n, 1)

    def test_global_small_query(self):
        # L_q 2 and 16 bound the split-K dispatch; 17 takes the prefill kernel.
        self._global(131072, 5000, 2)
        self._global(131072, 5000, 4)
        self._global(131072, 33333, 16)
        self._global(131072, 33333, 17)

    def test_prefill_kernel_small_query(self):
        """The prefill kernel serves every chunk of a dynamic-length export,
        including 1..16-token chunks and lengths that are not tile multiples."""
        for n_total, lq in ((5000, 1), (5000, 5), (777, 16), (4100, 100)):
            kc, vc, q, ref = self._global(8192, n_total, lq)
            out = torch.empty_like(q)
            kv_len = torch.tensor([n_total], device="cuda", dtype=torch.int32)
            K._launch_nvfp4_prefill(q, *kc, *vc, out, None, kv_len, SCALE, True)
            tag = f"prefill kernel kv={n_total} Lq={lq}"
            self.assertLess(rel(out, ref), 1e-2, tag)
            self.assertLess(row_err(out, ref), 2e-2, tag)

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
        self.assertLess(row_err(out, ref), 2e-2, f"{tag} dense mask")
        kv_len = torch.tensor([n_total], device="cuda")
        if keep == 1.0:
            out = torch.ops.triton.nvfp4_sdpa(
                q, *kc, *vc, None, False, SCALE, kv_len, window
            )
            self.assertLess(rel(out, ref), 1e-2, f"{tag} in-kernel window")
            self.assertLess(row_err(out, ref), 2e-2, f"{tag} in-kernel window")
        out = torch.ops.triton.nvfp4_sdpa(
            q, *kc, *vc, mask, False, SCALE, kv_len, window
        )
        self.assertLess(rel(out, ref), 1e-2, f"{tag} window + slot-indexed mask")
        self.assertLess(row_err(out, ref), 2e-2, f"{tag} window + slot-indexed mask")
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
    """One attention layer over an NVFP4 cache, called as production does: a
    ring (sliding window) or flat (causal) cache, RoPE or NoPE, and a 0-d
    ``kv_len`` of tokens written so far."""

    def __init__(self, length, ring, rope=True, window=16):
        super().__init__()
        self.ring = ring
        self.window = window if ring else 0
        k_q, k_s = new_cache(length)
        v_q, v_s = new_cache(length)
        self.register_buffer("k_q", k_q, persistent=False)
        self.register_buffer("k_s", k_s, persistent=False)
        self.register_buffer("v_q", v_q, persistent=False)
        self.register_buffer("v_s", v_s, persistent=False)
        inv_freq = 1.0 / (10000.0 ** (torch.arange(0, D, 2, device="cuda") / D))
        self.register_buffer("inv_freq", inv_freq if rope else None, persistent=False)

    def forward(self, q, k, v, input_pos):
        cache = (self.k_q, self.k_s, self.v_q, self.v_s)
        torch.ops.triton.nvfp4_kv_write(
            k, v, input_pos, *cache, self.inv_freq, self.ring, 1e-5
        )
        kv_len = input_pos[0] + input_pos.shape[0]
        return torch.ops.triton.nvfp4_sdpa(
            q, *cache, None, not self.ring, SCALE, kv_len, self.window
        )


def step_inputs(start, T):
    return (
        torch.randn(1, HQ, T, D, device="cuda", dtype=torch.bfloat16),
        torch.randn(1, T, HKV, D, device="cuda", dtype=torch.bfloat16),
        torch.randn(1, T, HKV, D, device="cuda", dtype=torch.bfloat16),
        torch.arange(start, start + T, device="cuda"),
    )


# Prefill, decode, then chunks that wrap a 48-slot ring (window 16 + the
# longest chunk - 1 <= 48) and stay inside a 64-slot flat cache.
STEPS = ((0, 24), (24, 1), (25, 8), (33, 20))
SEQ_DIM = torch.export.Dim("T", min=1, max=64)
SEQ_SHAPES = ({2: SEQ_DIM}, {1: SEQ_DIM}, {1: SEQ_DIM}, {0: SEQ_DIM})


def layer(ring, rope=True):
    return _WriteAndAttend(48 if ring else 64, ring, rope)


class TestNVFP4KVValidation(unittest.TestCase):
    """The ops reject malformed operands before launching (no GPU needed)."""

    def test_rejects_malformed_operands(self):
        def cache(batch=1, heads=HKV, length=16, dtype=torch.uint8):
            return [
                torch.zeros(batch, heads, length, D // 2, dtype=dtype),
                torch.zeros(batch, heads, length, D // 16, dtype=dtype),
            ]

        k = torch.zeros(1, 2, HKV, D, dtype=torch.bfloat16)
        pos = torch.arange(2)
        q = torch.zeros(1, HQ, 2, D, dtype=torch.bfloat16)
        write = torch.ops.triton.nvfp4_kv_write
        sdpa = torch.ops.triton.nvfp4_sdpa
        bad = [
            lambda: write(k, k, pos, *cache(dtype=torch.int8), *cache()),
            lambda: write(k, k, pos, *cache(batch=2), *cache(batch=2)),
            lambda: write(k, k[..., :64], pos, *cache(), *cache()),
            lambda: write(k, k, pos[:1], *cache(), *cache()),
            lambda: write(k, k, pos, *cache(), *cache(length=8)),
            lambda: sdpa(q.half(), *cache(), *cache()),
            lambda: sdpa(q[:, :3], *cache(), *cache()),
            lambda: sdpa(q[..., :64], *cache(), *cache()),
            # A negative eps is not "no K norm"; that is None.
            lambda: write(k, k, pos, *cache(), *cache(), None, False, -1.0),
            # A flat cache does not drop keys past its end silently.
            lambda: write(k, k, torch.arange(15, 17), *cache(), *cache()),
            # The kernel reads positions and frequencies with unit stride.
            lambda: write(k, k, torch.arange(4)[::2], *cache(), *cache()),
            lambda: write(k, k, pos, *cache(), *cache(), torch.ones(D)[::2]),
            # SDPA-style additive float masks would invert visibility.
            lambda: sdpa(q, *cache(), *cache(), torch.zeros(1, 1, 2, 16)),
            lambda: sdpa(q, *cache(), *cache(), torch.ones(1, HQ, 2, 16).bool()),
        ]
        for i, call in enumerate(bad):
            with self.assertRaises(ValueError, msg=str(i)):
                call()
        with self.assertRaises(RuntimeError):
            sdpa(q, *cache(), *cache(), None, True)


class TestLowerNVFP4KVWrite(unittest.TestCase):
    def setUp(self):
        if not torch.cuda.is_available() or torch.version.hip is not None:
            self.skipTest("CUDA required")
        torch.manual_seed(0)

    def test_pass_with_dynamic_seq_len_matches_eager(self):
        """Prefill, decode and ring-wrapping chunks through one dynamic-T
        export: the outputs and the cache buffers match eager after each step."""
        from executorch.backends.cuda.passes.lower_nvfp4_kv import LowerNVFP4KVWritePass

        steps = [step_inputs(*s) for s in STEPS]
        for ring, rope in ((False, False), (True, True)):
            eager = layer(ring, rope)
            ep = torch.export.export(
                layer(ring, rope), steps[0], dynamic_shapes=SEQ_SHAPES, strict=True
            )
            ep = LowerNVFP4KVWritePass()(ep.run_decompositions({}))
            self.assertNotIn("auto_functionalized", str(ep.graph))
            module = ep.module()
            for args in steps:
                want = eager(*args)
                # A dynamic T takes the prefill kernel; eager may take split-K.
                out = module(*args)
                tag = f"ring={ring} rope={rope} pos={int(args[3][0])}"
                self.assertLess(rel(out, want), 1e-2, tag)
                self.assertLess(row_err(out, want), 2e-2, tag)
                for got, ref in (
                    (module.k_q, eager.k_q),
                    (module.k_s, eager.k_s),
                    (module.v_q, eager.v_q),
                    (module.v_s, eager.v_s),
                ):
                    self.assertTrue(torch.equal(got, ref), tag)

    def _lower_and_run(self, ring, rope, steps, shapes):
        """Lowers the layer with the CUDA backend (and the pass) and runs every
        step through the program against eager, carrying the cache."""
        from executorch.backends.cuda.cuda_backend import CudaBackend
        from executorch.backends.cuda.cuda_partitioner import CudaPartitioner
        from executorch.exir import EdgeCompileConfig, to_edge_transform_and_lower
        from executorch.export.utils import is_fbcode

        eager = layer(ring, rope)
        expected = [eager(*args) for args in steps]
        ep = torch.export.export(
            layer(ring, rope), steps[0], dynamic_shapes=shapes, strict=True
        )
        lowered = to_edge_transform_and_lower(
            ep,
            partitioner=[
                CudaPartitioner(
                    [CudaBackend.generate_method_name_compile_spec("forward")]
                )
            ],
            compile_config=EdgeCompileConfig(_check_ir_validity=False),
        ).to_executorch()
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "kv.pte"
            with path.open("wb") as f:
                lowered.write_to_file(f)
            lowered.write_tensor_data_to_file(folder)
            if is_fbcode():
                # The in-process runtime in fbcode has no CUDA backend.
                return
            from executorch.runtime import Runtime

            data = next(Path(folder).glob("*.ptd"), None)
            program = Runtime.get().load_program(
                str(path), data_path=None if data is None else str(data)
            )
            method = program.load_method("forward")
            for args, want in zip(steps, expected):
                out = method.execute([a.cpu() for a in args])[0].to(want.device)
                tag = f"ring={ring} rope={rope} pos={int(args[3][0])}"
                self.assertLess(rel(out, want), 1e-2, tag)
                self.assertLess(row_err(out, want), 2e-2, tag)

    def test_lowered_program_matches_eager(self):
        """Dynamic T, as production prefill: the prefill attention kernel for
        every chunk, flat NoPE and ring RoPE layers."""
        steps = [step_inputs(*s) for s in STEPS]
        for ring, rope in ((False, False), (True, True)):
            self._lower_and_run(ring, rope, steps, SEQ_SHAPES)

    def test_lowered_static_decode_matches_eager(self):
        """Static T=1, as production decode: the split-K attention, stepped
        from position 0 past the 48-slot ring."""
        steps = [step_inputs(pos, 1) for pos in range(56)]
        for ring, rope in ((False, False), (True, True)):
            self._lower_and_run(ring, rope, steps, None)

    def test_pass_inlines_write_kernel(self):
        from executorch.backends.cuda.passes.lower_nvfp4_kv import LowerNVFP4KVWritePass

        args = step_inputs(40, 8)
        for ring, rope in itertools.product((False, True), repeat=2):
            expected = layer(ring, rope)(*args)
            # Older exporters emit auto_functionalized, newer ones _v2.
            for v2 in (False, True):
                # run_decompositions creates the auto_functionalized nodes.
                with torch._export.config.patch(
                    enable_auto_functionalized_v2_for_export=v2
                ):
                    ep = torch.export.export(layer(ring, rope), args, strict=True)
                    ep = ep.run_decompositions({})
                hop = (
                    torch.ops.higher_order.auto_functionalized_v2
                    if v2
                    else torch.ops.higher_order.auto_functionalized
                )
                self.assertIn(hop, {node.target for node in ep.graph.nodes})
                ep = LowerNVFP4KVWritePass()(ep)
                graph = str(ep.graph)
                self.assertNotIn("auto_functionalized", graph)
                self.assertIn("triton_kernel_wrapper_functional", graph)
                out = ep.module()(*args)
                tag = f"ring={ring} rope={rope} v2={v2}"
                self.assertTrue(torch.equal(out, expected), tag)
                mutated = [
                    spec
                    for spec in ep.graph_signature.output_specs
                    if spec.kind.name == "BUFFER_MUTATION"
                ]
                self.assertEqual(len(mutated), 4)


if __name__ == "__main__":
    unittest.main()
