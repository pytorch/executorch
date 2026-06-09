# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""fp32 fused SDPA (`sdpa_with_kv_cache`) export + golden for the WebGPU backend.

Exports a small sweep of GQA/MHA attention configs (prefill + decode) through
VulkanPartitioner and writes a torch-computed golden (the native binary has no
ATen) that the native test compares against.

Each config is identified by a name; the native test (test_webgpu_native.cpp)
mirrors the same CONFIGS table and reconstructs the identical deterministic
inputs bit-for-bit (/16 multipliers are exact in fp32).
"""

import os
import unittest
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn.functional as F

from executorch.backends.vulkan import VulkanPartitioner
from executorch.exir import to_edge_transform_and_lower
from executorch.extension.llm.custom_ops import (  # noqa: F401  registers llama ops
    custom_ops,
)


@dataclass(frozen=True)
class SdpaConfig:
    name: str
    hq: int  # query heads
    hkv: int  # key/value heads (GQA groups when hq != hkv)
    d: int  # head dim
    s: int  # new tokens this step
    cmax: int  # kv-cache capacity
    input_pos: int  # number of prior tokens already in the cache (decode)
    denom: float = 16.0  # ramp divisor; small denom -> large logits (softmax stress)


# Single source of truth, mirrored by the C++ CONFIGS table in the native test.
CONFIGS = [
    # name             Hq Hkv  D  S Cmax pos
    SdpaConfig("gqa31_prefill", 6, 2, 8, 4, 16, 0),  # GQA 3:1 (original case)
    SdpaConfig("mha_ctxodd", 4, 4, 16, 3, 8, 0),  # MHA; context_len=3 (odd)
    SdpaConfig("gqa21_prefill", 8, 4, 4, 5, 16, 0),  # GQA 2:1; multi-token S=5
    SdpaConfig("gqa31_decode", 6, 2, 8, 2, 16, 2),  # decode: 2 prior tokens
    # llama3-ish GQA, D=128, S=128.
    SdpaConfig("llama3_prefill", 24, 8, 128, 128, 256, 0),
    # Adversarial: denom=0.5 -> peak scaled logit ~177 (>88) overflows naive fp32 exp.
    SdpaConfig("mha_biglogit", 4, 4, 32, 4, 16, 0, 0.5),
]


@dataclass(frozen=True)
class ReplaySeq:
    """A prefill+mt+decode sequence replayed on a threaded KV cache.

    Mirrors a Vulkan VulkanSDPATest param set: each seq_lens entry is one step
    (big=prefill, mid=multi-token, 1=decode); input_pos advances by the cumulative
    sum. Field order (hq, hkv, d) is a reordering of Vulkan's positional call
    (head_dim, num_heads, num_kv_heads) -- values match, do not transpose.
    """

    name: str
    hq: int  # query heads
    hkv: int  # key/value heads
    d: int  # head dim
    cmax: int  # kv-cache capacity (>= sum(seq_lens))
    seq_lens: tuple[int, ...]


# Mirror Vulkan sdpa_test.cpp:856/867/875 (3 param sets); cmax = sum rounded up.
REPLAY_SEQS = [
    ReplaySeq("small", 8, 4, 4, 16, (3, 1, 1, 5, 1, 1, 2)),
    ReplaySeq("small_d", 6, 2, 8, 16, (3, 1, 1, 5, 1, 1)),
    ReplaySeq("llama3", 24, 8, 128, 256, (111, 1, 1, 1, 57, 1, 1)),
]

# (head_dim, num_heads, num_kv_heads) from sdpa_test.cpp:856/867/875 -- guards a
# transposition of the (hq, hkv, d) field order against the Vulkan source.
VULKAN_PARAMS = {"small": (4, 8, 4), "small_d": (8, 6, 2), "llama3": (128, 24, 8)}


class SdpaModule(torch.nn.Module):
    """Fused SDPA at a given input_pos (is_causal, no mask/dropout/scale)."""

    def __init__(self, input_pos: int = 0):
        super().__init__()
        self.input_pos = input_pos

    def forward(self, q, k, v, k_cache, v_cache):
        return torch.ops.llama.sdpa_with_kv_cache(
            q, k, v, k_cache, v_cache, self.input_pos, q.shape[1], None, 0.0, True, None
        )


def _ramp(n, mod, off, denom=16.0):
    """Ramp ((i % mod) - off) / denom; exact in fp32 for power-of-two denom."""
    a = (np.arange(n) % mod).astype(np.float32)
    return ((a - off) / np.float32(denom)).astype(np.float32)


def _ramp_t(n, mod, off, t):
    """Step-indexed /16 ramp; the C++ sdpa_ramp_t mirrors this bit-for-bit.

    The 31*t phase desyncs each step's q/k/v; integer modulo keeps it exact in
    fp32. arange(n)+31*t stays well within int range (max n ~341k, t<=6).
    """
    a = ((np.arange(n) + 31 * t) % mod).astype(np.float32)
    return ((a - off) / 16.0).astype(np.float32)


def _step_inputs(seq: "ReplaySeq", t: int, s: int):
    """Deterministic per-step q/k/v the native harness reconstructs bit-for-bit."""
    q = torch.from_numpy(_ramp_t(s * seq.hq * seq.d, 17, 8, t)).reshape(
        1, s, seq.hq, seq.d
    )
    k = torch.from_numpy(_ramp_t(s * seq.hkv * seq.d, 13, 6, t)).reshape(
        1, s, seq.hkv, seq.d
    )
    v = torch.from_numpy(_ramp_t(s * seq.hkv * seq.d, 11, 5, t)).reshape(
        1, s, seq.hkv, seq.d
    )
    return q, k, v


def _det_inputs(cfg: SdpaConfig):
    """Deterministic fp32 inputs the native test reconstructs bit-for-bit.

    q/k_new/v_new use the /cfg.denom ramps. For decode (input_pos > 0) the first
    input_pos rows of each cache are seeded with prior_k/prior_v (flat over
    input_pos*Hkv*D elements); all other cache rows are zero.
    """
    q = torch.from_numpy(_ramp(cfg.s * cfg.hq * cfg.d, 17, 8, cfg.denom)).reshape(
        1, cfg.s, cfg.hq, cfg.d
    )
    k = torch.from_numpy(_ramp(cfg.s * cfg.hkv * cfg.d, 13, 6, cfg.denom)).reshape(
        1, cfg.s, cfg.hkv, cfg.d
    )
    v = torch.from_numpy(_ramp(cfg.s * cfg.hkv * cfg.d, 11, 5, cfg.denom)).reshape(
        1, cfg.s, cfg.hkv, cfg.d
    )

    k_cache = torch.zeros(1, cfg.cmax, cfg.hkv, cfg.d)
    v_cache = torch.zeros(1, cfg.cmax, cfg.hkv, cfg.d)
    if cfg.input_pos > 0:
        prior_n = cfg.input_pos * cfg.hkv * cfg.d
        prior_k = torch.from_numpy(_ramp(prior_n, 7, 3)).reshape(
            cfg.input_pos, cfg.hkv, cfg.d
        )
        prior_v = torch.from_numpy(_ramp(prior_n, 5, 2)).reshape(
            cfg.input_pos, cfg.hkv, cfg.d
        )
        k_cache[0, : cfg.input_pos] = prior_k
        v_cache[0, : cfg.input_pos] = prior_v
    return q, k, v, k_cache, v_cache


def _golden(cfg: SdpaConfig, q, k, v, k_cache, v_cache) -> torch.Tensor:
    """Reference attention output [1,S,Hq,D], computed in fp64 then cast to fp32.

    fp64 makes the reference the true answer (rounding ~1e-15), so the baked
    golden carries no fp32 accumulation error -- the GPU's fp32 error is measured
    against truth, not against another fp32 approximation. Builds the full K/V
    over the context, expands GQA groups, applies a causal mask offset by
    input_pos. Mirrors Vulkan sdpa_test.cpp::sdpa_reference_impl (decomposed).
    """
    context_len = cfg.s + cfg.input_pos
    g = cfg.hq // cfg.hkv
    qd, kd, vd = q.double(), k.double(), v.double()
    kcd, vcd = k_cache.double(), v_cache.double()

    # Full K/V over the context: prior cache rows then the new tokens.
    k_full = torch.empty(context_len, cfg.hkv, cfg.d, dtype=torch.float64)
    v_full = torch.empty(context_len, cfg.hkv, cfg.d, dtype=torch.float64)
    if cfg.input_pos > 0:
        k_full[: cfg.input_pos] = kcd[0, : cfg.input_pos]
        v_full[: cfg.input_pos] = vcd[0, : cfg.input_pos]
    k_full[cfg.input_pos : context_len] = kd[0]
    v_full[cfg.input_pos : context_len] = vd[0]

    # GQA-expand to Hq heads, then [Hq, context_len, D].
    qh = qd[0].transpose(0, 1)  # [Hq, S, D]
    kh = k_full.repeat_interleave(g, dim=1).transpose(0, 1)  # [Hq, ctx, D]
    vh = v_full.repeat_interleave(g, dim=1).transpose(0, 1)

    # Causal mask with offset: row s attends to context cols <= s + input_pos.
    mask = torch.full((cfg.s, context_len), float("-inf"), dtype=torch.float64)
    for s in range(cfg.s):
        mask[s, : s + cfg.input_pos + 1] = 0.0

    out = F.scaled_dot_product_attention(qh, kh, vh, attn_mask=mask)  # [Hq,S,D] f64
    return (
        out.transpose(0, 1)
        .reshape(1, cfg.s, cfg.hq, cfg.d)
        .to(torch.float32)
        .contiguous()
    )


def _export_pte(cfg: SdpaConfig, q, k, v, kc, vc):
    ep = torch.export.export(SdpaModule(cfg.input_pos), (q, k, v, kc, vc))
    return to_edge_transform_and_lower(
        ep, partitioner=[VulkanPartitioner()]
    ).to_executorch()


class TestSdpa(unittest.TestCase):
    def test_sdpa_export_delegates(self) -> None:
        for cfg in CONFIGS:
            with self.subTest(config=cfg.name):
                q, k, v, kc, vc = _det_inputs(cfg)
                et = _export_pte(cfg, q, k, v, kc, vc)
                found = any(
                    d.id == "VulkanBackend"
                    for plan in et.executorch_program.execution_plan
                    for d in plan.delegates
                )
                self.assertTrue(
                    found, f"Expected VulkanBackend delegate in {cfg.name}.pte"
                )

    def test_golden_matches_eager_op(self) -> None:
        # Oracle self-validation (mirrors Vulkan test_reference_sdpa): the fp64
        # golden and the shipped fp32 CPU op are independent refs that must agree.
        for cfg in CONFIGS:
            with self.subTest(config=cfg.name):
                q, k, v, kc, vc = _det_inputs(cfg)
                eager = SdpaModule(cfg.input_pos)(q, k, v, kc.clone(), vc.clone())
                golden = _golden(cfg, q, k, v, kc, vc)
                torch.testing.assert_close(eager, golden, atol=1e-4, rtol=1e-4)

    def test_replay_golden_matches_eager(self) -> None:
        # Pure-torch proof of the threading model BEFORE any GPU run: replay the
        # eager llama op with a host-threaded cache and assert each step's output
        # equals the accumulated-context golden. Covers the large-S-at-offset mask
        # path (small step (5,5), llama3 step (57,114)) absent from CONFIGS.
        for seq in REPLAY_SEQS:
            with self.subTest(seq=seq.name):
                self.assertEqual(
                    (seq.d, seq.hq, seq.hkv),
                    VULKAN_PARAMS[seq.name],
                    f"{seq.name}: (d,hq,hkv) diverges from the Vulkan param set",
                )
                self.assertLessEqual(sum(seq.seq_lens), seq.cmax)
                kc = torch.zeros(1, seq.cmax, seq.hkv, seq.d)
                vc = torch.zeros(1, seq.cmax, seq.hkv, seq.d)
                input_pos = 0
                for t, s in enumerate(seq.seq_lens):
                    cfg = SdpaConfig(
                        f"{seq.name}_step{t}",
                        seq.hq,
                        seq.hkv,
                        seq.d,
                        s,
                        seq.cmax,
                        input_pos,
                    )
                    q, k, v = _step_inputs(seq, t, s)
                    golden = _golden(cfg, q, k, v, kc, vc)
                    eager = SdpaModule(input_pos)(q, k, v, kc.clone(), vc.clone())
                    torch.testing.assert_close(eager, golden, atol=1e-4, rtol=1e-4)
                    kc[0, input_pos : input_pos + s] = k[0]
                    vc[0, input_pos : input_pos + s] = v[0]
                    input_pos += s

    def test_replay_export_delegates(self) -> None:
        # Every step .pte (incl. llama3-scale) must delegate to VulkanBackend.
        for seq in REPLAY_SEQS:
            kc = torch.zeros(1, seq.cmax, seq.hkv, seq.d)
            vc = torch.zeros(1, seq.cmax, seq.hkv, seq.d)
            input_pos = 0
            for t, s in enumerate(seq.seq_lens):
                with self.subTest(seq=seq.name, step=t):
                    cfg = SdpaConfig(
                        f"{seq.name}_step{t}",
                        seq.hq,
                        seq.hkv,
                        seq.d,
                        s,
                        seq.cmax,
                        input_pos,
                    )
                    q, k, v = _step_inputs(seq, t, s)
                    et = _export_pte(cfg, q, k, v, kc, vc)
                    found = any(
                        d.id == "VulkanBackend"
                        for plan in et.executorch_program.execution_plan
                        for d in plan.delegates
                    )
                    self.assertTrue(found, f"no delegate in {seq.name} step{t}")
                input_pos += s


def export_sdpa_model(cfg: SdpaConfig, pte_path: str, golden_path: str) -> None:
    """Export one config's fused SDPA .pte and its torch golden (raw LE fp32)."""
    q, k, v, kc, vc = _det_inputs(cfg)
    et = _export_pte(cfg, q, k, v, kc, vc)
    with open(pte_path, "wb") as f:
        f.write(et.buffer)
    golden = _golden(cfg, q, k, v, kc, vc).numpy().astype("<f4")
    golden.tofile(golden_path)
    print(f"Exported {pte_path}; golden {golden_path} ({golden.size} floats)")


def export_all_sdpa_models(out_dir: str) -> None:
    """Write all configs' sdpa_<name>.pte + sdpa_<name>.golden.bin to out_dir."""
    for cfg in CONFIGS:
        pte_path = os.path.join(out_dir, f"sdpa_{cfg.name}.pte")
        golden_path = os.path.join(out_dir, f"sdpa_{cfg.name}.golden.bin")
        export_sdpa_model(cfg, pte_path, golden_path)


def export_replay_sequences(out_dir: str) -> None:
    """Export one .pte + golden per (S, input_pos) step of each replay sequence.

    Threads a host reference cache exactly as the native harness threads the
    device cache: at each step the golden attends over the accumulated context,
    then the step's k/v are scattered into the ref cache for the next step.
    """
    for seq in REPLAY_SEQS:
        assert sum(seq.seq_lens) <= seq.cmax, f"{seq.name}: seq exceeds cmax"
        ref_kc = torch.zeros(1, seq.cmax, seq.hkv, seq.d)
        ref_vc = torch.zeros(1, seq.cmax, seq.hkv, seq.d)
        input_pos = 0
        for t, s in enumerate(seq.seq_lens):
            cfg = SdpaConfig(
                f"{seq.name}_step{t}", seq.hq, seq.hkv, seq.d, s, seq.cmax, input_pos
            )
            q, k, v = _step_inputs(seq, t, s)
            et = _export_pte(cfg, q, k, v, ref_kc, ref_vc)
            base = os.path.join(out_dir, f"sdpa_{seq.name}_step{t}_S{s}_pos{input_pos}")
            with open(base + ".pte", "wb") as f:
                f.write(et.buffer)
            golden = _golden(cfg, q, k, v, ref_kc, ref_vc).numpy().astype("<f4")
            golden.tofile(base + ".golden.bin")
            ref_kc[0, input_pos : input_pos + s] = k[0]
            ref_vc[0, input_pos : input_pos + s] = v[0]
            input_pos += s
            print(f"Exported {base}.pte; golden ({golden.size} floats)")


# --- Dynamic input_pos (runtime SymInt) decode path -------------------------
# A single .pte (fixed S=1) is replayed across decode steps with input_pos
# supplied at runtime as a tensor; input_pos[0].item() lowers to a SymInt the
# WebGPU backend reads via a live uniform + per-step resize hook (mirrors the
# Vulkan SymInt path). The native test reuses ONE module and advances input_pos.

DYN_DECODE_STEPS = 6  # S=1 decode steps; input_pos = 0..N-1


class DynamicSdpaModule(torch.nn.Module):
    """Fused SDPA with a runtime-dynamic input_pos (decode)."""

    def forward(self, q, k, v, k_cache, v_cache, input_pos):
        start = input_pos[0].item()
        return torch.ops.llama.sdpa_with_kv_cache(
            q, k, v, k_cache, v_cache, start, q.shape[1], None, 0.0, True, None
        )


def _export_dyn_pte(seq: "ReplaySeq", s: int):
    """Export one dynamic-input_pos .pte (fixed S=s). Asserts the start arg
    lowers to a SymInt before lowering, so a folded constant Int never silently
    passes (R4)."""
    q = torch.from_numpy(_ramp_t(s * seq.hq * seq.d, 17, 8, 0)).reshape(
        1, s, seq.hq, seq.d
    )
    k = torch.from_numpy(_ramp_t(s * seq.hkv * seq.d, 13, 6, 0)).reshape(
        1, s, seq.hkv, seq.d
    )
    v = torch.from_numpy(_ramp_t(s * seq.hkv * seq.d, 11, 5, 0)).reshape(
        1, s, seq.hkv, seq.d
    )
    kc = torch.zeros(1, seq.cmax, seq.hkv, seq.d)
    vc = torch.zeros(1, seq.cmax, seq.hkv, seq.d)
    ip = torch.tensor([0], dtype=torch.long)  # placeholder input_pos at build
    # Scoped (not process-wide): input_pos[0].item() must lower to a SymInt.
    with torch._dynamo.config.patch(capture_scalar_outputs=True):
        ep = torch.export.export(DynamicSdpaModule(), (q, k, v, kc, vc, ip))
    symint_nodes = [
        n.name
        for n in ep.graph_module.graph.nodes
        if isinstance(n.meta.get("val", None), torch.SymInt)
    ]
    if not symint_nodes:
        raise AssertionError(
            f"{seq.name}: dynamic input_pos did not lower to a SymInt "
            "(folded to a constant Int?)"
        )
    return to_edge_transform_and_lower(
        ep, partitioner=[VulkanPartitioner()]
    ).to_executorch()


def export_dynamic_decode(out_dir: str) -> None:
    """One sdpa_dyn_<name>.pte (S=1, runtime input_pos) + per-step decode goldens.

    Mirrors the host accumulation the native test threads: at step t the golden
    attends over input_pos=t prior tokens plus the new token.
    """
    for seq in REPLAY_SEQS:
        assert DYN_DECODE_STEPS <= seq.cmax, f"{seq.name}: decode exceeds cmax"
        et = _export_dyn_pte(seq, 1)
        pte_path = os.path.join(out_dir, f"sdpa_dyn_{seq.name}.pte")
        with open(pte_path, "wb") as f:
            f.write(et.buffer)
        ref_kc = torch.zeros(1, seq.cmax, seq.hkv, seq.d)
        ref_vc = torch.zeros(1, seq.cmax, seq.hkv, seq.d)
        for t in range(DYN_DECODE_STEPS):
            cfg = SdpaConfig(
                f"dyn_{seq.name}_step{t}", seq.hq, seq.hkv, seq.d, 1, seq.cmax, t
            )
            q, k, v = _step_inputs(seq, t, 1)
            golden = _golden(cfg, q, k, v, ref_kc, ref_vc).numpy().astype("<f4")
            golden.tofile(
                os.path.join(out_dir, f"sdpa_dyn_{seq.name}_step{t}.golden.bin")
            )
            ref_kc[0, t : t + 1] = k[0]
            ref_vc[0, t : t + 1] = v[0]
        print(f"Exported {pte_path}; {DYN_DECODE_STEPS} decode goldens")


class TestSdpaDynamic(unittest.TestCase):
    def test_dynamic_export_emits_symint(self) -> None:
        # R4: a real export must carry a SymInt start_pos, not a folded Int.
        for seq in REPLAY_SEQS[:1]:
            _export_dyn_pte(seq, 1)  # raises if no SymInt node

    def test_dynamic_decode_golden_matches_eager(self) -> None:
        # The threaded-cache decode golden must equal the eager op step-by-step.
        for seq in REPLAY_SEQS:
            ref_kc = torch.zeros(1, seq.cmax, seq.hkv, seq.d)
            ref_vc = torch.zeros(1, seq.cmax, seq.hkv, seq.d)
            for t in range(DYN_DECODE_STEPS):
                cfg = SdpaConfig(
                    f"dyn_{seq.name}_step{t}", seq.hq, seq.hkv, seq.d, 1, seq.cmax, t
                )
                q, k, v = _step_inputs(seq, t, 1)
                golden = _golden(cfg, q, k, v, ref_kc, ref_vc)
                eager = SdpaModule(t)(q, k, v, ref_kc.clone(), ref_vc.clone())
                torch.testing.assert_close(eager, golden, atol=1e-4, rtol=1e-4)
                ref_kc[0, t : t + 1] = k[0]
                ref_vc[0, t : t + 1] = v[0]


# --- In-graph mutable KV cache (true autoregressive decode) -----------------
# The KV cache is held as register_buffers (mutable buffers), so forward() feeds
# ONLY the new token (q/k/v, S=1) + dynamic input_pos; sdpa_with_kv_cache mutates
# the caches in place and the runtime persists them in-graph across forward()
# calls (no host threading). Goldens are the same torch reference the host-
# threaded decode uses, so a GPU match proves in-graph accumulation.
class DecodeCacheModule(torch.nn.Module):
    def __init__(self, hkv: int, d: int, cmax: int):
        super().__init__()
        self.register_buffer("k_cache", torch.zeros(1, cmax, hkv, d))
        self.register_buffer("v_cache", torch.zeros(1, cmax, hkv, d))

    def forward(self, q, k, v, input_pos):
        start = input_pos[0].item()
        return torch.ops.llama.sdpa_with_kv_cache(
            q,
            k,
            v,
            self.k_cache,
            self.v_cache,
            start,
            q.shape[1],
            None,
            0.0,
            True,
            None,
        )


def export_incache_decode(out_dir: str) -> None:
    """One sdpa_incache_<name>.pte (mutable-buffer KV cache) + per-step decode
    goldens. forward() feeds only q/k/v + input_pos; the cache persists in-graph.
    """
    for seq in REPLAY_SEQS:
        assert DYN_DECODE_STEPS <= seq.cmax, f"{seq.name}: decode exceeds cmax"
        m = DecodeCacheModule(seq.hkv, seq.d, seq.cmax)
        q, k, v = _step_inputs(seq, 0, 1)
        ip = torch.tensor([0], dtype=torch.long)
        # Scoped (not process-wide): input_pos[0].item() must lower to a SymInt.
        with torch._dynamo.config.patch(capture_scalar_outputs=True):
            ep = torch.export.export(m, (q, k, v, ip))
        et = to_edge_transform_and_lower(
            ep, partitioner=[VulkanPartitioner()]
        ).to_executorch()
        pte = os.path.join(out_dir, f"sdpa_incache_{seq.name}.pte")
        with open(pte, "wb") as f:
            f.write(et.buffer)
        ref_kc = torch.zeros(1, seq.cmax, seq.hkv, seq.d)
        ref_vc = torch.zeros(1, seq.cmax, seq.hkv, seq.d)
        for t in range(DYN_DECODE_STEPS):
            cfg = SdpaConfig(
                f"incache_{seq.name}_step{t}", seq.hq, seq.hkv, seq.d, 1, seq.cmax, t
            )
            q, k, v = _step_inputs(seq, t, 1)
            golden = _golden(cfg, q, k, v, ref_kc, ref_vc).numpy().astype("<f4")
            golden.tofile(
                os.path.join(out_dir, f"sdpa_incache_{seq.name}_step{t}.golden.bin")
            )
            ref_kc[0, t : t + 1] = k[0]
            ref_vc[0, t : t + 1] = v[0]
        print(f"Exported {pte}; {DYN_DECODE_STEPS} in-graph-cache decode goldens")


if __name__ == "__main__":
    unittest.main()
