# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Dynamic tensor-shape (Option 2) export tests via VulkanPartitioner.

Exports ONE graph built at the upper-bound seq-len MAXS that the native runtime
(`test/native/test_dynamic_shape.cpp`) then runs at several live S, asserting the
output matches the torch golden and that the static path is unchanged. Numerics
are checked in the native test; this verifies the dynamic export side + writes
goldens.
"""

import os
import unittest

import torch
from executorch.backends.vulkan.partitioner.vulkan_partitioner import VulkanPartitioner
from executorch.exir import to_edge_transform_and_lower

MAXS = 128  # upper bound for the dynamic seq-len dim (within the 1D dispatch cap)
HIDDEN = 64


def _rms(x: torch.Tensor, weight: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    x_f32 = x.to(torch.float32)
    var = x_f32.pow(2).mean(dim=-1, keepdim=True)
    return (x_f32 * torch.rsqrt(var + eps)) * weight


class RmsNormModule(torch.nn.Module):
    def __init__(self, hidden: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.weight = torch.nn.Parameter(
            torch.linspace(0.5, 1.5, hidden, dtype=torch.float32)
        )
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return _rms(x, self.weight, self.eps)


class RmsChainModule(torch.nn.Module):
    """rms(rms(x)) — two ops; exercises the resize-cascade (DD-4)."""

    def __init__(self, hidden: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.w1 = torch.nn.Parameter(
            torch.linspace(0.5, 1.5, hidden, dtype=torch.float32)
        )
        self.w2 = torch.nn.Parameter(
            torch.linspace(1.5, 0.5, hidden, dtype=torch.float32)
        )
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return _rms(_rms(x, self.w1, self.eps), self.w2, self.eps)


class RmsResidualModule(torch.nn.Module):
    """rms(x) + x — rms op feeding an add op; proves the cross-op resize cascade."""

    def __init__(self, hidden: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.w = torch.nn.Parameter(
            torch.linspace(0.5, 1.5, hidden, dtype=torch.float32)
        )
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return _rms(x, self.w, self.eps) + x


class RmsMulModule(torch.nn.Module):
    """rms(x) * x — exercises the mul op (two same-shape dynamic operands)."""

    def __init__(self, hidden: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.w = torch.nn.Parameter(
            torch.linspace(0.5, 1.5, hidden, dtype=torch.float32)
        )
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return _rms(x, self.w, self.eps) * x


def _ramp(shape) -> torch.Tensor:
    n = 1
    for d in shape:
        n *= d
    return torch.linspace(-1.0, 1.0, n, dtype=torch.float32).reshape(shape)


def _export(model, example_inputs, dynamic_shapes, path: str) -> None:
    ep = torch.export.export(model, example_inputs, dynamic_shapes=dynamic_shapes)
    et = to_edge_transform_and_lower(ep, partitioner=[VulkanPartitioner()]).to_executorch()
    found = any(
        d.id == "VulkanBackend"
        for plan in et.executorch_program.execution_plan
        for d in plan.delegates
    )
    assert found, f"Expected VulkanBackend delegate in {path}"
    with open(path, "wb") as f:
        f.write(et.buffer)
    print(f"Exported {path}")


def _write_goldens(model, prefix: str, out_dir: str, s_values) -> None:
    for s in s_values:
        x = _ramp((1, 1, s, HIDDEN))
        with torch.no_grad():
            g = model(x)
        x.detach().numpy().astype("<f4").tofile(
            os.path.join(out_dir, f"{prefix}.S{s}.input.bin")
        )
        g.detach().numpy().astype("<f4").tofile(
            os.path.join(out_dir, f"{prefix}.S{s}.golden.bin")
        )
        print(f"  golden {prefix} S={s}")


def export_dynamic_shape_cases(out_dir: str) -> None:
    """Write the dynamic + static .pte's and per-S goldens for the native test."""
    os.makedirs(out_dir, exist_ok=True)
    s_dim = torch.export.Dim("s", min=1, max=MAXS)

    # 1) Single dynamic rms_norm, graph built at S=MAXS (upper bound).
    rms = RmsNormModule(HIDDEN)
    _export(
        rms,
        (_ramp((1, 1, MAXS, HIDDEN)),),
        {"x": {2: s_dim}},
        os.path.join(out_dir, "dyn_rms.pte"),
    )
    _write_goldens(rms, "dyn_rms", out_dir, [MAXS, 64, 8, 1])

    # 2) Two-op chain (cascade): rms(rms(x)).
    chain = RmsChainModule(HIDDEN)
    _export(
        chain,
        (_ramp((1, 1, MAXS, HIDDEN)),),
        {"x": {2: s_dim}},
        os.path.join(out_dir, "dyn_rms_chain.pte"),
    )
    _write_goldens(chain, "dyn_rms_chain", out_dir, [MAXS, 16, 1])

    # 2b) rms(x)+x residual — cross-op (rms->add) cascade.
    resid = RmsResidualModule(HIDDEN)
    _export(
        resid,
        (_ramp((1, 1, MAXS, HIDDEN)),),
        {"x": {2: s_dim}},
        os.path.join(out_dir, "dyn_residual.pte"),
    )
    _write_goldens(resid, "dyn_residual", out_dir, [MAXS, 32, 1])

    # 2c) rms(x)*x — exercises the mul op resize.
    rmsmul = RmsMulModule(HIDDEN)
    _export(
        rmsmul,
        (_ramp((1, 1, MAXS, HIDDEN)),),
        {"x": {2: s_dim}},
        os.path.join(out_dir, "dyn_rmsmul.pte"),
    )
    _write_goldens(rmsmul, "dyn_rmsmul", out_dir, [MAXS, 32, 1])

    # 2d) 4-bit quantized linear with a DYNAMIC rows (M) dim — prefill GEMM.
    _export_dynamic_linear(out_dir)

    # 2e) Fused SDPA with a DYNAMIC seq-len S (prefill, input_pos=0).
    _export_dynamic_sdpa(out_dir)

    # 2f) 4-bit embedding with a DYNAMIC token count (int64 indices).
    _export_dynamic_embedding(out_dir)

    # 2g) Interleaved RoPE with a DYNAMIC seq-len S (two outputs xq/xk).
    _export_dynamic_rope(out_dir)

    # 3) Static rms_norm (no dynamic dim) — regression: must stay byte-identical.
    static = RmsNormModule(HIDDEN)
    _export(
        static,
        (_ramp((1, 1, 8, HIDDEN)),),
        None,
        os.path.join(out_dir, "static_rms.pte"),
    )
    _write_goldens(static, "static_rms", out_dir, [8])


# Quantized linear: K x N weight, dynamic rows M; input [M, K], output [M, N].
LIN_K = 64
LIN_N = 128
LIN_GROUP = 32
LIN_MAXM = 128


def _export_dynamic_linear(out_dir: str) -> None:
    from executorch.backends.webgpu.test.ops.quantized_linear.test_quantized_linear import (
        _fp64_golden,
        _make_quantized_model,
    )

    model = _make_quantized_model(LIN_K, LIN_N, LIN_GROUP)
    x = _ramp((LIN_MAXM, LIN_K))
    m_dim = torch.export.Dim("m", min=1, max=LIN_MAXM)
    ep = torch.export.export(model, (x,), dynamic_shapes=({0: m_dim},))
    et = to_edge_transform_and_lower(ep, partitioner=[VulkanPartitioner()]).to_executorch()
    assert any(
        d.id == "VulkanBackend"
        for plan in et.executorch_program.execution_plan
        for d in plan.delegates
    ), "linear_q4gsw not delegated"
    with open(os.path.join(out_dir, "dyn_linear.pte"), "wb") as f:
        f.write(et.buffer)
    print("Exported dyn_linear.pte")
    for m in [LIN_MAXM, 32, 1]:
        xm = _ramp((m, LIN_K))
        g = _fp64_golden(model, xm).astype("<f4")  # [m, N]
        xm.detach().numpy().astype("<f4").tofile(
            os.path.join(out_dir, f"dyn_linear.S{m}.input.bin")
        )
        g.tofile(os.path.join(out_dir, f"dyn_linear.S{m}.golden.bin"))
        print(f"  golden dyn_linear M={m}")


# Dynamic SDPA: GQA prefill (input_pos=0), q/k/v seq-len dynamic.
SD_HQ = 8
SD_HKV = 2
SD_D = 16
SD_CMAX = 64
SD_MAXS = 64


def _export_dynamic_sdpa(out_dir: str) -> None:
    from executorch.backends.webgpu.test.ops.sdpa.test_sdpa import (
        _det_inputs,
        _golden,
        SdpaConfig,
        SdpaModule,
    )

    def cfg(s: int) -> "SdpaConfig":
        return SdpaConfig("dyn", SD_HQ, SD_HKV, SD_D, s, SD_CMAX, 0)

    model = SdpaModule(0)
    q, k, v, kc, vc = _det_inputs(cfg(SD_MAXS))
    s_dim = torch.export.Dim("s", min=1, max=SD_MAXS)
    ds = ({1: s_dim}, {1: s_dim}, {1: s_dim}, None, None)
    ep = torch.export.export(model, (q, k, v, kc, vc), dynamic_shapes=ds)
    et = to_edge_transform_and_lower(ep, partitioner=[VulkanPartitioner()]).to_executorch()
    assert any(
        d.id == "VulkanBackend"
        for plan in et.executorch_program.execution_plan
        for d in plan.delegates
    ), "sdpa not delegated"
    with open(os.path.join(out_dir, "sdpa_dyn.pte"), "wb") as f:
        f.write(et.buffer)
    print("Exported sdpa_dyn.pte")
    for s in [SD_MAXS, 16, 1]:
        c = cfg(s)
        q, k, v, kc, vc = _det_inputs(c)
        g = _golden(c, q, k, v, kc, vc)
        for name, t in [("q", q), ("k", k), ("v", v), ("kc", kc), ("vc", vc), ("golden", g)]:
            t.detach().cpu().numpy().astype("<f4").tofile(
                os.path.join(out_dir, f"sdpa_dyn.S{s}.{name}.bin")
            )
        print(f"  golden sdpa_dyn S={s} (golden shape {tuple(g.shape)})")


# Dynamic embedding: int64 token ids (dynamic count) -> [N, EMBED] fp32.
EMB_VOCAB = 64
EMB_DIM = 64
EMB_GROUP = 32
EMB_MAXN = 16


def _export_dynamic_embedding(out_dir: str) -> None:
    from executorch.backends.webgpu.test.ops.embedding_q4gsw.test_embedding_q4gsw import (
        _make_quantized_model,
        _quant_params,
        Shape,
    )

    shape = Shape("dyn", EMB_VOCAB, EMB_DIM, EMB_GROUP, list(range(EMB_MAXN)))
    qm = _make_quantized_model(shape)
    idx_max = torch.arange(EMB_MAXN, dtype=torch.long)
    n_dim = torch.export.Dim("n", min=1, max=EMB_MAXN)
    ep = torch.export.export(qm, (idx_max,), dynamic_shapes=({0: n_dim},))
    et = to_edge_transform_and_lower(ep, partitioner=[VulkanPartitioner()]).to_executorch()
    assert any(
        d.id == "VulkanBackend"
        for plan in et.executorch_program.execution_plan
        for d in plan.delegates
    ), "embedding_q4gsw not delegated"
    with open(os.path.join(out_dir, "emb_dyn.pte"), "wb") as f:
        f.write(et.buffer)
    print("Exported emb_dyn.pte")
    weight, scales, group_size = _quant_params(qm)
    for n in [EMB_MAXN, 8, 1]:
        idx = (torch.arange(n, dtype=torch.long) * 7) % EMB_VOCAB
        g = torch.ops.et_vk.embedding_q4gsw.default(
            weight, scales, group_size, idx, False
        )
        idx.detach().numpy().astype("<i8").tofile(
            os.path.join(out_dir, f"emb_dyn.S{n}.idx.bin")
        )
        g.detach().numpy().astype("<f4").tofile(
            os.path.join(out_dir, f"emb_dyn.S{n}.golden.bin")
        )
        print(f"  golden emb_dyn N={n} (shape {tuple(g.shape)})")


# Dynamic RoPE: xq/xk + freqs all share a dynamic seq-len S.
ROPE_NH = 8
ROPE_NKV = 2
ROPE_HD = 64
ROPE_MAXS = 16


def _export_dynamic_rope(out_dir: str) -> None:
    from executorch.backends.webgpu.test.ops.rope.test_rope import (
        _golden,
        _inputs,
        Shape,
    )
    from executorch.examples.models.llama.rope import RotaryEmbedding

    xq, xk, fc, fs = _inputs(Shape("dyn", 1, ROPE_MAXS, ROPE_NH, ROPE_NKV, ROPE_HD))
    s_dim = torch.export.Dim("s", min=1, max=ROPE_MAXS)
    ds = ({1: s_dim}, {1: s_dim}, {0: s_dim}, {0: s_dim})
    ep = torch.export.export(RotaryEmbedding().eval(), (xq, xk, fc, fs), dynamic_shapes=ds)
    et = to_edge_transform_and_lower(ep, partitioner=[VulkanPartitioner()]).to_executorch()
    assert any(
        d.id == "VulkanBackend"
        for plan in et.executorch_program.execution_plan
        for d in plan.delegates
    ), "apply_rotary_emb not delegated"
    with open(os.path.join(out_dir, "rope_dyn.pte"), "wb") as f:
        f.write(et.buffer)
    print("Exported rope_dyn.pte")
    for s in [ROPE_MAXS, 8, 1]:
        xq, xk, fc, fs = _inputs(Shape("dyn", 1, s, ROPE_NH, ROPE_NKV, ROPE_HD))
        gq, gk = _golden(xq, xk, fc, fs)
        for name, t in [("xq", xq), ("xk", xk), ("fc", fc), ("fs", fs), ("gq", gq), ("gk", gk)]:
            t.detach().cpu().numpy().astype("<f4").tofile(
                os.path.join(out_dir, f"rope_dyn.S{s}.{name}.bin")
            )
        print(f"  golden rope_dyn S={s}")


class TestDynamicShapeExport(unittest.TestCase):
    def test_export_dynamic_rms(self) -> None:
        import tempfile

        with tempfile.TemporaryDirectory() as d:
            export_dynamic_shape_cases(d)
            self.assertTrue(os.path.exists(os.path.join(d, "dyn_rms.pte")))
            self.assertTrue(os.path.exists(os.path.join(d, "dyn_rms.S1.golden.bin")))


if __name__ == "__main__":
    unittest.main()
