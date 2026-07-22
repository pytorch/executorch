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
from executorch.exir.backend.utils import get_delegates, get_non_lowered_nodes

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


class SigmoidModule(torch.nn.Module):
    """sigmoid(x) — elementwise; resize hook recomputes dispatch from live numel."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.sigmoid(x)


class SwiGluModule(torch.nn.Module):
    def __init__(
        self,
        reverse_inner: bool = False,
        reverse_outer: bool = False,
        extra_gate_consumer: bool = False,
        extra_sigmoid_consumer: bool = False,
        extra_silu_consumer: bool = False,
        expose_intermediates: bool = False,
        graph_output: str = "",
        separate_inputs: bool = False,
        interleaved_projection: bool = False,
        width: int = 8192,
    ) -> None:
        super().__init__()
        from torchao.quantization.granularity import PerGroup
        from torchao.quantization.quant_api import IntxWeightOnlyConfig, quantize_

        def make_q4(seed: int):
            torch.manual_seed(seed)
            linear = torch.nn.Linear(64, width, bias=False).eval()
            quantize_(
                linear,
                IntxWeightOnlyConfig(weight_dtype=torch.int4, granularity=PerGroup(32)),
            )
            return linear

        self.gate_proj = make_q4(0)
        self.up_proj = make_q4(1)
        self.interleaved_proj = make_q4(2) if interleaved_projection else None
        self.reverse_inner = reverse_inner
        self.reverse_outer = reverse_outer
        self.extra_gate_consumer = extra_gate_consumer
        self.extra_sigmoid_consumer = extra_sigmoid_consumer
        self.extra_silu_consumer = extra_silu_consumer
        self.expose_intermediates = expose_intermediates
        self.graph_output = graph_output
        self.separate_inputs = separate_inputs
        self.interleaved_projection = interleaved_projection

    def forward(self, x: torch.Tensor, up_input: torch.Tensor | None = None):
        gate = self.gate_proj(x)
        up = self.up_proj(up_input if self.separate_inputs else x)
        sigmoid = torch.sigmoid(gate)
        silu = sigmoid * gate if self.reverse_inner else gate * sigmoid
        interleaved = self.interleaved_proj(x) if self.interleaved_projection else None
        output = up * silu if self.reverse_outer else silu * up
        if self.extra_gate_consumer:
            return output + gate
        if self.extra_sigmoid_consumer:
            return output + sigmoid
        if self.extra_silu_consumer:
            return output + silu
        if self.expose_intermediates:
            return output, gate, sigmoid, silu
        if self.graph_output == "gate":
            return output, gate
        if self.graph_output == "sigmoid":
            return output, sigmoid
        if self.graph_output == "silu":
            return output, silu
        if interleaved is not None:
            return output + interleaved
        return output


class SelectModule(torch.nn.Module):
    """x.select(0, -1) — negative index resolved live + dynamic output dispatch."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x.select(0, -1)


def _ramp(shape) -> torch.Tensor:
    n = 1
    for d in shape:
        n *= d
    return torch.linspace(-1.0, 1.0, n, dtype=torch.float32).reshape(shape)


def _lower_fully_delegated(ep, label: str):
    edge = to_edge_transform_and_lower(ep, partitioner=[VulkanPartitioner()])
    graph = edge.exported_program().graph_module.graph
    delegates = get_delegates(graph)
    portable = get_non_lowered_nodes(graph)
    if len(delegates) != 1:
        raise RuntimeError(f"{label}: expected one delegate, got {len(delegates)}")
    if portable:
        raise RuntimeError(f"{label}: non-lowered nodes: {portable}")
    et = edge.to_executorch()
    delegate_ids = [
        delegate.id
        for plan in et.executorch_program.execution_plan
        for delegate in plan.delegates
    ]
    if delegate_ids != ["VulkanBackend"]:
        raise RuntimeError(f"{label}: serialized delegates: {delegate_ids}")
    return et


def _export(model, example_inputs, dynamic_shapes, path: str) -> None:
    ep = torch.export.export(model, example_inputs, dynamic_shapes=dynamic_shapes)
    et = _lower_fully_delegated(ep, path)
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

    # 2d) 4-bit quantized linear with a DYNAMIC rows (M) dim — prefill GEMM
    # (register-tiled N=128) + a shmem-GEMM-routed variant (N=2048).
    _export_dynamic_linear(out_dir)
    _export_dynamic_linear(
        out_dir,
        n=LIN_SHMEM_N,
        prefix="dyn_linear_shmem",
        k=LIN_ALT_K,
        group=LIN_ALT_GROUP,
    )
    _export_dynamic_linear(
        out_dir,
        prefix="dyn_linear_tiled",
        k=LIN_ALT_K,
        group=LIN_ALT_GROUP,
    )
    _export_static_linear(out_dir, 1, "static_linear_m1")
    _export_static_linear(out_dir, 32, "static_linear_m32")
    _export_dynamic_bk64_linear_cases(out_dir)

    # 2e) Fused SDPA with a DYNAMIC seq-len S (prefill, input_pos=0).
    _export_dynamic_sdpa(out_dir)
    _export_dynamic_k16_sdpa_cases(out_dir)
    _export_combined_routes(out_dir)
    _export_dynamic_sdpa_wide(out_dir)
    _export_static_sdpa(out_dir, 1, "static_sdpa_s1")
    _export_static_sdpa(out_dir, 16, "static_sdpa_s16")

    # 2f) 4-bit embedding with a DYNAMIC token count (int64 indices).
    _export_dynamic_embedding(out_dir)

    # 2g) Interleaved RoPE with a DYNAMIC seq-len S (two outputs xq/xk).
    _export_dynamic_rope(out_dir)

    # 2h) Elementwise sigmoid with a DYNAMIC seq-len S.
    sig = SigmoidModule()
    _export(
        sig,
        (_ramp((1, 1, MAXS, HIDDEN)),),
        {"x": {2: s_dim}},
        os.path.join(out_dir, "dyn_sigmoid.pte"),
    )
    _write_goldens(sig, "dyn_sigmoid", out_dir, [MAXS, 32, 1])

    # 2i) Dynamic SwiGLU graph patterns. The Llama-width fixture forces a 2D
    # fused dispatch at M=512; compact variants isolate commutative matching and
    # graph/intermediate ownership guards around same-input q4 projections.
    _export_dynamic_swiglu(out_dir)

    # 2j) select_copy(0, -1) over a DYNAMIC seq-len S (negative live index).
    _export_dynamic_select(out_dir)

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
LIN_ALT_K = 72  # K%16 != 0 disables Steel while K%8 keeps bicol eligible.
LIN_ALT_GROUP = 24
LIN_SHMEM_N = 2048
LIN_GROUP = 32
LIN_MAXM = 128

BK64_K = 2048
BK64_N = 2048
BK64_KV_N = 512
BK64_GATE_N = 8192
BK64_DOWN_K = 8192
BK64_GROUP = 64
BK64_MAXM = 512
BK64_LIVE_M = (BK64_MAXM, 511, 508, 128, 127, 1)
BK64_OPTIMIZED_M = (BK64_MAXM, 508, 128)
BK64_QKV_LIVE_M = (BK64_MAXM, 511, 508, 128, 127, 16, 2, 1)


SWIGLU_MAXM = 512
SWIGLU_WIDTH = 8192
SWIGLU_SMALL_WIDTH = 64
SWIGLU_K = 64


def _swiglu_inputs(model: SwiGluModule, m: int):
    x = _ramp((m, SWIGLU_K))
    if model.separate_inputs:
        return x, torch.flip(x, dims=[-1]).contiguous()
    return (x,)


def _write_swiglu_goldens(
    model: SwiGluModule,
    prefix: str,
    out_dir: str,
    m_values,
) -> None:
    for m in m_values:
        inputs = _swiglu_inputs(model, m)
        with torch.no_grad():
            golden = model(*inputs)
        base = os.path.join(out_dir, f"{prefix}.S{m}.")
        inputs[0].detach().numpy().astype("<f4").tofile(base + "input.bin")
        if model.separate_inputs:
            inputs[1].detach().numpy().astype("<f4").tofile(base + "up_input.bin")
        if isinstance(golden, tuple):
            for index, output in enumerate(golden):
                output.detach().numpy().astype("<f4").tofile(
                    base + f"golden{index}.bin"
                )
        else:
            golden.detach().numpy().astype("<f4").tofile(base + "golden.bin")
        print(f"  golden {prefix} M={m}")


def _export_swiglu_case(
    out_dir: str,
    prefix: str,
    model: SwiGluModule,
    m_values,
) -> None:
    inputs = _swiglu_inputs(model, SWIGLU_MAXM)
    m_dim = torch.export.Dim("m", min=1, max=SWIGLU_MAXM)
    dynamic_shapes = tuple({0: m_dim} for _ in inputs)
    ep = torch.export.export(
        model.eval(),
        inputs,
        dynamic_shapes=dynamic_shapes,
    )
    et = _lower_fully_delegated(ep, prefix)
    with open(os.path.join(out_dir, f"{prefix}.pte"), "wb") as f:
        f.write(et.buffer)
    print(f"Exported {prefix}.pte")
    _write_swiglu_goldens(model, prefix, out_dir, m_values)


def _export_dynamic_swiglu(out_dir: str) -> None:
    _export_swiglu_case(
        out_dir,
        "dyn_swiglu",
        SwiGluModule(width=SWIGLU_WIDTH),
        [SWIGLU_MAXM, 128, 1],
    )
    _export_swiglu_case(
        out_dir,
        "dyn_swiglu_inner_reversed",
        SwiGluModule(reverse_inner=True, width=SWIGLU_SMALL_WIDTH),
        [128],
    )
    for prefix, kwargs in (
        ("dyn_swiglu_extra_sigmoid_consumer", {"extra_sigmoid_consumer": True}),
        ("dyn_swiglu_extra_silu_consumer", {"extra_silu_consumer": True}),
        ("dyn_swiglu_gate_graph_output", {"graph_output": "gate"}),
        ("dyn_swiglu_sigmoid_graph_output", {"graph_output": "sigmoid"}),
        ("dyn_swiglu_silu_graph_output", {"graph_output": "silu"}),
        ("dyn_swiglu_different_inputs", {"separate_inputs": True}),
        (
            "dyn_swiglu_interleaved_q4",
            {"interleaved_projection": True},
        ),
    ):
        _export_swiglu_case(
            out_dir,
            prefix,
            SwiGluModule(width=SWIGLU_SMALL_WIDTH, **kwargs),
            [128],
        )
    _export_swiglu_case(
        out_dir,
        "dyn_swiglu_outer_reversed",
        SwiGluModule(reverse_outer=True, width=SWIGLU_SMALL_WIDTH),
        [128],
    )
    _export_swiglu_case(
        out_dir,
        "dyn_swiglu_extra_gate_consumer",
        SwiGluModule(extra_gate_consumer=True, width=SWIGLU_SMALL_WIDTH),
        [128],
    )
    _export_swiglu_case(
        out_dir,
        "dyn_swiglu_graph_outputs",
        SwiGluModule(expose_intermediates=True, width=SWIGLU_SMALL_WIDTH),
        [128],
    )


def _export_dynamic_linear(
    out_dir: str,
    n: int = LIN_N,
    prefix: str = "dyn_linear",
    k: int = LIN_K,
    group: int = LIN_GROUP,
) -> None:
    from executorch.backends.webgpu.test.ops.test_quantized_linear import (
        _fp64_golden,
        _make_quantized_model,
    )

    model = _make_quantized_model(k, n, group)
    x = _ramp((LIN_MAXM, k))
    m_dim = torch.export.Dim("m", min=1, max=LIN_MAXM)
    ep = torch.export.export(model, (x,), dynamic_shapes=({0: m_dim},))
    et = _lower_fully_delegated(ep, prefix)
    with open(os.path.join(out_dir, f"{prefix}.pte"), "wb") as f:
        f.write(et.buffer)
    print(f"Exported {prefix}.pte")
    for m in [LIN_MAXM, 32, 1]:
        xm = _ramp((m, k))
        g = _fp64_golden(model, xm).astype("<f4")  # [m, N]
        xm.detach().numpy().astype("<f4").tofile(
            os.path.join(out_dir, f"{prefix}.S{m}.input.bin")
        )
        g.tofile(os.path.join(out_dir, f"{prefix}.S{m}.golden.bin"))
        print(f"  golden {prefix} M={m}")


def _make_bk64_model(
    *,
    k: int = BK64_K,
    n: int = BK64_N,
    group: int = BK64_GROUP,
    bias: bool = False,
    seed: int = 11,
) -> torch.nn.Module:
    from torchao.quantization.granularity import PerGroup
    from torchao.quantization.quant_api import IntxWeightOnlyConfig, quantize_

    torch.manual_seed(seed)
    model = torch.nn.Linear(k, n, bias=bias).eval()
    if model.bias is not None:
        with torch.no_grad():
            model.bias.copy_(torch.linspace(-0.25, 0.25, n))
    quantize_(
        model,
        IntxWeightOnlyConfig(weight_dtype=torch.int4, granularity=PerGroup(group)),
    )
    return model


def _bk64_golden(model: torch.nn.Module, x: torch.Tensor) -> torch.Tensor:
    golden = x.double() @ model.weight.dequantize().double().t()
    if model.bias is not None:
        golden = golden + model.bias.double()
    return golden.to(torch.float32)


def _bk64_input(m: int, k: int) -> torch.Tensor:
    flat = torch.arange(m * k, dtype=torch.int64)
    hashed = (flat * 37 + torch.div(flat, 16, rounding_mode="floor") * 53) % 257
    return ((hashed.to(torch.float32) - 128.0) / 128.0).reshape(m, k)


class Bk64ShapeAwareLinear(torch.nn.Module):
    def __init__(self, projection: torch.nn.Module, output_width: int) -> None:
        super().__init__()
        self.projection = projection
        self.output_width = output_width

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.projection(x).reshape(1, x.shape[0], self.output_width)


class Bk64Qkv(torch.nn.Module):
    def __init__(
        self,
        *,
        widths=(BK64_N, BK64_KV_N, BK64_KV_N),
        group: int = BK64_GROUP,
        bias: bool = False,
        separate_v_input: bool = False,
    ) -> None:
        super().__init__()
        self.q = _make_bk64_model(n=widths[0], group=group, bias=bias, seed=21)
        self.k = _make_bk64_model(n=widths[1], group=group, bias=bias, seed=22)
        self.v = _make_bk64_model(n=widths[2], group=group, bias=bias, seed=23)
        self.widths = widths
        self.separate_v_input = separate_v_input

    def forward(
        self,
        x: torch.Tensor,
        v_input: torch.Tensor | None = None,
    ):
        q = self.q(x).reshape(1, x.shape[0], self.widths[0])
        k = self.k(x).reshape(1, x.shape[0], self.widths[1])
        v_source = v_input if self.separate_v_input else x
        v = self.v(v_source).reshape(1, v_source.shape[0], self.widths[2])
        return q, k, v


def _export_bk64_program(
    model: torch.nn.Module,
    x: torch.Tensor,
    n: int,
    prefix: str,
):
    export_model = Bk64ShapeAwareLinear(model, n).eval()
    m_dim = torch.export.Dim("m", min=1, max=BK64_MAXM)
    ep = torch.export.export(export_model, (x,), dynamic_shapes=({0: m_dim},))
    if not any(node.target == torch.ops.aten.sym_size.int for node in ep.graph.nodes):
        raise RuntimeError(f"{prefix}: dynamic q4 fixture lost aten.sym_size.int")
    return ep


def _export_dynamic_bk64_linear_case(
    out_dir: str,
    prefix: str,
    *,
    k: int = BK64_K,
    n: int = BK64_N,
    group: int = BK64_GROUP,
    bias: bool = False,
    live_m=BK64_LIVE_M,
) -> None:
    model = _make_bk64_model(k=k, n=n, group=group, bias=bias)
    x = _bk64_input(BK64_MAXM, k)
    ep = _export_bk64_program(model, x, n, prefix)
    et = _lower_fully_delegated(ep, prefix)
    with open(os.path.join(out_dir, f"{prefix}.pte"), "wb") as f:
        f.write(et.buffer)
    print(f"Exported {prefix}.pte")
    for m in live_m:
        xm = _bk64_input(m, k)
        golden = _bk64_golden(model, xm)
        xm.detach().numpy().astype("<f4").tofile(
            os.path.join(out_dir, f"{prefix}.S{m}.input.bin")
        )
        golden.detach().numpy().astype("<f4").tofile(
            os.path.join(out_dir, f"{prefix}.S{m}.golden.bin")
        )
        print(f"  golden {prefix} M={m}")


def _export_dynamic_bk64_linear_cases(out_dir: str) -> None:
    os.makedirs(out_dir, exist_ok=True)
    _export_dynamic_bk64_linear_case(out_dir, "dyn_linear_bk64")
    _export_dynamic_bk64_linear_case(
        out_dir,
        "dyn_linear_bk64_gate",
        n=BK64_GATE_N,
        live_m=BK64_OPTIMIZED_M,
    )
    _export_dynamic_bk64_linear_case(
        out_dir,
        "dyn_linear_bk64_down",
        k=BK64_DOWN_K,
        live_m=BK64_OPTIMIZED_M,
    )
    _export_dynamic_bk64_linear_case(
        out_dir,
        "dyn_linear_bk64_group32",
        group=32,
        live_m=[128],
    )
    _export_dynamic_bk64_linear_case(
        out_dir,
        "dyn_linear_bk64_bias",
        bias=True,
        live_m=[128],
    )
    _export_dynamic_bk64_linear_case(
        out_dir,
        "dyn_linear_bk64_kv_shape",
        n=BK64_KV_N,
        live_m=[128],
    )
    _export_dynamic_bk64_qkv_cases(out_dir)


def _export_dynamic_bk64_qkv_case(
    out_dir: str,
    prefix: str,
    model: Bk64Qkv,
    live_m,
) -> None:
    inputs = (_bk64_input(BK64_MAXM, BK64_K),)
    if model.separate_v_input:
        inputs += (torch.flip(inputs[0], dims=[-1]).contiguous(),)
    m_dim = torch.export.Dim("m", min=1, max=BK64_MAXM)
    dynamic_shapes = tuple({0: m_dim} for _ in inputs)
    ep = torch.export.export(model.eval(), inputs, dynamic_shapes=dynamic_shapes)
    if not any(node.target == torch.ops.aten.sym_size.int for node in ep.graph.nodes):
        raise RuntimeError(f"{prefix}: dynamic QKV fixture lost aten.sym_size.int")
    et = _lower_fully_delegated(ep, prefix)
    with open(os.path.join(out_dir, f"{prefix}.pte"), "wb") as f:
        f.write(et.buffer)
    print(f"Exported {prefix}.pte")
    for m in live_m:
        x = _bk64_input(m, BK64_K)
        case_inputs = (x,)
        if model.separate_v_input:
            case_inputs += (torch.flip(x, dims=[-1]).contiguous(),)
        outputs = (
            _bk64_golden(model.q, case_inputs[0]),
            _bk64_golden(model.k, case_inputs[0]),
            _bk64_golden(model.v, case_inputs[-1]),
        )
        base = os.path.join(out_dir, f"{prefix}.S{m}.")
        case_inputs[0].detach().numpy().astype("<f4").tofile(base + "input.bin")
        if model.separate_v_input:
            case_inputs[1].detach().numpy().astype("<f4").tofile(base + "v_input.bin")
        for name, output in zip(("q", "k", "v"), outputs):
            output.detach().numpy().astype("<f4").tofile(base + name + ".bin")
        print(f"  golden {prefix} M={m}")


def _export_dynamic_bk64_qkv_cases(out_dir: str) -> None:
    _export_dynamic_bk64_qkv_case(
        out_dir,
        "dyn_qkv_bk64",
        Bk64Qkv(),
        BK64_QKV_LIVE_M,
    )
    for prefix, model in (
        ("dyn_qkv_bk64_group32", Bk64Qkv(group=32)),
        ("dyn_qkv_bk64_bias", Bk64Qkv(bias=True)),
        (
            "dyn_qkv_bk64_wrong_width",
            Bk64Qkv(widths=(BK64_N, BK64_N, BK64_KV_N)),
        ),
        (
            "dyn_qkv_bk64_different_input",
            Bk64Qkv(separate_v_input=True),
        ),
    ):
        _export_dynamic_bk64_qkv_case(out_dir, prefix, model, [128])


def _export_static_linear(out_dir: str, m: int, prefix: str) -> None:
    from executorch.backends.webgpu.test.ops.test_quantized_linear import (
        _fp64_golden,
        _make_quantized_model,
    )

    model = _make_quantized_model(LIN_K, LIN_N, LIN_GROUP)
    x = _ramp((m, LIN_K))
    ep = torch.export.export(model, (x,))
    et = _lower_fully_delegated(ep, prefix)
    with open(os.path.join(out_dir, f"{prefix}.pte"), "wb") as f:
        f.write(et.buffer)
    x.detach().numpy().astype("<f4").tofile(
        os.path.join(out_dir, f"{prefix}.S{m}.input.bin")
    )
    _fp64_golden(model, x).astype("<f4").tofile(
        os.path.join(out_dir, f"{prefix}.S{m}.golden.bin")
    )
    print(f"Exported {prefix}.pte")


# Dynamic SDPA: GQA prefill (input_pos=0), q/k/v seq-len dynamic.
SD_HQ = 8
SD_HKV = 2
SD_D = 16
SD_CMAX = 64
SD_MAXS = 64

K16_HQ = 32
K16_HKV = 8
K16_D = 64
K16_CMAX = 525
K16_MAXS = 512
K16_INPUT_POS = 13
K16_PRIME_POS = 1
K16_PRIME_S = K16_INPUT_POS - K16_PRIME_POS
K16_POS_CONTROL = K16_INPUT_POS
K16_LIVE_S = (K16_MAXS, 508, 128, 127, 16, 1)


def _export_dynamic_sdpa(out_dir: str) -> None:
    from executorch.backends.webgpu.test.ops.test_sdpa import (
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
    et = _lower_fully_delegated(ep, "sdpa_dyn")
    with open(os.path.join(out_dir, "sdpa_dyn.pte"), "wb") as f:
        f.write(et.buffer)
    print("Exported sdpa_dyn.pte")
    for s in [SD_MAXS, 16, 1]:
        c = cfg(s)
        q, k, v, kc, vc = _det_inputs(c)
        g = _golden(c, q, k, v, kc, vc)
        for name, t in [
            ("q", q),
            ("k", k),
            ("v", v),
            ("kc", kc),
            ("vc", vc),
            ("golden", g),
        ]:
            t.detach().cpu().numpy().astype("<f4").tofile(
                os.path.join(out_dir, f"sdpa_dyn.S{s}.{name}.bin")
            )
        print(f"  golden sdpa_dyn S={s} (golden shape {tuple(g.shape)})")


def _export_dynamic_k16_sdpa_case(
    out_dir: str,
    prefix: str,
    hq: int,
    hkv: int,
    live_s,
    d: int = K16_D,
    scale: float | None = None,
) -> None:
    from backends.webgpu.test.ops.test_sdpa import (
        _det_inputs,
        _golden,
        SdpaConfig,
    )

    def cfg(s: int) -> "SdpaConfig":
        return SdpaConfig(
            prefix,
            hq,
            hkv,
            d,
            s,
            K16_CMAX,
            K16_INPUT_POS,
        )

    q, k, v, kc, vc = _det_inputs(cfg(K16_MAXS))
    kc.zero_()
    vc.zero_()

    class K16SdpaModule(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.register_buffer("k_cache", kc)
            self.register_buffer("v_cache", vc)

        def forward(self, q, k, v, pos_control):
            input_pos = pos_control.shape[1]
            return torch.ops.llama.sdpa_with_kv_cache(
                q,
                k,
                v,
                self.k_cache,
                self.v_cache,
                input_pos,
                q.shape[1],
                None,
                0.0,
                True,
                scale,
            )

    model = K16SdpaModule().eval()
    inputs = (q, k, v, torch.zeros(1, K16_POS_CONTROL))
    s_dim = torch.export.Dim(f"{prefix}_s", min=1, max=K16_MAXS)
    pos_dim = torch.export.Dim(f"{prefix}_pos_control", min=1, max=K16_POS_CONTROL)
    ep = torch.export.export(
        model,
        inputs,
        dynamic_shapes=({1: s_dim}, {1: s_dim}, {1: s_dim}, {1: pos_dim}),
    )
    sym_sizes = [
        node for node in ep.graph.nodes if node.target == torch.ops.aten.sym_size.int
    ]
    if len(sym_sizes) < 2:
        raise RuntimeError(f"{prefix}: dynamic K16 fixture lost S/position symbols")
    et = _lower_fully_delegated(ep, prefix)
    with open(os.path.join(out_dir, f"{prefix}.pte"), "wb") as f:
        f.write(et.buffer)
    print(f"Exported {prefix}.pte")

    def reference(live_cfg, q, k, v, kc, vc):
        if scale is None:
            return _golden(live_cfg, q, k, v, kc, vc)
        context_len = live_cfg.s + live_cfg.input_pos
        g = hq // hkv
        k_full = torch.cat((kc[0, : live_cfg.input_pos].double(), k[0].double()), dim=0)
        v_full = torch.cat((vc[0, : live_cfg.input_pos].double(), v[0].double()), dim=0)
        q_heads = q[0].double().transpose(0, 1)
        k_heads = k_full.repeat_interleave(g, dim=1).transpose(0, 1)
        v_heads = v_full.repeat_interleave(g, dim=1).transpose(0, 1)
        mask = torch.full((live_cfg.s, context_len), float("-inf"), dtype=torch.float64)
        for token in range(live_cfg.s):
            mask[token, : live_cfg.input_pos + token + 1] = 0.0
        golden = torch.nn.functional.scaled_dot_product_attention(
            q_heads, k_heads, v_heads, attn_mask=mask, scale=scale
        )
        return golden.transpose(0, 1).reshape(1, live_cfg.s, hq, d).float().contiguous()

    prime_cfg = SdpaConfig(prefix, hq, hkv, d, K16_PRIME_S, K16_CMAX, K16_PRIME_POS)
    prime_q, prime_k, prime_v, prime_kc, prime_vc = _det_inputs(prime_cfg)
    prime_kc.zero_()
    prime_vc.zero_()
    prime_golden = reference(prime_cfg, prime_q, prime_k, prime_v, prime_kc, prime_vc)
    prime_base = os.path.join(out_dir, f"{prefix}.prime.")
    for name, tensor in (
        ("q", prime_q),
        ("k", prime_k),
        ("v", prime_v),
        ("control", torch.zeros(1, 1)),
        ("golden", prime_golden),
    ):
        tensor.detach().numpy().astype("<f4").tofile(prime_base + name + ".bin")

    for s in live_s:
        live_cfg = cfg(s)
        q, k, v, kc, vc = _det_inputs(live_cfg)
        kc.zero_()
        vc.zero_()
        kc[0, K16_PRIME_POS:K16_INPUT_POS] = prime_k[0]
        vc[0, K16_PRIME_POS:K16_INPUT_POS] = prime_v[0]
        golden = reference(live_cfg, q, k, v, kc, vc)
        base = os.path.join(out_dir, f"{prefix}.S{s}.")
        for name, tensor in (
            ("q", q),
            ("k", k),
            ("v", v),
            ("control", torch.zeros(1, K16_POS_CONTROL)),
            ("kc", kc),
            ("vc", vc),
            ("golden", golden),
        ):
            tensor.detach().numpy().astype("<f4").tofile(base + name + ".bin")
        print(f"  golden {prefix} S={s}")


def _export_dynamic_k16_sdpa_cases(out_dir: str) -> None:
    _export_dynamic_k16_sdpa_case(
        out_dir,
        "sdpa_k16_llama",
        K16_HQ,
        K16_HKV,
        K16_LIVE_S,
    )
    _export_dynamic_k16_sdpa_case(
        out_dir,
        "sdpa_k16_wrong_geometry",
        14,
        2,
        (128, 1),
    )
    _export_dynamic_k16_sdpa_case(
        out_dir,
        "sdpa_k16_wrong_d",
        K16_HQ,
        K16_HKV,
        (128, 1),
        d=128,
        scale=0.125,
    )
    _export_dynamic_k16_sdpa_case(
        out_dir,
        "sdpa_k16_wrong_scale",
        K16_HQ,
        K16_HKV,
        (128, 1),
        scale=0.25,
    )


def _export_combined_routes(out_dir: str) -> None:
    from executorch.backends.webgpu.test.ops.test_quantized_linear import (
        _make_quantized_model,
    )
    from executorch.backends.webgpu.test.ops.test_sdpa import (
        _det_inputs,
        _golden,
        SdpaConfig,
    )
    from executorch.extension.llm.custom_ops import custom_ops  # noqa: F401

    class CombinedRoutes(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.linear = _make_quantized_model(LIN_K, SD_HQ * SD_D, LIN_GROUP)

        def forward(self, x, q, k, v, k_cache, v_cache):
            projected = self.linear(x).reshape(1, x.shape[0], SD_HQ, SD_D)
            return torch.ops.llama.sdpa_with_kv_cache(
                q + projected,
                k,
                v,
                k_cache,
                v_cache,
                0,
                q.shape[1],
                None,
                0.0,
                True,
                None,
            )

    model = CombinedRoutes().eval()
    cfg = SdpaConfig("combined", SD_HQ, SD_HKV, SD_D, SD_MAXS, SD_CMAX, 0)
    q, k, v, kc, vc = _det_inputs(cfg)
    x = _ramp((SD_MAXS, LIN_K))
    s_dim = torch.export.Dim("s", min=1, max=SD_MAXS)
    ep = torch.export.export(
        model,
        (x, q, k, v, kc, vc),
        dynamic_shapes=(
            {0: s_dim},
            {1: s_dim},
            {1: s_dim},
            {1: s_dim},
            None,
            None,
        ),
    )
    et = _lower_fully_delegated(ep, "combined_routes")
    with open(os.path.join(out_dir, "combined_routes.pte"), "wb") as f:
        f.write(et.buffer)
    print("Exported combined_routes.pte")

    for s in [SD_MAXS, 16, 1]:
        live_cfg = SdpaConfig("combined", SD_HQ, SD_HKV, SD_D, s, SD_CMAX, 0)
        q, k, v, kc, vc = _det_inputs(live_cfg)
        x = _ramp((s, LIN_K))
        with torch.no_grad():
            projected = model.linear(x).reshape(1, s, SD_HQ, SD_D)
            golden = _golden(live_cfg, q + projected, k, v, kc, vc)
        base = os.path.join(out_dir, f"combined_routes.S{s}.")
        for name, tensor in (
            ("x", x),
            ("q", q),
            ("k", k),
            ("v", v),
            ("kc", kc),
            ("vc", vc),
            ("golden", golden),
        ):
            tensor.detach().numpy().astype("<f4").tofile(base + name + ".bin")
        print(f"  golden combined_routes S={s}")


def _export_dynamic_sdpa_wide(out_dir: str) -> None:
    from executorch.backends.webgpu.test.ops.test_sdpa import (
        _det_inputs,
        _golden,
        SdpaConfig,
        SdpaModule,
    )

    hq, hkv, d, cmax, max_s = 8, 2, 132, 16, 16

    def cfg(s):
        return SdpaConfig("wide", hq, hkv, d, s, cmax, 0)

    model = SdpaModule(0)
    q, k, v, kc, vc = _det_inputs(cfg(max_s))
    s_dim = torch.export.Dim("s", min=1, max=max_s)
    ep = torch.export.export(
        model,
        (q, k, v, kc, vc),
        dynamic_shapes=({1: s_dim}, {1: s_dim}, {1: s_dim}, None, None),
    )
    et = _lower_fully_delegated(ep, "sdpa_wide")
    with open(os.path.join(out_dir, "sdpa_wide.pte"), "wb") as f:
        f.write(et.buffer)
    for s in [max_s, 1]:
        live = cfg(s)
        q, k, v, kc, vc = _det_inputs(live)
        golden = _golden(live, q, k, v, kc, vc)
        for name, tensor in (
            ("q", q),
            ("k", k),
            ("v", v),
            ("kc", kc),
            ("vc", vc),
            ("golden", golden),
        ):
            tensor.detach().numpy().astype("<f4").tofile(
                os.path.join(out_dir, f"sdpa_wide.S{s}.{name}.bin")
            )
    print("Exported sdpa_wide.pte")


def _export_static_sdpa(out_dir: str, s: int, prefix: str) -> None:
    from executorch.backends.webgpu.test.ops.test_sdpa import (
        _det_inputs,
        _golden,
        SdpaConfig,
        SdpaModule,
    )

    cfg = SdpaConfig(prefix, SD_HQ, SD_HKV, SD_D, s, SD_CMAX, 0)
    inputs = _det_inputs(cfg)
    ep = torch.export.export(SdpaModule(0), inputs)
    et = _lower_fully_delegated(ep, prefix)
    with open(os.path.join(out_dir, f"{prefix}.pte"), "wb") as f:
        f.write(et.buffer)
    golden = _golden(cfg, *inputs)
    for name, tensor in zip(("q", "k", "v", "kc", "vc"), inputs):
        tensor.detach().numpy().astype("<f4").tofile(
            os.path.join(out_dir, f"{prefix}.S{s}.{name}.bin")
        )
    golden.detach().numpy().astype("<f4").tofile(
        os.path.join(out_dir, f"{prefix}.S{s}.golden.bin")
    )
    print(f"Exported {prefix}.pte")


# Dynamic embedding: int64 token ids (dynamic count) -> [N, EMBED] fp32.
EMB_VOCAB = 64
EMB_DIM = 64
EMB_GROUP = 32
EMB_MAXN = 16


def _export_dynamic_embedding(out_dir: str) -> None:
    from executorch.backends.webgpu.test.ops.test_embedding_q4gsw import (
        _make_quantized_model,
        _quant_params,
        Shape,
    )

    shape = Shape("dyn", EMB_VOCAB, EMB_DIM, EMB_GROUP, list(range(EMB_MAXN)))
    qm = _make_quantized_model(shape)
    idx_max = torch.arange(EMB_MAXN, dtype=torch.long)
    n_dim = torch.export.Dim("n", min=1, max=EMB_MAXN)
    ep = torch.export.export(qm, (idx_max,), dynamic_shapes=({0: n_dim},))
    et = _lower_fully_delegated(ep, "emb_dyn")
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
    from executorch.backends.webgpu.test.ops.test_rope import (
        _golden,
        _inputs,
        Shape,
    )
    from executorch.examples.models.llama.rope import RotaryEmbedding

    xq, xk, fc, fs = _inputs(Shape("dyn", 1, ROPE_MAXS, ROPE_NH, ROPE_NKV, ROPE_HD))
    s_dim = torch.export.Dim("s", min=1, max=ROPE_MAXS)
    ds = ({1: s_dim}, {1: s_dim}, {0: s_dim}, {0: s_dim})
    ep = torch.export.export(
        RotaryEmbedding().eval(), (xq, xk, fc, fs), dynamic_shapes=ds
    )
    et = _lower_fully_delegated(ep, "rope_dyn")
    with open(os.path.join(out_dir, "rope_dyn.pte"), "wb") as f:
        f.write(et.buffer)
    print("Exported rope_dyn.pte")
    for s in [ROPE_MAXS, 8, 1]:
        xq, xk, fc, fs = _inputs(Shape("dyn", 1, s, ROPE_NH, ROPE_NKV, ROPE_HD))
        gq, gk = _golden(xq, xk, fc, fs)
        for name, t in [
            ("xq", xq),
            ("xk", xk),
            ("fc", fc),
            ("fs", fs),
            ("gq", gq),
            ("gk", gk),
        ]:
            t.detach().cpu().numpy().astype("<f4").tofile(
                os.path.join(out_dir, f"rope_dyn.S{s}.{name}.bin")
            )
        print(f"  golden rope_dyn S={s}")


# Dynamic select_copy: input [2, 1, S, HIDDEN], select(0, -1) -> [1, S, HIDDEN].
SEL_LEAD = 2


def _export_dynamic_select(out_dir: str) -> None:
    model = SelectModule()
    s_dim = torch.export.Dim("s", min=1, max=MAXS)
    ep = torch.export.export(
        model,
        (_ramp((SEL_LEAD, 1, MAXS, HIDDEN)),),
        dynamic_shapes=({2: s_dim},),
    )
    et = _lower_fully_delegated(ep, "dyn_select")
    with open(os.path.join(out_dir, "dyn_select.pte"), "wb") as f:
        f.write(et.buffer)
    print("Exported dyn_select.pte")
    for s in [MAXS, 32, 1]:
        x = _ramp((SEL_LEAD, 1, s, HIDDEN))
        with torch.no_grad():
            g = model(x)
        x.detach().numpy().astype("<f4").tofile(
            os.path.join(out_dir, f"dyn_select.S{s}.input.bin")
        )
        g.detach().numpy().astype("<f4").tofile(
            os.path.join(out_dir, f"dyn_select.S{s}.golden.bin")
        )
        print(f"  golden dyn_select S={s} (shape {tuple(g.shape)})")


class TestDynamicShapeExport(unittest.TestCase):
    def test_export_dynamic_rms(self) -> None:
        import tempfile

        with tempfile.TemporaryDirectory() as d:
            export_dynamic_shape_cases(d)
            self.assertTrue(os.path.exists(os.path.join(d, "dyn_rms.pte")))
            self.assertTrue(os.path.exists(os.path.join(d, "dyn_rms.S1.golden.bin")))
            expected = [
                "dyn_linear_bk64.pte",
                "dyn_linear_bk64.S512.input.bin",
                "dyn_linear_bk64.S512.golden.bin",
                "dyn_linear_bk64.S511.input.bin",
                "dyn_linear_bk64.S511.golden.bin",
                "dyn_linear_bk64.S508.golden.bin",
                "dyn_linear_bk64.S128.golden.bin",
                "dyn_linear_bk64.S127.golden.bin",
                "dyn_linear_bk64.S1.golden.bin",
                "dyn_linear_bk64_gate.pte",
                "dyn_linear_bk64_gate.S512.input.bin",
                "dyn_linear_bk64_gate.S512.golden.bin",
                "dyn_linear_bk64_gate.S508.golden.bin",
                "dyn_linear_bk64_gate.S128.golden.bin",
                "dyn_linear_bk64_down.pte",
                "dyn_linear_bk64_down.S512.input.bin",
                "dyn_linear_bk64_down.S512.golden.bin",
                "dyn_linear_bk64_down.S508.golden.bin",
                "dyn_linear_bk64_down.S128.golden.bin",
                "dyn_linear_bk64_group32.pte",
                "dyn_linear_bk64_bias.pte",
                "dyn_linear_bk64_kv_shape.pte",
                "dyn_qkv_bk64.pte",
                "dyn_qkv_bk64.S512.input.bin",
                "dyn_qkv_bk64.S512.q.bin",
                "dyn_qkv_bk64.S512.k.bin",
                "dyn_qkv_bk64.S512.v.bin",
                "dyn_qkv_bk64.S511.input.bin",
                "dyn_qkv_bk64.S511.q.bin",
                "dyn_qkv_bk64.S511.k.bin",
                "dyn_qkv_bk64.S511.v.bin",
                "dyn_qkv_bk64.S508.q.bin",
                "dyn_qkv_bk64.S508.k.bin",
                "dyn_qkv_bk64.S508.v.bin",
                "dyn_qkv_bk64.S128.q.bin",
                "dyn_qkv_bk64.S127.q.bin",
                "dyn_qkv_bk64.S16.q.bin",
                "dyn_qkv_bk64.S2.q.bin",
                "dyn_qkv_bk64.S1.q.bin",
                "dyn_qkv_bk64_group32.pte",
                "dyn_qkv_bk64_bias.pte",
                "dyn_qkv_bk64_wrong_width.pte",
                "dyn_qkv_bk64_different_input.pte",
            ]
            for name in expected:
                with self.subTest(artifact=name):
                    self.assertGreater(os.path.getsize(os.path.join(d, name)), 0)
            for prefix, live_s in (
                ("sdpa_k16_llama", K16_LIVE_S),
                ("sdpa_k16_wrong_geometry", (128, 1)),
                ("sdpa_k16_wrong_d", (128, 1)),
                ("sdpa_k16_wrong_scale", (128, 1)),
            ):
                with self.subTest(artifact=f"{prefix}.pte"):
                    self.assertGreater(
                        os.path.getsize(os.path.join(d, f"{prefix}.pte")), 0
                    )
                for kind in ("q", "k", "v", "control", "golden"):
                    name = f"{prefix}.prime.{kind}.bin"
                    with self.subTest(artifact=name):
                        self.assertGreater(os.path.getsize(os.path.join(d, name)), 0)
                for s in live_s:
                    for kind in ("q", "k", "v", "control", "golden"):
                        name = f"{prefix}.S{s}.{kind}.bin"
                        with self.subTest(artifact=name):
                            self.assertGreater(
                                os.path.getsize(os.path.join(d, name)), 0
                            )


if __name__ == "__main__":
    unittest.main()
