# Copyright 2026 Arm Limited and/or its affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.
"""Baseline benchmark for LLM-style SDPA decode on the standard Arm path.

This script is intentionally narrow. It focuses on decoder-style attention
shapes that are useful for comparing a future Arm ``tosa.CUSTOM`` SDPA path
against the existing decomposed SDPA path.

The baseline measured here is:
1. Eager PyTorch ``scaled_dot_product_attention`` with ``enable_gqa=True``.
2. The same exported graph after the Arm SDPA decomposition pass expands the
   operation into primitive ops.
3. The fused ``llama.custom_sdpa`` op wrapped with the same layout transposes
   used by the Arm custom-SDPA export path.

That makes this script useful before any custom Arm/VGF SDPA kernel exists,
while still exercising the same decomposition logic the standard TOSA path
relies on.

"""

from __future__ import annotations

import argparse
import copy
import statistics
import time

from collections import Counter
from dataclasses import dataclass
from typing import Iterable

import torch
from executorch.backends.arm._passes.decompose_sdpa_pass import (
    DecomposeScaledDotProductAttentionPass,
)
from torch.export import export


DEFAULT_KV_LENGTHS = (128, 512, 2048, 8192)


@dataclass(frozen=True)
class SdpaDecodeConfig:
    batch_size: int = 1
    query_heads: int = 32
    kv_heads: int = 8
    head_dim: int = 128
    query_len: int = 1
    dtype: torch.dtype = torch.float32


@dataclass(frozen=True)
class SdpaCase:
    name: str
    module: torch.nn.Module
    inputs: tuple[torch.Tensor, torch.Tensor, torch.Tensor]
    config: SdpaDecodeConfig
    kv_len: int


@dataclass(frozen=True)
class BenchmarkSummary:
    name: str
    median_ms: float
    mean_ms: float
    stdev_ms: float


def _get_custom_sdpa_implementation() -> str:
    try:
        from executorch.extension.llm.custom_ops import custom_ops  # noqa: F401
    except (AssertionError, ImportError, OSError, RuntimeError):
        return "python"
    return "native"


def _build_start_pos_causal_mask(
    query: torch.Tensor,
    key: torch.Tensor,
    start_pos: int,
) -> torch.Tensor:
    query_len = query.size(2)
    key_len = key.size(2)
    query_positions = start_pos + torch.arange(query_len, device=query.device)
    key_positions = torch.arange(key_len, device=key.device)
    return torch.where(
        key_positions.unsqueeze(0) <= query_positions.unsqueeze(1),
        torch.zeros(
            (query_len, key_len),
            dtype=torch.float32,
            device=query.device,
        ),
        torch.full(
            (query_len, key_len),
            float("-inf"),
            dtype=torch.float32,
            device=query.device,
        ),
    )


def _custom_sdpa_python_impl(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    start_pos: int,
    attn_mask: torch.Tensor | None = None,
    drpout_p: float = 0.0,
    is_causal: bool = False,
    scale: float | None = None,
) -> torch.Tensor:
    query = query.transpose(1, 2)
    key = key.transpose(1, 2)
    value = value.transpose(1, 2)

    num_heads_q = query.size(1)
    num_heads_kv = key.size(1)
    if num_heads_q != num_heads_kv:
        assert num_heads_q % num_heads_kv == 0
        repeats = num_heads_q // num_heads_kv
        key = key.repeat_interleave(repeats, dim=1)
        value = value.repeat_interleave(repeats, dim=1)

    if attn_mask is None and is_causal:
        attn_mask = _build_start_pos_causal_mask(query, key, start_pos)
        is_causal = False

    output = torch.nn.functional.scaled_dot_product_attention(
        query,
        key,
        value,
        attn_mask=attn_mask,
        dropout_p=drpout_p,
        is_causal=is_causal,
        scale=scale,
    )
    return output.transpose(1, 2).contiguous()


class DecodeSdpaModule(torch.nn.Module):
    def forward(
        self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor
    ) -> torch.Tensor:
        return torch.nn.functional.scaled_dot_product_attention(
            q,
            k,
            v,
            attn_mask=None,
            dropout_p=0.0,
            is_causal=False,
            enable_gqa=True,
        )


class DecodeCustomSdpaModule(torch.nn.Module):
    def __init__(self, implementation: str) -> None:
        super().__init__()
        self._implementation = implementation

    def forward(
        self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor
    ) -> torch.Tensor:
        query = q.permute(0, 2, 1, 3).contiguous()
        key_cache = k.permute(0, 2, 1, 3).contiguous()
        value_cache = v.permute(0, 2, 1, 3).contiguous()
        start_pos = key_cache.shape[1] - query.shape[1]
        if self._implementation == "native":
            output = torch.ops.llama.custom_sdpa.default(
                query,
                key_cache,
                value_cache,
                start_pos,
                None,
                0.0,
                True,
                None,
            )
        else:
            output = _custom_sdpa_python_impl(
                query,
                key_cache,
                value_cache,
                start_pos,
                None,
                0.0,
                True,
                None,
            )
        return output.permute(0, 2, 1, 3)


def build_decode_case(
    kv_len: int,
    config: SdpaDecodeConfig,
    seed: int = 0,
) -> SdpaCase:
    if config.query_len != 1:
        raise ValueError("This benchmark only supports single-token decode cases")
    if config.query_heads % config.kv_heads != 0:
        raise ValueError("query_heads must be divisible by kv_heads for GQA")

    generator = torch.Generator().manual_seed(seed + kv_len)
    shape_q = (
        config.batch_size,
        config.query_heads,
        config.query_len,
        config.head_dim,
    )
    shape_kv = (
        config.batch_size,
        config.kv_heads,
        kv_len,
        config.head_dim,
    )
    q = torch.randn(shape_q, generator=generator, dtype=config.dtype)
    k = torch.randn(shape_kv, generator=generator, dtype=config.dtype)
    v = torch.randn(shape_kv, generator=generator, dtype=config.dtype)
    return SdpaCase(
        name=(
            f"b{config.batch_size}_hq{config.query_heads}_hkv{config.kv_heads}"
            f"_d{config.head_dim}_q{config.query_len}_kv{kv_len}"
        ),
        module=DecodeSdpaModule().eval(),
        inputs=(q, k, v),
        config=config,
        kv_len=kv_len,
    )


def _time_callable(
    fn,
    args: tuple[torch.Tensor, ...],
    warmup_runs: int,
    benchmark_runs: int,
) -> BenchmarkSummary:
    for _ in range(warmup_runs):
        fn(*copy.deepcopy(args))

    samples_ms: list[float] = []
    for _ in range(benchmark_runs):
        bench_args = copy.deepcopy(args)
        start = time.perf_counter()
        fn(*bench_args)
        end = time.perf_counter()
        samples_ms.append((end - start) * 1000.0)

    return BenchmarkSummary(
        name=getattr(fn, "__name__", type(fn).__name__),
        median_ms=statistics.median(samples_ms),
        mean_ms=statistics.fmean(samples_ms),
        stdev_ms=statistics.pstdev(samples_ms) if len(samples_ms) > 1 else 0.0,
    )


def export_and_decompose(
    case: SdpaCase,
) -> tuple[torch.fx.GraphModule, Counter[str]]:
    graph_module = export(case.module, case.inputs, strict=True).module()
    DecomposeScaledDotProductAttentionPass()(graph_module)
    op_counts = summarize_graph_ops(graph_module)
    return graph_module, op_counts


def build_custom_sdpa_module() -> torch.nn.Module:
    implementation = _get_custom_sdpa_implementation()
    return DecodeCustomSdpaModule(implementation).eval()


def summarize_graph_ops(graph_module: torch.fx.GraphModule) -> Counter[str]:
    return Counter(
        str(node.target)
        for node in graph_module.graph.nodes
        if node.op == "call_function"
    )


def check_case_correctness(
    case: SdpaCase,
    atol: float = 1e-5,
    rtol: float = 1e-5,
) -> None:
    eager_output = case.module(*copy.deepcopy(case.inputs))
    decomposed_module, _ = export_and_decompose(case)
    decomposed_output = decomposed_module(*copy.deepcopy(case.inputs))
    custom_output = build_custom_sdpa_module()(*copy.deepcopy(case.inputs))
    torch.testing.assert_close(decomposed_output, eager_output, atol=atol, rtol=rtol)
    torch.testing.assert_close(custom_output, eager_output, atol=atol, rtol=rtol)


def format_results(
    case: SdpaCase,
    eager: BenchmarkSummary,
    decomposed: BenchmarkSummary,
    custom: BenchmarkSummary,
    op_counts: Counter[str],
    custom_impl: str,
) -> str:
    decomposed_ratio = (
        eager.median_ms / decomposed.median_ms if decomposed.median_ms else float("inf")
    )
    custom_ratio = (
        eager.median_ms / custom.median_ms if custom.median_ms else float("inf")
    )
    top_ops = ", ".join(f"{op}:{count}" for op, count in op_counts.most_common(6))
    return (
        f"{case.name}\n"
        f"  eager median:      {eager.median_ms:8.3f} ms\n"
        f"  decomposed median: {decomposed.median_ms:8.3f} ms\n"
        f"  custom median:     {custom.median_ms:8.3f} ms\n"
        f"  eager mean:        {eager.mean_ms:8.3f} ms\n"
        f"  decomposed mean:   {decomposed.mean_ms:8.3f} ms\n"
        f"  custom mean:       {custom.mean_ms:8.3f} ms\n"
        f"  eager/decomp:      {decomposed_ratio:8.3f}x\n"
        f"  eager/custom:      {custom_ratio:8.3f}x\n"
        f"  custom impl:       {custom_impl}\n"
        f"  top decomposed ops: {top_ops}\n"
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Baseline benchmark for the standard Arm SDPA decomposition path."
    )
    parser.add_argument(
        "--kv-lens",
        nargs="+",
        type=int,
        default=list(DEFAULT_KV_LENGTHS),
        help="Active KV lengths to benchmark.",
    )
    parser.add_argument("--warmup-runs", type=int, default=5)
    parser.add_argument("--benchmark-runs", type=int, default=20)
    parser.add_argument(
        "--skip-correctness",
        action="store_true",
        help="Skip eager-vs-decomposed correctness checks.",
    )
    return parser.parse_args()


def iter_cases(kv_lens: Iterable[int]) -> Iterable[SdpaCase]:
    config = SdpaDecodeConfig()
    for kv_len in kv_lens:
        yield build_decode_case(kv_len, config)


def main() -> None:
    args = parse_args()
    torch.manual_seed(0)

    for case in iter_cases(args.kv_lens):
        if not args.skip_correctness:
            check_case_correctness(case)

        decomposed_module, op_counts = export_and_decompose(case)
        custom_impl = _get_custom_sdpa_implementation()
        custom_module = build_custom_sdpa_module()
        eager_stats = _time_callable(
            case.module,
            case.inputs,
            warmup_runs=args.warmup_runs,
            benchmark_runs=args.benchmark_runs,
        )
        decomposed_stats = _time_callable(
            decomposed_module,
            case.inputs,
            warmup_runs=args.warmup_runs,
            benchmark_runs=args.benchmark_runs,
        )
        custom_stats = _time_callable(
            custom_module,
            case.inputs,
            warmup_runs=args.warmup_runs,
            benchmark_runs=args.benchmark_runs,
        )
        print(
            format_results(
                case,
                eager_stats,
                decomposed_stats,
                custom_stats,
                op_counts,
                custom_impl,
            )
        )


if __name__ == "__main__":
    main()
