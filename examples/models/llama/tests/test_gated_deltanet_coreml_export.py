# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import time
import unittest

import torch
from torch import nn

from executorch.examples.models.llama.attention import ATTENTION_REGISTRY
from executorch.examples.models.llama.model_args import ModelArgs
from executorch.examples.models.llama.rope import Rope

from executorch.backends.xnnpack.partition.xnnpack_partitioner import (
    XnnpackPartitioner,
)
from executorch.exir import EdgeCompileConfig, to_edge_transform_and_lower
from executorch.extension.llm.export.partitioner_lib import get_coreml_partitioner
from executorch.extension.pybindings.portable_lib import (
    _load_for_executorch_from_buffer,
)


SEQUENCE_LENGTHS = [1, 2, 4, 8, 16, 32, 64, 256, 512, 604]

WARMUP_ITERS = 5
BENCH_ITERS = 20


class _AttentionExportWrapper(nn.Module):
    """Wraps AttentionGatedDeltaNet for torch.export with explicit arguments.

    Supplies dummy frequency tensors (unused by DeltaNet) and routes input_pos.
    """

    def __init__(self, attn: nn.Module) -> None:
        super().__init__()
        self.attn = attn
        self.register_buffer("_dummy_freq", torch.zeros(1, 1))

    def forward(self, x: torch.Tensor, input_pos: torch.Tensor) -> torch.Tensor:
        out, _ = self.attn(
            x, self._dummy_freq, self._dummy_freq, input_pos=input_pos
        )
        return out


class TestGatedDeltaNetCoreMLExport(unittest.TestCase):
    """Verify AttentionGatedDeltaNet exports for CoreML at various sequence lengths.

    Hyperparameters match the Qwen3.5 0.8B linear-attention layer config:
      dim=1024, linear_key_head_dim=128, linear_value_head_dim=128,
      linear_num_key_heads=16, linear_num_value_heads=16, conv_kernel=4.
    """

    DIM = 1024

    @staticmethod
    def _make_args(seq_len: int) -> ModelArgs:
        return ModelArgs(
            dim=1024,
            n_layers=1,
            n_heads=8,
            n_kv_heads=2,
            head_dim=256,
            hidden_dim=3584,
            norm_eps=1e-6,
            vocab_size=248320,
            max_seq_len=max(seq_len, 64),
            max_context_len=max(seq_len, 64),
            max_batch_size=1,
            use_kv_cache=True,
            linear_conv_kernel_dim=4,
            linear_key_head_dim=128,
            linear_value_head_dim=128,
            linear_num_key_heads=16,
            linear_num_value_heads=16,
        )

    def _build_and_export(self, seq_len: int):
        torch.manual_seed(0)
        args = self._make_args(seq_len)
        rope = Rope(args)
        attn = ATTENTION_REGISTRY["gated_deltanet"](args, 0, rope)
        attn.eval()

        wrapper = _AttentionExportWrapper(attn)
        example_inputs = (
            torch.randn(1, seq_len, args.dim),
            torch.tensor([0], dtype=torch.long),
        )

        exported = torch.export.export(wrapper, example_inputs, strict=False)
        out = exported.module()(*example_inputs)
        self.assertEqual(out.shape, (1, seq_len, args.dim))
        return exported

    def _lower_to_coreml(self, exported):
        partitioner = get_coreml_partitioner(
            coreml_ios=18,
            embedding_quantize=None,
            pt2e_quantize=None,
            coreml_quantize=None,
            coreml_compute_units="cpu_and_ne",
        )
        edge = to_edge_transform_and_lower(
            exported,
            partitioner=[partitioner],
        )
        return edge

    def _export_to_buffer(self, seq_len: int) -> bytes:
        exported = self._build_and_export(seq_len)
        edge = self._lower_to_coreml(exported)
        et_program = edge.to_executorch()
        return et_program.buffer

    def test_torch_export(self):
        """Verify torch.export succeeds at all target sequence lengths."""
        for seq_len in SEQUENCE_LENGTHS:
            with self.subTest(seq_len=seq_len):
                self._build_and_export(seq_len)

    def test_coreml_lower(self):
        """Verify CoreML lowering succeeds at all target sequence lengths."""
        for seq_len in SEQUENCE_LENGTHS:
            with self.subTest(seq_len=seq_len):
                exported = self._build_and_export(seq_len)
                edge = self._lower_to_coreml(exported)
                self.assertIsNotNone(edge)

    def test_coreml_benchmark(self):
        """Export, load via ET pybindings, and benchmark forward at each seq length."""
        results = []

        for seq_len in SEQUENCE_LENGTHS:
            with self.subTest(seq_len=seq_len):
                pte_buffer = self._export_to_buffer(seq_len)
                et_module = _load_for_executorch_from_buffer(pte_buffer)

                inputs = [
                    torch.randn(1, seq_len, self.DIM),
                    torch.tensor([0], dtype=torch.long),
                ]

                # Warmup
                for _ in range(WARMUP_ITERS):
                    et_module.forward(inputs)

                # Benchmark
                start = time.perf_counter()
                for _ in range(BENCH_ITERS):
                    et_module.forward(inputs)
                elapsed = time.perf_counter() - start

                avg_ms = (elapsed / BENCH_ITERS) * 1000.0
                results.append((seq_len, avg_ms))

        # Display results table
        print("\n")
        print("=" * 50)
        print("GatedDeltaNet CoreML Forward Benchmark")
        print(f"  Qwen3.5 0.8B hyperparams, batch_size=1")
        print(f"  warmup={WARMUP_ITERS}, iters={BENCH_ITERS}")
        print("=" * 50)
        print(f"{'seq_len':>10}  {'avg (ms)':>10}  {'throughput (tok/s)':>18}")
        print("-" * 50)
        for seq_len, avg_ms in results:
            toks_per_sec = seq_len / (avg_ms / 1000.0)
            print(f"{seq_len:>10}  {avg_ms:>10.3f}  {toks_per_sec:>18.1f}")
        print("=" * 50)


XNNPACK_SEQUENCE_LENGTHS = [1, 2, 4, 8, 16, 32, 64]
CONTEXT_LEN = 1024
CONTEXT_FILL_REPS = 3


class _MHAExportWrapper(nn.Module):
    """Wraps AttentionMHA for torch.export with precomputed RoPE freq tables."""

    def __init__(self, attn: nn.Module, rope: Rope) -> None:
        super().__init__()
        self.attn = attn
        self.register_buffer("_freqs_cos", rope.freqs_cos.clone())
        self.register_buffer("_freqs_sin", rope.freqs_sin.clone())

    def forward(self, x: torch.Tensor, input_pos: torch.Tensor) -> torch.Tensor:
        freqs_cos = self._freqs_cos[input_pos]
        freqs_sin = self._freqs_sin[input_pos]
        out, _ = self.attn(x, freqs_cos, freqs_sin, input_pos=input_pos)
        return out


def _export_and_lower_xnnpack(wrapper, example_inputs):
    exported = torch.export.export(wrapper, example_inputs, strict=False)
    edge = to_edge_transform_and_lower(
        exported,
        partitioner=[XnnpackPartitioner()],
        compile_config=EdgeCompileConfig(
            _check_ir_validity=True,
            _skip_dim_order=True,
        ),
    )
    return edge


def _benchmark_single(edge, inputs):
    pte_buffer = edge.to_executorch().buffer
    et_module = _load_for_executorch_from_buffer(pte_buffer)
    for _ in range(WARMUP_ITERS):
        et_module.forward(inputs)
    start = time.perf_counter()
    for _ in range(BENCH_ITERS):
        et_module.forward(inputs)
    elapsed = time.perf_counter() - start
    return (elapsed / BENCH_ITERS) * 1000.0


def _benchmark_context_fill(edge, seq_len, dim, context_len, pos_fn):
    """Run actual multi-step context fill, advancing input_pos each step.

    pos_fn(step, seq_len) returns the input_pos tensor for that step.
    """
    pte_buffer = edge.to_executorch().buffer
    et_module = _load_for_executorch_from_buffer(pte_buffer)
    num_steps = context_len // seq_len

    # Warmup: one full context fill
    for step in range(num_steps):
        inputs = [torch.randn(1, seq_len, dim), pos_fn(step, seq_len)]
        et_module.forward(inputs)

    # Benchmark
    start = time.perf_counter()
    for _ in range(CONTEXT_FILL_REPS):
        for step in range(num_steps):
            inputs = [torch.randn(1, seq_len, dim), pos_fn(step, seq_len)]
            et_module.forward(inputs)
    elapsed = time.perf_counter() - start
    return (elapsed / CONTEXT_FILL_REPS) * 1000.0


def _gdn_pos(step, seq_len):
    return torch.tensor([step * seq_len], dtype=torch.long)


def _mha_pos(step, seq_len):
    start = step * seq_len
    return torch.arange(start, start + seq_len, dtype=torch.long)


def _make_gdn_args(seq_len: int, max_ctx: int = 0, chunk_size=None) -> ModelArgs:
    max_ctx = max(max_ctx, seq_len, 64)
    return ModelArgs(
        dim=1024,
        n_layers=1,
        n_heads=8,
        n_kv_heads=2,
        head_dim=256,
        hidden_dim=3584,
        norm_eps=1e-6,
        vocab_size=248320,
        max_seq_len=max_ctx,
        max_context_len=max_ctx,
        max_batch_size=1,
        use_kv_cache=True,
        linear_conv_kernel_dim=4,
        linear_key_head_dim=128,
        linear_value_head_dim=128,
        linear_num_key_heads=16,
        linear_num_value_heads=16,
        deltanet_chunk_size=chunk_size,
    )


def _make_mha_args(seq_len: int, max_ctx: int = 0) -> ModelArgs:
    max_ctx = max(max_ctx, seq_len, 64)
    return ModelArgs(
        dim=1024,
        n_layers=1,
        n_heads=16,
        n_kv_heads=4,
        head_dim=64,
        hidden_dim=3584,
        norm_eps=1e-6,
        vocab_size=248320,
        max_seq_len=max_ctx,
        max_context_len=max_ctx,
        max_batch_size=1,
        use_kv_cache=True,
    )


def _build_gdn(seq_len, max_ctx=0, chunk_size=None):
    torch.manual_seed(0)
    args = _make_gdn_args(seq_len, max_ctx, chunk_size)
    rope = Rope(args)
    attn = ATTENTION_REGISTRY["gated_deltanet"](args, 0, rope)
    attn.eval()
    wrapper = _AttentionExportWrapper(attn)
    example_inputs = (
        torch.randn(1, seq_len, args.dim),
        torch.tensor([0], dtype=torch.long),
    )
    return wrapper, example_inputs


def _build_mha(seq_len, max_ctx=0):
    from executorch.examples.models.llama.source_transformation.custom_kv_cache import (
        replace_kv_cache_with_custom_kv_cache,
    )
    from executorch.examples.models.llama.source_transformation.sdpa import (
        replace_sdpa_with_custom_op,
    )

    torch.manual_seed(0)
    args = _make_mha_args(seq_len, max_ctx)
    rope = Rope(args)
    attn = ATTENTION_REGISTRY["mha"](args, 0, rope)
    attn.eval()
    for p in attn.parameters():
        p.requires_grad_(False)
    for b in attn.buffers():
        b.requires_grad_(False)
    wrapper = _MHAExportWrapper(attn, rope)
    example_inputs = (
        torch.randn(1, seq_len, args.dim),
        torch.arange(0, seq_len, dtype=torch.long),
    )
    return wrapper, example_inputs


def _print_table(title, subtitle, results, col2="avg (ms)"):
    print("\n")
    print("=" * 55)
    print(title)
    print(f"  {subtitle}")
    print("=" * 55)
    print(f"{'seq_len':>10}  {col2:>12}")
    print("-" * 30)
    for row in results:
        print(f"{row[0]:>10}  {row[1]:>12.3f}")
    print("=" * 55)


class TestGatedDeltaNetXnnpackBenchmark(unittest.TestCase):
    """Export AttentionGatedDeltaNet to XNNPACK and benchmark at various seq lengths."""

    DIM = 1024

    def test_xnnpack_sequential_benchmark(self):
        results = []
        for seq_len in XNNPACK_SEQUENCE_LENGTHS:
            wrapper, inputs = _build_gdn(seq_len)
            edge = _export_and_lower_xnnpack(wrapper, inputs)
            avg_ms = _benchmark_single(edge, list(inputs))
            results.append((seq_len, avg_ms))
        _print_table(
            "GatedDeltaNet XNNPACK — fp32",
            "Qwen3.5 0.8B hyperparams, batch_size=1",
            results,
        )

    def test_xnnpack_context_fill(self):
        results = []
        for seq_len in XNNPACK_SEQUENCE_LENGTHS:
            wrapper, inputs = _build_gdn(seq_len, max_ctx=CONTEXT_LEN)
            edge = _export_and_lower_xnnpack(wrapper, inputs)
            total_ms = _benchmark_context_fill(
                edge, seq_len, self.DIM, CONTEXT_LEN, _gdn_pos
            )
            results.append((seq_len, total_ms))
        _print_table(
            f"GatedDeltaNet XNNPACK — fill {CONTEXT_LEN} tokens (fp32)",
            f"Qwen3.5 0.8B hyperparams, batch_size=1, reps={CONTEXT_FILL_REPS}",
            results,
            col2="total (ms)",
        )

class TestMHAXnnpackBenchmark(unittest.TestCase):
    """Export Qwen3.5 0.8B MHA block to XNNPACK and benchmark."""

    DIM = 1024

    def test_xnnpack_mha_benchmark(self):
        results = []
        for seq_len in XNNPACK_SEQUENCE_LENGTHS:
            wrapper, inputs = _build_mha(seq_len)
            edge = _export_and_lower_xnnpack(wrapper, inputs)
            avg_ms = _benchmark_single(edge, list(inputs))
            results.append((seq_len, avg_ms))
        _print_table(
            "MHA XNNPACK — fp32 (custom SDPA + KV cache)",
            "Qwen3.5 0.8B hyperparams, batch_size=1",
            results,
        )

    def test_xnnpack_mha_context_fill(self):
        results = []
        for seq_len in XNNPACK_SEQUENCE_LENGTHS:
            wrapper, inputs = _build_mha(seq_len, max_ctx=CONTEXT_LEN)
            edge = _export_and_lower_xnnpack(wrapper, inputs)
            total_ms = _benchmark_context_fill(
                edge, seq_len, self.DIM, CONTEXT_LEN, _mha_pos
            )
            results.append((seq_len, total_ms))
        _print_table(
            f"MHA XNNPACK — fill {CONTEXT_LEN} tokens (fp32, custom ops)",
            f"Qwen3.5 0.8B hyperparams, batch_size=1, reps={CONTEXT_FILL_REPS}",
            results,
            col2="total (ms)",
        )

class TestXnnpackGraphs(unittest.TestCase):
    """Print before/after XNNPACK lowering graphs for GDN and MHA."""

    def _print_before_after(self, label, wrapper, example_inputs):
        exported = torch.export.export(wrapper, example_inputs, strict=False)
        print("\n")
        print("=" * 70)
        print(f"{label} — BEFORE XNNPACK lowering")
        print("=" * 70)
        print(exported.graph_module.graph)

        edge = to_edge_transform_and_lower(
            exported,
            partitioner=[XnnpackPartitioner()],
            compile_config=EdgeCompileConfig(
                _check_ir_validity=True,
                _skip_dim_order=True,
            ),
        )
        print("\n")
        print("=" * 70)
        print(f"{label} — AFTER XNNPACK lowering")
        print("=" * 70)
        print(edge.exported_program().graph_module.graph)

    def test_gdn_graph(self):
        wrapper, inputs = _build_gdn(1)
        self._print_before_after("GDN (seq_len=1)", wrapper, inputs)

    def test_mha_graph(self):
        wrapper, inputs = _build_mha(1)
        self._print_before_after("MHA (seq_len=1)", wrapper, inputs)


if __name__ == "__main__":
    unittest.main()
