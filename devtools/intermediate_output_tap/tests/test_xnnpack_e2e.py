# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-unsafe

"""
Numerical-debugging tutorial for the ExecuTorch XNNPACK backend.

Showcases four ways to use the intermediate-output tap:

1. `test_tap_compare_xnnpack` — the one-shot `tap_compare(...)` helper.
   Shortest path; recommended for most users.

2. `test_low_level_pipeline_xnnpack` — the manual pipeline using
   `tap_intermediate_outputs_` + `strip_taps_` + `compare_aot_runtime_dataframe`
   directly. Use this when you need to insert custom edge-program transforms
   (e.g., `remove_graph_break_` in the CoreML pipeline) between lowering and
   stripping, or when you want delegation summaries / introspection on the
   edge program.

3. `test_tap_compare_static_transformer` — a small static-attention
   transformer with RMSNorm. Demonstrates per-layer tap selection: `wo`
   (attention output projection) in layers 2/5/8, plus every RMSNorm op in
   those same layers, using `select_by_module_class` + multi-pattern
   `select_by_module_path`.

4. `test_tap_compare_quantized_conv_bn_xnnpack` — Conv-BN fusion + 8-bit
   static quantization via PT2E + XNNPACK. End-to-end demonstration that
   the tap mechanism survives the full PT2E quantization pipeline.
"""

import math
import os
import sys
import tempfile
import unittest

import pandas as pd
import torch
import torch.utils._pytree as pytree
from executorch.backends.xnnpack.partition.xnnpack_partitioner import XnnpackPartitioner
from executorch.backends.xnnpack.quantizer.xnnpack_quantizer import (
    get_symmetric_quantization_config,
    XNNPACKQuantizer,
)
from executorch.devtools.backend_debug import get_delegation_info
from executorch.devtools.intermediate_output_tap import (
    compare_aot_runtime_dataframe,
    FULL_TENSOR,
    select_all,
    select_by_module_class,
    select_by_module_path,
    select_by_op_type,
    STATS,
    strip_taps_,
    tap_compare,
    tap_intermediate_outputs_,
)
from executorch.exir import to_edge_transform_and_lower
from executorch.runtime import Runtime, Verification
from torch.export import export
from torchao.quantization.pt2e.quantize_pt2e import convert_pt2e, prepare_pt2e


class _MLP(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.l1 = torch.nn.Linear(8, 16)
        self.l2 = torch.nn.Linear(16, 4)

    def forward(self, x):
        return self.l2(self.l1(x).relu())


# ----------------------------------------------------------------------
# Tiny static-attention transformer used by the third test.
# ----------------------------------------------------------------------
class _RMSNorm(torch.nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.ones(dim))
        self.eps = eps

    def forward(self, x):
        norm = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
        return norm * self.weight


class _Attention(torch.nn.Module):
    def __init__(self, dim: int, n_heads: int):
        super().__init__()
        self.n_heads = n_heads
        self.head_dim = dim // n_heads
        self.wq = torch.nn.Linear(dim, dim, bias=False)
        self.wk = torch.nn.Linear(dim, dim, bias=False)
        self.wv = torch.nn.Linear(dim, dim, bias=False)
        self.wo = torch.nn.Linear(dim, dim, bias=False)

    def forward(self, x):
        b, t, d = x.shape
        q = self.wq(x).view(b, t, self.n_heads, self.head_dim).transpose(1, 2)
        k = self.wk(x).view(b, t, self.n_heads, self.head_dim).transpose(1, 2)
        v = self.wv(x).view(b, t, self.n_heads, self.head_dim).transpose(1, 2)
        scores = torch.matmul(q, k.transpose(-2, -1)) / (self.head_dim**0.5)
        attn = torch.softmax(scores, dim=-1)
        out = torch.matmul(attn, v).transpose(1, 2).contiguous().view(b, t, d)
        return self.wo(out)


class _FeedForward(torch.nn.Module):
    def __init__(self, dim: int, hidden: int):
        super().__init__()
        self.w1 = torch.nn.Linear(dim, hidden, bias=False)
        self.w2 = torch.nn.Linear(hidden, dim, bias=False)

    def forward(self, x):
        return self.w2(torch.nn.functional.relu(self.w1(x)))


class _Block(torch.nn.Module):
    def __init__(self, dim: int, n_heads: int, hidden: int):
        super().__init__()
        self.attn_norm = _RMSNorm(dim)
        self.attention = _Attention(dim, n_heads)
        self.ffn_norm = _RMSNorm(dim)
        self.feed_forward = _FeedForward(dim, hidden)

    def forward(self, x):
        x = x + self.attention(self.attn_norm(x))
        x = x + self.feed_forward(self.ffn_norm(x))
        return x


class _StaticTransformer(torch.nn.Module):
    def __init__(
        self,
        n_layers: int = 10,
        dim: int = 32,
        n_heads: int = 4,
        hidden: int = 64,
        vocab_size: int = 64,
    ):
        super().__init__()
        self.tok_embeddings = torch.nn.Embedding(vocab_size, dim)
        self.layers = torch.nn.ModuleList(
            [_Block(dim, n_heads, hidden) for _ in range(n_layers)]
        )
        self.norm = _RMSNorm(dim)
        self.output = torch.nn.Linear(dim, vocab_size, bias=False)

    def forward(self, tokens):
        x = self.tok_embeddings(tokens)
        for layer in self.layers:
            x = layer(x)
        return self.output(self.norm(x))


def _build_rules():
    """Two-rule selection: FULL_TENSOR on `l1`, STATS on `l2`."""
    return [
        (
            select_all(
                select_by_op_type("aten.linear.default"),
                select_by_module_path("l1"),
            ),
            FULL_TENSOR,
        ),
        (
            select_all(
                select_by_op_type("aten.linear.default"),
                select_by_module_path("l2"),
            ),
            STATS,
        ),
    ]


def _print_df(df: pd.DataFrame, header: str) -> None:
    with pd.option_context(
        "display.max_columns",
        None,
        "display.width",
        280,
        "display.max_colwidth",
        30,
        "display.float_format",
        "{:.4g}".format,
    ):
        print(f"\n{header}")
        print(df.to_string(index=False))


def _assert_df_quality(
    test_case: unittest.TestCase,
    df: pd.DataFrame,
    sqnr_db_threshold: float = 30.0,
    mean_rtol: float = 1e-3,
    mean_atol: float = 1e-5,
) -> None:
    """For each row in the AOT-vs-runtime comparison DataFrame, assert quality.

    - FULL_TENSOR rows: SQNR (signal-to-noise ratio in dB) must exceed
      `sqnr_db_threshold`. 30 dB ≈ signal energy ≥ 1000× noise energy, well
      above the i8 quantization noise floor (~48 dB best case).
    - STATS rows: the `mean` field of the reduction must agree between AOT
      and runtime within `(mean_rtol, mean_atol)` (math.isclose semantics).
    """
    for _, row in df.iterrows():
        if row["reducer_name"] == "FULL_TENSOR":
            test_case.assertGreater(
                row["sqnr_db"],
                sqnr_db_threshold,
                f"low SQNR for {row['node_name']}: {row['sqnr_db']:.2f} dB "
                f"(threshold {sqnr_db_threshold} dB)",
            )
        else:
            aot_mean = row.get("aot_mean")
            rt_mean = row.get("rt_mean")
            test_case.assertTrue(
                aot_mean is not None and rt_mean is not None,
                f"non-FULL_TENSOR row for {row['node_name']} missing "
                "aot_mean/rt_mean columns",
            )
            test_case.assertTrue(
                math.isclose(aot_mean, rt_mean, rel_tol=mean_rtol, abs_tol=mean_atol),
                f"mean mismatch for {row['node_name']}: "
                f"aot={aot_mean}, rt={rt_mean}",
            )


def _assert_df_shape(df: pd.DataFrame, specs) -> None:
    """Common shape assertions for both test paths."""
    assert len(specs) == 2, f"expected 2 taps, got {len(specs)}"
    assert {s.reducer_name for s in specs} == {"FULL_TENSOR", "STATS"}
    assert "sqnr_db" in df.columns, "expected FULL_TENSOR sqnr_db column"
    assert "aot_min" in df.columns, "expected STATS aot_min column"


class _ConvBnReluConv(torch.nn.Module):
    """Conv2d → BatchNorm2d → ReLU → Conv2d.

    The Conv+BN pair is folded by `prepare_pt2e` so the post-quantization
    graph contains only conv2d ops (no batch_norm).
    """

    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(3, 8, kernel_size=3, padding=1)
        self.bn1 = torch.nn.BatchNorm2d(8)
        self.relu = torch.nn.ReLU()
        self.conv2 = torch.nn.Conv2d(8, 4, kernel_size=3, padding=1)

    def forward(self, x):
        return self.conv2(self.relu(self.bn1(self.conv1(x))))


@unittest.skipIf(
    sys.platform.startswith("win"), "ExecuTorch runtime not available on Windows"
)
class XnnpackEndToEndTest(unittest.TestCase):
    # ------------------------------------------------------------------
    # Path 1: One-shot `tap_compare(...)`
    # ------------------------------------------------------------------
    def test_tap_compare_xnnpack(self):
        model = _MLP()
        example_inputs = (torch.randn(2, 8),)

        df, specs = tap_compare(
            model,
            example_inputs,
            partitioner=[XnnpackPartitioner()],
            rules=_build_rules(),
        )

        _assert_df_shape(df, specs)
        _assert_df_quality(self, df)
        _print_df(df, f"[tap_compare] {len(specs)} tap(s) — AOT vs XNNPACK runtime:")

    # ------------------------------------------------------------------
    # Path 2: Low-level pipeline (manual steps)
    # ------------------------------------------------------------------
    def test_low_level_pipeline_xnnpack(self):
        model = _MLP()
        example_inputs = (torch.randn(2, 8),)

        # Step 1: Export.
        ep = export(model, example_inputs, strict=True)

        # Step 2: Insert taps.
        ep_t, specs = tap_intermediate_outputs_(ep, _build_rules())

        # Step 3: Capture AOT-side reference values via the tapped EP.
        # `tap.Tensor`'s eager impl applies the reducer, so the flat outputs
        # already contain reduced values at the same positions the runtime
        # will use.
        aot_out = ep_t.module()(*example_inputs)
        aot_flat, _ = pytree.tree_flatten(aot_out)

        # Step 4: Lower to XNNPACK and strip the taps.
        edge = to_edge_transform_and_lower(ep_t, partitioner=[XnnpackPartitioner()])
        # (At this point you can run any custom edge transforms — e.g.,
        # `remove_graph_break_(edge)` — before stripping.)
        strip_taps_(edge)

        # Bonus: print the delegation summary so you can see what XNNPACK
        # took. This is one of the things the low-level path gives you that
        # `tap_compare` hides.
        delegation_info = get_delegation_info(edge.exported_program().graph_module)
        print(
            "\n[low-level] === Delegation summary "
            f"(num_delegated_subgraphs={delegation_info.num_delegated_subgraphs}) ==="
        )
        print(delegation_info.get_summary())

        # Step 5: Save the .pte and run it through the ExecuTorch runtime.
        et_program = edge.to_executorch()
        with tempfile.TemporaryDirectory() as temp_dir:
            pte_path = os.path.join(temp_dir, "model.pte")
            et_program.save(pte_path)

            rt = Runtime.get()
            program = rt.load_program(pte_path, verification=Verification.Minimal)
            method = program.load_method("forward")
            flat_inputs, _ = pytree.tree_flatten(example_inputs)
            rt_flat = list(method.execute(flat_inputs))

        # Step 6: Diff AOT vs runtime.
        df = compare_aot_runtime_dataframe(specs, aot_flat, rt_flat)

        _assert_df_shape(df, specs)
        _assert_df_quality(self, df)
        _print_df(df, f"[low-level] {len(specs)} tap(s) — AOT vs XNNPACK runtime:")

    # ------------------------------------------------------------------
    # Path 3: Per-layer selection on a small static transformer.
    #
    # Taps `attention.wo` and every op inside `_RMSNorm` in layers 2, 5, 8.
    # Demonstrates `select_by_module_class` and the multi-pattern form of
    # `select_by_module_path`.
    # ------------------------------------------------------------------
    def test_tap_compare_static_transformer(self):
        torch.manual_seed(0)
        model = _StaticTransformer(
            n_layers=10, dim=32, n_heads=4, hidden=64, vocab_size=64
        )
        example_inputs = (torch.randint(0, 64, (1, 8)),)

        target_layers = (2, 5, 8)
        layer_patterns = [f"layers.{i}.*" for i in target_layers]

        rules = [
            # `wo` (attention output projection) in the target layers.
            (
                select_all(
                    select_by_op_type("aten.linear.default"),
                    select_by_module_path(
                        *[f"layers.{i}.attention.wo" for i in target_layers]
                    ),
                ),
                FULL_TENSOR,
            ),
            # The terminal output of each `_RMSNorm` instance in the target
            # layers (one tap per RMSNorm instance — not every internal op).
            (
                select_all(
                    select_by_module_class("_RMSNorm", output_only=True),
                    select_by_module_path(*layer_patterns),
                ),
                STATS,
            ),
        ]

        df, specs = tap_compare(
            model,
            example_inputs,
            partitioner=[XnnpackPartitioner()],
            rules=rules,
        )

        # We expect exactly three `wo` taps (one per target layer).
        wo_specs = [s for s in specs if s.reducer_name == "FULL_TENSOR"]
        norm_specs = [s for s in specs if s.reducer_name == "STATS"]
        self.assertEqual(len(wo_specs), 3, f"got {len(wo_specs)} wo taps: {wo_specs}")
        # Every `wo` tap should live in one of the target layers.
        for s in wo_specs:
            self.assertTrue(
                any(
                    f"layers.{i}.attention.wo" in (s.module_path or "")
                    for i in target_layers
                ),
                f"unexpected wo module_path: {s.module_path}",
            )
        # Two RMSNorms per block (attn_norm + ffn_norm) × 3 target layers
        # = 6 RMSNorm output taps.
        self.assertEqual(
            len(norm_specs), 6, f"got {len(norm_specs)} RMSNorm taps: {norm_specs}"
        )

        _assert_df_quality(self, df)
        _print_df(
            df,
            f"[transformer] {len(specs)} tap(s) — AOT vs XNNPACK runtime "
            f"({len(wo_specs)} wo + {len(norm_specs)} RMSNorm outputs):",
        )

    # ------------------------------------------------------------------
    # Path 4: Conv-BN fusion + 8-bit PT2E static quantization on XNNPACK.
    #
    # Insert taps PRE-PT2E so the quantizer observes the tap points and
    # assigns them stable scales (vs. tapping post-PT2E, which exposes
    # intermediate values whose scales were never calibrated and breaks
    # XNNPACK's fused multi-conv kernel).
    # ------------------------------------------------------------------
    def test_tap_compare_quantized_conv_bn_xnnpack(self):
        torch.manual_seed(0)
        model = _ConvBnReluConv().eval()
        example_inputs = (torch.randn(2, 3, 8, 8),)

        # 1. Capture as an EP and tap the conv2d outputs BEFORE PT2E.
        ep_pre = torch.export.export(model, example_inputs, strict=True)
        rules = [(select_by_op_type("aten.conv2d.default"), FULL_TENSOR)]
        ep_pre_tapped, specs = tap_intermediate_outputs_(ep_pre, rules)
        self.assertEqual(len(specs), 2, f"expected 2 conv2d taps, got {len(specs)}")

        # 2. Run PT2E on the tapped GraphModule. The tap.Tensor nodes are
        # passive (identity at runtime); we just need the quantizer to
        # observe the tapped values and assign them stable scales.
        captured = ep_pre_tapped.module()
        quantizer = XNNPACKQuantizer().set_global(
            get_symmetric_quantization_config(is_qat=False)
        )
        prepared = prepare_pt2e(captured, quantizer)

        # 3. Calibrate.
        for _ in range(20):
            prepared(torch.randn(2, 3, 8, 8))

        # 4. Convert.
        quantized = convert_pt2e(prepared)

        # 5. Re-export the tapped+quantized graph.
        ep = torch.export.export(quantized, example_inputs, strict=True)

        # Sanity: BN should be folded into conv — no batch_norm op left.
        targets = {str(n.target) for n in ep.graph_module.graph.nodes}
        self.assertFalse(
            any("batch_norm" in t for t in targets),
            "Conv+BN should fuse during prepare_pt2e; remaining targets: "
            f"{sorted(targets)}",
        )

        # 6. AOT-side reference values.
        aot_out = ep.module()(*example_inputs)
        aot_flat, _ = pytree.tree_flatten(aot_out)

        # 7. Lower with XNNPACK and strip taps.
        edge = to_edge_transform_and_lower(ep, partitioner=[XnnpackPartitioner()])
        strip_taps_(edge)
        et_program = edge.to_executorch()

        # 8. Run on the ExecuTorch runtime.
        with tempfile.TemporaryDirectory() as temp_dir:
            pte_path = os.path.join(temp_dir, "model.pte")
            et_program.save(pte_path)
            rt = Runtime.get()
            program = rt.load_program(pte_path, verification=Verification.Minimal)
            method = program.load_method("forward")
            flat_inputs, _ = pytree.tree_flatten(example_inputs)
            rt_flat = list(method.execute(flat_inputs))

        # 9. Diff AOT vs runtime. Spec output_indices were assigned against
        # the pre-PT2E EP, but PT2E preserves output ordering, so they're
        # still valid post-quant.
        df = compare_aot_runtime_dataframe(specs, aot_flat, rt_flat)
        _assert_df_quality(self, df)
        _print_df(
            df,
            f"[quant-conv-bn] {len(specs)} tap(s) — AOT vs XNNPACK runtime:",
        )
