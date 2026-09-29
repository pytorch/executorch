# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-unsafe

"""
Numerical-debugging tutorial for the ExecuTorch CoreML backend.

End-to-end use of the intermediate-output tap on the static-attention Llama
from `examples/apple/coreml/llama/`. Smoke test (random weights, tiny
ModelArgs — no checkpoint download required) producing two tables:

1. A delegation summary showing how many subgraphs ExecuTorch handed off
   to CoreML and which operators ran on which side.

2. An AOT-vs-runtime comparison of the tapped intermediate values, so you
   can see numerical drift between eager-PyTorch and the CoreML runtime
   at hand-picked points in the model.
"""

import math
import os
import platform
import sys
import tempfile
import types
import unittest

import pandas as pd
import torch
import torch.utils._pytree as pytree
from executorch.devtools.backend_debug import get_delegation_info
from executorch.devtools.intermediate_output_tap import (
    compare_aot_runtime_dataframe,
    FULL_TENSOR,
    select_all,
    select_any,
    select_by_module_path,
    select_by_op_type,
    STATS,
    strip_taps_,
    tap_intermediate_outputs_,
)
from executorch.exir import to_edge_transform_and_lower
from executorch.runtime import Runtime, Verification
from torch.export import export


def _assert_df_quality(
    test_case: unittest.TestCase,
    df: pd.DataFrame,
    sqnr_db_threshold: float = 30.0,
    mean_rtol: float = 5e-3,
    mean_atol: float = 1e-5,
) -> None:
    """For each row in the AOT-vs-runtime comparison DataFrame, assert quality.

    - FULL_TENSOR rows: SQNR (in dB) must exceed `sqnr_db_threshold`.
    - STATS rows: `aot_mean` ≈ `rt_mean` within `(mean_rtol, mean_atol)`.
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


def _macos_version() -> tuple[int, int]:
    """Return (major, minor) macOS version, or (0, 0) on non-macOS."""
    if sys.platform != "darwin":
        return (0, 0)
    release = platform.mac_ver()[0]
    if not release:
        return (0, 0)
    parts = release.split(".")
    major = int(parts[0]) if parts and parts[0].isdigit() else 0
    minor = int(parts[1]) if len(parts) > 1 and parts[1].isdigit() else 0
    return (major, minor)


# iOS18 CoreML models require macOS 15+ to execute at runtime.
_MACOS_SUPPORTS_IOS18_RUNTIME = _macos_version() >= (15, 0)


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


def _build_model():
    """Build a tiny static-attention Llama with random weights."""
    from executorch.examples.apple.coreml.llama.export_static_llm_coreml import (
        _transform_eager_model,
    )
    from executorch.examples.models.llama.llama_transformer import construct_transformer
    from executorch.examples.models.llama.model_args import ModelArgs

    args = ModelArgs(
        dim=64,
        n_layers=2,
        n_heads=4,
        n_kv_heads=2,
        vocab_size=128,
        hidden_dim=128,
        max_seq_len=64,
        max_context_len=64,
        generate_full_logits=True,
    )
    args.attention_type = "static_mha"
    args.attention_kwargs = {"decompose_sdpa_in_mha": True}

    model = construct_transformer(args)
    transform_args = types.SimpleNamespace(
        target_split_size=None,
        max_splits=8,
        embedding_quantize="",
        linear_quantize="c4w",
        no_graph_breaks=False,
    )
    model = _transform_eager_model(model, transform_args, torch.float16)
    return model, args


@unittest.skipIf(
    sys.platform != "darwin",
    "CoreML backend requires macOS / Apple silicon; skipping on this platform",
)
class CoreMLEndToEndTest(unittest.TestCase):
    def test_tap_compare_static_llama_coreml(self):
        import coremltools as ct
        from executorch.backends.apple.coreml.compiler import CoreMLBackend
        from executorch.backends.apple.coreml.partition import CoreMLPartitioner
        from executorch.examples.apple.coreml.llama.export_static_llm_coreml import (
            _create_example_inputs,
            remove_graph_break_,
        )

        # Step 1: Build and quantize the model.
        model, model_args = _build_model()

        # Step 2: Create example inputs and export.
        input_len = 8
        example_inputs, _ = _create_example_inputs(
            model_args,
            input_len=input_len,
            max_context_len=model_args.max_context_len,
            float_dtype=torch.float16,
        )
        with torch.no_grad():
            _ = model(*example_inputs)  # eager sanity check
        ep = export(model, example_inputs)

        # Step 3: Pick which intermediate values to tap.
        #
        # FULL_TENSOR for: embedding output, layer-1 wvs linear.
        # STATS for: output linear, all wq/wk linears, layer-0 wvs, and
        #            layer-1 RMSNorm output muls.
        #
        # Patterns use `*` between `layers.<i>.` and the inner module so they
        # match both the bare path (`layers.<i>.attention...`) and the wrapped
        # path (`layers.<i>.block.attention...`) introduced by
        # BlockWithGraphBreak at partition boundaries. Rules are tried in
        # order; first match wins per node.
        selector_full_tensor = select_any(
            select_by_op_type("aten.embedding.default"),
            select_all(
                select_by_op_type("aten.linear.default"),
                select_by_module_path("layers.1.*attention.wvs.*"),
            ),
        )
        selector_stats = select_any(
            select_all(
                select_by_op_type("aten.linear.default"),
                select_any(
                    select_by_module_path("output"),
                    select_by_module_path("*.attention.wqs.*"),
                    select_by_module_path("*.attention.wks.*"),
                    select_by_module_path("layers.0.*attention.wvs.*"),
                ),
            ),
            select_all(
                select_by_op_type("aten.mul.Tensor"),
                select_any(
                    select_by_module_path("layers.1.*attention_norm"),
                    select_by_module_path("layers.1.*attention_norm.*"),
                    select_by_module_path("layers.1.*ffn_norm"),
                    select_by_module_path("layers.1.*ffn_norm.*"),
                ),
            ),
        )

        ep_t, specs = tap_intermediate_outputs_(
            ep,
            rules=[
                (selector_full_tensor, FULL_TENSOR),
                (selector_stats, STATS),
            ],
        )
        self.assertGreater(len(specs), 0)

        # Step 4: AOT-side reference values.
        aot_out = ep_t.module()(*example_inputs)
        aot_flat, _ = pytree.tree_flatten(aot_out)

        # Step 5: Lower to CoreML, strip the taps, dump the delegation summary.
        coreml_partitioner = CoreMLPartitioner(
            compile_specs=CoreMLBackend.generate_compile_specs(
                minimum_deployment_target=ct.target.iOS18,
                compute_precision=ct.precision.FLOAT16,
                compute_unit=ct.ComputeUnit.CPU_AND_NE,
            ),
        )
        edge = to_edge_transform_and_lower(ep_t, partitioner=[coreml_partitioner])
        # Drop `executorch_utils::graph_break.Tensor` placeholders that
        # `_transform_eager_model` inserted to force partition boundaries —
        # they have no out-variant kernel and must not survive into runtime.
        remove_graph_break_(edge)
        strip_taps_(edge)

        delegation_info = get_delegation_info(edge.exported_program().graph_module)
        print(
            "\n=== Delegation summary "
            f"(num_delegated_subgraphs={delegation_info.num_delegated_subgraphs}) ==="
        )
        print(delegation_info.get_summary())

        # Step 6: Save the .pte and run it through the ExecuTorch runtime.
        et_program = edge.to_executorch()
        with tempfile.TemporaryDirectory() as temp_dir:
            pte_path = os.path.join(temp_dir, "model.pte")
            et_program.save(pte_path)

            # iOS18-targeted CoreML models require macOS 15+ at runtime; on
            # older macOS we exercise AOT lowering + .pte serialization only
            # and skip runtime execution and the AOT-vs-runtime comparison.
            if not _MACOS_SUPPORTS_IOS18_RUNTIME:
                self.skipTest(
                    "Skipping runtime portion: iOS18 CoreML models require "
                    f"macOS 15+, found macOS {platform.mac_ver()[0] or 'unknown'}"
                )

            rt = Runtime.get()
            program = rt.load_program(pte_path, verification=Verification.Minimal)
            method = program.load_method("forward")
            flat_inputs, _ = pytree.tree_flatten(example_inputs)
            rt_flat = list(method.execute(flat_inputs))

        # Step 7: Compare AOT vs runtime.
        df = compare_aot_runtime_dataframe(specs, aot_flat, rt_flat)
        _assert_df_quality(self, df)
        _print_df(df, f"{len(specs)} tap(s) — AOT vs CoreML runtime:")


if __name__ == "__main__":
    unittest.main()
