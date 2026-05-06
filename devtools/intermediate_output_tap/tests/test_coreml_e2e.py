# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-unsafe

"""
Numerical-debugging tutorial for the ExecuTorch CoreML backend.

This script walks through end-to-end use of the intermediate-output tap
infrastructure on the static-attention Llama from
`examples/apple/coreml/llama/`. It is a smoke test (random weights, tiny
ModelArgs — no checkpoint download required) and produces two tables:

1. A delegation summary showing how many subgraphs ExecuTorch handed off
   to CoreML and which operators ran on which side.

2. An AOT-vs-runtime comparison of the tapped intermediate values, so you
   can see numerical drift between eager-PyTorch and the CoreML runtime
   at hand-picked points in the model.

Run with:
    python swift_play/test_inspector_coreml.py
"""

import os
import tempfile

import coremltools as ct
import pandas as pd
import torch
import torch.utils._pytree as pytree
from executorch.backends.apple.coreml.compiler import CoreMLBackend
from executorch.backends.apple.coreml.partition import CoreMLPartitioner
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
    tap_intermediate_outputs,
)
import types

from executorch.examples.apple.coreml.llama.export_static_llm_coreml import (
    _create_example_inputs,
    _transform_eager_model,
    remove_graph_break_,
)
from executorch.examples.models.llama.llama_transformer import construct_transformer
from executorch.examples.models.llama.model_args import ModelArgs
from executorch.exir import to_edge_transform_and_lower
from executorch.runtime import Runtime, Verification
from torch.export import export


def _build_model() -> tuple[torch.nn.Module, ModelArgs]:
    """Build a tiny static-attention Llama with random weights."""
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


def main() -> None:
    # ------------------------------------------------------------------
    # Step 1: Build and quantize the model.
    # ------------------------------------------------------------------
    model, model_args = _build_model()

    # ------------------------------------------------------------------
    # Step 2: Create example inputs and export.
    # ------------------------------------------------------------------
    input_len = 8
    example_inputs, cache_len = _create_example_inputs(
        model_args,
        input_len=input_len,
        max_context_len=model_args.max_context_len,
        float_dtype=torch.float16,
    )
    print(f"input_len={input_len} cache_len={cache_len}")

    with torch.no_grad():
        _ = model(*example_inputs)  # eager sanity check

    print("Exporting...")
    ep = export(model, example_inputs)

    # ------------------------------------------------------------------
    # Step 3: Pick which intermediate values to tap.
    #
    # We use two reducers in two passes:
    #
    #   * FULL_TENSOR for `layers.1.attention.wvs.0` — surfaces the raw
    #     activation tensor; the comparison DataFrame computes SQNR over
    #     all elements.
    #
    #   * STATS for everything else (`output`, all wqs/wks linears, layer 0's
    #     wvs, and layer 1's RMSNorm output mul) — gives a rich set of
    #     debugging scalars (min/max/mean/std/rms/l1/l2/abs_max/abs_mean/
    #     nan_count/inf_count/zero_count/p99_abs).
    # ------------------------------------------------------------------
    # Patterns use `*` between `layers.<i>.` and the inner module so they match
    # both the bare path (`layers.<i>.attention...`) and the wrapped path
    # (`layers.<i>.block.attention...`) that BlockWithGraphBreak introduces
    # at the partition boundaries.
    selector_full_tensor = select_any(
        # Token-embedding output (one big tensor, before any transformer block).
        select_by_op_type("aten.embedding.default"),
        # First wvs linear in layer 1 — captures full activation post-Q/K/V.
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

    ep_t, specs_full = tap_intermediate_outputs(
        ep, selector=selector_full_tensor, reducer=FULL_TENSOR
    )
    ep_t, specs_stats = tap_intermediate_outputs(
        ep_t, selector=selector_stats, reducer=STATS
    )
    specs = list(specs_full) + list(specs_stats)
    print(
        f"Inserted {len(specs)} tap(s) "
        f"({len(specs_full)} FULL_TENSOR + {len(specs_stats)} STATS)."
    )

    # ------------------------------------------------------------------
    # Step 4: Capture the AOT-side reference values.
    #
    # `tap.Tensor`'s eager impl applies the reducer, so the flat outputs of
    # the tapped EP already contain reduced values at the same positions
    # the runtime will use. We pytree-flatten because the static-llama
    # forward returns nested (logits, (k_caches, v_caches)).
    # ------------------------------------------------------------------
    aot_out = ep_t.module()(*example_inputs)
    aot_flat, _ = pytree.tree_flatten(aot_out)

    # ------------------------------------------------------------------
    # Step 5: Lower to CoreML, strip the taps, and show what got delegated.
    # ------------------------------------------------------------------
    coreml_partitioner = CoreMLPartitioner(
        compile_specs=CoreMLBackend.generate_compile_specs(
            minimum_deployment_target=ct.target.iOS18,
            compute_precision=ct.precision.FLOAT16,
            compute_unit=ct.ComputeUnit.CPU_AND_NE,
        ),
    )
    print("Lowering to CoreML...")
    edge = to_edge_transform_and_lower(ep_t, partitioner=[coreml_partitioner])
    # Drop the `executorch_utils::graph_break.Tensor` placeholders that
    # `_transform_eager_model` inserted to force partition boundaries — they
    # have no out-variant kernel, so they must not survive into the runtime
    # program.
    remove_graph_break_(edge)
    specs = strip_taps_(edge, tap_specs=specs)

    delegation_info = get_delegation_info(edge.exported_program().graph_module)
    print(
        f"\n=== Delegation summary "
        f"(num_delegated_subgraphs={delegation_info.num_delegated_subgraphs}) ==="
    )
    print(delegation_info.get_summary())
    with pd.option_context(
        "display.max_columns", None,
        "display.width", 240,
        "display.max_colwidth", 60,
    ):
        print(
            delegation_info.get_operator_delegation_dataframe().to_string(index=False)
        )

    # ------------------------------------------------------------------
    # Step 6: Save the .pte and run it through the ExecuTorch runtime.
    # ------------------------------------------------------------------
    et_program = edge.to_executorch()
    with tempfile.TemporaryDirectory() as temp_dir:
        pte_path = os.path.join(temp_dir, "model.pte")
        et_program.save(pte_path)
        print(f"\nSaved PTE: {pte_path} ({os.path.getsize(pte_path)} bytes)")

        rt = Runtime.get()
        program = rt.load_program(pte_path, verification=Verification.Minimal)
        method = program.load_method("forward")
        # Runtime takes a flat tensor list — flatten the (tokens, options_dict)
        # pytree the same way torch.export did.
        flat_inputs, _ = pytree.tree_flatten(example_inputs)
        rt_flat = list(method.execute(flat_inputs))

    # ------------------------------------------------------------------
    # Step 7: Compare AOT vs runtime.
    # ------------------------------------------------------------------
    df = compare_aot_runtime_dataframe(specs, aot_flat, rt_flat)
    with pd.option_context(
        "display.max_columns", None,
        "display.width", 280,
        "display.max_colwidth", 30,
        "display.float_format", "{:.4g}".format,
    ):
        print(f"\n{len(specs)} tap(s) — AOT vs CoreML runtime:")
        print(df.to_string(index=False))


if __name__ == "__main__":
    main()
