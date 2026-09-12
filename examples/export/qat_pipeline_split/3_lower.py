# Copyright 2026 NXP
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# Stage 3: lower the quantized graph to an ExecuTorch .pte program.
#
# Two example paths (selected with --example):
#
#   minimal (default)
#       Pass the .pt2 file path directly to export().  ExportSession detects
#       an ExportedProgram input and auto-skips SOURCE_TRANSFORM, QUANTIZE and
#       TORCH_EXPORT; the effective pipeline starts at TO_EDGE_TRANSFORM_AND_LOWER.
#
#   sliced
#       Load the GraphModule and set pipeline_stages explicitly to
#       [TORCH_EXPORT, TO_EDGE_TRANSFORM_AND_LOWER, TO_EXECUTORCH] - the
#       mirror image of the [SOURCE_TRANSFORM] slice used in stage 1.
#
# Both paths assert delegation happened and write model.pte.
# The model is run in the separate stage 4 (4_run.py).

import argparse
import os

import torch

from model import get_example_inputs


def _build_recipe():
    from executorch.backends.xnnpack.recipes.xnnpack_recipe_types import (
        XNNPackRecipeType,
    )
    from executorch.export.recipe import ExportRecipe

    return ExportRecipe.get_recipe(XNNPackRecipeType.PT2E_INT8_STATIC_PER_TENSOR)


def _finish(sess, workdir: str) -> None:
    """Assert delegation occurred and save model.pte."""
    from executorch.export.types import StageType

    pte_buffer = sess.get_pte_buffer()
    assert (
        pte_buffer is not None and len(pte_buffer) > 0
    ), "get_pte_buffer() returned an empty buffer - lowering may have failed"
    print(f"  PTE buffer size: {len(pte_buffer)} bytes.")

    artifacts = sess.get_stage_artifacts()
    lowering_stage = next(
        (
            s
            for s in (
                StageType.TO_EDGE_TRANSFORM_AND_LOWER,
                StageType.TO_BACKEND,
            )
            if s in artifacts
        ),
        None,
    )
    assert (
        lowering_stage is not None
    ), "No lowering stage artifact found - did TO_EDGE_TRANSFORM_AND_LOWER run?"

    delegation_info = artifacts[lowering_stage].get_context("delegation_info")
    assert (
        delegation_info is not None
    ), "delegation_info context is None - the lowering stage did not populate it"

    num_delegated = delegation_info.num_delegated_subgraphs
    assert num_delegated > 0, (
        f"Expected at least one delegated subgraph, got {num_delegated}. "
        "XNNPACK partitioner did not claim any nodes."
    )
    print(f"  Delegated subgraphs: {num_delegated}  (assertion passed).")

    print("\n  Delegation summary:")
    sess.print_delegation_info()

    pte_path = os.path.join(workdir, "model.pte")
    sess.save_to_pte(os.path.join(workdir, "model"))
    assert os.path.isfile(
        pte_path
    ), f"Expected .pte at {pte_path} but file was not created"
    print(f"\n  Saved -> {pte_path}  ({os.path.getsize(pte_path)} bytes)")


def run_minimal(workdir: str) -> None:
    from executorch.export import export as et_export
    from executorch.export.types import StageType

    pt2_in = os.path.join(workdir, "stage2_minimal_quantized.pt2")
    assert os.path.isfile(pt2_in), (
        f"Input file not found: {pt2_in}  " "(run 2_qat.py --example minimal first)"
    )
    print(f"[minimal] Lowering quantized model from {pt2_in}")

    recipe = _build_recipe()

    # Pass the file path directly.  ExportSession auto-skips SOURCE_TRANSFORM,
    # QUANTIZE and TORCH_EXPORT for ExportedProgram input.
    sess = et_export(pt2_in, export_recipe=recipe)

    artifacts = sess.get_stage_artifacts()
    for skipped in (
        StageType.SOURCE_TRANSFORM,
        StageType.QUANTIZE,
        StageType.TORCH_EXPORT,
    ):
        assert skipped not in artifacts, (
            f"Stage {skipped} should have been skipped for ExportedProgram input "
            "but an artifact was found"
        )
    print(
        "[minimal] Auto-skip verified: SOURCE_TRANSFORM / QUANTIZE / "
        "TORCH_EXPORT absent from artifacts (assertion passed)."
    )

    _finish(sess, workdir)


def run_sliced(workdir: str) -> None:
    from executorch.export import export as et_export
    from executorch.export.types import StageType

    pt2_in = os.path.join(workdir, "stage2_sliced_quantized.pt2")
    assert os.path.isfile(pt2_in), (
        f"Input file not found: {pt2_in}  " "(run 2_qat.py --example sliced first)"
    )
    print(f"[sliced]  Lowering quantized model from {pt2_in}")

    quantized_gm = torch.export.load(pt2_in).module()

    assert isinstance(
        quantized_gm, torch.fx.GraphModule
    ), f"Expected GraphModule after load, got {type(quantized_gm)}"
    print(
        f"[sliced]  Loaded GraphModule ({len(list(quantized_gm.graph.nodes))} nodes)."
    )

    recipe = _build_recipe()

    # Mirror image of the [SOURCE_TRANSFORM] slice in stage 1: together they
    # cover the full pipeline with no overlap and no stage silently skipped.
    recipe.pipeline_stages = [
        StageType.TORCH_EXPORT,
        StageType.TO_EDGE_TRANSFORM_AND_LOWER,
        StageType.TO_EXECUTORCH,
    ]

    ex = get_example_inputs()
    sess = et_export(quantized_gm, example_inputs=[ex], export_recipe=recipe)

    artifacts = sess.get_stage_artifacts()
    for expected_stage in recipe.pipeline_stages:
        assert (
            expected_stage in artifacts
        ), f"Expected artifact for stage {expected_stage} but none was produced"
    print(
        "[sliced]  All explicit pipeline stages produced artifacts (assertion passed)."
    )

    _finish(sess, workdir)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Stage 3: lower the quantized graph from stage 2 to an ExecuTorch .pte."
    )
    parser.add_argument("--example", choices=["minimal", "sliced"], default="minimal")
    parser.add_argument("--workdir", required=True)
    args = parser.parse_args()

    if args.example == "minimal":
        run_minimal(args.workdir)
    else:
        run_sliced(args.workdir)

    print(f"\nStage 3 done.  model.pte written to: {args.workdir}")


if __name__ == "__main__":
    main()
