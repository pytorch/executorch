# Copyright 2026 NXP
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# Stage 1: capture the model up to the quantize boundary.
#
# Two example paths (selected with --example):
#
#   minimal (default)
#       Export the eager nn.Module to an ATEN-dialect ExportedProgram with a
#       single torch.export.export call.  No recipe, no ExportSession.
#
#   sliced
#       Drive only the SOURCE_TRANSFORM stage through ExportSession, then
#       export to ATEN and save.  This shows how to restrict the pipeline to
#       the stages that precede quantization.
#
# Both paths write a .pt2 file consumed by 2_qat.py.

import argparse
import os

import torch

from model import get_example_inputs, get_model


def _build_recipe():
    from executorch.backends.xnnpack.recipes.xnnpack_recipe_types import (
        XNNPackRecipeType,
    )
    from executorch.export.recipe import ExportRecipe

    return ExportRecipe.get_recipe(XNNPackRecipeType.PT2E_INT8_STATIC_PER_TENSOR)


def run_minimal(workdir: str) -> None:
    print("[minimal] Capturing ATEN graph with torch.export.export (no recipe).")

    model = get_model()
    ex = get_example_inputs()

    assert isinstance(
        model, torch.nn.Module
    ), f"Expected nn.Module from get_model(), got {type(model)}"

    ep = torch.export.export(model, ex, strict=True)

    assert ep is not None, "torch.export.export returned None"
    print(f"[minimal] Graph nodes: {len(list(ep.graph.nodes))}")

    out = os.path.join(workdir, "stage1_minimal.pt2")
    torch.export.save(ep, out)
    print(f"[minimal] Saved ExportedProgram -> {out}")

    reloaded_gm = torch.export.load(out).module()
    y = reloaded_gm(*ex)
    assert y.shape == (
        1,
        10,
    ), f"[minimal] Unexpected output shape after reload: {y.shape}"
    print(f"[minimal] Round-trip output shape: {y.shape}  (assertion passed)")


def run_sliced(workdir: str) -> None:
    print("[sliced]  Running SOURCE_TRANSFORM stage only via ExportSession.")

    from executorch.export import export as et_export
    from executorch.export.types import StageType

    recipe = _build_recipe()

    # Restrict the pipeline to the pre-quantize stages only.
    recipe.pipeline_stages = [StageType.SOURCE_TRANSFORM]

    model = get_model()
    ex = get_example_inputs()

    assert isinstance(
        model, torch.nn.Module
    ), f"Expected nn.Module from get_model(), got {type(model)}"

    sess = et_export(model, example_inputs=[ex], export_recipe=recipe)

    artifacts = sess.get_stage_artifacts()
    assert (
        StageType.SOURCE_TRANSFORM in artifacts
    ), "SOURCE_TRANSFORM artifact not found - did the stage run?"

    transformed = artifacts[StageType.SOURCE_TRANSFORM].data
    assert isinstance(
        transformed, dict
    ), f"Expected method-keyed dict from SOURCE_TRANSFORM, got {type(transformed)}"
    assert (
        "forward" in transformed
    ), "'forward' method missing from SOURCE_TRANSFORM artifact"

    transformed_module = transformed["forward"]
    assert isinstance(
        transformed_module, torch.nn.Module
    ), f"Expected nn.Module after SOURCE_TRANSFORM, got {type(transformed_module)}"
    print(
        f"[sliced]  SOURCE_TRANSFORM produced: {type(transformed_module).__name__}"
        " (pass-through for this PT2E recipe, as expected)"
    )

    ep = torch.export.export(transformed_module, ex, strict=True)
    assert ep is not None, "torch.export.export returned None"
    print(f"[sliced]  Graph nodes: {len(list(ep.graph.nodes))}")

    out = os.path.join(workdir, "stage1_sliced.pt2")
    torch.export.save(ep, out)
    print(f"[sliced]  Saved ExportedProgram -> {out}")

    reloaded_gm = torch.export.load(out).module()
    y = reloaded_gm(*ex)
    assert y.shape == (
        1,
        10,
    ), f"[sliced] Unexpected output shape after reload: {y.shape}"
    print(f"[sliced]  Round-trip output shape: {y.shape}  (assertion passed)")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Stage 1: capture the model up to the quantize boundary."
    )
    parser.add_argument(
        "--example",
        choices=["minimal", "sliced"],
        default="minimal",
    )
    parser.add_argument("--workdir", required=True)
    args = parser.parse_args()

    os.makedirs(args.workdir, exist_ok=True)

    if args.example == "minimal":
        run_minimal(args.workdir)
    else:
        run_sliced(args.workdir)

    print(f"\nStage 1 done.  Artifact written to: {args.workdir}")


if __name__ == "__main__":
    main()
