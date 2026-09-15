# Copyright 2026 NXP
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# Stage 2: perform arbitrary Quantization-Aware Training (QAT).
#
# Loads the captured ATEN graph from stage 1, prepares it for QAT, runs dummy
# forward passes to exercise the observers, saves and restores a checkpoint,
# then converts to a quantized graph and saves it for stage 3.
#
# No real QAT is performed: the dummy forward passes only exist to show that
# observers collect statistics in training mode.

import argparse
import copy
import os

import torch

from model import get_calibration_inputs, get_example_inputs


def _get_quantizer():
    from executorch.backends.xnnpack.recipes.xnnpack_recipe_types import (
        XNNPackRecipeType,
    )
    from executorch.export.recipe import ExportRecipe

    recipe = ExportRecipe.get_recipe(XNNPackRecipeType.PT2E_INT8_STATIC_PER_TENSOR)
    quantizers = recipe.quantization_recipe.quantizers
    assert (
        quantizers and len(quantizers) > 0
    ), "Recipe carries no quantizers - cannot prepare for QAT"
    return quantizers[0]


def _run_qat(captured_gm: torch.fx.GraphModule, workdir: str) -> torch.fx.GraphModule:
    from torchao.quantization.pt2e import (
        move_exported_model_to_eval,
        move_exported_model_to_train,
    )
    from torchao.quantization.pt2e.quantize_pt2e import convert_pt2e, prepare_qat_pt2e

    calib = get_calibration_inputs(n=4)
    quantizer = _get_quantizer()

    # prepare_qat_pt2e mutates the graph in-place; keep a pristine copy for the
    # checkpoint restore demo below.
    pristine_gm = copy.deepcopy(captured_gm)
    prepared = prepare_qat_pt2e(captured_gm, quantizer)

    assert prepared is not None, "prepare_qat_pt2e returned None"

    # Verify that observer / fake-quant modules were inserted.
    # Matched by call_module node target prefix because class identity differs
    # between torchao and torch.ao namespaces.
    obs_nodes = [
        n
        for n in prepared.graph.nodes
        if n.op == "call_module"
        and isinstance(n.target, str)
        and n.target.startswith("activation_post_process")
    ]
    assert len(obs_nodes) > 0, (
        "No activation_post_process nodes found after prepare_qat_pt2e - "
        "the quantizer may not have annotated this graph"
    )
    obs_type_names = ", ".join(
        type(dict(prepared.named_modules())[n.target]).__name__ for n in obs_nodes[:3]
    )
    print(
        f"  Inserted {len(obs_nodes)} observer/fake-quant nodes "
        f"({obs_type_names}{'...' if len(obs_nodes) > 3 else ''})."
    )

    prepared = move_exported_model_to_train(prepared)

    print("  Running dummy training steps (observers collecting statistics)...")
    for i, inputs in enumerate(calib):
        _ = prepared(*inputs)
        print(f"    step {i + 1}/{len(calib)}  input shape: {inputs[0].shape}")

    # Save observer state so training can be paused and resumed later.
    ckpt_path = os.path.join(workdir, "qat_ckpt.pt")
    torch.save(prepared.state_dict(), ckpt_path)
    print(f"  Checkpoint saved -> {ckpt_path}")

    # Restore into a fresh prepared module to demonstrate the pause/resume seam.
    print("  Demonstrating checkpoint restore into a fresh prepared module...")
    fresh_prepared = prepare_qat_pt2e(pristine_gm, quantizer)
    fresh_prepared = move_exported_model_to_train(fresh_prepared)

    saved_state = torch.load(ckpt_path, weights_only=True)
    fresh_prepared.load_state_dict(saved_state)

    restored_state = fresh_prepared.state_dict()
    assert set(saved_state.keys()) == set(
        restored_state.keys()
    ), "State dict keys differ after checkpoint restore"
    for key in saved_state:
        assert torch.allclose(
            saved_state[key], restored_state[key]
        ), f"Tensor mismatch for key '{key}' after checkpoint restore"
    print("  Checkpoint restore verified: all tensors match (assertion passed).")

    prepared = fresh_prepared

    prepared = move_exported_model_to_eval(prepared)
    converted = convert_pt2e(prepared)

    assert converted is not None, "convert_pt2e returned None"

    # After convert_pt2e the quantize/dequantize ops are in the
    # quantized_decomposed namespace; match by __name__ substring.
    qdq_nodes = [
        n
        for n in converted.graph.nodes
        if n.op == "call_function"
        and hasattr(n.target, "__name__")
        and (
            "quantize_per_tensor" in n.target.__name__
            or "dequantize_per_tensor" in n.target.__name__
            or "quantize_per_channel" in n.target.__name__
            or "dequantize_per_channel" in n.target.__name__
        )
    ]
    assert len(qdq_nodes) > 0, (
        "No quantize/dequantize nodes found after convert_pt2e - "
        "conversion may have failed silently"
    )
    print(f"  convert_pt2e inserted {len(qdq_nodes)} quantize/dequantize ops.")

    return converted


def run(example: str, workdir: str) -> None:
    pt2_in = os.path.join(workdir, f"stage1_{example}.pt2")
    assert os.path.isfile(pt2_in), (
        f"Input file not found: {pt2_in}  "
        f"(run 1_prepare.py --example {example} first)"
    )

    print(f"[{example}] Loading captured graph from {pt2_in}")
    captured_gm = torch.export.load(pt2_in).module()

    assert isinstance(
        captured_gm, torch.fx.GraphModule
    ), f"Expected GraphModule after load, got {type(captured_gm)}"
    print(
        f"[{example}] Loaded GraphModule with {len(list(captured_gm.graph.nodes))} nodes."
    )

    converted = _run_qat(captured_gm, workdir)

    ex = get_example_inputs()
    ep = torch.export.export(converted, ex, strict=True)
    out = os.path.join(workdir, f"stage2_{example}_quantized.pt2")
    torch.export.save(ep, out)
    print(f"[{example}] Saved quantized ExportedProgram -> {out}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Stage 2: perform arbitrary QAT on the captured graph from stage 1."
    )
    parser.add_argument("--example", choices=["minimal", "sliced"], default="minimal")
    parser.add_argument("--workdir", required=True)
    args = parser.parse_args()

    run(args.example, args.workdir)

    print(f"\nStage 2 done.  Artifacts in: {args.workdir}")


if __name__ == "__main__":
    main()
