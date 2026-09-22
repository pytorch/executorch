# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Helpers shared by the solo and DFlash export paths."""

import json
import math
import os

import torch
import torch.nn as nn


# CLI choice -> torch dtype for the activation compute path (and, for the MLX
# backend, the KV cache + unquantized weights/buffers).
ACTIVATION_DTYPES: dict[str, torch.dtype] = {
    "bfloat16": torch.bfloat16,
    "float16": torch.float16,
}


class BoundMethodForward:
    """Context manager: temporarily set ``model.forward`` to a bound method.

    All exported methods live on the SAME model instance, so exporting each with
    ``torch.export.export(model, ...)`` requires shadowing the class ``forward``
    with an instance-attribute bound method for the duration of the export. The
    model's class identity never changes, so every ExportedProgram shares
    identical mutable-buffer FQNs (``layers.X.self_attn.kv_cache.*``) and the
    CUDA backend can unify the prefill/decode KV-cache buffers at runtime.
    """

    def __init__(self, model: nn.Module, bound_method) -> None:
        self._model = model
        self._bound = bound_method

    def __enter__(self):
        self._model.forward = self._bound  # instance-attr shadows class method
        return self._model

    def __exit__(self, *exc):
        try:
            del self._model.forward
        except AttributeError:
            pass
        return False


def mutable_buffer_metadata(model: nn.Module) -> str:
    """JSON list of per-session mutable KV-cache buffer FQNs.

    The serving worker (muse_glimmer_engine) reads this via the ``get_mutable_buffer_metadata``
    constant method to bind each session's own KV cache, so one loaded model can
    host multiple isolated conversations. Matches gemma4_31b's filter.
    """
    mutable = [name for name, _ in model.named_buffers() if ".kv_cache." in name]
    return json.dumps({"version": 1, "mutable_buffers": mutable})


def activation_dtype_tag(dtype: torch.dtype) -> str:
    """Canonical string tag recorded in the .pte as ``get_activation_dtype``."""
    return {
        torch.bfloat16: "bfloat16",
        torch.float16: "float16",
    }[dtype]


def sample_vision_inputs(
    pos_embed_table: torch.Tensor,
    vision_config,
    grid_side: int = 32,
):
    """Build valid sample inputs for tracing ``vision_encoder``.

    Uses a zeros image of ``grid_side*patch`` px so the 9 host-precomputed
    tensors have realistic shapes/values (valid permutations and masks) for a
    strict-export trace. ``grid_side=32`` -> 1024 patches -> 256 soft tokens.
    """
    from executorch.examples.models.muse_glimmer.vision.precompute import (
        precompute_vision_inputs,
    )

    px = grid_side * vision_config.patch_size
    image = torch.zeros(3, px, px, dtype=torch.float32)
    inputs = precompute_vision_inputs(image, pos_embed_table, vision_config)
    return inputs.as_args()


def vision_sample_grid_side(max_patches: int, downsample_factor: int) -> int:
    side = min(32, math.isqrt(max_patches))
    side -= side % downsample_factor
    if side < downsample_factor:
        raise ValueError(
            "max_vision_patches must fit at least one downsampled patch group"
        )
    return side


def export_vision_encoder(
    vision_model: nn.Module,
    pos_embed_table: torch.Tensor,
    max_vision_patches: int,
):
    """Export ``vision_encoder`` with num_patches dynamic across all 9 inputs.

    The dynamic-shape spec is positional, so it must stay in step with the
    tensor order ``sample_vision_inputs`` produces.
    """
    from torch.export import Dim, export

    vis_cfg = vision_model.config
    ds2 = vis_cfg.downsample_factor**2
    max_groups = max(1, max_vision_patches // ds2)
    num_patches = ds2 * Dim("vis_groups", min=1, max=max_groups)

    print(f"Exporting vision_encoder (P in [{ds2}, {ds2 * max_groups}])...")
    vis_args = sample_vision_inputs(
        pos_embed_table,
        vis_cfg,
        grid_side=vision_sample_grid_side(
            max_vision_patches, vis_cfg.downsample_factor
        ),
    )
    # Shared dynamic num_patches across all vision inputs (masks use it on
    # both the query and key axes).
    vis_dynamic = (
        {1: num_patches},  # patches [1,P,1176]
        {1: num_patches},  # pos_emb [1,P,1536]
        {0: num_patches},  # cos_2d [P,48]
        {0: num_patches},  # sin_2d [P,48]
        {0: num_patches},  # sparse_perm [P]
        {0: num_patches},  # inv_perm [P]
        {2: num_patches, 3: num_patches},  # global_mask [1,1,P,P]
        {2: num_patches, 3: num_patches},  # sparse_mask [1,1,P,P]
        {0: num_patches},  # pixel_perm [P]
    )
    with torch.no_grad():
        return export(
            vision_model,
            vis_args,
            dynamic_shapes=vis_dynamic,
            strict=True,
        )


def save_pte(et_program, output_dir: str, pos_embed_table: torch.Tensor | None) -> None:
    """Write ``model.pte``, any ``.ptd`` tensor data, and the vision pos-embed table."""
    os.makedirs(output_dir, exist_ok=True)
    pte_path = os.path.join(output_dir, "model.pte")
    print(f"Saving to {pte_path}...")
    with open(pte_path, "wb") as f:
        et_program.write_to_file(f)
    print(f"  {os.path.getsize(pte_path) / 1024**2:.1f} MB")

    if et_program._tensor_data:
        et_program.write_tensor_data_to_file(output_dir)
        print(f"  Saved tensor data (.ptd) to {output_dir}/")

    if pos_embed_table is not None:
        # The 32x32 positional-embedding table is used by the runner for host
        # grid_sample interpolation. Written as raw float32 [pos_tokens, latent].
        pe_path = os.path.join(output_dir, "pos_embed.bin")
        pe = pos_embed_table.to(torch.float32).contiguous()
        with open(pe_path, "wb") as f:
            f.write(pe.numpy().tobytes())
        print(f"  Saved vision pos-embed table ({tuple(pe.shape)} f32) to {pe_path}")
