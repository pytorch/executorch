# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Vision-tower quantization + packing helpers.

This module is functionally ported from
``examples/models/gemma4/export_gemma4.py::_quantize_position_embedding_table``
(the E2B/E4B vision PE-int8 packer); the math is identical, only the
hidden-size constants differ. See the per-function ``Ported from`` notes.

Public API:

  * ``quantize_vision_position_table(vision_tower)`` -- in-place swap of
    the patch embedder's bf16 ``position_embedding_table`` Parameter with
    two buffers ``_pet_int8`` (per-channel int8) and ``_pet_scale``
    (fp32). The Gemma 4 31B vision PE table is (2, 10240, 1152) ≈ 47 MB
    bf16 → ~12 MB int8 + scale. Cosine sim vs bf16 reference > 0.999999
    in upstream Gemma 4 (E2B/E4B) experiments. Mirrors
    ``examples/models/gemma4/export_gemma4.py::_quantize_position_embedding_table``
    but operates on our own ported PatchEmbedder (which has hidden_size
    1152 instead of E2B's 768 and exposes ``_position_embeddings`` as the
    same instance method we monkey-patch here).

  * ``pack_vision_patch_embedder(patch_embedder, weights)`` — Gemma4-specific
    module packer that handles ``_pet_int8`` / ``_pet_scale`` plain tensors.
    This lets the generic ``load_and_pack_for_*`` APIs stream safetensors as
    usual while using the existing ``packers`` input for model-specific state
    adaptation.

  * ``install_int8_pe_dispatch(vision_tower)`` — same monkey-patch /
    buffer-shape installation but without quantizing existing data.

  * ``collect_vision_state_dict(vision_tower, embed_vision)`` — return a
    flat dict of all vision-side tensors (linears, norms, multimodal
    projector, plus the int8 PE buffers). All linears + norms are bf16;
    the PE table is in its int8/scale form.

  * ``has_vision_keys(safetensors_path)`` — peek at a saved checkpoint
    to detect whether it carries vision tensors. Used by
    load_prequantized_model so the new load path is purely additive.

These keys ride alongside the quantized LM in the same safetensors file
because torchao's ``flatten_tensor_state_dict`` accepts a mixed dict of
quantized subclass tensors + plain tensors and lists every name in
``metadata['tensor_names']``. The existing
the generic loaders iterate that list and route plain tensors through
``pack_one``; the Gemma4 patch-embedder packer handles ``_pet_int8`` /
``_pet_scale`` before the default register-buffer fallback.
"""

from __future__ import annotations

import types

import torch
import torch.nn as nn
import torch.nn.functional as F


# Anything in the model whose flat key starts with one of these is a
# vision-side tensor. Used by quantize_and_save.py and the loader to
# branch additively without touching the text-decoder code path.
VISION_PREFIXES: tuple[str, ...] = ("vision_tower.", "embed_vision.")


# ---------------------------------------------------------------------------
# Position-embedding-table quantization (the only "real" quantization on the
# vision side — every other vision weight stays bf16).
# ---------------------------------------------------------------------------


def _patch_position_embeddings_int8(patch_embedder: nn.Module) -> None:
    """Monkey-patch ``_position_embeddings`` to dequantize + index the int8 PE table.

    Ported from
    ``examples/models/gemma4/export_gemma4.py::_position_embeddings_int8``
    (Gemma 4 E2B/E4B int8 PE table lookup).

    Uses the same one-hot-matmul shape that HF's
    ``Gemma4VisionPatchEmbedder._position_embeddings`` produces, except we
    dequantize ``_pet_int8 * _pet_scale`` first. We also stick to
    ``F.embedding`` for the per-axis lookup to keep the graph tiny — that's
    also what the text-decoder vision_tower port uses (so the numerics are
    bit-for-bit those of the bf16 reference, modulo the int8 round-trip
    on the table itself).
    """

    def _position_embeddings(
        self,
        pixel_position_ids: torch.Tensor,  # [B, P, 2]
        padding_positions: torch.Tensor,  # [B, P] (True = padding)
    ) -> torch.Tensor:
        # Dequantize lazily so the bf16 graph stays bf16. (2, 10240, 1152)
        table = self._pet_int8.to(self._pet_scale.dtype) * self._pet_scale
        clamped = pixel_position_ids.clamp(min=0).long()
        emb_x = F.embedding(clamped[..., 0], table[0])
        emb_y = F.embedding(clamped[..., 1], table[1])
        pos_emb = (emb_x + emb_y).to(self.input_proj.weight.dtype)
        zero = torch.zeros_like(pos_emb)
        return torch.where(padding_positions.unsqueeze(-1), zero, pos_emb)

    patch_embedder._position_embeddings = types.MethodType(
        _position_embeddings, patch_embedder
    )


def quantize_vision_position_table(
    vision_tower: nn.Module,
    *,
    verbose: bool = False,
) -> None:
    """Replace ``vision_tower.patch_embedder.position_embedding_table`` with
    int8 per-channel data + fp32 scale buffers, and patch the lookup method.

    Ported from
    ``examples/models/gemma4/export_gemma4.py::_quantize_position_embedding_table``
    (the E2B/E4B vision PE-int8 packer). Same per-channel quant math; only the
    hidden-size constant differs (1152 here vs 768 in the E2B port).

    Idempotent: a second call is a no-op.
    """
    pe = vision_tower.patch_embedder
    pet = getattr(pe, "position_embedding_table", None)
    if pet is None:
        return  # already quantized

    if pet.device.type == "meta":
        raise RuntimeError(
            "quantize_vision_position_table requires a real (non-meta) "
            "position_embedding_table tensor. Load the HF weights first."
        )

    pet_fp = pet.data.to(torch.float32)
    # Per-channel along the last (hidden) dim — same axis as the gemma4
    # E2B/E4B reference.
    scale = pet_fp.abs().amax(dim=-1, keepdim=True).clamp(min=1e-8) / 127.0
    qdata = torch.round(pet_fp / scale).clamp(-128, 127).to(torch.int8)
    scale = scale.to(torch.float32)

    # Drop the parameter. After this, named_parameters() no longer yields it.
    del pe.position_embedding_table

    pe.register_buffer("_pet_int8", qdata, persistent=True)
    pe.register_buffer("_pet_scale", scale, persistent=True)

    _patch_position_embeddings_int8(pe)

    if verbose:
        bf16_mb = pet.numel() * 2 / (1024 * 1024)
        new_mb = (qdata.numel() + scale.numel() * 4) / (1024 * 1024)
        print(
            f"  vision PE table: bf16 -> int8 per-channel "
            f"({bf16_mb:.1f} MB -> {new_mb:.1f} MB)"
        )


def install_int8_pe_dispatch(
    vision_tower: nn.Module,
    *,
    verbose: bool = False,
) -> None:
    """Build-on-meta companion to ``quantize_vision_position_table``.

    Swaps the freshly-constructed ``position_embedding_table`` Parameter
    for zero placeholder buffers ``_pet_int8`` / ``_pet_scale`` and
    monkey-patches the lookup method.
    """
    pe = vision_tower.patch_embedder
    if not hasattr(pe, "position_embedding_table"):
        # Already swapped (idempotent).
        if not hasattr(pe, "_pet_int8"):
            raise RuntimeError(
                "install_int8_pe_dispatch: patch_embedder has neither the "
                "original position_embedding_table nor _pet_int8/_pet_scale."
            )
        return

    # Inspect shape from the existing parameter (works on meta).
    pet = pe.position_embedding_table
    shape = tuple(pet.shape)  # (2, position_embedding_size, hidden_size)
    del pe.position_embedding_table
    pe.register_buffer(
        "_pet_int8",
        torch.zeros(shape, dtype=torch.int8, device="meta"),
        persistent=True,
    )
    pe.register_buffer(
        "_pet_scale",
        torch.zeros((shape[0], shape[1], 1), dtype=torch.float32, device="meta"),
        persistent=True,
    )
    _patch_position_embeddings_int8(pe)

    if verbose:
        print("  vision PE table: int8 dispatch installed (placeholder buffers)")


def pack_vision_patch_embedder(
    patch_embedder: nn.Module,
    weights: dict[str, torch.Tensor],
) -> bool:
    """Install/load Gemma4 vision PE int8 buffers via generic packers."""
    if not any(k in weights for k in ("_pet_int8", "_pet_scale")):
        return False

    if hasattr(patch_embedder, "position_embedding_table"):
        dummy_tower = types.SimpleNamespace(patch_embedder=patch_embedder)
        install_int8_pe_dispatch(dummy_tower)

    for name, value in weights.items():
        if name not in ("_pet_int8", "_pet_scale"):
            return False
        patch_embedder.register_buffer(name, value, persistent=True)
    return True


# ---------------------------------------------------------------------------
# Vision state-dict collection (bf16 passthrough + int8 PE buffers).
# ---------------------------------------------------------------------------


def collect_vision_state_dict(
    vision_tower: nn.Module,
    embed_vision: nn.Module,
    *,
    dtype: torch.dtype = torch.bfloat16,
) -> dict[str, torch.Tensor]:
    """Return a flat dict of all vision-side tensors ready to mix into the
    saved safetensors.

    Caller is responsible for invoking ``quantize_vision_position_table``
    on ``vision_tower`` first if the int8 PE recipe is desired (this is
    what quantize_and_save.py does).

    Output keys:
      * ``vision_tower.*`` — every Parameter and persistent buffer of
        ``vision_tower`` (linears + norms in ``dtype``; the PE buffers
        are kept in their native int8 / fp32 dtypes).
      * ``embed_vision.*`` — the multimodal projector linear and norm.

    Norms / linear weights are cast to ``dtype`` (bf16 by default) so the
    file-level dtype mix matches the LM's quantized + bf16 plain
    tensors. Integer PE buffers and fp32 PE scale are passed through
    unchanged.
    """
    state: dict[str, torch.Tensor] = {}

    def _maybe_cast(name: str, t: torch.Tensor) -> torch.Tensor:
        # PE buffers + scale: keep native dtype.
        if name.endswith("._pet_int8") or name.endswith("._pet_scale"):
            return t.detach().contiguous()
        if t.dtype.is_floating_point:
            return t.detach().to(dtype).contiguous()
        return t.detach().contiguous()

    # vision_tower parameters
    for sub_fqn, param in vision_tower.named_parameters():
        key = f"vision_tower.{sub_fqn}"
        state[key] = _maybe_cast(key, param.data)
    # vision_tower buffers (only persistent ones — std_bias/std_scale, _pet_*)
    persistent = set(vision_tower.state_dict().keys())
    for sub_fqn, buf in vision_tower.named_buffers():
        if sub_fqn not in persistent:
            continue
        key = f"vision_tower.{sub_fqn}"
        if key in state:
            continue
        state[key] = _maybe_cast(key, buf.data)

    # embed_vision parameters
    for sub_fqn, param in embed_vision.named_parameters():
        key = f"embed_vision.{sub_fqn}"
        state[key] = _maybe_cast(key, param.data)
    # embed_vision buffers (RMSNormNoWeight has none; defensive walk anyway)
    persistent_ev = set(embed_vision.state_dict().keys())
    for sub_fqn, buf in embed_vision.named_buffers():
        if sub_fqn not in persistent_ev:
            continue
        key = f"embed_vision.{sub_fqn}"
        if key in state:
            continue
        state[key] = _maybe_cast(key, buf.data)

    return state


# ---------------------------------------------------------------------------
# Load-side helpers
# ---------------------------------------------------------------------------


def has_vision_keys(safetensors_path: str) -> bool:
    """Return True iff the file contains any ``vision_tower.*`` /
    ``embed_vision.*`` key.

    Used by ``load_prequantized_model`` so the existing text-only load path
    keeps working byte-for-byte when the checkpoint was saved with
    ``--no-vision`` (or by an old quantize_and_save.py).
    """
    from safetensors import safe_open

    with safe_open(safetensors_path, framework="pt", device="cpu") as f:
        for k in f.keys():
            if k.startswith(VISION_PREFIXES):
                return True
    return False


__all__ = [
    "VISION_PREFIXES",
    "quantize_vision_position_table",
    "install_int8_pe_dispatch",
    "collect_vision_state_dict",
    "has_vision_keys",
]
