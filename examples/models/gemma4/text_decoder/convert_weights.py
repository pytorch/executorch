# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# pyre-unsafe
# LICENSE file in the root directory of this source tree.

"""
Weight conversion from HuggingFace Gemma 4 to custom ExecuTorch format.

Maps HuggingFace model weights to the custom architecture, handling:
- Layer name mapping
- PLE weight handling
- Layer scalar mapping
- Tied embeddings (embed_tokens -> lm_head)
"""

import logging
from pathlib import Path
from typing import Dict, Optional

import torch

from .gemma4_config import Gemma4Config

logger = logging.getLogger(__name__)


def _download_manifold_file(manifold_path: str, local_path: Path) -> None:
    """Download a file from Manifold to a local path."""
    import io

    from manifold.clients.python import ManifoldClient

    if not manifold_path.startswith("manifold://"):
        raise ValueError(f"Invalid Manifold path: {manifold_path}")

    parts = manifold_path[len("manifold://") :].split("/", 1)
    bucket = parts[0]
    blob_path = parts[1] if len(parts) > 1 else ""

    logger.info(f"Downloading from Manifold: bucket={bucket}, path={blob_path}")

    with ManifoldClient.get_client(bucket) as client:
        output = io.BytesIO()
        client.sync_get(blob_path, output)
        local_path.parent.mkdir(parents=True, exist_ok=True)
        with open(local_path, "wb") as f:
            f.write(output.getvalue())


def _list_manifold_dir(manifold_path: str) -> list:
    """List files in a Manifold directory."""
    from manifold.clients.python import ManifoldClient

    if not manifold_path.startswith("manifold://"):
        raise ValueError(f"Invalid Manifold path: {manifold_path}")

    parts = manifold_path[len("manifold://") :].split("/", 1)
    bucket = parts[0]
    dir_path = parts[1] if len(parts) > 1 else ""

    files = []
    with ManifoldClient.get_client(bucket) as client:
        for name, _entry in client.sync_ls(dir_path):
            files.append(name)

    return files


def _load_safetensors_weights(checkpoint_path: str) -> Dict[str, torch.Tensor]:
    """Load weights from safetensors files.

    Args:
        checkpoint_path: Path to checkpoint directory containing safetensors files.
            Can be a local path or a Manifold path (manifold://bucket/path).

    Returns:
        Dictionary of weight tensors.
    """
    import tempfile

    try:
        from safetensors import safe_open
    except ImportError:
        raise ImportError(
            "safetensors is required for loading HuggingFace checkpoints. "
            "Install with: pip install safetensors"
        )

    state_dict = {}

    if checkpoint_path.startswith("manifold://"):
        all_files = _list_manifold_dir(checkpoint_path)
        safetensor_files = [f for f in all_files if f.endswith(".safetensors")]

        if not safetensor_files:
            raise FileNotFoundError(f"No safetensors files found in {checkpoint_path}")

        logger.info(f"Found {len(safetensor_files)} safetensors files in Manifold")

        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir_path = Path(tmpdir)
            for sf_name in safetensor_files:
                manifold_file_path = f"{checkpoint_path}/{sf_name}"
                local_file_path = tmpdir_path / sf_name

                logger.info(f"Downloading {sf_name}...")
                _download_manifold_file(manifold_file_path, local_file_path)

                with safe_open(str(local_file_path), framework="pt", device="cpu") as f:
                    for key in f.keys():
                        state_dict[key] = f.get_tensor(key)
    else:
        checkpoint_dir = Path(checkpoint_path)

        safetensor_files = list(checkpoint_dir.glob("*.safetensors"))
        if not safetensor_files:
            raise FileNotFoundError(f"No safetensors files found in {checkpoint_path}")

        for sf_file in safetensor_files:
            with safe_open(sf_file, framework="pt", device="cpu") as f:
                for key in f.keys():
                    state_dict[key] = f.get_tensor(key)

    return state_dict


def _get_weight_mapping(config: Gemma4Config) -> Dict[str, str]:
    """Get mapping from HuggingFace weight names to custom weight names."""
    mapping = {}

    # Embedding mappings
    mapping["model.language_model.embed_tokens.weight"] = (
        "model.self_decoder.embed_tokens.weight"
    )
    mapping["model.language_model.embed_tokens_per_layer.weight"] = (
        "model.self_decoder.embed_tokens_per_layer.weight"
    )
    mapping["model.language_model.per_layer_model_projection.weight"] = (
        "model.self_decoder.per_layer_model_projection.weight"
    )
    mapping["model.language_model.per_layer_projection_norm.weight"] = (
        "model.self_decoder.per_layer_projection_norm.weight"
    )

    # Final norm
    mapping["model.language_model.norm.weight"] = "model.norm.weight"

    # LM head (may not exist if tied)
    mapping["model.language_model.lm_head.weight"] = "model.lm_head.weight"

    # Layer mappings
    num_self_layers = config.num_self_decoder_layers
    for layer_idx in range(config.num_hidden_layers):
        hf_prefix = f"model.language_model.layers.{layer_idx}"

        if layer_idx < num_self_layers:
            custom_prefix = f"model.self_decoder.layers.{layer_idx}"
        else:
            cross_idx = layer_idx - num_self_layers
            custom_prefix = f"model.cross_decoder.layers.{cross_idx}"

        # Attention
        mapping[f"{hf_prefix}.self_attn.q_proj.weight"] = (
            f"{custom_prefix}.self_attn.q_proj.weight"
        )
        mapping[f"{hf_prefix}.self_attn.k_proj.weight"] = (
            f"{custom_prefix}.self_attn.k_proj.weight"
        )
        mapping[f"{hf_prefix}.self_attn.v_proj.weight"] = (
            f"{custom_prefix}.self_attn.v_proj.weight"
        )
        mapping[f"{hf_prefix}.self_attn.o_proj.weight"] = (
            f"{custom_prefix}.self_attn.o_proj.weight"
        )
        mapping[f"{hf_prefix}.self_attn.q_norm.weight"] = (
            f"{custom_prefix}.self_attn.q_norm.weight"
        )
        mapping[f"{hf_prefix}.self_attn.k_norm.weight"] = (
            f"{custom_prefix}.self_attn.k_norm.weight"
        )

        # MLP
        mapping[f"{hf_prefix}.mlp.gate_proj.weight"] = (
            f"{custom_prefix}.mlp.gate_proj.weight"
        )
        mapping[f"{hf_prefix}.mlp.up_proj.weight"] = (
            f"{custom_prefix}.mlp.up_proj.weight"
        )
        mapping[f"{hf_prefix}.mlp.down_proj.weight"] = (
            f"{custom_prefix}.mlp.down_proj.weight"
        )

        # Layer scalar
        mapping[f"{hf_prefix}.layer_scalar"] = f"{custom_prefix}.layer_scalar"

        # Per-layer input processing
        mapping[f"{hf_prefix}.per_layer_input_gate.weight"] = (
            f"{custom_prefix}.per_layer_input_gate.weight"
        )
        mapping[f"{hf_prefix}.per_layer_projection.weight"] = (
            f"{custom_prefix}.per_layer_projection.weight"
        )

        # LayerNorms
        mapping[f"{hf_prefix}.input_layernorm.weight"] = (
            f"{custom_prefix}.input_layernorm.weight"
        )
        mapping[f"{hf_prefix}.post_attention_layernorm.weight"] = (
            f"{custom_prefix}.post_attention_layernorm.weight"
        )
        mapping[f"{hf_prefix}.pre_feedforward_layernorm.weight"] = (
            f"{custom_prefix}.pre_feedforward_layernorm.weight"
        )
        mapping[f"{hf_prefix}.post_feedforward_layernorm.weight"] = (
            f"{custom_prefix}.post_feedforward_layernorm.weight"
        )
        mapping[f"{hf_prefix}.post_per_layer_input_norm.weight"] = (
            f"{custom_prefix}.post_per_layer_input_norm.weight"
        )

    return mapping


def convert_hf_to_custom(
    checkpoint_path: str,
    config: Gemma4Config,
    dtype: Optional[torch.dtype] = None,
) -> Dict[str, torch.Tensor]:
    """Convert HuggingFace checkpoint to custom format.

    Args:
        checkpoint_path: Path to HuggingFace checkpoint directory.
        config: Gemma4Config for the model.
        dtype: Optional dtype to convert weights to.

    Returns:
        State dict with converted weight names.
    """
    logger.info(f"Loading weights from {checkpoint_path}")

    hf_state_dict = _load_safetensors_weights(checkpoint_path)

    mapping = _get_weight_mapping(config)

    converted_state_dict = {}
    unmapped_keys = []

    for hf_key, tensor in hf_state_dict.items():
        if hf_key in mapping:
            custom_key = mapping[hf_key]
            if dtype is not None:
                tensor = tensor.to(dtype)
            converted_state_dict[custom_key] = tensor
            logger.debug(f"Mapped {hf_key} -> {custom_key}")
        else:
            unmapped_keys.append(hf_key)
            logger.warning(f"Unmapped key: {hf_key}")

    # Handle tied embeddings: copy embed_tokens.weight -> lm_head.weight if not in checkpoint
    lm_head_key = "model.lm_head.weight"
    embed_tokens_key = "model.self_decoder.embed_tokens.weight"
    if (
        lm_head_key not in converted_state_dict
        and embed_tokens_key in converted_state_dict
    ):
        logger.info(
            f"Using tied embeddings: copying {embed_tokens_key} to {lm_head_key}"
        )
        converted_state_dict[lm_head_key] = converted_state_dict[
            embed_tokens_key
        ].clone()

    logger.info(f"Converted {len(converted_state_dict)} weights")
    if unmapped_keys:
        logger.warning(f"Unmapped keys: {len(unmapped_keys)}")

    return converted_state_dict


def verify_conversion(
    hf_checkpoint_path: str,
    custom_state_dict: Dict[str, torch.Tensor],
    config: Gemma4Config,
) -> bool:
    """Verify that conversion preserved all necessary weights."""
    hf_state_dict = _load_safetensors_weights(hf_checkpoint_path)

    mapping = _get_weight_mapping(config)

    missing = []
    for hf_key, custom_key in mapping.items():
        if hf_key in hf_state_dict and custom_key not in custom_state_dict:
            missing.append((hf_key, custom_key))

    if missing:
        logger.error(f"Missing converted weights: {missing}")
        return False

    shape_mismatches = []
    for hf_key, custom_key in mapping.items():
        if hf_key in hf_state_dict and custom_key in custom_state_dict:
            hf_shape = hf_state_dict[hf_key].shape
            custom_shape = custom_state_dict[custom_key].shape
            if hf_shape != custom_shape:
                shape_mismatches.append((hf_key, hf_shape, custom_key, custom_shape))

    if shape_mismatches:
        logger.error(f"Shape mismatches: {shape_mismatches}")
        return False

    logger.info("Weight conversion verification passed")
    return True
