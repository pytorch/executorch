# Copyright 2026 Arm Limited and/or its affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import hashlib
import json
import os
import warnings
from pathlib import Path
from typing import Any

from transformers import CLIPTextConfig, T5Config


_EXECUTORCH_SD35_UPSTREAM_SYNC_ENV_VAR = "EXECUTORCH_SD35_UPSTREAM_SYNC"
_sd35_large_upstream_checked: set[str] = set()

_SD35_LARGE_UPSTREAM_FINGERPRINTS: dict[str, tuple[tuple[str, ...], str]] = {
    "text_encoder": (
        (
            "hidden_act",
            "hidden_size",
            "intermediate_size",
            "max_position_embeddings",
            "num_attention_heads",
            "num_hidden_layers",
            "projection_dim",
            "vocab_size",
        ),
        "0cb164627ea428c39166d8ecf70462f7ddba34a57071d383a1116603e114bf50",
    ),
    "text_encoder_2": (
        (
            "hidden_act",
            "hidden_size",
            "intermediate_size",
            "max_position_embeddings",
            "num_attention_heads",
            "num_hidden_layers",
            "projection_dim",
            "vocab_size",
        ),
        "4c082c890625573879240ef700b1dbe18e2a263ad8e76d222cba759553176e1e",
    ),
    "text_encoder_3": (
        (
            "d_ff",
            "d_kv",
            "d_model",
            "dense_act_fn",
            "feed_forward_proj",
            "num_heads",
            "num_layers",
            "relative_attention_num_buckets",
            "vocab_size",
        ),
        "da069a817a2fe4eb347fc5aefd7690bf97f8661d33dbe689058977809651023d",
    ),
    "transformer": (
        (
            "sample_size",
            "patch_size",
            "in_channels",
            "num_layers",
            "attention_head_dim",
            "num_attention_heads",
            "caption_projection_dim",
            "joint_attention_dim",
            "pooled_projection_dim",
            "out_channels",
            "pos_embed_max_size",
            "qk_norm",
        ),
        "bc87c7eefb80f7bc6e80479f5bd2af929fd14c2331a19ddb4e27a989546ebfb0",
    ),
    "vae": (
        (
            "sample_size",
            "in_channels",
            "out_channels",
            "down_block_types",
            "up_block_types",
            "block_out_channels",
            "layers_per_block",
            "latent_channels",
            "norm_num_groups",
            "act_fn",
            "mid_block_add_attention",
            "force_upcast",
            "use_quant_conv",
            "use_post_quant_conv",
            "scaling_factor",
            "shift_factor",
        ),
        "8019be48e6681895b9b7c54c0f2f48c3ddc9975d7b6c7ef2061b7743f8d55b71",
    ),
}


def _load_upstream_sd35_large_config(subfolder: str) -> dict[str, Any]:
    from executorch.examples.models.stable_diffusion_3_5_large.model import MODEL_ID
    from huggingface_hub import hf_hub_download

    config_path = hf_hub_download(  # nosec B615
        repo_id=MODEL_ID,
        filename="config.json",
        subfolder=subfolder,
        token=os.environ.get("HF_TOKEN"),
        etag_timeout=1.0,
    )
    return json.loads(Path(config_path).read_text())


def _fingerprint_sd35_large_config(
    config: dict[str, Any], fields: tuple[str, ...]
) -> str:
    selected_config = {field: config[field] for field in fields}
    serialized_config = json.dumps(
        selected_config,
        sort_keys=True,
        separators=(",", ":"),
    )
    return hashlib.sha256(serialized_config.encode()).hexdigest()


def _warn_if_sd35_large_config_differs_from_upstream(
    subfolder: str, warning_name: str
) -> None:
    if os.environ.get(_EXECUTORCH_SD35_UPSTREAM_SYNC_ENV_VAR, "0") != "1":
        return
    if subfolder in _sd35_large_upstream_checked:
        return

    _sd35_large_upstream_checked.add(subfolder)
    try:
        upstream_config = _load_upstream_sd35_large_config(subfolder)
        fields, expected_fingerprint = _SD35_LARGE_UPSTREAM_FINGERPRINTS[subfolder]
        upstream_fingerprint = _fingerprint_sd35_large_config(upstream_config, fields)
    except Exception as exc:
        warnings.warn(
            f"Unable to validate {warning_name} against upstream metadata: {exc}",
            RuntimeWarning,
            stacklevel=2,
        )
        return

    if upstream_fingerprint != expected_fingerprint:
        warnings.warn(
            f"Upstream {warning_name} architecture changed; review the tiny test config",
            RuntimeWarning,
            stacklevel=2,
        )


def get_tiny_sd35_large_text_encoder_config() -> CLIPTextConfig:
    """Create a tiny SD3.5 Large-like CLIP-L text encoder config for tests."""
    _warn_if_sd35_large_config_differs_from_upstream(
        "text_encoder", "SD3.5 Large CLIP text encoder"
    )
    return CLIPTextConfig(  # type: ignore[call-arg]
        architectures=["CLIPTextModelWithProjection"],
        attention_dropout=0.0,
        bos_token_id=0,
        dropout=0.0,
        eos_token_id=2,
        hidden_act="quick_gelu",
        hidden_size=32,
        initializer_factor=1.0,
        initializer_range=0.02,
        intermediate_size=128,
        layer_norm_eps=1e-5,
        max_position_embeddings=16,
        num_attention_heads=4,
        num_hidden_layers=2,
        pad_token_id=1,
        projection_dim=32,
        dtype="float16",
        vocab_size=256,
    )


def get_tiny_sd35_large_text_encoder_2_config() -> CLIPTextConfig:
    """Create a tiny SD3.5 Large-like CLIP-bigG text encoder config for
    tests.
    """
    _warn_if_sd35_large_config_differs_from_upstream(
        "text_encoder_2", "SD3.5 Large CLIP text encoder 2"
    )
    return CLIPTextConfig(  # type: ignore[call-arg]
        architectures=["CLIPTextModelWithProjection"],
        attention_dropout=0.0,
        bos_token_id=0,
        dropout=0.0,
        eos_token_id=2,
        hidden_act="gelu",
        hidden_size=48,
        initializer_factor=1.0,
        initializer_range=0.02,
        intermediate_size=192,
        layer_norm_eps=1e-5,
        max_position_embeddings=16,
        num_attention_heads=6,
        num_hidden_layers=2,
        pad_token_id=1,
        projection_dim=48,
        dtype="float16",
        vocab_size=256,
    )


def get_tiny_sd35_large_t5_config() -> T5Config:
    """Create a tiny SD3.5 Large-like T5 config for tests."""
    _warn_if_sd35_large_config_differs_from_upstream(
        "text_encoder_3", "SD3.5 Large T5 text encoder"
    )
    return T5Config(  # type: ignore[call-arg]
        architectures=["T5EncoderModel"],
        classifier_dropout=0.0,
        d_ff=64,
        d_kv=8,
        d_model=32,
        decoder_start_token_id=0,
        dense_act_fn="gelu_new",
        dropout_rate=0.1,
        eos_token_id=1,
        feed_forward_proj="gated-gelu",
        initializer_factor=1.0,
        is_encoder_decoder=True,
        is_gated_act=True,
        layer_norm_epsilon=1e-6,
        num_decoder_layers=2,
        num_heads=4,
        num_layers=2,
        output_past=True,
        pad_token_id=0,
        relative_attention_max_distance=128,
        relative_attention_num_buckets=8,
        tie_word_embeddings=False,
        dtype="float16",
        vocab_size=256,
        use_cache=True,
    )


def get_tiny_sd35_large_transformer_config() -> dict[str, Any]:
    """Create a tiny SD3.5 Large-like MMDiT config for tests."""
    _warn_if_sd35_large_config_differs_from_upstream(
        "transformer", "SD3.5 Large transformer"
    )
    return {
        "sample_size": 32,
        "patch_size": 2,
        "in_channels": 4,
        "num_layers": 2,
        "attention_head_dim": 8,
        "num_attention_heads": 2,
        "caption_projection_dim": 16,
        "joint_attention_dim": 16,
        "pooled_projection_dim": 32,
        "out_channels": 4,
        "pos_embed_max_size": 32,
        "qk_norm": "rms_norm",
    }


def get_tiny_sd35_large_vae_config() -> dict[str, Any]:
    """Create a tiny SD3.5 Large-like VAE config for tests."""
    _warn_if_sd35_large_config_differs_from_upstream("vae", "SD3.5 Large VAE")
    return {
        "sample_size": 32,
        "in_channels": 3,
        "out_channels": 3,
        "down_block_types": (
            "DownEncoderBlock2D",
            "DownEncoderBlock2D",
            "DownEncoderBlock2D",
            "DownEncoderBlock2D",
        ),
        "up_block_types": (
            "UpDecoderBlock2D",
            "UpDecoderBlock2D",
            "UpDecoderBlock2D",
            "UpDecoderBlock2D",
        ),
        "block_out_channels": (4, 8, 8, 8),
        "layers_per_block": 1,
        "latent_channels": 16,
        "norm_num_groups": 1,
        "act_fn": "silu",
        "mid_block_add_attention": True,
        "force_upcast": False,
        "use_quant_conv": False,
        "use_post_quant_conv": False,
        "scaling_factor": 1.5305,
        "shift_factor": 0.0609,
    }
