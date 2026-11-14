"""Static SmolVLA topology and model configuration."""

from __future__ import annotations

import json
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

import torch
from huggingface_hub import hf_hub_download
from safetensors.torch import load_file
from transformers import AutoConfig


# Vision encoder graph name
VISION_ENCODER_GRAPH_NAME = "vision_encoder"
# Prefix model graph name
PREFIX_MODEL_GRAPH_NAME = "prefix_model"
# Action expert graph name
ACTION_EXPERT_GRAPH_NAME = "action_expert"


@dataclass(frozen=True)
class SmolVLAConfig:
    # Read from the checkpoint config.json by from_hf_policy().
    policy_repo_id: str
    policy_config: Mapping[str, Any]
    image_size: int
    language_tokens: int
    state_dim: int
    chunk_size: int
    max_action_dim: int
    action_feature: tuple[int, ...]
    num_steps: int
    vlm_model_name: str

    num_cameras: int = 2
    image_tokens_per_camera: int = 64
    prefix_layers: int = 16
    camera_keys: tuple[str, ...] = (
        "observation.images.camera1",
        "observation.images.camera2",
    )
    state_key: str = "observation.state"
    language_tokens_key: str = "observation.language.tokens"
    language_mask_key: str = "observation.language.attention_mask"

    @classmethod
    def from_hf_policy(cls, repo_id: str) -> "SmolVLAConfig":
        with open(hf_hub_download(repo_id, "config.json")) as file:
            policy_config = json.load(file)
        image_width, image_height = policy_config["resize_imgs_with_padding"]
        if image_width != image_height:
            raise ValueError(
                "SmolVLA export expects square resize_imgs_with_padding, "
                f"got {policy_config['resize_imgs_with_padding']}"
            )
        return cls(
            policy_repo_id=repo_id,
            policy_config=policy_config,
            image_size=image_width,
            language_tokens=policy_config["tokenizer_max_length"],
            state_dim=policy_config["max_state_dim"],
            chunk_size=policy_config["chunk_size"],
            max_action_dim=policy_config["max_action_dim"],
            action_feature=tuple(policy_config["output_features"]["action"]["shape"]),
            num_steps=policy_config["num_steps"],
            vlm_model_name=policy_config["vlm_model_name"],
        )

    @property
    def vlm_config(self) -> Any:
        return AutoConfig.from_pretrained(self.vlm_model_name)

    @property
    def prefix_length(self) -> int:
        return self.num_cameras * self.image_tokens_per_camera + self.language_tokens + 1

    @property
    def prefix_tail_length(self) -> int:
        """Language and state tokens appended after the camera embeddings."""
        return self.language_tokens + 1

    @property
    def action_context_length(self) -> int:
        return self.prefix_length + self.chunk_size

    def load_policy_state_dict(self) -> dict[str, torch.Tensor]:
        return load_file(hf_hub_download(self.policy_repo_id, "model.safetensors"))

    @property
    def pte_filenames(self) -> dict[str, str]:
        return {
            VISION_ENCODER_GRAPH_NAME: "smolvla_vision_encoder.pte",
            PREFIX_MODEL_GRAPH_NAME: "smolvla_prefix_model.pte",
            ACTION_EXPERT_GRAPH_NAME: "smolvla_action_expert.pte",
        }

    def create_action_expert_config(self) -> "SmolVLAExpertConfig":
        return SmolVLAExpertConfig.from_policy_config(
            self.policy_config,
            self.prefix_length,
            text_config=self.vlm_config.text_config,
        )

    def create_prefix_model_args(self) -> Any:
        """Static Llama arguments of the VLM prefix, mirroring the reference SmolVLA export."""
        from executorch.examples.models.llama.model_args import ModelArgs

        text_config = self.vlm_config.text_config
        return ModelArgs(
            dim=text_config.hidden_size,
            n_heads=text_config.num_attention_heads,
            n_kv_heads=text_config.num_key_value_heads,
            head_dim=text_config.head_dim,
            hidden_dim=text_config.intermediate_size,
            n_layers=self.prefix_layers,
            norm_eps=text_config.rms_norm_eps,
            vocab_size=text_config.vocab_size,
            max_seq_len=self.prefix_length,
            max_context_len=self.prefix_length,
            # SmolVLA applies RoPE with max_wavelength=10000 in both the VLM and the expert.
            rope_freq_base=10000.0,
            use_hf_rope=True,
        )

    def create_action_expert_model_args(self) -> Any:
        """Static Llama arguments of the Action Expert, mirroring the reference SmolVLA export."""
        from executorch.examples.models.llama.model_args import ModelArgs

        expert_config = self.create_action_expert_config()
        return ModelArgs(
            dim=expert_config.hidden_size,
            n_heads=expert_config.num_attention_heads,
            n_kv_heads=expert_config.num_key_value_heads,
            head_dim=expert_config.head_dim,
            hidden_dim=expert_config.intermediate_size,
            n_layers=expert_config.num_layers,
            norm_eps=expert_config.norm_eps,
            vocab_size=self.vlm_config.text_config.vocab_size,
            max_seq_len=self.action_context_length,
            max_context_len=self.action_context_length,
            rope_freq_base=10000.0,
            use_hf_rope=True,
        )


@dataclass(frozen=True)
class SmolVLAExpertConfig:
    prefix_len: int
    suffix_len: int
    max_action_dim: int
    num_layers: int
    hidden_size: int
    intermediate_size: int
    num_attention_heads: int
    num_key_value_heads: int
    head_dim: int
    norm_eps: float
    self_attn_every_n_layers: int
    min_period: float
    max_period: float

    def is_self_attention_layer(self, layer_idx: int) -> bool:
        return layer_idx % self.self_attn_every_n_layers == 0

    @classmethod
    def from_policy_config(
        cls,
        policy_config: Mapping[str, Any],
        prefix_len: int,
        *,
        text_config: Any,
    ) -> "SmolVLAExpertConfig":
        num_layers = policy_config["num_expert_layers"]
        if num_layers <= 0:
            num_layers = policy_config["num_vlm_layers"]
        hidden_size = int(
            text_config.hidden_size * policy_config["expert_width_multiplier"]
        )
        intermediate_size = int(2 * hidden_size / 3)
        intermediate_size = int(
            getattr(text_config, "ffn_dim_multiplier", 4) * intermediate_size
        )
        multiple_of = getattr(text_config, "multiple_of", 256)
        intermediate_size = multiple_of * (
            (intermediate_size + multiple_of - 1) // multiple_of
        )
        return cls(
            prefix_len=prefix_len,
            suffix_len=policy_config["chunk_size"],
            max_action_dim=policy_config["max_action_dim"],
            num_layers=num_layers,
            hidden_size=hidden_size,
            intermediate_size=intermediate_size,
            num_attention_heads=text_config.num_attention_heads,
            num_key_value_heads=text_config.num_key_value_heads,
            head_dim=text_config.head_dim,
            norm_eps=text_config.rms_norm_eps,
            self_attn_every_n_layers=policy_config["self_attn_every_n_layers"],
            min_period=policy_config["min_period"],
            max_period=policy_config["max_period"],
        )
