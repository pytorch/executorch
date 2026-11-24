# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Tiny random models shared by the asset-independent tests."""

import math

import torch
from transformers import DynamicCache, Qwen3Config, Qwen3Model

from ..model.backbone import ReverbBackbone
from ..model.config import CODEBOOKS, CODES, MASK, ReverbConfig


def tiny_config(**overrides) -> ReverbConfig:
    values = {
        "vocab_size": 97,
        "hidden_size": 32,
        "intermediate_size": 48,
        "num_layers": 2,
        "num_heads": 4,
        "num_kv_heads": 2,
        "head_dim": 8,
        "rms_norm_eps": 1e-6,
        "rope_theta": 10000.0,
        "embedding_rank": 8,
    }
    values.update(overrides)
    return ReverbConfig(**values)


def checkpoint_dict(config: ReverbConfig) -> dict:
    """A config.json body in the released checkpoint's layout."""
    return {
        "model_type": "omnivoice",
        "num_audio_codebook": CODEBOOKS,
        "audio_vocab_size": CODES,
        "audio_mask_id": MASK,
        "llm_config": {
            "model_type": "qwen3",
            "vocab_size": config.vocab_size,
            "hidden_size": config.hidden_size,
            "intermediate_size": config.intermediate_size,
            "num_hidden_layers": config.num_layers,
            "num_attention_heads": config.num_heads,
            "num_key_value_heads": config.num_kv_heads,
            "head_dim": config.head_dim,
            "rms_norm_eps": config.rms_norm_eps,
            "rope_parameters": {
                "rope_theta": config.rope_theta,
                "rope_type": "default",
            },
        },
        "reverb": {"architecture": {"embedding_rank": config.embedding_rank}},
    }


def random_backbone(config: ReverbConfig, seed: int = 0) -> ReverbBackbone:
    torch.manual_seed(seed)
    model = ReverbBackbone(config).eval()
    with torch.no_grad():
        for parameter in model.parameters():
            parameter.normal_(0.0, 0.2)
    return model


class ReferenceBackbone:
    """The reference sampler's math on Hugging Face's Qwen3Model, for batch 1.

    Mirrors reverb_tts.sampler.Batch: the prefix runs once into a DynamicCache
    and every pass runs the target frames against it, with an additive bias on
    the reference frames in the polish passes.
    """

    def __init__(self, model: ReverbBackbone) -> None:
        config = model.config
        self.model = model
        self.llm = Qwen3Model(
            Qwen3Config(
                vocab_size=config.vocab_size,
                hidden_size=config.hidden_size,
                intermediate_size=config.intermediate_size,
                num_hidden_layers=config.num_layers,
                num_attention_heads=config.num_heads,
                num_key_value_heads=config.num_kv_heads,
                head_dim=config.head_dim,
                rms_norm_eps=config.rms_norm_eps,
                rope_parameters={
                    "rope_theta": config.rope_theta,
                    "rope_type": "default",
                },
                attn_implementation="sdpa",
            )
        ).eval()
        state = {
            key[len("llm.") :]: value
            for key, value in model.state_dict().items()
            if key.startswith("llm.") and "embed_tokens" not in key
        }
        missing, _ = self.llm.load_state_dict(state, strict=False)
        assert set(missing) == {"embed_tokens.weight"}
        self.heads = torch.cat(
            [
                model.audio_heads.main.weight,
                model.audio_heads.middle.weight,
                model.audio_heads.late.weight,
            ]
        )
        self.offsets = CODES * torch.arange(CODEBOOKS)

    def open(self, ids: torch.Tensor, codes: torch.Tensor, frames: int, bias: float):
        prefix = torch.cat(
            [
                self.model.llm.embed_tokens(ids),
                self.model.audio_embeddings(codes + self.offsets[:, None]).sum(0),
            ]
        )
        length, references = prefix.shape[0], codes.shape[1]
        self.cache = DynamicCache()
        self.llm(
            inputs_embeds=prefix[None],
            attention_mask=torch.zeros(1, 1, length, length),
            position_ids=torch.arange(length)[None],
            past_key_values=self.cache,
        )
        polish = torch.zeros(1, 1, frames, length + frames)
        polish[..., length - references : length] = bias
        self.masks = {False: torch.zeros(1, 1, frames, length + frames), True: polish}
        self.positions = (length + torch.arange(frames))[None]
        self.frames = frames

    def log_probs(self, tokens: torch.Tensor, codebooks: range, polish: bool = False):
        hidden = self.llm(
            inputs_embeds=self.model.audio_embeddings(
                tokens + self.offsets[:, None]
            ).sum(0)[None],
            attention_mask=self.masks[polish],
            position_ids=self.positions,
            past_key_values=self.cache,
        ).last_hidden_state[0]
        self.cache.crop(-self.frames)
        weight = self.heads[codebooks.start * CODES : codebooks.stop * CODES]
        logits = (hidden @ weight.T).float().view(self.frames, len(codebooks), CODES)
        return logits.transpose(0, 1).log_softmax(-1)[..., :MASK]


def masked_tokens(frames: int, seed: int = 0) -> torch.Tensor:
    generator = torch.Generator().manual_seed(seed)
    tokens = torch.randint(0, MASK, (CODEBOOKS, frames), generator=generator)
    tokens[:, ::3] = MASK
    tokens[2:] = MASK
    return tokens


def sine(samples: int, rate: int = 24000) -> torch.Tensor:
    t = torch.arange(samples) / rate
    return 0.3 * torch.sin(2 * math.pi * 220 * t) + 0.1 * torch.sin(
        2 * math.pi * 1250 * t
    )


def tiny_codec(seed: int = 0):
    from transformers import HiggsAudioV2TokenizerConfig, HiggsAudioV2TokenizerModel

    config = HiggsAudioV2TokenizerConfig(
        acoustic_model_config={
            "model_type": "dac",
            "encoder_hidden_size": 8,
            "decoder_hidden_size": 32,
            "hidden_size": 16,
            "codebook_dim": 4,
            "downsampling_ratios": [8, 5, 4, 2, 3],
            "upsampling_ratios": [8, 5, 4, 2, 3],
            "hop_length": 960,
            "n_codebooks": 9,
            "codebook_size": 1024,
        },
        semantic_model_config={
            "model_type": "hubert",
            "hidden_size": 32,
            "num_hidden_layers": 2,
            "num_attention_heads": 2,
            "intermediate_size": 64,
            "conv_dim": [16] * 7,
            "num_conv_pos_embeddings": 16,
            "num_conv_pos_embedding_groups": 4,
        },
        codebook_dim=8,
        codebook_size=1024,
        target_bandwidths=[0.5, 1, 1.5, 2],
        downsample_factor=320,
        sample_rate=24000,
        semantic_sample_rate=16000,
    )
    torch.manual_seed(seed)
    codec = HiggsAudioV2TokenizerModel(config).eval()
    with torch.no_grad():
        for quantizer in codec.quantizer.quantizers:
            quantizer.codebook.embed.normal_()
    return codec
