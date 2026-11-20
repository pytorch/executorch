# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Pocket TTS b6369a24, using the streaming building blocks of Phonon."""

from dataclasses import dataclass
from pathlib import Path

import torch
from executorch.examples.models.phonon.model import (
    FlowLMStep,
    FlowNet,
    MimiDecoder,
    MimiDecodeStep,
    Phonon,
    PhononConfig,
    Transformer,
)
from safetensors.torch import load_file
from torch import nn


@dataclass(frozen=True)
class PocketConfig(PhononConfig):
    dim: int = 1024
    num_heads: int = 16
    ffn_dim: int = 4096
    flow_dim: int = 512
    eos_threshold: float = -4.0
    temperature: float = 0.7
    max_context: int = 1024


class TimeVarianceNorm(nn.Module):
    """Pocket's RMSNorm uses sample variance, including Bessel's correction."""

    def __init__(self, dim: int):
        super().__init__()
        self.alpha = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * (self.alpha * torch.rsqrt(x.var(-1, keepdim=True) + 1e-5))


class PocketFlowNet(FlowNet):
    def __init__(self, cfg: PocketConfig):
        super().__init__(cfg.flow_dim, cfg.latent_dim, cfg.flow_depth)
        self.cond_embed = nn.Linear(cfg.dim, cfg.flow_dim)
        for embed in self.time_embed:
            embed.mlp[3] = TimeVarianceNorm(cfg.flow_dim)


class TextConditioner(nn.Module):
    def __init__(self, cfg: PocketConfig):
        super().__init__()
        self.embed = nn.Embedding(cfg.text_bins + 1, cfg.dim)


class PocketFlowLM(nn.Module):
    def __init__(self, cfg: PocketConfig):
        super().__init__()
        self.cfg = cfg
        self.bos_emb = nn.Parameter(torch.zeros(cfg.latent_dim))
        self.register_buffer("emb_mean", torch.zeros(cfg.latent_dim))
        self.register_buffer("emb_std", torch.ones(cfg.latent_dim))
        self.conditioner = TextConditioner(cfg)
        self.flow_net = PocketFlowNet(cfg)
        self.input_linear = nn.Linear(cfg.latent_dim, cfg.dim, bias=False)
        self.out_eos = nn.Linear(cfg.dim, 1)
        self.out_norm = nn.LayerNorm(cfg.dim, eps=1e-5)
        self.transformer = Transformer(cfg.num_layers, cfg.dim, cfg.ffn_dim, False)

    def summed_conditions(self) -> torch.Tensor:
        return self.bos_emb.new_zeros(1, 1, self.cfg.dim)

    def bos_embeds(self) -> torch.Tensor:
        return self.input_linear(self.bos_emb.view(1, 1, -1))


# Encoding a new voice is outside the exported text-to-audio path. Catalog
# voices already contain the projected audio_prompt embeddings.
ENCODER_PREFIXES = (
    "mimi.encoder.",
    "mimi.encoder_transformer.",
    "mimi.downsample.",
)


class PocketTTS(Phonon):
    def __init__(self, cfg: PocketConfig | None = None):
        nn.Module.__init__(self)
        self.cfg = cfg if cfg is not None else PocketConfig()
        self.flow_lm = PocketFlowLM(self.cfg)
        self.mimi = MimiDecoder(self.cfg)
        self.rope_layout = "interleaved"

    @classmethod
    def from_checkpoint(
        cls, checkpoint: str | Path, cfg: PocketConfig | None = None
    ) -> "PocketTTS":
        state = {
            name: tensor.float()
            for name, tensor in load_file(str(checkpoint)).items()
            if not name.startswith(ENCODER_PREFIXES)
            and name != "flow_lm.speaker_proj_weight"
        }
        model = cls(cfg)
        model.load_state_dict(state, strict=True)
        return model.eval().requires_grad_(False).to_half_split_rope()


class PocketMimiDecodeStep(MimiDecodeStep):
    def __init__(self, model: PocketTTS):
        super().__init__(model)
        t, w = model.cfg.upsample_stride, model.cfg.mimi_context
        i, j = torch.arange(t)[:, None], torch.arange(w + t)[None, :]
        # Pocket counts the current position in its context window.
        self.band = torch.where((j > i) & (j <= i + w), 0.0, -1.0e30)


def build_programs(
    model: PocketTTS, attention: str = "custom_sdpa"
) -> dict[str, nn.Module]:
    return {
        "text_embed": model.flow_lm.conditioner.embed,
        "flow_lm": FlowLMStep(model, attention).eval(),
        "mimi_decode": PocketMimiDecodeStep(model).eval(),
    }


def load_voice(path: str | Path, dim: int) -> torch.Tensor:
    tensors = load_file(str(path))
    if set(tensors) != {"audio_prompt"}:
        raise ValueError("Expected a Pocket catalog voice with one audio_prompt tensor")
    voice = tensors["audio_prompt"].float().contiguous()
    if voice.ndim != 3 or voice.shape[0] != 1 or voice.shape[2] != dim:
        raise ValueError(f"Expected audio_prompt shape [1, T, {dim}]")
    if voice.shape[1] == 0 or not torch.isfinite(voice).all():
        raise ValueError("audio_prompt must be nonempty and finite")
    return voice
