# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Exportable halves of the Higgs audio codec (`HiggsAudioV2TokenizerModel`).

Reverb speaks in the codec's first 8 codebooks at 25 frames per second, 960
samples of 24 kHz audio per frame. The decoder turns codes into a waveform; the
encoder turns a reference recording into codes for voice cloning.
"""

import math

import torch
import torch.nn.functional as F
from torch import nn
from torch.nn.utils import parametrize

from .backbone import sum_codebooks
from .config import CODEBOOKS

CODEBOOK_SIZE = 1024


class ELU(nn.Module):
    """`nn.ELU(alpha=1)` from ops the delegates lower: expm1, where and clamp."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.where(x > 0, x, torch.special.expm1(torch.clamp(x, max=0.0)))


def _replace_elu(module: nn.Module) -> None:
    for name, child in module.named_children():
        if isinstance(child, nn.ELU):
            if child.alpha != 1.0:
                raise ValueError("only ELU with alpha=1 is supported")
            setattr(module, name, ELU())
        else:
            _replace_elu(child)


def _nearest_code(quantizer, residual: torch.Tensor) -> torch.Tensor:
    """`quantizer.encode` with argmax in place of max(dim).indices: [1, C, T] -> [T]."""
    hidden = quantizer.project_in(residual[0].transpose(0, 1))  # [T, D]
    embed = quantizer.codebook.embed  # [1024, D]
    distance = -(
        hidden.pow(2).sum(1, keepdim=True)
        - 2 * hidden @ embed.T
        + embed.pow(2).sum(1)[None]
    )
    return distance.argmax(-1)


def _codebook_tables(codec) -> torch.Tensor:
    """Each codebook's entries after its output projection: [8 * 1024, hidden]."""
    tables = []
    for quantizer in codec.quantizer.quantizers[:CODEBOOKS]:
        tables.append(quantizer.project_out(quantizer.codebook.embed))
    return torch.cat(tables)


class CodecDecoder(nn.Module):
    """Codes [8, F] int64 -> waveform [F * 960] fp32 at 24 kHz.

    The residual quantizer sums one projected codebook entry per codebook and
    `fc2` maps that sum to the acoustic decoder's width. Both are linear, so
    `fc2` is folded into the tables and decoding starts with one gather.
    """

    def __init__(self, codec) -> None:
        super().__init__()
        if codec.config.num_quantizers < CODEBOOKS:
            raise ValueError(f"the codec must have at least {CODEBOOKS} codebooks")
        with torch.no_grad():
            tables = _codebook_tables(codec) @ codec.fc2.weight.T
        self.embeddings = nn.Embedding.from_pretrained(tables, freeze=True)
        self.bias = nn.Parameter(codec.fc2.bias.detach().clone(), requires_grad=False)
        self.decoder = codec.acoustic_decoder
        offsets = CODEBOOK_SIZE * torch.arange(CODEBOOKS, dtype=torch.int64)
        self.register_buffer("offsets", offsets[:, None])

    def forward(self, codes: torch.Tensor) -> torch.Tensor:
        hidden = sum_codebooks(self.embeddings(codes + self.offsets)) + self.bias
        return self.decoder(hidden.transpose(0, 1)[None]).reshape(-1).float()


class CodecEncoder(nn.Module):
    """Waveform [N] fp32 at 24 kHz, N a multiple of 960 -> codes [8, N / 960] int64.

    Follows `HiggsAudioV2TokenizerModel.encode`: HuBERT features of the 16 kHz
    audio (every hidden state averaged, every second frame kept) next to the
    acoustic encoder's features, quantized by the first 8 residual codebooks.
    The 24 -> 16 kHz resampling is torchaudio's default sinc filter with a
    precomputed kernel.
    """

    def __init__(self, codec) -> None:
        super().__init__()
        from torchaudio.functional.functional import _get_sinc_resample_kernel

        config = codec.config
        if config.sample_rate % config.hop_length:
            raise ValueError("the codec's hop must divide its sample rate")
        self.hop_length = config.hop_length
        self.semantic_stride = config.semantic_downsample_factor
        gcd = math.gcd(config.sample_rate, config.semantic_sample_rate)
        self.resample_from = config.sample_rate // gcd
        self.resample_to = config.semantic_sample_rate // gcd
        kernel, self.resample_width = _get_sinc_resample_kernel(
            config.sample_rate, config.semantic_sample_rate, gcd
        )
        self.register_buffer("resample_kernel", kernel)

        self.semantic_model = codec.semantic_model
        for module in self.semantic_model.modules():
            if parametrize.is_parametrized(module):
                for name in list(module.parametrizations):
                    parametrize.remove_parametrizations(module, name)
        self.encoder_semantic = codec.encoder_semantic
        _replace_elu(self.encoder_semantic)
        self.acoustic_encoder = codec.acoustic_encoder
        self.fc = codec.fc
        self.quantizers = codec.quantizer.quantizers[:CODEBOOKS]

    def _resample(self, wav: torch.Tensor) -> torch.Tensor:
        """[1, N] at the codec rate -> [1, N * to / from] at the semantic rate."""
        length = wav.shape[-1]
        width = self.resample_width
        padded = F.pad(wav, (width, width + self.resample_from))
        out = F.conv1d(padded[:, None], self.resample_kernel, stride=self.resample_from)
        out = out.transpose(1, 2).reshape(1, -1)
        return out[:, : (self.resample_to * length) // self.resample_from]

    def _semantic_features(self, wav: torch.Tensor) -> torch.Tensor:
        audio = F.pad(self._resample(wav), (160, 160))
        hidden_states = self.semantic_model(
            audio, output_hidden_states=True
        ).hidden_states
        features = hidden_states[0]
        for state in hidden_states[1:]:
            features = features + state
        features = features / len(hidden_states)
        return features[:, :: self.semantic_stride]  # [1, frames, 768]

    def forward(self, wav: torch.Tensor) -> torch.Tensor:
        wav = wav.to(self.fc.weight.dtype)[None]
        semantic = self.encoder_semantic(self._semantic_features(wav).transpose(1, 2))
        acoustic = self.acoustic_encoder(wav[:, None])
        embeddings = torch.cat([acoustic, semantic], dim=1)
        residual = self.fc(embeddings.transpose(1, 2)).transpose(1, 2)  # [1, C, frames]
        codes = []
        for quantizer in self.quantizers:
            indices = _nearest_code(quantizer, residual)  # [frames]
            residual = residual - quantizer.decode(indices[None])
            codes.append(indices)
        return torch.stack(codes)
