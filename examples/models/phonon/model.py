# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""PyTorch definition of Gradium's Phonon text-to-speech model.

Phonon (https://huggingface.co/gradium/phonon-31m-en) builds on Kyutai's
Pocket TTS and the Mimi codec. The modules below follow the official runtime
(xn-ptts, see NOTICE) and their parameter names match the keys of the official
`model.safetensors`, so the checkpoint loads with `strict=True`.

Generation runs four programs that `export_phonon.py` exports as methods of one
`.pte` file (see README.md for the I/O contract):

  text_embed(tokens)        text conditioner lookup, [1, S] -> [1, S, D]
  voice_embed(voice)        voice prompt projection, [1, T, D] -> [1, T, D]
  flow_lm(embeds, input_pos, noise)
                            causal transformer with its KV cache inside the
                            program, then the EOS logit and the one-step
                            flow-matching head for the last position
  mimi_decode(latent, *state)
                            one 80 ms frame of the streaming Mimi decoder; all
                            streaming state is explicit input and output
"""

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, List, Optional, Tuple

import torch
import torch.nn.functional as F
from executorch.examples.models.llama.rope import hf_apply_rotary_emb
from torch import nn

# Checkpoint tensors the official runtime drops: a training-only loss weighting
# network.
TRAINING_ONLY_PREFIXES = ("flow_lm.flow.w_s_t.",)

# Attention implementations for the flow LM KV cache (see FlowLMStep).
ATTENTION_IMPLS = ("custom_sdpa", "sdpa")


@dataclass(frozen=True)
class PhononConfig:
    # Flow LM
    dim: int = 512
    num_heads: int = 8
    num_layers: int = 6
    ffn_dim: int = 2048
    max_period: float = 10000.0
    text_bins: int = 4000
    latent_dim: int = 32
    flow_depth: int = 6
    eos_threshold: float = 4.0
    # Conditioners summed into every generated step
    num_speakers_bins: int = 31
    num_speakers_dim: int = 16
    num_speakers_values: Tuple[str, ...] = ("0", "1", "2")
    duration_dim: int = 128
    duration_scale: float = 1000.0
    duration_max_period: float = 10000.0
    # Mimi decoder
    sample_rate: int = 24000
    frame_rate: float = 12.5
    mimi_dim: int = 512
    n_filters: int = 64
    ratios: Tuple[int, ...] = (6, 5, 4)
    kernel_size: int = 7
    last_kernel_size: int = 3
    residual_kernel_size: int = 3
    compress: int = 2
    mimi_heads: int = 8
    mimi_layers: int = 2
    mimi_ffn_dim: int = 2048
    mimi_context: int = 250
    mimi_max_period: float = 10000.0
    # Export limits
    max_context: int = 512
    max_prefill: int = 128
    # Generation defaults of the official runtime
    temperature: float = 0.3

    @property
    def head_dim(self) -> int:
        return self.dim // self.num_heads

    @property
    def mimi_head_dim(self) -> int:
        return self.mimi_dim // self.mimi_heads

    @property
    def hop_length(self) -> int:
        return math.prod(self.ratios)

    @property
    def upsample_stride(self) -> int:
        stride = self.sample_rate / self.hop_length / self.frame_rate
        if stride != int(stride):
            raise ValueError("sample_rate / prod(ratios) / frame_rate must be whole")
        return int(stride)

    @property
    def frame_samples(self) -> int:
        return self.hop_length * self.upsample_stride

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "PhononConfig":
        """Parse the official `config.json`, rejecting variants this example
        does not implement."""
        lm, mimi = data["flow_lm"], data["mimi"]
        checks = {
            "lsd_decode_steps": (data.get("lsd_decode_steps"), 1),
            "flow_lm.time_rms_norm": (lm.get("time_rms_norm"), "rms"),
            "flow_lm.gelu": (lm.get("gelu"), "erf"),
            "mimi.gelu": (mimi.get("gelu"), "erf"),
            "mimi.channels": (mimi.get("channels"), 1),
            "mimi.n_residual_layers": (mimi.get("n_residual_layers"), 1),
            "flow_lm.lut_dim": (lm.get("lut_dim"), lm["d_model"]),
            "flow_lm.flow_dim": (lm.get("flow_dim"), lm["d_model"]),
            "fuser.sum": (
                data["fuser"]["sum"],
                ["num_speakers", "duration_delta"],
            ),
        }
        for name, (got, want) in checks.items():
            if got != want:
                raise ValueError(f"Unsupported Phonon config: {name}={got!r}")
        conds = {c["name"]: c for c in data["conditioners"]}
        ns, dd = conds["num_speakers"]["lut"], conds["duration_delta"]["continuous"]
        return cls(
            dim=lm["d_model"],
            num_heads=lm["num_heads"],
            num_layers=lm["num_layers"],
            ffn_dim=lm["dim_feedforward"],
            max_period=lm["max_period"],
            text_bins=lm["n_bins"],
            latent_dim=lm["ldim"],
            flow_depth=lm["flow_depth"],
            eos_threshold=data["eos_threshold"],
            num_speakers_bins=ns["n_bins"],
            num_speakers_dim=ns["dim"],
            num_speakers_values=tuple(ns["possible_values"]),
            duration_dim=dd["dim"],
            duration_scale=dd["scale_factor"],
            duration_max_period=dd["max_period"],
            sample_rate=mimi["sample_rate"],
            frame_rate=mimi["frame_rate"],
            mimi_dim=mimi["dimension"],
            n_filters=mimi["n_filters"],
            ratios=tuple(mimi["ratios"]),
            kernel_size=mimi["kernel_size"],
            last_kernel_size=mimi["last_kernel_size"],
            residual_kernel_size=mimi["residual_kernel_size"],
            compress=mimi["compress"],
            mimi_heads=mimi["transformer_num_heads"],
            mimi_layers=mimi["transformer_num_layers"],
            mimi_ffn_dim=mimi["transformer_dim_feedforward"],
            mimi_context=mimi["transformer_context"],
            mimi_max_period=mimi["transformer_max_period"],
        )

    @classmethod
    def from_json(cls, path: str | Path) -> "PhononConfig":
        return cls.from_dict(json.loads(Path(path).read_text()))


# ---------------------------------------------------------------------------
# Building blocks (parameter names follow the checkpoint)
# ---------------------------------------------------------------------------


class Attention(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.in_proj = nn.Linear(dim, 3 * dim, bias=False)
        self.out_proj = nn.Linear(dim, dim, bias=False)


class LayerScale(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.scale = nn.Parameter(torch.full((dim,), 0.01))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.scale * x


class TransformerLayer(nn.Module):
    """Pre-norm transformer layer: attention (computed by the caller) and an
    erf-GELU feed-forward block, with optional layer scale (Mimi)."""

    def __init__(self, dim: int, ffn_dim: int, layer_scale: bool):
        super().__init__()
        self.self_attn = Attention(dim)
        self.norm1 = nn.LayerNorm(dim, eps=1e-5)
        self.norm2 = nn.LayerNorm(dim, eps=1e-5)
        self.linear1 = nn.Linear(dim, ffn_dim, bias=False)
        self.linear2 = nn.Linear(ffn_dim, dim, bias=False)
        self.gelu = nn.GELU()
        if layer_scale:
            self.layer_scale_1 = LayerScale(dim)
            self.layer_scale_2 = LayerScale(dim)
        else:
            self.layer_scale_1 = nn.Identity()
            self.layer_scale_2 = nn.Identity()

    def feed_forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.linear2(self.gelu(self.linear1(self.norm2(x))))
        return x + self.layer_scale_2(h)


class Transformer(nn.Module):
    def __init__(self, n: int, dim: int, ffn_dim: int, layer_scale: bool):
        super().__init__()
        self.layers = nn.ModuleList(
            TransformerLayer(dim, ffn_dim, layer_scale) for _ in range(n)
        )


def rope_inv_freq(head_dim: int, max_period: float) -> torch.Tensor:
    half = head_dim // 2
    i = torch.arange(half, dtype=torch.float32)
    return 1.0 / torch.pow(torch.tensor(max_period, dtype=torch.float32), i / half)


def rope_tables(n: int, head_dim: int, max_period: float):
    """cos/sin [n, head_dim] for rotate-half RoPE at positions 0..n-1."""
    ang = torch.arange(n, dtype=torch.float32)[:, None] * rope_inv_freq(
        head_dim, max_period
    )
    cos, sin = torch.cos(ang), torch.sin(ang)
    return torch.cat([cos, cos], -1), torch.cat([sin, sin], -1)


def interleaved_to_half_split(in_proj: torch.Tensor, num_heads: int) -> torch.Tensor:
    """Reorder the q and k rows of a fused [3D, D] in_proj weight from
    interleaved RoPE pairs (2i, 2i+1) to rotate-half pairs (i, i + head_dim/2).

    Phonon is trained with interleaved RoPE. Applying the same permutation to q
    and k leaves every q.k product unchanged, and rotate-half RoPE on the new
    layout equals interleaved RoPE on the old one, so the model's function is
    unchanged while the graph uses the common rotate-half form that backends
    fuse (for example Vulkan's apply_rotary_emb_hf).
    """
    dim = in_proj.shape[1]
    head_dim = dim // num_heads
    per_head = torch.cat([torch.arange(0, head_dim, 2), torch.arange(1, head_dim, 2)])
    perm = torch.cat([h * head_dim + per_head for h in range(num_heads)])
    q, k, v = in_proj[:dim], in_proj[dim : 2 * dim], in_proj[2 * dim :]
    return torch.cat([q[perm], k[perm], v], 0).contiguous()


# ---------------------------------------------------------------------------
# Flow LM
# ---------------------------------------------------------------------------


class TimeRMSNorm(nn.Module):
    """Uncentred RMS norm with a learnt gain (config time_rms_norm: rms)."""

    def __init__(self, dim: int, eps: float = 1e-5):
        super().__init__()
        self.alpha = nn.Parameter(torch.ones(dim))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * torch.rsqrt((x * x).mean(-1, keepdim=True) + self.eps) * self.alpha


class TimestepEmbedder(nn.Module):
    def __init__(self, dim: int, freq_dim: int = 256):
        super().__init__()
        self.freqs = nn.Parameter(torch.zeros(freq_dim // 2))
        self.mlp = nn.Sequential(
            nn.Linear(freq_dim, dim),
            nn.SiLU(),
            nn.Linear(dim, dim),
            TimeRMSNorm(dim),
        )

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        args = t * self.freqs
        return self.mlp(torch.cat([torch.cos(args), torch.sin(args)], -1))


class AdaLNResBlock(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim
        self.in_ln = nn.LayerNorm(dim, eps=1e-6)
        self.mlp = nn.Sequential(nn.Linear(dim, dim), nn.SiLU(), nn.Linear(dim, dim))
        self.adaLN_modulation = nn.Sequential(nn.SiLU(), nn.Linear(dim, 3 * dim))

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        d = self.dim
        ada = self.adaLN_modulation(y)
        shift, scale, gate = ada[..., :d], ada[..., d : 2 * d], ada[..., 2 * d :]
        h = self.in_ln(x) * (scale + 1.0) + shift
        return x + gate * self.mlp(h)


class FinalLayer(nn.Module):
    def __init__(self, dim: int, out_dim: int):
        super().__init__()
        self.dim = dim
        self.norm_final = nn.LayerNorm(dim, eps=1e-6, elementwise_affine=False)
        self.linear = nn.Linear(dim, out_dim)
        self.adaLN_modulation = nn.Sequential(nn.SiLU(), nn.Linear(dim, 2 * dim))

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        ada = self.adaLN_modulation(y)
        shift, scale = ada[..., : self.dim], ada[..., self.dim :]
        return self.linear(self.norm_final(x) * (scale + 1.0) + shift)


class FlowNet(nn.Module):
    """SimpleMLPAdaLN velocity network of the flow-matching head."""

    def __init__(self, dim: int, latent_dim: int, depth: int):
        super().__init__()
        self.time_embed = nn.ModuleList([TimestepEmbedder(dim), TimestepEmbedder(dim)])
        self.cond_embed = nn.Linear(dim, dim)
        self.input_proj = nn.Linear(latent_dim, dim)
        self.res_blocks = nn.ModuleList(AdaLNResBlock(dim) for _ in range(depth))
        self.final_layer = FinalLayer(dim, latent_dim)

    def lsd_time_condition(self) -> torch.Tensor:
        """Time condition of a single lagrangian self-distillation step from
        s = 0 to t = 1: the mean of the two time embeddings, [1, dim]."""
        s, t = torch.zeros(1, 1), torch.ones(1, 1)
        return (self.time_embed[0](s) + self.time_embed[1](t)) / 2

    def forward(
        self, c: torch.Tensor, x: torch.Tensor, t_cond: torch.Tensor
    ) -> torch.Tensor:
        x = self.input_proj(x)
        y = t_cond + self.cond_embed(c)
        for block in self.res_blocks:
            x = block(x, y)
        return self.final_layer(x, y)


class Conditioner(nn.Module):
    def __init__(
        self,
        dim: int,
        embed: Optional[Tuple[int, int]] = None,
        proj_in: Optional[int] = None,
    ):
        super().__init__()
        # Unused at inference; kept so the checkpoint loads strictly.
        self.learnt_padding = nn.Parameter(torch.zeros(1, 1, dim))
        if embed is not None:
            self.embed = nn.Embedding(*embed)
        if proj_in is not None:
            self.output_proj = nn.Linear(proj_in, dim, bias=False)


class Conditioners(nn.Module):
    def __init__(self, cfg: PhononConfig):
        super().__init__()
        d = cfg.dim
        self.duration_delta = Conditioner(d, proj_in=cfg.duration_dim)
        self.num_speakers = Conditioner(
            d,
            embed=(cfg.num_speakers_bins + 1, cfg.num_speakers_dim),
            proj_in=cfg.num_speakers_dim,
        )
        self.speaker_wavs = Conditioner(d, proj_in=d)
        self.transcript_in_segment = Conditioner(d, embed=(cfg.text_bins + 1, d))


class ConditionProvider(nn.Module):
    def __init__(self, cfg: PhononConfig):
        super().__init__()
        self.conditioners = Conditioners(cfg)


def sinusoidal_embedding(position: float, dim: int, max_period: float) -> torch.Tensor:
    half = dim // 2
    i = torch.arange(half, dtype=torch.float32)
    phase = position / torch.pow(
        torch.tensor(max_period, dtype=torch.float32), i / (half - 1)
    )
    return torch.cat([torch.cos(phase), torch.sin(phase)])


class FlowLM(nn.Module):
    def __init__(self, cfg: PhononConfig):
        super().__init__()
        self.cfg = cfg
        d, ld = cfg.dim, cfg.latent_dim
        self.bos_emb = nn.Parameter(torch.zeros(ld))
        self.emb_mean = nn.Parameter(torch.zeros(ld))
        self.emb_std = nn.Parameter(torch.ones(ld))
        self.condition_provider = ConditionProvider(cfg)
        self.flow_net = FlowNet(d, ld, cfg.flow_depth)
        self.input_linear = nn.Linear(ld, d, bias=False)
        self.out_eos = nn.Linear(d, 1)
        self.out_norm = nn.LayerNorm(d, eps=1e-5)
        self.transformer = Transformer(cfg.num_layers, d, cfg.ffn_dim, False)

    @property
    def conditioners(self) -> Conditioners:
        return self.condition_provider.conditioners

    def summed_conditions(
        self, num_speakers: str = "1", duration_delta: float = 0.0
    ) -> torch.Tensor:
        """The conditions added to every generated step's input, [1, 1, D].
        Defaults are the official runtime's."""
        c, cfg = self.conditioners, self.cfg
        idx = cfg.num_speakers_values.index(num_speakers)
        ns = c.num_speakers.output_proj(c.num_speakers.embed.weight[idx : idx + 1])
        dd = c.duration_delta.output_proj(
            sinusoidal_embedding(
                cfg.duration_scale * duration_delta,
                cfg.duration_dim,
                cfg.duration_max_period,
            )[None]
        )
        return (ns + dd).view(1, 1, cfg.dim)

    def bos_embeds(self) -> torch.Tensor:
        """Input embedding of the first generated step, [1, 1, D]."""
        bos = self.input_linear(self.bos_emb.view(1, 1, -1))
        return bos + self.summed_conditions()


# ---------------------------------------------------------------------------
# Mimi decoder
# ---------------------------------------------------------------------------


class StreamingConv1d(nn.Module):
    """Causal stride-1 conv; the last kernel_size - 1 input frames carry over."""

    def __init__(self, cin: int, cout: int, kernel_size: int):
        super().__init__()
        self.conv = nn.Conv1d(cin, cout, kernel_size)

    @property
    def state_len(self) -> int:
        return self.conv.kernel_size[0] - 1

    def forward(
        self, x: torch.Tensor, prev: Optional[torch.Tensor]
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        if prev is None:
            return self.conv(x), None
        xx = torch.cat([prev, x], 2)
        return self.conv(xx), xx[:, :, xx.shape[2] - prev.shape[2] :]


class StreamingConvTranspose1d(nn.Module):
    """Transposed conv; the overlapping kernel_size - stride outputs are held
    back (without bias) and added to the next frame's head."""

    def __init__(
        self,
        cin: int,
        cout: int,
        kernel_size: int,
        stride: int,
        groups: int = 1,
        bias: bool = True,
    ):
        super().__init__()
        self.convtr = nn.ConvTranspose1d(
            cin, cout, kernel_size, stride=stride, groups=groups, bias=bias
        )

    @property
    def state_len(self) -> int:
        return self.convtr.kernel_size[0] - self.convtr.stride[0]

    def forward(
        self, x: torch.Tensor, partial: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        y = self.convtr(x)
        pt, n = partial.shape[2], y.shape[2]
        out = torch.cat([y[:, :, :pt] + partial, y[:, :, pt : n - pt]], 2)
        tail = y[:, :, n - pt :]
        bias = self.convtr.bias
        if bias is not None:
            tail = tail - bias.view(1, -1, 1)
        return out, tail


class SEANetResBlock(nn.Module):
    def __init__(self, dim: int, kernel_size: int, compress: int):
        super().__init__()
        hidden = dim // compress
        self.block = nn.ModuleList(
            [
                nn.ELU(),
                StreamingConv1d(dim, hidden, kernel_size),
                nn.ELU(),
                StreamingConv1d(hidden, dim, 1),
            ]
        )

    def forward(self, x: torch.Tensor, prev: torch.Tensor):
        h, prev = self.block[1](self.block[0](x), prev)
        h, _ = self.block[3](self.block[2](h), None)
        return x + h, prev


class SEANetDecoder(nn.Module):
    """Layout of `decoder.model`: conv, then (ELU, transposed conv, residual
    block) per ratio, then ELU and the output conv."""

    def __init__(self, cfg: PhononConfig):
        super().__init__()
        mult = 2 ** len(cfg.ratios)
        layers: List[nn.Module] = [
            StreamingConv1d(cfg.mimi_dim, mult * cfg.n_filters, cfg.kernel_size)
        ]
        for ratio in cfg.ratios:
            cin = mult * cfg.n_filters
            layers += [
                nn.ELU(),
                StreamingConvTranspose1d(cin, cin // 2, 2 * ratio, ratio),
                SEANetResBlock(cin // 2, cfg.residual_kernel_size, cfg.compress),
            ]
            mult //= 2
        layers += [nn.ELU(), StreamingConv1d(cfg.n_filters, 1, cfg.last_kernel_size)]
        self.model = nn.ModuleList(layers)


class MimiTransformer(nn.Module):
    def __init__(self, cfg: PhononConfig):
        super().__init__()
        self.transformer = Transformer(
            cfg.mimi_layers, cfg.mimi_dim, cfg.mimi_ffn_dim, True
        )


class Upsample(nn.Module):
    """Depthwise transposed conv from the codec frame rate to the transformer
    rate."""

    def __init__(self, dim: int, stride: int):
        super().__init__()
        self.convtr = StreamingConvTranspose1d(
            dim, dim, 2 * stride, stride, groups=dim, bias=False
        )

    def forward(self, x: torch.Tensor, partial: torch.Tensor):
        return self.convtr(x, partial)


class Quantizer(nn.Module):
    def __init__(self, cfg: PhononConfig):
        super().__init__()
        self.output_proj = nn.Conv1d(cfg.latent_dim, cfg.mimi_dim, 1, bias=False)


class MimiDecoder(nn.Module):
    def __init__(self, cfg: PhononConfig):
        super().__init__()
        s = cfg.upsample_stride
        self.quantizer = Quantizer(cfg)
        self.upsample = Upsample(cfg.mimi_dim, s)
        self.decoder_transformer = MimiTransformer(cfg)
        self.decoder = SEANetDecoder(cfg)


class Phonon(nn.Module):
    def __init__(self, cfg: Optional[PhononConfig] = None):
        super().__init__()
        cfg = cfg if cfg is not None else PhononConfig()
        self.cfg = cfg
        self.flow_lm = FlowLM(cfg)
        self.mimi = MimiDecoder(cfg)
        # Layout of the q/k rows of every in_proj (see interleaved_to_half_split).
        self.rope_layout = "interleaved"

    @classmethod
    def from_checkpoint(
        cls, checkpoint: str | Path, cfg: Optional[PhononConfig] = None
    ) -> "Phonon":
        """Load the official `model.safetensors` (bf16 flow LM, fp32 Mimi) in
        fp32. Every tensor is loaded with strict=True except the training-only
        tensors in TRAINING_ONLY_PREFIXES, which the official runtime drops too.
        """
        from safetensors.torch import load_file

        checkpoint = Path(checkpoint)
        if cfg is None:
            cfg = PhononConfig.from_json(checkpoint.parent / "config.json")
        state = {
            k: v.float()
            for k, v in load_file(str(checkpoint)).items()
            if not k.startswith(TRAINING_ONLY_PREFIXES)
        }
        model = cls(cfg)
        model.load_state_dict(state, strict=True)
        return model.eval().requires_grad_(False).to_half_split_rope()

    def to_half_split_rope(self) -> "Phonon":
        """Permute every in_proj once for rotate-half RoPE (exact, in place)."""
        if self.rope_layout == "half_split":
            return self
        blocks = [
            (self.flow_lm.transformer, self.cfg.num_heads),
            (self.mimi.decoder_transformer.transformer, self.cfg.mimi_heads),
        ]
        with torch.no_grad():
            for transformer, heads in blocks:
                for layer in transformer.layers:
                    w = layer.self_attn.in_proj.weight
                    w.copy_(interleaved_to_half_split(w, heads))
        self.rope_layout = "half_split"
        return self


def load_voice(path: str | Path) -> torch.Tensor:
    """Speaker latents [1, T, D] from a Phonon voice `.safetensors` file
    (stored as speaker_wavs [1, D, T])."""
    from safetensors.torch import load_file

    tensors = load_file(str(path))
    if list(tensors) != ["speaker_wavs"]:
        raise ValueError(f"{path}: expected one tensor named speaker_wavs")
    return tensors["speaker_wavs"].float().transpose(1, 2).contiguous()


# ---------------------------------------------------------------------------
# Export programs
# ---------------------------------------------------------------------------


class TextEmbedding(nn.Module):
    """tokens i64[1, S] -> embeds f32[1, S, D] (text conditioner lookup)."""

    def __init__(self, model: Phonon):
        super().__init__()
        self.embed = model.flow_lm.conditioners.transcript_in_segment.embed

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        return self.embed(tokens)


class VoiceEmbedding(nn.Module):
    """voice f32[1, T, D] -> embeds f32[1, T, D] (voice prompt projection).

    Taking the speaker latents time-major keeps the dynamic length out of any
    transpose, which not every delegate resizes."""

    def __init__(self, model: Phonon):
        super().__init__()
        self.proj = model.flow_lm.conditioners.speaker_wavs.output_proj

    def forward(self, voice: torch.Tensor) -> torch.Tensor:
        d = voice.shape[2]
        return self.proj(voice.reshape(-1, d)).reshape(1, -1, d)


class FlowLMStep(nn.Module):
    """forward(embeds f32[1, S, D], input_pos i64[1], noise f32[1, latent_dim])
    -> latent f32[1, latent_dim], eos_logit f32[1, 1], next_embeds f32[1, 1, D]

    Writes KV cache positions [input_pos, input_pos + S) and attends causally
    over [0, input_pos + S). The head runs on the last position:
    eos_logit = out_eos(h), latent = noise + flow_net(h, noise) (one flow step)
    and next_embeds = input_linear(latent) + conditions, the next step's input.
    Prompt calls (voice, then text) ignore the outputs.

    attention selects how the KV cache is stored and attended:
      custom_sdpa  torch.ops.llama.sdpa_with_kv_cache (ExecuTorch CPU custom op;
                   Vulkan and WebGPU lower it to a fused kernel)
      sdpa         index_copy_ into the cache + F.scaled_dot_product_attention
                   with an explicit causal mask (AOTInductor, MLX)
    """

    def __init__(self, model: Phonon, attention: str = "custom_sdpa"):
        super().__init__()
        if attention not in ATTENTION_IMPLS:
            raise ValueError(f"attention must be one of {ATTENTION_IMPLS}")
        if model.rope_layout != "half_split":
            raise ValueError("Call Phonon.to_half_split_rope() first")
        cfg, lm = model.cfg, model.flow_lm
        self.cfg, self.attention = cfg, attention
        self.layers = lm.transformer.layers
        self.out_norm, self.out_eos = lm.out_norm, lm.out_eos
        self.input_linear, self.flow_net = lm.input_linear, lm.flow_net
        with torch.no_grad():
            self.register_buffer("cond", lm.summed_conditions().clone())
            self.register_buffer("t_cond", lm.flow_net.lsd_time_condition().clone())
        cos, sin = rope_tables(cfg.max_context, cfg.head_dim, cfg.max_period)
        self.register_buffer("rope_cos", cos)
        self.register_buffer("rope_sin", sin)
        if attention == "custom_sdpa":
            from executorch.extension.llm.custom_ops import custom_ops  # noqa: F401

            shape = (1, cfg.max_context, cfg.num_heads, cfg.head_dim)
        else:
            shape = (1, cfg.num_heads, cfg.max_context, cfg.head_dim)
        for i in range(cfg.num_layers):
            self.register_buffer(f"k_cache_{i}", torch.zeros(shape), persistent=False)
            self.register_buffer(f"v_cache_{i}", torch.zeros(shape), persistent=False)

    def reset(self) -> None:
        for name, buf in self.named_buffers():
            if "_cache_" in name:
                buf.zero_()

    def _attend_custom(self, i, q, k, v, input_pos):
        cfg = self.cfg
        seq = q.shape[1]
        start = input_pos[0].item()
        torch._check(start >= 0)
        torch._check(start + seq <= cfg.max_context)
        cos = self.rope_cos.narrow(0, start, seq)
        sin = self.rope_sin.narrow(0, start, seq)
        q, k = hf_apply_rotary_emb(q, k, cos, sin)
        return torch.ops.llama.sdpa_with_kv_cache(
            q,
            k,
            v,
            getattr(self, f"k_cache_{i}"),
            getattr(self, f"v_cache_{i}"),
            start,
            seq,
            None,
            0.0,
            True,
            None,
        )

    def _attend_sdpa(self, i, q, k, v, input_pos):
        cfg = self.cfg
        seq = q.shape[1]
        pos = input_pos + torch.arange(seq, dtype=torch.long, device=input_pos.device)
        q, k = hf_apply_rotary_emb(q, k, self.rope_cos[pos], self.rope_sin[pos])
        k_cache, v_cache = getattr(self, f"k_cache_{i}"), getattr(self, f"v_cache_{i}")
        k_cache.index_copy_(2, pos, k.transpose(1, 2))
        v_cache.index_copy_(2, pos, v.transpose(1, 2))
        keys = torch.arange(cfg.max_context, dtype=torch.long, device=pos.device)
        mask = torch.where(keys[None, :] <= pos[:, None], 0.0, float("-inf"))
        o = F.scaled_dot_product_attention(
            q.transpose(1, 2), k_cache, v_cache, attn_mask=mask
        )
        return o.transpose(1, 2)

    def forward(
        self, embeds: torch.Tensor, input_pos: torch.Tensor, noise: torch.Tensor
    ):
        cfg = self.cfg
        seq = embeds.shape[1]
        heads, hd, d = cfg.num_heads, cfg.head_dim, cfg.dim
        attend = (
            self._attend_custom
            if self.attention == "custom_sdpa"
            else self._attend_sdpa
        )
        x = embeds
        for i, layer in enumerate(self.layers):
            qkv = layer.self_attn.in_proj(layer.norm1(x))
            q = qkv[..., :d].reshape(1, seq, heads, hd)
            k = qkv[..., d : 2 * d].reshape(1, seq, heads, hd)
            v = qkv[..., 2 * d :].reshape(1, seq, heads, hd)
            o = attend(i, q, k, v, input_pos)
            x = x + layer.self_attn.out_proj(o.reshape(1, seq, d))
            x = layer.feed_forward(x)
        h = self.out_norm(x[:, -1, :])
        eos_logit = self.out_eos(h)
        latent = noise + self.flow_net(h, noise, self.t_cond)
        next_embeds = self.input_linear(latent).reshape(1, 1, d) + self.cond
        return latent, eos_logit, next_embeds


def mimi_state_spec(cfg: PhononConfig) -> List[Tuple[str, Tuple[int, ...]]]:
    """Names and shapes of the streaming Mimi decoder state, in call order."""
    h, w, hd = cfg.mimi_heads, cfg.mimi_context, cfg.mimi_head_dim
    spec: List[Tuple[str, Tuple[int, ...]]] = [
        ("upsample_partial", (1, cfg.mimi_dim, cfg.upsample_stride))
    ]
    for i in range(cfg.mimi_layers):
        spec += [(f"k{i}", (1, h, w, hd)), (f"v{i}", (1, h, w, hd))]
    spec += [("kv_seen", (1, 1, 1, w)), ("rope_offset", (1,))]
    mult = 2 ** len(cfg.ratios)
    spec.append(("init_conv_prev", (1, cfg.mimi_dim, cfg.kernel_size - 1)))
    for j, ratio in enumerate(cfg.ratios, 1):
        c = mult * cfg.n_filters // 2
        spec += [
            (f"up{j}_partial", (1, c, ratio)),
            (f"res{j}_prev", (1, c, cfg.residual_kernel_size - 1)),
        ]
        mult //= 2
    spec.append(("final_conv_prev", (1, cfg.n_filters, cfg.last_kernel_size - 1)))
    return spec


def mimi_zero_state(cfg: PhononConfig) -> List[torch.Tensor]:
    return [torch.zeros(shape) for _, shape in mimi_state_spec(cfg)]


class MimiDecodeStep(nn.Module):
    """forward(latent f32[1, 1, latent_dim], *state) -> pcm f32[1, 1, frame],
    *new_state, with state as listed by mimi_state_spec (zeros to start).

    One latent frame per call: denormalize, project, upsample x16, a 2-layer
    transformer over a sliding window of the last mimi_context positions, and
    the SEANet decoder. Every piece of streaming state (conv tails, transposed
    conv partial sums, the window's keys and values, which window slots are
    filled, and the RoPE offset) is an explicit input and output, so no backend
    has to keep mutable state alive between calls.
    """

    def __init__(self, model: Phonon):
        super().__init__()
        if model.rope_layout != "half_split":
            raise ValueError("Call Phonon.to_half_split_rope() first")
        cfg, mimi = model.cfg, model.mimi
        self.cfg = cfg
        self.emb_std = model.flow_lm.emb_std
        self.emb_mean = model.flow_lm.emb_mean
        self.quantizer = mimi.quantizer
        self.upsample = mimi.upsample
        # Not `layers`: the CUDA backend shares weights between the methods of a
        # program by name, and FlowLMStep.layers holds other weights.
        self.transformer_layers = mimi.decoder_transformer.transformer.layers
        self.decoder = mimi.decoder.model
        t, w = cfg.upsample_stride, cfg.mimi_context
        self.register_buffer(
            "inv_freq", rope_inv_freq(cfg.mimi_head_dim, cfg.mimi_max_period)[None]
        )
        self.register_buffer("steps", torch.arange(t, dtype=torch.float32)[:, None])
        i, j = torch.arange(t)[:, None], torch.arange(w + t)[None, :]
        # Query i sits at window index w + i and sees keys i..w + i.
        band = torch.where((j >= i) & (j <= i + w), 0.0, -1.0e30)
        self.register_buffer("band", band)
        self.register_buffer("ones_seen", torch.ones(1, 1, 1, t))

    def _attention(self, layer, x, k_state, v_state, seen_all, cos, sin):
        cfg = self.cfg
        t, w, h, hd, d = (
            cfg.upsample_stride,
            cfg.mimi_context,
            cfg.mimi_heads,
            cfg.mimi_head_dim,
            cfg.mimi_dim,
        )
        qkv = layer.self_attn.in_proj(layer.norm1(x))  # [t, 3d]
        # [1, t, h, hd] is the layout of the rotate-half RoPE pattern.
        q = qkv[:, :d].reshape(1, t, h, hd)
        k = qkv[:, d : 2 * d].reshape(1, t, h, hd)
        v = qkv[:, 2 * d :].reshape(t, h, hd)
        q, k = hf_apply_rotary_emb(q, k, cos, sin)
        q, k = q.reshape(t, h, hd), k.reshape(t, h, hd)
        keys = torch.cat([k_state.reshape(h, w, hd), k.permute(1, 0, 2)], 1)
        values = torch.cat([v_state.reshape(h, w, hd), v.permute(1, 0, 2)], 1)
        mask = self.band + (seen_all.reshape(1, w + t) - 1.0) * 1.0e30
        scores = torch.bmm(q.permute(1, 0, 2), keys.transpose(1, 2)) * (hd**-0.5)
        probs = torch.softmax(scores + mask, -1)
        o = torch.bmm(probs, values).permute(1, 0, 2).reshape(t, d)
        x = x + layer.layer_scale_1(layer.self_attn.out_proj(o))
        x = layer.feed_forward(x)
        return (
            x,
            keys[:, t:].reshape(1, h, w, hd),
            values[:, t:].reshape(1, h, w, hd),
        )

    def forward(self, latent: torch.Tensor, *state: torch.Tensor):
        cfg = self.cfg
        n_layers, t = cfg.mimi_layers, cfg.upsample_stride
        state = list(state)
        up_partial = state[0]
        kv = state[1 : 1 + 2 * n_layers]
        seen, offset = state[1 + 2 * n_layers], state[2 + 2 * n_layers]
        convs = state[3 + 2 * n_layers :]

        x = latent * self.emb_std + self.emb_mean  # [1, 1, latent]
        x = self.quantizer.output_proj(x.transpose(1, 2))  # [1, d, 1]
        x, up_partial = self.upsample(x, up_partial)  # [1, d, t]

        ang = (self.steps + offset) * self.inv_freq  # [t, hd / 2]
        cos = torch.cat([torch.cos(ang), torch.cos(ang)], -1)
        sin = torch.cat([torch.sin(ang), torch.sin(ang)], -1)
        seen_all = torch.cat([seen, self.ones_seen], 3)
        x = x.reshape(cfg.mimi_dim, t).transpose(0, 1)  # [t, d]
        new_kv = []
        for i, layer in enumerate(self.transformer_layers):
            x, k, v = self._attention(
                layer, x, kv[2 * i], kv[2 * i + 1], seen_all, cos, sin
            )
            new_kv += [k, v]
        x = x.transpose(0, 1).reshape(1, cfg.mimi_dim, t)

        m = self.decoder
        new_convs = []
        x, c = m[0](x, convs[0])
        new_convs.append(c)
        for j in range(len(cfg.ratios)):
            elu, up, res = m[1 + 3 * j], m[2 + 3 * j], m[3 + 3 * j]
            x, p = up(elu(x), convs[1 + 2 * j])
            x, r = res(x, convs[2 + 2 * j])
            new_convs += [p, r]
        pcm, c = m[-1](m[-2](x), convs[-1])
        new_convs.append(c)
        return (
            pcm,
            up_partial,
            *new_kv,
            seen_all[..., t:],
            offset + float(t),
            *new_convs,
        )


def example_inputs(cfg: PhononConfig) -> dict[str, Tuple[torch.Tensor, ...]]:
    g = torch.Generator().manual_seed(0)
    seq = 5
    return {
        "text_embed": (torch.zeros(1, seq, dtype=torch.long),),
        "voice_embed": (torch.randn(1, seq, cfg.dim, generator=g),),
        "flow_lm": (
            torch.randn(1, seq, cfg.dim, generator=g),
            torch.tensor([0], dtype=torch.long),
            torch.zeros(1, cfg.latent_dim),
        ),
        "mimi_decode": (
            torch.randn(1, 1, cfg.latent_dim, generator=g),
            *mimi_zero_state(cfg),
        ),
    }


def dynamic_shapes(cfg: PhononConfig) -> dict[str, Any]:
    seq = torch.export.Dim("seq", min=1, max=cfg.max_prefill)
    return {
        "text_embed": ({1: seq},),
        "voice_embed": ({1: seq},),
        "flow_lm": ({1: seq}, None, None),
        "mimi_decode": None,
    }


def build_programs(
    model: Phonon, attention: str = "custom_sdpa"
) -> dict[str, nn.Module]:
    return {
        "text_embed": TextEmbedding(model).eval(),
        "voice_embed": VoiceEmbedding(model).eval(),
        "flow_lm": FlowLMStep(model, attention).eval(),
        "mimi_decode": MimiDecodeStep(model).eval(),
    }


def program_metadata(model: Phonon) -> dict[str, Any]:
    """Constant methods stored in the .pte next to the four programs."""
    cfg = model.cfg
    with torch.no_grad():
        bos = model.flow_lm.bos_embeds().contiguous()
    return {
        "format_version": 1,
        "sample_rate": cfg.sample_rate,
        "frame_samples": cfg.frame_samples,
        "frame_rate": cfg.frame_rate,
        "dim": cfg.dim,
        "latent_dim": cfg.latent_dim,
        "max_context": cfg.max_context,
        "max_prefill": cfg.max_prefill,
        "eos_threshold": cfg.eos_threshold,
        "temperature": cfg.temperature,
        "num_mimi_state": len(mimi_state_spec(cfg)),
        "bos_embeds": bos,
    }


def io_contract(cfg: PhononConfig) -> dict[str, Any]:
    """Inputs and outputs of every program; "S" is the dynamic sequence length
    (1 <= S <= max_prefill). Identical for every backend."""
    d, ld = cfg.dim, cfg.latent_dim
    state = [
        {"name": name, "shape": list(shape), "dtype": "float32"}
        for name, shape in mimi_state_spec(cfg)
    ]

    def t(name, shape, dtype="float32"):
        return {"name": name, "shape": shape, "dtype": dtype}

    return {
        "text_embed": {
            "inputs": [t("tokens", [1, "S"], "int64")],
            "outputs": [t("embeds", [1, "S", d])],
        },
        "voice_embed": {
            "inputs": [t("voice", [1, "S", d])],
            "outputs": [t("embeds", [1, "S", d])],
        },
        "flow_lm": {
            "inputs": [
                t("embeds", [1, "S", d]),
                t("input_pos", [1], "int64"),
                t("noise", [1, ld]),
            ],
            "outputs": [
                t("latent", [1, ld]),
                t("eos_logit", [1, 1]),
                t("next_embeds", [1, 1, d]),
            ],
        },
        "mimi_decode": {
            "inputs": [t("latent", [1, 1, ld])] + state,
            "outputs": [t("pcm", [1, 1, cfg.frame_samples])] + state,
        },
    }
