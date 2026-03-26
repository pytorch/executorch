# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Audio tokenizer decoder for Voxtral TTS.

Reimplements the vLLM VoxtralTTSAudioTokenizer decoder path for ExecuTorch export:
  - No einops (replaced with explicit permute/reshape)
  - No flash_attn (manual attention with ALiBi + sliding window)
  - weight_norm folded before export
  - Static input shapes (chunked processing)

Decoder architecture:
  codes (1, K, T) → quantizer.decode → (1, D_latent, T)
    → CausalConv1d (latent→dim)
    → [Transformer(n_layers, window) → CausalConvTranspose1d(dim, dim)] × N_stages
    → Transformer(n_layers, window)
    → CausalConv1d output_proj (dim→patch_size)
    → reshape → waveform (1, 1, T * total_upsample * patch_size)
"""

import math
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

from executorch.examples.models.voxtral_tts.model import RMSNorm


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------


@dataclass
class AudioTokenizerArgs:
    channels: int = 1
    sampling_rate: int = 24000
    pretransform_patch_size: int = 240
    patch_proj_kernel_size: int = 7
    semantic_codebook_size: int = 8192
    semantic_dim: int = 256
    acoustic_codebook_size: int = 21
    acoustic_dim: int = 36
    conv_weight_norm: bool = True
    causal: bool = True
    attn_sliding_window_size: int = 16
    half_attn_window_upon_downsampling: bool = True
    dim: int = 1024
    hidden_dim: int = 4096
    head_dim: int = 128
    n_heads: int = 8
    n_kv_heads: int = 8
    qk_norm_eps: float = 1e-6
    qk_norm: bool = True
    use_biases: bool = False
    norm_eps: float = 1e-2
    layer_scale: bool = True
    layer_scale_init: float | None = None
    # Decoder arch (comma-separated in config, parsed here)
    decoder_transformer_lengths: tuple[int, ...] = (2, 2, 2, 2)
    decoder_convs_kernels: tuple[int, ...] = (3, 4, 4, 4)
    decoder_convs_strides: tuple[int, ...] = (1, 2, 2, 2)
    # Encoder arch (for computing initial decoder window size)
    encoder_convs_strides: tuple[int, ...] = (2, 2, 2, 1)

    @staticmethod
    def from_dict(d: dict) -> "AudioTokenizerArgs":
        def _parse_ints(s):
            return tuple(int(x) for x in s.split(","))

        return AudioTokenizerArgs(
            channels=d.get("channels", 1),
            sampling_rate=d.get("sampling_rate", 24000),
            pretransform_patch_size=d.get("pretransform_patch_size", 240),
            patch_proj_kernel_size=d.get("patch_proj_kernel_size", 7),
            semantic_codebook_size=d.get("semantic_codebook_size", 8192),
            semantic_dim=d.get("semantic_dim", 256),
            acoustic_codebook_size=d.get("acoustic_codebook_size", 21),
            acoustic_dim=d.get("acoustic_dim", 36),
            conv_weight_norm=d.get("conv_weight_norm", True),
            causal=d.get("causal", True),
            attn_sliding_window_size=d.get("attn_sliding_window_size", 16),
            half_attn_window_upon_downsampling=d.get(
                "half_attn_window_upon_downsampling", True
            ),
            dim=d.get("dim", 1024),
            hidden_dim=d.get("hidden_dim", 4096),
            head_dim=d.get("head_dim", 128),
            n_heads=d.get("n_heads", 8),
            n_kv_heads=d.get("n_kv_heads", 8),
            qk_norm_eps=d.get("qk_norm_eps", 1e-6),
            qk_norm=d.get("qk_norm", True),
            use_biases=d.get("use_biases", False),
            norm_eps=d.get("norm_eps", 1e-2),
            layer_scale=d.get("layer_scale", True),
            layer_scale_init=d.get("layer_scale_init", None),
            decoder_transformer_lengths=_parse_ints(
                d.get("decoder_transformer_lengths_str", "2,2,2,2")
            ),
            decoder_convs_kernels=_parse_ints(
                d.get("decoder_convs_kernels_str", "3,4,4,4")
            ),
            decoder_convs_strides=_parse_ints(
                d.get("decoder_convs_strides_str", "1,2,2,2")
            ),
            encoder_convs_strides=_parse_ints(
                d.get("encoder_convs_strides_str", "2,2,2,1")
            ),
        )

    @property
    def decoder_initial_window_size(self) -> int:
        """Window size entering the decoder (after encoder downsampling)."""
        w = self.attn_sliding_window_size
        for s in self.encoder_convs_strides:
            if self.half_attn_window_upon_downsampling and s > 1:
                w = w // 2
        return w


# ---------------------------------------------------------------------------
# Quantizer (decode path only)
# ---------------------------------------------------------------------------


class SemanticCodebook(nn.Module):
    def __init__(self, codebook_size: int, codebook_dim: int):
        super().__init__()
        self.epsilon = 1e-5
        self.register_buffer("cluster_usage", torch.ones(codebook_size))
        self.register_buffer("embedding_sum", torch.zeros(codebook_size, codebook_dim))

    @property
    def embedding(self) -> torch.Tensor:
        return self.embedding_sum / self.cluster_usage.clamp(
            min=self.epsilon
        ).unsqueeze(1)

    def decode(self, codes: torch.Tensor) -> torch.Tensor:
        # codes: (B, 1, T) → (B, semantic_dim, T)
        codes = codes.squeeze(1)  # (B, T)
        quantized = F.embedding(codes, self.embedding)  # (B, T, D)
        return quantized.permute(0, 2, 1)  # (B, D, T)


class AcousticCodebook(nn.Module):
    def __init__(self, codebook_size: int, codebook_dim: int):
        super().__init__()
        self.n_levels = codebook_size
        self.num_codebooks = codebook_dim

    def decode(
        self, codes: torch.Tensor, dtype: torch.dtype = torch.float32
    ) -> torch.Tensor:
        # codes: (B, n_codebooks, T) → (B, n_codebooks, T)
        return (codes.to(dtype) * 2 / (self.n_levels - 1)) - 1


class AudioCodebook(nn.Module):
    def __init__(self, args: AudioTokenizerArgs):
        super().__init__()
        self.semantic = SemanticCodebook(args.semantic_codebook_size, args.semantic_dim)
        self.acoustic = AcousticCodebook(args.acoustic_codebook_size, args.acoustic_dim)
        self.semantic_dim = args.semantic_dim

    def decode(
        self, codes: torch.Tensor, dtype: torch.dtype = torch.float32
    ) -> torch.Tensor:
        # codes: (B, K, T) where K = 1 (semantic) + n_acoustic
        sem_codes = codes[:, :1, :]
        acou_codes = codes[:, 1:, :]
        sem_emb = self.semantic.decode(sem_codes).to(dtype)
        acou_emb = self.acoustic.decode(acou_codes, dtype)
        return torch.cat([sem_emb, acou_emb], dim=1)  # (B, D_latent, T)


# ---------------------------------------------------------------------------
# Conv layers (export-safe, no weight_norm at runtime)
# ---------------------------------------------------------------------------


def _pad1d(
    x: torch.Tensor, paddings: tuple[int, int], mode: str = "constant"
) -> torch.Tensor:
    left, right = paddings
    if mode == "replicate":
        length = x.shape[-1]
        max_pad = max(left, right)
        if length <= max_pad:
            extra = max_pad - length + 1
            x = F.pad(x, (0, extra))
            padded = F.pad(x, paddings, mode, 0.0)
            return padded[..., : padded.shape[-1] - extra]
        return F.pad(x, paddings, mode, 0.0)
    return F.pad(x, paddings, mode, 0.0)


_weight_norm = torch.nn.utils.parametrizations.weight_norm


class CodecCausalConv1d(nn.Module):
    """CausalConv1d for codec. Uses weight_norm to match checkpoint format;
    call remove_parametrizations() after loading to fold for export."""

    def __init__(
        self,
        in_ch: int,
        out_ch: int,
        kernel_size: int,
        stride: int = 1,
        dilation: int = 1,
        pad_mode: str = "replicate",
        use_weight_norm: bool = True,
        use_bias: bool = True,
    ):
        super().__init__()
        conv = nn.Conv1d(
            in_ch, out_ch, kernel_size, stride=stride, dilation=dilation, bias=use_bias
        )
        self.conv = _weight_norm(conv) if use_weight_norm else conv
        self.pad_mode = pad_mode
        self._stride = stride
        self._eff_ks = (kernel_size - 1) * dilation + 1
        self._pad_total = self._eff_ks - stride

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Static padding computation (use integer arithmetic for export)
        n_frames = (x.shape[-1] - self._eff_ks + self._pad_total) // self._stride + 1
        target = (n_frames - 1) * self._stride + (self._eff_ks - self._pad_total)
        extra = target - x.shape[-1]
        x = _pad1d(x, (self._pad_total, extra), mode=self.pad_mode)
        return self.conv(x)


class CodecCausalConvTranspose1d(nn.Module):
    def __init__(
        self,
        in_ch: int,
        out_ch: int,
        kernel_size: int,
        stride: int = 1,
        use_weight_norm: bool = True,
        use_bias: bool = True,
    ):
        super().__init__()
        conv = nn.ConvTranspose1d(
            in_ch, out_ch, kernel_size, stride=stride, bias=use_bias
        )
        self.conv = _weight_norm(conv) if use_weight_norm else conv
        self._kernel_size = kernel_size
        self._stride = stride

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        total_pad = self._kernel_size - self._stride
        out = self.conv(x)
        # Match vLLM: right = ceil(total_pad * trim_ratio), left = total_pad - right
        # With trim_ratio=1.0: right = total_pad, left = 0
        right = total_pad
        left = 0
        return out[..., left : out.shape[-1] - right]


# ---------------------------------------------------------------------------
# Attention (ALiBi + sliding window, export-safe)
# ---------------------------------------------------------------------------


def _get_alibi_slopes(n_heads: int) -> torch.Tensor:
    def _power_of_2(n: int) -> torch.Tensor:
        r = 2.0 ** (-8.0 / n)
        return torch.tensor([r**i for i in range(n)], dtype=torch.float32)

    if math.log2(n_heads).is_integer():
        return _power_of_2(n_heads)
    m = 2 ** math.floor(math.log2(n_heads))
    return torch.cat([_power_of_2(m), _power_of_2(2 * m)[::2][: n_heads - m]])


class CodecAttention(nn.Module):
    """ALiBi attention with causal sliding window. No flash_attn."""

    def __init__(self, args: AudioTokenizerArgs, layer_id: int):
        super().__init__()
        self.n_heads = args.n_heads
        self.n_kv_heads = args.n_kv_heads
        self.head_dim = args.head_dim
        self.n_rep = args.n_heads // args.n_kv_heads
        self.sliding_window = args.attn_sliding_window_size
        self.causal = args.causal

        self.register_buffer(
            "alibi_slopes", _get_alibi_slopes(self.n_heads), persistent=False
        )

        self.wq = nn.Linear(args.dim, args.n_heads * args.head_dim, bias=False)
        self.wk = nn.Linear(args.dim, args.n_kv_heads * args.head_dim, bias=False)
        self.wv = nn.Linear(args.dim, args.n_kv_heads * args.head_dim, bias=False)
        self.wo = nn.Linear(
            args.n_heads * args.head_dim, args.dim, bias=args.use_biases
        )

        if args.qk_norm:
            self.q_norm = RMSNorm(args.n_heads * args.head_dim, args.qk_norm_eps)
            self.k_norm = RMSNorm(args.n_kv_heads * args.head_dim, args.qk_norm_eps)
        self.has_qk_norm = args.qk_norm

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, _ = x.shape
        xq, xk, xv = self.wq(x), self.wk(x), self.wv(x)

        if self.has_qk_norm:
            xq = self.q_norm(xq)
            xk = self.k_norm(xk)

        xq = xq.view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        xk = xk.view(B, T, self.n_kv_heads, self.head_dim).transpose(1, 2)
        xv = xv.view(B, T, self.n_kv_heads, self.head_dim).transpose(1, 2)

        # GQA expansion
        if self.n_rep > 1:
            xk = xk.repeat_interleave(self.n_rep, dim=1)
            xv = xv.repeat_interleave(self.n_rep, dim=1)

        # Build ALiBi + causal + sliding window mask
        positions = torch.arange(T, device=x.device)
        rel_pos = positions.unsqueeze(0) - positions.unsqueeze(1)  # (T, T)
        slopes = self.alibi_slopes.to(dtype=x.dtype, device=x.device)
        attn_bias = slopes.view(self.n_heads, 1, 1) * rel_pos.unsqueeze(0).to(x.dtype)

        if self.causal:
            attn_bias = attn_bias.masked_fill(rel_pos.unsqueeze(0) > 0, float("-inf"))

        window_left = self.sliding_window
        window_right = 0 if self.causal else self.sliding_window
        outside = (rel_pos < -window_left) | (rel_pos > window_right)
        attn_bias = attn_bias.masked_fill(outside.unsqueeze(0), float("-inf"))

        y = F.scaled_dot_product_attention(
            xq, xk, xv, attn_mask=attn_bias.unsqueeze(0), is_causal=False
        )
        y = y.transpose(1, 2).contiguous().view(B, T, -1)
        return self.wo(y)


# ---------------------------------------------------------------------------
# Codec transformer block
# ---------------------------------------------------------------------------


class CodecFeedForward(nn.Module):
    def __init__(self, dim: int, hidden_dim: int, use_biases: bool):
        super().__init__()
        self.w1 = nn.Linear(dim, hidden_dim, bias=False)
        self.w2 = nn.Linear(hidden_dim, dim, bias=use_biases)
        self.w3 = nn.Linear(dim, hidden_dim, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.w2(F.silu(self.w1(x)) * self.w3(x))


class CodecTransformerBlock(nn.Module):
    def __init__(self, layer_id: int, args: AudioTokenizerArgs):
        super().__init__()
        self.attention = CodecAttention(args, layer_id)
        self.feed_forward = CodecFeedForward(args.dim, args.hidden_dim, args.use_biases)
        self.attention_norm = RMSNorm(args.dim, args.norm_eps)
        self.ffn_norm = RMSNorm(args.dim, args.norm_eps)

        self.has_layer_scale = args.layer_scale
        if args.layer_scale:
            if args.layer_scale_init is None:
                if layer_id < 18:
                    init_scale = 0.1
                elif layer_id <= 24:
                    init_scale = 1e-5
                else:
                    init_scale = 1e-6
            else:
                init_scale = args.layer_scale_init
            self.attention_scale = nn.Parameter(
                torch.full((args.dim,), init_scale, requires_grad=True)
            )
            self.ffn_scale = nn.Parameter(
                torch.full((args.dim,), init_scale, requires_grad=True)
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        r = self.attention(self.attention_norm(x))
        if self.has_layer_scale:
            r = self.attention_scale * r
        h = x + r
        r = self.feed_forward(self.ffn_norm(h))
        if self.has_layer_scale:
            r = self.ffn_scale * r
        return h + r


class CodecTransformer(nn.Module):
    """Wrapper matching vLLM's Transformer class (ModuleDict with string keys).
    Required for checkpoint key compatibility."""

    def __init__(self, args: AudioTokenizerArgs, n_layers: int):
        super().__init__()
        self.layers = nn.ModuleDict()
        for i in range(n_layers):
            self.layers[str(i)] = CodecTransformerBlock(i, args)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for layer in self.layers.values():
            x = layer(x)
        return x


# ---------------------------------------------------------------------------
# Codec decoder
# ---------------------------------------------------------------------------


class AudioTokenizerDecoder(nn.Module):
    """Audio tokenizer decoder: codes → waveform.

    Builds the decoder portion of the codec from AudioTokenizerArgs.
    The encoder is not needed for TTS inference (codes come from the
    acoustic transformer, not from encoding audio).
    """

    def __init__(self, args: AudioTokenizerArgs):
        super().__init__()
        self.args = args
        self.patch_size = args.pretransform_patch_size
        self.latent_dim = args.semantic_dim + args.acoustic_dim

        # Quantizer (decode only)
        self.quantizer = AudioCodebook(args)

        # Build decoder blocks
        decoder_blocks: list[nn.Module] = []

        # Compute initial window size after encoder downsampling
        cur_window = args.decoder_initial_window_size

        # First projection: latent_dim → dim
        decoder_blocks.append(
            CodecCausalConv1d(
                self.latent_dim,
                args.dim,
                kernel_size=args.decoder_convs_kernels[0],
                stride=args.decoder_convs_strides[0],
                pad_mode="replicate",
                use_weight_norm=args.conv_weight_norm,
                use_bias=False,
            )
        )
        if (
            args.half_attn_window_upon_downsampling
            and args.decoder_convs_strides[0] > 1
        ):
            cur_window *= 2

        for idx, n_layers in enumerate(args.decoder_transformer_lengths):
            # Transformer (wrapped in CodecTransformer for checkpoint compat)
            layer_args = deepcopy(args)
            layer_args.attn_sliding_window_size = cur_window
            decoder_blocks.append(CodecTransformer(layer_args, n_layers))

            # Upsampling conv (between stages, not after last)
            if idx + 1 < len(args.decoder_transformer_lengths):
                k = args.decoder_convs_kernels[idx + 1]
                s = args.decoder_convs_strides[idx + 1]
                if k != 1 or s != 1:
                    decoder_blocks.append(
                        CodecCausalConvTranspose1d(
                            args.dim,
                            args.dim,
                            k,
                            s,
                            use_weight_norm=args.conv_weight_norm,
                            use_bias=False,
                        )
                    )
                    if args.half_attn_window_upon_downsampling and s > 1:
                        cur_window *= 2

        self.decoder_blocks = nn.ModuleList(decoder_blocks)

        # Output projection (uses "reflect" pad_mode matching vLLM's default)
        self.output_proj = CodecCausalConv1d(
            args.dim,
            args.pretransform_patch_size,
            kernel_size=args.patch_proj_kernel_size,
            pad_mode="reflect",
            use_weight_norm=args.conv_weight_norm,
            use_bias=False,
        )

        # Downsample factor for waveform reconstruction
        self._downsample_factor = args.pretransform_patch_size * math.prod(
            args.decoder_convs_strides
        )

    @property
    def downsample_factor(self) -> int:
        return self._downsample_factor

    def forward(self, codes: torch.Tensor) -> torch.Tensor:
        """Decode audio codes to waveform.

        Args:
            codes: (B, K, T) integer audio codes.
        Returns:
            waveform: (B, 1, T * downsample_factor).
        """
        # Decode codes to continuous embeddings
        emb = self.quantizer.decode(codes, dtype=codes.new_zeros(1).float().dtype)
        # emb: (B, D_latent, T)

        # Forward through decoder
        # Convert to (B, T, D) for transformer blocks
        x = emb.permute(0, 2, 1).contiguous()

        for block in self.decoder_blocks:
            if isinstance(block, (CodecCausalConv1d, CodecCausalConvTranspose1d)):
                x = x.permute(0, 2, 1)  # (B, D, T)
                x = block(x)
                x = x.permute(0, 2, 1)  # (B, T, D)
            else:
                # CodecTransformer or CodecTransformerBlock: (B, T, D) → (B, T, D)
                x = block(x)

        # Output projection
        x = x.permute(0, 2, 1)  # (B, D, T)
        x = self.output_proj(x)  # (B, patch_size, T)

        # Unpatchify: (B, patch_size, T) → (B, 1, T * patch_size)
        # Must match vLLM's rearrange("b (c h) t -> b c (t h)", h=patch_size):
        # for each time step t, take all patch_size values → interleave by time.
        B, _, T = x.shape
        # x is (B, patch_size, T) → transpose to (B, T, patch_size) → flatten last two
        return x.transpose(1, 2).contiguous().reshape(B, 1, T * self.patch_size)


# ---------------------------------------------------------------------------
# Weight mapping for codec checkpoint loading
# ---------------------------------------------------------------------------


def _map_codec_key(ckpt_key: str) -> str | None:
    """Map codec checkpoint keys to AudioTokenizerDecoder state dict keys."""
    # Strip "audio_tokenizer." prefix if present
    if ckpt_key.startswith("audio_tokenizer."):
        ckpt_key = ckpt_key[len("audio_tokenizer.") :]

    # Skip encoder weights
    if ckpt_key.startswith("input_proj.") or ckpt_key.startswith("encoder_blocks."):
        return None

    # Skip audio_token_embedding (handled by model.pte)
    if ckpt_key.startswith("audio_token_embedding."):
        return None

    # Map quantizer weights
    if ckpt_key.startswith("quantizer.semantic_codebook."):
        return ckpt_key.replace("quantizer.semantic_codebook.", "quantizer.semantic.")
    if ckpt_key.startswith("quantizer.acoustic_codebook."):
        return ckpt_key.replace("quantizer.acoustic_codebook.", "quantizer.acoustic.")

    # Decoder blocks and output_proj map directly
    if ckpt_key.startswith("decoder_blocks.") or ckpt_key.startswith("output_proj."):
        return ckpt_key

    return None


def load_codec_decoder(
    model_path: str,
    codec_args_dict: dict,
    dtype: torch.dtype = torch.float32,
) -> AudioTokenizerDecoder:
    """Load codec decoder weights from checkpoint."""
    import torch.nn.utils.parametrize as parametrize
    from safetensors import safe_open

    args = AudioTokenizerArgs.from_dict(codec_args_dict)

    with torch.device("meta"):
        model = AudioTokenizerDecoder(args)

    ckpt_path = str(Path(model_path) / "consolidated.safetensors")
    state_dict = {}
    with safe_open(ckpt_path, framework="pt", device="cpu") as f:
        for key in f.keys():
            mapped = _map_codec_key(key)
            if mapped is None:
                continue
            state_dict[mapped] = f.get_tensor(key).to(dtype)

    missing, unexpected = model.load_state_dict(state_dict, strict=False, assign=True)

    # Materialize meta buffers
    for fqn, buf in list(model.named_buffers()):
        if buf.device.type == "meta":
            parts = fqn.rsplit(".", 1)
            parent = model.get_submodule(parts[0]) if len(parts) > 1 else model
            parent.register_buffer(
                parts[-1], torch.zeros(buf.shape, dtype=dtype, device="cpu")
            )

    # Recompute ALiBi slopes (persistent=False, lost during meta construction)
    for m in model.modules():
        if isinstance(m, CodecAttention):
            m.register_buffer(
                "alibi_slopes",
                _get_alibi_slopes(m.n_heads),
                persistent=False,
            )

    # Fold weight_norm parametrizations (if checkpoint has parametrized weights)
    for m in model.modules():
        if parametrize.is_parametrized(m, "weight"):
            parametrize.remove_parametrizations(m, "weight")

    if missing:
        print(f"  Codec missing keys ({len(missing)}): {missing[:5]}...")
    if unexpected:
        print(f"  Codec unexpected keys ({len(unexpected)}): {unexpected[:5]}...")

    model.eval()
    return model
