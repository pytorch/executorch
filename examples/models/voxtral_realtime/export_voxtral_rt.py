# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Export Voxtral-Mini-4B-Realtime-2602 to ExecuTorch.

Produces a single .pte with three methods:
  - audio_encoder:   mel (1, 128, T_mel) -> audio_embeds (1, T_audio, 3072)
  - text_decoder:    embeds (1, seq_len, 3072) + cache_position -> logits
  - token_embedding: token_ids (1, seq_len) -> embeds (1, seq_len, 3072)

With --streaming, produces a streaming .pte instead:
  - encode_audio_chunk: mel_chunk (1,128,8) + conv states + enc_pos -> audio_embeds + new states
  - text_decoder:       same as above
  - token_embedding:    same as above

Usage:
    python export_voxtral_rt.py --model-path ~/models/Voxtral-Mini-4B-Realtime-2602
    python export_voxtral_rt.py --model-path ~/models/Voxtral-Mini-4B-Realtime-2602 --streaming
"""

import argparse
import os

import torch
import torch.nn as nn

from executorch.examples.models.voxtral_realtime.model import load_model

from executorch.exir import (
    EdgeCompileConfig,
    ExecutorchBackendConfig,
    to_edge_transform_and_lower,
)
from executorch.exir.passes import MemoryPlanningPass
from torch.export import Dim, export


# ---------------------------------------------------------------------------
# Export wrappers
# ---------------------------------------------------------------------------


class AudioEncoderExport(nn.Module):
    """Wraps encoder + downsample + adapter for export."""

    def __init__(self, model):
        super().__init__()
        self.encoder = model.encoder
        self.adapter = model.adapter
        self.downsample_factor = model.config.downsample_factor

    def forward(self, mel: torch.Tensor) -> torch.Tensor:
        # mel: (1, 128, T_mel) channels-first. T_mel must be a multiple of 8.
        x = self.encoder(mel)  # (1, T_enc, 1280)
        B, T, D = x.shape
        # T = T_mel // 2 (conv2 stride), guaranteed divisible by 4
        x = x.reshape(B, T // self.downsample_factor, D * self.downsample_factor)
        return self.adapter(x)  # (1, T_audio, 3072)


class TextDecoderExport(nn.Module):
    """Wraps LM decoder for export. Time embedding is baked as a constant."""

    def __init__(self, model):
        super().__init__()
        self.decoder = model.decoder
        self.register_buffer("t_cond", model.t_cond)

    def forward(
        self, input_embeds: torch.Tensor, cache_position: torch.Tensor
    ) -> torch.Tensor:
        return self.decoder(input_embeds, cache_position, self.t_cond)


class TokenEmbeddingExport(nn.Module):
    """Wraps token embedding lookup for export."""

    def __init__(self, model):
        super().__init__()
        self.tok_embeddings = model.decoder.tok_embeddings

    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        return self.tok_embeddings(token_ids)


# ---------------------------------------------------------------------------
# Export
# ---------------------------------------------------------------------------


def _export_decoder_and_embedding(
    programs, model, max_seq_len, qlinear, qlinear_group_size, qembedding
):
    """Export text_decoder and token_embedding into programs dict."""
    from executorch.extension.llm.export.quantize import quantize_model_

    param_dtype = torch.float32

    print("\nExporting text_decoder...")
    text_decoder = TextDecoderExport(model)
    text_decoder.eval()

    if qlinear:
        print(f"  Quantizing decoder ({qlinear})...")
        quantize_model_(
            text_decoder,
            qlinear_config=qlinear,
            qlinear_group_size=qlinear_group_size,
        )

    seq_dim = Dim("seq_len", min=1, max=max_seq_len)
    sample_embeds = torch.randn(1, 4, model.config.dim, dtype=param_dtype)
    sample_pos = torch.arange(4, dtype=torch.long)
    programs["text_decoder"] = export(
        text_decoder,
        (sample_embeds, sample_pos),
        dynamic_shapes={
            "input_embeds": {1: seq_dim},
            "cache_position": {0: seq_dim},
        },
        strict=True,
    )
    print(f"  text_decoder exported (sample input: {sample_embeds.shape})")

    print("\nExporting token_embedding...")
    tok_emb = TokenEmbeddingExport(model)
    tok_emb.eval()

    if qembedding:
        print(f"  Quantizing embedding ({qembedding})...")
        quantize_model_(
            tok_emb,
            qembedding_config=qembedding,
        )

    tok_seq_dim = Dim("tok_seq_len", min=1, max=max_seq_len)
    sample_ids = torch.tensor([[0, 1, 2, 3]], dtype=torch.long)
    programs["token_embedding"] = export(
        tok_emb,
        (sample_ids,),
        dynamic_shapes={"token_ids": {1: tok_seq_dim}},
        strict=True,
    )
    print(f"  token_embedding exported (sample input: {sample_ids.shape})")


def export_all(
    model,
    max_seq_len,
    qlinear_encoder=None,
    qlinear_encoder_group_size=32,
    qlinear=None,
    qlinear_group_size=32,
    qembedding=None,
):
    """Export all three model components with per-component quantization."""
    from executorch.extension.llm.export.quantize import quantize_model_

    programs = {}
    param_dtype = torch.float32

    # 1. Audio encoder
    print("\nExporting audio_encoder...")
    audio_encoder = AudioEncoderExport(model)
    audio_encoder.eval()

    if qlinear_encoder:
        print(f"  Quantizing encoder ({qlinear_encoder})...")
        quantize_model_(
            audio_encoder,
            qlinear_config=qlinear_encoder,
            qlinear_group_size=qlinear_encoder_group_size,
        )

    _t_mel_base = Dim("_t_mel_base", min=1, max=3000)
    t_mel_dim = 8 * _t_mel_base
    sample_mel = torch.randn(1, model.config.num_mel_bins, 160, dtype=param_dtype)
    programs["audio_encoder"] = export(
        audio_encoder,
        (sample_mel,),
        dynamic_shapes={"mel": {2: t_mel_dim}},
        strict=True,
    )
    print(f"  audio_encoder exported (sample input: {sample_mel.shape})")

    # 2-3. Text decoder + token embedding
    _export_decoder_and_embedding(
        programs, model, max_seq_len, qlinear, qlinear_group_size, qembedding
    )

    metadata = {
        "sample_rate": 16000,
        "num_mel_bins": model.config.num_mel_bins,
        "hop_length": 160,
        "window_size": 400,
        "downsample_factor": model.config.downsample_factor,
        "dim": model.config.dim,
        "vocab_size": model.config.vocab_size,
        "max_seq_len": max_seq_len,
    }

    return programs, metadata


def export_streaming(
    model,
    max_seq_len,
    max_enc_len=750,
    qlinear_encoder=None,
    qlinear_encoder_group_size=32,
    qlinear=None,
    qlinear_group_size=32,
    qembedding=None,
):
    """Export streaming model components with per-component quantization."""
    from executorch.extension.llm.export.quantize import quantize_model_

    programs = {}
    param_dtype = torch.float32

    # 1. Streaming audio encoder
    print("\nExporting encode_audio_chunk...")
    from executorch.examples.models.voxtral_realtime.model import (
        StreamingAudioEncoderExport,
    )

    streaming_enc = StreamingAudioEncoderExport(model, max_enc_len=max_enc_len)
    streaming_enc.eval()

    if qlinear_encoder:
        print(f"  Quantizing encoder ({qlinear_encoder})...")
        quantize_model_(
            streaming_enc,
            qlinear_config=qlinear_encoder,
            qlinear_group_size=qlinear_encoder_group_size,
        )

    sample_mel_chunk = torch.randn(1, model.config.num_mel_bins, 8, dtype=param_dtype)
    sample_conv1_state = torch.zeros(1, model.config.num_mel_bins, 2, dtype=param_dtype)
    sample_conv2_state = torch.zeros(1, model.config.enc_dim, 2, dtype=param_dtype)
    sample_enc_pos = torch.arange(4, dtype=torch.long)

    programs["encode_audio_chunk"] = export(
        streaming_enc,
        (sample_mel_chunk, sample_conv1_state, sample_conv2_state, sample_enc_pos),
        dynamic_shapes=None,
        strict=True,
    )
    print(
        f"  encode_audio_chunk exported (fixed shapes: mel_chunk={sample_mel_chunk.shape})"
    )

    # 2-3. Text decoder + token embedding
    _export_decoder_and_embedding(
        programs, model, max_seq_len, qlinear, qlinear_group_size, qembedding
    )

    # Derive STFT overlap from audio parameters.
    # Left overlap: next multiple of hop_length >= n_fft/2
    # Right look-ahead: how far the last mel frame extends past the step end
    # mel_skip: number of overlap frames to skip at the start
    hop_length = 160
    n_fft = 400
    sample_rate = 16000
    frame_rate = 12.5
    step_samples = int(sample_rate / frame_rate)
    stft_left_overlap = ((n_fft // 2 + hop_length - 1) // hop_length) * hop_length
    mel_skip_frames = stft_left_overlap // hop_length
    chunk_mel_len = 8
    stft_right_lookahead = (
        (chunk_mel_len - 1) * hop_length + n_fft // 2 - chunk_mel_len * hop_length
    )
    # = (8-1)*160 + 200 - 8*160 = 1320 - 1280 = 40 samples = 2.5ms

    metadata = {
        "sample_rate": sample_rate,
        "num_mel_bins": model.config.num_mel_bins,
        "hop_length": hop_length,
        "window_size": n_fft,
        "downsample_factor": model.config.downsample_factor,
        "dim": model.config.dim,
        "enc_dim": model.config.enc_dim,
        "vocab_size": model.config.vocab_size,
        "max_seq_len": max_seq_len,
        "streaming": 1,
        "step_samples": step_samples,
        "chunk_mel_len": chunk_mel_len,
        "max_enc_len": max_enc_len,
        "conv1_pad": 2,
        "conv2_pad": 2,
        "stft_left_overlap": stft_left_overlap,
        "stft_right_lookahead": stft_right_lookahead,
        "mel_skip_frames": mel_skip_frames,
    }

    return programs, metadata


def lower_to_executorch(programs, metadata, backend="xnnpack"):
    """Lower exported programs to ExecuTorch."""
    if backend == "xnnpack":
        from executorch.backends.xnnpack.partition.xnnpack_partitioner import (
            XnnpackDynamicallyQuantizedPartitioner,
            XnnpackPartitioner,
        )

        print("\nLowering to ExecuTorch with XNNPACK...")
        partitioner = {
            key: [XnnpackDynamicallyQuantizedPartitioner(), XnnpackPartitioner()]
            for key in programs
        }
    else:
        print("\nLowering to ExecuTorch (portable)...")
        partitioner = []

    et_prog = to_edge_transform_and_lower(
        programs,
        partitioner=partitioner,
        compile_config=EdgeCompileConfig(
            _check_ir_validity=False,
            _skip_dim_order=True,
        ),
        constant_methods=metadata,
    )

    return et_prog.to_executorch(
        config=ExecutorchBackendConfig(
            extract_delegate_segments=True,
            do_quant_fusion_and_const_prop=True,
            memory_planning_pass=MemoryPlanningPass(alloc_graph_input=False),
        ),
    )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(
        description="Export Voxtral Realtime to ExecuTorch"
    )
    parser.add_argument(
        "--model-path",
        required=True,
        help="Directory with params.json + consolidated.safetensors",
    )
    parser.add_argument(
        "--backend",
        default="xnnpack",
        choices=["portable", "xnnpack"],
        help="Backend for acceleration (default: xnnpack)",
    )
    parser.add_argument(
        "--output-dir",
        default="./voxtral_rt_exports",
        help="Output directory (default: ./voxtral_rt_exports)",
    )
    parser.add_argument(
        "--max-seq-len",
        type=int,
        default=4096,
        help="KV cache length (default: 4096)",
    )
    parser.add_argument(
        "--delay-tokens",
        type=int,
        default=6,
        help="Transcription delay in tokens (default: 6 = 480ms)",
    )
    parser.add_argument(
        "--qlinear",
        default=None,
        choices=["4w", "8w", "8da4w", "8da8w"],
        help="Quantize decoder linear layers.",
    )
    parser.add_argument(
        "--qlinear-group-size",
        type=int,
        default=32,
        help="Group size for decoder linear quantization (default: 32).",
    )
    parser.add_argument(
        "--qlinear-encoder",
        default=None,
        choices=["4w", "8w", "8da4w", "8da8w"],
        help="Quantize encoder linear layers (separate from decoder).",
    )
    parser.add_argument(
        "--qlinear-encoder-group-size",
        type=int,
        default=32,
        help="Group size for encoder linear quantization (default: 32).",
    )
    parser.add_argument(
        "--qembedding",
        default=None,
        choices=["8w"],
        help="Quantize embedding layers (8-bit weight-only).",
    )
    parser.add_argument(
        "--streaming",
        action="store_true",
        help="Export streaming encoder (encode_audio_chunk) instead of offline encoder.",
    )
    parser.add_argument(
        "--max-enc-len",
        type=int,
        default=750,
        help="Encoder sliding window size for streaming (default: 750).",
    )
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    # Load model
    print("Loading model...")
    model = load_model(
        args.model_path,
        max_seq_len=args.max_seq_len,
        n_delay_tokens=args.delay_tokens,
    )

    # Untie output/embedding weights before quantization so each layer gets
    # its own quantization config (embedding: 8w, output linear: 8da4w).
    if args.qlinear or args.qembedding:
        model.decoder.output.weight = torch.nn.Parameter(
            model.decoder.tok_embeddings.weight.clone()
        )

    # Export (quantization is applied per-component inside export functions)
    print("\nExporting components...")
    quant_args = {
        "qlinear_encoder": args.qlinear_encoder,
        "qlinear_encoder_group_size": args.qlinear_encoder_group_size,
        "qlinear": args.qlinear,
        "qlinear_group_size": args.qlinear_group_size,
        "qembedding": args.qembedding,
    }
    if args.streaming:
        programs, metadata = export_streaming(
            model, args.max_seq_len, args.max_enc_len, **quant_args
        )
    else:
        programs, metadata = export_all(model, args.max_seq_len, **quant_args)

    # Lower
    et = lower_to_executorch(programs, metadata, backend=args.backend)

    # Save
    pte_path = os.path.join(args.output_dir, "model.pte")
    print(f"\nSaving to {pte_path}...")
    with open(pte_path, "wb") as f:
        et.write_to_file(f)
    size_mb = os.path.getsize(pte_path) / (1024 * 1024)
    print(f"Saved {size_mb:.1f} MB")

    print("\nDone!")


if __name__ == "__main__":
    main()
