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

Usage:
    python export_voxtral_rt.py --model-path ~/models/Voxtral-Mini-4B-Realtime-2602
"""

import argparse
import os

import torch
import torch.nn as nn

from executorch.exir import (
    EdgeCompileConfig,
    ExecutorchBackendConfig,
    to_edge_transform_and_lower,
)
from executorch.exir.passes import MemoryPlanningPass

from model import load_model
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


def export_all(model, max_seq_len):
    """Export all three model components."""
    programs = {}

    # Infer dtype from model weights for sample inputs
    param_dtype = torch.float32

    # 1. Audio encoder
    print("\nExporting audio_encoder...")
    audio_encoder = AudioEncoderExport(model)
    audio_encoder.eval()

    # T_mel must be a multiple of 8 (conv stride 2 + downsample 4)
    _t_mel_base = Dim("_t_mel_base", min=1, max=3000)
    t_mel_dim = 8 * _t_mel_base
    sample_mel = torch.randn(1, model.config.num_mel_bins, 160, dtype=param_dtype)
    programs["audio_encoder"] = export(
        audio_encoder,
        (sample_mel,),
        dynamic_shapes={"mel": {2: t_mel_dim}},
        strict=False,
    )
    print(f"  audio_encoder exported (sample input: {sample_mel.shape})")

    # 2. Text decoder
    print("\nExporting text_decoder...")
    text_decoder = TextDecoderExport(model)
    text_decoder.eval()

    seq_dim = Dim("seq_len", min=1, max=max_seq_len)
    # Use seq_len > 1 to avoid torch.export specializing on constant 1
    sample_embeds = torch.randn(1, 4, model.config.dim, dtype=param_dtype)
    sample_pos = torch.arange(4, dtype=torch.long)
    programs["text_decoder"] = export(
        text_decoder,
        (sample_embeds, sample_pos),
        dynamic_shapes={
            "input_embeds": {1: seq_dim},
            "cache_position": {0: seq_dim},
        },
        strict=False,
    )
    print(f"  text_decoder exported (sample input: {sample_embeds.shape})")

    # 3. Token embedding
    print("\nExporting token_embedding...")
    tok_emb = TokenEmbeddingExport(model)
    tok_emb.eval()

    tok_seq_dim = Dim("tok_seq_len", min=1, max=max_seq_len)
    sample_ids = torch.tensor([[0, 1, 2, 3]], dtype=torch.long)
    programs["token_embedding"] = export(
        tok_emb,
        (sample_ids,),
        dynamic_shapes={"token_ids": {1: tok_seq_dim}},
        strict=False,
    )
    print(f"  token_embedding exported (sample input: {sample_ids.shape})")

    # Metadata
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
        help="Quantize linear layers (e.g., 8da4w for 8-bit dynamic activation, 4-bit weight).",
    )
    parser.add_argument(
        "--qlinear-group-size",
        type=int,
        default=32,
        help="Group size for linear quantization (default: 32).",
    )
    parser.add_argument(
        "--qembedding",
        default=None,
        choices=["8w"],
        help="Quantize embedding layers (8-bit weight-only).",
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

    # Quantize (before export, source-transform style)
    if args.qlinear or args.qembedding:
        from executorch.extension.llm.export.quantize import quantize_model_

        print("\nQuantizing...")
        quantize_model_(
            model,
            qlinear_config=args.qlinear,
            qlinear_group_size=args.qlinear_group_size,
            qembedding_config=args.qembedding,
        )

    # Export
    print("\nExporting components...")
    programs, metadata = export_all(model, args.max_seq_len)

    # Lower
    et = lower_to_executorch(programs, metadata, backend=args.backend)

    # Save
    pte_path = os.path.join(args.output_dir, "voxtral_realtime.pte")
    print(f"\nSaving to {pte_path}...")
    with open(pte_path, "wb") as f:
        et.write_to_file(f)
    size_mb = os.path.getsize(pte_path) / (1024 * 1024)
    print(f"Saved {size_mb:.1f} MB")

    print("\nDone!")


if __name__ == "__main__":
    main()
