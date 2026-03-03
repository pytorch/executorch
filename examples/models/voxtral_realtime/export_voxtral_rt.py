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
  - encode_audio_chunk: mel_chunk (1,128,8) + enc_pos (4,) -> audio_embeds (1,1,3072)
  - text_decoder:       same as above
  - token_embedding:    same as above

Backend support:
  - XNNPACK (default): Uses custom SDPA op (torch.ops.llama.custom_sdpa) for optimal performance
  - Metal/AOTI: Uses MetalSDPA (_scaled_dot_product_attention_math_for_mps) for text_decoder
                and StandardEncoderSDPA (F.scaled_dot_product_attention) for streaming encoder,
                avoiding custom_sdpa which is incompatible with AOTI. Uses Dim.AUTO for audio
                encoder dynamic shapes (explicit bounds cause issues with AOTI).
  - CUDA/AOTI: Uses CudaSDPA (F.scaled_dot_product_attention with GQA expansion) for text_decoder
               and StandardEncoderSDPA for streaming encoder. Compiles to CUDA kernels via
               AOTInductor. Supports int4 quantization via _weight_int4pack_mm fallback kernel.
  - Portable: Uses custom SDPA like XNNPACK

Usage:
    python export_voxtral_rt.py --model-path ~/models/Voxtral-Mini-4B-Realtime-2602
    python export_voxtral_rt.py --model-path ~/models/Voxtral-Mini-4B-Realtime-2602 --streaming
    python export_voxtral_rt.py --model-path ~/models/Voxtral-Mini-4B-Realtime-2602 --backend metal
    python export_voxtral_rt.py --model-path ~/models/Voxtral-Mini-4B-Realtime-2602 --backend cuda --qlinear 4w
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
    programs,
    model,
    max_seq_len,
    qlinear,
    qlinear_group_size,
    qlinear_packing_format,
    qembedding,
    qembedding_group_size,
    device="cpu",
):
    """Export text_decoder and token_embedding into programs dict."""
    from executorch.extension.llm.export.quantize import quantize_model_

    param_dtype = next(model.parameters()).dtype

    print("\nExporting text_decoder...")
    text_decoder = TextDecoderExport(model)
    text_decoder.eval()

    if qlinear:
        print(f"  Quantizing decoder ({qlinear})...")
        quantize_model_(
            text_decoder,
            qlinear_config=qlinear,
            qlinear_group_size=qlinear_group_size,
            qlinear_packing_format=qlinear_packing_format,
        )

    seq_dim = Dim("seq_len", min=1, max=max_seq_len)
    sample_embeds = torch.randn(
        1, 4, model.config.dim, dtype=param_dtype, device=device
    )
    sample_pos = torch.arange(4, dtype=torch.long, device=device)
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
            qembedding_group_size=qembedding_group_size,
        )

    tok_seq_dim = Dim("tok_seq_len", min=1, max=max_seq_len)
    sample_ids = torch.tensor([[0, 1, 2, 3]], dtype=torch.long, device=device)
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
    qlinear_encoder_group_size=None,
    qlinear_encoder_packing_format=None,
    qlinear=None,
    qlinear_group_size=None,
    qlinear_packing_format=None,
    qembedding=None,
    qembedding_group_size=None,
    backend="xnnpack",
):
    """Export all three model components with per-component quantization."""
    from executorch.extension.llm.export.quantize import quantize_model_

    programs = {}
    param_dtype = next(model.parameters()).dtype
    device = "cuda" if backend == "cuda" else "cpu"

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
            qlinear_packing_format=qlinear_encoder_packing_format,
        )

    # For Metal/CUDA/AOTI: use max size as sample and Dim.AUTO (explicit bounds cause issues)
    # For XNNPACK: use small sample with explicit bounds
    if backend in ("metal", "cuda"):
        max_t_mel = 24000  # 3000 * 8
        sample_mel = torch.randn(
            1, model.config.num_mel_bins, max_t_mel, dtype=param_dtype, device=device
        )
        dynamic_shapes = {"mel": {2: Dim.AUTO}}
    else:
        _t_mel_base = Dim("_t_mel_base", min=1, max=3000)
        t_mel_dim = 8 * _t_mel_base
        sample_mel = torch.randn(
            1, model.config.num_mel_bins, 160, dtype=param_dtype, device=device
        )
        dynamic_shapes = {"mel": {2: t_mel_dim}}

    programs["audio_encoder"] = export(
        audio_encoder,
        (sample_mel,),
        dynamic_shapes=dynamic_shapes,
        strict=True,
    )
    print(f"  audio_encoder exported (sample input: {sample_mel.shape})")

    # 2-3. Text decoder + token embedding
    _export_decoder_and_embedding(
        programs,
        model,
        max_seq_len,
        qlinear,
        qlinear_group_size,
        qlinear_packing_format,
        qembedding,
        qembedding_group_size,
        device,
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
    qlinear_encoder_group_size=None,
    qlinear_encoder_packing_format=None,
    qlinear=None,
    qlinear_group_size=None,
    qlinear_packing_format=None,
    qembedding=None,
    qembedding_group_size=None,
    backend="xnnpack",
):
    """Export streaming model components with per-component quantization."""
    from executorch.extension.llm.export.quantize import quantize_model_

    programs = {}
    param_dtype = next(model.parameters()).dtype
    device = "cuda" if backend == "cuda" else "cpu"

    # 1. Streaming audio encoder
    print("\nExporting encode_audio_chunk...")
    from executorch.examples.models.voxtral_realtime.model import (
        StreamingAudioEncoderExport,
    )

    streaming_enc = StreamingAudioEncoderExport(model, max_enc_len=max_enc_len)
    streaming_enc.to(device=device, dtype=param_dtype)
    streaming_enc.eval()

    if qlinear_encoder:
        print(f"  Quantizing encoder ({qlinear_encoder})...")
        quantize_model_(
            streaming_enc,
            qlinear_config=qlinear_encoder,
            qlinear_group_size=qlinear_encoder_group_size,
            qlinear_packing_format=qlinear_encoder_packing_format,
        )

    sample_mel_chunk = torch.randn(
        1, model.config.num_mel_bins, 8, dtype=param_dtype, device=device
    )
    sample_enc_pos = torch.arange(4, dtype=torch.long, device=device)

    programs["encode_audio_chunk"] = export(
        streaming_enc,
        (sample_mel_chunk, sample_enc_pos),
        dynamic_shapes=None,
        strict=True,
    )
    print(
        f"  encode_audio_chunk exported (fixed shapes: mel_chunk={sample_mel_chunk.shape})"
    )

    # 2-3. Text decoder + token embedding
    _export_decoder_and_embedding(
        programs,
        model,
        max_seq_len,
        qlinear,
        qlinear_group_size,
        qlinear_packing_format,
        qembedding,
        qembedding_group_size,
        device,
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


# Custom decomposition for Metal backend compatibility.
# This decomposition is necessary to avoid issues with reinterpret_tensor_wrapper
# when linear layers have biases, which would cause ExecuTorch errors with 0 stride.
# TODO(manuelcandales): Remove this once ExecuTorch Metal backend supports bias in linear layers.
def _linear_bias_decomposition(input, weight, bias=None):
    """Decompose linear with bias into matmul + add."""
    weight_t = torch.ops.aten.t.default(weight)
    out = torch.ops.aten.matmul.default(input, weight_t)
    if bias is not None:
        return torch.ops.aten.add.Tensor(out, bias)
    return out


def export_preprocessor(output_dir, backend="xnnpack", streaming=False):
    """Export mel spectrogram preprocessor.

    Uses XNNPACK for all backends except MLX, which uses MLX partitioner.
    """
    from executorch.extension.audio.mel_spectrogram import WhisperAudioProcessor

    # Use MLX partitioner for mlx backend, XNNPACK for everything else
    pp_backend = "mlx" if backend == "mlx" else "xnnpack"
    print(f"  Using {pp_backend.upper()} partitioner for preprocessor...")

    model = WhisperAudioProcessor(
        feature_size=128,
        max_audio_len=300,
        streaming=streaming,
    )

    audio_tensor = torch.randn(93680)
    shapes_collection = torch.export.ShapesCollection()
    max_n_chunks = int(model.max_audio_len * model.n_samples)
    shapes_collection[audio_tensor] = {0: Dim.DYNAMIC(max=max_n_chunks)}

    with torch.no_grad(), torch.fx.experimental._config.patch(
        backed_size_oblivious=True
    ):
        ep = export(
            model, (audio_tensor,), dynamic_shapes=shapes_collection, strict=True
        )

        if pp_backend == "mlx":
            from executorch.backends.mlx.partitioner import MLXPartitioner

            partitioner = [MLXPartitioner()]
        else:
            from executorch.backends.xnnpack.partition.xnnpack_partitioner import (
                XnnpackPartitioner,
            )

            partitioner = [XnnpackPartitioner()]

        edge = to_edge_transform_and_lower(
            ep,
            partitioner=partitioner,
            compile_config=EdgeCompileConfig(_check_ir_validity=False),
        )
        exec_prog = edge.to_executorch()

        pp_path = os.path.join(output_dir, "preprocessor.pte")
        with open(pp_path, "wb") as f:
            exec_prog.write_to_file(f)
        size_mb = os.path.getsize(pp_path) / (1024 * 1024)
        print(f"  Saved preprocessor to {pp_path} ({size_mb:.1f} MB)")


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
    elif backend == "metal":
        from executorch.backends.apple.metal.metal_backend import MetalBackend
        from executorch.backends.apple.metal.metal_partitioner import MetalPartitioner

        print("\nLowering to ExecuTorch with Metal...")

        # Run decompositions for Metal backend
        updated_programs = {}
        for key, ep in programs.items():
            updated_programs[key] = ep.run_decompositions(
                {torch.ops.aten.linear.default: _linear_bias_decomposition}
            )
        programs = updated_programs

        partitioner = {}
        for key in programs:
            compile_specs = [MetalBackend.generate_method_name_compile_spec(key)]
            partitioner[key] = [MetalPartitioner(compile_specs)]
    elif backend in ("cuda", "cuda-windows"):
        from executorch.backends.cuda.cuda_backend import CudaBackend
        from executorch.backends.cuda.cuda_partitioner import CudaPartitioner
        from executorch.exir.backend.compile_spec_schema import CompileSpec
        from torch._inductor.decomposition import conv1d_to_conv2d

        print(
            f"\nLowering to ExecuTorch with CUDA{' (Windows)' if backend == 'cuda-windows' else ''}..."
        )

        # Run conv1d decomposition for CUDA backend
        updated_programs = {}
        for key, ep in programs.items():
            updated_programs[key] = ep.run_decompositions(
                {torch.ops.aten.conv1d.default: conv1d_to_conv2d}
            )
        programs = updated_programs

        partitioner = {}
        for key in programs:
            compile_specs = [CudaBackend.generate_method_name_compile_spec(key)]
            if backend == "cuda-windows":
                compile_specs.append(CompileSpec("platform", b"windows"))
            partitioner[key] = [CudaPartitioner(compile_specs)]
    elif backend == "mlx":
        from executorch.backends.mlx.partitioner import MLXPartitioner

        print("\nLowering to ExecuTorch with MLX...")
        partitioner = {key: [MLXPartitioner()] for key in programs}
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
        choices=["portable", "xnnpack", "mlx", "metal", "cuda", "cuda-windows"],
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
        choices=["4w", "8w", "8da4w", "8da8w", "fpa4w"],
        help="Quantize decoder linear layers.",
    )
    parser.add_argument(
        "--qlinear-group-size",
        type=int,
        default=None,
        help="Group size for decoder linear quantization (default: 32).",
    )
    parser.add_argument(
        "--qlinear-packing-format",
        default=None,
        choices=["tile_packed_to_4d"],
        help="Packing format for decoder 4w quantization (CUDA: tile_packed_to_4d).",
    )
    parser.add_argument(
        "--qlinear-encoder",
        default=None,
        choices=["4w", "8w", "8da4w", "8da8w", "fpa4w"],
        help="Quantize encoder linear layers (separate from decoder).",
    )
    parser.add_argument(
        "--qlinear-encoder-group-size",
        type=int,
        default=None,
        help="Group size for encoder linear quantization (default: 32).",
    )
    parser.add_argument(
        "--qlinear-encoder-packing-format",
        default=None,
        choices=["tile_packed_to_4d"],
        help="Packing format for encoder 4w quantization (CUDA: tile_packed_to_4d).",
    )
    parser.add_argument(
        "--qembedding",
        default=None,
        choices=["8w"],
        help="Quantize embedding layers (8-bit weight-only).",
    )
    parser.add_argument(
        "--qembedding-group-size",
        type=int,
        default=None,
        help="Group size for embedding quantization (default: 0 = per-channel).",
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
    parser.add_argument(
        "--dtype",
        default="fp32",
        choices=["fp32", "bf16"],
        help="Model dtype (default: fp32).",
    )
    parser.add_argument(
        "--export-preprocessor",
        action="store_true",
        help="Also export preprocessor.pte (uses XNNPACK, or MLX for --backend mlx).",
    )
    args = parser.parse_args()
    backend_for_export = "cuda" if args.backend == "cuda-windows" else args.backend

    # Validate fpa4w quantization requires Metal backend
    if args.qlinear == "fpa4w" and backend_for_export != "metal":
        parser.error("--qlinear=fpa4w can only be used with --backend=metal")
    if args.qlinear_encoder == "fpa4w" and backend_for_export != "metal":
        parser.error("--qlinear-encoder=fpa4w can only be used with --backend=metal")

    os.makedirs(args.output_dir, exist_ok=True)

    model_dtype = {"fp32": torch.float32, "bf16": torch.bfloat16}[args.dtype]

    print("Loading model...")
    model = load_model(
        args.model_path,
        max_seq_len=args.max_seq_len,
        n_delay_tokens=args.delay_tokens,
        dtype=model_dtype,
        backend=backend_for_export,
    )

    # Move to CUDA for CUDA backend export (AOTInductor needs CUDA tensors)
    if backend_for_export == "cuda":
        print("Moving model to CUDA...")
        model.cuda()

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
        "qlinear_encoder_packing_format": args.qlinear_encoder_packing_format,
        "qlinear": args.qlinear,
        "qlinear_group_size": args.qlinear_group_size,
        "qlinear_packing_format": args.qlinear_packing_format,
        "qembedding": args.qembedding,
        "qembedding_group_size": args.qembedding_group_size,
        "backend": backend_for_export,
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

    # Write tensor data for CUDA backend (.ptd file with compiled .so and weights)
    if et._tensor_data:
        et.write_tensor_data_to_file(args.output_dir)
        print(f"Saved tensor data to {args.output_dir}/")

    # Export preprocessor if requested
    if args.export_preprocessor:
        print("\nExporting preprocessor...")
        export_preprocessor(args.output_dir, args.backend, args.streaming)

    print("\nDone!")


if __name__ == "__main__":
    main()
