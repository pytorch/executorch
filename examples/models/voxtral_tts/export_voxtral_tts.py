# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Export Voxtral TTS to ExecuTorch.

Produces model.pte with five methods:
  - token_embedding:        token_ids (1, seq) -> embeds (1, seq, D)
  - text_decoder:           embeds (1, seq, D) + cache_pos (seq,) -> hidden (1, seq, D)
  - lm_head:                hidden (1, 1, D) -> logits (1, 1, vocab)
  - decode_audio_frame:     hidden (1, D) + noise (1, C) -> codes (1, 1+C)
  - audio_token_embedding:  codes (1, K, seq) -> embeds (1, seq, D)

Backend support:
  - Metal/AOTI: MetalSDPA, StaticKVCache, bf16, fpa4w quantization
  - XNNPACK (CPU): custom_sdpa, KVCache

Usage:
    python export_voxtral_tts.py --model-path ~/models/VoxtralTTS --backend metal --dtype bf16
    python export_voxtral_tts.py --model-path ~/models/VoxtralTTS --backend metal --dtype bf16 --qlinear fpa4w
    python export_voxtral_tts.py --model-path ~/models/VoxtralTTS --backend xnnpack --qlinear 8da4w
"""

import argparse
import os
from pathlib import Path

import torch
import torch.nn as nn

from executorch.examples.models.voxtral_tts.model import load_model

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


class TokenEmbeddingExport(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.tok_embeddings = model.decoder.tok_embeddings

    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        return self.tok_embeddings(token_ids)


class TextDecoderExport(nn.Module):
    """Wraps LM decoder. Returns normed hidden states (not logits)."""

    def __init__(self, model):
        super().__init__()
        self.decoder = model.decoder

    def forward(
        self, input_embeds: torch.Tensor, cache_position: torch.Tensor
    ) -> torch.Tensor:
        return self.decoder(input_embeds, cache_position)


class LMHeadExport(nn.Module):
    """Output projection: hidden -> logits."""

    def __init__(self, model):
        super().__init__()
        self.output = model.decoder.output

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return self.output(hidden_states)


class DecodeAudioFrameExport(nn.Module):
    """Flow matching: hidden + noise -> audio codes."""

    def __init__(self, model):
        super().__init__()
        self.acoustic_transformer = model.acoustic_transformer

    def forward(self, hidden_states: torch.Tensor, noise: torch.Tensor) -> torch.Tensor:
        return self.acoustic_transformer.decode_one_frame(hidden_states, noise)


class AudioTokenEmbeddingExport(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.audio_token_embedding = model.audio_token_embedding

    def forward(self, codes: torch.Tensor) -> torch.Tensor:
        return self.audio_token_embedding(codes)


# ---------------------------------------------------------------------------
# Export
# ---------------------------------------------------------------------------


def _quantize_with_skip(module, qlinear_config, group_size):
    """Apply quantization, skipping layers whose K dim isn't divisible by group_size.

    The acoustic transformer's input_projection has a small K dimension
    (n_acoustic_codebook, e.g. 36) that may not be divisible by group_size.
    This function applies quantization with a filter that skips such layers,
    regardless of the quantization config.
    """
    from executorch.extension.llm.export.quantize import quantize_model_

    if qlinear_config == "fpa4w":
        import torchao.experimental.ops.mps  # noqa: F401
        from torchao.experimental.quant_api import UIntxWeightOnlyConfig
        from torchao.quantization import quantize_

        config = UIntxWeightOnlyConfig(
            group_size=group_size,
            bitwidth=4,
            uintx_choose_qparams_algorithm="hqq",
        )

        def safe_filter(mod, fqn):
            if not isinstance(mod, torch.nn.Linear):
                return False
            k = mod.weight.shape[1]
            if k % group_size != 0:
                print(f"    Skipping {fqn}: K={k} not divisible by {group_size}")
                return False
            return True

        quantize_(module, config, filter_fn=safe_filter)
    else:
        # For non-fpa4w configs (e.g. 8da4w), quantize_model_ skips layers
        # whose K dim isn't divisible by group_size with a warning.
        # Use group_size=0 for per-axis to avoid the issue entirely on
        # small layers, or accept the skip behavior.
        quantize_model_(
            module,
            qlinear_config=qlinear_config,
            qlinear_group_size=group_size,
        )


def export_model(
    model,
    max_seq_len,
    qlinear=None,
    qlinear_group_size=32,
    qembedding=None,
    backend="metal",
):
    """Export all five model components."""
    from executorch.extension.llm.export.quantize import quantize_model_

    programs = {}
    param_dtype = next(model.parameters()).dtype

    # --- 1. Token embedding ---
    print("\nExporting token_embedding...")
    tok_emb = TokenEmbeddingExport(model)
    tok_emb.eval()

    if qembedding:
        print(f"  Quantizing embedding ({qembedding})...")
        quantize_model_(tok_emb, qembedding_config=qembedding)

    tok_seq_dim = Dim("tok_seq_len", min=1, max=max_seq_len)
    tok_dyn = {"token_ids": {1: tok_seq_dim}}
    sample_ids = torch.tensor([[0, 1, 2, 3]], dtype=torch.long)
    programs["token_embedding"] = export(
        tok_emb, (sample_ids,), dynamic_shapes=tok_dyn, strict=True
    )
    print("  token_embedding exported")

    # --- 2. Text decoder ---
    print("\nExporting text_decoder...")
    text_dec = TextDecoderExport(model)
    text_dec.eval()

    if qlinear:
        print(f"  Quantizing decoder ({qlinear})...")
        quantize_model_(
            text_dec,
            qlinear_config=qlinear,
            qlinear_group_size=qlinear_group_size,
        )

    # Metal's _scaled_dot_product_attention_math_for_mps requires seq_len > 2.
    # For decode (seq_len=1), use a separate static export or pad to 3.
    seq_min = 3 if backend == "metal" else 1
    seq_dim = Dim("seq_len", min=seq_min, max=max_seq_len)
    sample_embeds = torch.randn(1, 4, model.config.dim, dtype=param_dtype)
    sample_pos = torch.arange(4, dtype=torch.long)
    dec_dyn = {
        "input_embeds": {1: seq_dim},
        "cache_position": {0: seq_dim},
    }
    programs["text_decoder"] = export(
        text_dec, (sample_embeds, sample_pos), dynamic_shapes=dec_dyn, strict=True
    )
    print("  text_decoder exported")

    # --- 3. LM head ---
    print("\nExporting lm_head...")
    lm_head = LMHeadExport(model)
    lm_head.eval()

    sample_hidden = torch.randn(1, 1, model.config.dim, dtype=param_dtype)
    programs["lm_head"] = export(
        lm_head, (sample_hidden,), dynamic_shapes=None, strict=True
    )
    print("  lm_head exported")

    # --- 4. Decode audio frame ---
    print("\nExporting decode_audio_frame...")
    decode_frame = DecodeAudioFrameExport(model)
    decode_frame.eval()

    if qlinear:
        # The acoustic transformer's input_projection has a small K dimension
        # (n_acoustic_codebook, e.g. 36) that may not be divisible by
        # group_size. Quantize with a filter that skips incompatible layers.
        print(f"  Quantizing acoustic transformer ({qlinear})...")
        _quantize_with_skip(
            decode_frame,
            qlinear,
            qlinear_group_size,
        )

    sample_h = torch.randn(1, model.config.dim, dtype=param_dtype)
    sample_noise = torch.randn(1, model.config.n_acoustic_codebook, dtype=param_dtype)
    programs["decode_audio_frame"] = export(
        decode_frame, (sample_h, sample_noise), dynamic_shapes=None, strict=True
    )
    print("  decode_audio_frame exported")

    # --- 5. Audio token embedding ---
    print("\nExporting audio_token_embedding...")
    audio_tok = AudioTokenEmbeddingExport(model)
    audio_tok.eval()

    n_codebooks = model.config.n_acoustic_codebook + 1  # semantic + acoustic
    sample_codes = torch.zeros(1, n_codebooks, 4, dtype=torch.long)
    audio_seq_dim = Dim("audio_seq_len", min=1, max=max_seq_len)
    audio_dyn = {"codes": {2: audio_seq_dim}}
    programs["audio_token_embedding"] = export(
        audio_tok, (sample_codes,), dynamic_shapes=audio_dyn, strict=True
    )
    print("  audio_token_embedding exported")

    metadata = {
        "vocab_size": model.config.vocab_size,
        "dim": model.config.dim,
        "max_seq_len": max_seq_len,
        "n_acoustic_codebook": model.config.n_acoustic_codebook,
        "semantic_codebook_size": model.config.semantic_codebook_size,
        "acoustic_codebook_size": model.config.acoustic_codebook_size,
        "acoustic_decode_iters": 8,
        "cfg_alpha_x100": 120,  # 1.2 * 100 (integer for constant_methods)
        "noise_scale_x100": 100,  # 1.0 * 100
        "audio_tok_id": 24,  # Tekken special_ids.audio
    }

    return programs, metadata


# ---------------------------------------------------------------------------
# Metal linear bias decomposition
# ---------------------------------------------------------------------------


def _linear_bias_decomposition(input, weight, bias=None):
    """Decompose linear with bias into matmul + add for Metal compatibility."""
    weight_t = torch.ops.aten.t.default(weight)
    out = torch.ops.aten.matmul.default(input, weight_t)
    if bias is not None:
        return torch.ops.aten.add.Tensor(out, bias)
    return out


# ---------------------------------------------------------------------------
# Lower
# ---------------------------------------------------------------------------


def lower_to_executorch(programs, metadata, backend="metal"):
    if backend == "xnnpack":
        from executorch.backends.xnnpack.partition.xnnpack_partitioner import (
            XnnpackDynamicallyQuantizedPartitioner,
            XnnpackPartitioner,
        )

        print("\nLowering with XNNPACK...")
        partitioner = {
            key: [XnnpackDynamicallyQuantizedPartitioner(), XnnpackPartitioner()]
            for key in programs
        }
    elif backend == "metal":
        from executorch.backends.apple.metal.metal_backend import MetalBackend
        from executorch.backends.apple.metal.metal_partitioner import MetalPartitioner

        print("\nLowering with Metal...")

        updated = {}
        decomp_table = torch.export.default_decompositions()
        decomp_table[torch.ops.aten.linear.default] = _linear_bias_decomposition
        for key, ep in programs.items():
            updated[key] = ep.run_decompositions(decomp_table)
        programs = updated

        partitioner = {}
        for key in programs:
            compile_specs = [MetalBackend.generate_method_name_compile_spec(key)]
            partitioner[key] = [MetalPartitioner(compile_specs)]
    else:
        raise ValueError(f"Unsupported backend: {backend}. Use 'xnnpack' or 'metal'.")

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
    parser = argparse.ArgumentParser(description="Export Voxtral TTS to ExecuTorch")
    parser.add_argument(
        "--model-path",
        required=True,
        help="Directory with params.json + consolidated.safetensors",
    )
    parser.add_argument(
        "--backend",
        default="metal",
        choices=["xnnpack", "metal"],
        help="Backend (default: metal)",
    )
    parser.add_argument(
        "--output-dir",
        default="./voxtral_tts_exports",
        help="Output directory (default: ./voxtral_tts_exports)",
    )
    parser.add_argument(
        "--max-seq-len", type=int, default=4096, help="KV cache length (default: 4096)"
    )
    parser.add_argument(
        "--qlinear",
        default=None,
        choices=["8da4w", "fpa4w"],
        help="Quantize linear layers.",
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
        help="Quantize embedding layers.",
    )
    parser.add_argument(
        "--dtype",
        default="fp32",
        choices=["fp32", "bf16"],
        help="Model dtype (default: fp32).",
    )
    parser.add_argument(
        "--skip-codec",
        action="store_true",
        help="Skip codec decoder export (model.pte only).",
    )
    parser.add_argument(
        "--codec-chunk-size",
        type=int,
        default=375,
        help="Static chunk size for codec decoder (default: 375 frames).",
    )
    args = parser.parse_args()

    if args.qlinear == "fpa4w" and args.backend != "metal":
        parser.error("--qlinear=fpa4w requires --backend=metal")

    os.makedirs(args.output_dir, exist_ok=True)
    model_dtype = {"fp32": torch.float32, "bf16": torch.bfloat16}[args.dtype]

    print("Loading model...")
    model = load_model(
        args.model_path,
        max_seq_len=args.max_seq_len,
        dtype=model_dtype,
        backend=args.backend,
    )

    # Untie output/embedding weights before quantization
    if args.qlinear or args.qembedding:
        model.decoder.output.weight = nn.Parameter(
            model.decoder.tok_embeddings.weight.clone()
        )

    print("\nExporting components...")
    programs, metadata = export_model(
        model,
        args.max_seq_len,
        qlinear=args.qlinear,
        qlinear_group_size=args.qlinear_group_size,
        qembedding=args.qembedding,
        backend=args.backend,
    )

    et = lower_to_executorch(programs, metadata, backend=args.backend)

    pte_path = os.path.join(args.output_dir, "model.pte")
    print(f"\nSaving to {pte_path}...")
    with open(pte_path, "wb") as f:
        et.write_to_file(f)
    size_mb = os.path.getsize(pte_path) / (1024 * 1024)
    print(f"Saved {size_mb:.1f} MB")

    if et._tensor_data:
        et.write_tensor_data_to_file(args.output_dir)
        print(f"Saved tensor data to {args.output_dir}/")

    # --- Codec decoder (separate .pte) ---
    if not args.skip_codec:
        _export_codec(args, model_dtype)

    print("\nDone!")


def _export_codec(args, model_dtype):
    """Export codec decoder as a separate .pte file."""
    import json

    from executorch.examples.models.voxtral_tts.codec import load_codec_decoder

    model_dir = Path(args.model_path)
    with open(model_dir / "params.json") as f:
        params = json.load(f)

    codec_args_dict = params.get("multimodal", {}).get("audio_tokenizer_args", {})
    if not codec_args_dict:
        print("\nSkipping codec export: no audio_tokenizer_args in params.json")
        return

    # Codec always uses fp32 + XNNPACK regardless of the main model's backend/dtype.
    # The Metal AOTI backend has MPS buffer allocation issues with the codec's
    # ALiBi attention kernels. XNNPACK runs the codec on CPU which is fine since
    # the codec is only called once after generation completes.
    print("\nLoading codec decoder...")
    codec = load_codec_decoder(args.model_path, codec_args_dict, dtype=torch.float32)
    codec.eval()

    # Export with static shapes
    n_codebooks = 1 + codec.args.acoustic_dim  # semantic + acoustic
    T = args.codec_chunk_size
    sample_codes = torch.zeros(1, n_codebooks, T, dtype=torch.long)

    print("Exporting audio_decoder (xnnpack, fp32)...")
    codec_programs = {}
    codec_programs["audio_decoder"] = export(
        codec, (sample_codes,), dynamic_shapes=None, strict=True
    )
    print(f"  audio_decoder exported (static shape: codes={sample_codes.shape})")

    codec_metadata = {
        "downsample_factor": codec.downsample_factor,
        "n_codebooks": n_codebooks,
        "chunk_size": T,
        "sampling_rate": codec.args.sampling_rate,
    }

    codec_et = lower_to_executorch(codec_programs, codec_metadata, backend="xnnpack")

    codec_path = os.path.join(args.output_dir, "codec.pte")
    print(f"\nSaving codec to {codec_path}...")
    with open(codec_path, "wb") as f:
        codec_et.write_to_file(f)
    size_mb = os.path.getsize(codec_path) / (1024 * 1024)
    print(f"Saved {size_mb:.1f} MB")

    if codec_et._tensor_data:
        codec_et.write_tensor_data_to_file(args.output_dir)


if __name__ == "__main__":
    main()
