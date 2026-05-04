# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Export Voxtral-4B-TTS-2603 to ExecuTorch.

Produces two .pte files:
  model.pte (multi-method, like voxtral_realtime):
    - token_embedding:  token_ids (1, S) -> embeds (1, S, 3072)
    - audio_token_embedding: codes (1, 37, S) -> embeds (1, S, 3072)
    - text_decoder:     embeds (1, S, 3072) + cache_pos (S,) -> hidden (1, S, 3072)
    - semantic_head:    hidden (1, 3072) -> code (1,)
    - predict_velocity: x_t (1, 36) + t_idx (1,) + hidden (1, 3072) -> v_t (1, 36)

  codec_decoder.pte (single method):
    - forward: codes (1, 37, T) -> waveform (1, 1, T*1920)

Usage:
    python export_voxtral_tts.py --model-path ~/models/Voxtral-4B-TTS-2603 --qlinear 4w
    python export_voxtral_tts.py --model-path ~/models/Voxtral-4B-TTS-2603 --qlinear 4w --qembedding 4w
"""

import argparse
import os
from pathlib import Path

import torch
import torch.nn as nn
from executorch.exir import (
    EdgeCompileConfig,
    ExecutorchBackendConfig,
    to_edge_transform_and_lower,
)
from executorch.exir.passes import MemoryPlanningPass
from executorch.exir.passes.init_mutable_pass import InitializedMutableBufferPass
from executorch.extension.llm.export.quantize import quantize_model_
from torch.export import Dim, export

try:
    from .model import load_model
    from .voice import load_voice_from_model_dir
except ImportError:
    from model import load_model
    from voice import load_voice_from_model_dir

# ---------------------------------------------------------------------------
# Export wrappers
# ---------------------------------------------------------------------------


class TokenEmbeddingExport(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.tok_embeddings = model.decoder.tok_embeddings

    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        return self.tok_embeddings(token_ids)


class AudioTokenEmbeddingExport(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.audio_token_embedding = model.audio_token_embedding

    def forward(self, codes: torch.Tensor) -> torch.Tensor:
        return self.audio_token_embedding(codes)


class TextDecoderExport(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.decoder = model.decoder

    def forward(
        self, input_embeds: torch.Tensor, cache_position: torch.Tensor
    ) -> torch.Tensor:
        return self.decoder(input_embeds, cache_position)


class SemanticHeadExport(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.flow_head = model.flow_head

    def forward(self, hidden: torch.Tensor) -> torch.Tensor:
        return self.flow_head.semantic_logits(hidden)


class PredictVelocityExport(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.flow_head = model.flow_head

    def forward(
        self,
        x_t: torch.Tensor,
        t_idx: torch.Tensor,
        hidden: torch.Tensor,
    ) -> torch.Tensor:
        return self.flow_head.predict_velocity(x_t, t_idx, hidden)


class CodecDecoderExport(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.codec_decoder = model.codec_decoder

    def forward(self, codes: torch.Tensor) -> torch.Tensor:
        return self.codec_decoder(codes)


# ---------------------------------------------------------------------------
# Quantization policy
# ---------------------------------------------------------------------------


def _export_lm_pte(model, args, device: str) -> None:
    """Export model.pte (LM + flow head, 5 methods)."""
    print("\n" + "=" * 60)
    print("Exporting model.pte (5 methods)")
    print("=" * 60)
    programs, metadata = export_model(
        model,
        args.max_seq_len,
        streaming=args.streaming,
        device=device,
    )
    et_model = lower_to_executorch(programs, metadata, backend=args.backend)

    model_pte = os.path.join(args.output_dir, "model.pte")
    print(f"\nSaving to {model_pte}...")
    with open(model_pte, "wb") as f:
        et_model.write_to_file(f)
    size_mb = os.path.getsize(model_pte) / (1024 * 1024)
    print(f"Saved model.pte ({size_mb:.1f} MB)")

    # CUDA backend emits a .ptd containing the AOTI .so + weights.
    if et_model._tensor_data:
        et_model.write_tensor_data_to_file(args.output_dir)
        print(f"Saved model tensor data to {args.output_dir}/")


def _export_codec_pte(model, args, device: str) -> None:
    """Export codec_decoder.pte (single forward method).

    Codec convs are expressed as unfold + matmul / matmul + Fold
    (model.py:_conv1d_as_matmul / _conv_transpose1d_as_matmul) so AOTI's CUDA
    backend can lower them via Triton mm kernels. CodecAttention uses an
    additive ALiBi mask which is fine for ATen SDPA when triton_kernel_mode=OFF.
    """
    print("\n" + "=" * 60)
    print("Exporting codec_decoder.pte")
    print("=" * 60)
    codec_programs, codec_metadata = export_codec_decoder(
        model,
        max_codec_frames=args.max_codec_frames,
        qlinear_codec=args.qlinear_codec,
        qlinear_codec_group_size=args.qlinear_codec_group_size,
        device=device,
    )
    codec_backend = args.backend
    codec_triton_mode = "OFF" if codec_backend in ("cuda", "cuda-windows") else "ON"
    # triton_kernel_mode is a no-op for non-CUDA backends (xnnpack, mlx, portable).
    et_codec = lower_to_executorch(
        codec_programs,
        codec_metadata,
        backend=codec_backend,
        triton_kernel_mode=codec_triton_mode,
    )

    codec_pte = os.path.join(args.output_dir, "codec_decoder.pte")
    print(f"\nSaving to {codec_pte}...")
    with open(codec_pte, "wb") as f:
        et_codec.write_to_file(f)
    size_mb = os.path.getsize(codec_pte) / (1024 * 1024)
    print(f"Saved codec_decoder.pte ({size_mb:.1f} MB)")

    if et_codec._tensor_data:
        # Rename the codec's data blob so it doesn't collide with the LM's
        # `aoti_cuda_blob.ptd` (both default to the same filename).
        renamed = {}
        for k, v in et_codec._tensor_data.items():
            new_key = "codec_" + k if ("aoti_cuda" in k or k.startswith("model")) else k
            renamed[new_key] = v
        et_codec._tensor_data = renamed
        et_codec.write_tensor_data_to_file(args.output_dir)
        print(f"Saved codec tensor data to {args.output_dir}/")


def _apply_cuda_arg_defaults(parser, args, backend_for_export: str) -> None:
    """Auto-set CUDA-specific defaults: tile_packed_to_4d packing + bf16 dtype.

    Both are required for the AOTI _weight_int4pack_mm kernel path. Promoted
    automatically (with a print) so users don't have to remember the rule;
    explicit incompatible values are rejected via parser.error().
    """
    if backend_for_export == "cuda" and args.qlinear == "4w":
        if args.qlinear_packing_format is None:
            args.qlinear_packing_format = "tile_packed_to_4d"
            print(
                "Auto-selected --qlinear-packing-format=tile_packed_to_4d "
                "(required by _weight_int4pack_mm on CUDA)."
            )
        elif args.qlinear_packing_format != "tile_packed_to_4d":
            parser.error(
                "--qlinear=4w on CUDA requires "
                "--qlinear-packing-format=tile_packed_to_4d"
            )

    if backend_for_export == "cuda" and args.qlinear and args.dtype == "fp32":
        print(
            f"Auto-promoting --dtype to bf16 (CUDA --qlinear={args.qlinear} "
            "needs bf16 weights for the int-pack kernels)."
        )
        args.dtype = "bf16"


def _apply_mlx_arg_defaults(args, backend_for_export: str) -> None:
    """Choose MLX-compatible quantization defaults."""
    if (
        backend_for_export == "mlx"
        and args.qembedding is not None
        and args.qembedding_group_size is None
    ):
        args.qembedding_group_size = 128
        print(
            "Auto-selected --qembedding-group-size=128 "
            "(MLX embedding lowering requires grouped quantization)."
        )


def _validate_mlx_args(parser, args, backend_for_export: str) -> None:
    """Reject MLX combinations that produce runner-incompatible artifacts."""
    if backend_for_export == "mlx" and args.qlinear_codec is not None:
        parser.error(
            "--backend=mlx does not currently support --qlinear-codec because "
            "native MLX codec lowering is only validated for the unquantized "
            "codec decoder."
        )


def resolve_effective_quantization(
    *,
    backend: str,
    qlinear: str | None,
    qembedding: str | None,
) -> dict[str, str | None]:
    warning = None
    effective_qembedding = qembedding
    if backend == "xnnpack" and qembedding is not None:
        warning = (
            "XNNPACK runtime does not register quantized embedding kernels yet; "
            "disabling embedding quantization for this export."
        )
        effective_qembedding = None
    return {
        "qlinear": qlinear,
        "qembedding": effective_qembedding,
        "warning": warning,
    }


# ---------------------------------------------------------------------------
# Export functions
# ---------------------------------------------------------------------------


def export_model(
    model,
    max_seq_len,
    streaming=False,
    device="cpu",
):
    """Export LLM + acoustic head as a single multi-method model.pte.

    Quantization must be applied to the model BEFORE calling this function.
    """
    programs = {}
    param_dtype = next(model.parameters()).dtype
    config = model.config

    # 1. Text decoder
    print("\nExporting text_decoder...")
    text_decoder = TextDecoderExport(model)
    text_decoder.eval()
    seq_dim = Dim("seq_len", min=1, max=max_seq_len)
    sample_embeds = torch.randn(1, 4, config.dim, dtype=param_dtype, device=device)
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
    print(f"  text_decoder exported (sample: {sample_embeds.shape})")

    # 2. Token embedding
    print("\nExporting token_embedding...")
    tok_emb = TokenEmbeddingExport(model)
    tok_emb.eval()
    tok_seq_dim = Dim("tok_seq_len", min=1, max=max_seq_len)
    sample_ids = torch.tensor([[0, 1, 2, 3]], dtype=torch.long, device=device)
    programs["token_embedding"] = export(
        tok_emb,
        (sample_ids,),
        dynamic_shapes={"token_ids": {1: tok_seq_dim}},
        strict=True,
    )
    print(f"  token_embedding exported (sample: {sample_ids.shape})")

    # 3. Audio token embedding
    print("\nExporting audio_token_embedding...")
    audio_tok_emb = AudioTokenEmbeddingExport(model)
    audio_tok_emb.eval()
    sample_audio_codes = torch.zeros(
        1, config.n_codebooks, 1, dtype=torch.long, device=device
    )
    programs["audio_token_embedding"] = export(
        audio_tok_emb,
        (sample_audio_codes,),
        strict=True,
    )
    print("  audio_token_embedding exported " f"(sample: {sample_audio_codes.shape})")

    # 4. Semantic head
    print("\nExporting semantic_head...")
    sem_head = SemanticHeadExport(model)
    sem_head.eval()
    sample_hidden = torch.randn(1, config.dim, dtype=param_dtype, device=device)
    programs["semantic_head"] = export(
        sem_head,
        (sample_hidden,),
        strict=True,
    )
    print(f"  semantic_head exported (sample: {sample_hidden.shape})")

    # 5. Predict velocity
    print("\nExporting predict_velocity...")
    vel_pred = PredictVelocityExport(model)
    vel_pred.eval()
    sample_xt = torch.randn(1, config.acoustic_dim, dtype=param_dtype, device=device)
    sample_tidx = torch.tensor([0], dtype=torch.long, device=device)
    sample_hv = torch.randn(1, config.dim, dtype=param_dtype, device=device)
    programs["predict_velocity"] = export(
        vel_pred,
        (sample_xt, sample_tidx, sample_hv),
        strict=True,
    )
    print("  predict_velocity exported")

    # Tells the runner whether to stage fp32 buffers as bf16 before each
    # AOTI execute (1 = bf16 model, 0 = fp32 model). bf16 happens for
    # quantized exports (--qlinear); fp32 is the default mixed-precision path.
    lm_input_is_bf16 = 1 if param_dtype == torch.bfloat16 else 0

    # Determine the default voice embedding length from the real voice asset
    # instead of baking in casual_male-specific metadata.
    voice_embed_len = 0
    model_dir = Path(model.config_path) if hasattr(model, "config_path") else None
    if model_dir:
        try:
            v, _ = load_voice_from_model_dir(
                model_dir,
                None,
                dim=config.dim,
            )
            voice_embed_len = v.shape[0]
        except Exception:
            voice_embed_len = 0

    metadata = {
        "sample_rate": config.sampling_rate,
        "n_decoding_steps": config.n_decoding_steps,
        "cfg_alpha_x100": int(config.cfg_alpha * 100),
        "n_acoustic_codebook": config.acoustic_dim,
        "semantic_codebook_size": config.semantic_codebook_size,
        "acoustic_levels": config.acoustic_levels,
        "vocab_size": config.vocab_size,
        "max_seq_len": max_seq_len,
        "dim": config.dim,
        "downsample_factor": config.downsample_factor,
        "n_codebooks": config.n_codebooks,
        "end_audio_code": 1,
        "empty_audio_code": 0,
        "n_special_tokens": 2,
        "streaming": 1 if streaming else 0,
        "streaming_chunk_frames": 25,
        "streaming_initial_chunk": 5,
        "streaming_left_context": 25,
        "audio_token_id": config.audio_token_id,
        "begin_audio_token_id": config.begin_audio_token_id,
        "text_to_audio_token_id": config.text_to_audio_token_id,
        "repeat_audio_text_token_id": config.repeat_audio_text_token_id,
        "voice_embed_len": voice_embed_len,
        "lm_input_is_bf16": lm_input_is_bf16,
    }

    return programs, metadata


def export_codec_decoder(
    model,
    max_codec_frames=256,
    qlinear_codec=None,
    qlinear_codec_group_size=None,
    device="cpu",
):
    """Export codec decoder as a separate .pte."""
    from executorch.extension.llm.export.quantize import quantize_model_

    config = model.config

    print("\nExporting codec_decoder...")
    codec_dec = CodecDecoderExport(model)
    codec_dec.eval()

    if qlinear_codec:
        print(f"  Quantizing codec ({qlinear_codec})...")
        quantize_model_(
            codec_dec,
            qlinear_config=qlinear_codec,
            qlinear_group_size=qlinear_codec_group_size,
        )

    sample_codes = torch.zeros(
        1, config.n_codebooks, max_codec_frames, dtype=torch.long, device=device
    )
    # Static export: the codec's transformer/conv stages introduce tight
    # divisibility constraints under dynamic_shapes (upsample stride/kernel
    # math). Keeping the input static at max_codec_frames avoids those
    # constraint violations. The runner pads to max_codec_frames, but the
    # codec's transformer is only locally bidirectional (window<=16) so the
    # ALiBi-windowed attention contaminates a small boundary region; choose
    # max_codec_frames close to the expected per-utterance frame count to
    # minimize how many trailing zero codes the model attends to.
    programs = {"forward": export(codec_dec, (sample_codes,), strict=True)}
    print(
        f"  codec_decoder exported (codes: {sample_codes.shape}, "
        f"waveform: {max_codec_frames * config.downsample_factor} samples)"
    )

    metadata = {
        "max_codec_frames": max_codec_frames,
        "downsample_factor": config.downsample_factor,
        "n_codebooks": config.n_codebooks,
        "sample_rate": config.sampling_rate,
        "codec_supports_exact_frames": 0,
    }

    return programs, metadata


def apply_model_quantization(
    model,
    *,
    qlinear: str | None,
    qlinear_group_size: int | None,
    qlinear_packing_format: str | None,
    qembedding: str | None,
    qembedding_group_size: int | None,
    decoder_qlinear_scope: str = "all",
) -> None:
    if qlinear:
        qlinear_kwargs = {
            "qlinear_config": qlinear,
            "qlinear_group_size": qlinear_group_size,
            "qlinear_packing_format": qlinear_packing_format,
        }
        if decoder_qlinear_scope == "all":
            quantize_model_(model.decoder, **qlinear_kwargs)
        elif decoder_qlinear_scope == "attention":
            for layer in model.decoder.layers:
                quantize_model_(layer.attention, **qlinear_kwargs)
        elif decoder_qlinear_scope == "feed_forward":
            for layer in model.decoder.layers:
                quantize_model_(layer.feed_forward, **qlinear_kwargs)
        elif decoder_qlinear_scope != "none":
            raise ValueError(
                f"Unsupported decoder_qlinear_scope: {decoder_qlinear_scope}"
            )
        quantize_model_(
            model.flow_head,
            qlinear_config=qlinear,
            qlinear_group_size=qlinear_group_size,
            qlinear_packing_format=qlinear_packing_format,
            skip_incompatible_shapes=True,
        )

    if qembedding:
        tok_emb_wrapper = TokenEmbeddingExport(model)
        quantize_model_(
            tok_emb_wrapper,
            qembedding_config=qembedding,
            qembedding_group_size=qembedding_group_size,
        )
        audio_tok_emb_wrapper = AudioTokenEmbeddingExport(model)
        quantize_model_(
            audio_tok_emb_wrapper,
            qembedding_config=qembedding,
            qembedding_group_size=qembedding_group_size,
        )


def lower_to_executorch(programs, metadata, backend="xnnpack", triton_kernel_mode="ON"):
    """Lower exported programs to ExecuTorch.

    Args:
        triton_kernel_mode: For CUDA backend only. "ON" replaces ATen SDPA with
            Triton sdpa kernel (required for the LM decoder). "OFF" disables
            replacement so the codec's additive ALiBi mask SDPA can lower
            (Triton sdpa kernel only accepts bool masks).
    """
    mutable_buffer_passes = [InitializedMutableBufferPass(["k_cache", "v_cache"])]
    transform_passes = None
    if backend == "xnnpack":
        from executorch.backends.xnnpack.partition.xnnpack_partitioner import (
            XnnpackDynamicallyQuantizedPartitioner,
            XnnpackPartitioner,
        )

        print(f"\nLowering to ExecuTorch with XNNPACK ({len(programs)} methods)...")
        partitioner = {
            key: [XnnpackDynamicallyQuantizedPartitioner(), XnnpackPartitioner()]
            for key in programs
        }
    elif backend in ("cuda", "cuda-windows"):
        from executorch.backends.cuda.cuda_backend import CudaBackend
        from executorch.backends.cuda.cuda_partitioner import CudaPartitioner
        from executorch.exir.backend.compile_spec_schema import CompileSpec

        print(
            f"\nLowering to ExecuTorch with CUDA "
            f"{'(Windows) ' if backend == 'cuda-windows' else ''}"
            f"({len(programs)} methods, triton_kernel_mode={triton_kernel_mode})..."
        )
        # NB: conv1d_to_conv2d is applied inside CudaBackend.preprocess via its
        # decomposition_table. Doing it here too triggers an extra run_decompositions
        # pass that leaks unbacked symbols on the 26-layer Mistral text_decoder.

        partitioner = {}
        for key in programs:
            compile_specs = [CudaBackend.generate_method_name_compile_spec(key)]
            compile_specs.append(
                CompileSpec("triton_kernel_mode", triton_kernel_mode.encode("utf-8"))
            )
            if backend == "cuda-windows":
                compile_specs.append(CompileSpec("platform", b"windows"))
            partitioner[key] = [CudaPartitioner(compile_specs)]
    elif backend == "mlx":
        from executorch.backends.mlx.partitioner import MLXPartitioner
        from executorch.backends.mlx.passes import get_default_passes

        print(f"\nLowering to ExecuTorch with MLX ({len(programs)} methods)...")
        # One fresh partitioner per method (LM, audio_token_embedding,
        # semantic_head, predict_velocity, codec_decoder, ...).
        partitioner = {key: [MLXPartitioner()] for key in programs}
        # MLX requires reshape/permute/dtype canonicalization before lowering.
        transform_passes = get_default_passes()
    else:
        print(f"\nLowering to ExecuTorch (portable, {len(programs)} methods)...")
        partitioner = []

    et_prog = to_edge_transform_and_lower(
        programs,
        transform_passes=transform_passes,
        partitioner=partitioner,
        compile_config=EdgeCompileConfig(
            _check_ir_validity=False,
            _skip_dim_order=True,
        ),
        constant_methods=metadata,
    )

    if backend == "mlx":
        # MLX uses its own KV cache semantics and does not need the CUDA-only
        # mutable-buffer sharing / name emission paths. Match the
        # voxtral_realtime MLX export config.
        return et_prog.to_executorch(
            config=ExecutorchBackendConfig(
                extract_delegate_segments=True,
                do_quant_fusion_and_const_prop=True,
                memory_planning_pass=MemoryPlanningPass(alloc_graph_input=False),
            ),
        )

    return et_prog.to_executorch(
        config=ExecutorchBackendConfig(
            extract_delegate_segments=True,
            do_quant_fusion_and_const_prop=True,
            passes=mutable_buffer_passes,
            memory_planning_pass=MemoryPlanningPass(
                alloc_graph_input=False,
                share_mutable_buffers=True,
            ),
            emit_mutable_buffer_names=True,
        ),
    )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    import sys

    parser = argparse.ArgumentParser(description="Export Voxtral TTS to ExecuTorch")
    parser.add_argument(
        "--model-path",
        required=True,
        help="Directory with params.json + consolidated.safetensors",
    )
    parser.add_argument(
        "--backend",
        default="xnnpack",
        choices=["portable", "xnnpack", "cuda", "cuda-windows", "mlx"],
        help="Backend (default: xnnpack). cuda/cuda-windows compile via "
        "AOTInductor and emit model.pte + model.ptd. mlx targets Apple "
        "Silicon GPU via the MLX delegate (Darwin only).",
    )
    parser.add_argument(
        "--output-dir",
        default="./voxtral_tts_exports",
        help="Output directory (default: ./voxtral_tts_exports)",
    )
    parser.add_argument(
        "--export-target",
        default="all",
        choices=["all", "model", "codec"],
        help="Which artifacts to export (default: all).",
    )
    parser.add_argument(
        "--max-seq-len",
        type=int,
        default=4096,
        help="KV cache length (default: 4096)",
    )
    parser.add_argument(
        "--max-codec-frames",
        type=int,
        default=256,
        help="Max codec frames for decoder (default: 256 = ~20s audio)",
    )
    parser.add_argument(
        "--qlinear",
        default=None,
        choices=["4w", "8w", "8da4w", "8da8w"],
        help="Quantize ALL linear layers (LLM + acoustic head).",
    )
    parser.add_argument(
        "--qlinear-group-size",
        type=int,
        default=None,
        help="Group size for linear quantization.",
    )
    parser.add_argument(
        "--qlinear-packing-format",
        default=None,
        help="Packing format for 4w quantization.",
    )
    parser.add_argument(
        "--decoder-qlinear-scope",
        default="all",
        choices=["all", "attention", "feed_forward", "none"],
        help="Limit decoder linear quantization to a specific decoder sub-scope.",
    )
    parser.add_argument(
        "--qlinear-codec",
        default=None,
        choices=["4w", "8w"],
        help="Quantize codec decoder linear layers.",
    )
    parser.add_argument(
        "--qlinear-codec-group-size",
        type=int,
        default=None,
        help="Group size for codec linear quantization.",
    )
    parser.add_argument(
        "--qembedding",
        default=None,
        choices=["4w", "8w"],
        help="Quantize embedding layers.",
    )
    parser.add_argument(
        "--qembedding-group-size",
        type=int,
        default=None,
        help="Group size for embedding quantization.",
    )
    parser.add_argument(
        "--streaming",
        action="store_true",
        help="Enable streaming codec chunking metadata.",
    )
    parser.add_argument(
        "--dtype",
        default="fp32",
        choices=["fp32", "bf16"],
        help="Model dtype (default: fp32).",
    )
    args = parser.parse_args()

    backend_for_export = "cuda" if args.backend == "cuda-windows" else args.backend
    if backend_for_export == "mlx" and sys.platform != "darwin":
        parser.error("--backend=mlx requires macOS (Apple Silicon)")
    _apply_cuda_arg_defaults(parser, args, backend_for_export)
    _apply_mlx_arg_defaults(args, backend_for_export)
    _validate_mlx_args(parser, args, backend_for_export)

    os.makedirs(args.output_dir, exist_ok=True)
    model_dtype = {"fp32": torch.float32, "bf16": torch.bfloat16}[args.dtype]
    device = "cuda" if backend_for_export == "cuda" else "cpu"

    sys.stdout.reconfigure(line_buffering=True)

    print("Loading model...")
    model = load_model(
        args.model_path,
        max_seq_len=args.max_seq_len,
        dtype=model_dtype,
        backend=backend_for_export,
    )
    model.config_path = Path(args.model_path)

    if device == "cuda":
        print("Moving model to CUDA...")
        model.cuda()

    quant_plan = resolve_effective_quantization(
        backend=backend_for_export,
        qlinear=args.qlinear,
        qembedding=args.qembedding,
    )
    effective_qlinear = quant_plan["qlinear"]
    effective_qembedding = quant_plan["qembedding"]
    if quant_plan["warning"]:
        print(f"\nWarning: {quant_plan['warning']}")

    if effective_qlinear or effective_qembedding:
        if effective_qlinear:
            print(
                f"\nQuantizing linear layers ({effective_qlinear}, "
                f"decoder scope={args.decoder_qlinear_scope})..."
            )
        if effective_qembedding:
            print(f"Quantizing embedding ({effective_qembedding})...")
        apply_model_quantization(
            model,
            qlinear=effective_qlinear,
            qlinear_group_size=args.qlinear_group_size,
            qlinear_packing_format=args.qlinear_packing_format,
            qembedding=effective_qembedding,
            qembedding_group_size=args.qembedding_group_size,
            decoder_qlinear_scope=args.decoder_qlinear_scope,
        )

    if args.export_target in ("all", "model"):
        _export_lm_pte(model, args, device)

    if args.export_target in ("all", "codec"):
        _export_codec_pte(model, args, device)

    print("\n" + "=" * 60)
    print("DONE")
    print("=" * 60)
    for f in sorted(os.listdir(args.output_dir)):
        if f.endswith(".pte"):
            s = os.path.getsize(os.path.join(args.output_dir, f)) / (1024 * 1024)
            print(f"  {f}: {s:.1f} MB")


if __name__ == "__main__":
    main()
