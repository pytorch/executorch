"""Export Gemma4 E2B multimodal model to a single ExecuTorch .pte.

Conforms to the standard MultimodalPrefiller ABI so a single .pte serves
text-only, image+text, and audio+text via ExecuTorch's MultimodalRunner.

Methods (all match extension/llm/runner/constants.h):
  token_embedding: (token_ids[1,S]) -> (1,S,1536)         scaled by sqrt(hidden)
  text_decoder:    (embeds[1,S,1536], cache_position[S])  -> logits[1,vocab]
                       ↑ dynamic S, stateful KV cache, batched prefill
  vision_encoder:  (image[1,3,672,960] float [0,1]) -> (1,280,1536)
                       ↑ patchification + position_ids baked into the graph
  audio_encoder:   (mel[1,128,200] float)            -> (1,50,1536)
                       ↑ channels-first mel (matches Voxtral convention)
  audio_preprocessor: (waveform[1,N_pcm]) -> (1,T,128)
                       ↑ helper for the C++ runner to convert WAV to mel

Calling convention matches MultimodalRunner::generate(prompt) for text-only
and MultimodalRunner::generate(vector<MultimodalInput>) for image/audio+text.

Usage:
  python -m executorch.examples.models.gemma4.export \\
    --hf-model ~/models/gemma-4-E2B-it \\
    --et-checkpoint ~/models/gemma-4-E2B-it/model_et.pth \\
    --output ./gemma4_multimodal.pte \\
    --backend xnnpack \\
    --variant e2b \\
    --max-seq-len 1024 --audio-frames 1976 \\
    --qmode 8da4w --group-size 32
"""

from __future__ import annotations

import argparse
import copy
import sys
from pathlib import Path
from typing import Dict

import torch
import torch.nn as nn
from torch.export import Dim, export

# ExecuTorch export infrastructure
from executorch.exir import EdgeCompileConfig, ExecutorchBackendConfig, to_edge_transform_and_lower
from executorch.exir.passes import MemoryPlanningPass
from executorch.exir.passes.init_mutable_pass import InitializedMutableBufferPass

# Gemma4 multimodal encoder wrappers (vision, audio)
from executorch.examples.models.gemma4.encoders import AudioEncoderExport, VisionEncoderExport
from executorch.examples.models.gemma4.audio_preprocessor import Gemma4AudioPreprocessor

# LLM export pipeline for stateful text decoder with KV cache
from executorch.examples.models.llama.export_llama_lib import _prepare_for_llama_export
from executorch.extension.llm.export.config.llm_config import LlmConfig, ModelType


# ---------------------------------------------------------------------------
# Token-embedding wrapper (token_embedding method)
# ---------------------------------------------------------------------------


class TokenEmbeddingExport(nn.Module):
    """Returns scaled token embeddings ready for text_decoder.

    Bakes in sqrt(hidden_size) scale to match HF Gemma4TextScaledWordEmbedding,
    so the output is in the same space as vision_encoder / audio_encoder soft
    tokens. The standard MultimodalPrefiller passes our output directly into
    text_decoder without further scaling.
    """

    def __init__(self, transformer):
        super().__init__()
        self.tok_embeddings = transformer.tok_embeddings
        self.scale = transformer.params.embedding_scale_factor

    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        h = self.tok_embeddings(token_ids)
        if self.scale != 1.0:
            h = h * self.scale
        return h


# ---------------------------------------------------------------------------
# Text-decoder wrapper (text_decoder method)
# ---------------------------------------------------------------------------


class TextDecoderExport(nn.Module):
    """Gemma4 text decoder with Approach C PLI via pli_token_ids.

    Inputs:
      inputs_embeds  (1, S, hidden) — scaled embeddings, dynamic S
      cache_position (1,) long      — start position (static size 1)
      pli_token_ids  (1, S) long    — token IDs for PLI:
                                      text → real token IDs
                                      image → 255999 (<|image>)
                                      audio → 256000 (<|audio>)

    Returns logits (1, vocab_size).

    PLI = pli_projection(h) + pli_embeddings(pli_token_ids).
    Gemma4DecoderRunner in main.cpp passes pli_token_ids at each decode step.
    """

    def __init__(self, transformer):
        super().__init__()
        self.transformer = transformer

    def forward(
        self,
        inputs_embeds: torch.Tensor,   # (1, S, hidden)
        cache_position: torch.Tensor,  # (1,) long
        pli_token_ids: torch.Tensor,   # (1, S) long
    ) -> torch.Tensor:
        return self.transformer(
            h=inputs_embeds,
            attn_options={
                "input_pos": cache_position,
                "pli_token_ids": pli_token_ids,
            },
        )


# ---------------------------------------------------------------------------
# Export helpers
# ---------------------------------------------------------------------------


def _partitioners(programs: dict, backend: str):
    if backend == "xnnpack":
        from executorch.backends.xnnpack.partition.xnnpack_partitioner import (
            XnnpackDynamicallyQuantizedPartitioner,
            XnnpackPartitioner,
        )
        return {k: [XnnpackDynamicallyQuantizedPartitioner(), XnnpackPartitioner()]
                for k in programs}
    return []


def _apply_encoder_quantization(model: nn.Module, mode: str, group_size: int = 128) -> nn.Module:
    """Quantize Linear layers in a vision/audio encoder via the upstream
    `extension/llm/export/quantize.py:quantize_model_` entry point.

    Same entry point used by `examples/models/qwen3_5_moe/export.py`; the
    reviewer feedback on D99603811 (mnachin) was specifically that
    encoder/model quantization should reuse this rather than build a
    parallel helper. `skip_incompatible_shapes=True` makes the walker
    silently skip Linears whose hidden dim doesn't divide `group_size`
    (typical for the per-head projection inside ViT/Conformer blocks).
    """
    from executorch.extension.llm.export.quantize import quantize_model_
    quantize_model_(
        model,
        qlinear_config=mode,
        qlinear_group_size=group_size,
        skip_incompatible_shapes=True,
    )
    return model


def lower_all(
    programs: Dict[str, torch.export.ExportedProgram],
    metadata: dict,
    backend: str,
) -> object:
    """Lower all exported programs to a single ExecuTorch program."""
    mutable_passes = [InitializedMutableBufferPass(["k_cache", "v_cache"])]
    et_prog = to_edge_transform_and_lower(
        programs,
        partitioner=_partitioners(programs, backend),
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
            passes=mutable_passes,
            memory_planning_pass=MemoryPlanningPass(
                alloc_graph_input=False,
                share_mutable_buffers=True,
            ),
            emit_mutable_buffer_names=True,
        )
    )


# ---------------------------------------------------------------------------
# Text backbone export via LLM pipeline (P0 fix)
# ---------------------------------------------------------------------------


def export_text_programs(
    et_checkpoint: str,
    et_params: str,
    max_seq_len: int,
    qmode: str | None = None,
    group_size: int = 32,
    embedding_quantize: str | None = None,
    quantize_kv_cache: bool = False,
    tied_embedding: bool = False,
    split_text_decoder: bool = False,
) -> Dict[str, torch.export.ExportedProgram]:
    """Export token_embedding + text_decoder with proper KV cache.

    Uses _prepare_for_llama_export which applies:
      - replace_kv_cache_with_custom_kv_cache
      - replace_sdpa_with_custom_op
    making the model stateful with mutable KV cache buffers.
    """
    cfg = LlmConfig()
    cfg.base.model_class = ModelType.gemma4
    cfg.base.params = et_params
    cfg.base.checkpoint = et_checkpoint
    cfg.model.use_kv_cache = True
    cfg.model.use_sdpa_with_kv_cache = True
    cfg.model.enable_dynamic_shape = True
    cfg.export.max_seq_length = max_seq_len
    cfg.export.max_context_length = max_seq_len
    if quantize_kv_cache:
        cfg.model.quantize_kv_cache = True
    if tied_embedding:
        # When the checkpoint has tied embed_tokens<->lm_head, this directs
        # the LLM-export pipeline to use TorchAO's shared embedding kernel.
        # The runtime needs the C++ runner with shared-embedding kernels
        # linked (which is what `make gemma4-cpu` provides today).
        cfg.backend.torchao.use_torchao_kernels_tied_embedding = True
    if qmode is not None:
        cfg.quantization.qmode = qmode
        cfg.quantization.group_size = group_size
    if embedding_quantize is not None:
        # Validate: format is "<bits>,<groupsize>"; groupsize must divide
        # the model's hidden dim or be 0 (no grouping). Common bug: passing
        # "8,1024" against E2B (hidden=1536) fails because 1536 % 1024 != 0.
        try:
            bits_s, group_s = embedding_quantize.split(",")
            bits = int(bits_s); group = int(group_s)
        except ValueError:
            raise ValueError(
                f"--embedding-quantize must be '<bits>,<groupsize>', got {embedding_quantize!r}"
            )
        if group != 0 and group < 0:
            raise ValueError(
                f"--embedding-quantize groupsize must be >= 0 (0 = no grouping), got {group}"
            )
        if bits not in (4, 8):
            raise ValueError(
                f"--embedding-quantize bits must be 4 or 8, got {bits}"
            )
        cfg.quantization.embedding_quantize = embedding_quantize

    print(
        f"  Preparing Gemma4 text backbone (qmode={qmode}, group_size={group_size}, "
        f"emb_q={embedding_quantize}, kv_quant={quantize_kv_cache}, "
        f"tied_emb={tied_embedding})..."
    )
    builder = _prepare_for_llama_export(cfg)
    # builder.model is the Transformer with custom KV cache ops applied.
    transformer = builder.model

    programs: Dict[str, torch.export.ExportedProgram] = {}

    # -- token_embedding --
    print("  Exporting token_embedding...")
    tok_emb = TokenEmbeddingExport(transformer).eval()
    S_dim = Dim("S_emb", min=1, max=max_seq_len)
    with torch.no_grad():
        programs["token_embedding"] = export(
            tok_emb,
            (torch.zeros(1, 4, dtype=torch.long),),
            dynamic_shapes={"token_ids": {1: S_dim}},
            strict=True,
        )
    print(f"  token_embedding: (1,S) -> (1,S,{transformer.params.dim})")

    # -- text_decoder (stateful KV cache, standard MultimodalPrefiller ABI) --
    # Standard LLMEdgeManager pattern: input_pos is static size 1 (the START
    # position); the model internally indexes positions [start_pos..start_pos+S].
    # `inputs_embeds` is dynamic-S to serve both batched prefill and single-token decode.
    # `populate_start_pos_or_cache_position` detects size==1 and passes [start_pos].
    print("  Exporting text_decoder (KV cache, dynamic S, 3-input Approach C PLI)...")
    txt_dec = TextDecoderExport(transformer).eval()
    dim = transformer.params.dim
    S_dec = Dim("S_dec", min=1, max=max_seq_len - 1)
    with torch.no_grad():
        programs["text_decoder"] = export(
            txt_dec,
            (
                torch.zeros(1, 4, dim),              # trace with S=4 > 1
                torch.zeros(1, dtype=torch.long),    # cache_position: (1,) static
                torch.zeros(1, 4, dtype=torch.long), # pli_token_ids: (1, S)
            ),
            dynamic_shapes=(
                {1: S_dec},   # inputs_embeds: dim 1 (S) dynamic
                {},           # cache_position: all dims static (size-1)
                {1: S_dec},   # pli_token_ids: dim 1 (S) dynamic, matches embeds
            ),
            strict=True,
        )
    print(f"  text_decoder: (1,S,{dim}) + (1,) + (1,S) -> (1,{transformer.vocab_size})  [PLI]")

    # Optionally also export specialized prefill (S>=2, dynamic) and decode
    # (S=1, static) methods alongside text_decoder. Mirrors the qwen3_5_moe
    # pattern (examples/models/qwen3_5_moe/export.py:634-669): the runner
    # calls `prefill` for the batched prompt and `decode` for each step,
    # giving downstream backends (XNNPACK, CUDA AOTI) a chance to specialize
    # per call site (e.g. tensor-core matmul vs vec-mat). The unified
    # text_decoder method is preserved for backward compatibility with v11
    # ptes (the runner falls back to it when prefill/decode are absent).
    if split_text_decoder:
        S_prefill = Dim("S_prefill", min=2, max=max_seq_len - 1)
        print("  Exporting prefill (batched, dynamic S>=2)...")
        with torch.no_grad():
            programs["prefill"] = export(
                txt_dec,
                (
                    torch.zeros(1, 4, dim),
                    torch.zeros(1, dtype=torch.long),
                    torch.zeros(1, 4, dtype=torch.long),
                ),
                dynamic_shapes=(
                    {1: S_prefill},
                    {},
                    {1: S_prefill},
                ),
                strict=True,
            )
        print(f"  prefill: (1,S>=2,{dim}) + (1,) + (1,S>=2) -> (1,{transformer.vocab_size})")

        print("  Exporting decode (single-token, static S=1)...")
        with torch.no_grad():
            programs["decode"] = export(
                txt_dec,
                (
                    torch.zeros(1, 1, dim),
                    torch.zeros(1, dtype=torch.long),
                    torch.zeros(1, 1, dtype=torch.long),
                ),
                strict=True,
            )
        print(f"  decode: (1,1,{dim}) + (1,) + (1,1) -> (1,{transformer.vocab_size})")

    # Carry metadata from the builder
    text_metadata = builder.metadata or {}
    return programs, text_metadata


# ---------------------------------------------------------------------------
# Main export
# ---------------------------------------------------------------------------


_VARIANT_CONFIGS = {
    "e2b": "e2b_config.json",
    "e4b": "e4b_config.json",
}


def _params_for_variant(variant: str) -> str:
    cfg_name = _VARIANT_CONFIGS[variant]
    return str(Path(__file__).parent / "config" / cfg_name)


def export_gemma4_multimodal(
    hf_model_dir: str,
    et_checkpoint: str,
    output_path: str,
    backend: str = "xnnpack",
    max_seq_len: int = 512,
    audio_frames: int = 200,
    qmode: str | None = None,
    group_size: int = 32,
    embedding_quantize: str | None = None,
    variant: str = "e2b",
    vision_quantize: str | None = None,
    audio_quantize: str | None = None,
    encoder_group_size: int = 128,
    quantize_kv_cache: bool = False,
    tied_embedding: bool = False,
    split_text_decoder: bool = False,
) -> None:
    hf_model_dir = Path(hf_model_dir)
    output_path = Path(output_path)
    if variant not in _VARIANT_CONFIGS:
        raise ValueError(
            f"--variant must be one of {sorted(_VARIANT_CONFIGS)}, got {variant!r}"
        )
    et_params = _params_for_variant(variant)

    print(f"Exporting Gemma4 multimodal to {output_path}")
    print(f"  variant: {variant} (params: {et_params})")
    print(f"  hf_model_dir: {hf_model_dir}")
    print(f"  et_checkpoint: {et_checkpoint}")
    print(f"  backend: {backend}")

    # ---- Load HF model for vision + audio encoders ----
    print("\nLoading HF model for vision/audio encoders...")
    from transformers import AutoModelForCausalLM, AutoConfig
    hf_model = AutoModelForCausalLM.from_pretrained(
        str(hf_model_dir), dtype=torch.float32
    ).eval()
    cfg = AutoConfig.from_pretrained(str(hf_model_dir), trust_remote_code=True)

    n_mels = 128       # mel spectrogram bins
    text_hidden = cfg.text_config.hidden_size   # 1536
    vocab_size = cfg.text_config.vocab_size     # 262144
    img_h, img_w = 672, 960  # HF image processor target resolution → 2520 patches → 280 soft tokens

    programs: Dict[str, torch.export.ExportedProgram] = {}

    # ---- 1. Vision encoder (standard ABI: raw image in, soft tokens out) ----
    print("\nExporting vision_encoder...")
    vis_enc = VisionEncoderExport(
        hf_model.model.vision_tower, hf_model.model.embed_vision
    ).eval()
    if vision_quantize:
        print(f"  applying linear quantization: {vision_quantize}")
        vis_enc = _apply_encoder_quantization(
            vis_enc, vision_quantize, group_size=encoder_group_size
        )
        vis_enc.eval()
    with torch.no_grad():
        programs["vision_encoder"] = export(
            vis_enc,
            (torch.zeros(1, 3, img_h, img_w),),  # raw image in [0, 1]
            strict=True,
        )
    print(f"  vision_encoder: (1,3,{img_h},{img_w}) -> (1,280,{text_hidden})")

    # ---- 2. Audio preprocessor (PCM → mel spectrogram) ----
    print("\nExporting audio_preprocessor...")
    audio_pre = Gemma4AudioPreprocessor().eval()
    T_pcm_dim = Dim("T_pcm", min=1600, max=480000)
    with torch.no_grad():
        programs["audio_preprocessor"] = export(
            audio_pre,
            (torch.zeros(1, 32000),),
            dynamic_shapes={"waveform": {1: T_pcm_dim}},
            strict=True,
        )
    print(f"  audio_preprocessor: (1,N_pcm) -> (1,T,{n_mels})")

    # ---- 3. Audio encoder (channels-first mel → soft tokens) ----
    # Static T_mel=200 (stride-48 conv constraint: T = 48*k - 40 with k=5).
    # Channels-first input matches Voxtral convention; encoder transposes internally.
    print("\nExporting audio_encoder...")
    aud_enc = AudioEncoderExport(
        hf_model.model.audio_tower, hf_model.model.embed_audio
    ).eval()
    if audio_quantize:
        print(f"  applying linear quantization: {audio_quantize}")
        aud_enc = _apply_encoder_quantization(
            aud_enc, audio_quantize, group_size=encoder_group_size
        )
        aud_enc.eval()
    with torch.no_grad():
        programs["audio_encoder"] = export(
            aud_enc,
            (torch.zeros(1, n_mels, audio_frames),),  # (1, 128, 200) channels-first
            strict=True,
        )
    print(f"  audio_encoder: (1,{n_mels},{audio_frames}) -> (1,{audio_frames//4},{text_hidden})")

    # Free HF model before loading LLM pipeline (saves ~20 GB RAM)
    del hf_model
    torch.cuda.empty_cache()

    # ---- 4+5. Token embedding + Text decoder (KV cache) ----
    print("\nExporting text backbone (token_embedding + text_decoder with KV cache)...")
    text_programs, text_metadata = export_text_programs(
        et_checkpoint=et_checkpoint,
        et_params=et_params,
        max_seq_len=max_seq_len,
        qmode=qmode,
        group_size=group_size,
        embedding_quantize=embedding_quantize,
        quantize_kv_cache=quantize_kv_cache,
        tied_embedding=tied_embedding,
        split_text_decoder=split_text_decoder,
    )
    programs.update(text_programs)

    # ---- Metadata ----
    # Merge text_metadata (has use_kv_cache=True, use_sdpa_with_kv_cache=True, etc.)
    # with multimodal-specific entries.
    metadata = {
        **text_metadata,
        "get_bos_id": 2,
        "get_eos_ids": [1, 106, 50],  # <eos>=1, <turn|>=106, id=50
        "get_vocab_size": vocab_size,
        "get_max_seq_len": max_seq_len,
        "get_max_context_len": max_seq_len,
    }
    # Ensure KV cache flags are present (needed by create_multimodal_runner)
    metadata.setdefault("use_kv_cache", True)
    metadata.setdefault("use_sdpa_with_kv_cache", True)
    metadata.setdefault("enable_dynamic_shape", False)

    print(f"\nMethods to export: {sorted(programs.keys())}")
    print(f"KV cache metadata: use_kv_cache={metadata.get('use_kv_cache')}")

    # ---- Lower to ExecuTorch ----
    print(f"\nLowering to ExecuTorch ({backend})...")
    et_prog = lower_all(programs, metadata, backend=backend)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_bytes = et_prog.buffer
    output_path.write_bytes(output_bytes)
    size_mb = len(output_bytes) / (1024 * 1024)
    print(f"\nSaved {output_path} ({size_mb:.1f} MB)")
    print("Methods:", sorted(programs.keys()))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--hf-model", required=True,
                        help="HF Gemma4 model dir (for vision/audio encoders)")
    parser.add_argument("--et-checkpoint", required=True,
                        help="ExecuTorch checkpoint path (~/.../model_et.pth)")
    parser.add_argument("--output", default="./gemma4_multimodal.pte")
    parser.add_argument("--backend", default="portable",
                        choices=["portable", "xnnpack"])
    parser.add_argument("--max-seq-len", type=int, default=512)
    parser.add_argument("--audio-frames", type=int, default=200,
                        help="Mel frames for fixed-shape audio encoder export (~2s)")
    parser.add_argument("--qmode", default=None,
                        choices=[None, "8da4w", "4w", "int8"],
                        help="Quantization mode for text backbone (default: FP32)")
    parser.add_argument("--group-size", type=int, default=32,
                        help="Group size for weight quantization (default: 32)")
    parser.add_argument("--embedding-quantize", default=None,
                        help="Embedding quantization, format '<bits>,<groupsize>'. "
                             "Recommended: '8,0' (per-channel, works for any hidden size). "
                             "If you set a non-zero groupsize, it must divide the model's "
                             "hidden_size (E2B=1536, E4B=2560).")
    parser.add_argument("--variant", default="e2b",
                        choices=sorted(_VARIANT_CONFIGS.keys()),
                        help="Model variant to export (selects config file).")
    parser.add_argument("--vision-quantize", default=None,
                        choices=[None, "8da4w", "8da8w"],
                        help="Vision encoder linear quantization (default: FP32). "
                             "8da8w gives ~50%% size reduction with negligible quality loss.")
    parser.add_argument("--audio-quantize", default=None,
                        choices=[None, "8da4w", "8da8w"],
                        help="Audio encoder linear quantization (default: FP32). "
                             "8da4w shrinks the audio encoder ~75%%.")
    parser.add_argument("--encoder-group-size", type=int, default=128,
                        help="Group size for 8da4w encoder weight quantization "
                             "(default: 128; must divide encoder hidden dims).")
    parser.add_argument("--quantize-kv-cache", action="store_true", default=False,
                        help="Quantize the KV cache to INT8 per-token. "
                             "Reduces decode-time memory for long sequences.")
    parser.add_argument("--tied-embedding", action="store_true", default=False,
                        help="Use TorchAO shared-embedding kernel (assumes the "
                             "checkpoint has tied embed_tokens<->lm_head; the "
                             "C++ runner must be built with shared-embedding kernels).")
    parser.add_argument("--split-text-decoder", action="store_true", default=False,
                        help="Also export specialized 'prefill' (dynamic S>=2) "
                             "and 'decode' (static S=1) methods alongside "
                             "text_decoder. Mirrors qwen3_5_moe's pattern; the "
                             "runner uses these when present. Default off for "
                             "backward compatibility with v11 ptes.")
    args = parser.parse_args()

    export_gemma4_multimodal(
        hf_model_dir=args.hf_model,
        et_checkpoint=args.et_checkpoint,
        output_path=args.output,
        backend=args.backend,
        max_seq_len=args.max_seq_len,
        audio_frames=args.audio_frames,
        qmode=args.qmode,
        group_size=args.group_size,
        embedding_quantize=args.embedding_quantize,
        variant=args.variant,
        vision_quantize=args.vision_quantize,
        audio_quantize=args.audio_quantize,
        encoder_group_size=args.encoder_group_size,
        quantize_kv_cache=args.quantize_kv_cache,
        tied_embedding=args.tied_embedding,
        split_text_decoder=args.split_text_decoder,
    )


if __name__ == "__main__":
    main()
