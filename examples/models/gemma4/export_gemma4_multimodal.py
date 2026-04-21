"""Export Gemma4 E2B multimodal model to a single ExecuTorch .pte.

Produces gemma4_multimodal.pte with five methods:

  vision_encoder:     (pixel_values[1,2520,768], pixel_position_ids[1,2520,2]) -> (256,1536)
  audio_preprocessor: (waveform[1,N_samples]) -> mel_features[1,T,128]
  audio_encoder:      (mel_features[1,T,128]) -> (1,T//4,1536)
  token_embedding:    (token_ids[1,S]) -> (1,S,1536)
  text_decoder:       (inputs_embeds[1,S,1536], input_pos[S]) -> (1,vocab_size)
                        ↑ stateful with mutable KV-cache buffers (use_kv_cache=True)

MultimodalRunner expects exactly these method names. The text_decoder is
exported via the LLM export pipeline with proper KV-cache source transforms
so that decode can run step-by-step after prefill.

Usage:
  python export_gemma4_multimodal.py \\
    --hf-model ~/models/gemma-4-E2B-it \\
    --et-checkpoint ~/models/gemma-4-E2B-it/model_et.pth \\
    --output ./gemma4_multimodal.pte \\
    --backend portable
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
    """Wraps tok_embeddings from the KV-cache-prepared Gemma4 transformer.

    Returns raw (unscaled) embeddings. The C++ runner applies the embedding
    scale (sqrt(hidden_size) ≈ 39.19) after calling this method. Vision/audio
    soft tokens from their respective encoders must NOT be scaled.
    """

    def __init__(self, transformer):
        super().__init__()
        self.tok_embeddings = transformer.tok_embeddings

    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        return self.tok_embeddings(token_ids)


# ---------------------------------------------------------------------------
# Text-decoder wrapper (text_decoder method)
# ---------------------------------------------------------------------------


class TextDecoderExport(nn.Module):
    """Wraps the KV-cache-prepared Gemma4 Transformer for stateful decode.

    Inputs:
      inputs_embeds (1, 1, hidden): single-token embedding (scaled by sqrt(hidden))
      input_pos     (1,) long:      KV cache position index
      pli_token_ids (1, 1) long:    token ID for PLI computation
                                    (<|image>=255999, <|audio>=256000 for soft tokens)

    Returns logits (1, vocab_size) for the last position.

    PLI (Per-Layer Input) is computed inside the transformer from both the
    input embeddings (pli_projection of h) and the pli_token_ids. This matches
    HF Gemma4's unified multimodal forward pass exactly.
    """

    def __init__(self, transformer):
        super().__init__()
        self.transformer = transformer

    def forward(
        self,
        inputs_embeds: torch.Tensor,   # (1, 1, hidden)
        input_pos: torch.Tensor,        # (1,) long
        pli_token_ids: torch.Tensor,    # (1, 1) long
    ) -> torch.Tensor:
        return self.transformer(
            h=inputs_embeds,
            attn_options={
                "input_pos": input_pos,
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
    cfg.model.enable_dynamic_shape = False
    cfg.export.max_seq_length = max_seq_len
    cfg.export.max_context_length = max_seq_len

    print("  Preparing Gemma4 text backbone with KV-cache transforms...")
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

    # -- text_decoder (stateful KV cache) with PLI via pli_token_ids --
    # Takes 3 inputs: inputs_embeds, input_pos, pli_token_ids.
    # PLI is computed inside the transformer from both h (projection path) and
    # pli_token_ids (embedding path), combining them as in HF Gemma4.
    print("  Exporting text_decoder (KV cache, single-token decode, with PLI)...")
    txt_dec = TextDecoderExport(transformer).eval()
    dim = transformer.params.dim
    with torch.no_grad():
        programs["text_decoder"] = export(
            txt_dec,
            (
                torch.zeros(1, 1, dim),                      # inputs_embeds: (1, 1, dim)
                torch.zeros(1, dtype=torch.long),            # input_pos:     (1,)
                torch.zeros(1, 1, dtype=torch.long),         # pli_token_ids: (1, 1)
            ),
            strict=True,
        )
    print(f"  text_decoder: (1,1,{dim}) + (1,) + (1,1) -> (1,{transformer.vocab_size})  [PLI enabled]")

    # Carry metadata from the builder
    text_metadata = builder.metadata or {}
    return programs, text_metadata


# ---------------------------------------------------------------------------
# Main export
# ---------------------------------------------------------------------------


def export_gemma4_multimodal(
    hf_model_dir: str,
    et_checkpoint: str,
    output_path: str,
    backend: str = "xnnpack",
    max_seq_len: int = 512,
    audio_frames: int = 200,
) -> None:
    hf_model_dir = Path(hf_model_dir)
    output_path = Path(output_path)
    et_params = str(
        Path(__file__).parent / "config" / "e2b_config.json"
    )

    print(f"Exporting Gemma4 multimodal to {output_path}")
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

    n_patches = 2520   # HF image processor fixed output size
    patch_dim = 768    # 3 * 16^2
    n_mels = 128       # mel spectrogram bins
    text_hidden = cfg.text_config.hidden_size   # 1536
    vocab_size = cfg.text_config.vocab_size     # 262144

    # Build realistic position_ids for a 60×42 patch grid (60*42=2520).
    # All-zero position_ids collapse to 1 pooled token; realistic grid → 280 tokens.
    # pooling_kernel_size=3: pool_positions = 60//3 × 42//3 = 20×14 = 280 tokens.
    _grid_w, _grid_h = 60, 42  # 60*42 = 2520
    _pos_ids = [[x, y] for y in range(_grid_h) for x in range(_grid_w)]
    _vis_pos_ids = torch.tensor([_pos_ids], dtype=torch.long)  # (1, 2520, 2)
    n_vis_soft_tokens = (_grid_w // 3) * (_grid_h // 3)  # 20*14 = 280

    programs: Dict[str, torch.export.ExportedProgram] = {}

    # ---- 1. Vision encoder ----
    print("\nExporting vision_encoder...")
    vis_enc = VisionEncoderExport(
        hf_model.model.vision_tower, hf_model.model.embed_vision
    ).eval()
    with torch.no_grad():
        programs["vision_encoder"] = export(
            vis_enc,
            (
                torch.zeros(1, n_patches, patch_dim),
                _vis_pos_ids,
            ),
            strict=True,
        )
    print(f"  vision_encoder: (1,{n_patches},{patch_dim}) -> ({n_vis_soft_tokens},{text_hidden})")

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

    # ---- 3. Audio encoder (mel → soft tokens) ----
    # Export audio_encoder with static shape matching audio_frames (default 200).
    # The audio tower has stride-48 convolutions requiring T = 48*k - 40 (200 for k=5).
    # The C++ runner truncates/pads mel features to audio_frames before calling.
    print("\nExporting audio_encoder...")
    aud_enc = AudioEncoderExport(
        hf_model.model.audio_tower, hf_model.model.embed_audio
    ).eval()
    with torch.no_grad():
        programs["audio_encoder"] = export(
            aud_enc,
            (torch.zeros(1, audio_frames, n_mels),),
            strict=True,
        )
    print(f"  audio_encoder: (1,{audio_frames},{n_mels}) -> (1,{audio_frames//4},{text_hidden})")

    # Free HF model before loading LLM pipeline (saves ~20 GB RAM)
    del hf_model
    torch.cuda.empty_cache()

    # ---- 4+5. Token embedding + Text decoder (KV cache) ----
    print("\nExporting text backbone (token_embedding + text_decoder with KV cache)...")
    text_programs, text_metadata = export_text_programs(
        et_checkpoint=et_checkpoint,
        et_params=et_params,
        max_seq_len=max_seq_len,
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
    args = parser.parse_args()

    export_gemma4_multimodal(
        hf_model_dir=args.hf_model,
        et_checkpoint=args.et_checkpoint,
        output_path=args.output,
        backend=args.backend,
        max_seq_len=args.max_seq_len,
        audio_frames=args.audio_frames,
    )


if __name__ == "__main__":
    main()
