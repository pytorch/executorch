"""Export Qwen3-TTS as a single multi-method .pte for mobile deployment.

Produces one model.pte containing all pipeline stages:
  encode_text      — text token_ids → projected embeddings [1, S, 1024]
  talker           — composite embeddings → (logits, hidden) with KV cache
  code_predictor   — sub-code embeddings → hidden with KV cache
  codec_embed      — (token_id, group_idx) → embedding [1, 1, 1024]
  cp_head          — (hidden, head_idx) → logits [1, 2048]
  decode_audio     — audio codes [1, T, 16] → (waveform, lengths)

Follows the Parakeet multi-method export pattern.

Usage:
    python export_unified.py \
        --converted-dir qwen3_tts_artifacts \
        --talker-dir qwen3_tts_artifacts/talker_converted \
        --output-dir qwen3_tts_exports_unified \
        --backend xnnpack \
        --qlinear 8da4w
"""

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Dict, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.export import Dim, export

from executorch.exir import (
    EdgeCompileConfig,
    ExecutorchBackendConfig,
    to_edge_transform_and_lower,
)
from executorch.exir.passes import MemoryPlanningPass

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from model import DecoderExportMetadata, load_decoder_from_metadata


# ---------------------------------------------------------------------------
# Wrapper modules
# ---------------------------------------------------------------------------

class EncodeTextExport(nn.Module):
    """Text token_ids → projected embeddings [1, S, 1024].

    Wraps text_embedding (nn.Embedding) + text_projection (2-layer MLP).
    """

    def __init__(self, text_embedding: nn.Embedding, text_projection: nn.Module):
        super().__init__()
        self.text_embedding = text_embedding
        self.text_projection = text_projection

    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        embeds = self.text_embedding(token_ids)
        return self.text_projection(embeds)


class TextProjectionMLP(nn.Module):
    """2-layer MLP: text_hidden (2048) → intermediate (2048) → talker_dim (1024)."""

    def __init__(self, fc1_weight, fc1_bias, fc2_weight, fc2_bias):
        super().__init__()
        self.fc1 = nn.Linear(fc1_weight.shape[1], fc1_weight.shape[0])
        self.fc1.weight = nn.Parameter(fc1_weight)
        self.fc1.bias = nn.Parameter(fc1_bias)
        self.fc2 = nn.Linear(fc2_weight.shape[1], fc2_weight.shape[0])
        self.fc2.weight = nn.Parameter(fc2_weight)
        self.fc2.bias = nn.Parameter(fc2_bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc2(F.silu(self.fc1(x)))


class TalkerExport(nn.Module):
    """Talker transformer wrapper returning both logits and hidden state.

    The transformer runs with apply_output=False (returns normalized hidden).
    We apply codec_head manually to produce logits.
    """

    def __init__(self, transformer: nn.Module, codec_head_weight: torch.Tensor):
        super().__init__()
        self.transformer = transformer
        self.codec_head = nn.Linear(
            codec_head_weight.shape[1], codec_head_weight.shape[0], bias=False
        )
        self.codec_head.weight = nn.Parameter(codec_head_weight)

    def forward(
        self, embeds: torch.Tensor, input_pos: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        hidden = self.transformer(
            tokens=None, attn_options={"input_pos": input_pos}, h=embeds
        )
        if isinstance(hidden, tuple):
            hidden = hidden[0]
        logits = self.codec_head(hidden)
        return logits, hidden


class CodePredictorExport(nn.Module):
    """Code predictor transformer wrapper (returns hidden state only)."""

    def __init__(self, transformer: nn.Module):
        super().__init__()
        self.transformer = transformer

    def forward(
        self, embeds: torch.Tensor, input_pos: torch.Tensor
    ) -> torch.Tensor:
        hidden = self.transformer(
            tokens=None, attn_options={"input_pos": input_pos}, h=embeds
        )
        if isinstance(hidden, tuple):
            hidden = hidden[0]
        return hidden


class CodecEmbedExport(nn.Module):
    """All codec embeddings (main + 15 cp) stacked for index_select lookup.

    Main codec: vocab 3072, dim 1024 (group_idx=0)
    CP codec 0-14: vocab 2048, dim 1024 (group_idx=1..15)

    We pad CP embeddings to 3072 rows so all can be stacked into [16, 3072, 1024].
    """

    def __init__(
        self,
        main_codec_weight: torch.Tensor,
        cp_codec_weights: list,
    ):
        super().__init__()
        vocab_max = main_codec_weight.shape[0]
        dim = main_codec_weight.shape[1]
        num_groups = 1 + len(cp_codec_weights)

        stacked = torch.zeros(num_groups, vocab_max, dim, dtype=main_codec_weight.dtype)
        stacked[0] = main_codec_weight
        for i, w in enumerate(cp_codec_weights):
            stacked[i + 1, : w.shape[0]] = w

        self.register_buffer("stacked_embeds", stacked)

    def forward(
        self, token_id: torch.Tensor, group_idx: torch.Tensor
    ) -> torch.Tensor:
        table = torch.index_select(self.stacked_embeds, 0, group_idx).squeeze(0)
        return F.embedding(token_id, table).unsqueeze(0)


class CpHeadExport(nn.Module):
    """Code predictor per-group LM heads stacked for index_select.

    15 heads, each [2048, 1024]. Stacked to [15, 2048, 1024].
    """

    def __init__(self, head_weights: list):
        super().__init__()
        stacked = torch.stack(head_weights, dim=0)
        self.register_buffer("stacked_heads", stacked)

    def forward(
        self, hidden: torch.Tensor, head_idx: torch.Tensor
    ) -> torch.Tensor:
        head_weight = torch.index_select(
            self.stacked_heads, 0, head_idx
        ).squeeze(0)
        return F.linear(hidden, head_weight)


class DynamicDecoderExport(nn.Module):
    """Decoder wrapper with exportable padding (no math.ceil on SymInt)."""

    def __init__(self, decoder, decode_upsample_rate: int):
        super().__init__()
        self.decoder = decoder
        self.decode_upsample_rate = int(decode_upsample_rate)
        self._patch_causal_conv_padding()

    def _patch_causal_conv_padding(self):
        """Replace math.ceil-based padding with integer arithmetic."""
        for module in self.decoder.modules():
            cls_name = type(module).__name__
            if "CausalConvNet" in cls_name and hasattr(module, "stride"):
                _patch_conv_padding(module)

    def forward(self, audio_codes: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        audio_lengths = (audio_codes[..., 0] > -1).sum(1) * self.decode_upsample_rate
        clamped_codes = torch.clamp(audio_codes, min=0)
        wav = self.decoder(clamped_codes.transpose(1, 2)).squeeze(1)
        return wav, audio_lengths


def _patch_conv_padding(module):
    """Monkey-patch _get_extra_padding_for_conv1d to avoid math.ceil on SymInt."""
    kernel_size = module.kernel_size
    stride = module.stride
    padding = module.padding

    def _exportable_extra_padding(self, hidden_state):
        length = hidden_state.shape[-1]
        n_frames_num = length - kernel_size + padding + stride
        n_frames_ceil = (n_frames_num + stride - 1) // stride
        ideal_length = (n_frames_ceil - 1) * stride + (kernel_size - padding)
        return ideal_length - length

    import types
    module._get_extra_padding_for_conv1d = types.MethodType(
        _exportable_extra_padding, module
    )


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------

def load_talker_model(talker_dir: Path, max_seq_len: int):
    """Load the talker backbone using Llama infrastructure."""
    from executorch.examples.models.llama.model_args import ModelArgs
    from executorch.examples.models.llama.llama_transformer import construct_transformer

    config_path = talker_dir / "talker_config.json"
    with config_path.open("r") as f:
        params = json.load(f)

    params["use_kv_cache"] = True
    params["max_seq_len"] = max_seq_len
    params["max_context_len"] = max_seq_len
    params["max_batch_size"] = 1
    params["generate_full_logits"] = False
    params["apply_embedding"] = False
    params["apply_output"] = False

    model_args = ModelArgs(**params)
    model = construct_transformer(model_args)
    model.eval()

    ckpt = torch.load(talker_dir / "talker_main.pth", map_location="cpu", weights_only=True)
    missing, unexpected = model.load_state_dict(ckpt, strict=False)
    real_missing = [k for k in missing if "k_cache" not in k and "v_cache" not in k and "mask" not in k]
    if real_missing:
        print(f"WARNING: Talker missing keys: {real_missing}")

    return model, model_args


def load_code_predictor_model(talker_dir: Path, max_seq_len: int = 32):
    """Load the code predictor backbone."""
    from executorch.examples.models.llama.model_args import ModelArgs
    from executorch.examples.models.llama.llama_transformer import construct_transformer

    config_path = talker_dir / "code_predictor_config.json"
    with config_path.open("r") as f:
        params = json.load(f)

    params["use_kv_cache"] = True
    params["max_seq_len"] = max_seq_len
    params["max_context_len"] = max_seq_len
    params["max_batch_size"] = 1
    params["generate_full_logits"] = False
    params["apply_embedding"] = False
    params["apply_output"] = False

    model_args = ModelArgs(**params)
    model = construct_transformer(model_args)
    model.eval()

    ckpt = torch.load(
        talker_dir / "talker_code_predictor.pth", map_location="cpu", weights_only=True
    )
    missing, unexpected = model.load_state_dict(ckpt, strict=False)
    real_missing = [k for k in missing if "k_cache" not in k and "v_cache" not in k and "mask" not in k]
    if real_missing:
        print(f"WARNING: Code predictor missing keys: {real_missing}")

    return model, model_args


def load_aux_weights(talker_dir: Path):
    """Load auxiliary weights (embeddings, heads, projections)."""
    aux = torch.load(talker_dir / "talker_aux.pth", map_location="cpu", weights_only=True)
    return aux


# ---------------------------------------------------------------------------
# Export
# ---------------------------------------------------------------------------

def build_wrapper_modules(
    talker_dir: Path,
    converted_dir: Path,
    metadata: DecoderExportMetadata,
    max_seq_len: int,
    dtype: torch.dtype,
):
    """Build all wrapper modules for multi-method export."""
    aux = load_aux_weights(talker_dir)

    # 1. encode_text
    text_emb_weight = aux["model.text_embedding.weight"].to(dtype)
    text_embedding = nn.Embedding(
        text_emb_weight.shape[0], text_emb_weight.shape[1]
    )
    text_embedding.weight = nn.Parameter(text_emb_weight)

    text_projection = TextProjectionMLP(
        fc1_weight=aux["text_projection.linear_fc1.weight"].to(dtype),
        fc1_bias=aux["text_projection.linear_fc1.bias"].to(dtype),
        fc2_weight=aux["text_projection.linear_fc2.weight"].to(dtype),
        fc2_bias=aux["text_projection.linear_fc2.bias"].to(dtype),
    )
    encode_text = EncodeTextExport(text_embedding, text_projection)
    encode_text.eval()

    # 2. talker
    talker_model, talker_args = load_talker_model(talker_dir, max_seq_len)
    codec_head_weight = aux["codec_head.weight"].to(dtype)
    talker = TalkerExport(talker_model, codec_head_weight)
    talker.eval()

    # 3. code_predictor
    cp_model, cp_args = load_code_predictor_model(talker_dir, max_seq_len=32)
    code_predictor = CodePredictorExport(cp_model)
    code_predictor.eval()

    # 4. codec_embed
    main_codec_weight = aux["main_codec_embedding.weight"].to(dtype)
    cp_codec_weights = []
    for i in range(15):
        key = f"cp_codec_embedding.{i}.weight"
        cp_codec_weights.append(aux[key].to(dtype))
    codec_embed = CodecEmbedExport(main_codec_weight, cp_codec_weights)
    codec_embed.eval()

    # 5. cp_head
    cp_head_weights = []
    for i in range(15):
        key = f"code_predictor.lm_head.{i}.weight"
        cp_head_weights.append(aux[key].to(dtype))
    cp_head = CpHeadExport(cp_head_weights)
    cp_head.eval()

    # 6. decode_audio
    checkpoint_path = converted_dir / metadata.decoder_checkpoint
    decoder = load_decoder_from_metadata(metadata, checkpoint_path, dtype=dtype)
    decode_audio = DynamicDecoderExport(decoder, metadata.decode_upsample_rate)
    decode_audio.eval()
    decode_audio.to(dtype=dtype)

    for mod in [encode_text, talker, code_predictor, codec_embed, cp_head, decode_audio]:
        for p in mod.parameters():
            p.requires_grad_(False)
        for b in mod.buffers():
            b.requires_grad_(False)

    return {
        "encode_text": encode_text,
        "talker": talker,
        "code_predictor": code_predictor,
        "codec_embed": codec_embed,
        "cp_head": cp_head,
        "decode_audio": decode_audio,
    }, talker_args, cp_args


def export_all(
    modules: dict,
    talker_args,
    cp_args,
    metadata: DecoderExportMetadata,
    max_seq_len: int,
    backend: str,
    qlinear: str = None,
    qlinear_group_size: int = 32,
    qembedding: str = None,
):
    """Export all methods into a single .pte."""

    # Apply quantization before export.
    if qlinear is not None or qembedding is not None:
        from executorch.extension.llm.export.quantize import quantize_model_
        for name, mod in modules.items():
            if name in ("codec_embed", "cp_head"):
                continue
            q_linear = qlinear if name not in ("codec_embed",) else None
            q_embed = qembedding if name in ("encode_text",) else None
            if q_linear or q_embed:
                print(f"  Quantizing {name} (linear={q_linear}, embedding={q_embed})...")
                quantize_model_(
                    mod,
                    qlinear_config=q_linear,
                    qlinear_group_size=qlinear_group_size,
                    qembedding_config=q_embed,
                )

    programs = {}

    # 1. encode_text — dynamic sequence length
    print("Exporting encode_text...")
    seq_dim = Dim("seq_len", min=1, max=4096)
    sample_ids = torch.zeros(1, 10, dtype=torch.long)
    programs["encode_text"] = export(
        modules["encode_text"],
        (sample_ids,),
        dynamic_shapes={"token_ids": {1: seq_dim}},
        strict=False,
    )

    # 2. talker — dynamic sequence length for prefill+decode
    print("Exporting talker...")
    talker_seq = Dim("talker_seq", min=1, max=max_seq_len)
    sample_embeds = torch.randn(1, 4, talker_args.dim)
    sample_pos = torch.arange(4, dtype=torch.long)
    programs["talker"] = export(
        modules["talker"],
        (sample_embeds, sample_pos),
        dynamic_shapes={
            "embeds": {1: talker_seq},
            "input_pos": {0: talker_seq},
        },
        strict=False,
    )

    # 3. code_predictor — dynamic sequence length
    print("Exporting code_predictor...")
    cp_seq = Dim("cp_seq", min=1, max=32)
    sample_cp_embeds = torch.randn(1, 2, cp_args.dim)
    sample_cp_pos = torch.arange(2, dtype=torch.long)
    programs["code_predictor"] = export(
        modules["code_predictor"],
        (sample_cp_embeds, sample_cp_pos),
        dynamic_shapes={
            "embeds": {1: cp_seq},
            "input_pos": {0: cp_seq},
        },
        strict=False,
    )

    # 4. codec_embed — static shapes
    print("Exporting codec_embed...")
    sample_tid = torch.tensor([0], dtype=torch.long)
    sample_gidx = torch.tensor([0], dtype=torch.long)
    programs["codec_embed"] = export(
        modules["codec_embed"],
        (sample_tid, sample_gidx),
        strict=False,
    )

    # 5. cp_head — static shapes
    print("Exporting cp_head...")
    sample_hidden = torch.randn(1, cp_args.dim)
    sample_hidx = torch.tensor([0], dtype=torch.long)
    programs["cp_head"] = export(
        modules["cp_head"],
        (sample_hidden, sample_hidx),
        strict=False,
    )

    # 6. decode_audio — dynamic codes length
    print("Exporting decode_audio...")
    codes_dim = Dim("codes_len", min=1, max=2000)
    sample_codes = torch.randint(0, metadata.codebook_size, (1, 10, metadata.num_quantizers), dtype=torch.long)
    programs["decode_audio"] = export(
        modules["decode_audio"],
        (sample_codes,),
        dynamic_shapes={"audio_codes": {1: codes_dim}},
        strict=False,
    )

    # Build per-method partitioners.
    if backend == "xnnpack":
        from executorch.backends.xnnpack.partition.xnnpack_partitioner import (
            XnnpackDynamicallyQuantizedPartitioner,
            XnnpackPartitioner,
        )
        partitioner = {}
        for key in programs:
            if key in ("codec_embed",):
                partitioner[key] = []
            else:
                partitioner[key] = [
                    XnnpackDynamicallyQuantizedPartitioner(),
                    XnnpackPartitioner(),
                ]
    else:
        partitioner = {key: [] for key in programs}

    # Constant methods (metadata).
    constant_methods = metadata.to_constant_methods()
    constant_methods.update({
        "max_seq_len": max_seq_len,
        "talker_vocab_size": talker_args.vocab_size,
        "talker_dim": talker_args.dim,
        "talker_n_layers": talker_args.n_layers,
        "cp_n_layers": cp_args.n_layers,
        "num_code_groups": 16,
    })

    print("Lowering to ExecuTorch...")
    edge_prog = to_edge_transform_and_lower(
        programs,
        partitioner=partitioner,
        compile_config=EdgeCompileConfig(
            _check_ir_validity=False,
            _skip_dim_order=True,
        ),
        constant_methods=constant_methods,
    )
    et_prog = edge_prog.to_executorch(
        config=ExecutorchBackendConfig(
            extract_delegate_segments=True,
            do_quant_fusion_and_const_prop=True,
            memory_planning_pass=MemoryPlanningPass(alloc_graph_input=False),
        )
    )
    return et_prog


def parse_args():
    parser = argparse.ArgumentParser(
        description="Export Qwen3-TTS as single multi-method .pte"
    )
    parser.add_argument(
        "--converted-dir", type=Path, required=True,
        help="Directory with decoder_metadata.json and decoder checkpoint.",
    )
    parser.add_argument(
        "--talker-dir", type=Path, required=True,
        help="Directory with talker_main.pth, talker_code_predictor.pth, talker_aux.pth.",
    )
    parser.add_argument(
        "--output-dir", type=Path, default=Path("./qwen3_tts_exports_unified"),
    )
    parser.add_argument("--backend", choices=["portable", "xnnpack"], default="xnnpack")
    parser.add_argument("--max-seq-len", type=int, default=256)
    parser.add_argument("--dtype", choices=["fp32", "bf16"], default="fp32")
    parser.add_argument("--qlinear", choices=["4w", "8w", "8da4w", "8da8w"], default=None)
    parser.add_argument("--qlinear-group-size", type=int, default=32)
    parser.add_argument(
        "--qembedding", choices=["4w", "8w"], default=None,
        help="Embedding quantization. Reduces text_embedding from ~1.2GB to ~300-600MB.",
    )
    parser.add_argument("--output-name", type=str, default="model.pte")
    return parser.parse_args()


def main():
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    converted_dir = args.converted_dir.resolve()
    talker_dir = args.talker_dir.resolve()
    metadata = DecoderExportMetadata.from_json(converted_dir / "decoder_metadata.json")
    dtype = {"fp32": torch.float32, "bf16": torch.bfloat16}[args.dtype]

    print("Building wrapper modules...")
    modules, talker_args, cp_args = build_wrapper_modules(
        talker_dir=talker_dir,
        converted_dir=converted_dir,
        metadata=metadata,
        max_seq_len=args.max_seq_len,
        dtype=dtype,
    )

    print(f"\nModule summary:")
    for name, mod in modules.items():
        n_params = sum(p.numel() for p in mod.parameters())
        n_bufs = sum(b.numel() for b in mod.buffers())
        print(f"  {name}: {n_params:,} params, {n_bufs:,} buffer elements")

    et_prog = export_all(
        modules=modules,
        talker_args=talker_args,
        cp_args=cp_args,
        metadata=metadata,
        max_seq_len=args.max_seq_len,
        backend=args.backend,
        qlinear=args.qlinear,
        qlinear_group_size=args.qlinear_group_size,
        qembedding=args.qembedding,
    )

    model_path = args.output_dir / args.output_name
    with model_path.open("wb") as f:
        et_prog.write_to_file(f)
    file_size_mb = model_path.stat().st_size / (1024 * 1024)
    print(f"\nSaved: {model_path} ({file_size_mb:.1f} MB)")

    manifest = {
        "model_type": "qwen3_tts_unified",
        "backend": args.backend,
        "dtype": args.dtype,
        "qlinear": args.qlinear,
        "qembedding": args.qembedding,
        "max_seq_len": args.max_seq_len,
        "methods": list(modules.keys()),
        "num_code_groups": 16,
    }
    manifest_path = args.output_dir / "export_manifest.json"
    with manifest_path.open("w") as f:
        json.dump(manifest, f, indent=2, sort_keys=True)
    print(f"Saved: {manifest_path}")


if __name__ == "__main__":
    main()
