"""Export Qwen3-TTS as a single multi-method .pte for mobile deployment.

Produces one model.pte containing all pipeline stages:
  encode_text      — text token_ids → projected embeddings [1, S, 1024]
  talker           — composite embeddings → (logits, hidden) with KV cache
  code_predictor   — sub-code embeddings → hidden with KV cache
  codec_embed      — (token_id, group_idx) → embedding [1, 1, 1024]
  cp_head          — (hidden, head_idx) → logits [1, 2048]
  cp_generate      — fused 15-step code predictor loop
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
DEFAULT_MODEL_CONFIG_PATH = SCRIPT_DIR / "qwen3-tts-12Hz-0.6B-Base" / "config.json"
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from model import DecoderExportMetadata, load_decoder_from_metadata
from text_prompt_contract import (
    MIN_PROMPT_TOKEN_COUNT,
    TEXT_ONLY_PREFILL_TOKEN_COUNT,
    TEXT_ONLY_PREFILL_TOKEN_COUNT_WITH_LANGUAGE,
    TRAILING_TEMPLATE_TOKEN_COUNT,
)


def load_runtime_token_ids(model_config_path: Path) -> Dict[str, int]:
    with model_config_path.open("r", encoding="utf-8") as f:
        config = json.load(f)

    talker_config = config["talker_config"]
    return {
        "tts_pad_token_id": int(config["tts_pad_token_id"]),
        "tts_bos_token_id": int(config["tts_bos_token_id"]),
        "tts_eod_token_id": int(config["tts_eos_token_id"]),
        "codec_pad_id": int(talker_config["codec_pad_id"]),
        "codec_bos_id": int(talker_config["codec_bos_id"]),
        "codec_eos_id": int(talker_config["codec_eos_token_id"]),
        "codec_think_id": int(talker_config["codec_think_id"]),
        "codec_language_english_id": int(talker_config["codec_language_id"]["english"]),
        "codec_nothink_id": int(talker_config["codec_nothink_id"]),
        "codec_think_bos_id": int(talker_config["codec_think_bos_id"]),
        "codec_think_eos_id": int(talker_config["codec_think_eos_id"]),
        "im_start_token_id": int(config["im_start_token_id"]),
        "assistant_token_id": int(config["assistant_token_id"]),
        "newline_token_id": 198,
    }


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
        stacked[0, : main_codec_weight.shape[0]] = main_codec_weight
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


class CpGenerateExport(nn.Module):
    """Fused code predictor: 15 autoregressive steps in one graph.

    Unrolls the code predictor loop at export time. Each iteration:
    1. Apply per-group LM head to get logits
    2. Argmax to get greedy code (drives the autoregressive chain)
    3. Embed the code via per-group embedding table
    4. Run code predictor transformer step

    Returns all 15 logits (for optional C++ re-sampling) and the sum
    of all 16 group embeddings (for constructing the next talker input).

    The code predictor uses KV cache. Positions 0-16 are used per call.
    The causal mask prevents attending to stale future positions, so
    no explicit cache reset is needed between talker steps.
    """

    def __init__(
        self,
        cp_transformer: nn.Module,
        cp_head_weights: list,
        cp_embed_weights: list,
    ):
        super().__init__()
        self.cp_transformer = cp_transformer
        self.num_groups = len(cp_head_weights)

        for i, hw in enumerate(cp_head_weights):
            self.register_buffer(f"head_{i}", hw)
        for i, ew in enumerate(cp_embed_weights):
            self.register_buffer(f"embed_{i}", ew)

    def _cp_forward(self, embeds: torch.Tensor, pos: torch.Tensor) -> torch.Tensor:
        hidden = self.cp_transformer(
            tokens=None, attn_options={"input_pos": pos}, h=embeds
        )
        if isinstance(hidden, tuple):
            hidden = hidden[0]
        return hidden

    def forward(
        self,
        talker_hidden: torch.Tensor,
        code_0_embed: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        # Prefill: [talker_hidden, code_0_embed] at positions [0, 1]
        cp_input = torch.cat([talker_hidden, code_0_embed], dim=1)
        cp_pos = torch.arange(2, dtype=torch.long)
        cp_hidden = self._cp_forward(cp_input, cp_pos)

        # Start with code_0 embedding in the sum
        embed_sum = code_0_embed.reshape(-1)  # [1024]

        # Collect all 15 sub-code logits
        logits_list = []

        # Unrolled 15 iterations (traced by torch.export)
        head_0 = self.head_0
        logits_0 = F.linear(cp_hidden, head_0)
        logits_list.append(logits_0)
        code_0g = torch.argmax(logits_0, dim=-1)
        embed_0g = F.embedding(code_0g, self.embed_0)
        embed_sum = embed_sum + embed_0g.reshape(-1)
        cp_hidden = self._cp_forward(embed_0g.unsqueeze(0), torch.tensor([2], dtype=torch.long))

        head_1 = self.head_1
        logits_1 = F.linear(cp_hidden, head_1)
        logits_list.append(logits_1)
        code_1g = torch.argmax(logits_1, dim=-1)
        embed_1g = F.embedding(code_1g, self.embed_1)
        embed_sum = embed_sum + embed_1g.reshape(-1)
        cp_hidden = self._cp_forward(embed_1g.unsqueeze(0), torch.tensor([3], dtype=torch.long))

        head_2 = self.head_2
        logits_2 = F.linear(cp_hidden, head_2)
        logits_list.append(logits_2)
        code_2g = torch.argmax(logits_2, dim=-1)
        embed_2g = F.embedding(code_2g, self.embed_2)
        embed_sum = embed_sum + embed_2g.reshape(-1)
        cp_hidden = self._cp_forward(embed_2g.unsqueeze(0), torch.tensor([4], dtype=torch.long))

        head_3 = self.head_3
        logits_3 = F.linear(cp_hidden, head_3)
        logits_list.append(logits_3)
        code_3g = torch.argmax(logits_3, dim=-1)
        embed_3g = F.embedding(code_3g, self.embed_3)
        embed_sum = embed_sum + embed_3g.reshape(-1)
        cp_hidden = self._cp_forward(embed_3g.unsqueeze(0), torch.tensor([5], dtype=torch.long))

        head_4 = self.head_4
        logits_4 = F.linear(cp_hidden, head_4)
        logits_list.append(logits_4)
        code_4g = torch.argmax(logits_4, dim=-1)
        embed_4g = F.embedding(code_4g, self.embed_4)
        embed_sum = embed_sum + embed_4g.reshape(-1)
        cp_hidden = self._cp_forward(embed_4g.unsqueeze(0), torch.tensor([6], dtype=torch.long))

        head_5 = self.head_5
        logits_5 = F.linear(cp_hidden, head_5)
        logits_list.append(logits_5)
        code_5g = torch.argmax(logits_5, dim=-1)
        embed_5g = F.embedding(code_5g, self.embed_5)
        embed_sum = embed_sum + embed_5g.reshape(-1)
        cp_hidden = self._cp_forward(embed_5g.unsqueeze(0), torch.tensor([7], dtype=torch.long))

        head_6 = self.head_6
        logits_6 = F.linear(cp_hidden, head_6)
        logits_list.append(logits_6)
        code_6g = torch.argmax(logits_6, dim=-1)
        embed_6g = F.embedding(code_6g, self.embed_6)
        embed_sum = embed_sum + embed_6g.reshape(-1)
        cp_hidden = self._cp_forward(embed_6g.unsqueeze(0), torch.tensor([8], dtype=torch.long))

        head_7 = self.head_7
        logits_7 = F.linear(cp_hidden, head_7)
        logits_list.append(logits_7)
        code_7g = torch.argmax(logits_7, dim=-1)
        embed_7g = F.embedding(code_7g, self.embed_7)
        embed_sum = embed_sum + embed_7g.reshape(-1)
        cp_hidden = self._cp_forward(embed_7g.unsqueeze(0), torch.tensor([9], dtype=torch.long))

        head_8 = self.head_8
        logits_8 = F.linear(cp_hidden, head_8)
        logits_list.append(logits_8)
        code_8g = torch.argmax(logits_8, dim=-1)
        embed_8g = F.embedding(code_8g, self.embed_8)
        embed_sum = embed_sum + embed_8g.reshape(-1)
        cp_hidden = self._cp_forward(embed_8g.unsqueeze(0), torch.tensor([10], dtype=torch.long))

        head_9 = self.head_9
        logits_9 = F.linear(cp_hidden, head_9)
        logits_list.append(logits_9)
        code_9g = torch.argmax(logits_9, dim=-1)
        embed_9g = F.embedding(code_9g, self.embed_9)
        embed_sum = embed_sum + embed_9g.reshape(-1)
        cp_hidden = self._cp_forward(embed_9g.unsqueeze(0), torch.tensor([11], dtype=torch.long))

        head_10 = self.head_10
        logits_10 = F.linear(cp_hidden, head_10)
        logits_list.append(logits_10)
        code_10g = torch.argmax(logits_10, dim=-1)
        embed_10g = F.embedding(code_10g, self.embed_10)
        embed_sum = embed_sum + embed_10g.reshape(-1)
        cp_hidden = self._cp_forward(embed_10g.unsqueeze(0), torch.tensor([12], dtype=torch.long))

        head_11 = self.head_11
        logits_11 = F.linear(cp_hidden, head_11)
        logits_list.append(logits_11)
        code_11g = torch.argmax(logits_11, dim=-1)
        embed_11g = F.embedding(code_11g, self.embed_11)
        embed_sum = embed_sum + embed_11g.reshape(-1)
        cp_hidden = self._cp_forward(embed_11g.unsqueeze(0), torch.tensor([13], dtype=torch.long))

        head_12 = self.head_12
        logits_12 = F.linear(cp_hidden, head_12)
        logits_list.append(logits_12)
        code_12g = torch.argmax(logits_12, dim=-1)
        embed_12g = F.embedding(code_12g, self.embed_12)
        embed_sum = embed_sum + embed_12g.reshape(-1)
        cp_hidden = self._cp_forward(embed_12g.unsqueeze(0), torch.tensor([14], dtype=torch.long))

        head_13 = self.head_13
        logits_13 = F.linear(cp_hidden, head_13)
        logits_list.append(logits_13)
        code_13g = torch.argmax(logits_13, dim=-1)
        embed_13g = F.embedding(code_13g, self.embed_13)
        embed_sum = embed_sum + embed_13g.reshape(-1)
        cp_hidden = self._cp_forward(embed_13g.unsqueeze(0), torch.tensor([15], dtype=torch.long))

        # Last group: no need for CP forward after
        head_14 = self.head_14
        logits_14 = F.linear(cp_hidden, head_14)
        logits_list.append(logits_14)
        code_14g = torch.argmax(logits_14, dim=-1)
        embed_14g = F.embedding(code_14g, self.embed_14)
        embed_sum = embed_sum + embed_14g.reshape(-1)

        all_logits = torch.cat(logits_list, dim=0)  # [15, 2048]
        return all_logits, embed_sum


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

    # 3. code_predictor (standalone, kept for backward compat)
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

    # 5. cp_head (standalone, kept for backward compat)
    cp_head_weights = []
    for i in range(15):
        key = f"code_predictor.lm_head.{i}.weight"
        cp_head_weights.append(aux[key].to(dtype))
    cp_head = CpHeadExport(cp_head_weights)
    cp_head.eval()

    # 6. cp_generate (FUSED: 15-step code predictor in one graph)
    cp_model_fused, _ = load_code_predictor_model(talker_dir, max_seq_len=32)
    cp_generate = CpGenerateExport(
        cp_transformer=cp_model_fused,
        cp_head_weights=[w.to(dtype) for w in cp_head_weights],
        cp_embed_weights=[w.to(dtype) for w in cp_codec_weights],
    )
    cp_generate.eval()

    # 7. decode_audio
    checkpoint_path = converted_dir / metadata.decoder_checkpoint
    decoder = load_decoder_from_metadata(metadata, checkpoint_path, dtype=dtype)
    decode_audio = DynamicDecoderExport(decoder, metadata.decode_upsample_rate)
    decode_audio.eval()
    decode_audio.to(dtype=dtype)

    for mod in [encode_text, talker, code_predictor, codec_embed, cp_head,
                 cp_generate, decode_audio]:
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
        "cp_generate": cp_generate,
        "decode_audio": decode_audio,
    }, talker_args, cp_args


def export_all(
    modules: dict,
    talker_args,
    cp_args,
    metadata: DecoderExportMetadata,
    runtime_token_ids: Dict[str, int],
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

    # 6. cp_generate — fused 15-step code predictor (static shapes)
    print("Exporting cp_generate (fused 15-step loop)...")
    sample_talker_hidden = torch.randn(1, 1, cp_args.dim)
    sample_code0_embed = torch.randn(1, 1, cp_args.dim)
    programs["cp_generate"] = export(
        modules["cp_generate"],
        (sample_talker_hidden, sample_code0_embed),
        strict=False,
    )

    # 7. decode_audio — dynamic codes length
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
    elif backend == "metal":
        from executorch.backends.apple.metal.metal_backend import MetalBackend
        from executorch.backends.apple.metal.metal_partitioner import MetalPartitioner

        # Linear bias decomposition (following Voxtral pattern).
        def _linear_bias_decomposition(input_tensor, weight, bias=None):
            out = torch.matmul(input_tensor, weight.t())
            if bias is not None:
                out = out + bias
            return out

        updated_programs = {}
        for key, ep in programs.items():
            if key in ("codec_embed",):
                updated_programs[key] = ep
            else:
                updated_programs[key] = ep.run_decompositions(
                    {torch.ops.aten.linear.default: _linear_bias_decomposition}
                )
        programs = updated_programs

        partitioner = {}
        for key in programs:
            if key in ("codec_embed",):
                partitioner[key] = []
            elif key == "decode_audio":
                # decode_audio uses cumsum which lacks Metal fallback.
                # Use XNNPACK for GPU-incompatible methods.
                from executorch.backends.xnnpack.partition.xnnpack_partitioner import (
                    XnnpackDynamicallyQuantizedPartitioner,
                    XnnpackPartitioner,
                )
                partitioner[key] = [
                    XnnpackDynamicallyQuantizedPartitioner(),
                    XnnpackPartitioner(),
                ]
            else:
                compile_specs = [MetalBackend.generate_method_name_compile_spec(key)]
                partitioner[key] = [MetalPartitioner(compile_specs)]
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
        "text_prompt_min_token_count": MIN_PROMPT_TOKEN_COUNT,
        "text_prompt_prefill_token_count": TEXT_ONLY_PREFILL_TOKEN_COUNT,
        "text_prompt_prefill_token_count_with_language": TEXT_ONLY_PREFILL_TOKEN_COUNT_WITH_LANGUAGE,
        "text_prompt_trailing_template_token_count": TRAILING_TEMPLATE_TOKEN_COUNT,
    })
    constant_methods.update(runtime_token_ids)

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
    parser.add_argument("--backend", choices=["portable", "xnnpack", "metal"], default="xnnpack")
    parser.add_argument("--max-seq-len", type=int, default=256)
    parser.add_argument("--dtype", choices=["fp32", "bf16"], default="fp32")
    parser.add_argument("--qlinear", choices=["4w", "8w", "8da4w", "8da8w"], default=None)
    parser.add_argument("--qlinear-group-size", type=int, default=32)
    parser.add_argument(
        "--qembedding", choices=["4w", "8w"], default=None,
        help="Embedding quantization. Reduces text_embedding from ~1.2GB to ~300-600MB.",
    )
    parser.add_argument(
        "--model-config-path",
        type=Path,
        default=DEFAULT_MODEL_CONFIG_PATH,
        help="Path to the checked-in Qwen3-TTS config.json for runtime token IDs.",
    )
    parser.add_argument("--output-name", type=str, default="model.pte")
    return parser.parse_args()


def main():
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    converted_dir = args.converted_dir.resolve()
    talker_dir = args.talker_dir.resolve()
    model_config_path = args.model_config_path.resolve()
    metadata = DecoderExportMetadata.from_json(converted_dir / "decoder_metadata.json")
    runtime_token_ids = load_runtime_token_ids(model_config_path)
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
        runtime_token_ids=runtime_token_ids,
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
        "prompt_contract": "assistant_chat_text_v1",
        "requires_tokenizer": True,
        "supports_text_only_synthesis": True,
        "supports_voice_clone_synthesis": False,
        "text_prompt_min_token_count": MIN_PROMPT_TOKEN_COUNT,
        "text_prompt_prefill_token_count": TEXT_ONLY_PREFILL_TOKEN_COUNT,
        "text_prompt_prefill_token_count_with_language": TEXT_ONLY_PREFILL_TOKEN_COUNT_WITH_LANGUAGE,
        "text_prompt_trailing_template_token_count": TRAILING_TEMPLATE_TOKEN_COUNT,
        **runtime_token_ids,
    }
    manifest_path = args.output_dir / "export_manifest.json"
    with manifest_path.open("w") as f:
        json.dump(manifest, f, indent=2, sort_keys=True)
    print(f"Saved: {manifest_path}")


if __name__ == "__main__":
    main()
