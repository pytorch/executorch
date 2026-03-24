import json
import struct
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import torch
import torch.nn as nn
from transformers import modeling_rope_utils as hf_rope_utils
from transformers.utils import generic as hf_generic

if not hasattr(hf_generic, "check_model_inputs"):
    def _identity_check_model_inputs(*args, **kwargs):
        def decorator(fn):
            return fn

        return decorator

    hf_generic.check_model_inputs = _identity_check_model_inputs

if "default" not in hf_rope_utils.ROPE_INIT_FUNCTIONS:
    def _compute_default_rope_parameters(config, device=None, seq_len=None, layer_type=None):
        if hasattr(config, "standardize_rope_params"):
            config.standardize_rope_params()
        rope_parameters = getattr(config, "rope_parameters", None)
        if rope_parameters is None:
            base = getattr(config, "rope_theta", 10000.0)
            partial_rotary_factor = getattr(config, "partial_rotary_factor", 1.0)
        else:
            rope_parameters = (
                rope_parameters[layer_type] if layer_type is not None else rope_parameters
            )
            base = rope_parameters.get("rope_theta", getattr(config, "rope_theta", 10000.0))
            partial_rotary_factor = rope_parameters.get(
                "partial_rotary_factor",
                getattr(config, "partial_rotary_factor", 1.0),
            )
        head_dim = getattr(config, "head_dim", None) or (
            config.hidden_size // config.num_attention_heads
        )
        dim = int(head_dim * partial_rotary_factor)
        inv_freq = 1.0 / (
            base
            ** (
                torch.arange(0, dim, 2, dtype=torch.int64).to(
                    device=device, dtype=torch.float
                )
                / dim
            )
        )
        return inv_freq, 1.0

    hf_rope_utils.ROPE_INIT_FUNCTIONS["default"] = _compute_default_rope_parameters

from qwen_tts.core.tokenizer_12hz.configuration_qwen3_tts_tokenizer_v2 import (
    Qwen3TTSTokenizerV2DecoderConfig,
)
from qwen_tts.core.tokenizer_12hz.modeling_qwen3_tts_tokenizer_v2 import (
    Qwen3TTSTokenizerV2Decoder,
)


@dataclass
class DecoderExportMetadata:
    model_id_or_path: str
    tokenizer_type: str
    tts_model_type: str
    decoder_checkpoint: str
    output_sample_rate: int
    decode_upsample_rate: int
    num_quantizers: int
    codebook_size: int
    decoder_config: Dict

    def to_constant_methods(self) -> Dict[str, int]:
        return {
            "output_sample_rate": int(self.output_sample_rate),
            "decode_upsample_rate": int(self.decode_upsample_rate),
            "num_quantizers": int(self.num_quantizers),
            "codebook_size": int(self.codebook_size),
        }

    @classmethod
    def from_json(cls, path: Path) -> "DecoderExportMetadata":
        with path.open("r", encoding="utf-8") as f:
            raw = json.load(f)
        return cls(
            model_id_or_path=raw["model_id_or_path"],
            tokenizer_type=raw["tokenizer_type"],
            tts_model_type=raw["tts_model_type"],
            decoder_checkpoint=raw["decoder_checkpoint"],
            output_sample_rate=int(raw["output_sample_rate"]),
            decode_upsample_rate=int(raw["decode_upsample_rate"]),
            num_quantizers=int(raw["num_quantizers"]),
            codebook_size=int(raw["codebook_size"]),
            decoder_config=raw["decoder_config"],
        )

    def to_json(self, path: Path) -> None:
        payload = {
            "model_id_or_path": self.model_id_or_path,
            "tokenizer_type": self.tokenizer_type,
            "tts_model_type": self.tts_model_type,
            "decoder_checkpoint": self.decoder_checkpoint,
            "output_sample_rate": self.output_sample_rate,
            "decode_upsample_rate": self.decode_upsample_rate,
            "num_quantizers": self.num_quantizers,
            "codebook_size": self.codebook_size,
            "decoder_config": self.decoder_config,
        }
        with path.open("w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2, sort_keys=True)


class Qwen3TTSSpeechDecoderExport(nn.Module):
    """
    Export wrapper for speech tokenizer decode path (audio code -> waveform).
    """

    def __init__(self, decoder: Qwen3TTSTokenizerV2Decoder, decode_upsample_rate: int):
        super().__init__()
        self.decoder = decoder
        self.decode_upsample_rate = int(decode_upsample_rate)

    def forward(self, audio_codes: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        if audio_codes.dim() != 3:
            raise ValueError(
                f"audio_codes must be rank-3 [B, T, Q], got {tuple(audio_codes.shape)}"
            )
        audio_lengths = (audio_codes[..., 0] > -1).sum(1) * self.decode_upsample_rate
        # Decoder expects non-negative code ids.
        clamped_codes = torch.clamp(audio_codes, min=0)
        wav = self.decoder(clamped_codes.transpose(1, 2)).squeeze(1)
        return wav, audio_lengths


def load_decoder_from_metadata(
    metadata: DecoderExportMetadata, checkpoint_path: Path, dtype: torch.dtype
) -> Qwen3TTSTokenizerV2Decoder:
    decoder_cfg = Qwen3TTSTokenizerV2DecoderConfig(**metadata.decoder_config)
    decoder = Qwen3TTSTokenizerV2Decoder(decoder_cfg)
    state = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    decoder.load_state_dict(state, strict=True)
    decoder.eval()
    decoder.to(dtype=dtype)
    return decoder


def make_decode_export_module(
    metadata: DecoderExportMetadata, checkpoint_path: Path, dtype: torch.dtype
) -> Qwen3TTSSpeechDecoderExport:
    decoder = load_decoder_from_metadata(metadata, checkpoint_path, dtype=dtype)
    module = Qwen3TTSSpeechDecoderExport(
        decoder=decoder, decode_upsample_rate=metadata.decode_upsample_rate
    )
    module.eval()
    module.to(dtype=dtype)
    return module


def make_sample_codes(
    codebook_size: int,
    num_quantizers: int,
    code_len: int,
    device: str = "cpu",
) -> torch.Tensor:
    return torch.randint(
        low=0,
        high=codebook_size,
        size=(1, code_len, num_quantizers),
        dtype=torch.long,
        device=device,
    )


def write_codes_binary(path: Path, codes: torch.Tensor) -> None:
    """
    Write codec ids as a simple binary format:
      - int32 codes_len
      - int32 num_quantizers
      - int32[codes_len * num_quantizers] flattened row-major
    """
    if codes.dim() != 2:
        raise ValueError(
            f"codes tensor must be rank-2 [T, Q], got shape={tuple(codes.shape)}"
        )
    codes_i32 = codes.to(dtype=torch.int32).contiguous().cpu()
    t_len, num_q = int(codes_i32.shape[0]), int(codes_i32.shape[1])
    flat_values: List[int] = [int(v) for v in codes_i32.view(-1).tolist()]
    with path.open("wb") as f:
        f.write(struct.pack("<ii", t_len, num_q))
        f.write(struct.pack(f"<{len(flat_values)}i", *flat_values))


def read_codes_binary(path: Path) -> torch.Tensor:
    with path.open("rb") as f:
        header = f.read(8)
        if len(header) != 8:
            raise ValueError(f"Invalid codes file header: {path}")
        t_len, num_q = struct.unpack("<ii", header)
        payload = f.read()
    expected = t_len * num_q
    values = struct.unpack(f"<{expected}i", payload)
    tensor = torch.tensor(values, dtype=torch.long).reshape(t_len, num_q)
    return tensor
