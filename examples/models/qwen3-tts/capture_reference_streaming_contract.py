import argparse
import json
import random
import struct
import sys
from pathlib import Path

import numpy as np
import torch
from transformers import modeling_rope_utils as hf_rope_utils
from transformers.utils import generic as hf_generic

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))


STREAMING_CHUNK_SIZE = 300
STREAMING_LEFT_CONTEXT_SIZE = 25


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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Capture the upstream Qwen3-TTS streaming contract for parity checks."
    )
    parser.add_argument(
        "--upstream-repo",
        type=Path,
        default=Path("/Users/younghan/project/executorch-exp/Qwen3-TTS"),
    )
    parser.add_argument("--model-id-or-path", required=True, type=str)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--text", required=True, type=str)
    parser.add_argument("--language", default="English", type=str)
    parser.add_argument("--instruct", default="", type=str)
    parser.add_argument("--seed", default=42, type=int)
    parser.add_argument("--max-new-tokens", default=2048, type=int)
    parser.add_argument("--top-k", default=50, type=int)
    parser.add_argument("--top-p", default=1.0, type=float)
    parser.add_argument("--temperature", default=0.9, type=float)
    parser.add_argument("--repetition-penalty", default=1.05, type=float)
    parser.add_argument("--streaming-interval", default=2.0, type=float)
    parser.add_argument("--non-streaming-mode", action="store_true")
    return parser.parse_args()


def _default_reference_audio(duration_sec: float = 1.0, sample_rate: int = 24000):
    wav = np.zeros(int(duration_sec * sample_rate), dtype=np.float32)
    return wav, sample_rate


def _set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def _load_model(upstream_repo: Path, model_id_or_path: str):
    for module_name in list(sys.modules):
        if module_name == "qwen_tts" or module_name.startswith("qwen_tts."):
            sys.modules.pop(module_name)
    if str(upstream_repo) not in sys.path:
        sys.path.insert(0, str(upstream_repo))
    from qwen_tts.core.models.configuration_qwen3_tts import (  # noqa: WPS433
        Qwen3TTSConfig,
        Qwen3TTSTalkerCodePredictorConfig,
        Qwen3TTSTalkerConfig,
    )
    from qwen_tts.core.models.modeling_qwen3_tts import (  # noqa: WPS433
        Qwen3TTSForConditionalGeneration,
    )
    from qwen_tts.core.models.processing_qwen3_tts import Qwen3TTSProcessor  # noqa: WPS433

    def _patch_pad_token_id(config_cls, fallback_attr: str) -> None:
        original_init = config_cls.__init__

        def _wrapped_init(self, *args, **kwargs):
            original_init(self, *args, **kwargs)
            if not hasattr(self, "pad_token_id"):
                self.pad_token_id = getattr(self, fallback_attr, 0)

        config_cls.__init__ = _wrapped_init

    _patch_pad_token_id(Qwen3TTSTalkerConfig, "tts_pad_token_id")
    _patch_pad_token_id(Qwen3TTSTalkerCodePredictorConfig, "pad_token_id")
    from qwen_tts import Qwen3TTSModel  # noqa: WPS433
    from transformers import AutoConfig, AutoModel, AutoProcessor  # noqa: WPS433

    AutoConfig.register("qwen3_tts", Qwen3TTSConfig)
    AutoModel.register(Qwen3TTSConfig, Qwen3TTSForConditionalGeneration)
    AutoProcessor.register(Qwen3TTSConfig, Qwen3TTSProcessor)

    model = AutoModel.from_pretrained(
        model_id_or_path,
        device_map="cpu",
        dtype=torch.float32,
    )
    processor = AutoProcessor.from_pretrained(model_id_or_path)
    return Qwen3TTSModel(
        model=model,
        processor=processor,
        generate_defaults=model.generate_config,
    )


def write_codes_binary(path: Path, codes: torch.Tensor) -> None:
    codes_i32 = codes.to(dtype=torch.int32).contiguous().cpu()
    t_len, num_q = int(codes_i32.shape[0]), int(codes_i32.shape[1])
    flat_values = [int(v) for v in codes_i32.view(-1).tolist()]
    with path.open("wb") as f:
        f.write(struct.pack("<ii", t_len, num_q))
        f.write(struct.pack(f"<{len(flat_values)}i", *flat_values))


def _capture_voice_design_codes(model, args: argparse.Namespace):
    texts = [args.text]
    input_ids = model._tokenize_texts([model._build_assistant_text(t) for t in texts])
    instruct_ids = []
    if args.instruct:
        instruct_ids.append(model._tokenize_texts([model._build_instruct_text(args.instruct)])[0])
    else:
        instruct_ids.append(None)
    talker_codes_list, _ = model.model.generate(
        input_ids=input_ids,
        instruct_ids=instruct_ids,
        languages=[args.language],
        non_streaming_mode=args.non_streaming_mode,
        do_sample=True,
        top_k=args.top_k,
        top_p=args.top_p,
        temperature=args.temperature,
        repetition_penalty=args.repetition_penalty,
        subtalker_dosample=True,
        subtalker_top_k=args.top_k,
        subtalker_top_p=args.top_p,
        subtalker_temperature=args.temperature,
        max_new_tokens=args.max_new_tokens,
    )
    return talker_codes_list[0].detach().cpu()


def _capture_base_codes(model, args: argparse.Namespace):
    prompt_items = model.create_voice_clone_prompt(
        ref_audio=_default_reference_audio(),
        ref_text=None,
        x_vector_only_mode=True,
    )
    prompt_dict = model._prompt_items_to_voice_clone_prompt(prompt_items)
    ref_ids = [None]
    input_ids = model._tokenize_texts([model._build_assistant_text(args.text)])
    talker_codes_list, _ = model.model.generate(
        input_ids=input_ids,
        ref_ids=ref_ids,
        voice_clone_prompt=prompt_dict,
        languages=[args.language],
        non_streaming_mode=args.non_streaming_mode,
        do_sample=True,
        top_k=args.top_k,
        top_p=args.top_p,
        temperature=args.temperature,
        repetition_penalty=args.repetition_penalty,
        subtalker_dosample=True,
        subtalker_top_k=args.top_k,
        subtalker_top_p=args.top_p,
        subtalker_temperature=args.temperature,
        max_new_tokens=args.max_new_tokens,
    )
    return talker_codes_list[0].detach().cpu()


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    _set_seed(args.seed)

    model = _load_model(args.upstream_repo, args.model_id_or_path)
    codes = (
        _capture_voice_design_codes(model, args)
        if args.instruct
        else _capture_base_codes(model, args)
    )
    wavs, sample_rate = model.model.speech_tokenizer.decode([{"audio_codes": codes}])
    wav = np.asarray(wavs[0], dtype=np.float32)

    decode_upsample_rate = int(
        getattr(model.model.speech_tokenizer, "decode_upsample_rate", 1920)
    )
    codec_steps_per_second = sample_rate / decode_upsample_rate
    interval_steps = max(1, round(args.streaming_interval * codec_steps_per_second))
    chunk_boundaries = list(range(interval_steps, int(codes.shape[0]) + 1, interval_steps))
    if not chunk_boundaries or chunk_boundaries[-1] != int(codes.shape[0]):
        chunk_boundaries.append(int(codes.shape[0]))

    codes_path = args.output_dir / "reference_codes.bin"
    write_codes_binary(codes_path, codes)
    np.save(args.output_dir / "reference_audio.npy", wav)
    np.save(args.output_dir / "reference_codes.npy", codes.numpy())

    contract = {
        "model_id_or_path": args.model_id_or_path,
        "text": args.text,
        "language": args.language,
        "instruct": args.instruct,
        "seed": args.seed,
        "non_streaming_mode": args.non_streaming_mode,
        "temperature": args.temperature,
        "top_k": args.top_k,
        "top_p": args.top_p,
        "repetition_penalty": args.repetition_penalty,
        "streaming_interval_sec": args.streaming_interval,
        "streaming_chunk_size": 300,
        "streaming_left_context_size": 25,
        "codec_steps_per_second": codec_steps_per_second,
        "decode_upsample_rate": decode_upsample_rate,
        "num_codec_steps": int(codes.shape[0]),
        "num_quantizers": int(codes.shape[1]),
        "audio_duration_sec": float(len(wav) / sample_rate),
        "eos_position": int(codes.shape[0]),
        "chunk_boundaries": chunk_boundaries,
        "codec_trace": codes[:, 0].tolist(),
    }
    with (args.output_dir / "reference_contract.json").open("w", encoding="utf-8") as f:
        json.dump(contract, f, indent=2, sort_keys=True)

    print(f"Saved: {codes_path}")
    print(f"Saved: {args.output_dir / 'reference_contract.json'}")
    print(f"Saved: {args.output_dir / 'reference_audio.npy'}")


if __name__ == "__main__":
    main()
