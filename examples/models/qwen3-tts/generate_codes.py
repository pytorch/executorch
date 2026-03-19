import argparse
import json
import sys
from pathlib import Path
from typing import List, Optional

import numpy as np
import torch

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from model import write_codes_binary  # noqa: E402
from qwen_tts import Qwen3TTSModel  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate Qwen3-TTS codec ids for downstream ExecuTorch decode."
    )
    parser.add_argument("--model-id-or-path", required=True, type=str)
    parser.add_argument("--text", required=True, type=str)
    parser.add_argument("--language", default="English", type=str)
    parser.add_argument("--output-codes", required=True, type=Path)
    parser.add_argument("--output-meta", default=None, type=Path)
    parser.add_argument("--cache-dir", default=None, type=str)
    parser.add_argument("--ref-audio", default=None, type=str)
    parser.add_argument("--ref-text", default=None, type=str)
    parser.add_argument(
        "--x-vector-only-mode",
        action="store_true",
        help="Use x-vector only mode for voice clone prompt.",
    )
    parser.add_argument("--non-streaming-mode", action="store_true")
    parser.add_argument("--dtype", choices=["fp32", "bf16"], default="fp32")
    parser.add_argument("--max-new-tokens", type=int, default=None)
    parser.add_argument("--top-k", type=int, default=None)
    parser.add_argument("--top-p", type=float, default=None)
    parser.add_argument("--temperature", type=float, default=None)
    parser.add_argument("--repetition-penalty", type=float, default=None)
    parser.add_argument(
        "--trim-silence",
        action="store_true",
        help="Trim leading silence codes. In streaming mode, the model generates "
        "~N silent codes (where N ≈ text token count) before speech begins. "
        "This decodes a small chunk to detect where speech starts and strips "
        "the silent prefix, saving compute in the downstream decoder.",
    )
    parser.add_argument(
        "--trim-threshold",
        type=float,
        default=0.005,
        help="RMS threshold for silence detection (default: 0.005).",
    )
    return parser.parse_args()


def _default_reference_audio(duration_sec: float = 1.0, sample_rate: int = 24000):
    wav = np.zeros(int(duration_sec * sample_rate), dtype=np.float32)
    return wav, sample_rate


def _trim_silent_prefix(
    codes: torch.Tensor,
    model,
    metadata_decoder_config=None,
    threshold: float = 0.005,
    upsample_rate: int = 1920,
    sample_rate: int = 24000,
    chunk_size: int = 5,
) -> torch.Tensor:
    """Trim leading silent codes by decoding small chunks and checking RMS energy.

    In streaming mode, the talker generates ~N silent codes (N ≈ text token count)
    while absorbing text before producing speech. This function finds where speech
    starts by decoding codes in small chunks and checking audio energy.

    Returns the trimmed codes tensor [T', Q] with silent prefix removed.
    """
    t_len, n_q = codes.shape
    if t_len <= chunk_size:
        return codes

    decoder = model.model.talker.speech_tokenizer.decoder
    speech_start = 0

    for start in range(0, t_len - chunk_size + 1, chunk_size):
        end = min(start + chunk_size, t_len)
        chunk = codes[start:end].unsqueeze(0)
        chunk_clamped = torch.clamp(chunk, min=0)
        with torch.no_grad():
            wav = decoder(chunk_clamped.transpose(1, 2)).squeeze()
        rms = torch.sqrt(torch.mean(wav**2)).item()
        if rms > threshold:
            speech_start = max(0, start - 1)
            break
    else:
        return codes

    if speech_start > 0:
        print(
            f"Trimmed {speech_start} silent codes "
            f"({speech_start * upsample_rate / sample_rate:.1f}s silence)"
        )
    return codes[speech_start:]


def _build_ref_ids(
    model: Qwen3TTSModel, prompt_items
) -> List[Optional[torch.Tensor]]:
    ref_ids = []
    for item in prompt_items:
        if item.ref_text is None or item.ref_text == "":
            ref_ids.append(None)
            continue
        ref_tok = model._tokenize_texts([model._build_ref_text(item.ref_text)])[0]
        ref_ids.append(ref_tok)
    return ref_ids


def main() -> None:
    args = parse_args()
    args.output_codes.parent.mkdir(parents=True, exist_ok=True)

    dtype = {"fp32": torch.float32, "bf16": torch.bfloat16}[args.dtype]
    model = Qwen3TTSModel.from_pretrained(
        args.model_id_or_path,
        device_map="cpu",
        dtype=dtype,
        cache_dir=args.cache_dir,
    )

    if args.ref_audio is not None:
        prompt_items = model.create_voice_clone_prompt(
            ref_audio=args.ref_audio,
            ref_text=args.ref_text,
            x_vector_only_mode=args.x_vector_only_mode,
        )
    else:
        silence, sr = _default_reference_audio()
        prompt_items = model.create_voice_clone_prompt(
            ref_audio=(silence, sr),
            ref_text=None,
            x_vector_only_mode=True,
        )

    prompt_dict = model._prompt_items_to_voice_clone_prompt(prompt_items)
    input_ids = model._tokenize_texts([model._build_assistant_text(args.text)])
    ref_ids = _build_ref_ids(model, prompt_items)

    gen_kwargs = model._merge_generate_kwargs(
        max_new_tokens=args.max_new_tokens,
        top_k=args.top_k,
        top_p=args.top_p,
        temperature=args.temperature,
        repetition_penalty=args.repetition_penalty,
    )
    talker_codes_list, _ = model.model.generate(
        input_ids=input_ids,
        ref_ids=ref_ids,
        voice_clone_prompt=prompt_dict,
        languages=[args.language],
        non_streaming_mode=args.non_streaming_mode,
        **gen_kwargs,
    )
    codes = talker_codes_list[0].detach().cpu()

    if args.trim_silence:
        codes = _trim_silent_prefix(
            codes, model, metadata_decoder_config=None, threshold=args.trim_threshold
        )

    write_codes_binary(args.output_codes, codes)

    meta = {
        "model_id_or_path": args.model_id_or_path,
        "language": args.language,
        "text": args.text,
        "num_codes": int(codes.shape[0]),
        "num_quantizers": int(codes.shape[1]),
        "x_vector_only_mode": bool(
            args.x_vector_only_mode or args.ref_audio is None
        ),
        "ref_audio_provided": args.ref_audio is not None,
        "non_streaming_mode": args.non_streaming_mode,
        "trim_silence": args.trim_silence,
    }
    meta_path = args.output_meta or args.output_codes.with_suffix(".json")
    with meta_path.open("w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2, sort_keys=True)

    print(f"Saved codec ids: {args.output_codes}")
    print(f"Saved metadata: {meta_path}")


if __name__ == "__main__":
    main()
