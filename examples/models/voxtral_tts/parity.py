import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import torch


@dataclass
class PromptLayout:
    token_ids: list[int]
    voice_start: int
    voice_len: int


@dataclass
class SeedDecodeTrace:
    prefill_hidden: torch.Tensor
    seed_hidden: torch.Tensor
    seed_embed: torch.Tensor
    seed_position: int


def build_reference_prompt_ids(
    text_tokens: list[int],
    voice_len: int,
    begin_audio_token_id: int,
    audio_token_id: int,
    text_to_audio_token_id: int,
    repeat_audio_text_token_id: int,
    bos_token_id: int = 1,
) -> PromptLayout:
    token_ids = [bos_token_id, begin_audio_token_id]
    voice_start = len(token_ids)
    if voice_len > 0:
        token_ids.extend([audio_token_id] * voice_len)
    token_ids.append(text_to_audio_token_id)
    token_ids.extend(text_tokens)
    token_ids.append(repeat_audio_text_token_id)
    token_ids.append(begin_audio_token_id)
    return PromptLayout(
        token_ids=token_ids,
        voice_start=voice_start,
        voice_len=voice_len,
    )


def encode_speech_request_tokens(
    tokenizer_path: str | Path,
    text: str,
    voice: str,
) -> list[int]:
    from mistral_common.protocol.speech.request import SpeechRequest
    from mistral_common.tokens.tokenizers.mistral import MistralTokenizer

    tokenizer = MistralTokenizer.from_file(str(tokenizer_path))
    return tokenizer.encode_speech_request(
        SpeechRequest(input=text, voice=voice)
    ).tokens


def splice_voice_embeddings(
    prompt_embeds: torch.Tensor,
    voice_embed: torch.Tensor,
    voice_start: int,
) -> torch.Tensor:
    if voice_embed.numel() == 0:
        return prompt_embeds
    prompt_embeds = prompt_embeds.clone()
    voice_len = voice_embed.shape[0]
    prompt_embeds[:, voice_start : voice_start + voice_len, :] = voice_embed.unsqueeze(0)
    return prompt_embeds


def run_seed_decode(
    token_embedding: torch.nn.Module,
    decoder: torch.nn.Module,
    audio_token_id: int,
    prompt_embeds: torch.Tensor,
) -> SeedDecodeTrace:
    prompt_len = prompt_embeds.shape[1]
    device = prompt_embeds.device
    input_pos = torch.arange(prompt_len, dtype=torch.long, device=device)
    hidden_all = decoder(prompt_embeds, input_pos)
    prefill_hidden = hidden_all[:, -1, :].clone()

    seed_ids = torch.tensor([[audio_token_id]], dtype=torch.long, device=device)
    seed_embed = token_embedding(seed_ids)
    seed_pos = torch.tensor([prompt_len], dtype=torch.long, device=device)
    seed_hidden = decoder(seed_embed, seed_pos)[:, 0, :].clone()
    return SeedDecodeTrace(
        prefill_hidden=prefill_hidden,
        seed_hidden=seed_hidden,
        seed_embed=seed_embed.clone(),
        seed_position=prompt_len,
    )


def topk_pairs(logits: torch.Tensor, k: int = 5) -> list[dict[str, float | int]]:
    topk_vals, topk_ids = logits.float().topk(k)
    return [
        {"id": int(token_id), "logit": float(value)}
        for token_id, value in zip(topk_ids.tolist(), topk_vals.tolist())
    ]


def tensor_summary(tensor: torch.Tensor, limit: int = 8) -> dict[str, Any]:
    flat = tensor.detach().float().reshape(-1).cpu()
    values = flat[:limit].tolist()
    return {
        "shape": list(tensor.shape),
        "min": float(flat.min().item()) if flat.numel() else 0.0,
        "max": float(flat.max().item()) if flat.numel() else 0.0,
        "mean": float(flat.mean().item()) if flat.numel() else 0.0,
        "head": [float(v) for v in values],
    }


def _max_abs_diff(lhs: list[float], rhs: list[float]) -> float:
    if len(lhs) != len(rhs):
        return float("inf")
    if not lhs:
        return 0.0
    return max(abs(float(a) - float(b)) for a, b in zip(lhs, rhs))


def _compare_optional_tensor_field(
    reference: dict[str, Any],
    candidate: dict[str, Any],
    *,
    field: str,
    atol: float,
) -> dict[str, Any] | None:
    ref_value = reference.get(field)
    cand_value = candidate.get(field)
    if ref_value is None and cand_value is None:
        return None
    max_diff = _max_abs_diff(ref_value or [], cand_value or [])
    return {
        "name": field,
        "ok": max_diff <= atol,
        "max_abs_diff": max_diff,
        "hidden_atol": atol,
        "reference_len": len(ref_value or []),
        "candidate_len": len(cand_value or []),
    }


def _compare_optional_scalar_field(
    reference: dict[str, Any],
    candidate: dict[str, Any],
    *,
    field: str,
) -> dict[str, Any] | None:
    ref_value = reference.get(field)
    cand_value = candidate.get(field)
    if ref_value is None and cand_value is None:
        return None
    return {
        "name": field,
        "ok": ref_value == cand_value,
        "reference": ref_value,
        "candidate": cand_value,
    }


def compare_trace_payloads(
    reference: dict[str, Any],
    candidate: dict[str, Any],
    hidden_atol: float = 1e-4,
) -> dict[str, Any]:
    checks: list[dict[str, Any]] = []

    def add_check(name: str, ok: bool, details: dict[str, Any]) -> None:
        checks.append({"name": name, "ok": ok, **details})

    prompt_match = reference.get("prompt_token_ids") == candidate.get("prompt_token_ids")
    add_check(
        "prompt_token_ids",
        prompt_match,
        {
            "reference_len": len(reference.get("prompt_token_ids", [])),
            "candidate_len": len(candidate.get("prompt_token_ids", [])),
        },
    )

    voice_len_match = reference.get("voice_len") == candidate.get("voice_len")
    add_check(
        "voice_len",
        voice_len_match,
        {
            "reference": reference.get("voice_len"),
            "candidate": candidate.get("voice_len"),
        },
    )

    for field in (
        "prefill_hidden",
        "frame0_hidden",
        "seed_hidden",
        "frame0_audio_embed",
        "frame1_hidden",
    ):
        check = _compare_optional_tensor_field(
            reference,
            candidate,
            field=field,
            atol=hidden_atol,
        )
        if check is not None:
            add_check(
                check.pop("name"),
                check.pop("ok"),
                check,
            )

    for field in ("seed_position", "frame0_position", "frame1_position"):
        check = _compare_optional_scalar_field(reference, candidate, field=field)
        if check is not None:
            add_check(
                check.pop("name"),
                check.pop("ok"),
                check,
            )

    check = _compare_optional_scalar_field(reference, candidate, field="seed_step_applied")
    if check is not None:
        add_check(
            check.pop("name"),
            check.pop("ok"),
            check,
        )

    codes_check = _compare_optional_scalar_field(reference, candidate, field="frame0_full_codes")
    if codes_check is not None:
        add_check(
            codes_check.pop("name"),
            codes_check.pop("ok"),
            codes_check,
        )

    ref_frames = reference.get("frames", [])
    cand_frames = candidate.get("frames", [])
    compared_frames = min(len(ref_frames), len(cand_frames), 3)
    for frame_idx in range(compared_frames):
        ref_frame = ref_frames[frame_idx]
        cand_frame = cand_frames[frame_idx]
        semantic_match = ref_frame.get("semantic_code") == cand_frame.get("semantic_code")
        add_check(
            f"frame{frame_idx}_semantic_code",
            semantic_match,
            {
                "reference": ref_frame.get("semantic_code"),
                "candidate": cand_frame.get("semantic_code"),
            },
        )
        codes_match = ref_frame.get("full_codes") == cand_frame.get("full_codes")
        add_check(
            f"frame{frame_idx}_codes",
            codes_match,
            {
                "reference": ref_frame.get("full_codes"),
                "candidate": cand_frame.get("full_codes"),
            },
        )

    if len(ref_frames) != len(cand_frames):
        add_check(
            "frame_count",
            False,
            {
                "reference": len(ref_frames),
                "candidate": len(cand_frames),
            },
        )

    return {
        "ok": all(check["ok"] for check in checks),
        "checks": checks,
    }


def write_trace_json(path: str | Path, payload: dict[str, Any]) -> None:
    serializable = {}
    for key, value in payload.items():
        if isinstance(value, torch.Tensor):
            serializable[key] = tensor_summary(value)
        elif hasattr(value, "__dataclass_fields__"):
            serializable[key] = asdict(value)
        else:
            serializable[key] = value
    Path(path).write_text(json.dumps(serializable, indent=2, sort_keys=True) + "\n")
