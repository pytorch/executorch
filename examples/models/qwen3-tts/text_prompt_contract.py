from dataclasses import dataclass

import torch


ASSISTANT_ROLE_PREFIX = "<|im_start|>assistant\n"
ASSISTANT_ROLE_SUFFIX = "<|im_end|>\n<|im_start|>assistant\n"
ROLE_TOKEN_COUNT = 3
FIRST_TEXT_TOKEN_COUNT = 1
TRAILING_TEMPLATE_TOKEN_COUNT = 5
MIN_PROMPT_TOKEN_COUNT = (
    ROLE_TOKEN_COUNT + FIRST_TEXT_TOKEN_COUNT + TRAILING_TEMPLATE_TOKEN_COUNT
)
TEXT_ONLY_CODEC_PREFIX_TOKEN_COUNT = 5
TEXT_ONLY_COMBINED_PREFIX_TOKEN_COUNT = TEXT_ONLY_CODEC_PREFIX_TOKEN_COUNT - 1
TEXT_ONLY_PREFILL_TOKEN_COUNT = (
    ROLE_TOKEN_COUNT + TEXT_ONLY_COMBINED_PREFIX_TOKEN_COUNT + FIRST_TEXT_TOKEN_COUNT
)
TEXT_ONLY_PREFILL_TOKEN_COUNT_WITH_LANGUAGE = TEXT_ONLY_PREFILL_TOKEN_COUNT + 1


@dataclass
class PromptEmbeddingParts:
    role_embed: torch.Tensor
    first_text_embed: torch.Tensor
    trailing_text_hidden: torch.Tensor


@dataclass
class TextOnlyRuntimePlan:
    prefill_token_count: int
    trailing_token_count: int
    min_required_generation_steps: int


def build_assistant_prompt_text(text: str) -> str:
    return f"{ASSISTANT_ROLE_PREFIX}{text}{ASSISTANT_ROLE_SUFFIX}"


def split_prompt_embeddings(
    prompt_embeds: torch.Tensor,
    tts_eos_embed: torch.Tensor,
) -> PromptEmbeddingParts:
    if prompt_embeds.dim() != 3:
        raise ValueError(
            f"prompt_embeds must have shape [B, S, D], got {tuple(prompt_embeds.shape)}"
        )
    if tts_eos_embed.dim() != 3 or tts_eos_embed.shape[1] != 1:
        raise ValueError(
            f"tts_eos_embed must have shape [B, 1, D], got {tuple(tts_eos_embed.shape)}"
        )
    if prompt_embeds.shape[0] != tts_eos_embed.shape[0]:
        raise ValueError("prompt_embeds and tts_eos_embed batch dimensions must match")
    if prompt_embeds.shape[2] != tts_eos_embed.shape[2]:
        raise ValueError("prompt_embeds and tts_eos_embed hidden sizes must match")
    if prompt_embeds.shape[1] < MIN_PROMPT_TOKEN_COUNT:
        raise ValueError(
            "assistant prompt is too short to split into role, first text token, "
            "and trailing template segments"
        )

    role_embed = prompt_embeds[:, :ROLE_TOKEN_COUNT, :]
    first_text_embed = prompt_embeds[
        :, ROLE_TOKEN_COUNT : ROLE_TOKEN_COUNT + FIRST_TEXT_TOKEN_COUNT, :
    ]
    trailing_text_hidden = torch.cat(
        [
            prompt_embeds[
                :, ROLE_TOKEN_COUNT + FIRST_TEXT_TOKEN_COUNT : -TRAILING_TEMPLATE_TOKEN_COUNT, :
            ],
            tts_eos_embed,
        ],
        dim=1,
    )
    return PromptEmbeddingParts(
        role_embed=role_embed,
        first_text_embed=first_text_embed,
        trailing_text_hidden=trailing_text_hidden,
    )


def build_text_only_runtime_plan(
    prompt_token_count: int,
    max_seq_len: int,
    max_new_tokens: int,
    use_language_prefix: bool = False,
) -> TextOnlyRuntimePlan:
    if prompt_token_count < MIN_PROMPT_TOKEN_COUNT:
        raise ValueError(
            "assistant prompt is too short to produce the text-only runtime plan"
        )

    prefill_token_count = (
        TEXT_ONLY_PREFILL_TOKEN_COUNT_WITH_LANGUAGE
        if use_language_prefix
        else TEXT_ONLY_PREFILL_TOKEN_COUNT
    )
    trailing_token_count = (
        prompt_token_count
        - ROLE_TOKEN_COUNT
        - FIRST_TEXT_TOKEN_COUNT
        - TRAILING_TEMPLATE_TOKEN_COUNT
        + 1
    )
    if max_new_tokens < trailing_token_count:
        raise ValueError(
            "max_new_tokens is too small to consume the remaining prompt tokens"
        )
    if prefill_token_count + max_new_tokens > max_seq_len:
        raise ValueError(
            "max_seq_len is too small for the requested prefill and generation budget"
        )

    return TextOnlyRuntimePlan(
        prefill_token_count=prefill_token_count,
        trailing_token_count=trailing_token_count,
        min_required_generation_steps=trailing_token_count,
    )
