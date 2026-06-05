# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Shared Gemma 4 chat-template special-token IDs and the image+text input builder.

Single source of truth for the multimodal chat-template token layout. The
Python eager runner (``gemma4_31b/inference.py``) imports from here, and the C++
runner mirrors these exact values in ``gemma4/runner/chat_template.h`` so the two
implementations can never drift.

Image+text turn layout (matches the Gemma 4 HF chat template):

    <bos><start_of_turn>user\n<boi><image>*N<eoi>{prompt}<end_of_turn>\n
    <start_of_turn>model\n
"""

# Gemma 4 special token IDs (match the tokenizer + the C++ runner constants).
BOS_ID = 2
TURN_START_ID = 105  # <start_of_turn>
TURN_END_ID = 106  # <end_of_turn>
BOI_TOKEN_ID = 255999  # <start_of_image>
IMAGE_TOKEN_ID = 258880  # <image> soft-token placeholder
EOI_TOKEN_ID = 258882  # <end_of_image>


def build_vision_input_ids(
    tokenizer,
    prompt: str,
    num_vision_tokens: int,
    bos_id: int = BOS_ID,
) -> list[int]:
    """Build the chat-template token sequence for an image+text turn.

    Produces the same layout the C++ runner builds in
    ``gemma4/runner/chat_template.h::build_vision_input_ids``:

        <bos><start_of_turn>user\\n<boi><image>*N<eoi>{prompt}<end_of_turn>\\n
        <start_of_turn>model\\n

    Args:
        tokenizer: a ``tokenizers.Tokenizer``-like object exposing
            ``encode(str).ids``.
        prompt: the user text prompt.
        num_vision_tokens: number of ``<image>`` soft-token placeholders to
            insert (one per valid vision soft token).
        bos_id: beginning-of-sequence id (defaults to the Gemma 4 BOS).

    Returns:
        The flat list of token IDs for the turn.
    """
    user_tokens = tokenizer.encode("user\n").ids
    prompt_tokens = tokenizer.encode(prompt).ids
    newline_tokens = tokenizer.encode("\n").ids
    model_tokens = tokenizer.encode("model\n").ids

    ids: list[int] = []
    ids.append(bos_id)
    ids.append(TURN_START_ID)
    ids.extend(user_tokens)
    ids.append(BOI_TOKEN_ID)
    ids.extend([IMAGE_TOKEN_ID] * num_vision_tokens)
    ids.append(EOI_TOKEN_ID)
    ids.extend(prompt_tokens)
    ids.append(TURN_END_ID)
    ids.extend(newline_tokens)
    ids.append(TURN_START_ID)
    ids.extend(model_tokens)
    return ids


__all__ = [
    "BOS_ID",
    "TURN_START_ID",
    "TURN_END_ID",
    "BOI_TOKEN_ID",
    "IMAGE_TOKEN_ID",
    "EOI_TOKEN_ID",
    "build_vision_input_ids",
]
