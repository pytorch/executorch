# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Prompt matrix and the reference token builder behind the golden tests.

The builder states the runner's prompt contract rather than mirroring its
implementation, so a refactor that preserves the contract leaves the golden
untouched and a refactor that does not shows up as a golden diff.
"""

import hashlib
import os
from dataclasses import dataclass

BOS_ID = 200000
IMAGE_START_ID = 200080
IMAGE_END_ID = 200081
PATCH_ID = 200092
IMAGE_MARKER = "<img>"

# Tokens the image span costs beyond the patches themselves. Zero: the canonical
# format is a bare <|patch|> run, which is what the Muse Glimmer processor's
# replace_image_token method emits and what meta_reference_implementation splices on.
IMAGE_WRAPPER_TOKENS = 0

HF_DIR_ENV = "MUSE_GLIMMER_HF_DIR"

_SYSTEM = "<|start|>system<|message|>You are a helpful AI assistant.<|eot|>"
_ASSISTANT_HEADER = "<|start|>assistant"


@dataclass(frozen=True)
class Case:
    name: str
    prompt: str
    num_soft_tokens: int = 0


CASES = [
    Case("text_plain", "The meaning of life is"),
    Case(
        "harmony_single_turn",
        f"<|start|>user<|message|>What is 17 * 23?<|eot|>{_ASSISTANT_HEADER}",
    ),
    Case(
        "harmony_with_system",
        f"{_SYSTEM}<|start|>user<|message|>Name three primes.<|eot|>{_ASSISTANT_HEADER}",
    ),
    Case(
        "harmony_multi_turn",
        f"{_SYSTEM}<|start|>user<|message|>Hi.<|eot|>"
        "<|start|>assistant<|message|>Hello.<|eot|>"
        f"<|start|>user<|message|>And again?<|eot|>{_ASSISTANT_HEADER}",
    ),
    Case(
        "harmony_reasoning_then_answer",
        "<|start|>assistant to=self<|message|>Thinking.<|eom|>"
        "<|start|>assistant<|message|>Answer.<|eot|>",
    ),
    # The two prompts the README quotes, so the counts it cites stay honest.
    Case(
        "readme_minimal_user",
        "<|start|>user<|message|>What is the meaning of life?<|eot|>"
        f"{_ASSISTANT_HEADER}",
    ),
    Case(
        "readme_target_only",
        "<|start|>system<|message|>You are a helpful assistant.<|eot|>"
        "<|start|>user<|message|>What is the meaning of life?<|eot|>"
        f"{_ASSISTANT_HEADER}",
    ),
    Case(
        "image_small",
        f"<|start|>user<|message|>Describe this image: {IMAGE_MARKER}<|eot|>"
        f"{_ASSISTANT_HEADER}",
        num_soft_tokens=4,
    ),
    Case(
        "image_large",
        f"<|start|>user<|message|>Describe this image: {IMAGE_MARKER}<|eot|>"
        f"{_ASSISTANT_HEADER}",
        num_soft_tokens=256,
    ),
]


def text_only_len(golden_case, case):
    """Golden length with the image span removed, so a real image can be sized in."""
    if not case.num_soft_tokens:
        return golden_case["num_tokens"]
    return golden_case["num_tokens"] - case.num_soft_tokens - IMAGE_WRAPPER_TOKENS


def build_prompt_ids(encode, case):
    """Prompt contract: BOS, with the image patch run replacing ``<img>``.

    ``encode`` takes text and returns ids without special-token insertion.
    """
    ids = [BOS_ID]
    if not case.num_soft_tokens:
        if IMAGE_MARKER in case.prompt:
            raise ValueError("text-only prompt cannot contain <img>")
        ids.extend(encode(case.prompt))
        return ids

    if case.prompt.count(IMAGE_MARKER) != 1:
        raise ValueError("image prompt must contain exactly one <img>")
    before, after = case.prompt.split(IMAGE_MARKER)
    ids.extend(encode(before))
    ids.extend([PATCH_ID] * case.num_soft_tokens)
    ids.extend(encode(after))
    return ids


def stub_encode(text):
    """Deterministic stand-in for the BPE, so shape can be tested without assets.

    Special-token substrings resolve to their real ids; everything else becomes
    one id per character in a range that cannot collide with them.
    """
    specials = {
        "<|begin_of_text|>": BOS_ID,
        "<|end_of_text|>": 200001,
        "<|eom|>": 200007,
        "<|eot|>": 200008,
        "<|start|>": 200022,
        "<|message|>": 200023,
        "<|image_start|>": IMAGE_START_ID,
        "<|image_end|>": IMAGE_END_ID,
        "<|patch|>": PATCH_ID,
    }
    ids = []
    i = 0
    while i < len(text):
        for piece, pid in specials.items():
            if text.startswith(piece, i):
                ids.append(pid)
                i += len(piece)
                break
        else:
            ids.append(ord(text[i]) % 1000)
            i += 1
    return ids


def hf_dir():
    """Directory holding the real tokenizer.json, or None when unset."""
    path = os.environ.get(HF_DIR_ENV)
    if path and os.path.isfile(
        os.path.join(os.path.expanduser(path), "tokenizer.json")
    ):
        return os.path.expanduser(path)
    return None


def real_encoder(directory):
    from tokenizers import Tokenizer

    tok = Tokenizer.from_file(os.path.join(directory, "tokenizer.json"))
    return lambda text: tok.encode(text, add_special_tokens=False).ids


def ids_sha256(ids):
    """Stable digest of an id sequence. Committed instead of the ids themselves."""
    return hashlib.sha256(",".join(str(i) for i in ids).encode()).hexdigest()


def file_sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def special_runs(ids):
    """Run-length view of the special-token skeleton: [[index, id, count], ...].

    Run-length keeps a 256-patch image case readable, and makes a golden diff say
    what actually moved instead of renumbering every following position.
    """
    runs = []
    for i, tok in enumerate(ids):
        if tok < 200000:
            continue
        if runs and runs[-1][1] == tok and i == runs[-1][0] + runs[-1][2]:
            runs[-1][2] += 1
        else:
            runs.append([i, tok, 1])
    return runs
