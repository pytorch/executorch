# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import json
import re
import unicodedata
from pathlib import Path
from typing import Sequence

import numpy as np
from numpy.typing import NDArray

AVAILABLE_LANGUAGES = (
    "en",
    "ko",
    "ja",
    "ar",
    "bg",
    "cs",
    "da",
    "de",
    "el",
    "es",
    "et",
    "fi",
    "fr",
    "hi",
    "hr",
    "hu",
    "id",
    "it",
    "lt",
    "lv",
    "nl",
    "pl",
    "pt",
    "ro",
    "ru",
    "sk",
    "sl",
    "sv",
    "tr",
    "uk",
    "vi",
    "na",
)

_EMOJI_PATTERN = re.compile(
    "[\U0001f600-\U0001f64f"
    "\U0001f300-\U0001f5ff"
    "\U0001f680-\U0001f6ff"
    "\U0001f700-\U0001f77f"
    "\U0001f780-\U0001f7ff"
    "\U0001f800-\U0001f8ff"
    "\U0001f900-\U0001f9ff"
    "\U0001fa00-\U0001fa6f"
    "\U0001fa70-\U0001faff"
    "\u2600-\u26ff"
    "\u2700-\u27bf"
    "\U0001f1e6-\U0001f1ff]+"
)
_SENTENCE_BOUNDARY_PATTERN = re.compile(
    r"(?<!Mr\.)(?<!Mrs\.)(?<!Ms\.)(?<!Dr\.)(?<!Prof\.)(?<!Sr\.)(?<!Jr\.)"
    r"(?<!Ph\.D\.)(?<!etc\.)(?<!e\.g\.)(?<!i\.e\.)(?<!vs\.)(?<!Inc\.)"
    r"(?<!Ltd\.)(?<!Co\.)(?<!Corp\.)(?<!St\.)(?<!Ave\.)(?<!Blvd\.)"
    r"(?<!\b[A-Z]\.)(?:(?<=[.!?])\s+|(?<=[。！？])(?![。！？])\s*)"
)


def preprocess_text(text: str, language: str) -> str:
    if language not in AVAILABLE_LANGUAGES:
        raise ValueError(f"Invalid language: {language}")

    text = unicodedata.normalize("NFKD", text)
    text = _EMOJI_PATTERN.sub("", text)
    for old, new in {
        "–": "-",
        "‑": "-",
        "—": "-",
        "_": " ",
        "“": '"',
        "”": '"',
        "‘": "'",
        "’": "'",
        "´": "'",
        "`": "'",
        "[": " ",
        "]": " ",
        "|": " ",
        "/": " ",
        "#": " ",
        "→": " ",
        "←": " ",
    }.items():
        text = text.replace(old, new)
    text = re.sub(r"[♥☆♡©\\]", "", text)
    for old, new in {
        "@": " at ",
        "e.g.,": "for example, ",
        "i.e.,": "that is, ",
    }.items():
        text = text.replace(old, new)
    text = re.sub(r" ([,.!?;:'])", r"\1", text)
    while '""' in text:
        text = text.replace('""', '"')
    while "''" in text:
        text = text.replace("''", "'")
    while "``" in text:
        text = text.replace("``", "`")
    text = re.sub(r"\s+", " ", text).strip()
    if not re.search(r"[.!?！？，;:,'\"')\]}…。」』】〉》›»]$", text):
        text += "."
    return f"<{language}>{text}</{language}>"


def length_to_mask(
    lengths: NDArray[np.int64], max_length: int | None = None
) -> NDArray[np.float32]:
    max_length = max_length if max_length is not None else int(lengths.max())
    positions = np.arange(max_length)
    return (positions < lengths[:, None]).astype(np.float32)[:, None, :]


class UnicodeProcessor:
    def __init__(
        self, unicode_indexer_path: str | Path, *, vocabulary_size: int
    ) -> None:
        if vocabulary_size <= 0:
            raise ValueError("text vocabulary size must be positive")
        with Path(unicode_indexer_path).open(encoding="utf-8") as indexer_file:
            self.indexer: Sequence[int] | dict[str, int] = json.load(indexer_file)
        token_ids = self.indexer.values() if isinstance(self.indexer, dict) else self.indexer
        if any(
            not isinstance(token_id, int)
            or isinstance(token_id, bool)
            or token_id < -1
            or token_id >= vocabulary_size
            for token_id in token_ids
        ):
            raise ValueError("Unicode indexer contains an invalid vocabulary token")
        self.vocabulary_size = vocabulary_size

    def _index(self, codepoint: int) -> int:
        try:
            if isinstance(self.indexer, dict):
                token_id = self.indexer[str(codepoint)]
            else:
                token_id = self.indexer[codepoint]
        except (IndexError, KeyError) as error:
            raise ValueError(
                f"Unicode indexer has no entry for codepoint {codepoint}"
            ) from error
        if token_id < 0:
            raise ValueError(f"unsupported Unicode codepoint {codepoint}")
        return token_id

    def __call__(
        self, texts: Sequence[str], languages: Sequence[str]
    ) -> tuple[NDArray[np.int64], NDArray[np.float32]]:
        if len(texts) != len(languages):
            raise ValueError("texts and languages must have the same cardinality")
        if len(texts) == 0:
            raise ValueError("expected at least one text and language")

        processed = [
            preprocess_text(text, language)
            for text, language in zip(texts, languages)
        ]
        lengths = np.asarray([len(text) for text in processed], dtype=np.int64)
        text_ids = np.zeros((len(processed), int(lengths.max())), dtype=np.int64)
        for index, text in enumerate(processed):
            text_ids[index, : len(text)] = [self._index(ord(char)) for char in text]
        return text_ids, length_to_mask(lengths)


def chunk_text(text: str, max_len: int = 300) -> list[str]:
    """Pack whole sentences using ``max_len`` as a soft threshold.

    A sentence is never split, so one sentence may exceed ``max_len``.
    """
    paragraphs = [
        paragraph.strip()
        for paragraph in re.split(r"\n\s*\n+", text.strip())
        if paragraph.strip()
    ]
    chunks: list[str] = []
    for paragraph in paragraphs:
        current_chunk = ""
        for sentence in _SENTENCE_BOUNDARY_PATTERN.split(paragraph):
            if len(current_chunk) + len(sentence) + 1 <= max_len:
                current_chunk += (" " if current_chunk else "") + sentence
            else:
                if current_chunk:
                    chunks.append(current_chunk.strip())
                current_chunk = sentence
        if current_chunk:
            chunks.append(current_chunk.strip())
    return chunks


def chunk_text_for_language(text: str, language: str) -> list[str]:
    """Use soft packing thresholds of 120 for ko/ja and 300 otherwise."""
    return chunk_text(text, max_len=120 if language in ("ko", "ja") else 300)
