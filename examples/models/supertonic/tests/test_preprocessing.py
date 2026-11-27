# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import json

import numpy as np
import pytest

from examples.models.supertonic.preprocessing import (
    AVAILABLE_LANGUAGES,
    chunk_text_for_language,
    preprocess_text,
    UnicodeProcessor,
)


def test_preprocess_text_normalizes_nfkd_and_adds_language_tags() -> None:
    assert preprocess_text("Café", "en") == "<en>Cafe\u0301.</en>"


def test_available_languages_match_published_model() -> None:
    assert AVAILABLE_LANGUAGES == (
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


def test_preprocess_text_cleans_punctuation_emoji_and_expressions() -> None:
    text = "“Hello” — world_🙂 @ x ♥ e.g., i.e., [done]"

    assert (
        preprocess_text(text, "en")
        == '<en>"Hello" - world at x for example, that is, done.</en>'
    )


def test_preprocess_text_rejects_invalid_language() -> None:
    with pytest.raises(ValueError, match="Invalid language: xx"):
        preprocess_text("Hello", "xx")


def test_unicode_processor_returns_deterministic_ids_and_masks(tmp_path) -> None:
    indexer_path = tmp_path / "unicode_indexer.json"
    indexer_path.write_text(json.dumps(list(range(128))), encoding="utf-8")
    processor = UnicodeProcessor(indexer_path, vocabulary_size=128)

    text_ids, text_mask = processor(["A", "Hi!"], ["en", "en"])

    np.testing.assert_array_equal(
        text_ids[0],
        [60, 101, 110, 62, 65, 46, 60, 47, 101, 110, 62, 0],
    )
    np.testing.assert_array_equal(
        text_ids[1],
        [60, 101, 110, 62, 72, 105, 33, 60, 47, 101, 110, 62],
    )
    np.testing.assert_array_equal(text_mask[0, 0], [1] * 11 + [0])
    np.testing.assert_array_equal(text_mask[1, 0], [1] * 12)
    assert text_ids.dtype == np.int64
    assert text_mask.dtype == np.float32


def test_unicode_processor_rejects_empty_batch(tmp_path) -> None:
    indexer_path = tmp_path / "unicode_indexer.json"
    indexer_path.write_text(json.dumps(list(range(128))), encoding="utf-8")
    processor = UnicodeProcessor(indexer_path, vocabulary_size=128)

    with pytest.raises(ValueError, match="at least one text and language"):
        processor([], [])


def test_unicode_processor_rejects_mismatched_cardinality(tmp_path) -> None:
    indexer_path = tmp_path / "unicode_indexer.json"
    indexer_path.write_text(json.dumps(list(range(128))), encoding="utf-8")
    processor = UnicodeProcessor(indexer_path, vocabulary_size=128)

    with pytest.raises(ValueError, match="same cardinality"):
        processor(["Hello", "World"], ["en"])


def test_unicode_processor_rejects_unsupported_and_out_of_range_tokens(
    tmp_path,
) -> None:
    unsupported = list(range(128))
    unsupported[ord("A")] = -1
    unsupported_path = tmp_path / "unsupported.json"
    unsupported_path.write_text(json.dumps(unsupported), encoding="utf-8")
    processor = UnicodeProcessor(unsupported_path, vocabulary_size=128)
    with pytest.raises(ValueError, match="unsupported Unicode codepoint 65"):
        processor(["A"], ["en"])

    out_of_range = list(range(128))
    out_of_range[ord("A")] = 128
    out_of_range_path = tmp_path / "out_of_range.json"
    out_of_range_path.write_text(json.dumps(out_of_range), encoding="utf-8")
    with pytest.raises(ValueError, match="invalid vocabulary token"):
        UnicodeProcessor(out_of_range_path, vocabulary_size=128)

    invalid_type = list(range(128))
    invalid_type[ord("A")] = True
    invalid_type_path = tmp_path / "invalid_type.json"
    invalid_type_path.write_text(json.dumps(invalid_type), encoding="utf-8")
    with pytest.raises(ValueError, match="invalid vocabulary token"):
        UnicodeProcessor(invalid_type_path, vocabulary_size=128)

    with pytest.raises(ValueError, match="text vocabulary size must be positive"):
        UnicodeProcessor(unsupported_path, vocabulary_size=0)


def test_chunk_text_uses_120_for_korean_and_300_for_english() -> None:
    text = f"{'가' * 60}. {'나' * 60}. {'다' * 10}."

    assert chunk_text_for_language(text, "ko") == [
        f"{'가' * 60}.",
        f"{'나' * 60}. {'다' * 10}.",
    ]
    assert chunk_text_for_language(text, "en") == [text]


def test_chunk_text_splits_cjk_terminators_without_spaces() -> None:
    first = "あ" * 60 + "。"
    second = "い" * 59 + "！？"
    third = "う" * 10 + "？"

    expected = [first, second + " " + third]
    assert chunk_text_for_language(first + second + third, "ja") == expected
    assert chunk_text_for_language(first + " \u3000" + second + third, "ja") == expected


@pytest.mark.parametrize(("language", "threshold"), [("ko", 120), ("en", 300)])
def test_chunk_text_keeps_a_single_sentence_over_soft_limit(
    language: str, threshold: int
) -> None:
    sentence = f"{'x' * threshold}."

    assert chunk_text_for_language(sentence, language) == [sentence]
