#pragma once

#include "types.h"

#include <optional>
#include <string>
#include <unordered_set>
#include <vector>

#include <tokenizer.h>

namespace parakeet::timestamp_utils {

// throws if any tokenizer calls fail
std::vector<TokenWithTextInfo> get_tokens_with_text_info(
    const std::vector<Token>& tokens,
    const tokenizers::Tokenizer& tokenizer,
    const std::unordered_set<std::string>& supported_punctuation);

// ref:
// https://github.com/NVIDIA-NeMo/NeMo/blob/bf583c9/nemo/collections/asr/parts/utils/timestamp_utils.py#L54
// assumes BPE tokenizer type
std::vector<TextWithOffsets> get_words_offsets(
    const std::vector<TokenWithTextInfo>& tokens,
    const tokenizers::Tokenizer& tokenizer,
    const std::unordered_set<std::string>& supported_punctuation,
    const std::string& word_delimiter_char = " ");

// ref
// https://github.com/NVIDIA-NeMo/NeMo/blob/bf583c9/nemo/collections/asr/parts/utils/timestamp_utils.py#L227
std::vector<TextWithOffsets> get_segment_offsets(
    const std::vector<TextWithOffsets>& word_offsets,
    const std::vector<std::string>& segment_delimiters = {".", "?", "!"},
    const std::optional<int64_t>& segment_gap_threshold = std::nullopt);

} // namespace parakeet::timestamp_utils
