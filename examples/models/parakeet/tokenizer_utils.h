#pragma once

#include "types.h"

#include <string>
#include <unordered_set>
#include <vector>

#include <tokenizer.h>

namespace parakeet::tokenizer_utils {

// Matches NeMo extract_punctuation_from_vocab method
// https://github.com/NVIDIA-NeMo/NeMo/blob/b90a528/nemo/collections/asr/parts/utils/tokenizer_utils.py#L20
std::unordered_set<std::string> derive_supported_punctuation(
    const tokenizers::Tokenizer& tokenizer);

std::string decode_token_sequence(
    const std::vector<TokenId>& tokens,
    const tokenizers::Tokenizer& tokenizer);

// convenience overload
std::string decode_token_sequence(
    const std::vector<Token>& decoded_tokens,
    const tokenizers::Tokenizer& tokenizer);

} // namespace parakeet::tokenizer_utils
