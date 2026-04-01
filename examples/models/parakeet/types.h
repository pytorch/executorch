#pragma once

#include <cstdint>
#include <string>

namespace parakeet {

// Matches output type of tokenizers::Tokenizer methods
using TokenId = uint64_t;

struct Token {
  TokenId id;
  int64_t start_offset;
  int64_t duration;
};

struct TokenWithTextInfo {
  TokenId id;
  // Raw vocabulary piece for the token_id (i.e., "##ing", "▁hello")
  std::string raw_piece;
  // Decoded text for the token_id (i.e., "ing", " hello")
  std::string decoded_text;
  int64_t start_offset;
  int64_t end_offset;
};

struct TextWithOffsets {
  std::string text;
  int64_t start_offset;
  int64_t end_offset;
};

} // namespace parakeet
