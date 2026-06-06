#pragma once

#include <cstdint>
#include <string>

#include <executorch/extension/asr/runner/transducer_runner.h>

namespace parakeet {

// Use the shared Token type from the ASR runner.
using Token = ::executorch::extension::asr::Token;
using TokenId = uint64_t;

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
