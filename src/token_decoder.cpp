/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */
// @lint-ignore-every LICENSELINT

#include <pytorch/tokenizers/token_decoder.h>

// Standard
#include <cstdarg>

// Third Party
#include <nlohmann/json.hpp>

// Local
#include <pytorch/tokenizers/third-party/llama.cpp-unicode/unicode.h>

using json = nlohmann::json;

namespace tokenizers {

// TokenDecoderConfig //////////////////////////////////////////////////////////

TokenDecoderConfig::TokenDecoderConfig(std::string type)
    : type(std::move(type)) {}

TokenDecoder::Ptr TokenDecoderConfig::create() const {
  // NOTE: These types must line up with the type strings found in the
  //  tokenizers library
  //  https://github.com/huggingface/tokenizers/blob/main/tokenizers/src/decoders/mod.rs#L55
  if (type == "ByteLevel") {
    return TokenDecoder::Ptr(new ByteLevelTokenDecoder());
  }
  throw std::runtime_error("Unsupported TokenDecoder type: " + type);
}

TokenDecoderConfig &TokenDecoderConfig::parse_json(const json &json_config) {
  type = json_config.at("type");
  if (type == "ByteLevel") {
    // No parameters to parse
  } else {
    throw std::runtime_error("Unsupported TokenDecoder type: " + type);
  }
  return *this;
}

// ByteLevel ///////////////////////////////////////////////////////////////////

namespace {

// Copied from llama.cpp
// CITE:
// https://github.com/ggerganov/llama.cpp/blob/master/src/llama-vocab.cpp#L20
static std::string format(const char *fmt, ...) {
  va_list ap;
  va_list ap2;
  va_start(ap, fmt);
  va_copy(ap2, ap);
  int size = vsnprintf(NULL, 0, fmt, ap);
  // GGML_ASSERT(size >= 0 && size < INT_MAX); // NOLINT
  std::vector<char> buf(size + 1);
  // int size2 = vsnprintf(buf.data(), size + 1, fmt, ap2);
  // GGML_ASSERT(size2 == size);
  va_end(ap2);
  va_end(ap);
  return std::string(buf.data(), size);
}

} // namespace

std::string ByteLevelTokenDecoder::decode(re2::StringPiece token) const {
  // This is borrowed and lightly tweaked from llama.cpp
  // CITE:
  // https://github.com/ggerganov/llama.cpp/blob/master/src/llama-vocab.cpp#L1755
  std::string decoded_text;
  // TODO: This could be more efficient since what we really need is a string
  //  const ref.
  std::string text(token);
  const auto cpts = unicode_cpts_from_utf8(text);
  for (const auto cpt : cpts) {
    const auto utf8 = unicode_cpt_to_utf8(cpt);
    try {
      decoded_text += unicode_utf8_to_byte(utf8);
    } catch (const std::out_of_range & /*e*/) {
      decoded_text += "[UNK_BYTE_0x";
      for (const auto c : utf8) {
        decoded_text += format("%02x", (uint8_t)c);
      }
      decoded_text += text + "]";
    }
  }

  return decoded_text;
}

} // end  namespace tokenizers
