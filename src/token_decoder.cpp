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
#include <unicode.h>

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
  } else if (type == "Replace") {
    // Use parsed pattern and content from JSON
    return TokenDecoder::Ptr(
        new ReplaceTokenDecoder(replace_pattern, replace_content));
  } else if (type == "ByteFallback") {
    return TokenDecoder::Ptr(new ByteFallbackTokenDecoder());
  } else if (type == "Fuse") {
    return TokenDecoder::Ptr(new FuseTokenDecoder());
  } else if (type == "Sequence") {
    // Parse the decoders array from JSON and create sub-decoders
    std::vector<TokenDecoder::Ptr> decoders;
    for (const auto& decoder_json : sequence_decoders) {
      TokenDecoderConfig sub_config;
      sub_config.parse_json(decoder_json);
      decoders.push_back(sub_config.create());
    }
    return TokenDecoder::Ptr(new SequenceTokenDecoder(std::move(decoders)));
  }
  throw std::runtime_error("Unsupported TokenDecoder type: " + type);
}

TokenDecoderConfig& TokenDecoderConfig::parse_json(const json& json_config) {
  type = json_config.at("type");
  if (type == "ByteLevel") {
    // No parameters to parse
  } else if (type == "Replace") {
    // Parse pattern and content for Replace decoder
    if (json_config.contains("pattern") && json_config.contains("content")) {
      if (json_config["pattern"].contains("String")) {
        replace_pattern = json_config["pattern"]["String"];
      }
      replace_content = json_config["content"];
    }
  } else if (type == "ByteFallback") {
    // No parameters to parse
  } else if (type == "Fuse") {
    // No parameters to parse
  } else if (type == "Sequence") {
    // Parse decoders array for Sequence decoder
    if (json_config.contains("decoders")) {
      sequence_decoders = json_config["decoders"];
    }
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
static std::string format(const char* fmt, ...) {
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

std::string ByteLevelTokenDecoder::decode(const std::string& token) const {
  // This is borrowed and lightly tweaked from llama.cpp
  // CITE:
  // https://github.com/ggerganov/llama.cpp/blob/master/src/llama-vocab.cpp#L1755
  std::string decoded_text;
  // TODO: This could be more efficient since what we really need is a string
  //  const ref.
  const auto cpts = unicode_cpts_from_utf8(token);
  for (const auto cpt : cpts) {
    const auto utf8 = unicode_cpt_to_utf8(cpt);
    try {
      decoded_text += unicode_utf8_to_byte(utf8);
    } catch (const std::out_of_range& /*e*/) {
      decoded_text += "[UNK_BYTE_0x";
      for (const auto c : utf8) {
        decoded_text += format("%02x", (uint8_t)c);
      }
      decoded_text += token + "]";
    }
  }

  return decoded_text;
}

// ReplaceTokenDecoder ////////////////////////////////////////////////////////

ReplaceTokenDecoder::ReplaceTokenDecoder(
    const std::string& pattern,
    const std::string& content)
    : pattern_(pattern), content_(content) {}

std::string ReplaceTokenDecoder::decode(const std::string& token) const {
  // Guard against empty pattern to prevent infinite loop
  if (pattern_.empty()) {
    return token;
  }

  std::string result = token;
  size_t pos = 0;
  while ((pos = result.find(pattern_, pos)) != std::string::npos) {
    result.replace(pos, pattern_.length(), content_);
    pos += content_.length();
  }
  return result;
}

// ByteFallbackTokenDecoder ///////////////////////////////////////////////////

std::string ByteFallbackTokenDecoder::decode(const std::string& token) const {
  // ByteFallback handles tokens that represent individual bytes
  // For tokens that start with <0x and end with >, extract the hex value
  if (token.length() >= 5 && token.substr(0, 3) == "<0x" &&
      token.back() == '>') {
    std::string hex_str = token.substr(3, token.length() - 4);
    try {
      unsigned long byte_val = std::stoul(hex_str, nullptr, 16);
      if (byte_val <= 255) {
        return std::string(1, static_cast<char>(byte_val));
      }
    } catch (const std::exception&) {
      // Fall through to return original token
    }
  }
  return token;
}

// FuseTokenDecoder ///////////////////////////////////////////////////////////

std::string FuseTokenDecoder::decode(const std::string& token) const {
  // Fuse decoder typically just returns the token as-is
  // The actual "fusing" happens at a higher level when multiple tokens are
  // combined
  return token;
}

// SequenceTokenDecoder ///////////////////////////////////////////////////////

SequenceTokenDecoder::SequenceTokenDecoder(
    std::vector<TokenDecoder::Ptr> decoders)
    : decoders_(std::move(decoders)) {}

std::string SequenceTokenDecoder::decode(const std::string& token) const {
  std::string result = token;
  for (const auto& decoder : decoders_) {
    result = decoder->decode(result);
  }
  return result;
}

} // end  namespace tokenizers
