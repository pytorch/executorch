/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */
// @lint-ignore-every LICENSELINT

// Local
#include <pytorch/tokenizers/pre_tokenizer.h>
#include <unicode.h>

// Standard
#include <algorithm>
#include <iterator>
#include <utility>

// Third Party
#include <nlohmann/json.hpp>

using json = nlohmann::json;

namespace tokenizers {

// PreTokenizerConfig //////////////////////////////////////////////////////////

PreTokenizerConfig::PreTokenizerConfig(std::string type)
    : type(std::move(type)) {}

PreTokenizer::Ptr PreTokenizerConfig::create() const {
  // NOTE: These types must line up with the type strings found in the
  //  tokenizers library
  //  https://github.com/huggingface/tokenizers/blob/main/tokenizers/src/pre_tokenizers/mod.rs#L73
  if (type == "Split") {
    if (!pattern) {
      throw std::runtime_error(
          "Missing pattern for PreTokenizer of type Split");
    }

    // Validate behavior parameter
    std::string behavior_str = behavior ? *behavior : "";
    if (!behavior_str.empty() && behavior_str != "MergedWithPrevious") {
      throw std::runtime_error(
          "Unsupported behavior '" + behavior_str +
          "' for Split PreTokenizer. Only 'MergedWithPrevious' is supported.");
    }

    // Validate invert parameter
    bool invert_flag = invert ? *invert : false;
    if (invert_flag) {
      throw std::runtime_error(
          "invert=true is not supported for Split PreTokenizer. Only invert=false is supported.");
    }

    return PreTokenizer::Ptr(new RegexPreTokenizer(
        *pattern, is_delimiter ? *is_delimiter : false, behavior_str));
  }
  if (type == "Digits") {
    if (individual_digits) {
      return PreTokenizer::Ptr(new DigitsPreTokenizer(*individual_digits));
    }
    return PreTokenizer::Ptr(new DigitsPreTokenizer());
  }
  if (type == "ByteLevel") {
    if (add_prefix_space && pattern) {
      return PreTokenizer::Ptr(
          new ByteLevelPreTokenizer(*add_prefix_space, *pattern));
    }
    if (add_prefix_space) {
      return PreTokenizer::Ptr(new ByteLevelPreTokenizer(*add_prefix_space));
    }
    if (pattern) {
      return PreTokenizer::Ptr(new ByteLevelPreTokenizer(*pattern));
    }
    return PreTokenizer::Ptr(new ByteLevelPreTokenizer());
  }
  if (type == "Sequence") {
    if (!pretokenizers or pretokenizers->empty()) {
      throw std::runtime_error(
          "Missing pretokenizers for PreTokenizer of type Sequence");
    }
    std::vector<PreTokenizer::Ptr> pretoks;
    std::transform(
        pretokenizers->begin(),
        pretokenizers->end(),
        std::back_inserter(pretoks),
        [](const PreTokenizerConfig& cfg) { return cfg.create(); });
    return PreTokenizer::Ptr(new SequencePreTokenizer(pretoks));
  }
  throw std::runtime_error("Unsupported PreTokenizer type: " + type);
}

PreTokenizerConfig& PreTokenizerConfig::parse_json(const json& json_config) {
  type = json_config.at("type");
  if (type == "Split") {
    try {
      pattern = json_config.at("pattern").at("Regex");
      is_delimiter = false;
    } catch (json::out_of_range&) {
      // "Regex" is not there, check "String", which is a delimiter
      std::string delimiter = json_config.at("pattern").at("String");
      // For string patterns, escape regex special characters to treat them as
      // literal strings (same as Rust's regex::escape)
      pattern = IRegex::escape(delimiter);
      is_delimiter = true;
    }

    // Parse behavior and invert fields
    try {
      behavior = json_config.at("behavior");
    } catch (json::out_of_range&) {
      // behavior is optional, default to empty string
    }

    try {
      invert = json_config.at("invert");
    } catch (json::out_of_range&) {
      // invert is optional, default to false
    }
  } else if (type == "Digits") {
    try {
      individual_digits = json_config.at("individual_digits");
    } catch (json::out_of_range&) {
    }
  } else if (type == "ByteLevel") {
    try {
      add_prefix_space = json_config.at("add_prefix_space");
    } catch (json::out_of_range&) {
    }
    // TODO: trim_offsets, use_regex
  } else if (type == "Sequence") {
    pretokenizers = std::vector<PreTokenizerConfig>();
    for (const auto& entry : json_config.at("pretokenizers")) {
      pretokenizers->push_back(PreTokenizerConfig().parse_json(entry));
    }
  } else {
    throw std::runtime_error("Unsupported PreTokenizer type: " + type);
  }
  return *this;
}

// RegexPreTokenizer ///////////////////////////////////////////////////////////

std::unique_ptr<IRegex> RegexPreTokenizer::create_regex_(
    const std::string& pattern) {
  assert(!pattern.empty());
  return TK_UNWRAP_THROW(create_regex(pattern));
}

std::vector<std::string> RegexPreTokenizer::pre_tokenize(
    const std::string& input) const {
  if (!regex_)
    return {};

  std::vector<std::string> results;
  auto matches = regex_->find_all(input);

  if (!is_delimiter_) {
    // Original behavior: return the matches themselves
    for (const auto& match : matches) {
      results.push_back(input.substr(match.start, match.end - match.start));
    }
  } else {
    // Delimiter behavior
    if (matches.empty()) {
      // No matches found, return the entire input
      results.push_back(input);
      return results;
    }

    if (behavior_ == "MergedWithPrevious") {
      // MergedWithPrevious: Include delimiter with previous token
      // Example: "the-final--countdown" with delimiter "-"
      // -> ["the-", "final-", "-", "countdown"]
      size_t last_end = 0;

      for (size_t i = 0; i < matches.size(); ++i) {
        const auto& match = matches[i];

        // Add text before the match plus the delimiter
        if (match.start > last_end) {
          std::string token = input.substr(last_end, match.end - last_end);
          results.push_back(token);
        } else {
          // Only delimiter, no preceding text
          std::string delimiter =
              input.substr(match.start, match.end - match.start);
          results.push_back(delimiter);
        }

        last_end = match.end;
      }

      // Add remaining text after the last match (if any)
      if (last_end < input.length()) {
        results.push_back(input.substr(last_end));
      }
    } else {
      // Default delimiter behavior (split on delimiters)
      size_t last_end = 0;
      for (const auto& match : matches) {
        // Add text before the match (if any)
        if (match.start > last_end) {
          results.push_back(input.substr(last_end, match.start - last_end));
        }
        last_end = match.end;
      }

      // Add remaining text after the last match (if any)
      if (last_end < input.length()) {
        results.push_back(input.substr(last_end));
      }
    }
  }
  return results;
}

// ByteLevelPreTokenizer ///////////////////////////////////////////////////////

//////////////////
// Impl Details //
//////////////////
namespace {

// Standard GPT2 regex
// https://github.com/openai/gpt-2/blob/master/src/encoder.py#L53
constexpr char GPT2_EXPR[] =
    R"('s|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+)";

} // namespace

//////////////////
// Construction //
//////////////////

ByteLevelPreTokenizer::ByteLevelPreTokenizer(
    bool add_prefix_space,
    const std::string& pattern)
    : pattern_(pattern.empty() ? GPT2_EXPR : pattern),
      add_prefix_space_(add_prefix_space) {}

std::vector<std::string> ByteLevelPreTokenizer::pre_tokenize(
    const std::string& input) const {
  // Add the prefix space if configured to do so.
  std::string formatted_input = input;
  if (add_prefix_space_ && !formatted_input.empty() &&
      formatted_input[0] != ' ') {
    formatted_input.insert(formatted_input.begin(), ' ');
  }

  return unicode_regex_split(formatted_input, {pattern_});
}

// SequencePreTokenizer ////////////////////////////////////////////////////////

SequencePreTokenizer::SequencePreTokenizer(
    std::vector<PreTokenizer::Ptr> pre_tokenizers)
    : pre_tokenizers_(std::move(pre_tokenizers)) {}

std::vector<std::string> SequencePreTokenizer::pre_tokenize(
    const std::string& input) const {
  std::vector<std::string> pieces{std::string(input)};
  for (const auto& pre_tokenizer : pre_tokenizers_) {
    std::vector<std::string> new_pieces;
    for (const auto& piece : pieces) {
      for (const auto& subpiece : pre_tokenizer->pre_tokenize(piece)) {
        new_pieces.push_back(subpiece);
      }
    }
    pieces = std::move(new_pieces);
  }
  return pieces;
}

} // namespace tokenizers
