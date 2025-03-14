/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */
// @lint-ignore-every LICENSELINT

#pragma once

// Standard
#include <memory>
#include <optional>
#include <string>
#include <vector>

// Third Party
#include <nlohmann/json.hpp>
#include <re2/re2.h>

namespace tokenizers {

// -- Base ---------------------------------------------------------------------

/**
 * Base class for all pre-tokenizers with a single virtual method to split the
 * input string piece
 */
class PreTokenizer {
public:
  /** Shared pointer type */
  typedef std::shared_ptr<PreTokenizer> Ptr;

  /** Split the input string piece into sub-pieces
   *
   * This pre-tokenization may result in sub-pieces that are not contained
   * within the original input, therefore the resulting pieces will be owned by
   * the caller.
   *
   * NOTE: Pass by value per best practice
   *  https://abseil.io/docs/cpp/guides/strings#string_view
   */
  virtual std::vector<std::string>
  pre_tokenize(re2::StringPiece input) const = 0;

  virtual ~PreTokenizer() = default;
}; // end class PreTokenizer

// -- Factory ------------------------------------------------------------------

// Helper macro to standardize addition of config member fields
#define CONFIG_MEMBER(type, name)                                              \
  std::optional<type> name;                                                    \
  PreTokenizerConfig &set_##name(type arg) {                                   \
    this->name = std::move(arg);                                               \
    return *this;                                                              \
  }

/**
 * Factory and config class for creating a new PreTokenizer
 *
 * This class is the central method for instantiating a PreTokenizer instance.
 * It contains the common construction logic and config parameter names for all
 * pre tokenizer constructor args.
 *
 * NOTE: When adding a new pre tokenizer, you must ensure its arguments are
 *  added to this class and it's constructor is added in the implementation!
 *
 * Usage Example:
 *
 * const auto pre_tokenizer = PreTokenizerConfig("Sequence").set_pretokenizers(
 *   {PreTokenizerConfig("Digits"), PreTokenizerConfig("ByteLevel")}
 * );
 * const auto pre_tokenized = pre_tokenizer->pre_tokenize("Hello World!");
 */
class PreTokenizerConfig {
public:
  /*------------------------*/
  /* Public mutable members */
  /*------------------------*/

  /**
   * The Type name string matching from tokenizers
   * https://github.com/huggingface/tokenizers/blob/main/tokenizers/src/pre_tokenizers/mod.rs#L73
   */
  std::string type;

  /**
   * Used by: RegexPreTokenizer, ByteLevelPreTokenizer
   */
  CONFIG_MEMBER(std::string, pattern)

  /**
   * Used by: DigitsPreTokenizer
   */
  CONFIG_MEMBER(bool, individual_digits)

  /**
   * Used by: ByteLevelPreTokenizer
   */
  CONFIG_MEMBER(bool, add_prefix_space)

  /**
   * Used by: SequencePreTokenizer
   */
  CONFIG_MEMBER(std::vector<PreTokenizerConfig>, pretokenizers)

  /*----------------*/
  /* Public methods */
  /*----------------*/

  /**
   * Construct with the type
   */
  explicit PreTokenizerConfig(std::string type = "");

  /**
   * Construct the pre tokenizer instance from the member data
   */
  PreTokenizer::Ptr create() const;

  /**
   * Populate from a json config file
   */
  PreTokenizerConfig &parse_json(const nlohmann::json &json_config);

}; // end class PreTokenizerConfig

// -- Regex --------------------------------------------------------------------
// Used for general-purpose single-regex pre tokenization
// CITE:
// https://github.com/huggingface/tokenizers/blob/main/tokenizers/src/pre_tokenizers/split.rs
//
// TODO: Support for "behavior" and "invert" options
//  https://github.com/huggingface/tokenizers/blob/main/tokenizers/src/tokenizer/normalizer.rs#L82
//  https://github.com/huggingface/tokenizers/blob/main/tokenizers/src/tokenizer/pattern.rs#L128

class RegexPreTokenizer : public PreTokenizer {
public:
  typedef std::unique_ptr<re2::RE2> Re2UPtr;

  explicit RegexPreTokenizer(const std::string &pattern)
      : regex_(RegexPreTokenizer::create_regex_(pattern)) {}

  /** Pre-tokenize with the stored regex */
  std::vector<std::string> pre_tokenize(re2::StringPiece input) const;

protected:
  static Re2UPtr create_regex_(const std::string &pattern);

  Re2UPtr regex_;

}; // end class RegexPreTokenizer

// -- Digits -------------------------------------------------------------------
// Used by tokenizers
// CITE:
// https://github.com/huggingface/tokenizers/blob/main/tokenizers/src/pre_tokenizers/digits.rs

class DigitsPreTokenizer : public RegexPreTokenizer {
public:
  explicit DigitsPreTokenizer(bool individual_digits = false)
      : RegexPreTokenizer(individual_digits ? R"([^\p{N}]+|\p{N})"
                                            : R"([^\p{N}]+|[\p{N}]+)") {}
}; // end class DigitsPreTokenizer

// -- ByteLevel ----------------------------------------------------------------
// Used by tokenizers
// CITE:
// https://github.com/huggingface/tokenizers/blob/main/tokenizers/src/pre_tokenizers/byte_level.rs

class ByteLevelPreTokenizer : public PreTokenizer {
public:
  /**
   * @param add_prefix_space: Whether to add a leading space to the first word
   * @param pattern: A user-supplied regex to use for token splitting. If not
   *    provided, it use the standard GPT2 pattern.
   */
  ByteLevelPreTokenizer(bool add_prefix_space = true,
                        const std::string &pattern = "");
  explicit ByteLevelPreTokenizer(const std::string &pattern)
      : ByteLevelPreTokenizer(true, pattern) {}

  /** Perform pre-tokenization */
  std::vector<std::string> pre_tokenize(re2::StringPiece input) const override;

private:
  const std::string pattern_;
  const bool add_prefix_space_;

}; // end class ByteLevelPreTokenizer

// -- Sequence -----------------------------------------------------------------
// Used by tokenizers
// CITE:
// https://github.com/huggingface/tokenizers/blob/main/tokenizers/src/pre_tokenizers/sequence.rs

class SequencePreTokenizer : public PreTokenizer {
public:
  /**
   * @param pre_tokenizers: The sequence of owned pre-tokenizer objects to use
   */
  explicit SequencePreTokenizer(std::vector<PreTokenizer::Ptr> pre_tokenizers);

  /** Perform pre-tokenization */
  std::vector<std::string> pre_tokenize(re2::StringPiece input) const override;

private:
  const std::vector<PreTokenizer::Ptr> pre_tokenizers_;

}; // end class ByteLevelPreTokenizer

} // namespace tokenizers
