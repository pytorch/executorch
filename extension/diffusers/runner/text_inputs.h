/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <executorch/runtime/core/result.h>

#include <pytorch/tokenizers/tokenizer.h>

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

namespace executorch::extension::diffusers {

/** Inputs of one exported text encoder. */
struct TextEncoderInputs {
  std::vector<int64_t> input_ids;
  std::vector<int64_t> attention_mask;
};

/**
 * Tokenizes prompts for one exported text encoder, as Hugging Face
 * tokenizer(prompt, padding="max_length", max_length=..., truncation=True).
 */
class PromptTokenizer {
 public:
  /**
   * Loads the tokenizer of a Hugging Face snapshot slot directory, such as
   * <snapshot>/tokenizer_2: spiece.model when present, else tokenizer.json.
   */
  static runtime::Result<std::unique_ptr<PromptTokenizer>> load(
      const std::string& directory,
      int64_t max_length);

  /** Pads or truncates to max_length, keeping the special tokens. */
  runtime::Result<TextEncoderInputs> encode(const std::string& text) const;

 private:
  PromptTokenizer(
      std::unique_ptr<::tokenizers::Tokenizer> tokenizer,
      bool sentencepiece,
      bool add_bos,
      uint64_t pad_id,
      int64_t max_length,
      bool pad_left);

  runtime::Result<std::vector<uint64_t>> tokenize(
      const std::string& text,
      bool add_special_tokens) const;

  std::unique_ptr<::tokenizers::Tokenizer> tokenizer_;
  bool sentencepiece_;
  bool add_bos_;
  uint64_t pad_id_;
  int64_t max_length_;
  bool pad_left_;
};

} // namespace executorch::extension::diffusers
