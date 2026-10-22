/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <executorch/extension/llm/batching/types.h>
#include <executorch/extension/llm/runner/text_stream.h>
#include <executorch/extension/llm/runner/util.h>

#include <algorithm>
#include <cstddef>
#include <optional>
#include <utility>

namespace executorch::extension::llm::serving::detail {

// Per-request rendering state. The caller keeps the complete batching history
// separately: string stops can hide part of a token or a speculative update.
class TextOutput {
 public:
  TextOutput(
      const tokenizers::Tokenizer& tokenizer,
      batching::Token previous,
      std::vector<batching::Token> stop_tokens,
      std::vector<std::string> stop_strings,
      TextStream::Sink sink)
      : stop_tokens_(std::move(stop_tokens)),
        stop_strings_(std::move(stop_strings)),
        sink_(std::move(sink)),
        stream_(
            tokenizer,
            [this](const std::string& piece) { accept(piece); },
            previous) {}

  TextOutput(const TextOutput&) = delete;
  TextOutput& operator=(const TextOutput&) = delete;
  TextOutput(TextOutput&&) = delete;
  TextOutput& operator=(TextOutput&&) = delete;

  runtime::Error append(const std::vector<batching::Token>& tokens) {
    if (finished_) {
      return runtime::Error::InvalidState;
    }
    if (error_ != runtime::Error::Ok) {
      return error_;
    }
    for (const auto token : tokens) {
      if (stopped()) {
        break;
      }
      if (std::find(stop_tokens_.begin(), stop_tokens_.end(), token) !=
          stop_tokens_.end()) {
        token_stop_ = true;
        break;
      }
      error_ = stream_.append(token);
      if (error_ != runtime::Error::Ok) {
        return error_;
      }
      generated_.push_back(token);
    }
    return runtime::Error::Ok;
  }

  void finish() {
    if (finished_) {
      return;
    }
    finished_ = true;
    if (error_ != runtime::Error::Ok || string_stop_) {
      return;
    }
    stream_.flush();
    if (!string_stop_ && !pending_.empty()) {
      if (sink_) {
        sink_(pending_);
      }
      pending_.clear();
    }
  }

  bool stopped() const {
    return token_stop_ || string_stop_;
  }

  bool string_stopped() const {
    return string_stop_;
  }

  std::size_t completion_tokens() const {
    return generated_.size();
  }

  // Cancellation additionally invalidates replay at the serving layer.
  std::optional<std::vector<batching::Token>> generated_token_ids() const {
    if (!finished_ || string_stop_ || error_ != runtime::Error::Ok) {
      return std::nullopt;
    }
    return generated_;
  }

 private:
  void accept(const std::string& piece) {
    if (string_stop_) {
      return;
    }
    pending_ += piece;
    const auto safe =
        stop_safe_prefix_len(pending_, stop_strings_, string_stop_);
    if (safe != 0 && sink_) {
      sink_(pending_.substr(0, safe));
    }
    pending_.erase(0, safe);
    if (string_stop_) {
      pending_.clear();
    }
  }

  const std::vector<batching::Token> stop_tokens_;
  const std::vector<std::string> stop_strings_;
  TextStream::Sink sink_;
  TextStream stream_;
  std::string pending_;
  std::vector<batching::Token> generated_;
  runtime::Error error_ = runtime::Error::Ok;
  bool token_stop_ = false;
  bool string_stop_ = false;
  bool finished_ = false;
};

} // namespace executorch::extension::llm::serving::detail
