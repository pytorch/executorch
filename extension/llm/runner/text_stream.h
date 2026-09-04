/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

// Assembles a stream of token ids into text a caller can hand straight to a
// UI, a socket, or a JSON encoder.
//
// A byte-level tokenizer can emit a token that is only part of a character, so
// decoding each token on its own yields pieces that are not valid UTF-8 even
// though their concatenation is. Printing to a terminal survives that;
// anything that treats a piece as a standalone string does not. Holding the
// incomplete tail back until it completes is the state this exists to own,
// because a character split across tokens outlives any one delivery of them.
//
// Stop strings are a separate stage. Assemble here, then filter the text with
// stop_safe_prefix_len before it reaches the sink.

#include <cstdint>
#include <functional>
#include <string>
#include <utility>
#include <vector>

#include <executorch/extension/llm/runner/util.h>
#include <executorch/runtime/core/error.h>
#include <pytorch/tokenizers/tokenizer.h>

namespace executorch {
namespace extension {
namespace llm {

class ET_EXPERIMENTAL TextStream {
 public:
  // Never called with an empty string.
  using Sink = std::function<void(const std::string&)>;

  // The tokenizer is borrowed and must outlive the stream.
  //
  // `previous` is the token the next one follows, normally the last token of
  // the prompt. Only a SentencePiece tokenizer reads it, to drop the leading
  // space of the first token after BOS; every BPETokenizerBase discards it.
  // The default is for BPE, which cannot observe the value. A SentencePiece
  // caller must pass the real previous token: 0 is a valid id there, so
  // leaving it defaulted silently mis-handles the space on the first token.
  TextStream(
      const tokenizers::Tokenizer& tokenizer,
      Sink on_text,
      uint64_t previous = 0)
      : tokenizer_(tokenizer),
        on_text_(std::move(on_text)),
        previous_(previous) {}

  // Emits every character the token completed. A token that only extends an
  // unfinished character emits nothing and is not lost.
  //
  // On a tokenizer error the stream stops emitting and stays failed, rather
  // than leaving the caller to guess which tokens reached the sink.
  ::executorch::runtime::Error append(uint64_t token) {
    if (failed_) {
      return ::executorch::runtime::Error::InvalidState;
    }
    const tokenizers::Result<std::string> piece =
        tokenizer_.decode(previous_, token);
    if (!piece.ok()) {
      failed_ = true;
      return ::executorch::runtime::Error::InvalidArgument;
    }
    previous_ = token;
    pending_ += *piece;
    emit_(utf8_complete_prefix_len(pending_));
    return ::executorch::runtime::Error::Ok;
  }

  // Stops at the first token that fails.
  ::executorch::runtime::Error append(const std::vector<uint64_t>& tokens) {
    for (uint64_t token : tokens) {
      const ::executorch::runtime::Error error = append(token);
      if (error != ::executorch::runtime::Error::Ok) {
        return error;
      }
    }
    return ::executorch::runtime::Error::Ok;
  }

  // Emits whatever is held back, including a trailing character the tokens
  // never finished. Call it once the generation has ended, or those bytes are
  // dropped. Idempotent.
  void flush() {
    emit_(pending_.size());
  }

  // A character whose bytes have not all arrived yet.
  bool has_pending() const {
    return !pending_.empty();
  }

  bool failed() const {
    return failed_;
  }

 private:
  void emit_(size_t length) {
    if (length == 0) {
      return;
    }
    std::string ready = pending_.substr(0, length);
    pending_.erase(0, length);
    if (on_text_) {
      on_text_(ready);
    }
  }

  const tokenizers::Tokenizer& tokenizer_;
  Sink on_text_;
  uint64_t previous_;
  std::string pending_;
  bool failed_ = false;
};

} // namespace llm
} // namespace extension
} // namespace executorch
