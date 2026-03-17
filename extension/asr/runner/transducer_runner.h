/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <cstdint>
#include <functional>
#include <memory>
#include <optional>
#include <string>
#include <vector>

#include <executorch/extension/llm/runner/stats.h>
#include <executorch/extension/module/module.h>
#include <executorch/extension/tensor/tensor_ptr.h>
#include <executorch/runtime/core/error.h>
#include <executorch/runtime/core/result.h>
#include <pytorch/tokenizers/tokenizer.h>

namespace executorch::extension::asr {

using ::executorch::extension::Module;
using ::executorch::extension::llm::Stats;
using ::executorch::runtime::Error;
using ::executorch::runtime::Result;

/**
 * A decoded token produced by transducer greedy decoding, carrying the
 * frame offset at which the token was emitted and its TDT duration.
 */
struct ET_EXPERIMENTAL TransducerToken {
  uint64_t id;
  int64_t start_offset;
  int64_t duration;
};

/**
 * Configuration for the transducer (RNN-T / TDT) decode loop.
 */
struct ET_EXPERIMENTAL TransducerConfig {
  int64_t blank_id = 0;
  int64_t num_rnn_layers = 2;
  int64_t pred_hidden = 640;
  int64_t max_symbols_per_step = 10;
  // TDT duration values. Empty means standard RNN-T (duration always 1).
  std::vector<int> durations = {};
};

/**
 * Runner for transducer-based ASR models (RNN-T, TDT, HAT).
 *
 * The module is expected to expose the following callable methods:
 *  - "encoder":      processes audio features into projected encoder states.
 *  - "decoder_step": LSTM predictor step — takes (token, h, c) and returns
 *                    (g_proj, new_h, new_c).
 *  - "joint":        joint network — takes (f_t, g_proj) and returns
 *                    (token_idx, duration_idx).
 *
 * An optional "preprocessor" method is supported for models that bundle
 * audio preprocessing inside the same .pte file.
 */
class ET_EXPERIMENTAL TransducerRunner {
 public:
  TransducerRunner(
      const std::string& module_path,
      const std::string& tokenizer_path,
      TransducerConfig config,
      std::optional<std::string> data_path = std::nullopt);

  bool is_loaded() const;

  Error load();

  /**
   * Runs greedy transducer decode on already-projected encoder output.
   *
   * @param encoder_output Projected encoder tensor of shape [1, T,
   * joint_hidden].
   * @param encoder_len    Number of valid time frames in encoder_output.
   * @param token_callback Optional functor invoked for each decoded text piece.
   * @returns Decoded tokens with frame offsets, or an error.
   */
  Result<std::vector<TransducerToken>> transcribe(
      const ::executorch::aten::Tensor& encoder_output,
      int64_t encoder_len,
      std::function<void(const std::string&)> token_callback = {});

  /**
   * Provides access to the underlying module for running additional methods
   * (e.g. "preprocessor", "encoder") that are model-specific.
   */
  Module& module() {
    return *module_;
  }

  const Stats& get_stats() const {
    return stats_;
  }

 private:
  std::string module_path_;
  std::string tokenizer_path_;
  TransducerConfig config_;

  std::unique_ptr<Module> module_;
  std::unique_ptr<::tokenizers::Tokenizer> tokenizer_;

  bool decoder_step_method_loaded_ = false;
  bool joint_method_loaded_ = false;

  Stats stats_;
};

} // namespace executorch::extension::asr
