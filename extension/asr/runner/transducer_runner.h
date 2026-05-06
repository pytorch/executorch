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
 * A decoded token with frame-level timing information.
 */
struct Token {
  uint64_t id;
  int64_t start_offset; // Frame index in the encoder output
  // For TDT: predicted duration (number of encoder frames consumed by this
  // token). For standard RNN-T: always 1.
  int64_t duration;
};

/**
 * Runtime configuration for the Transducer decode loop.
 *
 * Model-intrinsic parameters (blank_id, num_rnn_layers, pred_hidden) are
 * auto-detected from the model's constant_methods at load() time and do not
 * need to be specified here.
 */
struct ET_EXPERIMENTAL TransducerConfig {
  int64_t max_symbols_per_step = 10;
  // TDT duration values. Empty means standard RNN-T (duration always 1).
  std::vector<int> durations = {};
};

/**
 * Runner for Transducer-based ASR models (RNN-T, TDT, HAT).
 *
 * Transducer models use a fundamentally different decode paradigm from
 * Seq2Seq: frame-by-frame scanning of encoder output with a joint network,
 * emitting tokens conditionally at each frame.
 *
 * Required module methods:
 *  - "encoder":      (mel, mel_len) -> (f_proj, encoded_len)
 *  - "decoder_step": (token, h, c) -> (g_proj, h', c')
 *  - "joint":        (f_t, g_proj) -> (token_id, duration_idx)
 *
 * Optional module methods:
 *  - "preprocessor": (audio, audio_len) -> (mel, mel_len)
 *
 * Auto-detected constant_methods (read at load() time):
 *  - "blank_id", "num_rnn_layers", "pred_hidden"
 */
class ET_EXPERIMENTAL TransducerRunner {
 public:
  TransducerRunner(
      const std::string& module_path,
      const std::string& tokenizer_path,
      TransducerConfig config = {},
      std::optional<std::string> data_path = std::nullopt);

  /**
   * Returns true when the module and tokenizer are ready for inference.
   */
  bool is_loaded() const;

  /**
   * Loads the module, validates required methods, reads model metadata
   * from constant_methods, and initialises tokenizer.
   */
  Error load();

  /**
   * Runs the model's preprocessor method on raw audio.
   *
   * @param raw_audio 1-D tensor of float audio samples.
   * @returns Preprocessed features tensor (e.g., mel spectrogram),
   *   ready to pass to transcribe().
   */
  Result<::executorch::extension::TensorPtr> preprocess(
      ::executorch::extension::TensorPtr raw_audio);

  /**
   * Runs transducer greedy decode on pre-encoded features.
   *
   * @param preprocessed_features Encoder input tensor (e.g., mel spectrogram)
   *   of shape [batch, time, features]. The encoder will be called internally.
   * @param token_callback Optional functor invoked for each decoded piece of
   *   text emitted during generation.
   *
   * @returns Result containing decoded tokens with frame offsets, or an error.
   */
  Result<std::vector<Token>> transcribe(
      ::executorch::extension::TensorPtr preprocessed_features,
      std::function<void(const std::string&)> token_callback = {});

  /**
   * Returns a reference to the loaded tokenizer, or nullptr if not loaded.
   */
  const ::tokenizers::Tokenizer* tokenizer() const {
    return tokenizer_.get();
  }

 private:
  Error load_tokenizer();
  Error load_model_metadata();

  std::string module_path_;
  std::string data_path_;
  std::string tokenizer_path_;
  TransducerConfig config_;

  // Model metadata (populated by load_model_metadata)
  int64_t blank_id_ = 0;
  int64_t num_rnn_layers_ = 2;
  int64_t pred_hidden_ = 640;

  std::unique_ptr<Module> module_;
  std::unique_ptr<::tokenizers::Tokenizer> tokenizer_;

  bool encoder_method_loaded_ = false;
  bool decoder_method_loaded_ = false;
  bool joint_method_loaded_ = false;
  bool preprocessor_method_loaded_ = false;
  bool preprocessor_method_present_ = false;

  Stats stats_;
};

} // namespace executorch::extension::asr
