/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <functional>
#include <memory>
#include <optional>
#include <string>
#include <unordered_set>
#include <vector>

#include <executorch/extension/llm/runner/llm_runner_helper.h>
#include <executorch/extension/llm/runner/stats.h>
#include <executorch/extension/llm/sampler/sampler.h>
#include <executorch/extension/module/module.h>
#include <executorch/extension/tensor/tensor_ptr.h>
#include <executorch/runtime/core/error.h>
#include <executorch/runtime/core/result.h>
#include <pytorch/tokenizers/tokenizer.h>

namespace executorch::extension::asr {

using ::executorch::extension::Module;
using ::executorch::extension::llm::get_eos_ids;
using ::executorch::extension::llm::load_tokenizer;
using ::executorch::extension::llm::print_report;
using ::executorch::extension::llm::Sampler;
using ::executorch::extension::llm::Stats;
using ::executorch::runtime::Error;
using ::executorch::runtime::Result;

using TokenCallback = std::function<void(const std::string&)>;

/**
 * Configuration for the ASR transcription loop.
 *
 * max_new_tokens controls the number of tokens generated after the prompt.
 * Temperature controls the randomness of the output.
 */
struct ET_EXPERIMENTAL AsrTranscribeConfig {
  int64_t max_new_tokens = 128;
  std::unordered_set<int64_t> eos_token_ids = {};
  float temperature = 0.0f;
  int64_t decoder_start_token_id = 0;
};

/**
 * Runner that owns a ASR model encoder + decoder pair exported as a single
 * ExecuTorch module. A good example is Whisper
 * (https://huggingface.co/openai/whisper-small)
 *
 * The module is expected to expose two callable methods:
 *  - "encoder": processes precomputed audio features into encoder states.
 *  - "text_decoder": consumes the decoder input ids, encoder output and cache
 *    positions to autoregressively generate logits.
 */
class ET_EXPERIMENTAL AsrRunner {
 public:
  AsrRunner(
      const std::string& module_path,
      std::optional<std::string> data_path,
      const std::string& tokenizer_path);

  /**
   * Returns true when the module and tokenizer are ready for inference.
   */
  bool is_loaded() const;

  /**
   * Loads the module, validates required methods and initialises tokenizer.
   */
  ::executorch::runtime::Error load();

  /**
   * Executes an end-to-end transcription cycle.
   *
   * @param preprocessed_features Audio features tensor of shape [batch, time,
   * features] already processed by a preprocessor module. Typically produced
   * by an audio feature extractor (e.g., mel-spectrogram computation).
   * @param config Controls generation length and termination criteria.
   * @param token_callback Optional functor invoked for each decoded piece of
   * text emitted during generation.
   *
   * @returns Result containing the final transcribed text, or an error.
   */
  ::executorch::runtime::Result<std::string> transcribe(
      ::executorch::extension::TensorPtr preprocessed_features,
      AsrTranscribeConfig config = {},
      std::optional<TokenCallback> token_callback = std::nullopt);

 private:
  ::executorch::runtime::Error load_tokenizer();
  inline const std::unordered_set<int64_t>& eos_token_ids() const {
    return eos_token_ids_;
  }

  std::string module_path_;
  std::string data_path_;
  std::string tokenizer_path_;

  std::unique_ptr<Module> module_;
  std::unique_ptr<::tokenizers::Tokenizer> tokenizer_;
  std::unordered_set<int64_t> eos_token_ids_;

  bool encoder_method_loaded_ = false;
  bool decoder_method_loaded_ = false;
  bool sampler_method_loaded_ = false;
  bool sampler_method_present_ = false;

  Stats stats_;
};

} // namespace executorch::extension::asr
