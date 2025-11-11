/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

// Generic encoder prefiller that handles multimodal inputs (image and audio)
// to prefill the KV cache of a multimodal LLM.

#pragma once

#include <executorch/extension/llm/runner/multimodal_decoder_runner.h>
#include <executorch/extension/llm/runner/multimodal_input.h>
#include <executorch/extension/llm/runner/text_decoder_runner.h>
#include <executorch/extension/llm/sampler/sampler.h>
#include <executorch/extension/module/module.h>
#include <executorch/runtime/platform/compiler.h>
#include <pytorch/tokenizers/tokenizer.h>

namespace executorch::extension::llm {

using runtime::Error;
using runtime::Result;
using tokenizers::Tokenizer;

// Assuming kv cache and parallel prefill are enabled.
// This prefiller supports both image and audio inputs
class ET_EXPERIMENTAL MultimodalPrefiller {
 public:
  explicit MultimodalPrefiller(
      Module* module,
      MultimodalDecoderRunner* decoder_runner,
      Tokenizer* tokenizer,
      IOManager* io_manager);

  /**
   * Prefill an LLM Module with the given multimodal input.
   * @param input The multimodal input (image or audio) to the multimodal LLM.
   * @param start_pos The starting position in KV cache of the input in the LLM.
   * It's passed as reference and will be updated inside this function.
   * @return The next token of the LLM Module after prefill.
   */
  virtual Result<uint64_t> prefill(
      const MultimodalInput& input,
      int64_t& start_pos);

  virtual Error load();
  virtual bool is_method_loaded();

  virtual ~MultimodalPrefiller() = default;

 protected:
  Module* module_;
  MultimodalDecoderRunner* text_decoder_runner_;
  Tokenizer* tokenizer_;
  IOManager* io_manager_;
};

} // namespace executorch::extension::llm
