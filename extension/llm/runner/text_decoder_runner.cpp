/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

// Given inputs, run a text decoder and return logits.

#include <executorch/extension/llm/runner/text_decoder_runner.h>

#include <ctime>

#include <executorch/extension/llm/runner/stats.h>

namespace executorch {
namespace extension {
namespace llm {

// NOTE: we observed ~2x loading performance increase on iPhone 15
// and a ~5% improvement on Galaxy S22 by switching to
// FileDataLoader instead of MmapDataLoader + UseMlockIgnoreErrors.
TextDecoderRunner::TextDecoderRunner(
    Module* module,
    bool use_kv_cache,
    int32_t vocab_size,
    float temperature)
    : module_(module),
      sampler_(std::make_unique<Sampler>(
          vocab_size,
          temperature,
          kTopp,
          static_cast<unsigned long long>(std::time(nullptr)))),
      use_kv_cache_(use_kv_cache) {}

// This function is functional, meaning it shouldn't modify any state of the
// input. It should be safe to call multiple times with the same inputs. The
// outer loop (call site) is responsible for managing state.
::executorch::runtime::Result<exec_aten::Tensor> TextDecoderRunner::step(
    TensorPtr& tokens,
    TensorPtr& start_pos) {
  // ET_LOG(Info, "Input token %" PRIu64, input_token);
  if (use_kv_cache_) {
    auto outputs_res = module_->forward({tokens, start_pos});
    ET_CHECK_OK_OR_RETURN_ERROR(outputs_res.error());
    ET_CHECK_MSG(
        outputs_res.get().size() == 1,
        "More then one output returned from executing LLM.");
    ET_CHECK_MSG(
        outputs_res.get()[0].isTensor(),
        "Non Tensor Output returned from executing LLM");

    // Return the logits tensor
    return outputs_res.get()[0].toTensor();
  } else { // no kv cache
    (void)start_pos; // unused

    auto outputs_res = module_->forward(tokens);
    ET_CHECK_OK_OR_RETURN_ERROR(outputs_res.error());
    ET_CHECK_MSG(
        outputs_res.get().size() == 1,
        "More then one output returned from executing LLM.");
    ET_CHECK_MSG(
        outputs_res.get()[0].isTensor(),
        "Non Tensor Output returned from executing LLM");

    // Return the logits tensor
    return outputs_res.get()[0].toTensor();
  }
}

} // namespace llm
} // namespace extension
} // namespace executorch
