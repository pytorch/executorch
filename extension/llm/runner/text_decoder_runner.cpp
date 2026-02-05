/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

// Given inputs, run a text decoder and return logits.

#include <executorch/extension/llm/runner/text_decoder_runner.h>
#include <executorch/kernels/portable/cpu/util/arange_util.h>

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
    IOManager* io_manager,
    std::string method_name)
    : module_(module),
      io_manager_(io_manager),
      method_name_(std::move(method_name)) {}

// This function is functional, meaning it shouldn't modify any state of the
// input. It should be safe to call multiple times with the same inputs. The
// outer loop (call site) is responsible for managing state.
::executorch::runtime::Result<executorch::aten::Tensor> TextDecoderRunner::step(
    TensorPtr& tokens,
    int64_t start_pos) {
  // ET_LOG(Info, "Input token %" PRIu64, input_token);
  auto method_meta_result = module_->method_meta(method_name_);
  if (!method_meta_result.ok()) {
    return method_meta_result.error();
  }
  auto method_meta = std::move(*method_meta_result);
  // If only 1 input, we are not using kv cache
  bool use_kv_cache = method_meta.num_inputs() > 1;

  std::vector<int64_t> cache_positions;

  if (use_kv_cache) {
    auto start_pos_tensor_result = populate_start_pos_or_cache_position(
        module_, start_pos, cache_positions, tokens->numel(), method_name_.c_str());
    if (!start_pos_tensor_result.ok()) {
      return start_pos_tensor_result.error();
    }
    auto start_pos_tensor = std::move(*start_pos_tensor_result);

    std::vector<runtime::EValue> inputs;
    auto inputs_res =
        io_manager_->prepare_decode(tokens, start_pos_tensor, method_name_);
    ET_CHECK_OK_OR_RETURN_ERROR(inputs_res.error());
    inputs = inputs_res.get();
    auto outputs_res = module_->execute(method_name_, inputs);
    ET_CHECK_OK_OR_RETURN_ERROR(outputs_res.error());

    auto update_err = io_manager_->update_decode(outputs_res.get(), method_name_);
    ET_CHECK_OK_OR_RETURN_ERROR(update_err);

    ET_CHECK_MSG(
        outputs_res.get().size() == 1,
        "More than one output returned from executing LLM.");
    ET_CHECK_MSG(
        outputs_res.get()[0].isTensor(),
        "Non Tensor Output returned from executing LLM");

    // Return the logits tensor
    return outputs_res.get()[0].toTensor();
  } else { // no kv cache
    (void)start_pos; // unused

    std::vector<runtime::EValue> inputs{tokens};
    auto outputs_res = module_->execute(method_name_, inputs);
    ET_CHECK_OK_OR_RETURN_ERROR(outputs_res.error());
    ET_CHECK_MSG(
        outputs_res.get().size() == 1,
        "More than one output returned from executing LLM.");
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
