/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 * Copyright 2026 Arm Limited and/or its affiliates.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

// Given inputs, run a text decoder and return logits.

#include <executorch/extension/llm/runner/text_decoder_runner.h>
#include <executorch/kernels/portable/cpu/util/arange_util.h>

#include <algorithm>
#include <ctime>
#include <limits>

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
    std::string method_name,
    std::unique_ptr<Sampler> sampler,
    Stats* stats,
    bool enable_dynamic_shape)
    : module_(module),
      io_manager_(io_manager),
      method_name_(std::move(method_name)),
      sampler_(std::move(sampler)),
      stats_(stats),
      enable_dynamic_shape_(enable_dynamic_shape) {}

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
        module_,
        start_pos,
        cache_positions,
        tokens->numel(),
        method_name_.c_str());
    if (!start_pos_tensor_result.ok()) {
      return start_pos_tensor_result.error();
    }
    auto start_pos_tensor = std::move(*start_pos_tensor_result);

    std::vector<runtime::EValue> inputs;
    auto inputs_res =
        io_manager_->prepare_decode(tokens, start_pos_tensor, method_name_);
    ET_CHECK_OK_OR_RETURN_ERROR(inputs_res.error());
    inputs = inputs_res.get();
    if (stats_ != nullptr) {
      stats_->on_model_execution_begin();
    }
    auto outputs_res = module_->execute(method_name_, inputs);
    if (stats_ != nullptr) {
      stats_->on_model_execution_end();
    }
    ET_CHECK_OK_OR_RETURN_ERROR(outputs_res.error());

    auto update_err =
        io_manager_->update_decode(outputs_res.get(), method_name_);
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

    auto input_meta = method_meta.input_tensor_meta(0);
    ET_CHECK_OK_OR_RETURN_ERROR(input_meta.error());
    const auto sizes = input_meta->sizes();
    ET_CHECK_OR_RETURN_ERROR(
        sizes.size() == 2 && sizes[0] == 1 && tokens->dim() == 2 &&
            tokens->size(0) == 1,
        InvalidArgument,
        "Non-KV text input must have shape [1, sequence_length]");
    const auto valid_length = tokens->size(1);
    const auto input_length = static_cast<aten::SizesType>(
        enable_dynamic_shape_ ? valid_length : sizes[1]);
    ET_CHECK_OR_RETURN_ERROR(
        valid_length > 0 && valid_length <= input_length,
        InvalidArgument,
        "Token count exceeds the static non-KV input window");
    ET_CHECK_OR_RETURN_ERROR(
        tokens->scalar_type() == aten::ScalarType::Long &&
            (input_meta->scalar_type() == aten::ScalarType::Int ||
             input_meta->scalar_type() == aten::ScalarType::Long),
        InvalidArgument,
        "Non-KV text input must use int32 or int64 tokens");
    TensorPtr model_tokens;
    if (input_meta->scalar_type() == aten::ScalarType::Int) {
      std::vector<int32_t> padded(input_length, 0);
      const auto* data = tokens->const_data_ptr<int64_t>();
      for (int64_t i = 0; i < valid_length; ++i) {
        ET_CHECK_OR_RETURN_ERROR(
            data[i] >= 0 && data[i] <= std::numeric_limits<int32_t>::max(),
            InvalidArgument,
            "Token ID cannot be represented as int32");
        padded[i] = static_cast<int32_t>(data[i]);
      }
      model_tokens = make_tensor_ptr({1, input_length}, std::move(padded));
    } else {
      std::vector<int64_t> padded(input_length, 0);
      const auto* data = tokens->const_data_ptr<int64_t>();
      std::copy(data, data + valid_length, padded.begin());
      model_tokens = make_tensor_ptr({1, input_length}, std::move(padded));
    }
    std::vector<runtime::EValue> inputs{model_tokens};
    if (stats_ != nullptr) {
      stats_->on_model_execution_begin();
    }
    auto outputs_res = module_->execute(method_name_, inputs);
    if (stats_ != nullptr) {
      stats_->on_model_execution_end();
    }
    ET_CHECK_OK_OR_RETURN_ERROR(outputs_res.error());
    ET_CHECK_MSG(
        outputs_res.get().size() == 1,
        "More than one output returned from executing LLM.");
    ET_CHECK_MSG(
        outputs_res.get()[0].isTensor(),
        "Non Tensor Output returned from executing LLM");

    auto logits = outputs_res.get()[0].toTensor();
    if (!enable_dynamic_shape_) {
      ET_CHECK_OR_RETURN_ERROR(
          logits.dim() == 3 && logits.size(0) == 1 &&
              logits.size(1) == input_length,
          InvalidProgram,
          "Static non-KV models need full logits; export with "
          "--generate_full_logits");
      const auto vocab_size = static_cast<aten::SizesType>(logits.size(2));
      auto* last_row = static_cast<char*>(logits.mutable_data_ptr()) +
          (valid_length - 1) * vocab_size * logits.element_size();
      last_logits_ = from_blob(last_row, {1, vocab_size}, logits.scalar_type());
      return *last_logits_;
    }
    return logits;
  }
}

} // namespace llm
} // namespace extension
} // namespace executorch
