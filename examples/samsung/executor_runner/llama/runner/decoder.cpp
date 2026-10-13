/*
 * Copyright (c) 2025 Samsung Electronics Co. LTD
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

// Given inputs, run a text decoder and return logits.

#include <executorch/kernels/portable/cpu/util/arange_util.h>

#include <ctime>

#include <executorch/extension/llm/runner/stats.h>
#include <executorch/extension/llm/runner/text_decoder_runner.h>

#include <executorch/examples/samsung/executor_runner/llama/runner/decoder.h>

namespace example {

class IExecutionIOBinder {
  virtual ~IExecutionIOBinder() = default;
  virtual void beforeExecution() = 0;
  virtual void afterExecution() = 0;
};

::executorch::runtime::Result<executorch::aten::Tensor> Decoder::step(
    executorch::extension::TensorPtr& tokens,
    int64_t start_pos) {

  std::string method_stage;
  if (tokens->numel() > 1) {
    method_stage = methods_.name[static_cast<std::size_t>(DecoderMode::PREFILL)];
  } else {
    method_stage = methods_.name[static_cast<std::size_t>(DecoderMode::DECODE)];
  }

  // ET_LOG(Info, "Input token %" PRIu64, input_token);
  auto method_meta = ET_UNWRAP(module_->method_meta(method_stage.c_str()));
  // If only 1 input, we are not using kv cache
  bool use_kv_cache = method_meta.num_inputs() > 1;

  if (use_kv_cache) {
    // Size of the second argument. This could be either input_pos or
    // cache_positions

    // Check if we are using cache positions instead of input pos.
    auto second_input_info = ET_UNWRAP(method_meta.input_tensor_meta(1));
    // For input_pos, numel is 1, for cache_positions, numel is max_seq_len
    auto sizes = second_input_info.sizes();
    // Assuming 1D tensor
    ET_CHECK_OR_RETURN_ERROR(
        sizes.size() == 1,
        InvalidProgram,
        "The second input tensor is not 1D tensor. Got dimension (%zu)",
        sizes.size());
    auto numel = sizes[0];
    std::vector<::executorch::aten::SizesType> sizes_vec = {numel};

    auto n_inputs = method_meta.num_inputs();
    auto input_info = ET_UNWRAP(method_meta.input_tensor_meta(2));
    sizes = input_info.sizes();
    if (sizes.size() != 2) ET_LOG(Error, "third input of %zu is invalid dimension : %zu", n_inputs, sizes.size());

    input_info = ET_UNWRAP(method_meta.input_tensor_meta(3));
    sizes = input_info.sizes();
    if (sizes.size() != 4) ET_LOG(Error, "forth input of %zu is invalid dimension : %zu", n_inputs, sizes.size());

    executorch::extension::TensorPtr start_pos_tensor;

    // Assuming model is exported with input_pos, create a tensor with size 1
    start_pos_tensor = executorch::extension::from_blob(
        &start_pos, sizes_vec, ::executorch::aten::ScalarType::Long);

    std::vector<executorch::runtime::EValue> inputs;

    if (method_stage == methods_.name[static_cast<std::size_t>(DecoderMode::PREFILL)]) {
      auto inputs_res =
          io_manager_->prepare_prefill(tokens, start_pos_tensor);
      ET_CHECK_OK_OR_RETURN_ERROR(inputs_res.error());
      inputs = inputs_res.get();
    } else if (method_stage == methods_.name[static_cast<std::size_t>(DecoderMode::DECODE)]) {
      auto inputs_res =
          io_manager_->prepare_decode(tokens, start_pos_tensor);
      ET_CHECK_OK_OR_RETURN_ERROR(inputs_res.error());
      inputs = inputs_res.get();
    }

    auto outputs_res = module_->execute(method_stage.c_str(), inputs);
    ET_CHECK_OK_OR_RETURN_ERROR(outputs_res.error());
    ET_CHECK_MSG(
        outputs_res.get()[0].isTensor(),
        "Non Tensor Output returned from executing LLM");

    if (method_stage == methods_.name[static_cast<std::size_t>(DecoderMode::PREFILL)]) {
      if (io_manager_->update_prefill(outputs_res.get()) !=
          executorch::runtime::Error::Ok) {
        ET_LOG(Error, "Failed to updated prefill output by io_manager");
      }
    } else if (method_stage == methods_.name[static_cast<std::size_t>(DecoderMode::DECODE)]) {
      if (io_manager_->update_decode(outputs_res.get()) !=
          executorch::runtime::Error::Ok) {
        ET_LOG(Error, "Failed to updated decode output by io_manager");
      }
    }

    // Return the logits tensor
    return outputs_res.get()[0].toTensor();
  } else { // no kv cache
    (void)start_pos; // unused

    auto outputs_res = module_->execute(method_stage.c_str(), tokens);
    ET_LOG(Info, "After module forward");
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

::executorch::runtime::Error Decoder::create_shared_planned_memory() {
  executorch::runtime::Error ret = executorch::runtime::Error::Ok;

  if (shared_method_memory_ != nullptr) {
    ET_LOG(Info, "Shared Planned memory is already created");
    return ret;
  }

  auto prefill_meta = module_->method_meta(
          methods_.name[static_cast<std::size_t>(DecoderMode::PREFILL)].c_str());
  auto decode_meta = module_->method_meta(
          methods_.name[static_cast<std::size_t>(DecoderMode::DECODE)].c_str());
  if (!prefill_meta.ok() || !decode_meta.ok()) {
    ET_LOG(Error, "Failed to get method metadata for shared memory setup");
    return executorch::runtime::Error::InvalidArgument;
  }

  size_t prefill_bufs = prefill_meta->num_memory_planned_buffers();
  size_t decode_bufs = decode_meta->num_memory_planned_buffers();
  size_t num_buffers = std::max(prefill_bufs, decode_bufs);

  ET_LOG(Info, "Prefill_bufs(%ld), decode_bufs(%ld) num_buffers(%ld)",
          prefill_bufs, decode_bufs, num_buffers);

  shared_method_memory_ =
        std::make_unique<SharedMethodMemory>();
  shared_method_memory_->buffers.reserve(num_buffers);
  shared_method_memory_->spans.reserve(num_buffers);

  for (size_t i = 0; i < num_buffers; ++i) {
    int64_t prefill_size =
        (i < prefill_bufs)
        ? prefill_meta->memory_planned_buffer_size(i).get()
        : 0;
    int64_t decode_size =
        (i < decode_bufs)
        ? decode_meta->memory_planned_buffer_size(i).get()
        : 0;
    size_t max_size = static_cast<size_t>(std::max(prefill_size, decode_size));
    ET_LOG(Info, "Prefill_size(%ld), decode_size(%ld) max_size(%ld)",
                  prefill_size, decode_size, max_size);

    shared_method_memory_->buffers.emplace_back(max_size);
    shared_method_memory_->spans.emplace_back(
        shared_method_memory_->buffers.back().data(), max_size);
  }

  shared_method_memory_->allocator =
      std::make_unique<::executorch::runtime::HierarchicalAllocator>(
          ::executorch::runtime::Span<::executorch::runtime::Span<uint8_t>>(
              shared_method_memory_->spans.data(),
              shared_method_memory_->spans.size()));
  return ret;
}



} // namespace example
