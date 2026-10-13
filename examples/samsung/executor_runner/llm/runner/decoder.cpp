/*
 * Copyright (c) 2025 Samsung Electronics Co. LTD
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

// Given inputs, run a text decoder and return logits.

#include <executorch/kernels/portable/cpu/util/arange_util.h>
#include <string>

#include <ctime>

#include <executorch/examples/samsung/executor_runner/llm/runner/decoder.h>
#include <executorch/extension/llm/runner/stats.h>
#include <executorch/extension/llm/runner/text_decoder_runner.h>

namespace example {

namespace runtime = executorch::runtime;

class IExecutionIOBinder {
  virtual ~IExecutionIOBinder() = default;
  virtual void beforeExecution() = 0;
  virtual void afterExecution() = 0;
};

runtime::Result<aten::Tensor> slice_kth_token(
    const runtime::EValue& output_evalue,
    int64_t k) {
  ET_CHECK_MSG(
      output_evalue.isTensor(), "The provided EValue is not a tensor.");
  aten::Tensor logits = output_evalue.toTensor();

  constexpr int64_t kExpectedNumDim = 3;
  constexpr int64_t kSequenceDim = 1;
  const int64_t max_seq_len = logits.size(kSequenceDim);
  const int64_t actual_dim = logits.dim();

  const auto strides = logits.strides();
  const auto sizes = logits.sizes();

  ET_CHECK_MSG(
      k >= 0 && k < max_seq_len,
      "Index k (%ld) out of bounds (max: %ld)",
      k,
      max_seq_len);

  ET_CHECK_MSG(
      actual_dim == kExpectedNumDim,
      "num dim shape of logit(%ld) is not expected(%ld)",
      actual_dim,
      kExpectedNumDim);

  // Calculate Memory Offset (zero-copy)
  int64_t offset = k * strides[kSequenceDim];

  // Define New Shape for the k-th token (1, 1, VocabSize)
  // Using a fixed-size array or vector that matches SizesType
  std::vector<aten::SizesType> new_sizes = {
      static_cast<aten::SizesType>(sizes[0]),
      1, // Target single token
      static_cast<aten::SizesType>(sizes[2])};

  // Create TensorPtr and dereference it to return Result<Tensor>
  auto tensor_ptr = extension::from_blob(
      logits.mutable_data_ptr<float>() + offset,
      new_sizes,
      logits.scalar_type());

  // Dereference the shared_ptr to get the Tensor object
  return *tensor_ptr;
}

runtime::Error Decoder::load() {
  extract_llm_methods();

  if (method_names_.size() > 1) {
    auto ret_shared = create_shared_planned_memory();
    ET_CHECK_OR_RETURN_ERROR(
        ret_shared == runtime::Error::Ok,
        InvalidArgument,
        "Failed to create share planned memory");
  }

  for (const auto& method : method_names_) {
    if (method.empty() || module_->is_method_loaded(method)) {
      continue;
    }

    runtime::Error ret_load;
    if (shared_method_memory_) {
      ret_load =
          module_->load_method(method, shared_method_memory_->allocator.get());
    } else {
      ret_load = module_->load_method(method);
    }
    ET_CHECK_OR_RETURN_ERROR(
        ret_load == runtime::Error::Ok,
        InvalidArgument,
        "Failed to load priority method %s",
        method.c_str());
    ET_LOG(Info, "Loading Priority Method (%s)", method.c_str());
  }

  // Load remaining utility methods (e.g., get_genai_mode)
  for (const auto& method : method_names_) {
    if (module_->is_method_loaded(method)) {
      continue;
    }

    runtime::Error ret_load = module_->load_method(method);
    ET_CHECK_OR_RETURN_ERROR(
        ret_load == runtime::Error::Ok,
        InvalidArgument,
        "Failed to load utility method %s",
        method.c_str());
    ET_LOG(Info, "Loading Utility Method (%s)", method.c_str());
  }

  auto ret = load_io_manager();
  ET_CHECK_OR_RETURN_ERROR(
      ret == runtime::Error::Ok, InvalidArgument, "Failed to load io manager");

  return ret;
}

runtime::Result<aten::Tensor> Decoder::step(
    extension::TensorPtr& tokens,
    int64_t start_pos) {
  std::string method_stage;
  if (tokens->numel() > 1) {
    method_stage = prefill_method_;
  } else {
    method_stage = decode_method_;
  }

  auto method_meta = ET_UNWRAP(module_->method_meta(method_stage.c_str()));

  const size_t total_input_count = method_meta.num_inputs();
  bool use_kv_cache =
      total_input_count > 1; // kv cache is not used, when input is 1
  if (use_kv_cache) {
    // Size of the position tracking argument. This could be either input_pos or
    // cache_positions
    constexpr int64_t kExpectedPositionDimensions = 3;
    constexpr int64_t kPositionTensorIndex = 3;
    constexpr int64_t kAttentionSetupTensorIndex = 1;

    constexpr int64_t kKVCacheExpectedDimensions = 4;
    constexpr int64_t kHeadsDimIndex = 1;
    constexpr int64_t kHeadSizeDimIndex = 3;

    // Check if we are using cache positions instead of input pos.
    auto position_tensor_meta =
        ET_UNWRAP(method_meta.input_tensor_meta(kPositionTensorIndex));
    auto position_shape = position_tensor_meta.sizes();
    ET_CHECK_OR_RETURN_ERROR(
        position_shape.size() == kExpectedPositionDimensions,
        InvalidProgram,
        "The pos input tensor is not expected. Got dimension (%zu)",
        position_shape.size());
    std::vector<aten::SizesType> sizes_vec = {position_shape[0]};

    auto attention_setup_tensor_meta =
        ET_UNWRAP(method_meta.input_tensor_meta(kAttentionSetupTensorIndex));
    auto attention_setup_shape = attention_setup_tensor_meta.sizes();
    auto* llm_io_manager = static_cast<example::LlmIOManager*>(io_manager_);
    const auto& config = llm_io_manager->get_config();
    const size_t kv_cache_tensor_index = kAttentionSetupTensorIndex +
        llm_io_manager->get_attention_setup_tensor_count();

    auto kv_cache_tensor_meta =
        ET_UNWRAP(method_meta.input_tensor_meta(kv_cache_tensor_index));
    auto kv_cache_shape = kv_cache_tensor_meta.sizes();
    ET_CHECK_OR_RETURN_ERROR(
        static_cast<int64_t>(kv_cache_shape.size()) ==
            kKVCacheExpectedDimensions,
        InvalidArgument,
        "Input tensor at index %" PRId64
        " (KV cache slot) has invalid dimensionality: %" PRId64
        " (expected %" PRId64 ")",
        kv_cache_tensor_index,
        static_cast<int64_t>(kv_cache_shape.size()),
        kKVCacheExpectedDimensions);

    ET_CHECK_OR_RETURN_ERROR(
        static_cast<size_t>(kv_cache_shape[kHeadsDimIndex]) ==
            config.n_kv_heads,
        InvalidArgument,
        "KV cache tensor head count mismatch at index %" PRId64
        ": found %" PRId64 " (expected %" PRId64 " via configuration)",
        kv_cache_tensor_index,
        static_cast<size_t>(kv_cache_shape[kHeadsDimIndex]),
        config.n_kv_heads);

    ET_CHECK_OR_RETURN_ERROR(
        static_cast<size_t>(kv_cache_shape[kHeadSizeDimIndex]) ==
            config.head_dim,
        InvalidArgument,
        "KV cache tensor head dimension size mismatch at index %" PRId64
        ": found %" PRId64 " (expected %" PRId64 " via configuration)",
        kv_cache_tensor_index,
        static_cast<size_t>(kv_cache_shape[kHeadSizeDimIndex]),
        config.head_dim);

    extension::TensorPtr start_pos_tensor;
    // Assuming model is exported with input_pos, create a tensor with size 1
    start_pos_tensor =
        extension::from_blob(&start_pos, sizes_vec, aten::ScalarType::Long);

    std::vector<runtime::EValue> inputs;

    if (method_stage == prefill_method_) {
      auto inputs_res = io_manager_->prepare_prefill(tokens, start_pos_tensor);
      ET_CHECK_OK_OR_RETURN_ERROR(inputs_res.error());
      inputs = inputs_res.get();
    } else if (method_stage == decode_method_) {
      auto inputs_res = io_manager_->prepare_decode(tokens, start_pos_tensor);
      ET_CHECK_OK_OR_RETURN_ERROR(inputs_res.error());
      inputs = inputs_res.get();
    } else {
      ET_LOG(
          Debug,
          "Skipping IO prepare for non-execution method stage: %s",
          method_stage.c_str());
    }

    auto outputs_res = module_->execute(method_stage.c_str(), inputs);
    ET_CHECK_OK_OR_RETURN_ERROR(outputs_res.error());
    ET_CHECK_MSG(
        outputs_res.get()[0].isTensor(),
        "Non Tensor Output returned from executing LLM");

    if (method_stage == prefill_method_) {
      ET_CHECK_OK_OR_RETURN_ERROR(
          io_manager_->update_prefill(outputs_res.get()),
          "Failed to update prefill output by io_manager");
    } else if (method_stage == decode_method_) {
      ET_CHECK_OK_OR_RETURN_ERROR(
          io_manager_->update_decode(outputs_res.get()),
          "Failed to update decode output by io_manager");
    } else {
      ET_LOG(
          Debug,
          "Skipping IO update for non-execution method stage: %s",
          method_stage.c_str());
    }
    // Return the logits tensor
    return slice_kth_token(outputs_res.get()[0], (tokens->numel() - 1));
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

void Decoder::extract_llm_methods() {
  if (method_names_.size() > 1) {
    for (auto it = method_names_.begin(); it != method_names_.end(); ++it) {
      if (it->find("prefill") != std::string::npos) {
        prefill_method_ = *it;
      } else if (it->find("decode") != std::string::npos) {
        decode_method_ = *it;
      } else {
        ET_LOG(
            Info,
            "skipping io manager identify with %s method name.",
            (*it).c_str());
        continue;
      }
    }
  } else if (method_names_.size() == 1) {
    const std::string& method = *method_names_.begin();
    prefill_method_ = method;
    decode_method_ = method;
  }
}

runtime::Error Decoder::load_io_manager() {
  if (method_names_.empty()) {
    return runtime::Error::InvalidArgument;
  }
  return io_manager_->load(prefill_method_, decode_method_);
}

runtime::Error Decoder::create_shared_planned_memory() {
  if (shared_method_memory_ != nullptr) {
    ET_LOG(Info, "Shared Planned memory is already created");
    return runtime::Error::Ok;
  }

  auto prefill_meta = module_->method_meta(prefill_method_);
  auto decode_meta = module_->method_meta(decode_method_);
  if (!prefill_meta.ok() || !decode_meta.ok()) {
    ET_LOG(Error, "Failed to get method metadata for shared memory setup");
    return runtime::Error::InvalidArgument;
  }

  size_t prefill_bufs = prefill_meta->num_memory_planned_buffers();
  size_t decode_bufs = decode_meta->num_memory_planned_buffers();
  size_t num_buffers = std::max(prefill_bufs, decode_bufs);

  ET_LOG(
      Info,
      "Prefill_bufs(%ld), decode_bufs(%ld) num_buffers(%ld)",
      prefill_bufs,
      decode_bufs,
      num_buffers);

  shared_method_memory_ = std::make_unique<SharedMethodMemory>();
  shared_method_memory_->buffers.reserve(num_buffers);
  shared_method_memory_->spans.reserve(num_buffers);

  for (size_t i = 0; i < num_buffers; ++i) {
    int64_t prefill_size = (i < prefill_bufs)
        ? prefill_meta->memory_planned_buffer_size(i).get()
        : 0;
    int64_t decode_size = (i < decode_bufs)
        ? decode_meta->memory_planned_buffer_size(i).get()
        : 0;
    size_t max_size = static_cast<size_t>(std::max(prefill_size, decode_size));
    ET_LOG(
        Info,
        "Prefill_size(%ld), decode_size(%ld) max_size(%ld)",
        prefill_size,
        decode_size,
        max_size);

    shared_method_memory_->buffers.emplace_back(max_size);
    shared_method_memory_->spans.emplace_back(
        shared_method_memory_->buffers.back().data(), max_size);
  }

  shared_method_memory_->allocator =
      std::make_unique<runtime::HierarchicalAllocator>(
          runtime::Span<runtime::Span<uint8_t>>(
              shared_method_memory_->spans.data(),
              shared_method_memory_->spans.size()));

  return runtime::Error::Ok;
}

} // namespace example
