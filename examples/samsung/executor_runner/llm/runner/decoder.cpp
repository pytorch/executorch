/*
 * Copyright (c) 2025 Samsung Electronics Co. LTD
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

// Given inputs, run a text decoder and return logits.

#include <executorch/examples/samsung/executor_runner/llm/runner/decoder.h>
#include <executorch/extension/llm/runner/stats.h>
#include <executorch/extension/llm/runner/text_decoder_runner.h>
#include <executorch/kernels/portable/cpu/util/arange_util.h>

#include <ctime>
#include <string>

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

  bool use_kv_cache = total_input_count > 1;
  if (use_kv_cache) {
    auto* llm_io_manager = static_cast<example::LlmIOManager*>(io_manager_);
    const auto& config = llm_io_manager->get_config();
    auto model_version_config = getModelVersionConfig(config.model_version);

    //
    // Validate attention-mask tensors.
    //
    for (size_t i = 0; i < model_version_config.attention_mask_tensor_count;
         ++i) {
      const size_t tensor_index =
          model_version_config.attention_mask_tensor_index + i;

      auto tensor_meta = ET_UNWRAP(method_meta.input_tensor_meta(tensor_index));

      const auto tensor_shape = tensor_meta.sizes();

      ET_CHECK_OR_RETURN_ERROR(
          tensor_shape.size() == model_version_config.attention_mask_rank,
          InvalidProgram,
          "Unexpected attention mask rank at input %zu. "
          "Expected %zu, got %zu.",
          tensor_index,
          model_version_config.attention_mask_rank,
          tensor_shape.size());
    }

    //
    // Validate position / RoPE tensors.
    //
    // Gemma3:
    //   position[0], position[1]
    //
    // Gemma4:
    //   global_cos, global_sin, local_cos, local_sin
    //
    for (size_t i = 0; i < model_version_config.position_tensor_count; ++i) {
      const size_t tensor_index =
          model_version_config.position_tensor_index + i;

      auto tensor_meta = ET_UNWRAP(method_meta.input_tensor_meta(tensor_index));

      const auto tensor_shape = tensor_meta.sizes();

      ET_CHECK_OR_RETURN_ERROR(
          tensor_shape.size() == model_version_config.position_rank,
          InvalidProgram,
          "Unexpected position tensor rank at input %zu. "
          "Expected %zu, got %zu.",
          tensor_index,
          model_version_config.position_rank,
          tensor_shape.size());
    }

    //
    // Validate KV-cache tensors.
    //
    // We only need one cache tensor to determine/validate the common
    // KV-cache layout [B, H, S, D], assuming all cache leaves use
    // the same rank/layout.
    //
    auto k_cache_tensor_meta = ET_UNWRAP(method_meta.input_tensor_meta(
        model_version_config.k_cache_tensor_index));

    const auto k_cache_shape = k_cache_tensor_meta.sizes();

    ET_CHECK_OR_RETURN_ERROR(
        k_cache_shape.size() == model_version_config.kv_cache_rank,
        InvalidProgram,
        "Unexpected K-cache rank. Expected %zu, got %zu.",
        model_version_config.kv_cache_rank,
        k_cache_shape.size());

    auto v_cache_tensor_meta = ET_UNWRAP(method_meta.input_tensor_meta(
        model_version_config.v_cache_tensor_index));

    const auto v_cache_shape = v_cache_tensor_meta.sizes();

    ET_CHECK_OR_RETURN_ERROR(
        v_cache_shape.size() == model_version_config.kv_cache_rank,
        InvalidProgram,
        "Unexpected V-cache rank. Expected %zu, got %zu.",
        model_version_config.kv_cache_rank,
        v_cache_shape.size());

    //
    // Validate the configured KV-cache axes before indexing them.
    //
    ET_CHECK_OR_RETURN_ERROR(
        model_version_config.heads_dim_index < k_cache_shape.size(),
        InvalidProgram,
        "KV heads dimension index %zu exceeds cache rank %zu.",
        model_version_config.heads_dim_index,
        k_cache_shape.size());

    ET_CHECK_OR_RETURN_ERROR(
        model_version_config.head_size_dim_index < k_cache_shape.size(),
        InvalidProgram,
        "KV head-size dimension index %zu exceeds cache rank %zu.",
        model_version_config.head_size_dim_index,
        k_cache_shape.size());

    const auto num_kv_heads =
        k_cache_shape[model_version_config.heads_dim_index];

    const auto kv_head_size =
        k_cache_shape[model_version_config.head_size_dim_index];

    ET_LOG(
        Debug,
        "KV cache: K base=%zu, V base=%zu, count=%zu, "
        "heads=%d, head_size=%d",
        model_version_config.k_cache_tensor_index,
        model_version_config.v_cache_tensor_index,
        model_version_config.kv_cache_tensor_count,
        num_kv_heads,
        kv_head_size);

    //
    // Optional sanity check that the final V-cache input actually
    // exists in this PTE method.
    //
    const size_t last_v_cache_index =
        model_version_config.v_cache_tensor_index +
        model_version_config.kv_cache_tensor_count - 1;

    ET_CHECK_OR_RETURN_ERROR(
        last_v_cache_index < method_meta.num_inputs(),
        InvalidProgram,
        "KV cache layout exceeds PTE input count. "
        "Last V-cache input=%zu, num_inputs=%zu.",
        last_v_cache_index,
        method_meta.num_inputs());

    // Assuming model is exported with input_pos, create a tensor with size 1
    extension::TensorPtr start_pos_tensor;
    std::vector<aten::SizesType> start_pos_sizes = {1};
    start_pos_tensor = extension::from_blob(
        &start_pos, start_pos_sizes, aten::ScalarType::Long);
    std::vector<runtime::EValue> inputs;
    if (method_stage == prefill_method_) {
      auto inputs_res = io_manager_->prepare_prefill(tokens, start_pos_tensor);
      ET_CHECK_OK_OR_RETURN_ERROR(inputs_res.error());
      inputs = inputs_res.get();

      ET_LOG(
          Info,
          "Prepared %zu inputs, PTE expects %zu inputs",
          inputs.size(),
          method_meta.num_inputs());

      const size_t count = std::min(inputs.size(), method_meta.num_inputs());

      for (size_t i = 0; i < count; ++i) {
        if (!inputs[i].isTensor()) {
          ET_LOG(Info, "input[%zu]: prepared EValue is not Tensor", i);
          continue;
        }

        const auto prepared_tensor = inputs[i].toTensor();
        const auto prepared_sizes = prepared_tensor.sizes();

        auto expected_meta = ET_UNWRAP(method_meta.input_tensor_meta(i));
        const auto expected_sizes = expected_meta.sizes();

        ET_LOG(
            Info,
            "input[%zu]: prepared_rank=%zu expected_rank=%zu",
            i,
            prepared_sizes.size(),
            expected_sizes.size());

        for (size_t d = 0; d < prepared_sizes.size(); ++d) {
          ET_LOG(Info, "  prepared[%zu]=%d", d, prepared_sizes[d]);
        }

        for (size_t d = 0; d < expected_sizes.size(); ++d) {
          ET_LOG(Info, "  expected[%zu]=%d", d, expected_sizes[d]);
        }
      }

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
  } else { // kv cache is not used, when input is 1
    (void)start_pos;
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
