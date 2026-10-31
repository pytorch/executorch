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

#include <cinttypes>
#include <string>
#include <vector>

namespace example {

namespace runtime = executorch::runtime;

namespace {

runtime::Error validate_inputs(
    const runtime::MethodMeta& method_meta,
    const std::vector<runtime::EValue>& inputs) {
  ET_CHECK_OR_RETURN_ERROR(
      inputs.size() == method_meta.num_inputs(),
      InvalidProgram,
      "Method %s expects %zu inputs but the runner prepared %zu",
      method_meta.name(),
      method_meta.num_inputs(),
      inputs.size());
  for (size_t i = 0; i < inputs.size(); ++i) {
    if (!inputs[i].isTensor()) {
      continue;
    }
    auto expected = ET_UNWRAP(method_meta.input_tensor_meta(i));
    const auto prepared_rank = inputs[i].toTensor().dim();
    ET_CHECK_OR_RETURN_ERROR(
        static_cast<size_t>(prepared_rank) == expected.sizes().size(),
        InvalidProgram,
        "Method %s input %zu: expected rank %zu, prepared rank %zu",
        method_meta.name(),
        i,
        expected.sizes().size(),
        static_cast<size_t>(prepared_rank));
  }
  return runtime::Error::Ok;
}

} // namespace

runtime::Result<aten::Tensor> Decoder::slice_kth_token(
    const runtime::EValue& output,
    int64_t k) {
  ET_CHECK_OR_RETURN_ERROR(
      output.isTensor(), InvalidProgram, "Logits output is not a tensor");
  aten::Tensor logits = output.toTensor();

  constexpr int64_t kExpectedNumDim = 3;
  constexpr int64_t kSequenceDim = 1;
  ET_CHECK_OR_RETURN_ERROR(
      logits.dim() == kExpectedNumDim,
      InvalidProgram,
      "Logits rank %zu is not %" PRId64,
      static_cast<size_t>(logits.dim()),
      kExpectedNumDim);
  const int64_t seq_len = logits.size(kSequenceDim);
  ET_CHECK_OR_RETURN_ERROR(
      k >= 0 && k < seq_len,
      InvalidArgument,
      "Token index %" PRId64 " out of bounds for %" PRId64 " logits rows",
      k,
      seq_len);

  const auto sizes = logits.sizes();
  std::vector<aten::SizesType> new_sizes = {
      static_cast<aten::SizesType>(sizes[0]),
      1,
      static_cast<aten::SizesType>(sizes[2])};
  uint8_t* row = logits.mutable_data_ptr<uint8_t>() +
      k * logits.strides()[kSequenceDim] * logits.element_size();
  last_logits_ = extension::from_blob(row, new_sizes, logits.scalar_type());
  return *last_logits_;
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

bool Decoder::is_method_loaded() {
  return !prefill_method_.empty() && !decode_method_.empty() &&
      module_->is_method_loaded(prefill_method_) &&
      module_->is_method_loaded(decode_method_);
}

runtime::Result<aten::Tensor> Decoder::step(
    extension::TensorPtr& tokens,
    int64_t start_pos) {
  const bool is_prefill = tokens->numel() > 1;
  const std::string& method_stage =
      is_prefill ? prefill_method_ : decode_method_;

  auto method_meta = ET_UNWRAP(module_->method_meta(method_stage.c_str()));

  if (method_meta.num_inputs() <= 1) {
    auto outputs_res = module_->execute(method_stage.c_str(), tokens);
    ET_CHECK_OK_OR_RETURN_ERROR(outputs_res.error());
    ET_CHECK_OR_RETURN_ERROR(
        outputs_res.get().size() == 1 && outputs_res.get()[0].isTensor(),
        InvalidProgram,
        "Expected a single logits tensor from %s",
        method_stage.c_str());
    return outputs_res.get()[0].toTensor();
  }

  std::vector<aten::SizesType> start_pos_sizes = {1};
  auto start_pos_tensor =
      extension::from_blob(&start_pos, start_pos_sizes, aten::ScalarType::Long);
  auto inputs_res = is_prefill
      ? io_manager_->prepare_prefill(tokens, start_pos_tensor, method_stage)
      : io_manager_->prepare_decode(tokens, start_pos_tensor, method_stage);
  ET_CHECK_OK_OR_RETURN_ERROR(inputs_res.error());
  std::vector<runtime::EValue> inputs = std::move(inputs_res.get());
  ET_CHECK_OK_OR_RETURN_ERROR(validate_inputs(method_meta, inputs));

  auto outputs_res = module_->execute(method_stage.c_str(), inputs);
  ET_CHECK_OK_OR_RETURN_ERROR(outputs_res.error());
  ET_CHECK_OR_RETURN_ERROR(
      !outputs_res.get().empty(),
      InvalidProgram,
      "%s returned no outputs",
      method_stage.c_str());

  if (is_prefill) {
    ET_CHECK_OK_OR_RETURN_ERROR(
        io_manager_->update_prefill(outputs_res.get(), method_stage),
        "Failed to update prefill output by io_manager");
  } else {
    ET_CHECK_OK_OR_RETURN_ERROR(
        io_manager_->update_decode(outputs_res.get(), method_stage),
        "Failed to update decode output by io_manager");
  }
  return slice_kth_token(outputs_res.get()[0], tokens->numel() - 1);
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
      "Prefill_bufs(%zu), decode_bufs(%zu) num_buffers(%zu)",
      prefill_bufs,
      decode_bufs,
      num_buffers);

  shared_method_memory_ = std::make_unique<SharedMethodMemory>();
  shared_method_memory_->buffers.reserve(num_buffers);
  shared_method_memory_->spans.reserve(num_buffers);

  for (size_t i = 0; i < num_buffers; ++i) {
    int64_t prefill_size = 0;
    if (i < prefill_bufs) {
      prefill_size = ET_UNWRAP(prefill_meta->memory_planned_buffer_size(i));
    }
    int64_t decode_size = 0;
    if (i < decode_bufs) {
      decode_size = ET_UNWRAP(decode_meta->memory_planned_buffer_size(i));
    }
    size_t max_size = static_cast<size_t>(std::max(prefill_size, decode_size));
    ET_LOG(
        Info,
        "Prefill_size(%" PRId64 "), decode_size(%" PRId64 ") max_size(%zu)",
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
