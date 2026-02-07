/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <executorch/extension/llm/runner/io_manager/attention_sink_io_manager.h>

namespace executorch {
namespace extension {
namespace llm {

AttentionSinkIOManager::AttentionSinkIOManager(
    ET_MODULE_NAMESPACE::Module& module,
    int64_t max_context_len,
    AttentionSinkConfig config)
    : IOManager(module),
      max_context_len_(max_context_len),
      config_(config),
      logical_pos_(0) {
  ET_CHECK_MSG(
      config_.sink_size >= 0,
      "sink_size must be non-negative, got %" PRId64,
      config_.sink_size);
  ET_CHECK_MSG(
      config_.window_size > 0,
      "window_size must be positive, got %" PRId64,
      config_.window_size);
}

runtime::Error AttentionSinkIOManager::load(
    const std::string& prefill_method,
    const std::string& decode_method) {
  (void)prefill_method;
  (void)decode_method;

  ET_LOG(
      Info,
      "AttentionSinkIOManager loaded: sink_size=%" PRId64
      ", window_size=%" PRId64 ", max_context_len=%" PRId64,
      config_.sink_size,
      config_.window_size,
      max_context_len_);

  return runtime::Error::Ok;
}

runtime::Error AttentionSinkIOManager::reset(
    const std::string& prefill_method,
    const std::string& decode_method) {
  (void)prefill_method;
  (void)decode_method;

  logical_pos_ = 0;

  ET_LOG(Debug, "AttentionSinkIOManager reset");
  return runtime::Error::Ok;
}

runtime::Result<std::vector<runtime::EValue>>
AttentionSinkIOManager::prepare_prefill(
    const TensorPtr& input,
    const TensorPtr& start_pos,
    const std::string& prefill_method) {
  int64_t logical_start = start_pos->data_ptr<int64_t>()[0];
  int64_t seq_len = input->numel();

  logical_pos_ = logical_start + seq_len;

  ET_LOG(
      Debug,
      "AttentionSinkIOManager::prepare_prefill: logical_start=%" PRId64
      ", seq_len=%" PRId64 ", logical_pos_after=%" PRId64
      ", cache_full=%s",
      logical_start,
      seq_len,
      logical_pos_,
      is_cache_full() ? "true" : "false");

  // Check if we need to provide cache_indices (3rd input)
  auto method_meta = module_.method_meta(prefill_method);
  if (method_meta.ok() && method_meta->num_inputs() == 3) {
    update_indices_tensor(logical_start, seq_len);
    return std::vector<runtime::EValue>{input, start_pos, *indices_tensor_};
  }

  // Pass through to model as-is. 
  return std::vector<runtime::EValue>{input, start_pos};
}

runtime::Result<std::vector<runtime::EValue>>
AttentionSinkIOManager::prepare_decode(
    const TensorPtr& input,
    const TensorPtr& start_pos,
    const std::string& decode_method) {
  int64_t logical_start = start_pos->data_ptr<int64_t>()[0];
  int64_t seq_len = input->numel();

  logical_pos_ = logical_start + seq_len;

  ET_LOG(
      Debug,
      "AttentionSinkIOManager::prepare_decode: logical_start=%" PRId64
      ", logical_pos_after=%" PRId64
      ", cache_full=%s",
      logical_start,
      logical_pos_,
      is_cache_full() ? "true" : "false");

  // Check if we need to provide cache_indices (3rd input)
  auto method_meta = module_.method_meta(decode_method);
  if (method_meta.ok() && method_meta->num_inputs() == 3) {
    update_indices_tensor(logical_start, seq_len);
    return std::vector<runtime::EValue>{input, start_pos, *indices_tensor_};
  }

  // Pass through to model as-is.
  return std::vector<runtime::EValue>{input, start_pos};
}

void AttentionSinkIOManager::update_indices_tensor(
    int64_t logical_start,
    int64_t seq_len) {
  int64_t ring_size = max_context_len_ - config_.sink_size;
  ET_CHECK_MSG(ring_size > 0, "ring_size must be positive, got %" PRId64, ring_size);
  ET_CHECK_MSG(
      ring_size >= config_.window_size,
      "ring_size (%" PRId64 ") must be >= window_size (%" PRId64 ")",
      ring_size,
      config_.window_size);
  indices_buffer_.resize(seq_len);
  for (int64_t i = 0; i < seq_len; ++i) {
    int64_t pos = logical_start + i;
    if (pos < config_.sink_size) {
      indices_buffer_[i] = pos;
    } else {
      indices_buffer_[i] =
          config_.sink_size + (pos - config_.sink_size) % ring_size;
    }
  }

  // Wrap in tensor
  if (!indices_tensor_impl_ || indices_tensor_impl_->size(0) != seq_len) {
    sizes_vec_ = {static_cast<exec_aten::TensorImpl::SizesType>(seq_len)};
    dim_order_vec_ = {0};
    strides_vec_ = {1};

    indices_tensor_impl_ = std::make_unique<exec_aten::TensorImpl>(
        exec_aten::ScalarType::Long,
        1,
        sizes_vec_.data(),
        static_cast<void*>(indices_buffer_.data()),
        dim_order_vec_.data(),
        strides_vec_.data(),
        exec_aten::TensorShapeDynamism::DYNAMIC_BOUND);
    indices_tensor_ =
        std::make_unique<exec_aten::Tensor>(indices_tensor_impl_.get());
  } else {
    // Update logic if buffer moved (vector resize might reallocate)
    // Just re-create to be safe as data ptr is used
    sizes_vec_ = {static_cast<exec_aten::TensorImpl::SizesType>(seq_len)};
    dim_order_vec_ = {0};
    strides_vec_ = {1};

    indices_tensor_impl_ = std::make_unique<exec_aten::TensorImpl>(
        exec_aten::ScalarType::Long,
        1,
        sizes_vec_.data(),
        static_cast<void*>(indices_buffer_.data()),
        dim_order_vec_.data(),
        strides_vec_.data(),
        exec_aten::TensorShapeDynamism::DYNAMIC_BOUND);
    indices_tensor_ =
        std::make_unique<exec_aten::Tensor>(indices_tensor_impl_.get());
  }
}

} // namespace llm
} // namespace extension
} // namespace executorch
