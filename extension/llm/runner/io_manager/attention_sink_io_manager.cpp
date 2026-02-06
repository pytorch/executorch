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
    int64_t max_cache_size,
    AttentionSinkConfig config)
    : IOManager(module),
      max_cache_size_(max_cache_size),
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
      ", window_size=%" PRId64 ", max_cache_size=%" PRId64,
      config_.sink_size,
      config_.window_size,
      max_cache_size_);

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

  // Pass through to model as-is. The model's KVCacheWithAttentionSink
  // (or RingKVCache) handles position-to-index mapping and mask creation.
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

  // Pass through to model as-is. The model's KVCacheWithAttentionSink
  // (or RingKVCache) handles position-to-index mapping and mask creation.
  return std::vector<runtime::EValue>{input, start_pos};
}

} // namespace llm
} // namespace extension
} // namespace executorch
