/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <executorch/extension/llm/runner/io_manager/io_manager.h>
#include <executorch/extension/tensor/tensor.h>

namespace executorch {
namespace extension {
namespace llm {
namespace exec_aten = ::executorch::aten;

/**
 * @brief Configuration for attention sink behavior.
 */
struct AttentionSinkConfig {
  /// Number of initial tokens to always keep (sink tokens).
  int64_t sink_size = 4;

  /// Size of the sliding window for non-sink tokens.
  int64_t window_size = 508;

  /// When the cache is full, evict this many tokens at once.
  int64_t eviction_batch_size = 256;
};

/**
 * @brief IOManager that supports attention sink models for infinite context.
 *
 * This IOManager is designed to work with models that have been exported with
 * attention sink support (KVCacheWithAttentionSink). The model internally
 * manages:
 * - Cache write indices (sink tokens at fixed positions, rest in ring buffer)
 * - Attention mask creation (sink tokens always visible + sliding window)
 * - Position-based RoPE embeddings
 *
 * The IOManager's role is to:
 * 1. Pass through input_pos as-is (the model handles position mapping)
 * 2. Track logical position for runner bookkeeping
 * 3. Allow generation to continue past max_cache_size without errors
 *
 * This works with models exported using:
 * - KVCacheWithAttentionSink (model-side attention sink)
 * - RingKVCache (standard ring buffer - provides sliding window without sink)
 *
 * Cache layout (managed by model, not runner):
 *   [sink tokens: 0 to sink_size-1] [ring buffer: sink_size to cache_size-1]
 *
 * Usage:
 *   1. Export model with attention sink config (use_ring_kv_cache=True, etc.)
 *   2. Runner detects attention sink metadata and creates this IOManager
 *   3. IOManager passes positions through; model handles cache management
 */
class ET_EXPERIMENTAL AttentionSinkIOManager : public IOManager {
 public:
  /**
   * @brief Construct an AttentionSinkIOManager.
   *
   * @param module The Module used for querying method metadata and execution.
   * @param max_cache_size The maximum size of the KV cache in the model.
   * @param config Configuration for attention sink behavior.
   */
  AttentionSinkIOManager(
      ET_MODULE_NAMESPACE::Module& module,
      int64_t max_context_len,
      AttentionSinkConfig config = AttentionSinkConfig());

  /**
   * @brief Load the IO manager with method metadata.
   */
  ET_NODISCARD runtime::Error load(
      const std::string& prefill_method,
      const std::string& decode_method) override;

  /**
   * @brief Reset the IO manager state.
   *
   * Resets the logical position counter.
   */
  ET_NODISCARD runtime::Error reset(
      const std::string& prefill_method,
      const std::string& decode_method) override;

  /**
   * @brief Prepare inputs for the prefill phase.
   *
   * Passes through input_pos to the model. The model's internal
   * KVCacheWithAttentionSink handles position-to-index mapping and masking.
   */
  runtime::Result<std::vector<runtime::EValue>> prepare_prefill(
      const TensorPtr& input,
      const TensorPtr& start_pos,
      const std::string& prefill_method) override;

  /**
   * @brief Prepare inputs for the decode phase.
   *
   * Passes through input_pos to the model. The model's internal
   * KVCacheWithAttentionSink handles position-to-index mapping and masking.
   */
  runtime::Result<std::vector<runtime::EValue>> prepare_decode(
      const TensorPtr& input,
      const TensorPtr& start_pos,
      const std::string& decode_method) override;

  /**
   * @brief Get the current logical position.
   *
   * This is the position in the full context, which may exceed the cache size.
   * The model handles wrapping internally via ring buffer.
   */
  int64_t logical_position() const {
    return logical_pos_;
  }

  /**
   * @brief Get the attention sink configuration.
   */
  const AttentionSinkConfig& config() const {
    return config_;
  }

  /**
   * @brief Check if the cache is in the "infinite context" regime.
   *
   * Returns true when the logical position exceeds the effective cache
   * capacity, meaning the ring buffer has wrapped and old tokens are being
   * overwritten.
   */
  bool is_cache_full() const {
    return logical_pos_ >= max_context_len_;
  }

 private:
  /// Maximum size of the KV cache in the model
  int64_t max_context_len_;

  /// Attention sink configuration
  AttentionSinkConfig config_;

  /// Current logical position (may exceed max_cache_size)
  int64_t logical_pos_ = 0;

  /**
   * @brief Update the internal indices buffer and tensor for a given position and length.
   */
  void update_indices_tensor(int64_t logical_start, int64_t seq_len);

  // Buffer for cache indices
  std::vector<int64_t> indices_buffer_;

  // Tensor wrapper for indices
  std::unique_ptr<exec_aten::TensorImpl> indices_tensor_impl_;
  std::unique_ptr<exec_aten::Tensor> indices_tensor_;

  // Metadata storage for TensorImpl
  std::vector<exec_aten::TensorImpl::SizesType> sizes_vec_;
  std::vector<exec_aten::TensorImpl::DimOrderType> dim_order_vec_;
  std::vector<exec_aten::TensorImpl::StridesType> strides_vec_;
};

} // namespace llm
} // namespace extension
} // namespace executorch
