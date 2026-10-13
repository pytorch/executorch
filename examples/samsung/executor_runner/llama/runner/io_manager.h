/*
 * Copyright (c) 2025 Samsung Electronics Co. LTD
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <vector>

#include <executorch/extension/llm/runner/io_manager/io_manager.h>
#include <executorch/examples/samsung/executor_runner/llama/runner/constants.h>

namespace example {

/**
 * IO manager for models where the runner manages KV-cache externally.
 *
 * The model receives past KV-cache as input and produces new KV-cache entries
 * as output. The runner is responsible for concatenating outputs into the
 * cache and feeding the updated cache back on the next step.
 *
 * Supports micro-batch prefill: the runner may call prepare_prefill /
 * update_prefill multiple times (each processing prefill_seq_len tokens)
 * before switching to decode. The KV-cache accumulates in the prefill
 * buffer across micro-batches and is copied to the decode buffer on the
 * first prepare_decode call.
 *
 * Prefill (per micro-batch):
 *   Input:  token[prefill_seq_len], start_pos, attn_mask[R,C],
 *           k_cache[1,H,past,D] x L, v_cache[1,H,past,D] x L
 *   Output: logits, k_cache[1,H,prefill_seq_len,D] x L,
 *           v_cache[1,H,prefill_seq_len,D] x L
 *   where past = max_seq_len - prefill_seq_len
 *
 * Decode:
 *   Input:  token[1], start_pos, attn_mask[R,C],
 *           k_cache[1,H,max-1,D] x L, v_cache[1,H,max-1,D] x L
 *   Output: logits, k_cache[1,H,1,D] x L, v_cache[1,H,1,D] x L
 */
class ET_EXPERIMENTAL LlmIOManager
    : public executorch::extension::llm::IOManager {
 public:
   struct Config {
    size_t max_seq_len;
    size_t prefill_seq_len;
    size_t n_layers;
    size_t n_kv_heads;
    size_t head_dim;
    size_t attn_mask_rows = 128;
    size_t attn_mask_cols = 1024;
  };

  /**
   * @brief Construct an IOManager bound to a Module.
   *
   * @param module The Module used for querying method metadata and execution.
   */
  LlmIOManager(executorch::extension::Module& module, Config config)
      : IOManager(module), module_(module), config_(config) {
    ET_LOG(Info,
            "(%s) layers(%ld), kv_heads(%ld) max_sqe_len(%ld) \
            head_dim(%ld) prefill_seq_len(%ld)",
            ET_FUNCTION, config_.n_layers, config_.n_kv_heads,
            config_.max_seq_len, config_.head_dim, config_.prefill_seq_len);
    (void)module_;

    prefill_cache_len_ = config_.max_seq_len - config_.prefill_seq_len;
    decode_cache_len_ = config_.max_seq_len - 1;

    allocate_buffers();
  }

  /**
   * @brief Virtual destructor to allow proper cleanup in derived classes.
   */
  virtual ~LlmIOManager() = default;

  /**
   * @brief Load the IO manager with method metadata for prefill and
   * decode operations.
   *
   * @param prefill_method The prefill method to initialize with.
   * @param decode_method The decode method to initialize with.
   */
  ET_NODISCARD virtual executorch::runtime::Error load(
      const std::string& prefill_method,
      const std::string& decode_method) override {
    ET_LOG(Info,
            "(%s) layers(%ld), kv_heads(%ld) max_sqe_len(%ld) \
            head_dim(%ld) prefill_seq_len(%ld)",
            ET_FUNCTION, config_.n_layers, config_.n_kv_heads,
            config_.max_seq_len, config_.head_dim, config_.prefill_seq_len);

    return executorch::runtime::Error::Ok;
  }

  /**
   * @brief Reset the IO manager state.
   *
   * @param prefill_method The prefill method to reset with.
   * @param decode_method The decode method to reset with.
   */
  ET_NODISCARD virtual executorch::runtime::Error reset(
      const std::string& prefill_method,
      const std::string& decode_method) override {
    (void)prefill_method;
    (void)decode_method;

    dump_step_ = 0;

    current_pos_ = 0;
    prefill_done_ = false;

    for (size_t l = 0; l < config_.n_layers; l++) {
      std::fill(
          prefill_k_input_[l].begin(), prefill_k_input_[l].end(), 0.0f);
      std::fill(
          prefill_v_input_[l].begin(), prefill_v_input_[l].end(), 0.0f);
      std::fill(
          decode_k_input_[l].begin(), decode_k_input_[l].end(), 0.0f);
      std::fill(
          decode_v_input_[l].begin(), decode_v_input_[l].end(), 0.0f);
    }
    std::fill(attn_mask_.begin(), attn_mask_.end(), 0.0f);

    return executorch::runtime::Error::Ok;
  }

  /**
   * @brief Prepare inputs for the prefill phase of LLM inference.
   *
   * @param input The input tensor containing token IDs.
   * @param start_pos The tensor containing the starting position of the current
   * input within the context.
   * @param prefill_method The prefill method to prepare inputs for.
   * @return std::vector<executorch::runtime::EValue> Vector of prepared inputs
   * for the prefill method.
   */
  virtual executorch::runtime::Result<std::vector<executorch::runtime::EValue>>
  prepare_prefill(
      const executorch::extension::TensorPtr& input,
      const executorch::extension::TensorPtr& start_pos,
      const std::string& prefill_method) override;

  /**
   * @brief Prepare inputs for the prefill phase using the default method name.
   *
   * Uses "forward" as the prefill method.
   *
   * @param input The input tensor containing token IDs.
   * @param start_pos The tensor containing the starting position.
   * @return Vector of prepared inputs for the prefill method.
   */
  executorch::runtime::Result<std::vector<executorch::runtime::EValue>>
  prepare_prefill(
      const executorch::extension::TensorPtr& input,
      const executorch::extension::TensorPtr& start_pos) {
    return prepare_prefill(input, start_pos, example::default_method_name_prefill);
  }

  /**
   * @brief Prepare inputs for the decode phase of LLM inference.
   *
   * @param input The input tensor containing token IDs.
   * @param start_pos The tensor containing the starting position of the current
   * input within the context.
   * @param decode_method The decode method to prepare inputs for.
   * @return std::vector<executorch::runtime::EValue> Vector of prepared inputs
   * for the decode method.
   */
  virtual executorch::runtime::Result<std::vector<executorch::runtime::EValue>>
  prepare_decode(
      const executorch::extension::TensorPtr& input,
      const executorch::extension::TensorPtr& start_pos,
      const std::string& decode_method) override;

  /**
   * @brief Prepare inputs for the decode phase using the default method name.
   *
   * Uses "forward" as the decode method.
   *
   * @param input The input tensor containing token IDs.
   * @param start_pos The tensor containing the starting position.
   * @return Vector of prepared inputs for the decode method.
   */
  executorch::runtime::Result<std::vector<executorch::runtime::EValue>>
  prepare_decode(
      const executorch::extension::TensorPtr& input,
      const executorch::extension::TensorPtr& start_pos) {
    return prepare_decode(input, start_pos, example::default_method_name_decode);
  }

  /**
   * @brief Process and update internal state with outputs from the prefill
   * phase.
   *
   * @param prefill_method The prefill method to update with outputs.
   * @param model_outputs Vector of outputs from the prefill method execution.
   */
  ET_NODISCARD virtual executorch::runtime::Error update_prefill(
      const std::vector<executorch::runtime::EValue>& model_outputs,
      const std::string& prefill_method) override;

  /**
   * @brief Process outputs from the prefill phase using the default method.
   *
   * Uses "forward" as the prefill method.
   *
   * @param model_outputs Vector of outputs from the prefill execution.
   * @return Error code.
   */
  ET_NODISCARD executorch::runtime::Error update_prefill(
      const std::vector<executorch::runtime::EValue>& model_outputs) {
    return update_prefill(model_outputs, example::default_method_name_prefill);
  }

  /**
   * @brief Process and update internal state with outputs from the decode
   * phase.
   *
   * @param decode_method The decode method to update with outputs.
   * @param model_outputs Vector of outputs from the decode method execution.
   */
  ET_NODISCARD virtual executorch::runtime::Error update_decode(
      const std::vector<executorch::runtime::EValue>& model_outputs,
      const std::string& decode_method) override;

  /**
   * @brief Process outputs from the decode phase using the default method.
   *
   * Uses "forward" as the decode method.
   *
   * @param model_outputs Vector of outputs from the decode execution.
   * @return Error code.
   */
  ET_NODISCARD executorch::runtime::Error update_decode(
      const std::vector<executorch::runtime::EValue>& model_outputs) {
    return update_decode(model_outputs, example::default_method_name_decode);
  }

  size_t current_pos() const {
    return current_pos_;
  }

 private:
 void dump_buffer(
      const std::string& tag,
      size_t layer,
      const _Float16* data,
      size_t size);
  void dump_kv_input(const std::string& phase);
  void dump_kv_output(
      const std::string& phase,
      const std::vector<executorch::runtime::EValue>& model_outputs);

  void allocate_buffers();

  bool dump_enabled_ = false;
  mutable size_t dump_step_ = 0;
  std::string dump_dir_ = "/data/local/tmp/kv_cache_dump";

    /**
   * Copy prefill output KV cache (layout [1,H,prefill_cache_len,D]) into
   * the decode input buffer (layout [1,H,decode_cache_len,D]).
   *
   * Since the head stride differs between the two layouts, we copy per-head.
   */
  void copy_prefill_to_decode_input();

  /**
   * Copy new cache entries from output (layout [1,H,new_len,D]) into the
   * decode input buffer (layout [1,H,cache_len,D]) at the given position.
   *
   * For each head h:
   *   cache[h * cache_len * D + pos * D .. + (pos + new_len) * D]
   *     = src[h * new_len * D .. + new_len * D]
   */
  void copy_to_cache(const _Float16* src, size_t new_len,
      _Float16* cache, size_t cache_len, size_t pos);

  /**
   * Update the attention mask for a causal (lower-triangular) pattern.
   * For each query row r in [0, seq_len), the mask allows attending to
   * positions [0, pos + r] (value 1) and masks future positions (value 0).
   */
  void update_attn_mask(size_t pos, size_t seq_len);

  /**
   * @brief Reference to the Module used for method metadata and execution.
   */
  executorch::extension::module::Module& module_;
  Config config_;

  size_t prefill_cache_len_;
  size_t decode_cache_len_;
  size_t current_pos_ = 0;
  bool prefill_done_ = false;

  std::vector<int64_t> input_tokens_;
  std::vector<int64_t> pos_ids_;

  // Per-layer input buffers: [1, n_kv_heads, cache_len, head_dim] flattened
  std::vector<std::vector<_Float16>> prefill_k_input_;
  std::vector<std::vector<_Float16>> prefill_v_input_;
  std::vector<std::vector<_Float16>> decode_k_input_;
  std::vector<std::vector<_Float16>> decode_v_input_;

  // Attention mask buffer: [attn_mask_rows, attn_mask_cols] flattened
  std::vector<float> attn_mask_;
  executorch::extension::TensorPtr attn_mask_tensor_;

  // Tensor wrappers (kept alive between prepare and model execution)
  std::vector<executorch::extension::TensorPtr> prefill_k_input_tensors_;
  std::vector<executorch::extension::TensorPtr> prefill_v_input_tensors_;
  std::vector<executorch::extension::TensorPtr> prefill_k_output_tensors_;
  std::vector<executorch::extension::TensorPtr> prefill_v_output_tensors_;
  std::vector<executorch::extension::TensorPtr> decode_k_input_tensors_;
  std::vector<executorch::extension::TensorPtr> decode_v_input_tensors_;
  std::vector<executorch::extension::TensorPtr> decode_k_output_tensors_;
  std::vector<executorch::extension::TensorPtr> decode_v_output_tensors_;
};

} // namespace example
