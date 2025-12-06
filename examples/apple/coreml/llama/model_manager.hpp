/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <executorch/extension/module/module.h>
#include <executorch/runtime/core/error.h>
#include <executorch/runtime/core/evalue.h>
#include <executorch/runtime/executor/method_meta.h>

#include <cstdint>
#include <filesystem>
#include <memory>
#include <string>
#include <vector>

namespace executorch {
namespace extension {

/**
 * Manages loading and execution of a piecewise LLaMA model consisting of:
 * - input_block.pte
 * - transformer_block_0.pte, transformer_block_1.pte, ..., transformer_block_{n_layers-1}.pte
 * - output_block.pte
 *
 * The C++ version automatically infers n_layers from the number of
 * transformer_block_*.pte files present in the model directory.
 *
 * The ModelManager internally manages KV caches and provides a simple
 * forward() API similar to CPU models.
 */
class ModelManager {
 public:
  /**
   * Constructs a ModelManager by loading all model pieces from the
   * specified directory.
   *
   * @param model_path Path to directory containing the piecewise model files
   *                   (input_block.pte, transformer_block.pte, output_block.pte)
   *
   * The constructor automatically:
   * - Loads the single transformer_block.pte containing all layers
   * - Infers n_layers from the transformer block's output count
   * - Extracts metadata to determine: max_batch_size, n_kv_heads, cache_size,
   *   head_dim, seq_length, max_seq_length, and float_dtype
   * - Allocates and initializes KV caches
   */
  explicit ModelManager(const std::string& model_path);

  // Prevent copying and moving
  ModelManager(const ModelManager&) = delete;
  ModelManager& operator=(const ModelManager&) = delete;
  ModelManager(ModelManager&&) = delete;
  ModelManager& operator=(ModelManager&&) = delete;

  ~ModelManager() = default;

  /**
   * Runs inference on the model with the given tokens at the specified position.
   *
   * This method handles tokens of any size by:
   * - Padding to seq_length if tokens < seq_length
   * - Processing in seq_length chunks if tokens > seq_length
   *
   * @param tokens Input token IDs tensor, shape [1, num_tokens]
   * @param input_pos Position tensor, shape [1] indicating where these tokens start
   *
   * @return Result containing the logits tensor (shape: [batch_size, vocab_size])
   */
  runtime::Result<executorch::aten::Tensor> forward(
      const executorch::aten::Tensor& tokens,
      const executorch::aten::Tensor& input_pos);

  /**
   * Resets the KV caches to zero. Useful for starting a new sequence.
   */
  void reset_caches();

  // Accessors for model metadata
  int64_t get_n_layers() const { return n_layers_; }
  int64_t get_max_batch_size() const { return max_batch_size_; }
  int64_t get_n_kv_heads() const { return n_kv_heads_; }
  int64_t get_cache_size() const { return cache_size_; }
  int64_t get_head_dim() const { return head_dim_; }
  int64_t get_seq_length() const { return seq_length_; }
  int64_t get_max_seq_length() const { return max_seq_length_; }

 private:
  /**
   * Infers the number of layers by counting transformer_block_*.pte files
   * in the model directory.
   *
   * @param model_path Path to the model directory
   * @return Number of transformer blocks found
   */
  int64_t infer_n_layers(const std::string& model_path);

  /**
   * Loads the input projection block.
   *
   * @param model_path Path to the model directory
   * @return Error indicating success or failure
   */
  runtime::Error load_input_block(const std::string& model_path);

  /**
   * Loads all transformer blocks.
   *
   * @param model_path Path to the model directory
   * @return Error indicating success or failure
   */
  runtime::Error load_transformer_blocks(const std::string& model_path);

  /**
   * Loads the output projection block.
   *
   * @param model_path Path to the model directory
   * @return Error indicating success or failure
   */
  runtime::Error load_output_block(const std::string& model_path);

  /**
   * Extracts metadata from the first transformer block.
   *
   * @return Error indicating success or failure
   */
  runtime::Error extract_metadata();

  /**
   * Updates a single layer's KV caches with new cache values.
   * This matches the Python InputManager._update_cache() logic.
   *
   * @param layer_id Which layer to update
   * @param amount_to_copy Number of tokens to copy (pre-calculated)
   * @param new_k_cache New K cache tensor for this layer
   * @param new_v_cache New V cache tensor for this layer
   */
  void update_cache(
      int64_t layer_id,
      int64_t amount_to_copy,
      const executorch::aten::Tensor& new_k_cache,
      const executorch::aten::Tensor& new_v_cache);

  /**
   * Updates the attention mask for the current input position.
   * This matches the Python InputManager.update() mask update logic.
   *
   * @param input_pos Current input position
   * @param amount_to_copy Number of tokens to update in the mask
   */
  void update_mask(int64_t input_pos, int64_t amount_to_copy);

  /**
   * Private core inference method that requires tokens to be exactly seq_length.
   * This is the actual forward pass implementation - the public forward() method
   * handles chunking/padding and calls this.
   *
   * @param tokens Input token IDs tensor, shape [1, seq_length] (must be exact)
   * @param input_pos Position tensor, shape [1]
   * @param num_tokens Actual number of valid (non-padding) tokens in the input
   *
   * @return Result containing the logits tensor (shape: [batch_size, vocab_size])
   */
  runtime::Result<executorch::aten::Tensor> forward_(
      const executorch::aten::Tensor& tokens,
      const executorch::aten::Tensor& input_pos,
      int64_t num_tokens);

  // Model pieces
  std::unique_ptr<executorch::extension::Module> input_proj_module_;
  std::unique_ptr<executorch::extension::Module> output_proj_module_;
  std::vector<std::unique_ptr<executorch::extension::Module>> transformer_modules_;

  // Model metadata extracted from the transformer block
  int64_t n_layers_;
  int64_t max_batch_size_;
  int64_t n_kv_heads_;
  int64_t cache_size_;
  int64_t head_dim_;
  int64_t seq_length_;
  int64_t max_seq_length_;

  // KV cache storage (managed internally)
  // KV cache data storage (one per layer): [max_batch_size, n_kv_heads, cache_size, head_dim]
  // Using uint8_t to store raw bytes (supports both FP16 and FP32)
  std::vector<std::vector<uint8_t>> k_caches_data_;
  std::vector<std::vector<uint8_t>> v_caches_data_;

  // KV cache tensor views (pre-created for each layer to avoid overhead in forward())
  std::vector<std::shared_ptr<executorch::aten::Tensor>> k_caches_;
  std::vector<std::shared_ptr<executorch::aten::Tensor>> v_caches_;

  // Attention mask data storage: [seq_length, max_seq_length]
  // Using uint8_t to store raw bytes (supports both FP16 and FP32)
  std::vector<uint8_t> mask_data_;

  // Attention mask tensor view (pre-created to avoid overhead in forward())
  std::shared_ptr<executorch::aten::Tensor> attn_mask_;

  // Cache position tracking (managed internally)
  int64_t cache_pos_;

  // Pre-allocated buffers for forward() chunking/padding (to avoid allocation overhead)
  std::vector<int64_t> chunk_buffer_;     // Size: seq_length
  std::vector<int64_t> pos_buffer_;       // Size: 1
  std::vector<int64_t> input_length_buffer_;  // Size: 1

  // Pre-allocated EValue vector for layer execution (to avoid repeated allocations)
  std::vector<executorch::runtime::EValue> layer_inputs_;
};

} // namespace extension
} // namespace executorch
