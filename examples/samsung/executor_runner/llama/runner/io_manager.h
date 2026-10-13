/*
 * Copyright (c) 2025 Samsung Electronics Co. LTD
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <cstring>
#include <vector>

#include <executorch/examples/samsung/executor_runner/llama/runner/constants.h>
#include <executorch/examples/samsung/executor_runner/llama/runner/rotary_embedding.h>
#include <executorch/extension/llm/runner/io_manager/io_manager.h>

namespace example {

namespace llm = ::executorch::extension::llm;
namespace runtime = ::executorch::runtime;
namespace aten = ::executorch::aten;
namespace extension = ::executorch::extension;

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
class ET_EXPERIMENTAL LlmIOManager : public llm::IOManager {
 public:
  struct Config {
    size_t max_seq_len;
    size_t max_context_len;
    size_t prefill_seq_len;
    size_t decode_seq_len;
    size_t n_layers;
    size_t n_heads;
    size_t n_kv_heads;
    size_t head_dim;
    size_t hidden_dim;
    size_t vocab_size;
    size_t attn_mask_rows = 128;
    size_t attn_mask_cols = 1024;
    std::optional<size_t> sliding_window;
    EmbeddingStrategy embedding_strategy = EmbeddingStrategy::kGlobalOnly;
    MaskingStrategy masking_strategy = MaskingStrategy::kCausal;
  };

  LlmIOManager(extension::Module& module, Config config);

  const Config& get_config() const {
    return config_;
  }

  ET_NODISCARD virtual runtime::Error load_rotary_embedding(
      std::optional<const std::string> pos_embedding_path);

  size_t get_attention_setup_tensor_count() const {
    constexpr size_t kGlobalMaskCount = 1;
    constexpr size_t kGlobalRopeCosCount = 1;
    constexpr size_t kGlobalRopeSinCount = 1;
    constexpr size_t kBaselineMetadataCount =
        kGlobalMaskCount + kGlobalRopeCosCount + kGlobalRopeSinCount;

    size_t setup_tensor_count = kBaselineMetadataCount;

    const bool requires_local_mask =
        (config_.masking_strategy == MaskingStrategy::kCausalLocal ||
         config_.masking_strategy == MaskingStrategy::kHybridConvAttention);
    if (requires_local_mask) {
      constexpr size_t kLocalMaskCount = 1;
      setup_tensor_count += kLocalMaskCount;
    }

    const bool requires_local_rope =
        (config_.embedding_strategy == EmbeddingStrategy::kGlobalAndLocal);
    if (requires_local_rope) {
      constexpr size_t kLocalRopeCosCount = 1;
      constexpr size_t kLocalRopeSinCount = 1;
      setup_tensor_count += (kLocalRopeCosCount + kLocalRopeSinCount);
    }

    return setup_tensor_count;
  }

  /**
   * Calculates the total number of tensor slots required in the model's input
   * buffer.
   *
   * Memory Layout Layout:
   * [0]                  : Token/Prompt IDs
   * [1 ... N]            : Attention Setup Overheads (Masks, RoPE Sin/Cos
   * Tensors) [N+1 ... End]        : Alternating Key and Value Cache Tensors for
   * all Layers
   */
  size_t get_input_reserve_size() const {
    constexpr size_t kTokenInputCount = 1;
    constexpr size_t kKVSlotsPerLayer = 2;

    return kTokenInputCount + get_attention_setup_tensor_count() +
        (kKVSlotsPerLayer * config_.n_layers);
  }

  virtual ~LlmIOManager() = default;

  ET_NODISCARD virtual runtime::Error load(
      const std::string& prefill_method,
      const std::string& decode_method) override {
    (void)prefill_method;
    (void)decode_method;

    if ((config_.masking_strategy == MaskingStrategy::kCausalLocal ||
         config_.masking_strategy == MaskingStrategy::kHybridConvAttention) &&
        !config_.sliding_window.has_value()) {
      ET_LOG(
          Error,
          "Sliding window configuration is required for the selected masking strategy");
      return runtime::Error::InvalidArgument;
    }

    ET_LOG(
        Info,
        "(%s) Model Config: layers(%zu), kv_heads(%zu), hidden_dim(%zu), head_dim(%zu), \
          vocab(%zu), max_seq(%zu), max_ctx(%zu), prefill_seq(%zu), decode_seq(%zu), \
          mask_rows(%zu), mask_cols(%zu), slide_win(%zu), emb_strat(%d), mask_strat(%d)",
        ET_FUNCTION,
        config_.n_layers,
        config_.n_kv_heads,
        config_.hidden_dim,
        config_.head_dim,
        config_.vocab_size,
        config_.max_seq_len,
        config_.max_context_len,
        config_.prefill_seq_len,
        config_.decode_seq_len,
        config_.attn_mask_rows,
        config_.attn_mask_cols,
        config_.sliding_window.value_or(0),
        static_cast<int>(config_.embedding_strategy),
        static_cast<int>(config_.masking_strategy));

    return runtime::Error::Ok;
  }

  /**
   * @brief Reset the IO manager state.
   *
   * @param prefill_method The prefill method to reset with.
   * @param decode_method The decode method to reset with.
   */
  ET_NODISCARD virtual runtime::Error reset(
      const std::string& prefill_method,
      const std::string& decode_method) override {
    (void)prefill_method;
    (void)decode_method;

    dump_step_ = 0;

    current_pos_ = 0;
    prefill_done_ = false;

    for (size_t l = 0; l < config_.n_layers; l++) {
      std::fill(prefill_k_input_[l].begin(), prefill_k_input_[l].end(), 0);
      std::fill(prefill_v_input_[l].begin(), prefill_v_input_[l].end(), 0);
      std::fill(decode_k_input_[l].begin(), decode_k_input_[l].end(), 0);
      std::fill(decode_v_input_[l].begin(), decode_v_input_[l].end(), 0);
    }
    std::fill(attn_mask_.begin(), attn_mask_.end(), 0.0f);

    return runtime::Error::Ok;
  }

  /**
   * @brief Prepare inputs for the prefill phase of LLM inference.
   *
   * @param input The input tensor containing token IDs.
   * @param start_pos The tensor containing the starting position of the current
   * input within the context.
   * @param prefill_method The prefill method to prepare inputs for.
   * @return std::vector<runtime::EValue> Vector of prepared inputs
   * for the prefill method.
   */
  virtual runtime::Result<std::vector<runtime::EValue>> prepare_prefill(
      const extension::TensorPtr& input,
      const extension::TensorPtr& start_pos,
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
  runtime::Result<std::vector<runtime::EValue>> prepare_prefill(
      const extension::TensorPtr& input,
      const extension::TensorPtr& start_pos) {
    return prepare_prefill(
        input, start_pos, example::default_method_name_prefill);
  }

  /**
   * @brief Prepare inputs for the decode phase of LLM inference.
   *
   * @param input The input tensor containing token IDs.
   * @param start_pos The tensor containing the starting position of the current
   * input within the context.
   * @param decode_method The decode method to prepare inputs for.
   * @return std::vector<runtime::EValue> Vector of prepared inputs
   * for the decode method.
   */
  virtual runtime::Result<std::vector<runtime::EValue>> prepare_decode(
      const extension::TensorPtr& input,
      const extension::TensorPtr& start_pos,
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
  runtime::Result<std::vector<runtime::EValue>> prepare_decode(
      const extension::TensorPtr& input,
      const extension::TensorPtr& start_pos) {
    return prepare_decode(
        input, start_pos, example::default_method_name_decode);
  }

  /**
   * @brief Process and update internal state with outputs from the prefill
   * phase.
   *
   * @param prefill_method The prefill method to update with outputs.
   * @param model_outputs Vector of outputs from the prefill method execution.
   */
  ET_NODISCARD virtual runtime::Error update_prefill(
      const std::vector<runtime::EValue>& model_outputs,
      const std::string& prefill_method) override;

  /**
   * @brief Process outputs from the prefill phase using the default method.
   *
   * Uses "forward" as the prefill method.
   *
   * @param model_outputs Vector of outputs from the prefill execution.
   * @return Error code.
   */
  ET_NODISCARD runtime::Error update_prefill(
      const std::vector<runtime::EValue>& model_outputs) {
    return update_prefill(model_outputs, example::default_method_name_prefill);
  }

  /**
   * @brief Process and update internal state with outputs from the decode
   * phase.
   *
   * @param decode_method The decode method to update with outputs.
   * @param model_outputs Vector of outputs from the decode method execution.
   */
  ET_NODISCARD virtual runtime::Error update_decode(
      const std::vector<runtime::EValue>& model_outputs,
      const std::string& decode_method) override;

  /**
   * @brief Process outputs from the decode phase using the default method.
   *
   * Uses "forward" as the decode method.
   *
   * @param model_outputs Vector of outputs from the decode execution.
   * @return Error code.
   */
  ET_NODISCARD runtime::Error update_decode(
      const std::vector<runtime::EValue>& model_outputs) {
    return update_decode(model_outputs, example::default_method_name_decode);
  }

  size_t current_pos() const {
    return current_pos_;
  }

 private:
  void dump_buffer(
      const std::string& tag,
      size_t layer,
      const char* data,
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
   * Copy model KV-cache outputs into the given buffers and dump them.
   *
   * @param model_outputs  [logits, k0..k_{L-1}, v0..v_{L-1}]
   * @param k_bufs  per-layer K cache buffers (input/output)
   * @param v_bufs  per-layer V cache buffers (input/output)
   * @param cache_len  length of each cache buffer
   * @param new_len  number of new tokens to copy (prefill_seq_len or 1)
   * @param phase  dump label ("prefill" or "decode")
   */
  void update_kv_cache(
      const std::vector<executorch::runtime::EValue>& model_outputs,
      std::vector<std::vector<char>>& k_bufs,
      std::vector<std::vector<char>>& v_bufs,
      size_t cache_len,
      size_t new_len,
      const std::string& phase);

  /**
   * Copy new cache entries from source data (layout [1,H,new_len,D]) into the
   * cache buffer (layout [1,H,cache_len,D]) at the given position.
   *
   * For each head h:
   *   cache[h * cache_len * D + pos * D .. + (pos + new_len) * D]
   *     = src[h * new_len * D .. + new_len * D]
   */
  template <typename T>
  void copy_to_cache(
      const T* src,
      size_t new_len,
      char* cache,
      size_t cache_len,
      size_t pos) {
    for (size_t h = 0; h < config_.n_kv_heads; h++) {
      const T* src_head = src + h * new_len * config_.head_dim;
      char* dst_head = cache + h * cache_len * config_.head_dim * sizeof(T) +
          pos * config_.head_dim * sizeof(T);
      std::memcpy(dst_head, src_head, new_len * config_.head_dim * sizeof(T));
    }
  }

  /**
   * Copy tensor data to cache buffer, dispatching based on kv_scalar_type.
   * Handles both Half (float16) and Float (float32) types.
   */
  void copy_tensor_to_cache(
      const executorch::aten::Tensor& tensor,
      size_t new_len,
      char* cache,
      size_t cache_len,
      size_t pos) {
    const float* data = tensor.const_data_ptr<float>();
    copy_to_cache<float>(data, new_len, cache, cache_len, pos);
  }

  /**
   * Append KV-cache input tensors for all layers.
   *
   * For each layer l, creates k and v tensors from the given buffers
   * (shape [1, n_kv_heads, cache_len, head_dim]) and appends them
   * to both the per-phase tensor wrappers and the model inputs vector.
   */
  void append_kv_input_tensors(
      std::vector<std::vector<char>>& k_bufs,
      std::vector<std::vector<char>>& v_bufs,
      std::vector<executorch::extension::TensorPtr>& k_tensors,
      std::vector<executorch::extension::TensorPtr>& v_tensors,
      executorch::aten::ScalarType kv_scalar_type,
      size_t cache_len,
      std::vector<executorch::runtime::EValue>& inputs) const;

  /**
   * Update the attention mask for a causal (lower-triangular) pattern.
   * For each query row r in [0, seq_len), the mask allows attending to
   * positions [0, pos + r] (value 1) and masks future positions (value 0).
   */
  void update_attn_mask(size_t pos, size_t seq_len);

  /**
   * @brief Prepare rotary embedding inputs for quantized models.
   *
   * This helper function computes and adds the rotary embedding tensors
   * (global_cos, global_sin, local_cos, local_sin) to the inputs vector.
   * It is used by both prepare_prefill and prepare_decode to avoid code
   * duplication.
   *
   * @param seq_len The sequence length to use for the embedding computation.
   * @param inputs Vector of EValue inputs to append the embedding tensors to.
   */
  void prepare_rotary_embedding_inputs(
      size_t seq_len,
      std::vector<executorch::runtime::EValue>& inputs);

  /**
   * Update the local_attention mask for a causal (lower-triangular) pattern.
   * For each query row r in [0, seq_len), the mask allows attending to
   * positions [0, pos + r] (value 1) and masks future positions (value 0).
   */
  void update_local_attn_mask(size_t pos, size_t seq_len);
  Config config_;

  size_t prefill_cache_len_;
  size_t decode_cache_len_;
  size_t current_pos_ = 0;
  bool prefill_done_ = false;

  std::vector<int64_t> input_tokens_;
  executorch::extension::TensorPtr input_tokens_tensor_;
  std::vector<int64_t> pos_ids_;

  // Per-layer input buffers: [1, n_kv_heads, cache_len, head_dim] flattened
  // Use std::vector<char> to support both float and half (float16) data types
  std::vector<std::vector<char>> prefill_k_input_;
  std::vector<std::vector<char>> prefill_v_input_;
  std::vector<std::vector<char>> decode_k_input_;
  std::vector<std::vector<char>> decode_v_input_;

  std::unique_ptr<example::RotaryEmbedding> rotary_embedding_;

  std::vector<float> global_cos_data_;
  std::vector<float> global_sin_data_;
  std::vector<float> local_cos_data_;
  std::vector<float> local_sin_data_;

  executorch::extension::TensorPtr global_cos_tensor_;
  executorch::extension::TensorPtr global_sin_tensor_;
  executorch::extension::TensorPtr local_cos_tensor_;
  executorch::extension::TensorPtr local_sin_tensor_;

  // Attention mask buffer: [attn_mask_rows, attn_mask_cols] flattened
  std::vector<float> attn_mask_;
  std::vector<float> local_attn_mask_;
  executorch::extension::TensorPtr attn_mask_tensor_;
  executorch::extension::TensorPtr local_attn_mask_tensor_;

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
