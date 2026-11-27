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

#include <executorch/examples/samsung/executor_runner/llm/runner/config.h>
#include <executorch/examples/samsung/executor_runner/llm/runner/rotary_embedding.h>
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
  LlmIOManager(extension::Module& module, const RunnerConfig& config);
  ~LlmIOManager() override = default;
  LlmIOManager(const LlmIOManager& other) = delete;
  LlmIOManager& operator=(const LlmIOManager& rhs) = delete;
  LlmIOManager(LlmIOManager&& other) = delete;
  LlmIOManager& operator=(const LlmIOManager&& rhs) = delete;

  ET_NODISCARD runtime::Error load(
      const std::string& prefill_method,
      const std::string& decode_method) override;

  ET_NODISCARD runtime::Error reset(
      const std::string& prefill_method,
      const std::string& decode_method) override;

  ET_NODISCARD virtual runtime::Error load_rotary_embedding(
      std::optional<const std::string> pos_embedding_path);

  virtual runtime::Result<std::vector<runtime::EValue>> prepare_prefill(
      const extension::TensorPtr& input,
      const extension::TensorPtr& start_pos,
      const std::string& prefill_method) override;

  runtime::Result<std::vector<runtime::EValue>> prepare_prefill(
      const extension::TensorPtr& input,
      const extension::TensorPtr& start_pos) {
    return prepare_prefill(
        input, start_pos, example::default_method_name_prefill);
  }

  runtime::Result<std::vector<runtime::EValue>> prepare_decode(
      const extension::TensorPtr& input,
      const extension::TensorPtr& start_pos,
      const std::string& decode_method) override;

  runtime::Result<std::vector<runtime::EValue>> prepare_decode(
      const extension::TensorPtr& input,
      const extension::TensorPtr& start_pos) {
    return prepare_decode(
        input, start_pos, example::default_method_name_decode);
  }

  ET_NODISCARD runtime::Error update_prefill(
      const std::vector<runtime::EValue>& model_outputs,
      const std::string& prefill_method) override;

  ET_NODISCARD runtime::Error update_prefill(
      const std::vector<runtime::EValue>& model_outputs) {
    return update_prefill(model_outputs, example::default_method_name_prefill);
  }

  ET_NODISCARD runtime::Error update_decode(
      const std::vector<runtime::EValue>& model_outputs,
      const std::string& decode_method) override;

  ET_NODISCARD runtime::Error update_decode(
      const std::vector<runtime::EValue>& model_outputs) {
    return update_decode(model_outputs, example::default_method_name_decode);
  }

  /**
   * Supply pre-computed vision embeddings for Gemma 4 multimodal inference.
   *
   * @param embeds Packed valid vision embeddings (num_vision_tokens * hidden,
   *               contiguous float). The caller must have dropped padding.
   * @param num_tokens Number of valid vision tokens (= embeds.size()/hidden).
   */
  void set_vision_embeddings(std::vector<float> embeds, int64_t num_tokens);

  size_t num_vision_embeds_consumed() const {
    return vision_embed_idx_;
  }

  const RunnerConfig& get_config() const {
    return config_;
  }

  size_t get_attention_setup_tensor_count() const;

  // [tokens, (Gemma 4: inputs_embeds, mm_mask), masks, RoPE cos/sin, K x L,
  //  V x L]
  size_t get_input_reserve_size() const {
    const size_t prelude =
        config_.model_version == DecoderModelVersion::kGemma4 ? 2 : 0;
    return 1 + prelude + get_attention_setup_tensor_count() +
        2 * static_cast<size_t>(config_.model_args->get_n_layers());
  }

  size_t current_pos() const {
    return current_pos_;
  }

 private:
  void allocate_buffers();

  /**
   * Copy prefill output KV cache (layout [1,H,prefill_cache_len,D]) into
   * the decode input buffer (layout [1,H,decode_cache_len,D]).
   *
   * Since the head stride differs between the two layouts, we copy per-head.
   */
  void copy_prefill_to_decode_input();

  /**
   * Copy model KV-cache outputs into the given buffers.
   *
   * @param model_outputs  [logits, k0..k_{L-1}, v0..v_{L-1}]
   * @param k_bufs  per-layer K cache buffers (input/output)
   * @param v_bufs  per-layer V cache buffers (input/output)
   * @param cache_len  length of each cache buffer
   * @param new_len  number of new tokens to copy (prefill_seq_len or 1)
   */
  ET_NODISCARD runtime::Error update_kv_cache(
      const std::vector<executorch::runtime::EValue>& model_outputs,
      std::vector<std::vector<char>>& k_bufs,
      std::vector<std::vector<char>>& v_bufs,
      size_t cache_len,
      size_t new_len);

  /**
   * Copy the first `rows` of new cache entries from source data (layout
   * [1,H,new_len,D]) into the cache buffer (layout [1,H,cache_len,D]) at pos.
   */
  template <typename T>
  void copy_to_cache(
      const T* src,
      size_t new_len,
      size_t rows,
      char* cache,
      size_t cache_len,
      size_t pos,
      size_t head_dim) {
    for (size_t h = 0; h < config_.model_args->get_n_kv_heads(); h++) {
      const T* src_head = src + h * new_len * head_dim;
      char* dst_head = cache + h * cache_len * head_dim * sizeof(T) +
          pos * head_dim * sizeof(T);
      std::memcpy(dst_head, src_head, rows * head_dim * sizeof(T));
    }
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
      std::vector<char>& dummy_buf,
      std::vector<executorch::runtime::EValue>& inputs) const;

  /**
   * Append the Gemma 4 text-decoder prelude inputs that follow the token ids:
   * inputs_embeds [1, seq_len, hidden_dim] and mm_mask [1, seq_len].
   *
   * For text-only inference both are zero-filled: the decoder embeds the token
   * ids internally and mm_mask == 0 selects those text embeddings everywhere,
   * so inputs_embeds (the multimodal injection path) is ignored.
   */
  void append_prelude_inputs(
      size_t seq_len,
      std::vector<executorch::runtime::EValue>& inputs);

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
  ET_NODISCARD runtime::Error prepare_rotary_embedding_inputs(
      size_t seq_len,
      std::vector<executorch::runtime::EValue>& inputs);

  /**
   * Update the local_attention mask for a causal (lower-triangular) pattern.
   * For each query row r in [0, seq_len), the mask allows attending to
   * positions [0, pos + r] (value 1) and masks future positions (value 0).
   */
  void update_local_attn_mask(size_t pos, size_t seq_len);
  const RunnerConfig& config_;

  size_t prefill_cache_len_;
  size_t decode_cache_len_;
  size_t current_pos_ = 0;
  bool prefill_done_ = false;

  std::vector<int64_t> input_tokens_;
  executorch::extension::TensorPtr input_tokens_tensor_;
  std::vector<int64_t> pos_ids_;

  // Gemma 4 prelude inputs (zero-filled for text-only inference).
  std::vector<float> inputs_embeds_data_;
  executorch::extension::TensorPtr inputs_embeds_tensor_;
  std::vector<float> mm_mask_data_;
  executorch::extension::TensorPtr mm_mask_tensor_;

  // Gemma 4 vision embeddings injected at image-token positions.
  // vision_embeds_ holds only the valid (non-padding) embeddings packed
  // contiguously (num_vision_tokens * hidden). append_prelude_inputs copies
  // them into inputs_embeds_data_ at the image-token slots and marks those
  // slots in mm_mask_data_.
  std::vector<float> vision_embeds_;
  int64_t num_vision_tokens_ = 0;
  // Cursor into vision_embeds_ consumed by append_prelude_inputs. Persists
  // across micro-batch prefill chunks so image tokens split across chunks map
  // to the correct embeddings. Reset on set_vision_embeddings.
  size_t vision_embed_idx_ = 0;

  // Zero-filled scratch feeding the KV input slots of KV-shared (cross-decoder)
  // layers. These slots are declared model inputs but never read by the graph,
  // so a single shared buffer (sized for the widest head_dim) backs them all.
  std::vector<char> prefill_dummy_kv_;
  std::vector<char> decode_dummy_kv_;
  mutable std::vector<executorch::extension::TensorPtr> dummy_kv_tensors_;

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
