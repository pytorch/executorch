/*
 * Copyright (c) 2026 Samsung Electronics Co. LTD
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <executorch/extension/llm/runner/util.h>

#include <cstddef>
#include <memory>
#include <optional>
#include <string>
#include <unordered_map>
#include <vector>

namespace example {

// Exynos Runtime metadata key constants
inline constexpr auto kPrefillArLen = "get_prefill_ar_len";
inline constexpr auto kNumOfLayers = "get_n_layers";
inline constexpr auto kNumOfHiddenDim = "get_hidden_dim";
inline constexpr auto kNumOfHeads = "get_n_heads";
inline constexpr auto kNumOfKVHeads = "get_n_kv_heads";

inline constexpr auto default_method_name = "forward";
inline constexpr auto default_method_name_prefill = "prefill_forward";
inline constexpr auto default_method_name_decode = "decode_forward";

// Gemma 4 multimodal special tokens (matches the model's tokenizer).
inline constexpr int64_t kGemma4PadId = 0;
inline constexpr int64_t kGemma4BosId = 2;
inline constexpr int64_t kGemma4TurnStartId = 105; // <|turn|
inline constexpr int64_t kGemma4TurnEndId = 106; // <turn|>
inline constexpr int64_t kGemma4ImageTokenId = 258880;
inline constexpr int64_t kGemma4BoiTokenId = 255999; // <|boi|>
inline constexpr int64_t kGemma4EoiTokenId = 258882; // <|eoi|>

enum class DecoderModelVersion {
  kLlama3,
  kGemma3,
  kGemma4,
  kGeneral,
};

struct ModelVersionConfig {
  size_t position_rank;
  size_t attention_mask_rank;
  size_t kv_cache_rank;

  size_t attention_mask_tensor_index;
  size_t attention_mask_tensor_count;

  size_t position_tensor_index;
  size_t position_tensor_count;

  size_t k_cache_tensor_index;
  size_t v_cache_tensor_index;
  size_t kv_cache_tensor_count;

  size_t heads_dim_index;
  size_t head_size_dim_index;

  bool has_local_global_rope;
  bool has_partial_rope;
  bool has_shared_kv;

  size_t sliding_window;
};

inline ModelVersionConfig getModelVersionConfig(DecoderModelVersion version) {
  switch (version) {
    case DecoderModelVersion::kGemma3:
      return {
          .position_rank = 3,
          .attention_mask_rank = 3,
          .kv_cache_rank = 4,

          // Flat PTE:
          // 0      input_ids
          // 1-2    attention_setup
          // 3-4    masks
          // 5-6    position
          // 7-32   k_cache_0..25
          // 33-58  v_cache_0..25
          .attention_mask_tensor_index = 3,
          .attention_mask_tensor_count = 2,

          .position_tensor_index = 5,
          .position_tensor_count = 2,

          .k_cache_tensor_index = 7,
          .v_cache_tensor_index = 33,
          .kv_cache_tensor_count = 26,

          .heads_dim_index = 1,
          .head_size_dim_index = 3,

          .has_local_global_rope = true,
          .has_partial_rope = false,
          .has_shared_kv = false,
          .sliding_window = 512,
      };

    case DecoderModelVersion::kGemma4:
      return {
          .position_rank = 3,
          .attention_mask_rank = 4,
          .kv_cache_rank = 4,

          // Flat PTE:
          // 0      input_ids
          // 1      inputs_embeds
          // 2      mm_mask
          // 3      attn_mask_full
          // 4      attn_mask_sliding
          // 5      pos_emb_global_cos
          // 6      pos_emb_global_sin
          // 7      pos_emb_local_cos
          // 8      pos_emb_local_sin
          // 9-43   k_cache_0..34
          // 44-78  v_cache_0..34
          .attention_mask_tensor_index = 3,
          .attention_mask_tensor_count = 2,

          .position_tensor_index = 5,
          .position_tensor_count = 4,

          .k_cache_tensor_index = 9,
          .v_cache_tensor_index = 44,
          .kv_cache_tensor_count = 35,

          .heads_dim_index = 1,
          .head_size_dim_index = 3,

          .has_local_global_rope = true,
          .has_partial_rope = true,
          .has_shared_kv = true,
          .sliding_window = 512,
      };

    case DecoderModelVersion::kLlama3:
      return {
          .position_rank = 3,
          .attention_mask_rank = 4,
          .kv_cache_rank = 4,

          .attention_mask_tensor_index = 1,
          .attention_mask_tensor_count = 1,

          .position_tensor_index = 2,
          .position_tensor_count = 2,

          .k_cache_tensor_index = 4,
          .v_cache_tensor_index = 20,
          .kv_cache_tensor_count = 16,

          .heads_dim_index = 1,
          .head_size_dim_index = 3,

          .has_local_global_rope = false,
          .has_partial_rope = false,
          .has_shared_kv = false,
          .sliding_window = 0,
      };
    default:
      return {
          .position_rank = 3,
          .kv_cache_rank = 4,
          .has_local_global_rope = false,
          .has_partial_rope = false,
          .has_shared_kv = false,
          .sliding_window = 0,
      };
  }
}

enum class EmbeddingStrategy {
  kGlobalOnly,
  kGlobalAndLocal,
};

enum class MaskingStrategy {
  kCausal,
  kCausalLocal,
  kHybridConvAttention, // LIV-conv + GQA mask constraint
};

class RunnerPaths {
 public:
  std::string model_path;
  std::string tokenizer_path;
  std::optional<std::string> pos_embedding_path = std::nullopt;
  std::optional<std::string> data_path = std::nullopt;
  std::optional<std::string> vision_model_path = std::nullopt;
};

class BaseModelArgs {
 public:
  virtual ~BaseModelArgs() = default;
  virtual DecoderModelVersion get_version() const = 0;
  virtual int64_t get_hidden_dim() const {
    return 0;
  }
  virtual int64_t get_n_heads() const {
    return 0;
  }
  virtual int64_t get_n_kv_heads() const {
    return 0;
  }
  virtual int64_t get_n_layers() const {
    return 0;
  }
  virtual int64_t get_head_dim() const {
    return 0;
  }
  virtual int64_t get_global_head_dim() const {
    return 0;
  }
  virtual int64_t get_vocab_size() const {
    return 0;
  }
  virtual int64_t get_max_seq_len() const {
    return 128;
  }
  virtual int64_t get_max_context_len() const {
    return 128;
  }
  virtual int64_t get_prefill_seq_len() const {
    return 128;
  }
  virtual int64_t get_decode_seq_len() const {
    return 1;
  }
  virtual int64_t get_attn_mask_rows() const {
    return 128;
  }
  virtual int64_t get_attn_mask_cols() const {
    return 1024;
  }
  virtual std::optional<int64_t> get_sliding_window() const {
    return std::nullopt;
  }
  // Number of trailing layers that reuse (share) KV cache from earlier layers
  // and therefore do not maintain their own externally-managed cache.
  virtual int64_t get_num_kv_shared_layers() const {
    return 0;
  }
  // Per-layer head dimension. Defaults to a uniform head_dim; models with
  // mixed local/global attention (e.g. Gemma 4) override this.
  virtual int64_t get_head_dim_for_layer(int64_t layer_idx) const {
    (void)layer_idx;
    return get_head_dim();
  }
  // Layers that actually produce/consume an externally-managed KV cache
  // (the leading self-decoder layers). Shared layers are handled inside the
  // delegate and are not cached by the runner.
  int64_t get_num_cache_layers() const {
    return get_n_layers() - get_num_kv_shared_layers();
  }
};

class Llama3Args : public BaseModelArgs {
 public:
  DecoderModelVersion get_version() const override {
    return DecoderModelVersion::kLlama3;
  }
  int64_t hidden_dim = 0;
  int64_t n_heads = 0;
  int64_t n_kv_heads = 0;
  int64_t n_layers = 0;
  int64_t head_dim = 0;
  int64_t vocab_size = 0;
  int64_t max_seq_len = 128;
  int64_t max_context_len = 128;
  int64_t prefill_seq_len = 128;
  int64_t decode_seq_len = 1;
  int64_t attn_mask_rows = 128;
  int64_t attn_mask_cols = 1024;
  std::optional<int64_t> sliding_window;

  int64_t get_hidden_dim() const override {
    return hidden_dim;
  }
  int64_t get_n_heads() const override {
    return n_heads;
  }
  int64_t get_n_kv_heads() const override {
    return n_kv_heads;
  }
  int64_t get_n_layers() const override {
    return n_layers;
  }
  int64_t get_head_dim() const override {
    return head_dim;
  }
  int64_t get_global_head_dim() const override {
    return head_dim;
  }
  int64_t get_vocab_size() const override {
    return vocab_size;
  }
  int64_t get_max_seq_len() const override {
    return max_seq_len;
  }
  int64_t get_max_context_len() const override {
    return max_context_len;
  }
  int64_t get_prefill_seq_len() const override {
    return prefill_seq_len;
  }
  int64_t get_decode_seq_len() const override {
    return decode_seq_len;
  }
  int64_t get_attn_mask_rows() const override {
    return attn_mask_rows;
  }
  int64_t get_attn_mask_cols() const override {
    return attn_mask_cols;
  }
  std::optional<int64_t> get_sliding_window() const override {
    return sliding_window;
  }
};

class Gemma3Args : public BaseModelArgs {
 public:
  DecoderModelVersion get_version() const override {
    return DecoderModelVersion::kGemma3;
  }
  int64_t hidden_dim = 0;
  int64_t n_heads = 0;
  int64_t n_kv_heads = 0;
  int64_t n_layers = 0;
  int64_t head_dim = 0;
  int64_t vocab_size = 0;
  int64_t max_seq_len = 128;
  int64_t max_context_len = 128;
  int64_t prefill_seq_len = 128;
  int64_t decode_seq_len = 1;
  int64_t attn_mask_rows = 128;
  int64_t attn_mask_cols = 1024;
  std::optional<int64_t> sliding_window;

  int64_t get_hidden_dim() const override {
    return hidden_dim;
  }
  int64_t get_n_heads() const override {
    return n_heads;
  }
  int64_t get_n_kv_heads() const override {
    return n_kv_heads;
  }
  int64_t get_n_layers() const override {
    return n_layers;
  }
  int64_t get_head_dim() const override {
    return head_dim;
  }
  int64_t get_global_head_dim() const override {
    return head_dim;
  }
  int64_t get_vocab_size() const override {
    return vocab_size;
  }
  int64_t get_max_seq_len() const override {
    return max_seq_len;
  }
  int64_t get_max_context_len() const override {
    return max_context_len;
  }
  int64_t get_prefill_seq_len() const override {
    return prefill_seq_len;
  }
  int64_t get_decode_seq_len() const override {
    return decode_seq_len;
  }
  int64_t get_attn_mask_rows() const override {
    return attn_mask_rows;
  }
  int64_t get_attn_mask_cols() const override {
    return attn_mask_cols;
  }
  std::optional<int64_t> get_sliding_window() const override {
    return sliding_window;
  }
};

class Gemma4Args : public BaseModelArgs {
 public:
  DecoderModelVersion get_version() const override {
    return DecoderModelVersion::kGemma4;
  }

  struct TextConfig {
    int64_t hidden_size = 0;
    int64_t intermediate_size = 0;
    bool use_double_wide_mlp = false;
    int64_t num_hidden_layers = 0;
    int64_t num_attention_heads = 0;
    int64_t head_dim = 0;
    int64_t global_head_dim = 0;
    int64_t num_key_value_heads = 0;
    int64_t vocab_size = 0;
    int64_t max_seq_len = 0;
    int64_t max_position_embeddings = 0;
    std::optional<int64_t> sliding_window;

    // RoPE specific configuration for Gemma 4
    float rope_percentage = 0.25f;
    float base_freq_local = 10000.0f;
    float base_freq_global = 1000000.0f;
  } text_config;

  struct AudioConfig {
    int64_t hidden_size = 0;
    int64_t num_hidden_layers = 0;
    int64_t num_attention_heads = 0;
  } audio_config;

  int64_t audio_seq_length = 0;
  int64_t attn_mask_rows = 128;
  int64_t attn_mask_cols = 1024;

  // YOCO KV sharing: the last num_kv_shared_layers reuse earlier layers' KV.
  int64_t num_kv_shared_layers = 0;
  // Attention pattern period (every pattern-th layer is full_attention).
  int64_t sliding_window_pattern = 5;
  // Parsed from "layer_types"; true where the layer uses full (global)
  // attention. Empty falls back to the sliding_window_pattern derivation.
  std::vector<bool> is_full_attention;

  int64_t get_hidden_dim() const override {
    return text_config.hidden_size;
  }
  int64_t get_n_heads() const override {
    return text_config.num_attention_heads;
  }
  int64_t get_n_kv_heads() const override {
    return text_config.num_key_value_heads;
  }
  int64_t get_n_layers() const override {
    return text_config.num_hidden_layers;
  }
  int64_t get_head_dim() const override {
    return text_config.head_dim;
  }
  int64_t get_global_head_dim() const override {
    return text_config.global_head_dim;
  }
  int64_t get_vocab_size() const override {
    return text_config.vocab_size;
  }
  int64_t get_max_seq_len() const override {
    return text_config.max_seq_len;
  }
  int64_t get_max_context_len() const override {
    return text_config.max_position_embeddings;
  }
  std::optional<int64_t> get_sliding_window() const override {
    return text_config.sliding_window;
  }
  // Attention mask spans [prefill_seq_len, max_seq_len] (rows x cols).
  int64_t get_attn_mask_rows() const override {
    return get_prefill_seq_len();
  }
  int64_t get_attn_mask_cols() const override {
    return text_config.max_seq_len;
  }
  int64_t get_num_kv_shared_layers() const override {
    return num_kv_shared_layers;
  }
  int64_t get_head_dim_for_layer(int64_t layer_idx) const override {
    bool full;
    if (layer_idx >= 0 &&
        layer_idx < static_cast<int64_t>(is_full_attention.size())) {
      full = is_full_attention[layer_idx];
    } else {
      full = ((layer_idx + 1) % sliding_window_pattern == 0);
    }
    return full ? text_config.global_head_dim : text_config.head_dim;
  }
};

class RunnerConfig {
 public:
  std::string config_path;
  bool is_decode_only = false;
  float temperature = -1.0f;

  // Stream generated tokens to stdout (in addition to the token callback).
  bool verbose = false;

  // Empty => no system prompt.
  std::string system_prompt;

  // Text appended verbatim to the very end of the rendered prompt (after the
  // assistant turn header), i.e. a pre-filled assistant response. Ignored in
  // raw_prompt mode where the caller supplies the full prompt already.
  std::string assistant_prefill_prompt;

  // Treat the prompt as already fully formatted: skip chat-template wrapping
  // and system_prompt so a caller can feed raw special-token input verbatim.
  bool raw_prompt = false;

  // Vision inputs (Gemma 4 multimodal). Empty image_path => text-only.
  std::string image_path;
  std::string prompt_before_image;

  // Strategies
  EmbeddingStrategy embedding_strategy = EmbeddingStrategy::kGlobalOnly;
  MaskingStrategy masking_strategy = MaskingStrategy::kCausal;

  std::string tokenizer_type;

  DecoderModelVersion model_version;
  std::unique_ptr<BaseModelArgs> model_args;

  void load_from_json();
};

} // namespace example