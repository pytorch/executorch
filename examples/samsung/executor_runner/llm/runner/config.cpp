/*
 * Copyright (c) 2025 Samsung Electronics Co. LTD
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <executorch/examples/samsung/executor_runner/llm/runner/config.h>
#include <executorch/runtime/platform/runtime.h>
#include <nlohmann/json.hpp>
#include <fstream>

#include <executorch/runtime/platform/runtime.h>

namespace example {

using json = nlohmann::json;

void RunnerConfig::load_from_json() {
  if (config_path.empty()) {
    ET_LOG(Info, "Config path is empty, skipping JSON load");
    return;
  }

  std::ifstream file(config_path);
  if (!file.is_open()) {
    ET_LOG(Error, "Could not open config file: %s", config_path.c_str());
    return;
  }

  json j = json::parse(file, nullptr, false);
  if (j.is_discarded()) {
    ET_LOG(
        Error, "Invalid JSON format in config file: %s", config_path.c_str());
    return;
  }

  if (j.contains("embedding_strategy") && j["embedding_strategy"].is_string()) {
    std::string strategy = j["embedding_strategy"].get<std::string>();
    if (strategy == "global_only") {
      embedding_strategy = EmbeddingStrategy::kGlobalOnly;
    } else if (strategy == "global_and_local") {
      embedding_strategy = EmbeddingStrategy::kGlobalAndLocal;
    }
  }

  if (j.contains("masking_strategy") && j["masking_strategy"].is_string()) {
    std::string strategy = j["masking_strategy"].get<std::string>();
    if (strategy == "causal") {
      masking_strategy = MaskingStrategy::kCausal;
    } else if (strategy == "causal_local") {
      masking_strategy = MaskingStrategy::kCausalLocal;
    } else if (strategy == "hybrid_conv") {
      masking_strategy = MaskingStrategy::kHybridConvAttention;
    }
  }

  if (j.contains("model_version") && j["model_version"].is_string()) {
    std::string version_str = j["model_version"].get<std::string>();
    if (version_str == "llama3") {
      model_version = DecoderModelVersion::kLlama3;
    } else if (version_str == "gemma3") {
      model_version = DecoderModelVersion::kGemma3;
    } else if (version_str == "gemma4") {
      model_version = DecoderModelVersion::kGemma4;
    } else if (version_str == "general") {
      model_version = DecoderModelVersion::kGeneral;
    }
  }

  if (j.contains("tokenizer_type") && j["tokenizer_type"].is_string()) {
    tokenizer_type = j["tokenizer_type"].get<std::string>();
  }

  // Use text_config if present, otherwise fallback to root
  const json* config_source = &j;
  if (j.contains("text_config") && j["text_config"].is_object()) {
    config_source = &j["text_config"];
  }
  const auto& src = *config_source;

  switch (model_version) {
    case DecoderModelVersion::kGemma4: {
      auto args = std::make_unique<Gemma4Args>();
      if (src.contains("hidden_size") &&
          src["hidden_size"].is_number_integer()) {
        args->text_config.hidden_size = src["hidden_size"].get<int64_t>();
      }
      if (src.contains("num_hidden_layers") &&
          src["num_hidden_layers"].is_number_integer()) {
        args->text_config.num_hidden_layers =
            src["num_hidden_layers"].get<int64_t>();
      }
      if (src.contains("num_attention_heads") &&
          src["num_attention_heads"].is_number_integer()) {
        args->text_config.num_attention_heads =
            src["num_attention_heads"].get<int64_t>();
      }
      if (src.contains("num_key_value_heads") &&
          src["num_key_value_heads"].is_number_integer()) {
        args->text_config.num_key_value_heads =
            src["num_key_value_heads"].get<int64_t>();
      }
      if (src.contains("head_dim") && src["head_dim"].is_number_integer()) {
        args->text_config.head_dim = src["head_dim"].get<int64_t>();
      }
      if (src.contains("global_head_dim") &&
          src["global_head_dim"].is_number_integer()) {
        args->text_config.global_head_dim =
            src["global_head_dim"].get<int64_t>();
      }
      if (src.contains("partial_rotary_factor") &&
          src["partial_rotary_factor"].is_number()) {
        args->text_config.rope_percentage =
            src["partial_rotary_factor"].get<float>();
      }
      if (src.contains("rope_theta") && src["rope_theta"].is_number()) {
        args->text_config.base_freq_global = src["rope_theta"].get<float>();
      }
      if (src.contains("rope_local_base_freq") &&
          src["rope_local_base_freq"].is_number()) {
        args->text_config.base_freq_local =
            src["rope_local_base_freq"].get<float>();
      }
      if (src.contains("vocab_size") && src["vocab_size"].is_number_integer()) {
        args->text_config.vocab_size = src["vocab_size"].get<int64_t>();
      }
      if (src.contains("max_seq_len") &&
          src["max_seq_len"].is_number_integer()) {
        args->text_config.max_seq_len = src["max_seq_len"].get<int64_t>();
      }
      if (src.contains("max_position_embeddings") &&
          src["max_position_embeddings"].is_number_integer()) {
        args->text_config.max_position_embeddings =
            src["max_position_embeddings"].get<int64_t>();
      }
      if (src.contains("sliding_window") &&
          src["sliding_window"].is_number_integer()) {
        args->text_config.sliding_window = src["sliding_window"].get<int64_t>();
      }
      model_args = std::move(args);
      break;
    }
    case DecoderModelVersion::kLlama3: {
      auto args = std::make_unique<Llama3Args>();
      if (src.contains("hidden_dim") && src["hidden_dim"].is_number_integer()) {
        args->hidden_dim = src["hidden_dim"].get<int64_t>();
      } else if (
          src.contains("hidden_size") &&
          src["hidden_size"].is_number_integer()) {
        args->hidden_dim = src["hidden_size"].get<int64_t>();
      }
      if (src.contains("n_heads") && src["n_heads"].is_number_integer()) {
        args->n_heads = src["n_heads"].get<int64_t>();
      } else if (
          src.contains("num_attention_heads") &&
          src["num_attention_heads"].is_number_integer()) {
        args->n_heads = src["num_attention_heads"].get<int64_t>();
      }
      if (src.contains("n_kv_heads") && src["n_kv_heads"].is_number_integer()) {
        args->n_kv_heads = src["n_kv_heads"].get<int64_t>();
      } else if (
          src.contains("num_key_value_heads") &&
          src["num_key_value_heads"].is_number_integer()) {
        args->n_kv_heads = src["num_key_value_heads"].get<int64_t>();
      }
      if (src.contains("n_layers") && src["n_layers"].is_number_integer()) {
        args->n_layers = src["n_layers"].get<int64_t>();
      } else if (
          src.contains("num_hidden_layers") &&
          src["num_hidden_layers"].is_number_integer()) {
        args->n_layers = src["num_hidden_layers"].get<int64_t>();
      }
      if (src.contains("head_dim") && src["head_dim"].is_number_integer()) {
        args->head_dim = src["head_dim"].get<int64_t>();
      }
      if (src.contains("vocab_size") && src["vocab_size"].is_number_integer()) {
        args->vocab_size = src["vocab_size"].get<int64_t>();
      }
      if (src.contains("max_seq_len") &&
          src["max_seq_len"].is_number_integer()) {
        args->max_seq_len = src["max_seq_len"].get<int64_t>();
      } else if (
          src.contains("max_position_embeddings") &&
          src["max_position_embeddings"].is_number_integer()) {
        args->max_seq_len = src["max_position_embeddings"].get<int64_t>();
      }
      if (src.contains("max_context_len") &&
          src["max_context_len"].is_number_integer()) {
        args->max_context_len = src["max_context_len"].get<int64_t>();
      } else if (
          src.contains("max_position_embeddings") &&
          src["max_position_embeddings"].is_number_integer()) {
        args->max_context_len = src["max_position_embeddings"].get<int64_t>();
      }
      if (src.contains("prefill_seq_len") &&
          src["prefill_seq_len"].is_number_integer()) {
        args->prefill_seq_len = src["prefill_seq_len"].get<int64_t>();
      }
      if (src.contains("decode_seq_len") &&
          src["decode_seq_len"].is_number_integer()) {
        args->decode_seq_len = src["decode_seq_len"].get<int64_t>();
      }
      if (src.contains("attn_mask_rows") &&
          src["attn_mask_rows"].is_number_integer()) {
        args->attn_mask_rows = src["attn_mask_rows"].get<int64_t>();
      }
      if (src.contains("attn_mask_cols") &&
          src["attn_mask_cols"].is_number_integer()) {
        args->attn_mask_cols = src["attn_mask_cols"].get<int64_t>();
      }
      if (src.contains("sliding_window") &&
          src["sliding_window"].is_number_integer()) {
        args->sliding_window = src["sliding_window"].get<int64_t>();
      } else {
        args->sliding_window = std::nullopt;
      }
      model_args = std::move(args);
      break;
    }
    case DecoderModelVersion::kGemma3: {
      auto args = std::make_unique<Gemma3Args>();
      if (src.contains("hidden_dim") && src["hidden_dim"].is_number_integer()) {
        args->hidden_dim = src["hidden_dim"].get<int64_t>();
      } else if (
          src.contains("hidden_size") &&
          src["hidden_size"].is_number_integer()) {
        args->hidden_dim = src["hidden_size"].get<int64_t>();
      }

      if (src.contains("n_heads") && src["n_heads"].is_number_integer()) {
        args->n_heads = src["n_heads"].get<int64_t>();
      } else if (
          src.contains("num_attention_heads") &&
          src["num_attention_heads"].is_number_integer()) {
        args->n_heads = src["num_attention_heads"].get<int64_t>();
      }

      if (src.contains("n_kv_heads") && src["n_kv_heads"].is_number_integer()) {
        args->n_kv_heads = src["n_kv_heads"].get<int64_t>();
      } else if (
          src.contains("num_key_value_heads") &&
          src["num_key_value_heads"].is_number_integer()) {
        args->n_kv_heads = src["num_key_value_heads"].get<int64_t>();
      }

      if (src.contains("n_layers") && src["n_layers"].is_number_integer()) {
        args->n_layers = src["n_layers"].get<int64_t>();
      } else if (
          src.contains("num_hidden_layers") &&
          src["num_hidden_layers"].is_number_integer()) {
        args->n_layers = src["num_hidden_layers"].get<int64_t>();
      }

      if (src.contains("head_dim") && src["head_dim"].is_number_integer()) {
        args->head_dim = src["head_dim"].get<int64_t>();
      }

      if (src.contains("vocab_size") && src["vocab_size"].is_number_integer()) {
        args->vocab_size = src["vocab_size"].get<int64_t>();
      }

      if (src.contains("max_seq_len") &&
          src["max_seq_len"].is_number_integer()) {
        args->max_seq_len = src["max_seq_len"].get<int64_t>();
      } else if (
          src.contains("max_position_embeddings") &&
          src["max_position_embeddings"].is_number_integer()) {
        args->max_seq_len = src["max_position_embeddings"].get<int64_t>();
      }

      if (src.contains("max_context_len") &&
          src["max_context_len"].is_number_integer()) {
        args->max_context_len = src["max_context_len"].get<int64_t>();
      } else if (
          src.contains("max_position_embeddings") &&
          src["max_position_embeddings"].is_number_integer()) {
        args->max_context_len = src["max_position_embeddings"].get<int64_t>();
      }

      if (src.contains("prefill_seq_len") &&
          src["prefill_seq_len"].is_number_integer()) {
        args->prefill_seq_len = src["prefill_seq_len"].get<int64_t>();
      }
      if (src.contains("decode_seq_len") &&
          src["decode_seq_len"].is_number_integer()) {
        args->decode_seq_len = src["decode_seq_len"].get<int64_t>();
      }
      if (src.contains("attn_mask_rows") &&
          src["attn_mask_rows"].is_number_integer()) {
        args->attn_mask_rows = src["attn_mask_rows"].get<int64_t>();
      }
      if (src.contains("attn_mask_cols") &&
          src["attn_mask_cols"].is_number_integer()) {
        args->attn_mask_cols = src["attn_mask_cols"].get<int64_t>();
      }
      if (src.contains("sliding_window") &&
          src["sliding_window"].is_number_integer()) {
        args->sliding_window = src["sliding_window"].get<int64_t>();
      } else {
        args->sliding_window = std::nullopt;
      }
      model_args = std::move(args);
      break;
    }
    case DecoderModelVersion::kGeneral:
    default:
      break;
  }
}
} // namespace example