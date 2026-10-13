/*
 * Copyright (c) 2026 Samsung Electronics Co. LTD
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <cassert>
#include <cmath>
#include <cstring>
#include <fstream>
#include <iostream>

#include <executorch/examples/samsung/executor_runner/llm/runner/rotary_embedding.h>

namespace example {

namespace runtime = ::executorch::runtime;

RotaryEmbedding::RotaryEmbedding(const RunnerConfig& config)
    : config_(config),
      loaded_(false),
      seq_len_(config.model_args->get_max_seq_len()),
      base_freq_global_(1000000.0f),
      base_freq_local_(1000000.0f),
      rope_percentage_(1.0f) {
  switch (config_.model_version) {
    case DecoderModelVersion::kGemma4: {
      auto* args = static_cast<Gemma4Args*>(config_.model_args.get());
      normalized_global_head_dim_ = args->text_config.global_head_dim;
      normalized_local_head_dim_ = args->text_config.head_dim;
      base_freq_global_ = args->text_config.base_freq_global;
      base_freq_local_ = args->text_config.base_freq_local;
      rope_percentage_ = args->text_config.rope_percentage;
      break;
    }
    case DecoderModelVersion::kLlama3: {
      auto* args = static_cast<Llama3Args*>(config_.model_args.get());
      normalized_global_head_dim_ = args->get_head_dim();
      normalized_local_head_dim_ = args->get_head_dim();
      base_freq_global_ = 500000.0f;
      rope_percentage_ = 1.0f;
      break;
    }
    case DecoderModelVersion::kGemma3: {
      auto* args = static_cast<Gemma3Args*>(config_.model_args.get());
      normalized_global_head_dim_ = args->get_head_dim();
      normalized_local_head_dim_ = args->get_head_dim();
      base_freq_global_ = 1000000.0f;
      base_freq_local_ = 10000.0f;
      rope_percentage_ = 1.0f;
      break;
    }
    default: {
      break;
    }
  }

  // TODO(brucekk.kim): size should be configurable
  global_bytes_ = config_.model_args->get_max_seq_len() *
      normalized_global_head_dim_ * sizeof(float);
  local_bytes_ = config_.model_args->get_max_seq_len() *
      normalized_local_head_dim_ * sizeof(float);
}

runtime::Error RotaryEmbedding::load(
    std::optional<const std::string> pos_embedding_path) {
  allocate_buffers();

  if (pos_embedding_path.has_value() && !pos_embedding_path->empty()) {
    runtime::Error err = load_from_file(pos_embedding_path.value());
    if (err == runtime::Error::Ok) {
      return runtime::Error::Ok;
    }
    ET_LOG(
        Info,
        "Failed to load embeddings from file, falling back to precompute.");
  } else {
    ET_LOG(
        Info,
        "No valid position embedding path provided, falling back to precompute.");
  }

  return load_by_precompute_rope_tables();
}

runtime::Error RotaryEmbedding::load_from_file(const std::string& path) {
  std::ifstream file(path, std::ios::binary);
  ET_CHECK_OR_RETURN_ERROR(
      file.is_open(),
      InvalidArgument,
      "File I/O error: Cannot open file at '%s'.",
      path.c_str());

  file.seekg(0, std::ios::end);
  size_t file_size = file.tellg();
  file.seekg(0, std::ios::beg);

  size_t expected_size = global_bytes_ * 2;
  if (config_.embedding_strategy == EmbeddingStrategy::kGlobalAndLocal) {
    expected_size += local_bytes_ * 2;
  }

  ET_CHECK_OR_RETURN_ERROR(
      file_size == expected_size,
      InvalidArgument,
      "Embedding file size mismatch. Expected %zu, got %zu.",
      expected_size,
      file_size);

  // =========================================================================
  // BINARY ASSET LAYOUT ASSUMPTION WARNING:
  // The raw .bin file contains NO metadata, headers, or structural markers.
  // This code relies entirely on a strict, sequential file-pointer progression.
  // Any discrepancy in the packing sequence or byte padding during model
  // export will cause silent memory corruption or misaligned frequencies.
  //
  // Expected File Byte Ordering Layout:
  // +-----------------------------+-----------------------------+
  // | global_cos (global_bytes_)  | global_sin (global_bytes_)  |
  // +-----------------------------+-----------------------------+
  // | local_cos  (local_bytes_)*  | local_sin  (local_bytes_)*  |
  // +-----------------------------+-----------------------------+
  // (*Optional: Only if embedding_strategy == kGlobalAndLocal)
  // =========================================================================

  // Stream out the mandatory global attention layer matrices
  file.read(global_cos_.get(), global_bytes_);
  file.read(global_sin_.get(), global_bytes_);
  if (config_.embedding_strategy == EmbeddingStrategy::kGlobalAndLocal) {
    file.read(local_cos_.get(), local_bytes_);
    file.read(local_sin_.get(), local_bytes_);
  }

  loaded_ = true;
  return runtime::Error::Ok;
}

runtime::Error RotaryEmbedding::load_by_precompute_rope_tables() {
  float* global_cos_ptr = reinterpret_cast<float*>(global_cos_.get());
  float* global_sin_ptr = reinterpret_cast<float*>(global_sin_.get());

  assert(
      reinterpret_cast<uintptr_t>(global_cos_ptr) % alignof(float) == 0 &&
      "global_cos_ is misaligned for float access!");
  assert(
      reinterpret_cast<uintptr_t>(global_sin_ptr) % alignof(float) == 0 &&
      "global_sin_ is misaligned for float access!");

  compute_rope_table(
      global_cos_ptr,
      global_sin_ptr,
      seq_len_,
      normalized_global_head_dim_,
      base_freq_global_,
      rope_percentage_);

  if (config_.embedding_strategy == EmbeddingStrategy::kGlobalAndLocal) {
    float* local_cos_ptr = reinterpret_cast<float*>(local_cos_.get());
    float* local_sin_ptr = reinterpret_cast<float*>(local_sin_.get());

    assert(
        reinterpret_cast<uintptr_t>(local_cos_ptr) % alignof(float) == 0 &&
        "local_cos_ is misaligned for float access!");
    assert(
        reinterpret_cast<uintptr_t>(local_sin_ptr) % alignof(float) == 0 &&
        "local_sin_ is misaligned for float access!");

    // Sliding/local layers use full RoPE; the partial rotary factor applies
    // only to the global (full-attention) layers.
    compute_rope_table(
        local_cos_ptr,
        local_sin_ptr,
        seq_len_,
        normalized_local_head_dim_,
        base_freq_local_,
        /*rope_pct=*/1.0f);
  }
  loaded_ = true;
  ET_LOG(Info, "Rotary Embedding tables precomputed successfully.");
  return runtime::Error::Ok;
}

void RotaryEmbedding::allocate_buffers() {
  global_cos_ = std::make_unique<char[]>(global_bytes_);
  global_sin_ = std::make_unique<char[]>(global_bytes_);

  if (config_.embedding_strategy == EmbeddingStrategy::kGlobalAndLocal) {
    local_cos_ = std::make_unique<char[]>(local_bytes_);
    local_sin_ = std::make_unique<char[]>(local_bytes_);
  }
}

runtime::Error RotaryEmbedding::compute_rope_table(
    float* cos_ptr,
    float* sin_ptr,
    size_t seq_len,
    size_t dim,
    float base,
    float rope_pct) {
  ET_CHECK_OR_RETURN_ERROR(
      cos_ptr != nullptr, InvalidArgument, "cos_ptr cannot be nullptr.");
  ET_CHECK_OR_RETURN_ERROR(
      sin_ptr != nullptr, InvalidArgument, "sin_ptr cannot be nullptr.");
  ET_CHECK_OR_RETURN_ERROR(
      dim % 2 == 0,
      InvalidArgument,
      "The head dimension (dim) must be an even number for rotary pairs.");
  ET_CHECK_OR_RETURN_ERROR(
      base > 0.0f,
      InvalidArgument,
      "The RoPE base frequency must be greater than zero.");
  ET_CHECK_OR_RETURN_ERROR(
      rope_pct >= 0.0f && rope_pct <= 1.0f,
      InvalidArgument,
      "The rope_pct parameter must be a percentage value between 0.0 and 1.0 inclusive.");

  const size_t half_dim = dim / 2;
  size_t rotary_dim = static_cast<size_t>(static_cast<float>(dim) * rope_pct);
  rotary_dim = std::min(rotary_dim, dim);
  rotary_dim &= ~size_t{1};

  ET_CHECK_OR_RETURN_ERROR(
      rotary_dim > 0,
      InvalidArgument,
      "rope_pct results in zero rotary dimensions.");

  const size_t rotary_half = rotary_dim / 2;

  // Partial RoPE uses the full head_dim as the inv_freq denominator (Gemma
  // convention), so for full RoPE this reduces to the standard formula.
  const float freq_denom = static_cast<float>(dim);

  for (size_t pos = 0; pos < seq_len; ++pos) {
    float* cos_row = cos_ptr + pos * dim;
    float* sin_row = sin_ptr + pos * dim;

    // Non-rotary dimensions (partial RoPE) pass through unchanged: cos=1,
    // sin=0.
    for (size_t i = 0; i < dim; ++i) {
      cos_row[i] = 1.0f;
      sin_row[i] = 0.0f;
    }

    for (size_t i = 0; i < rotary_half; ++i) {
      // inv_freq_i = base^(-2i / head_dim)
      const float exponent = -2.0f * static_cast<float>(i) / freq_denom;
      const float inv_freq = std::pow(base, exponent);
      const float angle = static_cast<float>(pos) * inv_freq;

      const float c = std::cos(angle);
      const float s = std::sin(angle);

      // Split-half convention:
      //
      //   first half:   i
      //   second half:  half_dim + i
      //
      //    x = [x_0 ... x_{d/2-1} | x_{d/2} ... x_{d-1}]
      //    where each rotary pair is: (x_i, x_{i + d/2})
      // Both coordinates of the pair use the same cos/sin angle.

      cos_row[i] = c;
      sin_row[i] = s;

      cos_row[half_dim + i] = c;
      sin_row[half_dim + i] = s;
    }
  }

  return runtime::Error::Ok;
}

runtime::Error RotaryEmbedding::update(
    const std::vector<int64_t>& pos_ids,
    float* global_cos,
    float* global_sin,
    float* local_cos,
    float* local_sin) {
  if (!loaded_) {
    ET_LOG(Error, "RotaryEmbedding::update() called before load() succeeded.");
    return runtime::Error::InvalidState;
  }
  if (pos_ids.empty()) {
    return runtime::Error::InvalidArgument;
  }
  if (global_cos == nullptr || global_sin == nullptr) {
    return runtime::Error::InvalidArgument;
  }
  if (config_.embedding_strategy == EmbeddingStrategy::kGlobalAndLocal &&
      (local_cos == nullptr || local_sin == nullptr)) {
    return runtime::Error::InvalidArgument;
  }

  size_t pos = static_cast<size_t>(pos_ids[0]);
  size_t length = pos_ids.size();

  // Global Copy
  size_t g_head_dim = normalized_global_head_dim_;
  size_t g_bytes_to_copy = length * g_head_dim * sizeof(float);
  size_t g_element_offset = pos * g_head_dim * sizeof(float);
  std::memcpy(
      global_cos, global_cos_.get() + g_element_offset, g_bytes_to_copy);
  std::memcpy(
      global_sin, global_sin_.get() + g_element_offset, g_bytes_to_copy);

  // Local Copy
  if (config_.embedding_strategy == EmbeddingStrategy::kGlobalAndLocal) {
    size_t l_head_dim = config_.model_args->get_head_dim();
    size_t l_bytes_to_copy = length * l_head_dim * sizeof(float);
    size_t l_element_offset = pos * l_head_dim * sizeof(float);
    std::memcpy(
        local_cos, local_cos_.get() + l_element_offset, l_bytes_to_copy);
    std::memcpy(
        local_sin, local_sin_.get() + l_element_offset, l_bytes_to_copy);
  }

  return runtime::Error::Ok;
}

} // namespace example