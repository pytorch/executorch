/*
 * Copyright (c) 2025 Samsung Electronics Co. LTD
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <cstring>

#include <executorch/examples/samsung/executor_runner/llm/runner/constants.h>
#include <executorch/examples/samsung/executor_runner/llm/runner/io_manager.h>

namespace example {

namespace llm = ::executorch::extension::llm;
namespace runtime = ::executorch::runtime;
namespace aten = ::executorch::aten;
namespace extension = ::executorch::extension;

namespace {

// Helper to fill a float region with a value, using memset when possible.
inline void fill_float_region(float* begin, float* end, float value) {
  if (begin >= end) {
    return;
  }
  if (value == 0.0f) {
    std::memset(begin, 0, static_cast<size_t>(end - begin) * sizeof(float));
  } else {
    std::fill(begin, end, value);
  }
}

inline void
fill_float_region(float* base, size_t offset, size_t len, float value) {
  fill_float_region(base + offset, base + offset + len, value);
}

} // namespace

LlmIOManager::LlmIOManager(extension::Module& module, Config config)
    : IOManager(module), config_(config) {
  ET_LOG(
      Info,
      "LlmIOManager Constructor: dim(%zu), vocab(%zu), max_seq(%zu)",
      config_.hidden_dim,
      config_.vocab_size,
      config_.max_seq_len);

  rotary_embedding_ = std::make_unique<RotaryEmbedding>(RotaryEmbedding::Config{
      config_.head_dim, config_.max_seq_len, config_.embedding_strategy});

  ET_CHECK_MSG(
      config_.max_seq_len >= config_.prefill_seq_len,
      "Inconsistent configuration: max_seq_len (%zu) must be greater than or equal to prefill_seq_len (%zu)",
      config_.max_seq_len,
      config_.prefill_seq_len);

  ET_CHECK_MSG(
      config_.max_seq_len >= 1,
      "Inconsistent configuration: max_seq_len must be at least 1, but got %zu",
      config_.max_seq_len);

  prefill_cache_len_ = config_.max_seq_len - config_.prefill_seq_len;
  decode_cache_len_ = config_.max_seq_len - 1;

  allocate_buffers();
}

executorch::runtime::Error LlmIOManager::load_rotary_embedding(
    std::optional<const std::string> pos_embedding_path) {
  if (pos_embedding_path.has_value()) {
    return rotary_embedding_->load(pos_embedding_path.value());
  }
  return executorch::runtime::Error::Ok;
}

executorch::runtime::Result<std::vector<executorch::runtime::EValue>>
LlmIOManager::prepare_prefill(
    const executorch::extension::TensorPtr& input,
    const executorch::extension::TensorPtr& start_pos,
    const std::string& prefill_method) {
  (void)prefill_method;
  std::vector<executorch::runtime::EValue> inputs;
  inputs.reserve(get_input_reserve_size());

  // re-producessing input token tensor
  const int64_t* prompt_tokens = input->mutable_data_ptr<int64_t>();
  size_t prompt_token_len = input->numel();
  input_tokens_.resize(prompt_token_len);
  std::copy(
      prompt_tokens, prompt_tokens + prompt_token_len, input_tokens_.begin());
  input_tokens_.resize(config_.prefill_seq_len, 0);

  input_tokens_tensor_ = executorch::extension::from_blob(
      input_tokens_.data(),
      {1, static_cast<int>(config_.prefill_seq_len)},
      executorch::aten::ScalarType::Long);
  inputs.emplace_back(input_tokens_tensor_);

  // re-producessing start_pos tensor
  const int64_t* start_position = start_pos->mutable_data_ptr<int64_t>();
  pos_ids_.resize(config_.prefill_seq_len);
  current_pos_ = *start_position;
  for (int64_t i = 0; i < config_.prefill_seq_len; ++i) {
    pos_ids_[i] = current_pos_ + i;
  }

  // attention tensor
  update_attn_mask(current_pos_, prompt_token_len);
  attn_mask_tensor_ = executorch::extension::from_blob(
      attn_mask_.data(),
      {1,
       1,
       static_cast<executorch::aten::SizesType>(config_.attn_mask_rows),
       static_cast<executorch::aten::SizesType>(config_.attn_mask_cols)});

  inputs.emplace_back(attn_mask_tensor_);

  switch (config_.masking_strategy) {
    case MaskingStrategy::kCausal:
      break;
    case MaskingStrategy::kCausalLocal:
    case MaskingStrategy::kHybridConvAttention:
      update_local_attn_mask(current_pos_, prompt_token_len);
      local_attn_mask_tensor_ = executorch::extension::from_blob(
          local_attn_mask_.data(),
          {1,
           1,
           static_cast<executorch::aten::SizesType>(config_.attn_mask_rows),
           static_cast<executorch::aten::SizesType>(config_.attn_mask_cols)});
      inputs.emplace_back(local_attn_mask_tensor_);
      break;
  }

  ET_CHECK_OK_OR_RETURN_ERROR(
      prepare_rotary_embedding_inputs(config_.prefill_seq_len, inputs));

  prefill_k_input_tensors_.clear();
  prefill_v_input_tensors_.clear();
  prefill_k_output_tensors_.clear();
  prefill_v_output_tensors_.clear();
  append_kv_input_tensors(
      prefill_k_input_,
      prefill_v_input_,
      prefill_k_input_tensors_,
      prefill_v_input_tensors_,
      executorch::aten::ScalarType::Float,
      prefill_cache_len_,
      inputs);
  dump_kv_input("prefill");

  return inputs;
}
executorch::runtime::Result<std::vector<executorch::runtime::EValue>>
LlmIOManager::prepare_decode(
    const executorch::extension::TensorPtr& input,
    const executorch::extension::TensorPtr& start_pos,
    const std::string& decode_method) {
  (void)decode_method;
  if (!prefill_done_) {
    copy_prefill_to_decode_input();
    prefill_done_ = true;
  }
  const int64_t* start_position = start_pos->mutable_data_ptr<int64_t>();
  current_pos_ = *start_position;
  update_attn_mask(current_pos_, config_.decode_seq_len);
  std::vector<executorch::runtime::EValue> inputs;
  inputs.reserve(get_input_reserve_size());
  inputs.emplace_back(input);

  attn_mask_tensor_ = executorch::extension::from_blob(
      attn_mask_.data(),
      {1,
       1,
       1,
       static_cast<executorch::aten::SizesType>(config_.attn_mask_cols)});
  inputs.emplace_back(attn_mask_tensor_);
  switch (config_.masking_strategy) {
    case MaskingStrategy::kCausal:
      break;
    case MaskingStrategy::kCausalLocal:
    case MaskingStrategy::kHybridConvAttention:
      update_local_attn_mask(current_pos_, 1);
      local_attn_mask_tensor_ = executorch::extension::from_blob(
          local_attn_mask_.data(),
          {1,
           1,
           1,
           static_cast<executorch::aten::SizesType>(config_.attn_mask_cols)});
      inputs.emplace_back(local_attn_mask_tensor_);
      break;
  }

  pos_ids_.clear();
  pos_ids_.resize(config_.decode_seq_len);
  pos_ids_[0] = current_pos_;
  ET_CHECK_OK_OR_RETURN_ERROR(
      prepare_rotary_embedding_inputs(config_.decode_seq_len, inputs));

  decode_k_input_tensors_.clear();
  decode_v_input_tensors_.clear();
  decode_k_output_tensors_.clear();
  decode_v_output_tensors_.clear();
  append_kv_input_tensors(
      decode_k_input_,
      decode_v_input_,
      decode_k_input_tensors_,
      decode_v_input_tensors_,
      executorch::aten::ScalarType::Float,
      decode_cache_len_,
      inputs);
  dump_kv_input("decode");

  return inputs;
}
executorch::runtime::Error LlmIOManager::update_prefill(
    const std::vector<executorch::runtime::EValue>& model_outputs,
    const std::string& prefill_method) {
  (void)prefill_method;
  update_kv_cache(
      model_outputs,
      prefill_k_input_,
      prefill_v_input_,
      prefill_cache_len_,
      config_.prefill_seq_len,
      "prefill");
  return executorch::runtime::Error::Ok;
}
executorch::runtime::Error LlmIOManager::update_decode(
    const std::vector<executorch::runtime::EValue>& model_outputs,
    const std::string& decode_method) {
  (void)decode_method;
  update_kv_cache(
      model_outputs,
      decode_k_input_,
      decode_v_input_,
      decode_cache_len_,
      1,
      "decode");
  return executorch::runtime::Error::Ok;
}
// TOOD: sizeof(float) dtype hard-codes the buffer dtype to be float.
void LlmIOManager::allocate_buffers() {
  size_t prefill_in_size = config_.n_kv_heads * prefill_cache_len_ *
      config_.head_dim * sizeof(float);
  size_t decode_in_size =
      config_.n_kv_heads * decode_cache_len_ * config_.head_dim * sizeof(float);
  prefill_k_input_.resize(config_.n_layers);
  prefill_v_input_.resize(config_.n_layers);
  decode_k_input_.resize(config_.n_layers);
  decode_v_input_.resize(config_.n_layers);
  for (size_t l = 0; l < config_.n_layers; l++) {
    prefill_k_input_[l].resize(prefill_in_size, 0);
    prefill_v_input_[l].resize(prefill_in_size, 0);
    decode_k_input_[l].resize(decode_in_size, 0);
    decode_v_input_[l].resize(decode_in_size, 0);
  }
  attn_mask_.resize(config_.attn_mask_rows * config_.attn_mask_cols, -1.0f);
  local_attn_mask_.resize(
      config_.attn_mask_rows * config_.attn_mask_cols, -1.0f);
}
void LlmIOManager::copy_prefill_to_decode_input() {
  size_t copy_size = prefill_cache_len_ * config_.head_dim * sizeof(float);
  for (size_t l = 0; l < config_.n_layers; l++) {
    for (size_t h = 0; h < config_.n_kv_heads; h++) {
      const char* src = prefill_k_input_[l].data() +
          h * prefill_cache_len_ * config_.head_dim * sizeof(float);
      char* dst = decode_k_input_[l].data() +
          h * decode_cache_len_ * config_.head_dim * sizeof(float);
      std::memcpy(dst, src, copy_size);
    }
    for (size_t h = 0; h < config_.n_kv_heads; h++) {
      const char* src = prefill_v_input_[l].data() +
          h * prefill_cache_len_ * config_.head_dim * sizeof(float);
      char* dst = decode_v_input_[l].data() +
          h * decode_cache_len_ * config_.head_dim * sizeof(float);
      std::memcpy(dst, src, copy_size);
    }
  }
}

void LlmIOManager::append_kv_input_tensors(
    std::vector<std::vector<char>>& k_bufs,
    std::vector<std::vector<char>>& v_bufs,
    std::vector<executorch::extension::TensorPtr>& k_tensors,
    std::vector<executorch::extension::TensorPtr>& v_tensors,
    executorch::aten::ScalarType kv_scalar_type,
    size_t cache_len,
    std::vector<executorch::runtime::EValue>& inputs) const {
  for (size_t l = 0; l < config_.n_layers; l++) {
    auto k_in = executorch::extension::from_blob(
        k_bufs[l].data(),
        {1,
         static_cast<executorch::aten::SizesType>(config_.n_kv_heads),
         static_cast<executorch::aten::SizesType>(cache_len),
         static_cast<executorch::aten::SizesType>(config_.head_dim)},
        kv_scalar_type);
    k_tensors.push_back(k_in);
    inputs.emplace_back(k_in);
  }
  for (size_t l = 0; l < config_.n_layers; l++) {
    auto v_in = executorch::extension::from_blob(
        v_bufs[l].data(),
        {1,
         static_cast<executorch::aten::SizesType>(config_.n_kv_heads),
         static_cast<executorch::aten::SizesType>(cache_len),
         static_cast<executorch::aten::SizesType>(config_.head_dim)},
        kv_scalar_type);
    v_tensors.push_back(v_in);
    inputs.emplace_back(v_in);
  }
}

void LlmIOManager::update_attn_mask(size_t pos, size_t seq_len) {
  const float maskFalse = -255.0f;
  const float maskTrue = 0.0f;

  const size_t rows = config_.attn_mask_rows;
  const size_t cols = config_.attn_mask_cols;
  const size_t start_col = cols - rows;
  const size_t total_size = rows * cols;

  // Initialize entire mask to maskFalse (uses memset for 0.0f)
  fill_float_region(attn_mask_.data(), 0, total_size, maskFalse);

  if (seq_len > 1) {
    // Prefill phase
    const size_t rows = config_.attn_mask_rows;
    const size_t cols = config_.attn_mask_cols;
    const size_t start_col = config_.attn_mask_cols - config_.attn_mask_rows;

    for (int i = 0; i < seq_len; ++i) {
      int start_idx = cols - rows;
      int end_idx = start_idx + seq_len;

      for (int j = start_idx; j < end_idx && j <= start_idx + i; ++j) {
        attn_mask_[i * cols + j] = maskTrue;
      }
    }

    for (int i = 0; i < rows; ++i) {
      for (int j = 0; j < pos; ++j) {
        attn_mask_[i * cols + j] = maskTrue;
      }
    }
  } else {
    const size_t left_len = std::min(pos, cols);
    if (left_len > 0) {
      fill_float_region(attn_mask_.data(), 0, left_len, maskTrue);
    }

    if (cols > 0) {
      attn_mask_[cols - 1] = maskTrue;
    }
  }
}

void LlmIOManager::update_local_attn_mask(size_t pos, size_t seq_len) {
  const float maskFalse = -255.0f;
  const float maskTrue = 0.0f;

  const size_t rows = config_.attn_mask_rows;
  const size_t cols = config_.attn_mask_cols;
  const size_t sliding_window = config_.sliding_window.value();
  const size_t total_size = rows * cols;

  // Start with a copy of the full attention mask
  std::memcpy(
      local_attn_mask_.data(), attn_mask_.data(), total_size * sizeof(float));

  if (seq_len > 1) {
    // Prefill phase with sliding window
    if ((pos + seq_len) >= sliding_window) {
      // Mask out positions before the sliding window
      const size_t mask_start = sliding_window - seq_len;
      const size_t mask_end = pos; // 512
      const size_t mask_len = mask_end - mask_start;

      if (mask_len > 0) {
        for (size_t i = 0; i < rows; ++i) {
          fill_float_region(
              local_attn_mask_.data(),
              i * cols + mask_start,
              mask_len,
              maskFalse);
        }
      }
    }
  } else {
    // Decode phase (single token)
    // Set all positions up to pos to maskTrue
    const size_t end = std::min(static_cast<size_t>(pos), cols);
    fill_float_region(local_attn_mask_.data(), 0, end, maskTrue);

    // If there are more than 512 True values, we need to keep only the last 512
    // Count True values efficiently by scanning from the end
    size_t valid_count = 1;
    for (size_t i = 0; i < cols; ++i) {
      if (local_attn_mask_[i] == maskTrue) {
        valid_count++;
      } else {
        break;
      }
    }

    if (valid_count > sliding_window) {
      const size_t to_remove = valid_count - sliding_window;
      // Set the oldest True values to maskFalse
      for (size_t i = 0; i < to_remove; ++i) {
        if (local_attn_mask_[i] == maskTrue) {
          local_attn_mask_[i] = maskFalse;
        }
      }
    }

    if (cols > 0) {
      attn_mask_[cols - 1] = maskTrue;
    }
  }
}

void LlmIOManager::dump_buffer(
    const std::string& tag,
    size_t layer,
    const char* data,
    size_t size) {
  if (!dump_enabled_) {
    return;
  }
  char filename[512];
  std::snprintf(
      filename,
      sizeof(filename),
      "%s/%s_layer%zu_pos%zu_step%zu.bin",
      dump_dir_.c_str(),
      tag.c_str(),
      layer,
      current_pos_,
      dump_step_);
  FILE* f = std::fopen(filename, "wb");
  if (f) {
    std::fwrite(data, 1, size, f);
    std::fclose(f);
  }
}
void LlmIOManager::dump_kv_input(const std::string& phase) {
  if (!dump_enabled_) {
    return;
  }
  const bool is_prefill = (phase == "prefill");
  const auto& k_bufs = is_prefill ? prefill_k_input_ : decode_k_input_;
  const auto& v_bufs = is_prefill ? prefill_v_input_ : decode_v_input_;
  for (size_t l = 0; l < config_.n_layers; l++) {
    dump_buffer(phase + "_k_input", l, k_bufs[l].data(), k_bufs[l].size());
    dump_buffer(phase + "_v_input", l, v_bufs[l].data(), v_bufs[l].size());
  }
}
void LlmIOManager::update_kv_cache(
    const std::vector<executorch::runtime::EValue>& model_outputs,
    std::vector<std::vector<char>>& k_bufs,
    std::vector<std::vector<char>>& v_bufs,
    size_t cache_len,
    size_t new_len,
    const std::string& phase) {
  // model_outputs: [logits, k0..k_{L-1}, v0..v_{L-1}]
  for (size_t l = 0; l < config_.n_layers; l++) {
    copy_tensor_to_cache(
        model_outputs[1 + l].toTensor(),
        new_len,
        k_bufs[l].data(),
        cache_len,
        current_pos_);
  }
  for (size_t l = 0; l < config_.n_layers; l++) {
    copy_tensor_to_cache(
        model_outputs[1 + config_.n_layers + l].toTensor(),
        new_len,
        v_bufs[l].data(),
        cache_len,
        current_pos_);
  }
  dump_kv_output(phase, model_outputs);
  dump_step_++;
}
void LlmIOManager::dump_kv_output(
    const std::string& phase,
    const std::vector<executorch::runtime::EValue>& model_outputs) {
  if (!dump_enabled_) {
    return;
  }
  for (size_t l = 0; l < config_.n_layers; l++) {
    const auto& k_out = model_outputs[1 + l].toTensor();
    const auto& v_out = model_outputs[1 + config_.n_layers + l].toTensor();
    dump_buffer(
        phase + "_k_output",
        l,
        reinterpret_cast<const char*>(k_out.const_data_ptr<float>()),
        k_out.numel() * sizeof(float));
    dump_buffer(
        phase + "_v_output",
        l,
        reinterpret_cast<const char*>(v_out.const_data_ptr<float>()),
        v_out.numel() * sizeof(float));
  }
}

executorch::runtime::Error LlmIOManager::prepare_rotary_embedding_inputs(
    size_t seq_len,
    std::vector<executorch::runtime::EValue>& inputs) {
  const size_t element_size = seq_len * config_.head_dim;
  global_cos_data_.resize(element_size);
  global_sin_data_.resize(element_size);
  switch (config_.embedding_strategy) {
    case EmbeddingStrategy::kGlobalOnly:
      break;
    case EmbeddingStrategy::kGlobalAndLocal:
      local_cos_data_.resize(element_size);
      local_sin_data_.resize(element_size);
      break;
  }

  auto update_err = rotary_embedding_->update(
      pos_ids_,
      global_cos_data_.data(),
      global_sin_data_.data(),
      local_cos_data_.data(),
      local_sin_data_.data());
  if (update_err != executorch::runtime::Error::Ok) {
    ET_LOG(Error, "update pos_embedding failed !");
    return update_err;
  }

  global_cos_tensor_ = executorch::extension::from_blob(
      global_cos_data_.data(),
      {1,
       static_cast<executorch::aten::SizesType>(seq_len),
       static_cast<executorch::aten::SizesType>(config_.head_dim)});
  global_sin_tensor_ = executorch::extension::from_blob(
      global_sin_data_.data(),
      {1,
       static_cast<executorch::aten::SizesType>(seq_len),
       static_cast<executorch::aten::SizesType>(config_.head_dim)});
  switch (config_.embedding_strategy) {
    case EmbeddingStrategy::kGlobalOnly:
      break;
    case EmbeddingStrategy::kGlobalAndLocal:
      local_cos_tensor_ = executorch::extension::from_blob(
          local_cos_data_.data(),
          {1,
           static_cast<executorch::aten::SizesType>(seq_len),
           static_cast<executorch::aten::SizesType>(config_.head_dim)});
      local_sin_tensor_ = executorch::extension::from_blob(
          local_sin_data_.data(),
          {1,
           static_cast<executorch::aten::SizesType>(seq_len),
           static_cast<executorch::aten::SizesType>(config_.head_dim)});
      break;
  }

  inputs.emplace_back(global_cos_tensor_);
  inputs.emplace_back(global_sin_tensor_);
  switch (config_.embedding_strategy) {
    case EmbeddingStrategy::kGlobalOnly:
      break;
    case EmbeddingStrategy::kGlobalAndLocal:
      inputs.emplace_back(local_cos_tensor_);
      inputs.emplace_back(local_sin_tensor_);
      break;
  }

  return executorch::runtime::Error::Ok;
}

} // namespace example
