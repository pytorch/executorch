/*
 * Copyright (c) 2025 Samsung Electronics Co. LTD
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <executorch/examples/samsung/executor_runner/llama/runner/constants.h>
#include <executorch/examples/samsung/executor_runner/llama/runner/io_manager.h>

namespace example {
  executorch::runtime::Result<std::vector<executorch::runtime::EValue>>
  LlmIOManager::prepare_prefill(
      const executorch::extension::TensorPtr& input,
      const executorch::extension::TensorPtr& start_pos,
      const std::string& prefill_method) {
    (void)prefill_method;

    std::vector<executorch::runtime::EValue> inputs;
    inputs.reserve(3 + config_.n_layers * 2);

    //re-producessing input token tensor
    const int64_t* prompt_tokens = input->mutable_data_ptr<int64_t>();
    size_t prompt_token_len = input->numel();

    input_tokens_.clear();
    input_tokens_.assign(prompt_tokens, prompt_tokens + prompt_token_len);
    input_tokens_.resize(config_.prefill_seq_len, input_tokens_.back());

    auto input_tokens_tensor = executorch::extension::from_blob(
      input_tokens_.data(), {1, static_cast<int>(config_.prefill_seq_len)},
      executorch::aten::ScalarType::Long);

    inputs.emplace_back(input_tokens_tensor);

    //re-producessing start_pos tensor
    const int64_t* start_position = start_pos->mutable_data_ptr<int64_t>();

    pos_ids_.clear();
    pos_ids_.resize(config_.prefill_seq_len);
    for(int64_t i = 0; i < config_.prefill_seq_len; ++i) {
      if (i < prompt_token_len) pos_ids_[i] = i;
      else pos_ids_[i] = prompt_token_len - 1;
    }
    auto start_pos_tensor = executorch::extension::from_blob(
      pos_ids_.data(), {static_cast<int>(config_.prefill_seq_len)}, executorch::aten::ScalarType::Long);

    inputs.emplace_back(start_pos_tensor);

    current_pos_ = *start_position;

    //attention tensor
    update_attn_mask(current_pos_, prompt_token_len);

    attn_mask_tensor_ = executorch::extension::from_blob(
        attn_mask_.data(),
        {static_cast<executorch::aten::SizesType>(config_.attn_mask_rows),
         static_cast<executorch::aten::SizesType>(config_.attn_mask_cols)});

    inputs.emplace_back(attn_mask_tensor_);

    prefill_k_input_tensors_.clear();
    prefill_v_input_tensors_.clear();
    prefill_k_output_tensors_.clear();
    prefill_v_output_tensors_.clear();

    for (size_t l = 0; l < config_.n_layers; l++) {
      auto k_in = executorch::extension::from_blob(
          prefill_k_input_[l].data(),
          {1,
           static_cast<executorch::aten::SizesType>(config_.n_kv_heads),
           static_cast<executorch::aten::SizesType>(prefill_cache_len_),
           static_cast<executorch::aten::SizesType>(config_.head_dim)},
          executorch::aten::ScalarType::Half);
      prefill_k_input_tensors_.push_back(k_in);
      inputs.emplace_back(k_in);
    }

    for (size_t l = 0; l < config_.n_layers; l++) {
      auto v_in = executorch::extension::from_blob(
          prefill_v_input_[l].data(),
          {1,
           static_cast<executorch::aten::SizesType>(config_.n_kv_heads),
           static_cast<executorch::aten::SizesType>(prefill_cache_len_),
           static_cast<executorch::aten::SizesType>(config_.head_dim)},
          executorch::aten::ScalarType::Half);
      prefill_v_input_tensors_.push_back(v_in);
      inputs.emplace_back(v_in);
    }

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

    update_attn_mask(current_pos_, 1);

    std::vector<executorch::runtime::EValue> inputs;
    inputs.reserve(3 + config_.n_layers * 2);
    inputs.emplace_back(input);
    inputs.emplace_back(start_pos);

    attn_mask_tensor_ = executorch::extension::from_blob(
        attn_mask_.data(),
        {1,
         static_cast<executorch::aten::SizesType>(config_.attn_mask_cols)});
    inputs.emplace_back(attn_mask_tensor_);

    decode_k_input_tensors_.clear();
    decode_v_input_tensors_.clear();
    decode_k_output_tensors_.clear();
    decode_v_output_tensors_.clear();

    for (size_t l = 0; l < config_.n_layers; l++) {
      auto k_in = executorch::extension::from_blob(
          decode_k_input_[l].data(),
          {1,
           static_cast<executorch::aten::SizesType>(config_.n_kv_heads),
           static_cast<executorch::aten::SizesType>(decode_cache_len_),
           static_cast<executorch::aten::SizesType>(config_.head_dim)},
          executorch::aten::ScalarType::Half);
      decode_k_input_tensors_.push_back(k_in);
      inputs.emplace_back(k_in);
    }
    for (size_t l = 0; l < config_.n_layers; l++) {
      auto v_in = executorch::extension::from_blob(
          decode_v_input_[l].data(),
          {1,
           static_cast<executorch::aten::SizesType>(config_.n_kv_heads),
           static_cast<executorch::aten::SizesType>(decode_cache_len_),
           static_cast<executorch::aten::SizesType>(config_.head_dim)},
          executorch::aten::ScalarType::Half);
      decode_v_input_tensors_.push_back(v_in);
      inputs.emplace_back(v_in);
    }

    dump_kv_input("decode");

    return inputs;
  }

  executorch::runtime::Error LlmIOManager::update_prefill(
      const std::vector<executorch::runtime::EValue>& model_outputs,
      const std::string& prefill_method) {
    (void)prefill_method;
    // model_outputs: [logits, k0, ..., k_{L-1}, v0, ..., v_{L-1}]
    // k_out shape: [1, n_kv_heads, prefill_seq_len, head_dim]
    // Accumulate prefill output into prefill input buffers. The copy to
    // decode input is deferred until the first prepare_decode call so that
    // multiple micro-batch prefill iterations work correctly.

    for (size_t l = 0; l < config_.n_layers; l++) {
      const auto& k_out = model_outputs[1 + l].toTensor();
      const _Float16* k_data = k_out.const_data_ptr<_Float16>();
      copy_to_cache(
          k_data,
          config_.prefill_seq_len,
          prefill_k_input_[l].data(),
          prefill_cache_len_,
          current_pos_);
    }
    for (size_t l = 0; l < config_.n_layers; l++) {
      const auto& v_out =
          model_outputs[1 + config_.n_layers + l].toTensor();
      const _Float16* v_data = v_out.const_data_ptr<_Float16>();
      copy_to_cache(
          v_data,
          config_.prefill_seq_len,
          prefill_v_input_[l].data(),
          prefill_cache_len_,
          current_pos_);
    }

    dump_kv_output("prefill", model_outputs);
    dump_step_++;

    return executorch::runtime::Error::Ok;
  }

  executorch::runtime::Error LlmIOManager::update_decode(
      const std::vector<executorch::runtime::EValue>& model_outputs,
      const std::string& decode_method) {
    (void)decode_method;
    // model_outputs: [logits, k0, ..., k_{L-1}, v0, ..., v_{L-1}]
    // k_out shape: [1, n_kv_heads, 1, head_dim]
    // Copy decode output into decode input cache at current_pos_.

    for (size_t l = 0; l < config_.n_layers; l++) {
      const auto& k_out = model_outputs[1 + l].toTensor();
      const _Float16* k_data = k_out.const_data_ptr<_Float16>();
      copy_to_cache(
          k_data,
          1,
          decode_k_input_[l].data(),
          decode_cache_len_,
          current_pos_);
    }
    for (size_t l = 0; l < config_.n_layers; l++) {
      const auto& v_out =
          model_outputs[1 + config_.n_layers + l].toTensor();
      const _Float16* v_data = v_out.const_data_ptr<_Float16>();
      copy_to_cache(
          v_data,
          1,
          decode_v_input_[l].data(),
          decode_cache_len_,
          current_pos_);
    }

    dump_kv_output("decode", model_outputs);
    dump_step_++;

    return executorch::runtime::Error::Ok;
  }

  void LlmIOManager::allocate_buffers() {
    size_t prefill_in_size = config_.n_kv_heads * prefill_cache_len_ *
        config_.head_dim;
    size_t decode_in_size = config_.n_kv_heads * decode_cache_len_ *
        config_.head_dim;

    prefill_k_input_.resize(config_.n_layers);
    prefill_v_input_.resize(config_.n_layers);
    decode_k_input_.resize(config_.n_layers);
    decode_v_input_.resize(config_.n_layers);

    for (size_t l = 0; l < config_.n_layers; l++) {
      prefill_k_input_[l].resize(prefill_in_size, 0.0f);
      prefill_v_input_[l].resize(prefill_in_size, 0.0f);
      decode_k_input_[l].resize(decode_in_size, 0.0f);
      decode_v_input_[l].resize(decode_in_size, 0.0f);
    }

    attn_mask_.resize(
        config_.attn_mask_rows * config_.attn_mask_cols, -1.0f);
  }

  void LlmIOManager::copy_prefill_to_decode_input() {
    for (size_t l = 0; l < config_.n_layers; l++) {
      for (size_t h = 0; h < config_.n_kv_heads; h++) {
        const _Float16* src = prefill_k_input_[l].data() +
            h * prefill_cache_len_ * config_.head_dim;
        _Float16* dst = decode_k_input_[l].data() +
            h * decode_cache_len_ * config_.head_dim;
        std::memcpy(
            dst, src, prefill_cache_len_ * config_.head_dim * sizeof(_Float16));
      }
      for (size_t h = 0; h < config_.n_kv_heads; h++) {
        const _Float16* src = prefill_v_input_[l].data() +
            h * prefill_cache_len_ * config_.head_dim;
        _Float16* dst = decode_v_input_[l].data() +
            h * decode_cache_len_ * config_.head_dim;
        std::memcpy(
            dst, src, prefill_cache_len_ * config_.head_dim * sizeof(_Float16));
      }
    }
  }

  void LlmIOManager::copy_to_cache(
      const _Float16* src,
      size_t new_len,
      _Float16* cache,
      size_t cache_len,
      size_t pos) {
    for (size_t h = 0; h < config_.n_kv_heads; h++) {
      const _Float16* src_head = src + h * new_len * config_.head_dim;
      _Float16* dst_head =
          cache + h * cache_len * config_.head_dim + pos * config_.head_dim;
      std::memcpy(
          dst_head, src_head, new_len * config_.head_dim * sizeof(_Float16));
    }
  }

  void LlmIOManager::update_attn_mask(size_t pos, size_t seq_len) {
    static int num_prompt_token = 0;  // temporary variable to share num_prompt_token info from prefill to decode
    std::fill(attn_mask_.begin(), attn_mask_.end(), -std::numeric_limits<float>::infinity());

    const size_t start_col = config_.attn_mask_cols - config_.attn_mask_rows;
    if (seq_len > 1) { // prefill : start from 896 and fill same mask as last_token to all empty rows.
        size_t last_valid_row = 0;
        num_prompt_token  = seq_len;

        for (size_t r = 0; r < seq_len && r < config_.attn_mask_rows; r++) {
            size_t visible_count = pos + r + 1;
            for (size_t c = 0; c < visible_count; c++) {
                size_t target_col = start_col + c;
                if (target_col < config_.attn_mask_cols) {
                    attn_mask_[r * config_.attn_mask_cols + target_col] = 0.0f;
                }
            }
            last_valid_row = r;
        }
        for (size_t r = last_valid_row + 1; r < config_.attn_mask_rows; r++) {
            float* src = &attn_mask_[last_valid_row * config_.attn_mask_cols];
            float* dst = &attn_mask_[r * config_.attn_mask_cols];
            std::memcpy(dst, src, config_.attn_mask_cols * sizeof(float));
        }
    }
    else {
        int pos_decode = pos - num_prompt_token + 1;
        for (int r = 0; r < (pos); r++) {
          attn_mask_[r] = 0.0f;
        }

        attn_mask_[config_.attn_mask_cols - 1] = 0.0f;
    }
  }

  void LlmIOManager::dump_buffer(
      const std::string& tag, size_t layer, const _Float16* data, size_t size) {
    if (!dump_enabled_) {
      return;
    }
    char filename[512];
    std::snprintf(
        filename, sizeof(filename),
        "%s/%s_layer%zu_pos%zu_step%zu.bin",
        dump_dir_.c_str(), tag.c_str(), layer, current_pos_, dump_step_);

    FILE* f = std::fopen(filename, "wb");
    if (f) {
      std::fwrite(data, sizeof(_Float16), size, f);
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

  void LlmIOManager::dump_kv_output(
      const std::string& phase,
      const std::vector<executorch::runtime::EValue>& model_outputs) {
    if (!dump_enabled_) {
      return;
    }
    for (size_t l = 0; l < config_.n_layers; l++) {
      const auto& k_out = model_outputs[1 + l].toTensor();
      const auto& v_out =
          model_outputs[1 + config_.n_layers + l].toTensor();
      dump_buffer(
          phase + "_k_output",
          l,
          k_out.const_data_ptr<_Float16>(),
          k_out.numel());
      dump_buffer(
          phase + "_v_output",
          l,
          v_out.const_data_ptr<_Float16>(),
          v_out.numel());
    }
  }
} // namespace example
