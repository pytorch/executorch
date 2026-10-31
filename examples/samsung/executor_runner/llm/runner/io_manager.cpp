/*
 * Copyright (c) 2025 Samsung Electronics Co. LTD
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <algorithm>
#include <cinttypes>
#include <cstring>

#include <executorch/examples/samsung/executor_runner/llm/runner/config.h>
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

LlmIOManager::LlmIOManager(
    extension::Module& module,
    const RunnerConfig& config)
    : IOManager(module), config_(config) {
  rotary_embedding_ = std::make_unique<RotaryEmbedding>(config);

  // RunnerConfig::validate() guarantees 1 <= prefill_seq_len <= max_seq_len.
  prefill_cache_len_ = config_.model_args->get_max_seq_len() -
      config_.model_args->get_prefill_seq_len();
  decode_cache_len_ = config_.model_args->get_max_seq_len() - 1;

  allocate_buffers();
}

size_t LlmIOManager::get_attention_setup_tensor_count() const {
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

ET_NODISCARD runtime::Error LlmIOManager::load(
    const std::string& prefill_method,
    const std::string& decode_method) {
  (void)prefill_method;
  (void)decode_method;

  if ((config_.masking_strategy == MaskingStrategy::kCausalLocal ||
       config_.masking_strategy == MaskingStrategy::kHybridConvAttention) &&
      !config_.model_args->get_sliding_window().has_value()) {
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
      config_.model_args->get_n_layers(),
      config_.model_args->get_n_kv_heads(),
      config_.model_args->get_hidden_dim(),
      config_.model_args->get_head_dim(),
      config_.model_args->get_vocab_size(),
      config_.model_args->get_max_seq_len(),
      config_.model_args->get_max_context_len(),
      config_.model_args->get_prefill_seq_len(),
      config_.model_args->get_decode_seq_len(),
      config_.model_args->get_attn_mask_rows(),
      config_.model_args->get_attn_mask_cols(),
      config_.model_args->get_sliding_window().value_or(0),
      static_cast<int>(config_.embedding_strategy),
      static_cast<int>(config_.masking_strategy));

  return runtime::Error::Ok;
}

ET_NODISCARD runtime::Error LlmIOManager::reset(
    const std::string& prefill_method,
    const std::string& decode_method) {
  (void)prefill_method;
  (void)decode_method;

  current_pos_ = 0;
  prefill_done_ = false;

  for (size_t l = 0; l < config_.model_args->get_num_cache_layers(); l++) {
    std::fill(prefill_k_input_[l].begin(), prefill_k_input_[l].end(), 0);
    std::fill(prefill_v_input_[l].begin(), prefill_v_input_[l].end(), 0);
    std::fill(decode_k_input_[l].begin(), decode_k_input_[l].end(), 0);
    std::fill(decode_v_input_[l].begin(), decode_v_input_[l].end(), 0);
  }
  std::fill(attn_mask_.begin(), attn_mask_.end(), 0.0f);

  return runtime::Error::Ok;
}

executorch::runtime::Error LlmIOManager::load_rotary_embedding(
    std::optional<const std::string> pos_embedding_path) {
  return rotary_embedding_->load(pos_embedding_path);
}

executorch::runtime::Result<std::vector<executorch::runtime::EValue>>
LlmIOManager::prepare_prefill(
    const executorch::extension::TensorPtr& input,
    const executorch::extension::TensorPtr& start_pos,
    const std::string& prefill_method) {
  (void)prefill_method;
  std::vector<executorch::runtime::EValue> inputs;
  inputs.reserve(get_input_reserve_size());
  const size_t prefill_seq_len = config_.model_args->get_prefill_seq_len();
  const size_t prompt_token_len = input->numel();

  ET_CHECK_OR_RETURN_ERROR(
      prompt_token_len <= prefill_seq_len,
      InvalidArgument,
      "Prompt chunk length %zu exceeds prefill sequence length %zu.",
      prompt_token_len,
      prefill_seq_len);

  const int64_t* prompt_tokens = input->const_data_ptr<int64_t>();
  input_tokens_.assign(prefill_seq_len, 0);

  std::copy(
      prompt_tokens, prompt_tokens + prompt_token_len, input_tokens_.begin());

  input_tokens_tensor_ = executorch::extension::from_blob(
      input_tokens_.data(),
      {1, static_cast<int>(prefill_seq_len)},
      executorch::aten::ScalarType::Long);
  inputs.emplace_back(input_tokens_tensor_);

  append_prelude_inputs(prefill_seq_len, inputs);

  const int64_t start_position = *start_pos->const_data_ptr<int64_t>();
  ET_CHECK_OR_RETURN_ERROR(
      start_position >= 0 &&
          static_cast<size_t>(start_position) + prompt_token_len <=
              prefill_cache_len_,
      InvalidArgument,
      "Prompt does not fit: chunk at position %" PRId64
      " with %zu tokens exceeds the %zu-token prefill cache",
      start_position,
      prompt_token_len,
      prefill_cache_len_);
  current_pos_ = static_cast<size_t>(start_position);
  pos_ids_.resize(prefill_seq_len);
  for (size_t i = 0; i < prefill_seq_len; ++i) {
    pos_ids_[i] = current_pos_ + static_cast<int64_t>(i);
  }

  // attention tensor
  update_attn_mask(current_pos_, prompt_token_len);
  attn_mask_tensor_ = executorch::extension::from_blob(
      attn_mask_.data(),
      {1,
       1,
       static_cast<executorch::aten::SizesType>(
           config_.model_args->get_attn_mask_rows()),
       static_cast<executorch::aten::SizesType>(
           config_.model_args->get_attn_mask_cols())});

  inputs.emplace_back(attn_mask_tensor_);

  const auto masking_strategy = config_.masking_strategy;
  switch (masking_strategy) {
    case MaskingStrategy::kCausal:
      break;
    case MaskingStrategy::kCausalLocal:
    case MaskingStrategy::kHybridConvAttention:
      update_local_attn_mask(current_pos_, prompt_token_len);
      const size_t mask_rows = config_.model_args->get_attn_mask_rows();
      const size_t mask_cols = config_.model_args->get_attn_mask_cols();
      local_attn_mask_tensor_ = executorch::extension::from_blob(
          local_attn_mask_.data(),
          {1,
           1,
           static_cast<executorch::aten::SizesType>(mask_rows),
           static_cast<executorch::aten::SizesType>(mask_cols)});
      inputs.emplace_back(local_attn_mask_tensor_);
      break;
  }

  ET_CHECK_OK_OR_RETURN_ERROR(
      prepare_rotary_embedding_inputs(prefill_seq_len, inputs));

  prefill_k_input_tensors_.clear();
  prefill_v_input_tensors_.clear();
  prefill_k_output_tensors_.clear();
  prefill_v_output_tensors_.clear();
  dummy_kv_tensors_.clear();
  append_kv_input_tensors(
      prefill_k_input_,
      prefill_v_input_,
      prefill_k_input_tensors_,
      prefill_v_input_tensors_,
      executorch::aten::ScalarType::Float,
      prefill_cache_len_,
      prefill_dummy_kv_,
      inputs);
  return inputs;
}
executorch::runtime::Result<std::vector<executorch::runtime::EValue>>
LlmIOManager::prepare_decode(
    const executorch::extension::TensorPtr& input,
    const executorch::extension::TensorPtr& start_pos,
    const std::string& decode_method) {
  (void)decode_method;
  const int64_t start_position = *start_pos->const_data_ptr<int64_t>();
  ET_CHECK_OR_RETURN_ERROR(
      start_position >= 0 &&
          static_cast<size_t>(start_position) <= decode_cache_len_,
      InvalidArgument,
      "Decode position %" PRId64 " exceeds max_seq_len %" PRId64,
      start_position,
      config_.model_args->get_max_seq_len());
  if (!prefill_done_) {
    copy_prefill_to_decode_input();
    prefill_done_ = true;
  }
  current_pos_ = static_cast<size_t>(start_position);
  update_attn_mask(current_pos_, config_.model_args->get_decode_seq_len());
  std::vector<executorch::runtime::EValue> inputs;
  inputs.reserve(get_input_reserve_size());
  inputs.emplace_back(input);

  append_prelude_inputs(config_.model_args->get_decode_seq_len(), inputs);

  attn_mask_tensor_ = executorch::extension::from_blob(
      attn_mask_.data(),
      {1,
       1,
       1,
       static_cast<executorch::aten::SizesType>(
           config_.model_args->get_attn_mask_cols())});
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
           static_cast<executorch::aten::SizesType>(
               config_.model_args->get_attn_mask_cols())});
      inputs.emplace_back(local_attn_mask_tensor_);
      break;
  }

  pos_ids_.clear();
  pos_ids_.resize(config_.model_args->get_decode_seq_len());
  pos_ids_[0] = current_pos_;
  ET_CHECK_OK_OR_RETURN_ERROR(prepare_rotary_embedding_inputs(
      config_.model_args->get_decode_seq_len(), inputs));

  decode_k_input_tensors_.clear();
  decode_v_input_tensors_.clear();
  decode_k_output_tensors_.clear();
  decode_v_output_tensors_.clear();
  dummy_kv_tensors_.clear();
  append_kv_input_tensors(
      decode_k_input_,
      decode_v_input_,
      decode_k_input_tensors_,
      decode_v_input_tensors_,
      executorch::aten::ScalarType::Float,
      decode_cache_len_,
      decode_dummy_kv_,
      inputs);
  return inputs;
}
executorch::runtime::Error LlmIOManager::update_prefill(
    const std::vector<executorch::runtime::EValue>& model_outputs,
    const std::string& prefill_method) {
  (void)prefill_method;
  return update_kv_cache(
      model_outputs,
      prefill_k_input_,
      prefill_v_input_,
      prefill_cache_len_,
      config_.model_args->get_prefill_seq_len());
}
executorch::runtime::Error LlmIOManager::update_decode(
    const std::vector<executorch::runtime::EValue>& model_outputs,
    const std::string& decode_method) {
  (void)decode_method;
  return update_kv_cache(
      model_outputs, decode_k_input_, decode_v_input_, decode_cache_len_, 1);
}
void LlmIOManager::allocate_buffers() {
  const size_t n_kv_heads = config_.model_args->get_n_kv_heads();
  const size_t num_cache_layers = config_.model_args->get_num_cache_layers();
  prefill_k_input_.resize(num_cache_layers);
  prefill_v_input_.resize(num_cache_layers);
  decode_k_input_.resize(num_cache_layers);
  decode_v_input_.resize(num_cache_layers);
  for (size_t l = 0; l < num_cache_layers; l++) {
    const size_t head_dim = config_.model_args->get_head_dim_for_layer(l);
    const size_t prefill_in_size =
        n_kv_heads * prefill_cache_len_ * head_dim * sizeof(float);
    const size_t decode_in_size =
        n_kv_heads * decode_cache_len_ * head_dim * sizeof(float);
    prefill_k_input_[l].resize(prefill_in_size, 0);
    prefill_v_input_[l].resize(prefill_in_size, 0);
    decode_k_input_[l].resize(decode_in_size, 0);
    decode_v_input_[l].resize(decode_in_size, 0);
  }

  // Dummy scratch for KV-shared layer slots, sized for the widest head_dim.
  const size_t max_head_dim = std::max<size_t>(
      config_.model_args->get_head_dim(),
      config_.model_args->get_global_head_dim());
  prefill_dummy_kv_.assign(
      n_kv_heads * prefill_cache_len_ * max_head_dim * sizeof(float), 0);
  decode_dummy_kv_.assign(
      n_kv_heads * decode_cache_len_ * max_head_dim * sizeof(float), 0);
  attn_mask_.resize(
      config_.model_args->get_attn_mask_rows() *
          config_.model_args->get_attn_mask_cols(),
      -1.0f);
  local_attn_mask_.resize(
      config_.model_args->get_attn_mask_rows() *
          config_.model_args->get_attn_mask_cols(),
      -1.0f);
}
void LlmIOManager::copy_prefill_to_decode_input() {
  const size_t n_kv_heads = config_.model_args->get_n_kv_heads();
  for (size_t l = 0; l < config_.model_args->get_num_cache_layers(); l++) {
    const size_t head_dim = config_.model_args->get_head_dim_for_layer(l);
    const size_t copy_size = prefill_cache_len_ * head_dim * sizeof(float);
    for (size_t h = 0; h < n_kv_heads; h++) {
      const char* k_src = prefill_k_input_[l].data() +
          h * prefill_cache_len_ * head_dim * sizeof(float);
      char* k_dst = decode_k_input_[l].data() +
          h * decode_cache_len_ * head_dim * sizeof(float);
      std::memcpy(k_dst, k_src, copy_size);
      const char* v_src = prefill_v_input_[l].data() +
          h * prefill_cache_len_ * head_dim * sizeof(float);
      char* v_dst = decode_v_input_[l].data() +
          h * decode_cache_len_ * head_dim * sizeof(float);
      std::memcpy(v_dst, v_src, copy_size);
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
    std::vector<char>& dummy_buf,
    std::vector<executorch::runtime::EValue>& inputs) const {
  const size_t n_layers = config_.model_args->get_n_layers();
  const size_t num_cache_layers = config_.model_args->get_num_cache_layers();
  const auto n_kv_heads = static_cast<executorch::aten::SizesType>(
      config_.model_args->get_n_kv_heads());

  // The model declares a KV input slot for every layer, in K[0..L-1] then
  // V[0..L-1] order. The leading self-decoder layers carry the real cache;
  // the trailing KV-shared (cross-decoder) layers are resolved inside the
  // delegate and their slots are never read, so point them at zeroed scratch.
  auto make_tensor = [&](std::vector<std::vector<char>>& bufs,
                         size_t layer) -> executorch::extension::TensorPtr {
    char* data =
        (layer < num_cache_layers) ? bufs[layer].data() : dummy_buf.data();
    return executorch::extension::from_blob(
        data,
        {1,
         n_kv_heads,
         static_cast<executorch::aten::SizesType>(cache_len),
         static_cast<executorch::aten::SizesType>(
             config_.model_args->get_head_dim_for_layer(layer))},
        kv_scalar_type);
  };

  for (size_t l = 0; l < n_layers; l++) {
    auto k_in = make_tensor(k_bufs, l);
    if (l < num_cache_layers) {
      k_tensors.push_back(k_in);
    } else {
      dummy_kv_tensors_.push_back(k_in);
    }
    inputs.emplace_back(k_in);
  }
  for (size_t l = 0; l < n_layers; l++) {
    auto v_in = make_tensor(v_bufs, l);
    if (l < num_cache_layers) {
      v_tensors.push_back(v_in);
    } else {
      dummy_kv_tensors_.push_back(v_in);
    }
    inputs.emplace_back(v_in);
  }
}

void LlmIOManager::update_attn_mask(size_t pos, size_t seq_len) {
  const float maskFalse = -255.0f;
  const float maskTrue = 0.0f;

  const size_t rows = config_.model_args->get_attn_mask_rows();
  const size_t cols = config_.model_args->get_attn_mask_cols();
  const size_t total_size = rows * cols;

  fill_float_region(attn_mask_.data(), 0, total_size, maskFalse);

  if (seq_len > 1) {
    // Prefill: past positions [0, pos) and the causal block for the chunk,
    // which occupies the last `rows` columns.
    const size_t start_col = cols - rows;
    for (size_t i = 0; i < seq_len; ++i) {
      fill_float_region(
          attn_mask_.data(), i * cols + start_col, i + 1, maskTrue);
    }
    for (size_t i = 0; i < rows; ++i) {
      fill_float_region(attn_mask_.data(), i * cols, pos, maskTrue);
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

  const size_t rows = config_.model_args->get_attn_mask_rows();
  const size_t cols = config_.model_args->get_attn_mask_cols();
  const size_t sliding_window =
      config_.model_args->get_sliding_window().value();
  const size_t total_size = rows * cols;

  // Start with a copy of the full attention mask
  std::memcpy(
      local_attn_mask_.data(), attn_mask_.data(), total_size * sizeof(float));

  if (seq_len > 1) {
    // Prefill phase with sliding window: row i sits at position pos + i and
    // sees the last sliding_window keys, so mask the oldest cached columns.
    for (size_t i = 0; i < rows; ++i) {
      const size_t query_pos = pos + i;
      if (query_pos + 1 <= sliding_window) {
        continue;
      }
      const size_t mask_len = std::min(query_pos + 1 - sliding_window, pos);
      fill_float_region(local_attn_mask_.data(), i * cols, mask_len, maskFalse);
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
      local_attn_mask_[cols - 1] = maskTrue;
    }
  }
}

executorch::runtime::Error LlmIOManager::update_kv_cache(
    const std::vector<executorch::runtime::EValue>& model_outputs,
    std::vector<std::vector<char>>& k_bufs,
    std::vector<std::vector<char>>& v_bufs,
    size_t cache_len,
    size_t new_len) {
  // model_outputs: [logits, k0..k_{C-1}, v0..v_{C-1}] where C is the number of
  // cached (self-decoder) layers; KV-shared layers emit no new cache.
  const size_t num_cache_layers = config_.model_args->get_num_cache_layers();
  ET_CHECK_OR_RETURN_ERROR(
      model_outputs.size() == 1 + 2 * num_cache_layers,
      InvalidProgram,
      "Expected %zu outputs (logits + K/V for %zu layers), got %zu",
      1 + 2 * num_cache_layers,
      num_cache_layers,
      model_outputs.size());
  // The last decode step may run at pos == cache_len; its KV is never read.
  if (current_pos_ >= cache_len) {
    return executorch::runtime::Error::Ok;
  }
  const size_t rows = std::min(new_len, cache_len - current_pos_);
  const size_t n_kv_heads = config_.model_args->get_n_kv_heads();
  for (size_t i = 0; i < 2 * num_cache_layers; i++) {
    const size_t l = i % num_cache_layers;
    const size_t head_dim = config_.model_args->get_head_dim_for_layer(l);
    const runtime::EValue& out = model_outputs[1 + i];
    ET_CHECK_OR_RETURN_ERROR(
        out.isTensor() &&
            out.toTensor().scalar_type() == aten::ScalarType::Float &&
            static_cast<size_t>(out.toTensor().numel()) ==
                n_kv_heads * new_len * head_dim,
        InvalidProgram,
        "KV output %zu must be a float tensor of %zu elements",
        1 + i,
        n_kv_heads * new_len * head_dim);
    auto& bufs = i < num_cache_layers ? k_bufs : v_bufs;
    copy_to_cache<float>(
        out.toTensor().const_data_ptr<float>(),
        new_len,
        rows,
        bufs[l].data(),
        cache_len,
        current_pos_,
        head_dim);
  }
  return executorch::runtime::Error::Ok;
}

executorch::runtime::Error LlmIOManager::prepare_rotary_embedding_inputs(
    size_t seq_len,
    std::vector<executorch::runtime::EValue>& inputs) {
  const int64_t head_dim = config_.model_args->get_head_dim();
  const int64_t global_head_dim = config_.model_args->get_global_head_dim();

  const size_t global_element_size = seq_len * global_head_dim;
  global_cos_data_.resize(global_element_size);
  global_sin_data_.resize(global_element_size);

  switch (config_.embedding_strategy) {
    case EmbeddingStrategy::kGlobalOnly:
      break;
    case EmbeddingStrategy::kGlobalAndLocal: {
      const size_t local_element_size = seq_len * head_dim;
      local_cos_data_.resize(local_element_size);
      local_sin_data_.resize(local_element_size);
      break;
    }
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
       static_cast<executorch::aten::SizesType>(global_head_dim)});
  global_sin_tensor_ = executorch::extension::from_blob(
      global_sin_data_.data(),
      {1,
       static_cast<executorch::aten::SizesType>(seq_len),
       static_cast<executorch::aten::SizesType>(global_head_dim)});
  switch (config_.embedding_strategy) {
    case EmbeddingStrategy::kGlobalOnly:
      break;
    case EmbeddingStrategy::kGlobalAndLocal:
      local_cos_tensor_ = executorch::extension::from_blob(
          local_cos_data_.data(),
          {1,
           static_cast<executorch::aten::SizesType>(seq_len),
           static_cast<executorch::aten::SizesType>(head_dim)});
      local_sin_tensor_ = executorch::extension::from_blob(
          local_sin_data_.data(),
          {1,
           static_cast<executorch::aten::SizesType>(seq_len),
           static_cast<executorch::aten::SizesType>(head_dim)});
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

void LlmIOManager::set_vision_embeddings(
    std::vector<float> embeds,
    int64_t num_tokens) {
  vision_embeds_ = std::move(embeds);
  num_vision_tokens_ = num_tokens;
  vision_embed_idx_ = 0;
  ET_LOG(
      Info,
      "Vision embeddings set: %lld tokens, %zu floats",
      static_cast<long long>(num_vision_tokens_),
      vision_embeds_.size());
}

void LlmIOManager::append_prelude_inputs(
    size_t seq_len,
    std::vector<executorch::runtime::EValue>& inputs) {
  if (config_.model_version != DecoderModelVersion::kGemma4) {
    return;
  }
  const int64_t hidden_dim = config_.model_args->get_hidden_dim();

  // inputs_embeds [1, seq_len, hidden_dim]: zeros for text-only inference. The
  // decoder embeds token ids internally; this slot only matters at multimodal
  // placeholder positions, which text-only prompts never contain.
  inputs_embeds_data_.assign(seq_len * hidden_dim, 0.0f);

  // mm_mask [1, seq_len]: zeros select the internal text embeddings everywhere.
  mm_mask_data_.assign(seq_len, 0.0f);

  // Inject vision embeddings at image-token positions (Gemma 4 multimodal).
  // Only during prefill (seq_len > 1); decode tokens are never image tokens.
  // Each position whose token is the image-token id consumes the next packed
  // vision embedding and flips its mm_mask bit so the decoder picks
  // inputs_embeds at that position. vision_embed_idx_ persists across
  // micro-batch prefill chunks.
  // The token id is also replaced with pad: the exported text decoder looks up
  // per-layer embeddings (PLE) from input_ids directly, and HF / the export
  // reference (Gemma4Model_ENN.forward) feed pad ids at multimodal positions.
  // input_tokens_tensor_ aliases input_tokens_, so the in-place edit is seen
  // by the model.
  size_t vision_idx = vision_embed_idx_;
  if (!vision_embeds_.empty() && seq_len > 1) {
    for (size_t i = 0; i < seq_len; ++i) {
      if (static_cast<int64_t>(input_tokens_[i]) == kGemma4ImageTokenId &&
          vision_idx < static_cast<size_t>(num_vision_tokens_)) {
        std::copy_n(
            vision_embeds_.data() + vision_idx * hidden_dim,
            hidden_dim,
            inputs_embeds_data_.data() + i * hidden_dim);
        mm_mask_data_[i] = 1.0f;
        input_tokens_[i] = kGemma4PadId;
        ++vision_idx;
      }
    }
  }
  vision_embed_idx_ = vision_idx;

  inputs_embeds_tensor_ = executorch::extension::from_blob(
      inputs_embeds_data_.data(),
      {1,
       static_cast<executorch::aten::SizesType>(seq_len),
       static_cast<executorch::aten::SizesType>(hidden_dim)});
  inputs.emplace_back(inputs_embeds_tensor_);

  mm_mask_tensor_ = executorch::extension::from_blob(
      mm_mask_data_.data(),
      {1, static_cast<executorch::aten::SizesType>(seq_len)});
  inputs.emplace_back(mm_mask_tensor_);
}

} // namespace example
