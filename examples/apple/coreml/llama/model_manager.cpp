/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include "model_manager.hpp"

#include <executorch/extension/tensor/tensor.h>
#include <executorch/runtime/platform/log.h>

#include <algorithm>
#include <chrono>
#include <filesystem>
#include <sstream>
#include <stdexcept>

namespace executorch {
namespace extension {

namespace fs = std::filesystem;

using runtime::Error;
using runtime::EValue;
using runtime::MethodMeta;
using runtime::Result;
using aten::ScalarType;
using aten::Tensor;

ModelManager::ModelManager(const std::string& model_path)
    : n_layers_(0),
        max_batch_size_(0),
        n_kv_heads_(0),
        cache_size_(0),
        head_dim_(0),
        seq_length_(0),
        max_seq_length_(0) {
  // Infer number of layers from the single transformer_block.pte
  n_layers_ = infer_n_layers(model_path);
  if (n_layers_ == 0) {
    std::stringstream ss;
    ss << "No transformer_block.pte file found or invalid output count in directory: " << model_path;
    throw std::runtime_error(ss.str());
  }

  ET_LOG(Info, "Detected piecewise model with %lld layers", (long long)n_layers_);

  // Load all model pieces
  Error error = load_input_block(model_path);
  if (error != Error::Ok) {
    std::stringstream ss;
    ss << "Failed to load input block: " << static_cast<uint32_t>(error);
    throw std::runtime_error(ss.str());
  }

  error = load_transformer_blocks(model_path);
  if (error != Error::Ok) {
    std::stringstream ss;
    ss << "Failed to load transformer blocks: " << static_cast<uint32_t>(error);
    throw std::runtime_error(ss.str());
  }

  error = load_output_block(model_path);
  if (error != Error::Ok) {
    std::stringstream ss;
    ss << "Failed to load output block: " << static_cast<uint32_t>(error);
    throw std::runtime_error(ss.str());
  }

  // Extract metadata from the first transformer block
  error = extract_metadata();
  if (error != Error::Ok) {
    std::stringstream ss;
    ss << "Failed to extract metadata: " << static_cast<uint32_t>(error);
    throw std::runtime_error(ss.str());
  }

  ET_LOG(Info, "Model loaded successfully");
  ET_LOG(Info, "  max_batch_size: %lld", (long long)max_batch_size_);
  ET_LOG(Info, "  n_kv_heads: %lld", (long long)n_kv_heads_);
  ET_LOG(Info, "  cache_size: %lld", (long long)cache_size_);
  ET_LOG(Info, "  head_dim: %lld", (long long)head_dim_);
  ET_LOG(Info, "  seq_length: %lld", (long long)seq_length_);
  ET_LOG(Info, "  max_seq_length: %lld", (long long)max_seq_length_);

  // Allocate KV caches and attention mask
  // Using fp16 (2 bytes per element)
  size_t bytes_per_elem = 2;
  size_t cache_elem_count = max_batch_size_ * n_kv_heads_ * cache_size_ * head_dim_;
  size_t cache_byte_count = cache_elem_count * bytes_per_elem;

  k_caches_data_.resize(n_layers_);
  v_caches_data_.resize(n_layers_);
  for (int64_t i = 0; i < n_layers_; ++i) {
    // Allocate as bytes and initialize to zero
    k_caches_data_[i].resize(cache_byte_count, 0);
    v_caches_data_[i].resize(cache_byte_count, 0);
  }

  // Allocate attention mask
  size_t mask_elem_count = seq_length_ * max_seq_length_;
  size_t mask_byte_count = mask_elem_count * bytes_per_elem;
  mask_data_.resize(mask_byte_count, 0);

  // attn_cache = minus_infinity * torch.ones(seq_length, cache_size)  // attn for past tokens
  // attn_seq = torch.triu(minus_infinity * torch.ones(seq_length, seq_length), diagonal=1)  // attn for current tokens
  // attn_mask = concat([attn_cache, attn_seq], dim=-1)

  // Use -30000.0 as minus_infinity value to prevent under/overflow in FP16
  exec_aten::Half* mask_ptr = reinterpret_cast<exec_aten::Half*>(mask_data_.data());
  exec_aten::Half minus_inf_half(-30000.0f);
  exec_aten::Half zero_half(0.0f);

  for (int64_t row = 0; row < seq_length_; ++row) {
    // First cache_size columns: all minus_inf
    for (int64_t col = 0; col < cache_size_; ++col) {
      mask_ptr[row * max_seq_length_ + col] = minus_inf_half;
    }
    // Next seq_length columns: upper triangular (diagonal=1)
    // diagonal=1 means diagonal and below are 0, above diagonal is minus_inf
    for (int64_t col = 0; col < seq_length_; ++col) {
      if (col > row) {
        mask_ptr[row * max_seq_length_ + cache_size_ + col] = minus_inf_half;
      } else {
        mask_ptr[row * max_seq_length_ + cache_size_ + col] = zero_half;
      }
    }
  }

  // Initialize cache position tracking
  cache_pos_ = 0;

  // Pre-create tensor views for k/v caches and attention mask
  // This avoids overhead of creating tensors in the forward() loop
  k_caches_.reserve(n_layers_);
  v_caches_.reserve(n_layers_);

  for (int64_t i = 0; i < n_layers_; ++i) {
    k_caches_.push_back(from_blob(
        k_caches_data_[i].data(),
        {static_cast<int>(max_batch_size_), static_cast<int>(n_kv_heads_),
         static_cast<int>(cache_size_), static_cast<int>(head_dim_)},
        ScalarType::Half));

    v_caches_.push_back(from_blob(
        v_caches_data_[i].data(),
        {static_cast<int>(max_batch_size_), static_cast<int>(n_kv_heads_),
         static_cast<int>(cache_size_), static_cast<int>(head_dim_)},
        ScalarType::Half));
  }

  attn_mask_ = from_blob(
      mask_data_.data(),
      {static_cast<int>(seq_length_), static_cast<int>(max_seq_length_)},
      ScalarType::Half);

  // Pre-allocate buffers for forward() chunking/padding
  chunk_buffer_.resize(seq_length_, 0);
  pos_buffer_.resize(1, 0);
  input_length_buffer_.resize(1, 0);

  // Pre-allocate EValue vector for layer execution (6 inputs per layer)
  layer_inputs_.reserve(6);

  ET_LOG(Info, "Allocated KV caches and attention mask");
}

int64_t ModelManager::infer_n_layers(const std::string& model_path) {
  // Check for the single transformer_block.pte file
  std::string single_block_path = model_path + "/transformer_block.pte";
  if (!fs::exists(single_block_path)) {
    ET_LOG(Error, "transformer_block.pte not found in directory: %s", model_path.c_str());
    return 0;
  }

  // Load temporarily to get metadata
  auto temp_module = std::make_unique<executorch::extension::Module>(single_block_path);
  Error error = temp_module->load();
  if (error != Error::Ok) {
    ET_LOG(Error, "Failed to load transformer block: 0x%" PRIx32, static_cast<uint32_t>(error));
    return 0;
  }

  error = temp_module->load_method("forward");
  if (error != Error::Ok) {
    ET_LOG(Error, "Failed to load forward method: 0x%" PRIx32, static_cast<uint32_t>(error));
    return 0;
  }

  // Get method metadata to determine output count
  auto meta_result = temp_module->method_meta("forward");
  if (!meta_result.ok()) {
    ET_LOG(Error, "Failed to get method metadata: 0x%" PRIx32, static_cast<uint32_t>(meta_result.error()));
    return 0;
  }

  const MethodMeta& metadata = *meta_result;
  size_t num_outputs = metadata.num_outputs();

  // Output format: (h, k_cache_0, ..., k_cache_n, v_cache_0, ..., v_cache_n)
  // So: num_outputs = 1 + n_layers + n_layers = 1 + 2*n_layers
  // Therefore: n_layers = (num_outputs - 1) / 2
  if ((num_outputs - 1) % 2 != 0) {
    ET_LOG(Error, "Invalid output count %zu for transformer block", num_outputs);
    return 0;
  }

  int64_t n_layers = (num_outputs - 1) / 2;
  ET_LOG(Info, "Found transformer_block.pte with %lld layers", (long long)n_layers);
  return n_layers;
}

Error ModelManager::load_input_block(const std::string& model_path) {
  std::string input_block_path = model_path + "/input_block.pte";

  if (!fs::exists(input_block_path)) {
    ET_LOG(Error, "Input block not found: %s", input_block_path.c_str());
    return Error::InvalidArgument;
  }

  input_proj_module_ = std::make_unique<executorch::extension::Module>(input_block_path);
  Error error = input_proj_module_->load();
  if (error != Error::Ok) {
    ET_LOG(Error, "Failed to load input block: 0x%" PRIx32, static_cast<uint32_t>(error));
    return error;
  }

  error = input_proj_module_->load_method("forward");
  if (error != Error::Ok) {
    ET_LOG(Error, "Failed to load forward method for input block: 0x%" PRIx32, static_cast<uint32_t>(error));
    return error;
  }

  ET_LOG(Info, "Loaded input block");
  return Error::Ok;
}

Error ModelManager::load_transformer_blocks(const std::string& model_path) {
  // Load single transformer_block.pte file
  std::string block_path = model_path + "/transformer_block.pte";

  if (!fs::exists(block_path)) {
    ET_LOG(Error, "Transformer block not found: %s", block_path.c_str());
    return Error::InvalidArgument;
  }

  auto module = std::make_unique<executorch::extension::Module>(block_path);
  Error error = module->load();
  if (error != Error::Ok) {
    ET_LOG(Error, "Failed to load transformer block: 0x%" PRIx32, static_cast<uint32_t>(error));
    return error;
  }

  error = module->load_method("forward");
  if (error != Error::Ok) {
    ET_LOG(Error, "Failed to load forward method for transformer block: 0x%" PRIx32, static_cast<uint32_t>(error));
    return error;
  }

  transformer_modules_.push_back(std::move(module));
  ET_LOG(Info, "Loaded transformer block");

  return Error::Ok;
}

Error ModelManager::load_output_block(const std::string& model_path) {
  std::string output_block_path = model_path + "/output_block.pte";

  if (!fs::exists(output_block_path)) {
    ET_LOG(Error, "Output block not found: %s", output_block_path.c_str());
    return Error::InvalidArgument;
  }

  output_proj_module_ = std::make_unique<executorch::extension::Module>(output_block_path);
  Error error = output_proj_module_->load();
  if (error != Error::Ok) {
    ET_LOG(Error, "Failed to load output block: 0x%" PRIx32, static_cast<uint32_t>(error));
    return error;
  }

  error = output_proj_module_->load_method("forward");
  if (error != Error::Ok) {
    ET_LOG(Error, "Failed to load forward method for output block: 0x%" PRIx32, static_cast<uint32_t>(error));
    return error;
  }

  ET_LOG(Info, "Loaded output block");
  return Error::Ok;
}

Error ModelManager::extract_metadata() {
  if (transformer_modules_.empty()) {
    ET_LOG(Error, "No transformer modules loaded");
    return Error::InvalidState;
  }

  // Get metadata from the transformer block
  auto meta_result = transformer_modules_[0]->method_meta("forward");
  if (!meta_result.ok()) {
    ET_LOG(Error, "Failed to get method metadata: 0x%" PRIx32, static_cast<uint32_t>(meta_result.error()));
    return meta_result.error();
  }

  const MethodMeta& metadata = *meta_result;

  ET_LOG(Info, "Method metadata (transformer block):");
  ET_LOG(Info, "  Number of inputs: %zu", metadata.num_inputs());

  // Single transformer block with list interface: 3 + n_layers + n_layers + 1 = 4 + 2*n_layers
  size_t expected_inputs = 4 + 2 * n_layers_;

  if (metadata.num_inputs() != expected_inputs) {
    ET_LOG(Error, "Expected %zu inputs for transformer block (4 + 2*%lld layers), got %zu",
           expected_inputs, (long long)n_layers_, metadata.num_inputs());
    return Error::InvalidArgument;
  }

  // Extract k_cache input metadata (first k_cache is at index 3)
  size_t k_cache_index = 3;
  auto k_cache_meta_result = metadata.input_tensor_meta(k_cache_index);
  if (!k_cache_meta_result.ok()) {
    ET_LOG(Error, "Failed to get k_cache tensor metadata");
    return k_cache_meta_result.error();
  }

  const auto& k_cache_meta = *k_cache_meta_result;
  auto k_cache_sizes = k_cache_meta.sizes();

  if (k_cache_sizes.size() != 4) {
    ET_LOG(Error, "Expected 4 dimensions for k_cache, got %zu", k_cache_sizes.size());
    return Error::InvalidArgument;
  }

  max_batch_size_ = k_cache_sizes[0];
  n_kv_heads_ = k_cache_sizes[1];
  cache_size_ = k_cache_sizes[2];
  head_dim_ = k_cache_sizes[3];

  // Assert that the model uses fp16
  ScalarType model_dtype = k_cache_meta.scalar_type();
  if (model_dtype != ScalarType::Half) {
    ET_LOG(Error, "Model must use fp16 (Half) dtype, got dtype: %d", static_cast<int>(model_dtype));
    return Error::InvalidArgument;
  }

  ET_LOG(Info, "  k_cache shape: [%lld, %lld, %lld, %lld]",
         (long long)max_batch_size_,
         (long long)n_kv_heads_,
         (long long)cache_size_,
         (long long)head_dim_);
  ET_LOG(Info, "  dtype: fp16");

  // Extract mask input metadata (mask is at index 3 + 2*n_layers)
  size_t mask_index = 3 + 2 * n_layers_;
  auto mask_meta_result = metadata.input_tensor_meta(mask_index);
  if (!mask_meta_result.ok()) {
    ET_LOG(Error, "Failed to get mask tensor metadata at index %zu", mask_index);
    return mask_meta_result.error();
  }

  const auto& mask_meta = *mask_meta_result;
  auto mask_sizes = mask_meta.sizes();

  if (mask_sizes.size() != 2) {
    ET_LOG(Error, "Expected 2 dimensions for mask, got %zu", mask_sizes.size());
    return Error::InvalidArgument;
  }

  seq_length_ = mask_sizes[0];
  max_seq_length_ = mask_sizes[1];

  ET_LOG(Info, "  mask shape: [%lld, %lld]",
         (long long)seq_length_,
         (long long)max_seq_length_);

  return Error::Ok;
}

void ModelManager::reset_caches() {
  // Reset KV caches to zero
  for (int64_t i = 0; i < n_layers_; ++i) {
    std::fill(k_caches_data_[i].begin(), k_caches_data_[i].end(), 0.0f);
    std::fill(v_caches_data_[i].begin(), v_caches_data_[i].end(), 0.0f);
  }
  ET_LOG(Info, "Reset KV caches");
}

Result<Tensor> ModelManager::forward(
    const Tensor& tokens,
    const Tensor& input_pos) {
  // Validate inputs
  if (tokens.dim() != 2) {
    ET_LOG(Error, "tokens tensor must be 2D, got %zd dimensions", tokens.dim());
    return Error::InvalidArgument;
  }

  if (input_pos.dim() != 1 || input_pos.size(0) != 1) {
    ET_LOG(Error, "input_pos tensor must be shape [1], got dim=%zd size=%lld",
           input_pos.dim(), (long long)(input_pos.dim() > 0 ? input_pos.size(0) : 0));
    return Error::InvalidArgument;
  }

  // Get input_pos value
  int64_t input_pos_value = input_pos.const_data_ptr<int64_t>()[0];

  // For inference, we assume all tokens in the input are valid
  // (the caller is responsible for proper padding)
  const int64_t* tokens_data = tokens.const_data_ptr<int64_t>();
  int64_t token_dim_size = tokens.size(1);  // Second dimension of [batch, num_tokens]
  int64_t num_tokens = token_dim_size;  // Use full dimension as num_tokens

  // Case 1: num_tokens <= seq_length - pad and process once
  if (num_tokens <= seq_length_) {
    // Fill chunk_buffer_ with tokens, rest is already zeros (padded)
    std::fill(chunk_buffer_.begin(), chunk_buffer_.end(), 0);
    std::copy(tokens_data, tokens_data + num_tokens, chunk_buffer_.begin());

    auto chunk_tensor = from_blob(
        chunk_buffer_.data(),
        {1, static_cast<int>(seq_length_)},
        ScalarType::Long);

    pos_buffer_[0] = input_pos_value;
    auto pos_tensor = from_blob(
        pos_buffer_.data(),
        {1},
        ScalarType::Long);

    // Pass actual num_tokens to forward_() so it can pass it to output_block
    return forward_(*chunk_tensor, *pos_tensor, num_tokens);
  }

  // Case 2: num_tokens > seq_length - process in chunks
  std::optional<Tensor> last_logits;
  int64_t current_pos = input_pos_value;

  for (int64_t offset = 0; offset < num_tokens; offset += seq_length_) {
    int64_t chunk_size = std::min(seq_length_, num_tokens - offset);

    // Clear buffer and fill with chunk
    std::fill(chunk_buffer_.begin(), chunk_buffer_.end(), 0);
    std::copy(tokens_data + offset, tokens_data + offset + chunk_size, chunk_buffer_.begin());

    auto chunk_tensor = from_blob(
        chunk_buffer_.data(),
        {1, static_cast<int>(seq_length_)},
        ScalarType::Long);

    pos_buffer_[0] = current_pos;
    auto pos_tensor = from_blob(
        pos_buffer_.data(),
        {1},
        ScalarType::Long);

    // Pass actual chunk_size to forward_() so it can pass it to output_block
    auto result = forward_(*chunk_tensor, *pos_tensor, chunk_size);
    if (!result.ok()) {
      return result;
    }

    last_logits = std::move(*result);
    current_pos += chunk_size;
  }

  return std::move(*last_logits);
}

Result<Tensor> ModelManager::forward_(
    const Tensor& tokens,
    const Tensor& input_pos,
    int64_t num_tokens) {
  // Validate that tokens is exactly seq_length
  if (tokens.dim() != 2 || tokens.size(1) != seq_length_) {
    ET_LOG(Error, "forward_() requires tokens shape [1, %lld], got [%lld, %lld]",
           (long long)seq_length_,
           (long long)tokens.size(0),
           (long long)tokens.size(1));
    return Error::InvalidArgument;
  }

  // Extract input_pos value
  if (input_pos.dim() != 1 || input_pos.size(0) != 1) {
    ET_LOG(Error, "input_pos tensor must be shape [1], got dim=%zd size=%lld",
           input_pos.dim(), (long long)(input_pos.dim() > 0 ? input_pos.size(0) : 0));
    return Error::InvalidArgument;
  }
  int64_t input_pos_value = input_pos.const_data_ptr<int64_t>()[0];

  // Use the num_tokens parameter passed from forward()
  // This is the actual number of valid (non-padding) tokens in the input

  if (input_pos_value + num_tokens > max_seq_length_) {
    ET_LOG(Error, "Position (%lld) + num_tokens (%lld) exceeds max_seq_length (%lld)",
           (long long)input_pos_value, (long long)num_tokens, (long long)max_seq_length_);
    return Error::InvalidArgument;
  }

  // Create input_length tensor using pre-allocated buffer
  input_length_buffer_[0] = num_tokens;
  auto input_length_tensor = from_blob(
      input_length_buffer_.data(),
      {1},
      ScalarType::Long);

  // Run input block: (tokens, input_pos) -> (h, freqs_cos, freqs_sin)
  auto input_result = input_proj_module_->execute(
      "forward",
      {EValue(tokens), EValue(input_pos)});

  if (!input_result.ok()) {
    ET_LOG(Error, "Input block execution failed: 0x%" PRIx32,
           static_cast<uint32_t>(input_result.error()));
    return input_result.error();
  }

  std::vector<EValue> input_outputs = std::move(*input_result);
  if (input_outputs.size() != 3) {
    ET_LOG(Error, "Expected 3 outputs from input block, got %zu", input_outputs.size());
    return Error::InvalidState;
  }

  Tensor h = input_outputs[0].toTensor();
  Tensor freqs_cos = input_outputs[1].toTensor();
  Tensor freqs_sin = input_outputs[2].toTensor();

  int64_t amount_to_copy = std::min(num_tokens, cache_size_ - cache_pos_);

  update_mask(input_pos_value, amount_to_copy);

  // Single transformer block mode - one module processes all layers at once
  layer_inputs_.clear();
  layer_inputs_.emplace_back(h);
  layer_inputs_.emplace_back(freqs_cos);
  layer_inputs_.emplace_back(freqs_sin);

  // Pass all k_caches and v_caches as separate EValues
  for (int64_t i = 0; i < n_layers_; ++i) {
    layer_inputs_.emplace_back(k_caches_[i]);
  }
  for (int64_t i = 0; i < n_layers_; ++i) {
    layer_inputs_.emplace_back(v_caches_[i]);
  }
  layer_inputs_.emplace_back(attn_mask_);

  auto result = transformer_modules_[0]->execute("forward", layer_inputs_);
  if (!result.ok()) {
    ET_LOG(Error, "Single transformer block execution failed: 0x%" PRIx32,
           static_cast<uint32_t>(result.error()));
    return result.error();
  }

  std::vector<EValue> outputs = std::move(*result);
  size_t expected_outputs = 1 + 2 * n_layers_;
  if (outputs.size() != expected_outputs) {
    ET_LOG(Error, "Expected %zu outputs from single transformer block (1 + 2*%lld layers), got %zu",
           expected_outputs, (long long)n_layers_, outputs.size());
    return Error::InvalidState;
  }

  h = outputs[0].toTensor();

  // Update caches for all layers
  for (int64_t i = 0; i < n_layers_; ++i) {
    Tensor new_k = outputs[1 + i].toTensor();
    Tensor new_v = outputs[1 + n_layers_ + i].toTensor();
    update_cache(i, amount_to_copy, new_k, new_v);
  }

  // Update cache position after all layers
  cache_pos_ += amount_to_copy;
  if (cache_pos_ >= cache_size_) {
    cache_pos_ = 0;
  }

  // Run output block: (h, input_length) -> (logits,)
  auto output_result = output_proj_module_->execute(
      "forward",
      {EValue(h), EValue(*input_length_tensor)});

  if (!output_result.ok()) {
    ET_LOG(Error, "Output block execution failed: 0x%" PRIx32,
           static_cast<uint32_t>(output_result.error()));
    return output_result.error();
  }

  std::vector<EValue> output_outputs = std::move(*output_result);
  if (output_outputs.size() != 1) {
    ET_LOG(Error, "Expected 1 output from output block, got %zu", output_outputs.size());
    return Error::InvalidState;
  }

  return output_outputs[0].toTensor();
}

void ModelManager::update_cache(
    int64_t layer_id,
    int64_t amount_to_copy,
    const Tensor& new_k_cache,
    const Tensor& new_v_cache) {
  // Using fp16 (2 bytes per element)
  size_t elem_size = 2;
  size_t row_bytes = head_dim_ * elem_size;

  const char* new_k_bytes = reinterpret_cast<const char*>(new_k_cache.const_data_ptr());
  const char* new_v_bytes = reinterpret_cast<const char*>(new_v_cache.const_data_ptr());
  char* k_cache_bytes = reinterpret_cast<char*>(k_caches_data_[layer_id].data());
  char* v_cache_bytes = reinterpret_cast<char*>(v_caches_data_[layer_id].data());

  // Copy new_cache[:, :, 0:amount_to_copy, :] -> cache[:, :, cache_pos_:cache_pos_+amount_to_copy, :]
  // Optimization: Copy all positions for each (batch, head) in a single memcpy
  // since positions are contiguous in memory
  size_t copy_bytes = amount_to_copy * row_bytes;

  for (int64_t batch = 0; batch < max_batch_size_; ++batch) {
    for (int64_t head = 0; head < n_kv_heads_; ++head) {
      // Source: new_cache[batch, head, 0:amount_to_copy, :]
      size_t src_offset = ((batch * n_kv_heads_ + head) * seq_length_) * row_bytes;
      // Dest: persistent_cache[batch, head, cache_pos_:cache_pos_+amount_to_copy, :]
      size_t dst_offset = ((batch * n_kv_heads_ + head) * cache_size_ + cache_pos_) * row_bytes;

      std::memcpy(k_cache_bytes + dst_offset, new_k_bytes + src_offset, copy_bytes);
      std::memcpy(v_cache_bytes + dst_offset, new_v_bytes + src_offset, copy_bytes);
    }
  }
}

void ModelManager::update_mask(int64_t input_pos, int64_t amount_to_copy) {
  if (input_pos <= cache_size_) {
    char* mask_bytes = reinterpret_cast<char*>(mask_data_.data());
    size_t elem_size = 2;  // fp16 uses 2 bytes

    int64_t num_cols_to_zero = std::min(amount_to_copy, max_seq_length_ - input_pos);
    for (int64_t row = 0; row < seq_length_; ++row) {
      size_t offset = (row * max_seq_length_ + input_pos) * elem_size;
      size_t count = num_cols_to_zero * elem_size;
      std::memset(mask_bytes + offset, 0, count);
    }
  }
}

} // namespace extension
} // namespace executorch
