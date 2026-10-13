/*
 * Copyright (c) 2025 Samsung Electronics Co. LTD
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <executorch/examples/samsung/executor_runner/llama/runner/rotary_embedding.h>
#include <cstring>
#include <fstream>
#include <iostream>

namespace example {

namespace runtime = ::executorch::runtime;

RotaryEmbedding::RotaryEmbedding(Config config) : config_(config) {}

runtime::Error RotaryEmbedding::load(
    std::optional<const std::string> pos_embedding_path) {
  if (!pos_embedding_path.has_value()) {
    ET_LOG(
        Error,
        "Missing position embedding path. Verify that the file path is provided in the configuration.");
    return runtime::Error::InvalidArgument;
  }

  const size_t total_bytes =
      config_.max_seq_len * config_.head_dim * sizeof(float);
  global_cos_ = std::make_unique<char[]>(total_bytes);
  global_sin_ = std::make_unique<char[]>(total_bytes);
  if (config_.strategy == EmbeddingStrategy::kGlobalAndLocal) {
    local_cos_ = std::make_unique<char[]>(total_bytes);
    local_sin_ = std::make_unique<char[]>(total_bytes);
  }

  std::ifstream file(pos_embedding_path.value(), std::ios::binary);
  if (!file) {
    ET_LOG(
        Error,
        "File I/O error: Cannot open position embedding file at '%s'.",
        pos_embedding_path.value().c_str());
    return runtime::Error::InvalidArgument;
  }

  file.seekg(0, std::ios::end);
  size_t file_size = file.tellg();
  file.seekg(0, std::ios::beg);

  int bin_num =
      (config_.strategy == EmbeddingStrategy::kGlobalAndLocal) ? 4 : 2;
  size_t expected_size = total_bytes * bin_num;
  if (file_size != expected_size) {
    ET_LOG(
        Error,
        "Embedding file size mismatch. Expected %zu bytes (total_bytes: %zu * bin_num: %d based on strategy), but got %zu bytes. File may be corrupted or from a different model version.",
        expected_size,
        total_bytes,
        bin_num,
        file_size);
    return runtime::Error::InvalidArgument;
  }

  file.read(global_cos_.get(), total_bytes);
  file.read(global_sin_.get(), total_bytes);
  if (config_.strategy == EmbeddingStrategy::kGlobalAndLocal) {
    file.read(local_cos_.get(), total_bytes);
    file.read(local_sin_.get(), total_bytes);
  }

  file.close();
  return runtime::Error::Ok;
}

runtime::Error RotaryEmbedding::update(
    const std::vector<int64_t>& pos_ids,
    float* global_cos,
    float* global_sin,
    float* local_cos,
    float* local_sin) {
  if (pos_ids.empty()) {
    return runtime::Error::InvalidArgument;
  }
  if (global_cos == nullptr || global_sin == nullptr) {
    return runtime::Error::InvalidArgument;
  }
  if (config_.strategy == EmbeddingStrategy::kGlobalAndLocal &&
      (local_cos == nullptr || local_sin == nullptr)) {
    return runtime::Error::InvalidArgument;
  }

  size_t pos = static_cast<size_t>(pos_ids[0]);
  size_t length = pos_ids.size();
  size_t head_dim = config_.head_dim;
  size_t bytes_to_copy = length * head_dim * sizeof(float);
  size_t element_offset = pos * head_dim * sizeof(float);

  std::memcpy(global_cos, global_cos_.get() + element_offset, bytes_to_copy);
  std::memcpy(global_sin, global_sin_.get() + element_offset, bytes_to_copy);

  if (config_.strategy == EmbeddingStrategy::kGlobalAndLocal) {
    std::memcpy(local_cos, local_cos_.get() + element_offset, bytes_to_copy);
    std::memcpy(local_sin, local_sin_.get() + element_offset, bytes_to_copy);
  }

  return runtime::Error::Ok;
}

} // namespace example
