/*
 * Copyright (c) 2026 Samsung Electronics Co. LTD
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <memory>
#include <optional>
#include <string>
#include <vector>

#include <executorch/examples/samsung/executor_runner/llm/runner/config.h>
#include <executorch/runtime/core/error.h>

namespace example {

namespace runtime = ::executorch::runtime;

class RotaryEmbedding {
 public:
  RotaryEmbedding(const RunnerConfig& config);
  ~RotaryEmbedding() = default;

  runtime::Error load(std::optional<const std::string> pos_embedding_path);
  runtime::Error update(
      const std::vector<int64_t>& pos_ids,
      float* global_cos,
      float* global_sin,
      float* local_cos,
      float* local_sin);

 private:
  runtime::Error load_from_file(const std::string& path);
  runtime::Error load_by_precompute_rope_tables();
  void allocate_buffers();
  runtime::Error compute_rope_table(
      float* cos_ptr,
      float* sin_ptr,
      size_t seq_len,
      size_t dim,
      float base,
      float rope_pct);

  const RunnerConfig& config_;
  bool loaded_;

  size_t seq_len_;
  size_t normalized_global_head_dim_;
  size_t normalized_local_head_dim_;
  size_t global_bytes_;
  size_t local_bytes_;
  float base_freq_global_;
  float base_freq_local_;
  float rope_percentage_;

  std::unique_ptr<char[]> global_cos_;
  std::unique_ptr<char[]> global_sin_;
  std::unique_ptr<char[]> local_cos_;
  std::unique_ptr<char[]> local_sin_;
};

} // namespace example
