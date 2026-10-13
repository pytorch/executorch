/*
 * Copyright (c) 2025 Samsung Electronics Co. LTD
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <executorch/examples/samsung/executor_runner/llama/runner/constants.h>
#include <executorch/runtime/core/error.h>
#include <memory>
#include <optional>
#include <string>
#include <vector>

namespace example {

namespace runtime = ::executorch::runtime;

class RotaryEmbedding {
 public:
  struct Config {
    size_t head_dim;
    size_t max_seq_len;
    EmbeddingStrategy strategy = EmbeddingStrategy::kGlobalOnly;
  };

  RotaryEmbedding(Config config);
  ~RotaryEmbedding() = default;

  runtime::Error load(std::optional<const std::string> pos_embedding_path);
  runtime::Error update(
      const std::vector<int64_t>& pos_ids,
      float* global_cos,
      float* global_sin,
      float* local_cos,
      float* local_sin);

 private:
  Config config_;
  std::unique_ptr<char[]> global_cos_;
  std::unique_ptr<char[]> global_sin_;
  std::unique_ptr<char[]> local_cos_;
  std::unique_ptr<char[]> local_sin_;
};

} // namespace example
