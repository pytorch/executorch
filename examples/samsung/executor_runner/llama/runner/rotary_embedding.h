/*
 * Copyright (c) 2025 Samsung Electronics Co. LTD
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <vector>
#include <string>
#include <memory>
#include <executorch/runtime/core/error.h>

namespace example {

class RotaryEmbedding {
 public:
  struct Config {
    size_t head_dim;
    size_t max_seq_len;
    bool use_gemma;
  };

  RotaryEmbedding(Config config);
  ~RotaryEmbedding() = default;

  executorch::runtime::Error load(std::optional<const std::string> pos_embedding_path);
  executorch::runtime::Error update(const std::vector<int64_t>& pos_ids,
                                    float* global_cos, float* global_sin,
                                    float* local_cos, float* local_sin);
 private:
  Config config_;
  std::unique_ptr<char[]> global_cos_;
  std::unique_ptr<char[]> global_sin_;
  std::unique_ptr<char[]> local_cos_;
  std::unique_ptr<char[]> local_sin_;
};

} // namespace example
