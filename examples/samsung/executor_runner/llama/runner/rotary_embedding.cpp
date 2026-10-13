/*
 * Copyright (c) 2025 Samsung Electronics Co. LTD
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <executorch/examples/samsung/executor_runner/llama/runner/rotary_embedding.h>
#include <fstream>
#include <iostream>
#include <cstring>

namespace example {

RotaryEmbedding::RotaryEmbedding(Config config) : config_(config) {}

executorch::runtime::Error RotaryEmbedding::load(std::optional<const std::string> pos_embedding_path) {
    if (!pos_embedding_path.has_value()) {
        ET_LOG(Error, "RotaryEmbedding not has value");
        return executorch::runtime::Error::Ok;
    }

    const size_t total_bytes = config_.max_seq_len * config_.head_dim * sizeof(float);
    global_cos_ = std::make_unique<char[]>(total_bytes);
    global_sin_ = std::make_unique<char[]>(total_bytes);
    if (config_.use_gemma) {
        local_cos_  = std::make_unique<char[]>(total_bytes);
        local_sin_  = std::make_unique<char[]>(total_bytes);
    }

    std::ifstream file(pos_embedding_path.value(), std::ios::binary);
    if (!file) {
        ET_LOG(Error, "can't open the file");
        return executorch::runtime::Error::Internal;
    }

    file.seekg(0, std::ios::end);
    size_t file_size = file.tellg();
    file.seekg(0, std::ios::beg);

    int bin_num = config_.use_gemma ? 4 : 2;
    if (file_size != total_bytes * bin_num) {
        ET_LOG(Error, "file size is not right");
        return executorch::runtime::Error::Internal;
    }

    file.read(global_cos_.get(), total_bytes);
    file.read(global_sin_.get(), total_bytes);
    if (config_.use_gemma) {
        file.read(local_cos_.get(),  total_bytes);
        file.read(local_sin_.get(),  total_bytes);
    }

    file.close();
    return executorch::runtime::Error::Ok;
}

executorch::runtime::Error RotaryEmbedding::update(const std::vector<int64_t>& pos_ids,
                                                float* global_cos, float* global_sin,
                                                float* local_cos, float* local_sin) {
    if (pos_ids.empty()) {
        return executorch::runtime::Error::Internal;
    }

    size_t pos = static_cast<size_t>(pos_ids[0]);
    size_t length = pos_ids.size();
    size_t head_dim = config_.head_dim;

    std::memcpy(global_cos, global_cos_.get() + pos * head_dim * sizeof(float), length * head_dim * sizeof(float));
    std::memcpy(global_sin, global_sin_.get() + pos * head_dim * sizeof(float), length * head_dim * sizeof(float));
    if (config_.use_gemma) {
        std::memcpy(local_cos, local_cos_.get() + pos * head_dim * sizeof(float), length * head_dim * sizeof(float));
        std::memcpy(local_sin, local_sin_.get() + pos * head_dim * sizeof(float), length * head_dim * sizeof(float));
    }

    return executorch::runtime::Error::Ok;
}

} // namespace example
