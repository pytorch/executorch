// Copyright (c) Meta Platforms, Inc. and affiliates.
// All rights reserved.
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#pragma once

#include <cstddef>
#include <cstdint>
#include <mutex>
#include <string>
#include <string_view>
#include <vector>

#include <executorch/extension/module/module.h>
#include <executorch/runtime/core/result.h>
#include <executorch/runtime/core/span.h>

namespace executorch::extension::llm {

struct PreparedMuseGlimmerImage {
  std::vector<uint16_t> embeddings;
  int64_t num_soft_tokens = 0;
  int64_t hidden_dim = 0;
  double vision_encoder_ms = 0.0;
};

struct MuseGlimmerVisionRuntimeConfig {
  Module* module = nullptr;
  std::mutex* execution_mutex = nullptr;
  std::string pos_embed_path;
  executorch::aten::ScalarType activation_dtype =
      executorch::aten::ScalarType::BFloat16;
  int64_t expected_hidden_dim = 0;
  int64_t max_image_tokens = 4096;
  size_t max_encoded_bytes = 32 * 1024 * 1024;
  int32_t max_image_dimension = 32768;
  int64_t max_image_pixels = 64 * 1024 * 1024;
};

class MuseGlimmerVisionRuntime final {
 public:
  explicit MuseGlimmerVisionRuntime(MuseGlimmerVisionRuntimeConfig config);

  ::executorch::runtime::Result<PreparedMuseGlimmerImage>
  prepare_image_from_file(const std::string& image_path) const;

  ::executorch::runtime::Result<PreparedMuseGlimmerImage>
  prepare_image_from_bytes(
      ::executorch::runtime::Span<const uint8_t> encoded_image) const;

  ::executorch::runtime::Result<PreparedMuseGlimmerImage>
  prepare_image_from_bytes(const std::vector<uint8_t>& encoded_image) const {
    return prepare_image_from_bytes(::executorch::runtime::Span<const uint8_t>(
        encoded_image.data(), encoded_image.size()));
  }

 private:
  ::executorch::runtime::Result<PreparedMuseGlimmerImage> prepare_decoded_image(
      const uint8_t* rgb,
      int32_t width,
      int32_t height) const;

  MuseGlimmerVisionRuntimeConfig config_;
  std::vector<float> pos_embed_table_;
};

::executorch::runtime::Result<std::vector<uint8_t>>
decode_muse_glimmer_base64_strict(
    std::string_view encoded,
    size_t max_decoded_bytes);

} // namespace executorch::extension::llm
