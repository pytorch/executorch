/*
 * Copyright (c) 2026 Samsung Electronics Co. LTD
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <executorch/examples/samsung/executor_runner/llm/runner/config.h>
#include <executorch/examples/samsung/executor_runner/llm/runner/image_utils.h>
#include <executorch/extension/module/module.h>

#include <memory>
#include <string>
#include <vector>

namespace example {

namespace runtime = ::executorch::runtime;
namespace extension = ::executorch::extension;

/**
 * Runs the Gemma 4 vision encoder (gemma4_vision.pte) on an input image and
 * produces LM-ready vision embeddings plus the number of valid (non-padding)
 * vision tokens.
 *
 * The vision encoder is a separate module from the text decoder. Its PTE has a
 * single "vision_encoder" method accepting pre-patchified pixel values
 * [1, max_patches, 768] and 2D patch positions [1, max_patches, 2], and
 * returning embeddings [1, pooled, hidden] plus a BOOL validity mask [1,
 * pooled].
 */
class VisionEncoder {
 public:
  explicit VisionEncoder(const std::string& model_path);

  runtime::Error load();

  /// Encode an image file and produce vision embeddings.
  /// On success fills vision_embeds_ ([1, pooled, hidden], float) and sets
  /// num_valid_tokens_ to the number of real (non-padding) vision tokens.
  runtime::Error encode(const std::string& image_path);

  const std::vector<float>& vision_embeds() const {
    return vision_embeds_;
  }
  int64_t num_valid_tokens() const {
    return num_valid_tokens_;
  }
  int64_t hidden_size() const {
    return hidden_size_;
  }
  bool loaded() const {
    return loaded_;
  }

 private:
  std::string model_path_;
  std::unique_ptr<extension::Module> module_;
  bool loaded_ = false;
  // Patch budget and position-id dtype the PTE was exported with; read from
  // the method metadata in load().
  int32_t max_soft_tokens_ = kVisionMaxSoftTokens;
  executorch::aten::ScalarType position_ids_dtype_ =
      executorch::aten::ScalarType::Long;
  int64_t hidden_size_ = 0;
  int64_t num_valid_tokens_ = 0;
  std::vector<float> vision_embeds_;
};

} // namespace example
