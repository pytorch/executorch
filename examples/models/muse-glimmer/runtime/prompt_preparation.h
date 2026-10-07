/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */
#pragma once

#include <executorch/examples/models/muse-glimmer/runtime/engine/muse_glimmer_vision_runtime.h>
#include <executorch/extension/llm/serving/types.h>
#include <nlohmann/json_fwd.hpp>
#include <memory>
#include <vector>

namespace executorch::extension::llm {

// One instance per executor. Sharing this immutable value with the CPU preparer
// binds its payloads to that exact executor, not just compatible tensor shapes.
struct MuseGlimmerPreparationSpec final {
  const aten::ScalarType activation_dtype;
  const int32_t hidden_dim;
  const int32_t max_context_length;
  const int32_t max_forward_tokens;
  const int32_t vocab_size;
  const bool has_vision;
  const int64_t max_soft_tokens;
  const MuseGlimmerImageLimits image_limits;
  const batching::Token bos_id = 200000;
  const batching::Token patch_id = 200092;
};

struct MuseGlimmerImageSpan {
  size_t offset;
  size_t size;
};

class MuseGlimmerMaterializer;

class MuseGlimmerPreparedInput final : public batching::PreparedInput {
 public:
  MuseGlimmerPreparedInput(
      std::shared_ptr<const MuseGlimmerPreparationSpec> spec,
      std::vector<batching::Token> tokens,
      MuseGlimmerRGBImage image,
      MuseGlimmerImageGrid grid,
      MuseGlimmerImageSpan image_span);

  const void* kind() const override;
  size_t size() const override {
    return tokens_.size();
  }
  static const void* kind_tag();
  bool compatible(const MuseGlimmerPreparationSpec& spec) const;
  batching::Token previous_token() const {
    return tokens_.back();
  }
  const MuseGlimmerImageSpan& image_span() const {
    return image_span_;
  }

 private:
  friend class MuseGlimmerMaterializer;
  const std::shared_ptr<const MuseGlimmerPreparationSpec> spec_;
  const std::vector<batching::Token> tokens_;
  const MuseGlimmerRGBImage image_;
  const MuseGlimmerImageGrid grid_;
  const MuseGlimmerImageSpan image_span_;
  // Only the bound executor's engine thread may read/write this host cache.
  // Its destruction is independent of Module, delegates, and engine lifetime.
  mutable std::vector<uint16_t> image_embeddings_;
};

serving::PromptPreparationResult prepare_muse_glimmer_prompt(
    const nlohmann::json& request,
    const serving::PromptPreparationContext& context,
    std::shared_ptr<const MuseGlimmerPreparationSpec> spec);

} // namespace executorch::extension::llm
