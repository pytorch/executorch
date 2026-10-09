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
#include <optional>
#include <utility>
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
    return backing_->tokens_->size() - start_;
  }
  static const void* kind_tag();
  bool compatible(const MuseGlimmerPreparationSpec& spec) const;
  batching::PreparedInputPtr suffix(size_t start) const override;
  batching::PrefixIdentityPtr prefix_identity() const override;

 private:
  friend class MuseGlimmerMaterializer;
  struct Backing {
    Backing(
        std::shared_ptr<const MuseGlimmerPreparationSpec> spec,
        std::vector<batching::Token> tokens,
        MuseGlimmerRGBImage image,
        MuseGlimmerImageGrid grid,
        MuseGlimmerImageSpan image_span);
    bool validate_structure() const;

    const std::shared_ptr<const MuseGlimmerPreparationSpec> spec_;
    // Independently owned: retained identity must not retain image backing.
    const batching::TokenInputPtr tokens_;
    mutable std::optional<MuseGlimmerRGBImage> image_;
    const MuseGlimmerImageGrid grid_;
    const MuseGlimmerImageSpan image_span_;
    const bool valid_;
    const batching::ContentKey image_key_;
    // Only the bound executor's engine thread accesses the lazy image/cache.
    mutable std::vector<uint16_t> image_embeddings_;
  };
  MuseGlimmerPreparedInput(std::shared_ptr<const Backing> backing, size_t start)
      : backing_(std::move(backing)), start_(start) {}
  const std::shared_ptr<const Backing> backing_;
  const size_t start_ = 0;
};

serving::PromptPreparationResult prepare_muse_glimmer_prompt(
    const nlohmann::json& request,
    const serving::PromptPreparationContext& context,
    std::shared_ptr<const MuseGlimmerPreparationSpec> spec);

serving::ModelPreparationResult prepare_muse_glimmer_input(
    const serving::PromptPreparationContext& context,
    const serving::PromptInput& input,
    std::shared_ptr<const MuseGlimmerPreparationSpec> spec);

} // namespace executorch::extension::llm
