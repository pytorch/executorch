/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */
#pragma once

#include <executorch/examples/models/muse-glimmer/runtime/prompt_preparation.h>
#include <executorch/extension/tensor/tensor_ptr.h>
#include <functional>

namespace executorch::extension::llm {

// Executor-private physical slice assembly. Callbacks are synchronous engine
// operations, also allowing model-independent tests of layout and ownership.
class MuseGlimmerMaterializer final {
 public:
  using EncodeImage = std::function<runtime::Result<PreparedMuseGlimmerImage>(
      const MuseGlimmerRGBImage&)>;
  using EmbedText =
      std::function<runtime::Result<TensorPtr>(const std::vector<int64_t>&)>;

  explicit MuseGlimmerMaterializer(
      std::shared_ptr<const MuseGlimmerPreparationSpec> spec)
      : spec_(std::move(spec)) {}
  bool accepts(const batching::PreparedInput& input) const;
  runtime::Result<TensorPtr> materialize(
      const std::vector<batching::Input>& inputs,
      const EncodeImage& encode_image,
      const EmbedText& embed_text) const;

 private:
  std::shared_ptr<const MuseGlimmerPreparationSpec> spec_;
};

} // namespace executorch::extension::llm
