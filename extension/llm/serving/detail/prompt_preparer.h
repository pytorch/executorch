/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <executorch/extension/llm/serving/detail/image_header.h>
#include <executorch/extension/llm/serving/serving_runtime.h>
#include <executorch/runtime/core/result.h>
#include <pytorch/tokenizers/tokenizer.h>

#include <algorithm>
#include <cstddef>
#include <limits>
#include <optional>
#include <utility>

namespace executorch::extension::llm::serving::detail {

struct PreparedPrompt {
  std::vector<batching::Token> tokens;
  batching::PreparationInput input;
  std::optional<batching::Token> previous_token;
  bool has_images = false;
};

inline bool image_preparation_supported(
    const ServingRuntimeConfig& config,
    const batching::PreparationConfig& preparation) {
  const auto overhead =
      sizeof(batching::PreparationInput) + sizeof(MultimodalInput);
  return config.max_images == 1 && config.image_preprocessor &&
      config.max_image_encoded_bytes > 0 &&
      config.max_image_encoded_bytes <= 512 * 1024 &&
      config.max_image_dimension > 0 && config.max_image_dimension <= 4096 &&
      config.max_image_pixels > 0 &&
      config.max_image_pixels <= 4 * 1024 * 1024 &&
      preparation.max_images > 0 && config.max_image_preprocessed_bytes > 0 &&
      preparation.max_workspace_bytes >= overhead &&
      config.max_image_preprocessed_bytes <=
      preparation.max_workspace_bytes - overhead;
}

// CPU-only source preparation. Model preparation belongs to Runner. The
// optional preparation config is the executor's immutable construction config,
// not a per-call limit; image preparation requires it.
inline runtime::Result<PreparedPrompt> prepare_prompt(
    const tokenizers::Tokenizer& tokenizer,
    const PromptInput& input,
    std::size_t max_tokens = std::numeric_limits<batching::Position>::max(),
    const ServingRuntimeConfig* config = nullptr,
    const batching::PreparationConfig* preparation = nullptr) {
  const auto limit = std::min(
      max_tokens,
      static_cast<std::size_t>(std::numeric_limits<batching::Position>::max()));
  PreparedPrompt prepared;
  const EncodedImage* source = nullptr;
  for (const auto& segment : input.segments) {
    if (const auto* image = segment.try_get_encoded_image()) {
      if (!config || !preparation ||
          !image_preparation_supported(*config, *preparation) || source ||
          image->data.empty() ||
          image->data.size() > config->max_image_encoded_bytes ||
          !bounded_image_header(
              *image, config->max_image_dimension, config->max_image_pixels)) {
        return runtime::Error::InvalidArgument;
      }
      source = image;
    }
  }
  std::size_t image_index = 0;
  for (const auto& segment : input.segments) {
    if (segment.is_encoded_image()) {
      image_index = prepared.input.segments.size();
      prepared.input.segments.emplace_back(Image{});
      prepared.has_images = true;
      prepared.previous_token.reset();
      continue;
    }
    std::vector<batching::Token> encoded;
    const std::vector<batching::Token>* ids;
    if (const auto* text = segment.try_get_text()) {
      auto result = tokenizer.encode(*text, /*bos=*/0, /*eos=*/0);
      if (!result.ok()) {
        return runtime::Error::InvalidArgument;
      }
      encoded = std::move(*result);
      ids = &encoded;
    } else if (const auto* tokens = segment.try_get_tokens()) {
      ids = tokens;
    } else {
      return runtime::Error::InvalidArgument;
    }
    if (ids->size() > limit - prepared.tokens.size()) {
      return runtime::Error::InvalidArgument;
    }
    prepared.tokens.insert(prepared.tokens.end(), ids->begin(), ids->end());
    if (!ids->empty()) {
      prepared.input.segments.emplace_back(*ids);
      prepared.previous_token = ids->back();
    }
  }
  if (prepared.tokens.empty() && !prepared.has_images) {
    return runtime::Error::InvalidArgument;
  }
  if (source) {
    // Account for every token segment and final container capacity before the
    // trusted hook allocates pixels. The empty image occupies its final slot.
    const auto staged_bytes = preparation->input_retained_bytes(prepared.input);
    if (!staged_bytes ||
        config->max_image_preprocessed_bytes >
            preparation->max_workspace_bytes - *staged_bytes) {
      return runtime::Error::InvalidArgument;
    }
    auto result = config->image_preprocessor(*source);
    if (!result.ok()) {
      return runtime::Error::InvalidArgument;
    }
    auto& decoded = prepared.input.segments[image_index].get_image();
    decoded = std::move(*result);
    const auto capacity = decoded.is_uint8()
        ? decoded.get_uint8_data().capacity()
        : decoded.get_float_data().capacity();
    const auto element_bytes = decoded.is_float() ? sizeof(float) : 1;
    if (capacity > config->max_image_preprocessed_bytes / element_bytes ||
        decoded.width() <= 0 || decoded.height() <= 0 ||
        decoded.channels() <= 0 || decoded.channels() > 4) {
      return runtime::Error::InvalidArgument;
    }
    const auto width = static_cast<std::size_t>(decoded.width());
    const auto height = static_cast<std::size_t>(decoded.height());
    if (width > config->max_image_dimension ||
        height > config->max_image_dimension ||
        width > config->max_image_pixels / height) {
      return runtime::Error::InvalidArgument;
    }
    const auto pixels = width * height;
    const auto channels = static_cast<std::size_t>(decoded.channels());
    const auto size_limit = std::numeric_limits<std::size_t>::max();
    if (pixels > size_limit / channels ||
        pixels * channels > size_limit / element_bytes) {
      return runtime::Error::InvalidArgument;
    }
    const auto elements = pixels * channels;
    if ((decoded.is_uint8() && decoded.get_uint8_data().size() != elements) ||
        (decoded.is_float() && decoded.get_float_data().size() != elements)) {
      return runtime::Error::InvalidArgument;
    }
  }
  return prepared;
}

} // namespace executorch::extension::llm::serving::detail
