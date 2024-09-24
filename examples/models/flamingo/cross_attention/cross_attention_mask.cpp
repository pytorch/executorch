/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <executorch/examples/models/flamingo/cross_attention/cross_attention_mask.h>

#include <algorithm>
#include <string>

namespace example {

using ::executorch::aten::ScalarType;
using ::executorch::aten::Tensor;
using ::executorch::aten::TensorImpl;

// Fowrward declaration needed for ARM compilers.
int32_t safe_size_t_to_sizes_type(size_t value);
std::vector<std::vector<int>> _get_image_attention_intervals(
    const std::vector<int>& tokens,
    int image_token_id);

int32_t safe_size_t_to_sizes_type(size_t value) {
  if (value >
      static_cast<size_t>(std::numeric_limits<TensorImpl::SizesType>::max())) {
    throw std::overflow_error(
        "size_t value too large for TensorImpl::SizesType");
  }
  return static_cast<TensorImpl::SizesType>(value);
}

/**
 * Returns a list of lists of the form [start, end) where start is the index
 * of the current image token and end is the index of the next image token,
 * exclusive.
 *
 * Example:
 *     >>> text = "<img1><img2>These are two dogs. <img3>This is a cat."
 *     >>> size_t image_token_id = 1;
 *     >>> std::vector<int> tokens = {1, 1, 9673, 527, 1403, 12875, 13, 1, 1115,
 * 374, 264, 8415]};
 *     >>> transform = VisionCrossAttentionMask(tile_size=400, patch_size=40,
 * image_token_id=1)
 *     >>> intervals = _get_image_attention_intervals(tokens, image_token_id)
 *     [[0, 7], [1, 7], [7, 12]]
 *
 * @param tokens List of token IDs in the text sequence.
 * @param image_token_id The value of the image token.
 *
 * @returns Vector of vectors of the form [start, end) indicating the range of
 * positions in the text sequence that should attend to the image.
 */
std::vector<std::vector<int>> _get_image_attention_intervals(
    const std::vector<int>& tokens,
    int image_token_id) {
  std::vector<std::vector<int>> vision_masks;
  int end = tokens.size();
  std::vector<int> vision_token_locations;

  // Find all vision token locations.
  for (int i = 0; i < tokens.size(); ++i) {
    if (tokens[i] == image_token_id) {
      vision_token_locations.push_back(i);
    }
  }

  // Return empty vector if there are no images.
  if (vision_token_locations.empty()) {
    return vision_masks;
  }

  // If there is only one image, it will attend to subsequent text until end.
  if (vision_token_locations.size() == 1) {
    vision_masks.push_back({vision_token_locations[0], end});
    return vision_masks;
  }

  // Construct intervals from previous image token to next image token.
  for (int i = 0; i < vision_token_locations.size() - 1; ++i) {
    vision_masks.push_back(
        {vision_token_locations[i], vision_token_locations[i + 1]});
  }

  // Last image will attend to subsequent text until end.
  vision_masks.push_back({vision_token_locations.back(), end});

  // If there are consecutive vision tokens, they should all attend to the
  // same subsequent text.
  int last_mask_end = vision_masks.back()[1];
  for (auto it = vision_masks.rbegin(); it != vision_masks.rend(); ++it) {
    if ((*it)[0] == (*it)[1] - 1) {
      (*it)[1] = last_mask_end;
    }
    last_mask_end = (*it)[1];
  }

  return vision_masks;
}

std::vector<executorch::extension::TensorPtr> cross_attention_mask(
    const std::vector<int>& tokens,
    const std::vector<Tensor>& images,
    size_t tile_size,
    size_t patch_size,
    int image_token_id,
    std::vector<std::vector<int>>& out) {
  size_t patch_grid_size = tile_size / patch_size;
  size_t patches_per_tile = patch_grid_size * patch_grid_size;

  std::vector<std::vector<int>> image_intervals =
      _get_image_attention_intervals(tokens, image_token_id);

  if (image_intervals.size() != images.size()) {
    throw std::runtime_error(
        "The number of image tokens (" +
        std::to_string(image_intervals.size()) +
        ") does not match the number of images (" +
        std::to_string(images.size()) + ")");
  }

  // Create mask for each individual image based on its number of tokens,
  // which can vary based on number of tiles since they are not yet tile padded.
  // The masks are padded and concatenated together in the batch collator.
  std::vector<executorch::extension::TensorPtr> cross_attention_masks;
  size_t text_seq_len = tokens.size();
  for (size_t image_idx = 0; image_idx < image_intervals.size(); ++image_idx) {
    size_t n_tiles = images[image_idx].size(0);
    size_t image_seq_len =
        n_tiles * (patches_per_tile + 1); // +1 for the CLS token.

    // Mask will be block of 1s at the corresponding interval in the text.
    // It is not a causal block because all the image tokens correspond
    // to a single image, so text tokens attend to all the image's tokens.
    std::vector<TensorImpl::SizesType> sizes = {
        safe_size_t_to_sizes_type(text_seq_len),
        safe_size_t_to_sizes_type(image_seq_len)};

    // Allocate the underlying data to be handled by the managed tensor.
    size_t num_elements = text_seq_len * image_seq_len;
    size_t stride = image_seq_len;
    std::vector<int> mask_data(num_elements);

    auto mask = executorch::extension::from_blob(
        mask_data.data(), sizes, ScalarType::Int);
    cross_attention_masks.emplace_back(std::move(mask));

    // Add the allocated data to the output vector.
    out.emplace_back(std::move(mask_data));

    // All rows of tensor in the text_seq_len dimension within the interval are
    // set to 1 (true).
    size_t start = image_intervals[image_idx][0];
    size_t end = image_intervals[image_idx][1]; // End is exclusive.
    for (size_t i = start; i < end; ++i) {
      for (size_t j = 0; j < image_seq_len; ++j) {
        size_t unrolled_index = i * image_seq_len + j;
        if (unrolled_index >= out[image_idx].size()) {
          throw std::out_of_range(
              "Index " + std::to_string(unrolled_index) +
              " out of range of output tensor.");
        }
        out[image_idx][i * stride + j] = 1;
      }
    }
  }

  return cross_attention_masks;
}

} // namespace example
