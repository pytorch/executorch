/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <executorch/extension/image/image_processor.h>

#include <algorithm>
#include <cstdint>
#include <utility>
#include <vector>

// Platform-independent ImageProcessor methods, compiled on all platforms. The
// per-platform translation units (image_processor.cpp /
// image_processor_apple.cpp) are selected mutually exclusively and provide the
// rest of the class; these geometry-only methods live here once instead of
// being duplicated in both.
namespace executorch {
namespace extension {
namespace image {

std::vector<int32_t> ImageProcessor::compute_output_shape(
    int32_t input_width,
    int32_t input_height,
    Orientation /*orientation*/,
    NormalizedRect roi) const {
  // Clamp to >= 1 so a sub-pixel ROI cannot truncate a dimension to 0, which
  // would divide by zero in compute_resize_dims (LETTERBOX) and yield NaN.
  // Mirrors the min-1 crop guard in process_into.
  const int32_t roi_w =
      std::max(1, static_cast<int32_t>(input_width * roi.width));
  const int32_t roi_h =
      std::max(1, static_cast<int32_t>(input_height * roi.height));

  int32_t resize_w, resize_h, final_w, final_h;
  compute_resize_dims(
      roi_w, roi_h, config(), resize_w, resize_h, final_w, final_h);

  // Output is CHW with a leading batch dimension. The channel count is
  // ImageProcessorConfig::kOutputChannels (alpha discarded; YUV decodes to
  // RGB), matching what process() produces.
  return {1, ImageProcessorConfig::kOutputChannels, final_h, final_w};
}

std::pair<int32_t, int32_t> ImageProcessor::compute_letterbox_padding(
    int32_t input_width,
    int32_t input_height,
    NormalizedRect roi) const {
  // Clamp to >= 1 to avoid a divide-by-zero -> NaN in compute_resize_dims for a
  // sub-pixel ROI (see compute_output_shape).
  const int32_t roi_w =
      std::max(1, static_cast<int32_t>(input_width * roi.width));
  const int32_t roi_h =
      std::max(1, static_cast<int32_t>(input_height * roi.height));

  int32_t resize_w, resize_h, final_w, final_h;
  compute_resize_dims(
      roi_w, roi_h, config(), resize_w, resize_h, final_w, final_h);

  // Same offset the pipelines use to place resized content, so callers can
  // exactly invert the padding.
  return compute_letterbox_offset(
      resize_w, resize_h, final_w, final_h, config().letterbox_anchor);
}

} // namespace image
} // namespace extension
} // namespace executorch
