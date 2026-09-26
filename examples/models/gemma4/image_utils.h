/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <executorch/extension/tensor/tensor_ptr.h>
#include <executorch/extension/tensor/tensor_ptr_maker.h>

#include <stb_image_resize.h>

#include <cmath>
#include <cstdint>
#include <stdexcept>
#include <string>
#include <vector>

namespace executorch::examples::gemma4 {

using ::executorch::extension::TensorPtr;

// Vision encoder constants
static constexpr int32_t kPatchSize = 16;
static constexpr int32_t kPatchDim = 3 * kPatchSize * kPatchSize; // 768
static constexpr int32_t kPoolingKernel = 3;
static constexpr int32_t kCellSize = kPatchSize * kPoolingKernel; // 48

struct ImageData {
  TensorPtr pixel_values; // (1, num_patches, 768) float32
  TensorPtr pixel_position_ids; // (1, num_patches, 2) int64
  int64_t num_valid_patches;
  int32_t original_width;
  int32_t original_height;
};

/// Compute target dimensions preserving aspect ratio.
/// Both dimensions will be divisible by kCellSize (48).
inline std::pair<int32_t, int32_t> get_aspect_ratio_preserving_size(
    int32_t height,
    int32_t width,
    int32_t max_patches) {
  double target_px = static_cast<double>(max_patches) * kPatchSize * kPatchSize;
  double factor = std::sqrt(target_px / (height * width));
  double ideal_h = factor * height;
  double ideal_w = factor * width;

  auto target_h =
      static_cast<int32_t>(std::floor(ideal_h / kCellSize)) * kCellSize;
  auto target_w =
      static_cast<int32_t>(std::floor(ideal_w / kCellSize)) * kCellSize;

  int32_t max_side =
      (max_patches / (kPoolingKernel * kPoolingKernel)) * kCellSize;
  if (target_h == 0 && target_w == 0) {
    throw std::runtime_error("Image too small for patch size");
  }
  if (target_h == 0) {
    target_h = kCellSize;
    target_w = std::min(
        static_cast<int32_t>(std::floor(static_cast<double>(width) / height)) *
            kCellSize,
        max_side);
  } else if (target_w == 0) {
    target_w = kCellSize;
    target_h = std::min(
        static_cast<int32_t>(std::floor(static_cast<double>(height) / width)) *
            kCellSize,
        max_side);
  }
  return {target_h, target_w};
}

/// Compute number of soft tokens after pooling.
inline int64_t compute_vision_num_tokens(int64_t num_patches) {
  return num_patches / (kPoolingKernel * kPoolingKernel);
}

/// Resize RGB image (HWC uint8) to target dimensions using bicubic
/// (Catmull-Rom) interpolation, matching Python's PIL.Image.BICUBIC.
inline std::vector<uint8_t> resize_rgb(
    const uint8_t* src,
    int32_t src_w,
    int32_t src_h,
    int32_t dst_w,
    int32_t dst_h) {
  std::vector<uint8_t> dst(dst_w * dst_h * 3);
  stbir_resize_uint8_generic(
      src,
      src_w,
      src_h,
      0,
      dst.data(),
      dst_w,
      dst_h,
      0,
      3,
      STBIR_ALPHA_CHANNEL_NONE,
      0,
      STBIR_EDGE_CLAMP,
      STBIR_FILTER_CATMULLROM,
      STBIR_COLORSPACE_SRGB,
      nullptr);
  return dst;
}

/// Patchify an RGB image (HWC uint8, already resized) into pixel_values
/// and position_ids tensors, padded to max_patches.
///
/// pixel_values: (1, max_patches, 768) float32 in [0, 1]
/// pixel_position_ids: (1, max_patches, 2) int64, padding = -1
inline ImageData patchify_rgb_image(
    const uint8_t* rgb,
    int32_t width,
    int32_t height,
    int32_t max_soft_tokens = 280) {
  int32_t max_patches = max_soft_tokens * kPoolingKernel * kPoolingKernel;

  // Resize if needed
  auto [target_h, target_w] =
      get_aspect_ratio_preserving_size(height, width, max_patches);

  std::vector<uint8_t> resized;
  const uint8_t* img_data = rgb;
  if (target_h != height || target_w != width) {
    resized = resize_rgb(rgb, width, height, target_w, target_h);
    img_data = resized.data();
    height = target_h;
    width = target_w;
  }

  int32_t h_patches = height / kPatchSize;
  int32_t w_patches = width / kPatchSize;
  int32_t num_patches = h_patches * w_patches;

  // Allocate padded tensors
  auto pixel_values = executorch::extension::zeros(
      {1, max_patches, kPatchDim}, executorch::aten::ScalarType::Float);
  auto pixel_position_ids = executorch::extension::full(
      {1, max_patches, 2}, -1, executorch::aten::ScalarType::Long);

  float* pv_data = pixel_values->mutable_data_ptr<float>();
  int64_t* pos_data = pixel_position_ids->mutable_data_ptr<int64_t>();

  // Extract patches in row-major order (y then x), matching HF's layout:
  //   patches[py * w_patches + px] = image[py*16:(py+1)*16, px*16:(px+1)*16]
  for (int32_t py = 0; py < h_patches; ++py) {
    for (int32_t px = 0; px < w_patches; ++px) {
      int32_t patch_idx = py * w_patches + px;
      float* patch_out = pv_data + patch_idx * kPatchDim;

      // Extract 16x16 patch, flatten to (patch_h, patch_w, 3) row-major
      for (int32_t dy = 0; dy < kPatchSize; ++dy) {
        for (int32_t dx = 0; dx < kPatchSize; ++dx) {
          int32_t img_y = py * kPatchSize + dy;
          int32_t img_x = px * kPatchSize + dx;
          int32_t src_idx = (img_y * width + img_x) * 3;
          int32_t dst_idx = (dy * kPatchSize + dx) * 3;
          for (int32_t c = 0; c < 3; ++c) {
            patch_out[dst_idx + c] = img_data[src_idx + c] / 255.0f;
          }
        }
      }

      // Position IDs: (x, y) coordinates
      pos_data[patch_idx * 2] = px;
      pos_data[patch_idx * 2 + 1] = py;
    }
  }

  return ImageData{
      std::move(pixel_values),
      std::move(pixel_position_ids),
      num_patches,
      width,
      height,
  };
}

} // namespace executorch::examples::gemma4
