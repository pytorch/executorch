/*
 * Copyright (c) 2026 Samsung Electronics Co. LTD
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <executorch/extension/tensor/tensor_ptr.h>
#include <executorch/extension/tensor/tensor_ptr_maker.h>

#include <stb_image.h>
#include <stb_image_resize.h>

#include <cmath>
#include <cstdint>
#include <cstring>
#include <memory>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace example {

using ::executorch::extension::TensorPtr;

// Vision encoder constants (Gemma 4). Matches HF's image processor.
inline constexpr int32_t kVisionPatchSize = 16;
inline constexpr int32_t kVisionPatchDim =
    3 * kVisionPatchSize * kVisionPatchSize; // 768
inline constexpr int32_t kVisionPoolingKernel = 3;
inline constexpr int32_t kVisionCellSize =
    kVisionPatchSize * kVisionPoolingKernel; // 48

// The shipped gemma4_vision.pte accepts [1, 1260, 768] pixel_values and
// [1, 1260, 2] pixel_position_ids (STATIC shapes). 1260 = 140 pooled
// tokens * 9 (pooling kernel squared).
inline constexpr int32_t kVisionMaxSoftTokens = 140;
inline constexpr int32_t kVisionMaxPatches =
    kVisionMaxSoftTokens * kVisionPoolingKernel * kVisionPoolingKernel; // 1260
inline constexpr int32_t kVisionPooledTokens =
    kVisionMaxPatches / (kVisionPoolingKernel * kVisionPoolingKernel); // 140

struct ImageData {
  TensorPtr pixel_values; // (1, num_patches, 768) float32
  TensorPtr pixel_position_ids; // (1, num_patches, 2) int64, padding = -1
  int64_t num_valid_patches;
  int32_t original_width;
  int32_t original_height;
};

/// Load an image file (JPEG/PNG/etc.) into HWC interleaved RGB bytes.
inline std::vector<uint8_t>
load_image_rgb(const std::string& path, int32_t* width, int32_t* height) {
  int w = 0;
  int h = 0;
  int channels = 0;
  unsigned char* data = stbi_load(path.c_str(), &w, &h, &channels, 3);
  if (data == nullptr) {
    throw std::runtime_error(
        "Failed to load image: " + path + " (" + stbi_failure_reason() + ")");
  }
  std::vector<uint8_t> rgb(data, data + static_cast<size_t>(w) * h * 3);
  stbi_image_free(data);
  *width = w;
  *height = h;
  return rgb;
}

/// Compute target dimensions preserving aspect ratio.
/// Both dimensions will be divisible by kVisionCellSize (48).
inline std::pair<int32_t, int32_t> get_aspect_ratio_preserving_size(
    int32_t height,
    int32_t width,
    int32_t max_patches) {
  double target_px =
      static_cast<double>(max_patches) * kVisionPatchSize * kVisionPatchSize;
  double factor = std::sqrt(target_px / (static_cast<double>(height) * width));
  double ideal_h = factor * height;
  double ideal_w = factor * width;

  auto target_h = static_cast<int32_t>(std::floor(ideal_h / kVisionCellSize)) *
      kVisionCellSize;
  auto target_w = static_cast<int32_t>(std::floor(ideal_w / kVisionCellSize)) *
      kVisionCellSize;

  int32_t max_side =
      (max_patches / (kVisionPoolingKernel * kVisionPoolingKernel)) *
      kVisionCellSize;
  if (target_h == 0 && target_w == 0) {
    throw std::runtime_error("Image too small for patch size");
  }
  if (target_h == 0) {
    target_h = kVisionCellSize;
    target_w = std::min(
        static_cast<int32_t>(std::floor(static_cast<double>(width) / height)) *
            kVisionCellSize,
        max_side);
  } else if (target_w == 0) {
    target_w = kVisionCellSize;
    target_h = std::min(
        static_cast<int32_t>(std::floor(static_cast<double>(height) / width)) *
            kVisionCellSize,
        max_side);
  }
  return {target_h, target_w};
}

/// Compute number of soft tokens after pooling.
inline int64_t compute_vision_num_tokens(int64_t num_patches) {
  return num_patches / (kVisionPoolingKernel * kVisionPoolingKernel);
}

/// Resize RGB image (HWC uint8) to target dimensions using bicubic
/// (Catmull-Rom) interpolation, matching Python's PIL.Image.BICUBIC.
/// PIL filters the stored sRGB values directly, so stb must not linearize
/// them (STBIR_COLORSPACE_SRGB would gamma-decode before filtering).
inline std::vector<uint8_t> resize_rgb(
    const uint8_t* src,
    int32_t src_w,
    int32_t src_h,
    int32_t dst_w,
    int32_t dst_h) {
  std::vector<uint8_t> dst(static_cast<size_t>(dst_w) * dst_h * 3);
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
      STBIR_COLORSPACE_LINEAR,
      nullptr);
  return dst;
}

/// Patchify an RGB image (HWC uint8, already resized) into pixel_values
/// and position_ids tensors, padded to max_patches.
inline ImageData patchify_rgb_image(
    const uint8_t* rgb,
    int32_t width,
    int32_t height,
    int32_t max_soft_tokens) {
  int32_t max_patches =
      max_soft_tokens * kVisionPoolingKernel * kVisionPoolingKernel;

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

  int32_t h_patches = height / kVisionPatchSize;
  int32_t w_patches = width / kVisionPatchSize;
  int32_t num_patches = h_patches * w_patches;

  // Allocate padded tensors
  auto pixel_values = executorch::extension::zeros(
      {1, max_patches, kVisionPatchDim}, executorch::aten::ScalarType::Float);
  auto pixel_position_ids = executorch::extension::full(
      {1, max_patches, 2}, -1, executorch::aten::ScalarType::Long);

  float* pv_data = pixel_values->mutable_data_ptr<float>();
  int64_t* pos_data = pixel_position_ids->mutable_data_ptr<int64_t>();

  // Extract patches in row-major order (y then x), matching HF's layout:
  //   patches[py * w_patches + px] = image[py*16:(py+1)*16, px*16:(px+1)*16]
  for (int32_t py = 0; py < h_patches; ++py) {
    for (int32_t px = 0; px < w_patches; ++px) {
      int32_t patch_idx = py * w_patches + px;
      float* patch_out =
          pv_data + static_cast<size_t>(patch_idx) * kVisionPatchDim;

      for (int32_t dy = 0; dy < kVisionPatchSize; ++dy) {
        for (int32_t dx = 0; dx < kVisionPatchSize; ++dx) {
          int32_t img_y = py * kVisionPatchSize + dy;
          int32_t img_x = px * kVisionPatchSize + dx;
          int32_t src_idx = (img_y * width + img_x) * 3;
          int32_t dst_idx = (dy * kVisionPatchSize + dx) * 3;
          for (int32_t c = 0; c < 3; ++c) {
            patch_out[dst_idx + c] = img_data[src_idx + c] / 255.0f;
          }
        }
      }

      // Position IDs: (x, y) patch coordinates
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

} // namespace example
