/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <utility>

namespace executorch {
namespace extension {
namespace image {

struct NormalizedRect {
  float x = 0.0f;
  float y = 0.0f;
  float width = 1.0f;
  float height = 1.0f;
};

inline constexpr NormalizedRect kFullImage = {0.0f, 0.0f, 1.0f, 1.0f};

enum class ColorFormat : uint8_t {
  BGRA,
  RGBA,
};

enum class YUVFormat : uint8_t {
  NV12,
  NV21,
};

// Quantization range of YUV samples. This is intrinsic to the encoding (not
// platform specific): VIDEO is studio/limited range (Y in [16, 235], chroma in
// [16, 240]); FULL spans the entire [0, 255]. Decoding with the wrong range
// over/under-stretches contrast and shifts color. Defaults to VIDEO, the most
// common camera/codec output.
enum class YUVRange : uint8_t {
  VIDEO,
  FULL,
};

enum class ResizeMode : uint8_t {
  STRETCH,
  LETTERBOX,
};

enum class LetterboxAnchor : uint8_t {
  CENTER,
  TOP_LEFT,
};

// EXIF orientation codes describing how to rotate the source so it displays
// upright. Only the four rotation values are supported (no mirrored variants);
// these match the codes Core Image's imageByApplyingOrientation: applies.
enum class Orientation : uint8_t {
  UP = 1, // no rotation
  DOWN = 3, // 180 degrees
  RIGHT = 6, // 90 degrees clockwise
  LEFT = 8, // 90 degrees counter-clockwise
};

struct Normalization {
  float scale_factor;
  // Per-channel mean/std applied as: (pixel * scale_factor - mean[c]) /
  // std_dev[c]. Only indices [0, kOutputChannels) (i.e. [0, 3) — RGB) are read
  // by the pipeline today; the 4th slot is reserved for a future 4-channel
  // (RGBA/alpha) output and is otherwise unused. Keep the reserved slot as an
  // identity normalization (mean 0, std_dev 1) so it stays divide-safe if a
  // future path ever reads it. std_dev entries that are read must be nonzero
  // (the loop divides by them); prefer the factories below over hand-rolled
  // aggregates, which value-initialize omitted entries to 0.
  float mean[4];
  float std_dev[4];

  static constexpr Normalization zeroToOne() {
    return {1.0f / 255.0f, {0.0f, 0.0f, 0.0f, 0.0f}, {1.0f, 1.0f, 1.0f, 1.0f}};
  }

  static constexpr Normalization imagenet() {
    return {
        1.0f / 255.0f,
        {0.485f, 0.456f, 0.406f, 0.0f},
        {0.229f, 0.224f, 0.225f, 1.0f}};
  }
};

struct ImageProcessorConfig {
  // Sentinels for gpu_min_input_pixels.
  static constexpr int64_t kGpuAlways = 0; // always use GPU
  static constexpr int64_t kGpuNever = INT64_MAX; // always use CPU

  // Default threshold: inputs larger than 1080p may use the GPU; 1080p and
  // smaller run on the CPU (where the GPU's fixed per-call overhead is not
  // worth it).
  static constexpr int64_t kDefaultGpuMinInputPixels = 1920 * 1080 + 1;

  // Channels in the produced output tensor. The processor currently always
  // emits RGB (alpha discarded; YUV decoded to RGB). This is the *output* axis;
  // for the channels a given input ColorFormat decodes to, use num_channels().
  static constexpr int32_t kOutputChannels = 3;

  int32_t target_width = 224;
  int32_t target_height = 224;
  ResizeMode resize_mode = ResizeMode::STRETCH;
  LetterboxAnchor letterbox_anchor = LetterboxAnchor::CENTER;
  float pad_value = 0.0f;
  Normalization normalization = Normalization::zeroToOne();
  // Minimum source pixel count (width * height) at which the GPU path may be
  // used; smaller inputs run on the CPU. kGpuAlways (0) forces GPU, kGpuNever
  // forces CPU.
  int64_t gpu_min_input_pixels = kDefaultGpuMinInputPixels;
};

// True if a source of width*height pixels should use the GPU path.
// kGpuNever (INT64_MAX) is never reached, so it forces CPU; kGpuAlways (0) is
// always satisfied, so it forces GPU.
inline bool should_use_gpu(
    const ImageProcessorConfig& config,
    int32_t width,
    int32_t height) {
  return static_cast<int64_t>(width) * static_cast<int64_t>(height) >=
      config.gpu_min_input_pixels;
}

// True if the config never uses the GPU regardless of input size.
inline bool is_cpu_only(const ImageProcessorConfig& config) {
  return config.gpu_min_input_pixels == ImageProcessorConfig::kGpuNever;
}

inline constexpr int32_t bytes_per_pixel(ColorFormat /*format*/) {
  // BGRA and RGBA are both 4 bytes per pixel.
  return 4;
}

inline constexpr int32_t num_channels(ColorFormat /*format*/) {
  // Channels a given input format decodes to (the input/decode axis): BGRA and
  // RGBA are processed as 3-channel RGB (alpha discarded). For the output
  // tensor's channel count, see ImageProcessorConfig::kOutputChannels.
  return 3;
}

// Compute resize_w/resize_h (post-scaling dims) and final_w/final_h (post-pad
// dims) for the given input. STRETCH scales to target dims directly; LETTERBOX
// scales to fit within target while preserving aspect ratio (the caller pads up
// to final dims).
inline void compute_resize_dims(
    int32_t input_w,
    int32_t input_h,
    const ImageProcessorConfig& config,
    int32_t& resize_w,
    int32_t& resize_h,
    int32_t& final_w,
    int32_t& final_h) {
  const int32_t tw = config.target_width;
  const int32_t th = config.target_height;

  // Default to STRETCH dims so a future ResizeMode left unhandled is still
  // well-defined (no UB reading uninitialized out-params) on builds without
  // -Wswitch (the internal build curates it out). The switch intentionally has
  // no default: case, so OSS -Wall/-Werror still flags a missing case at
  // compile time.
  resize_w = tw;
  resize_h = th;

  switch (config.resize_mode) {
    case ResizeMode::STRETCH:
      // Already tw/th from the defaults above.
      break;
    case ResizeMode::LETTERBOX: {
      const float scale = std::min(
          static_cast<float>(tw) / input_w, static_cast<float>(th) / input_h);
      // Rounding an extreme aspect ratio down can hit 0; keep at least one
      // pixel so the resized buffer is never empty.
      resize_w = std::max(1, static_cast<int32_t>(std::round(input_w * scale)));
      resize_h = std::max(1, static_cast<int32_t>(std::round(input_h * scale)));
      break;
    }
  }
  final_w = tw;
  final_h = th;
}

// Offset (per side) for centering resized content within the final canvas.
// Returns {0, 0} for the TOP_LEFT anchor.
inline std::pair<int32_t, int32_t> compute_letterbox_offset(
    int32_t width,
    int32_t height,
    int32_t final_width,
    int32_t final_height,
    LetterboxAnchor anchor) {
  if (anchor == LetterboxAnchor::TOP_LEFT) {
    return {0, 0};
  }
  return {(final_width - width) / 2, (final_height - height) / 2};
}

// True if `orientation` is one of the supported rotation codes.
inline bool is_supported_orientation(Orientation orientation) {
  return orientation == Orientation::UP || orientation == Orientation::DOWN ||
      orientation == Orientation::RIGHT || orientation == Orientation::LEFT;
}

// True for the 90-degree rotations (RIGHT/LEFT), which swap width and height.
inline bool is_transposed(Orientation orientation) {
  return orientation == Orientation::RIGHT || orientation == Orientation::LEFT;
}

// Source dimensions after applying `orientation`: width/height are swapped for
// the 90-degree rotations, unchanged otherwise.
inline std::pair<int32_t, int32_t>
oriented_dims(int32_t width, int32_t height, Orientation orientation) {
  if (is_transposed(orientation)) {
    return {height, width};
  }
  return {width, height};
}

} // namespace image
} // namespace extension
} // namespace executorch
