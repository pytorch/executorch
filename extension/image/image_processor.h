/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <cstdint>
#include <memory>
#include <utility>
#include <vector>

#include <executorch/extension/image/image_processor_config.h>
#include <executorch/extension/tensor/tensor_ptr.h>
#include <executorch/runtime/core/error.h>
#include <executorch/runtime/core/result.h>

namespace executorch {
namespace extension {
namespace image {

class ImageProcessor {
 public:
  ImageProcessor();
  explicit ImageProcessor(ImageProcessorConfig config);
  ~ImageProcessor();

  // Movable but not copyable. The Impl (pImpl) is owned by unique_ptr and
  // and shouldn't be deep-copied; callers that want a fresh instance should
  // construct one from the config().
  ImageProcessor(ImageProcessor&&) noexcept;
  ImageProcessor& operator=(ImageProcessor&&) noexcept;
  ImageProcessor(const ImageProcessor&) = delete;
  ImageProcessor& operator=(const ImageProcessor&) = delete;

  /// Output tensor shape `[1, 3, target_height, target_width]` for the given
  /// input. The channel count is always `ImageProcessorConfig::kOutputChannels`
  /// (3 — alpha is discarded; YUV decodes to RGB), matching the tensor
  /// `process()` produces.
  std::vector<int32_t> compute_output_shape(
      int32_t input_width,
      int32_t input_height,
      Orientation orientation = Orientation::UP,
      NormalizedRect roi = kFullImage) const;

  /// Letterbox padding (per side, in pixels) the processor applies for the
  /// given input size, returned as `{x, y}`: `x` is the horizontal pad
  /// (left/right, along the width axis) and `y` the vertical pad (top/bottom,
  /// along the height axis) of the resized content. Returns `{0, 0}` for
  /// STRETCH or the TOP_LEFT anchor. Lets callers map the padded output back to
  /// the source region without replicating the resize geometry.
  std::pair<int32_t, int32_t> compute_letterbox_padding(
      int32_t input_width,
      int32_t input_height,
      Orientation orientation = Orientation::UP,
      NormalizedRect roi = kFullImage) const;

  /// Process an image into a normalized float tensor.
  ///
  /// @note **Not thread-safe per instance.** Implementations may keep
  /// per-instance state and reuse internal scratch buffers across calls, so
  /// concurrent calls to `process()` / `process_yuv()` on the same
  /// `ImageProcessor` from different threads are not safe. Use one instance per
  /// thread, or serialize calls externally. Different instances are always
  /// independent.
  runtime::Result<TensorPtr> process(
      const uint8_t* data,
      int32_t width,
      int32_t height,
      int32_t stride_bytes,
      ColorFormat input_format,
      Orientation orientation = Orientation::UP,
      NormalizedRect roi = kFullImage) const;

  /// Process semi-planar YUV (NV12/NV21) into a normalized float tensor.
  /// @note Not thread-safe per instance — see `process()`.
  runtime::Result<TensorPtr> process_yuv(
      const uint8_t* y_plane,
      int32_t y_stride,
      const uint8_t* uv_plane,
      int32_t uv_stride,
      int32_t width,
      int32_t height,
      YUVFormat format,
      Orientation orientation = Orientation::UP,
      NormalizedRect roi = kFullImage,
      YUVRange range = YUVRange::VIDEO) const;

  /// Process an image into a caller-provided output tensor, avoiding per-call
  /// output allocation (e.g. to reuse one tensor across video frames). `out`
  /// must be a contiguous Float tensor shaped [1, 3, target_height,
  /// target_width]. `process()` is a thin allocating wrapper over this.
  /// @note Not thread-safe per instance — see `process()`.
  runtime::Error process_into(
      const uint8_t* data,
      int32_t width,
      int32_t height,
      int32_t stride_bytes,
      ColorFormat input_format,
      ::executorch::aten::Tensor& out,
      Orientation orientation = Orientation::UP,
      NormalizedRect roi = kFullImage) const;

  /// Semi-planar YUV (NV12/NV21) variant of `process_into`.
  /// @note Not thread-safe per instance — see `process()`.
  runtime::Error process_yuv_into(
      const uint8_t* y_plane,
      int32_t y_stride,
      const uint8_t* uv_plane,
      int32_t uv_stride,
      int32_t width,
      int32_t height,
      YUVFormat format,
      ::executorch::aten::Tensor& out,
      Orientation orientation = Orientation::UP,
      NormalizedRect roi = kFullImage,
      YUVRange range = YUVRange::VIDEO) const;

  const ImageProcessorConfig& config() const;

  /// Platform-specific implementation. Forward-declared here; the full
  /// definition lives in each platform's translation unit. External callers
  /// receive an opaque reference: the type is only usable from a translation
  /// unit that includes the platform implementation.
  class Impl;

  /// Internal accessor used by the platform-specific free functions and the
  /// file-local helpers in this library's implementation. External callers
  /// should not use this; the Impl type is opaque outside the implementation.
  Impl& impl() const noexcept;

 private:
  std::unique_ptr<Impl> impl_;
};

} // namespace image
} // namespace extension
} // namespace executorch
