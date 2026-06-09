/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <executorch/extension/image/image_processor.h>

#include <algorithm>
#include <cstring>
#include <memory>

#define STB_IMAGE_RESIZE_IMPLEMENTATION
#include <stb_image_resize.h>

#include <executorch/runtime/core/error.h>
#include <executorch/runtime/core/exec_aten/util/tensor_util.h>

namespace executorch {
namespace extension {
namespace image {

using runtime::Error;
using runtime::Result;

namespace {

inline uint8_t clamp_uint8(int v) {
  return static_cast<uint8_t>(std::max(0, std::min(255, v)));
}

// Apply a rotation to an interleaved 8-bit image, writing a tightly-packed
// result to `dst` (capacity out_width * out_height * channels). Supports the
// rotation codes UP/DOWN/RIGHT/LEFT with `channels` of 3 or 4.
// out_width/out_height receive the post-rotation dims (swapped for RIGHT/LEFT).
//
// The destination pixel (r, c) maps to source (sr, sc), an affine function of
// (r, c). The per-orientation coefficients are computed once (no per-pixel
// branch) and the source index is stepped incrementally across the loop.
void apply_orientation_interleaved(
    const uint8_t* src,
    int32_t width,
    int32_t height,
    int32_t stride,
    int32_t channels,
    Orientation orientation,
    uint8_t* dst,
    int32_t& out_width,
    int32_t& out_height) {
  const auto od = oriented_dims(width, height, orientation);
  out_width = od.first;
  out_height = od.second;
  const int32_t dst_stride = out_width * channels;
  const size_t px = static_cast<size_t>(channels);

  // sr = sr0 + r*dsr_dr + c*dsr_dc;  sc = sc0 + r*dsc_dr + c*dsc_dc.
  int32_t sr0, sc0, dsr_dr, dsr_dc, dsc_dr, dsc_dc;
  switch (orientation) {
    case Orientation::DOWN: // 180 degrees
      sr0 = height - 1;
      dsr_dr = -1;
      dsr_dc = 0;
      sc0 = width - 1;
      dsc_dr = 0;
      dsc_dc = -1;
      break;
    case Orientation::RIGHT: // 90 degrees clockwise
      sr0 = height - 1;
      dsr_dr = 0;
      dsr_dc = -1;
      sc0 = 0;
      dsc_dr = 1;
      dsc_dc = 0;
      break;
    case Orientation::LEFT: // 90 degrees counter-clockwise
      sr0 = 0;
      dsr_dr = 0;
      dsr_dc = 1;
      sc0 = width - 1;
      dsc_dr = -1;
      dsc_dc = 0;
      break;
    case Orientation::UP:
    default:
      sr0 = 0;
      dsr_dr = 1;
      dsr_dc = 0;
      sc0 = 0;
      dsc_dr = 0;
      dsc_dc = 1;
      break;
  }

  for (int32_t r = 0; r < out_height; ++r) {
    int32_t sr = sr0 + r * dsr_dr;
    int32_t sc = sc0 + r * dsc_dr;
    uint8_t* d = dst + static_cast<size_t>(r) * dst_stride;
    for (int32_t c = 0; c < out_width; ++c) {
      std::memcpy(
          d,
          src + static_cast<size_t>(sr) * stride + static_cast<size_t>(sc) * px,
          px);
      d += channels;
      sr += dsr_dc;
      sc += dsc_dc;
    }
  }
}

// Convert NV12 (UV-interleaved) or NV21 (VU-interleaved) to RGBA using BT.601,
// honoring the sample quantization range and packing a constant alpha=255.
// Writing RGBA directly (rather than RGB + a separate widen pass) lets the
// result feed process_into, which is BGRA/RGBA-only. Caller guarantees width
// and height are even.
void yuv_to_rgba_semi_planar(
    const uint8_t* y_plane,
    int32_t y_stride,
    const uint8_t* uv_plane,
    int32_t uv_stride,
    int32_t width,
    int32_t height,
    YUVFormat format,
    YUVRange range,
    uint8_t* rgba_out,
    int32_t rgba_stride) {
  const bool is_nv12 = (format == YUVFormat::NV12);
  const bool is_full = (range == YUVRange::FULL);
  for (int32_t y = 0; y < height; ++y) {
    const uint8_t* y_row = y_plane + y * y_stride;
    const uint8_t* uv_row = uv_plane + (y / 2) * uv_stride;
    uint8_t* out_row = rgba_out + y * rgba_stride;

    for (int32_t x = 0; x < width; ++x) {
      const int32_t uv_idx = (x / 2) * 2;
      const uint8_t u = is_nv12 ? uv_row[uv_idx] : uv_row[uv_idx + 1];
      const uint8_t v = is_nv12 ? uv_row[uv_idx + 1] : uv_row[uv_idx];

      const int32_t d = u - 128;
      const int32_t e = v - 128;

      if (is_full) {
        // Full range: unity luma gain, no luma offset.
        const int32_t yv = y_row[x];
        out_row[x * 4] = clamp_uint8(yv + ((359 * e + 128) >> 8));
        out_row[x * 4 + 1] = clamp_uint8(yv - ((88 * d + 183 * e + 128) >> 8));
        out_row[x * 4 + 2] = clamp_uint8(yv + ((454 * d + 128) >> 8));
      } else {
        // Video range: luma scaled by 255/219 about a 16 offset.
        const int32_t c = y_row[x] - 16;
        out_row[x * 4] = clamp_uint8((298 * c + 409 * e + 128) >> 8);
        out_row[x * 4 + 1] =
            clamp_uint8((298 * c - 100 * d - 208 * e + 128) >> 8);
        out_row[x * 4 + 2] = clamp_uint8((298 * c + 516 * d + 128) >> 8);
      }
      out_row[x * 4 + 3] = 255;
    }
  }
}

// Swizzle BGRA/RGBA → RGB (alpha discarded).
void swizzle_to_rgb(
    const uint8_t* src,
    int32_t width,
    int32_t height,
    int32_t src_stride,
    ColorFormat format,
    uint8_t* rgb_out,
    int32_t rgb_stride) {
  for (int32_t y = 0; y < height; ++y) {
    const uint8_t* in_row = src + y * src_stride;
    uint8_t* out_row = rgb_out + y * rgb_stride;
    if (format == ColorFormat::RGBA) {
      for (int32_t x = 0; x < width; ++x) {
        out_row[x * 3] = in_row[x * 4];
        out_row[x * 3 + 1] = in_row[x * 4 + 1];
        out_row[x * 3 + 2] = in_row[x * 4 + 2];
      }
    } else { // BGRA
      for (int32_t x = 0; x < width; ++x) {
        out_row[x * 3] = in_row[x * 4 + 2];
        out_row[x * 3 + 1] = in_row[x * 4 + 1];
        out_row[x * 3 + 2] = in_row[x * 4];
      }
    }
  }
}

// Bilinear resize via stb_image_resize. An identity resize (matching source and
// destination dimensions) is copied row by row so it stays pixel-exact,
// matching the accelerated backends instead of running content through the
// resampler.
Error resize_bilinear(
    const uint8_t* src,
    int32_t src_w,
    int32_t src_h,
    int32_t src_stride,
    int32_t channels,
    uint8_t* dst,
    int32_t dst_w,
    int32_t dst_h,
    int32_t dst_stride) {
  if (src_w == dst_w && src_h == dst_h) {
    const int32_t row_bytes = src_w * channels;
    for (int32_t y = 0; y < src_h; ++y) {
      std::memcpy(dst + y * dst_stride, src + y * src_stride, row_bytes);
    }
    return Error::Ok;
  }
  // stbir_resize_uint8 defaults to a bicubic kernel (Catmull-Rom upsampling,
  // Mitchell downsampling). Use the generic API with an explicit triangle
  // filter so the resampler is genuinely bilinear, matching the hardware
  // bilinear filtering of the accelerated backends, as the name implies.
  // Samples are clamped at the edges and treated as linear (no sRGB gamma).
  int result = stbir_resize_uint8_generic(
      src,
      src_w,
      src_h,
      src_stride,
      dst,
      dst_w,
      dst_h,
      dst_stride,
      channels,
      STBIR_ALPHA_CHANNEL_NONE,
      /*flags=*/0,
      STBIR_EDGE_CLAMP,
      STBIR_FILTER_TRIANGLE,
      STBIR_COLORSPACE_LINEAR,
      /*alloc_context=*/nullptr);
  ET_CHECK_OR_RETURN_ERROR(
      result != 0, Internal, "stbir_resize_uint8_generic failed");
  return Error::Ok;
}

} // namespace

// --- ImageProcessor class ---

// Portable backend's per-instance state holds only the config.
class ImageProcessor::Impl {
 public:
  ImageProcessorConfig config;
};

ImageProcessor::ImageProcessor() : impl_(std::make_unique<Impl>()) {}

ImageProcessor::ImageProcessor(ImageProcessorConfig config)
    : impl_(std::make_unique<Impl>()) {
  impl_->config = config;
}

ImageProcessor::~ImageProcessor() = default;
ImageProcessor::ImageProcessor(ImageProcessor&&) noexcept = default;
ImageProcessor& ImageProcessor::operator=(ImageProcessor&&) noexcept = default;

ImageProcessor::Impl& ImageProcessor::impl() const noexcept {
  return *impl_;
}

const ImageProcessorConfig& ImageProcessor::config() const {
  return impl_->config;
}

Error ImageProcessor::process_into(
    const uint8_t* data,
    int32_t width,
    int32_t height,
    int32_t stride_bytes,
    ColorFormat input_format,
    executorch::aten::Tensor& out,
    Orientation orientation,
    NormalizedRect roi) const {
  ET_CHECK_OR_RETURN_ERROR(data != nullptr, InvalidArgument, "data is null");
  ET_CHECK_OR_RETURN_ERROR(
      width > 0 && height > 0, InvalidArgument, "invalid dimensions");
  ET_CHECK_OR_RETURN_ERROR(
      config().target_width > 0 && config().target_height > 0,
      InvalidArgument,
      "invalid target dimensions");
  ET_CHECK_OR_RETURN_ERROR(
      stride_bytes >= width * bytes_per_pixel(input_format),
      InvalidArgument,
      "stride too small");
  ET_CHECK_OR_RETURN_ERROR(
      roi.x >= 0 && roi.y >= 0 && roi.width > 0 && roi.height > 0 &&
          roi.x + roi.width <= 1.0f + 1e-6f &&
          roi.y + roi.height <= 1.0f + 1e-6f,
      InvalidArgument,
      "invalid ROI");
  ET_CHECK_OR_RETURN_ERROR(
      out.scalar_type() == executorch::aten::ScalarType::Float &&
          out.dim() == 4 && out.size(0) == 1 &&
          out.size(1) == ImageProcessorConfig::kOutputChannels &&
          out.size(2) == config().target_height &&
          out.size(3) == config().target_width,
      InvalidArgument,
      "out must be a Float [1, 3, target_h, target_w] tensor");
  // The CHW write below indexes `out` as tightly packed; a non-contiguous
  // tensor would scatter the result and corrupt memory.
  ET_CHECK_OR_RETURN_ERROR(
      executorch::ET_RUNTIME_NAMESPACE::tensor_is_contiguous(out),
      InvalidArgument,
      "out must be contiguous");
  ET_CHECK_OR_RETURN_ERROR(
      is_supported_orientation(orientation),
      InvalidArgument,
      "unsupported orientation");

  // Channels decoded from the input format (used for the intermediate RGB
  // buffers) vs. channels written to the output tensor. Equal today (both are
  // 3-channel RGB); kept distinct so the field each site reads stays correct if
  // a future single-channel input/output is added.
  const int32_t input_channels = num_channels(input_format);
  constexpr int32_t output_channels = ImageProcessorConfig::kOutputChannels;
  int32_t cur_w = width;
  int32_t cur_h = height;
  const uint8_t* cur_data = data;
  int32_t cur_stride = stride_bytes;

  // Step 1: orientation (orient -> ROI -> resize). Produce an oriented copy of
  // the interleaved input so the ROI/resize below run in display space. UP
  // keeps the zero-copy fast path.
  std::vector<uint8_t> oriented_buf;
  if (orientation != Orientation::UP) {
    const int32_t bpp = bytes_per_pixel(input_format);
    oriented_buf.resize(static_cast<size_t>(width) * height * bpp);
    int32_t oriented_w, oriented_h;
    apply_orientation_interleaved(
        cur_data,
        cur_w,
        cur_h,
        cur_stride,
        bpp,
        orientation,
        oriented_buf.data(),
        oriented_w,
        oriented_h);
    cur_data = oriented_buf.data();
    cur_w = oriented_w;
    cur_h = oriented_h;
    cur_stride = oriented_w * bpp;
  }

  // Step 2: ROI crop (pointer arithmetic).
  if (roi.x != 0.0f || roi.y != 0.0f || roi.width != 1.0f ||
      roi.height != 1.0f) {
    const int32_t bpp = bytes_per_pixel(input_format);
    const int32_t src_w = cur_w;
    const int32_t src_h = cur_h;
    // Guard against a sub-pixel ROI truncating to a zero-size crop, which would
    // produce an empty buffer and a 0-dim resize; keep at least one pixel.
    cur_w = std::max(1, static_cast<int32_t>(src_w * roi.width));
    cur_h = std::max(1, static_cast<int32_t>(src_h * roi.height));
    // Clamp the crop origin so the (min-1-clamped) crop stays inside the
    // source. Without this, a high roi.x/roi.y can push the read window past
    // the row or buffer end -> out-of-bounds read in swizzle_to_rgb below.
    const int32_t roi_x =
        std::min(static_cast<int32_t>(src_w * roi.x), src_w - cur_w);
    const int32_t roi_y =
        std::min(static_cast<int32_t>(src_h * roi.y), src_h - cur_h);
    cur_data = cur_data + roi_y * cur_stride + roi_x * bpp;
    // cur_stride stays the same.
  }

  // Step 3: Swizzle BGRA/RGBA → RGB (alpha discarded).
  std::vector<uint8_t> rgb_buf(
      static_cast<size_t>(cur_w) * cur_h * input_channels);
  swizzle_to_rgb(
      cur_data,
      cur_w,
      cur_h,
      cur_stride,
      input_format,
      rgb_buf.data(),
      cur_w * input_channels);
  cur_data = rgb_buf.data();
  cur_stride = cur_w * input_channels;

  // Step 4: Resize.
  int32_t resize_w, resize_h, final_w, final_h;
  compute_resize_dims(
      cur_w, cur_h, config(), resize_w, resize_h, final_w, final_h);

  std::vector<uint8_t> resized_buf(
      static_cast<size_t>(resize_w) * resize_h * input_channels);
  auto err = resize_bilinear(
      cur_data,
      cur_w,
      cur_h,
      cur_stride,
      input_channels,
      resized_buf.data(),
      resize_w,
      resize_h,
      resize_w * input_channels);
  if (err != Error::Ok) {
    return err;
  }

  // Step 5: Normalize + layout into the caller's CHW output (padded).
  float* output = out.mutable_data_ptr<float>();
  std::fill(
      output,
      output + static_cast<size_t>(output_channels) * final_w * final_h,
      config().pad_value);

  // Same helper compute_letterbox_padding() uses, so the placement here and
  // the padding we report to callers can never drift apart.
  const auto [offset_x, offset_y] = compute_letterbox_offset(
      resize_w, resize_h, final_w, final_h, config().letterbox_anchor);

  const auto& norm = config().normalization;
  // The per-channel divide below requires nonzero std_dev. The factories
  // guarantee this, but a hand-rolled Normalization could pass a 0.
  for (int32_t c = 0; c < output_channels; ++c) {
    ET_CHECK_OR_RETURN_ERROR(
        norm.std_dev[c] != 0.0f,
        InvalidArgument,
        "normalization std_dev must be nonzero");
  }
  // Source (resized RGB) carries input_channels; the output tensor carries
  // output_channels. They are equal today, so channels map 1:1; a future
  // divergence (e.g. grayscale) would need an explicit channel map here.
  for (int32_t y = 0; y < resize_h; ++y) {
    for (int32_t x = 0; x < resize_w; ++x) {
      const int32_t src_idx = (y * resize_w + x) * input_channels;
      const int32_t dst_y = y + offset_y;
      const int32_t dst_x = x + offset_x;
      for (int32_t c = 0; c < output_channels; ++c) {
        const float val =
            (resized_buf[src_idx + c] * norm.scale_factor - norm.mean[c]) /
            norm.std_dev[c];
        const size_t out_idx = static_cast<size_t>(c) * final_w * final_h +
            static_cast<size_t>(dst_y) * final_w + dst_x;
        output[out_idx] = val;
      }
    }
  }
  return Error::Ok;
}

Error ImageProcessor::process_yuv_into(
    const uint8_t* y_plane,
    int32_t y_stride,
    const uint8_t* uv_plane,
    int32_t uv_stride,
    int32_t width,
    int32_t height,
    YUVFormat format,
    executorch::aten::Tensor& out,
    Orientation orientation,
    NormalizedRect roi,
    YUVRange range) const {
  ET_CHECK_OR_RETURN_ERROR(
      y_plane != nullptr, InvalidArgument, "y_plane is null");
  ET_CHECK_OR_RETURN_ERROR(
      uv_plane != nullptr, InvalidArgument, "uv_plane is null");
  ET_CHECK_OR_RETURN_ERROR(
      width > 0 && height > 0, InvalidArgument, "invalid dimensions");
  ET_CHECK_OR_RETURN_ERROR(
      width % 2 == 0 && height % 2 == 0,
      InvalidArgument,
      "width and height must be even");
  // Each Y row needs `width` bytes; each UV row holds width/2 chroma pairs of
  // 2 bytes = `width` bytes.
  ET_CHECK_OR_RETURN_ERROR(
      y_stride >= width, InvalidArgument, "y_stride too small");
  ET_CHECK_OR_RETURN_ERROR(
      uv_stride >= width, InvalidArgument, "uv_stride too small");
  // yuv_to_rgb_semi_planar reduces format/range to a single bool each, treating
  // anything other than NV12/FULL as NV21/VIDEO. Reject unknown enum values so
  // a bogus cast (or a future variant the decoder doesn't yet handle) fails
  // fast instead of being silently mis-decoded.
  ET_CHECK_OR_RETURN_ERROR(
      format == YUVFormat::NV12 || format == YUVFormat::NV21,
      InvalidArgument,
      "unsupported YUV format");
  ET_CHECK_OR_RETURN_ERROR(
      range == YUVRange::VIDEO || range == YUVRange::FULL,
      InvalidArgument,
      "unsupported YUV range");
  // Validate the ROI before converting so a malformed rect fails fast instead
  // of after a full-frame decode.
  ET_CHECK_OR_RETURN_ERROR(
      roi.x >= 0 && roi.y >= 0 && roi.width > 0 && roi.height > 0 &&
          roi.x + roi.width <= 1.0f + 1e-6f &&
          roi.y + roi.height <= 1.0f + 1e-6f,
      InvalidArgument,
      "invalid ROI");

  // Convert YUV directly into an RGBA buffer (process_into is BGRA/RGBA-only).
  // Writing RGBA in one pass avoids a separate RGB buffer and an O(n) widen
  // copy; the converter packs alpha=255.
  std::vector<uint8_t> rgba(static_cast<size_t>(width) * height * 4);
  yuv_to_rgba_semi_planar(
      y_plane,
      y_stride,
      uv_plane,
      uv_stride,
      width,
      height,
      format,
      range,
      rgba.data(),
      width * 4);
  return process_into(
      rgba.data(),
      width,
      height,
      width * 4,
      ColorFormat::RGBA,
      out,
      orientation,
      roi);
}

// Allocate a CHW float tensor sized to the configured target and fill it via
// process_into.
Result<TensorPtr> ImageProcessor::process(
    const uint8_t* data,
    int32_t width,
    int32_t height,
    int32_t stride_bytes,
    ColorFormat input_format,
    Orientation orientation,
    NormalizedRect roi) const {
  ET_CHECK_OR_RETURN_ERROR(
      config().target_width > 0 && config().target_height > 0,
      InvalidArgument,
      "invalid target dimensions");

  const int32_t final_w = config().target_width;
  const int32_t final_h = config().target_height;
  auto out = make_tensor_ptr(
      {1, ImageProcessorConfig::kOutputChannels, final_h, final_w},
      std::vector<float>(
          static_cast<size_t>(ImageProcessorConfig::kOutputChannels) * final_w *
          final_h));

  auto err = process_into(
      data, width, height, stride_bytes, input_format, *out, orientation, roi);
  if (err != Error::Ok) {
    return err;
  }
  return out;
}

// Allocate a CHW float tensor sized to the configured target and fill it via
// process_yuv_into.
Result<TensorPtr> ImageProcessor::process_yuv(
    const uint8_t* y_plane,
    int32_t y_stride,
    const uint8_t* uv_plane,
    int32_t uv_stride,
    int32_t width,
    int32_t height,
    YUVFormat format,
    Orientation orientation,
    NormalizedRect roi,
    YUVRange range) const {
  ET_CHECK_OR_RETURN_ERROR(
      config().target_width > 0 && config().target_height > 0,
      InvalidArgument,
      "invalid target dimensions");

  const int32_t final_w = config().target_width;
  const int32_t final_h = config().target_height;
  auto out = make_tensor_ptr(
      {1, ImageProcessorConfig::kOutputChannels, final_h, final_w},
      std::vector<float>(
          static_cast<size_t>(ImageProcessorConfig::kOutputChannels) * final_w *
          final_h));

  auto err = process_yuv_into(
      y_plane,
      y_stride,
      uv_plane,
      uv_stride,
      width,
      height,
      format,
      *out,
      orientation,
      roi,
      range);
  if (err != Error::Ok) {
    return err;
  }
  return out;
}

} // namespace image
} // namespace extension
} // namespace executorch
