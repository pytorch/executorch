/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

// Apple-accelerated implementation of ImageProcessor. Compiled only on Apple
// targets via build rules. The CPU pipeline uses Accelerate (vImage/vDSP) and
// CoreGraphics, both pure C APIs; the GPU fast paths call into the Core Image
// helpers in image_processor_apple_gpu.mm.
//
// Supported inputs:
//   ColorFormat:     BGRA, RGBA
//   YUVFormat:       NV12, NV21
//   ResizeMode:      STRETCH, LETTERBOX
//   LetterboxAnchor: CENTER, TOP_LEFT
//   Orientation:     UP, DOWN (180), RIGHT (90 CW), LEFT (90 CCW)

#include <executorch/extension/image/image_processor.h>
#include <executorch/extension/image/image_processor_apple.h>

#include <algorithm>
#include <cstring>
#include <memory>
#include <vector>

#include <Accelerate/Accelerate.h>
#include <CoreGraphics/CoreGraphics.h>

#if defined(__ARM_NEON)
#include <arm_neon.h>
#endif

#include <executorch/runtime/core/error.h>
#include <executorch/runtime/core/exec_aten/util/tensor_util.h>
#include "image_processor_apple_gpu.h"

namespace executorch {
namespace extension {
namespace image {

using runtime::Error;
using runtime::Result;

namespace {

// Standard video-range pixel range for ITU_R_601_4 YUV→RGB conversion. The
// signal occupies [16, 235] (luma) / [16, 240] (chroma); vImage derives the
// expansion gain from these bounds, mapping that range to the full [0, 255]
// output. (Using full-range bounds here would apply unity gain and decode
// video-range frames with washed-out contrast.)
constexpr vImage_YpCbCrPixelRange kYpCbCrPixelRange_Video = {
    .Yp_bias = 16,
    .CbCr_bias = 128,
    .YpRangeMax = 235,
    .CbCrRangeMax = 240,
    .YpMax = 235,
    .YpMin = 16,
    .CbCrMax = 240,
    .CbCrMin = 16};

// Full-range pixel range: luma and chroma span the entire [0, 255].
constexpr vImage_YpCbCrPixelRange kYpCbCrPixelRange_Full = {
    .Yp_bias = 0,
    .CbCr_bias = 128,
    .YpRangeMax = 255,
    .CbCrRangeMax = 255,
    .YpMax = 255,
    .YpMin = 0,
    .CbCrMax = 255,
    .CbCrMin = 0};

// Convert an Orientation to the EXIF orientation code (1-8) that the Core Image
// helpers (ci_process_*) expect. The enum is laid out to match the EXIF
// numbering; the cast's validity is anchored by the static_assert here, the one
// place that knows both the enum and the EXIF contract.
constexpr int32_t to_exif_orientation(Orientation orientation) {
  static_assert(
      static_cast<int32_t>(Orientation::UP) == 1,
      "Orientation::UP must equal the EXIF code for up (1)");
  return static_cast<int32_t>(orientation);
}

// CVPixelBuffer formats process_pixelbuffer can handle. Both the GPU and CPU
// paths are limited to these, so the format is validated once up front.
bool is_supported_pixel_format(OSType pixel_format) {
  switch (pixel_format) {
    case kCVPixelFormatType_32BGRA:
    case kCVPixelFormatType_32RGBA:
    case kCVPixelFormatType_420YpCbCr8BiPlanarVideoRange:
    case kCVPixelFormatType_420YpCbCr8BiPlanarFullRange:
    case kCVPixelFormatType_420YpCbCr10BiPlanarVideoRange:
    case kCVPixelFormatType_420YpCbCr10BiPlanarFullRange:
      return true;
    default:
      return false;
  }
}

// Scratch buffer storage strategy.
//
// ImageProcessor owns an Impl struct (pImpl) containing the config plus
// several ScratchBuffer<T> members for intermediate work in
// process()/process_yuv(). Each ScratchBuffer reuses its allocation across
// calls on the same processor; resize() reuses existing capacity and
// shrinks if capacity > 4× needed AND > 1MB to bound peak memory.
template <typename T>
class ScratchBuffer {
 public:
  T* resize(size_t needed) {
    constexpr size_t kShrinkThreshold = 1024 * 1024 / sizeof(T);
    const bool shrink = capacity_ > needed * 4 && capacity_ > kShrinkThreshold;
    if (needed > capacity_ || shrink) {
      // new T[] leaves trivial T uninitialized (no zero-fill), matching a raw
      // allocation; std::vector::resize would value-initialize on growth.
      buf_.reset(needed ? new T[needed] : nullptr);
      capacity_ = needed;
    }
    return buf_.get();
  }
  T* data() {
    return buf_.get();
  }

 private:
  std::unique_ptr<T[]> buf_;
  size_t capacity_ = 0;
};

} // namespace

// Platform-specific implementation for ImageProcessor (pImpl).
//
// One Impl instance per ImageProcessor. Buffers grow on demand and are
// reused across calls on the same processor. NOT thread-safe: callers must
// not call process()/process_yuv() on the same instance from multiple
// threads (see image_processor.h).
class ImageProcessor::Impl {
 public:
  ImageProcessorConfig config;
  ScratchBuffer<uint8_t> conv; // to_bgra() output
  ScratchBuffer<uint8_t> resized; // resize_and_pad_bgra() output
  ScratchBuffer<uint8_t> scale_temp; // vImageScale_ARGB8888 temp buffer
  ScratchBuffer<uint8_t> gpu_resized; // GPU path intermediate buffer
  ScratchBuffer<uint8_t> oriented; // orientation transform output
  ScratchBuffer<uint8_t> bgra; // process_yuv() intermediate BGRA
  ScratchBuffer<uint8_t> narrow_y; // P010→8-bit narrowed Y plane
  ScratchBuffer<uint8_t> narrow_uv; // P010→8-bit narrowed CbCr plane
  ScratchBuffer<uint8_t> uv_swap; // NV21→NV12 chroma-swapped CbCr plane

  // Lazy force-CPU proxy used when the owning processor can use the GPU but a
  // frame must run on the CPU pipeline (small input, GPU readback, or GPU
  // failure). The proxy never attempts the GPU. Allocated on first need so
  // CPU-only processors do not pay for it.
  std::unique_ptr<ImageProcessor> cpu_proxy;
};

namespace {

// Narrow a semi-planar 16-bit plane to 8-bit by taking the high byte of each
// sample. P010 stores its 10 valid bits in the high bits of each 16-bit word,
// so the high byte is the top 8 bits (matching the previous scalar `>> 8`).
// Uses NEON (8 samples/iteration) where available, with a scalar fallback for
// the row remainder and non-ARM targets.
void narrow_plane_p010_to_8bit(
    const uint8_t* src_base,
    int32_t src_stride_bytes,
    uint8_t* dst,
    int32_t samples_per_row,
    int32_t rows) {
  for (int32_t row = 0; row < rows; ++row) {
    const auto* src = reinterpret_cast<const uint16_t*>(
        src_base + static_cast<size_t>(row) * src_stride_bytes);
    uint8_t* d = dst + static_cast<size_t>(row) * samples_per_row;
    int32_t i = 0;
#if defined(__ARM_NEON)
    for (; i + 8 <= samples_per_row; i += 8) {
      vst1_u8(d + i, vshrn_n_u16(vld1q_u16(src + i), 8));
    }
#endif
    for (; i < samples_per_row; ++i) {
      d[i] = static_cast<uint8_t>(src[i] >> 8);
    }
  }
}

// Swap the two interleaved chroma channels (Cb<->Cr) of a CbCr8 plane into a
// tightly-packed destination (stride = chroma_w * 2). Converts NV21 (Cr,Cb)
// chroma to NV12 (Cb,Cr) so the standard NV12 conversion can be reused.
// Swapping the chroma is the correct NV21 handling; swapping the decoded R/B is
// not, because BT.601 weights Cr (->R) and Cb (->B) differently and the green
// channel mixes both.
//
// Each CbCr pair is a 16-bit unit, so the swap is a byte reversal within each
// halfword. NEON does 8 pairs (16 bytes) per vrev16q_u8; the scalar loop covers
// the row remainder and non-ARM targets.
void swap_chroma_cbcr(
    const uint8_t* src,
    int32_t src_stride,
    uint8_t* dst,
    int32_t chroma_w,
    int32_t chroma_h) {
  const int32_t row_bytes = chroma_w * 2;
  for (int32_t row = 0; row < chroma_h; ++row) {
    const uint8_t* s = src + static_cast<size_t>(row) * src_stride;
    uint8_t* d = dst + static_cast<size_t>(row) * row_bytes;
    int32_t i = 0;
#if defined(__ARM_NEON)
    for (; i + 16 <= row_bytes; i += 16) {
      vst1q_u8(d + i, vrev16q_u8(vld1q_u8(s + i)));
    }
#endif
    for (; i + 2 <= row_bytes; i += 2) {
      d[i] = s[i + 1];
      d[i + 1] = s[i];
    }
  }
}

// Convert BGRA/RGBA input to BGRA8888.
// `height * dst_stride` bytes; `dst_stride` must be at least `width * 4`.
Error to_bgra(
    const uint8_t* src,
    int32_t width,
    int32_t height,
    int32_t src_stride,
    ColorFormat format,
    uint8_t* dst,
    size_t dst_stride) {
  if (format == ColorFormat::BGRA) {
    for (int32_t y = 0; y < height; ++y) {
      std::memcpy(
          dst + static_cast<size_t>(y) * dst_stride,
          src + static_cast<size_t>(y) * src_stride,
          static_cast<size_t>(width) * 4);
    }
    return Error::Ok;
  }

  // RGBA→BGRA: swap channels 0↔2 with vImage (NEON accelerated)
  vImage_Buffer src_buf = {
      const_cast<uint8_t*>(src),
      static_cast<vImagePixelCount>(height),
      static_cast<vImagePixelCount>(width),
      static_cast<size_t>(src_stride)};
  vImage_Buffer dst_buf = {
      dst,
      static_cast<vImagePixelCount>(height),
      static_cast<vImagePixelCount>(width),
      dst_stride};
  const uint8_t permuteMap[4] = {2, 1, 0, 3}; // RGBA→BGRA
  vImagePermuteChannels_ARGB8888(
      &src_buf, &dst_buf, permuteMap, kvImageNoFlags);
  return Error::Ok;
}

// GPU resize dimension parameters.
struct GpuResizeDims {
  int32_t resize_w, resize_h, final_w, final_h;
};

// Compute GPU resize dimensions. The GPU handles crop + resize; padding
// (LETTERBOX) is applied during normalize.
void compute_gpu_dims(
    int32_t width,
    int32_t height,
    NormalizedRect roi,
    Orientation orientation,
    const ImageProcessorConfig& config,
    GpuResizeDims& out) {
  // ROI is in oriented (display) space, so orient the source dims first.
  const auto od = oriented_dims(width, height, orientation);
  const int32_t roi_w = static_cast<int32_t>(od.first * roi.width);
  const int32_t roi_h = static_cast<int32_t>(od.second * roi.height);
  compute_resize_dims(
      roi_w,
      roi_h,
      config,
      out.resize_w,
      out.resize_h,
      out.final_w,
      out.final_h);
}

// Apply ROI crop on BGRA data via pointer arithmetic.
// Updates cur_data/cur_w/cur_h in place; cur_stride is unchanged.
void apply_roi_crop_bgra(
    uint8_t*& cur_data,
    int32_t& cur_w,
    int32_t& cur_h,
    int32_t cur_stride,
    NormalizedRect roi) {
  if (roi.x == 0.0f && roi.y == 0.0f && roi.width == 1.0f &&
      roi.height == 1.0f) {
    return;
  }
  const int32_t src_w = cur_w;
  const int32_t src_h = cur_h;
  // Guard against a sub-pixel ROI truncating to a zero-size crop, which would
  // produce an empty buffer and a 0-dim resize; keep at least one pixel.
  cur_w = std::max(1, static_cast<int32_t>(src_w * roi.width));
  cur_h = std::max(1, static_cast<int32_t>(src_h * roi.height));
  // Clamp the crop origin so the (min-1-clamped) crop stays inside the source.
  // Without this, a high roi.x/roi.y can push the read window past the buffer
  // end -> out-of-bounds read in the downstream resize.
  const int32_t roi_x =
      std::min(static_cast<int32_t>(src_w * roi.x), src_w - cur_w);
  const int32_t roi_y =
      std::min(static_cast<int32_t>(src_h * roi.y), src_h - cur_h);
  cur_data = cur_data + roi_y * cur_stride + roi_x * 4;
}

// Result view into a thread-local BGRA buffer after resize.
struct BgraView {
  const uint8_t* data;
  int32_t width, height, stride;
};

// Resize BGRA data using vImageScale (bilinear, NEON-accelerated).
// Letterbox padding is applied during normalization so pad pixels get the
// correct pad_value instead of being normalized from zero.
//
// Caller pre-sizes `dst` to at least `resize_h * resize_w * 4` bytes (where
// resize_w/resize_h come from compute_resize_dims) and passes a
// scale_temp buffer pointer (use compute_scale_temp_size to query the size,
// or pass nullptr to skip the temp). Returns the BgraView plus final
// dimensions via out params.
Error resize_and_pad_bgra(
    const uint8_t* src,
    int32_t cur_w,
    int32_t cur_h,
    int32_t src_stride,
    const ImageProcessorConfig& config,
    uint8_t* dst,
    size_t dst_stride,
    void* scale_temp,
    BgraView& result,
    int32_t& final_w_out,
    int32_t& final_h_out) {
  int32_t resize_w, resize_h, final_w, final_h;
  compute_resize_dims(
      cur_w, cur_h, config, resize_w, resize_h, final_w, final_h);
  final_w_out = final_w;
  final_h_out = final_h;

  vImage_Buffer src_buf = {
      const_cast<uint8_t*>(src),
      static_cast<vImagePixelCount>(cur_h),
      static_cast<vImagePixelCount>(cur_w),
      static_cast<size_t>(src_stride)};
  vImage_Buffer dst_buf = {
      dst,
      static_cast<vImagePixelCount>(resize_h),
      static_cast<vImagePixelCount>(resize_w),
      dst_stride};

  vImage_Error verr =
      vImageScale_ARGB8888(&src_buf, &dst_buf, scale_temp, kvImageNoFlags);
  ET_CHECK_OR_RETURN_ERROR(
      verr == kvImageNoError, Internal, "vImageScale_ARGB8888 failed");

  result.data = dst;
  result.width = resize_w;
  result.height = resize_h;
  result.stride = static_cast<int32_t>(dst_stride);
  return Error::Ok;
}

// Query the temp buffer size required by vImageScale_ARGB8888 (bilinear)
// for the given source/destination dimensions. Returns 0 when no temp
// buffer is needed.
size_t compute_scale_temp_size(
    int32_t src_w,
    int32_t src_h,
    int32_t dst_w,
    int32_t dst_h) {
  vImage_Buffer src_buf = {
      nullptr,
      static_cast<vImagePixelCount>(src_h),
      static_cast<vImagePixelCount>(src_w),
      static_cast<size_t>(src_w) * 4};
  vImage_Buffer dst_buf = {
      nullptr,
      static_cast<vImagePixelCount>(dst_h),
      static_cast<vImagePixelCount>(dst_w),
      static_cast<size_t>(dst_w) * 4};
  vImage_Error temp_size = vImageScale_ARGB8888(
      &src_buf, &dst_buf, nullptr, kvImageGetTempBufferSize);
  return temp_size > 0 ? static_cast<size_t>(temp_size) : 0;
}

// Deinterleave BGRA uint8 → planar RGB float with fused normalization.
// Handles offset for letterbox padding.
//
// Per channel (R, G, B): vDSP_vfltu8 reads the matching byte from BGRA via
// stride=4 and converts uint8→float, then vDSP_vsmsa applies the fused
// affine `out = in * (scale_factor / std_dev) + (-mean / std_dev)` in-place.
Error deinterleave_bgra_to_chw(
    const uint8_t* bgra_data,
    int32_t src_w,
    int32_t src_h,
    int32_t src_stride,
    float* output,
    int32_t final_w,
    int32_t final_h,
    int32_t offset_x,
    int32_t offset_y,
    const Normalization& norm) {
  const size_t spatial = static_cast<size_t>(final_w) * final_h;

  // Per-channel affine coefficients for `out = in * a + b`.
  // BGRA byte layout: byte 0 = B, byte 1 = G, byte 2 = R; norm.{mean,std_dev}
  // are indexed in RGB order (channel 0 = R, 1 = G, 2 = B).
  const float a_r = norm.scale_factor / norm.std_dev[0];
  const float a_g = norm.scale_factor / norm.std_dev[1];
  const float a_b = norm.scale_factor / norm.std_dev[2];
  const float b_r = -norm.mean[0] / norm.std_dev[0];
  const float b_g = -norm.mean[1] / norm.std_dev[1];
  const float b_b = -norm.mean[2] / norm.std_dev[2];

  // When the bias is zero (e.g. zeroToOne / mean=0), a plain scale (vsmul) is
  // cheaper than the fused scale+add (vsmsa).
  const bool no_offset = (b_r == 0.0f && b_g == 0.0f && b_b == 0.0f);
  auto scale_bias =
      [no_offset](float* p, const float* a, const float* b, vDSP_Length n) {
        if (no_offset) {
          vDSP_vsmul(p, 1, a, p, 1, n);
        } else {
          vDSP_vsmsa(p, 1, a, b, p, 1, n);
        }
      };

  // Output planes in CHW order: R, G, B. Each plane is final_w × final_h
  // floats; we write a src_h × src_w region starting at (offset_y, offset_x).
  float* r_plane = output + 0 * spatial;
  float* g_plane = output + 1 * spatial;
  float* b_plane = output + 2 * spatial;

  // Fast path: source is contiguous and destination region is the entire
  // plane (offsets 0, src dims == final dims).
  if (src_stride == src_w * 4 && offset_x == 0 && offset_y == 0 &&
      src_w == final_w && src_h == final_h) {
    const vDSP_Length n = static_cast<vDSP_Length>(src_w) * src_h;
    vDSP_vfltu8(bgra_data + 2, 4, r_plane, 1, n);
    scale_bias(r_plane, &a_r, &b_r, n);
    vDSP_vfltu8(bgra_data + 1, 4, g_plane, 1, n);
    scale_bias(g_plane, &a_g, &b_g, n);
    vDSP_vfltu8(bgra_data + 0, 4, b_plane, 1, n);
    scale_bias(b_plane, &a_b, &b_b, n);
    return Error::Ok;
  }

  // Slow path: row-by-row to handle stride padding and/or letterbox offsets.
  for (int32_t y = 0; y < src_h; ++y) {
    const uint8_t* src_row = bgra_data + y * src_stride;
    const ptrdiff_t dst_off = (y + offset_y) * final_w + offset_x;
    float* r_dst = r_plane + dst_off;
    float* g_dst = g_plane + dst_off;
    float* b_dst = b_plane + dst_off;
    const vDSP_Length n = static_cast<vDSP_Length>(src_w);
    vDSP_vfltu8(src_row + 2, 4, r_dst, 1, n);
    scale_bias(r_dst, &a_r, &b_r, n);
    vDSP_vfltu8(src_row + 1, 4, g_dst, 1, n);
    scale_bias(g_dst, &a_g, &b_g, n);
    vDSP_vfltu8(src_row + 0, 4, b_dst, 1, n);
    scale_bias(b_dst, &a_b, &b_b, n);
  }
  return Error::Ok;
}

// Rotate an interleaved BGRA (ARGB8888 layout) buffer by `orientation` using
// vImage's SIMD/cache-aware 90-degree rotation, writing a tightly-packed result
// into `scratch`. UP is handled by the caller (no rotation). out_data/out_w/
// out_h/out_stride describe the rotated buffer (dims swapped for RIGHT/LEFT).
Error rotate_bgra(
    const uint8_t* src,
    int32_t width,
    int32_t height,
    int32_t stride,
    Orientation orientation,
    ScratchBuffer<uint8_t>& scratch,
    uint8_t*& out_data,
    int32_t& out_w,
    int32_t& out_h,
    int32_t& out_stride) {
  uint8_t rotation;
  switch (orientation) {
    case Orientation::RIGHT: // 90 degrees clockwise
      rotation = kRotate90DegreesClockwise;
      break;
    case Orientation::LEFT: // 90 degrees counter-clockwise
      rotation = kRotate90DegreesCounterClockwise;
      break;
    case Orientation::DOWN: // 180 degrees
      rotation = kRotate180DegreesClockwise;
      break;
    default:
      return Error::InvalidArgument;
  }

  const auto od = oriented_dims(width, height, orientation);
  out_w = od.first;
  out_h = od.second;
  out_stride = out_w * 4;
  out_data = scratch.resize(static_cast<size_t>(out_h) * out_stride);

  vImage_Buffer srcBuf = {
      const_cast<uint8_t*>(src),
      static_cast<vImagePixelCount>(height),
      static_cast<vImagePixelCount>(width),
      static_cast<size_t>(stride)};
  vImage_Buffer dstBuf = {
      out_data,
      static_cast<vImagePixelCount>(out_h),
      static_cast<vImagePixelCount>(out_w),
      static_cast<size_t>(out_stride)};
  const Pixel_8888 backColor = {0, 0, 0, 0};
  vImage_Error verr = vImageRotate90_ARGB8888(
      &srcBuf, &dstBuf, rotation, backColor, kvImageNoFlags);
  ET_CHECK_OR_RETURN_ERROR(
      verr == kvImageNoError,
      Internal,
      "vImageRotate90_ARGB8888 failed: %zd",
      verr);
  return Error::Ok;
}

} // namespace

// --- ImageProcessor class ---

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

// --- File-local normalization helpers ---
//
// These back the Apple GPU/CPU pipelines and process_pixelbuffer(); they
// are intentionally not part of the public surface (image_processor_apple.h
// exposes only process_pixelbuffer).

namespace {

// Fill a caller-owned CHW float buffer from resized BGRA8 data. `out` must hold
// 3*final_w*final_h floats. For LETTERBOX (content smaller than output) the pad
// region is set to pad_value and content is placed at the anchor offset;
// otherwise every element is written and the fill is skipped.
Error normalize_bgra_into(
    const ImageProcessor& proc,
    const uint8_t* bgra_data,
    int32_t width,
    int32_t height,
    int32_t final_w,
    int32_t final_h,
    int32_t stride,
    float* out) {
  ET_CHECK_OR_RETURN_ERROR(
      bgra_data != nullptr, InvalidArgument, "data is null");
  ET_CHECK_OR_RETURN_ERROR(
      width <= final_w && height <= final_h,
      InvalidArgument,
      "data dimensions must not exceed final dimensions");

  const auto& config = proc.config();
  const size_t total = static_cast<size_t>(3) * final_w * final_h;

  int32_t offset_x = 0, offset_y = 0;
  if (width != final_w || height != final_h) {
    std::fill(out, out + total, config.pad_value);
    const auto offset = compute_letterbox_offset(
        width, height, final_w, final_h, config.letterbox_anchor);
    offset_x = offset.first;
    offset_y = offset.second;
  }

  return deinterleave_bgra_to_chw(
      bgra_data,
      width,
      height,
      stride,
      out,
      final_w,
      final_h,
      offset_x,
      offset_y,
      config.normalization);
}

// CPU fallback that writes the normalized result into `out`. Routes through a
// force-CPU proxy when the processor can use the GPU so its scratch is reused.
Error process_bgra_cpu_only_into(
    const ImageProcessor& proc,
    const uint8_t* bgra,
    int32_t width,
    int32_t height,
    Orientation orientation,
    NormalizedRect roi,
    executorch::aten::Tensor& out) {
  if (is_cpu_only(proc.config())) {
    return proc.process_into(
        bgra,
        width,
        height,
        width * 4,
        ColorFormat::BGRA,
        out,
        orientation,
        roi);
  }
  auto& cpu_proxy = proc.impl().cpu_proxy;
  if (!cpu_proxy) {
    ImageProcessorConfig cpu_config = proc.config();
    cpu_config.gpu_min_input_pixels = ImageProcessorConfig::kGpuNever;
    cpu_proxy = std::make_unique<ImageProcessor>(cpu_config);
  }
  return cpu_proxy->process_into(
      bgra, width, height, width * 4, ColorFormat::BGRA, out, orientation, roi);
}

// Validate that `out` is a contiguous Float [1, 3, target_h, target_w] tensor.
Error check_out_tensor(
    const ImageProcessorConfig& config,
    executorch::aten::Tensor& out) {
  ET_CHECK_OR_RETURN_ERROR(
      out.scalar_type() == executorch::aten::ScalarType::Float &&
          out.dim() == 4 && out.size(0) == 1 && out.size(1) == 3 &&
          out.size(2) == config.target_height &&
          out.size(3) == config.target_width,
      InvalidArgument,
      "out must be a Float [1, 3, target_h, target_w] tensor");
  // The CHW write indexes `out` as tightly packed; a non-contiguous tensor
  // would scatter the result and corrupt memory.
  ET_CHECK_OR_RETURN_ERROR(
      executorch::ET_RUNTIME_NAMESPACE::tensor_is_contiguous(out),
      InvalidArgument,
      "out must be contiguous");
  return Error::Ok;
}

} // namespace

Error ImageProcessor::process_into(
    const uint8_t* data,
    int32_t width,
    int32_t height,
    int32_t stride_bytes,
    ColorFormat input_format,
    executorch::aten::Tensor& out,
    Orientation orientation,
    NormalizedRect roi) const {
  const auto& config = impl_->config;
  ET_CHECK_OR_RETURN_ERROR(data != nullptr, InvalidArgument, "data is null");
  ET_CHECK_OR_RETURN_ERROR(
      width > 0 && height > 0, InvalidArgument, "invalid dimensions");
  ET_CHECK_OR_RETURN_ERROR(
      config.target_width > 0 && config.target_height > 0,
      InvalidArgument,
      "invalid target dimensions");
  // The fused normalization divides by std_dev per channel. The factories
  // guarantee nonzero, but a hand-rolled Normalization could pass a 0.
  for (int32_t c = 0; c < 3; ++c) {
    ET_CHECK_OR_RETURN_ERROR(
        config.normalization.std_dev[c] != 0.0f,
        InvalidArgument,
        "normalization std_dev must be nonzero");
  }
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
      is_supported_orientation(orientation),
      InvalidArgument,
      "unsupported orientation");
  auto out_err = check_out_tensor(config, out);
  if (out_err != Error::Ok) {
    return out_err;
  }
  float* out_ptr = out.mutable_data_ptr<float>();

  // GPU fast path: crop + resize in a single Core Image pass.
  if (should_use_gpu(config, width, height)) {
    const CIPixelFormatValue ci_format = (input_format == ColorFormat::BGRA)
        ? CI_PIXEL_FORMAT_BGRA8
        : CI_PIXEL_FORMAT_RGBA8;
    GpuResizeDims gpu;
    compute_gpu_dims(width, height, roi, orientation, config, gpu);
    auto& gpu_resized = impl_->gpu_resized;
    gpu_resized.resize(static_cast<size_t>(gpu.resize_w) * gpu.resize_h * 4);
    int ret = ci_process_to_bgra(
        data,
        width,
        height,
        stride_bytes,
        ci_format,
        to_exif_orientation(orientation),
        roi.x,
        roi.y,
        roi.width,
        roi.height,
        gpu.resize_w,
        gpu.resize_h,
        gpu_resized.data(),
        gpu.resize_w * 4);
    if (ret == 0) {
      return normalize_bgra_into(
          *this,
          gpu_resized.data(),
          gpu.resize_w,
          gpu.resize_h,
          gpu.final_w,
          gpu.final_h,
          gpu.resize_w * 4,
          out_ptr);
    }
    ET_LOG(Debug, "GPU BGRA resize failed (ret=%d), falling back to CPU", ret);
  }

  // CPU path. Step 1: convert to BGRA.
  uint8_t* bgra_data = nullptr;
  int32_t cur_w = width;
  int32_t cur_h = height;
  int32_t cur_stride;
  if (input_format == ColorFormat::BGRA) {
    bgra_data = const_cast<uint8_t*>(data);
    cur_stride = stride_bytes;
  } else {
    const size_t conv_stride = static_cast<size_t>(width) * 4;
    bgra_data = impl_->conv.resize(conv_stride * height);
    auto err = to_bgra(
        data,
        width,
        height,
        stride_bytes,
        input_format,
        bgra_data,
        conv_stride);
    if (err != Error::Ok) {
      return err;
    }
    cur_stride = static_cast<int32_t>(conv_stride);
  }

  // Step 2: orientation. Rotate the BGRA buffer (vImage) so ROI/resize run in
  // display space (orient -> ROI -> resize). UP leaves the buffer untouched.
  uint8_t* cur_data = bgra_data;
  if (orientation != Orientation::UP) {
    uint8_t* rotated;
    int32_t rot_w, rot_h, rot_stride;
    auto rot_err = rotate_bgra(
        cur_data,
        cur_w,
        cur_h,
        cur_stride,
        orientation,
        impl_->oriented,
        rotated,
        rot_w,
        rot_h,
        rot_stride);
    if (rot_err != Error::Ok) {
      return rot_err;
    }
    cur_data = rotated;
    cur_w = rot_w;
    cur_h = rot_h;
    cur_stride = rot_stride;
  }

  // Step 3: ROI crop (pointer arithmetic on BGRA data).
  apply_roi_crop_bgra(cur_data, cur_w, cur_h, cur_stride, roi);

  // Step 4: resize. Letterbox padding is applied during normalization.
  BgraView resized;
  int32_t final_w, final_h;
  {
    int32_t resize_w, resize_h, fw, fh;
    compute_resize_dims(cur_w, cur_h, config, resize_w, resize_h, fw, fh);
    const size_t resized_stride = static_cast<size_t>(resize_w) * 4;
    uint8_t* resize_dst = impl_->resized.resize(resized_stride * resize_h);
    const size_t temp_size =
        compute_scale_temp_size(cur_w, cur_h, resize_w, resize_h);
    void* scale_temp =
        temp_size > 0 ? impl_->scale_temp.resize(temp_size) : nullptr;
    auto resize_err = resize_and_pad_bgra(
        cur_data,
        cur_w,
        cur_h,
        cur_stride,
        config,
        resize_dst,
        resized_stride,
        scale_temp,
        resized,
        final_w,
        final_h);
    if (resize_err != Error::Ok) {
      return resize_err;
    }
  }

  // Step 5: normalize BGRA → CHW float buffer.
  return normalize_bgra_into(
      *this,
      resized.data,
      resized.width,
      resized.height,
      final_w,
      final_h,
      resized.stride,
      out_ptr);
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
  const auto& config = impl_->config;
  ET_CHECK_OR_RETURN_ERROR(
      y_plane != nullptr, InvalidArgument, "y_plane is null");
  ET_CHECK_OR_RETURN_ERROR(
      uv_plane != nullptr, InvalidArgument, "uv_plane is null");
  ET_CHECK_OR_RETURN_ERROR(
      format == YUVFormat::NV12 || format == YUVFormat::NV21,
      InvalidArgument,
      "semi-planar overload requires NV12 or NV21");
  ET_CHECK_OR_RETURN_ERROR(
      width > 0 && height > 0, InvalidArgument, "invalid dimensions");
  ET_CHECK_OR_RETURN_ERROR(
      width % 2 == 0 && height % 2 == 0,
      InvalidArgument,
      "width and height must be even");
  ET_CHECK_OR_RETURN_ERROR(
      y_stride >= width, InvalidArgument, "y_stride too small");
  ET_CHECK_OR_RETURN_ERROR(
      uv_stride >= width, InvalidArgument, "uv_stride too small");
  ET_CHECK_OR_RETURN_ERROR(
      config.target_width > 0 && config.target_height > 0,
      InvalidArgument,
      "invalid target dimensions");
  ET_CHECK_OR_RETURN_ERROR(
      is_supported_orientation(orientation),
      InvalidArgument,
      "unsupported orientation");
  auto out_err = check_out_tensor(config, out);
  if (out_err != Error::Ok) {
    return out_err;
  }
  float* out_ptr = out.mutable_data_ptr<float>();

  // NV21 stores chroma as Cr,Cb. Swap it to NV12's Cb,Cr ordering once, up
  // front, so both the GPU and CPU paths below are format-agnostic (always
  // NV12).
  const uint8_t* cbcr = uv_plane;
  int32_t cbcr_stride = uv_stride;
  if (format == YUVFormat::NV21) {
    const int32_t chroma_w = (width + 1) / 2;
    const int32_t chroma_h = (height + 1) / 2;
    uint8_t* swapped =
        impl_->uv_swap.resize(static_cast<size_t>(chroma_w) * 2 * chroma_h);
    swap_chroma_cbcr(uv_plane, uv_stride, swapped, chroma_w, chroma_h);
    cbcr = swapped;
    cbcr_stride = chroma_w * 2;
  }

  // GPU fast path: YUV→RGB + crop + resize in a single Core Image pass.
  if (should_use_gpu(config, width, height)) {
    GpuResizeDims gpu;
    compute_gpu_dims(width, height, roi, orientation, config, gpu);
    auto& gpu_resized = impl_->gpu_resized;
    gpu_resized.resize(static_cast<size_t>(gpu.resize_w) * gpu.resize_h * 4);
    int ret = ci_process_yuv_to_bgra(
        y_plane,
        y_stride,
        cbcr,
        cbcr_stride,
        width,
        height,
        static_cast<int32_t>(range),
        to_exif_orientation(orientation),
        roi.x,
        roi.y,
        roi.width,
        roi.height,
        gpu.resize_w,
        gpu.resize_h,
        gpu_resized.data(),
        gpu.resize_w * 4);
    if (ret == 0) {
      return normalize_bgra_into(
          *this,
          gpu_resized.data(),
          gpu.resize_w,
          gpu.resize_h,
          gpu.final_w,
          gpu.final_h,
          gpu.resize_w * 4,
          out_ptr);
    }
    ET_LOG(Debug, "GPU YUV resize failed (ret=%d), falling back to CPU", ret);
  }

  // CPU path: vImage YUV→BGRA (ITU-R 601), honoring the sample range.
  auto makeConversion = [](const vImage_YpCbCrPixelRange& pixel_range) {
    vImage_YpCbCrToARGB info;
    vImageConvert_YpCbCrToARGB_GenerateConversion(
        kvImage_YpCbCrToARGBMatrix_ITU_R_601_4,
        &pixel_range,
        &info,
        kvImage420Yp8_CbCr8,
        kvImageARGB8888,
        kvImageNoFlags);
    return info;
  };
  static const vImage_YpCbCrToARGB cachedVideo =
      makeConversion(kYpCbCrPixelRange_Video);
  static const vImage_YpCbCrToARGB cachedFull =
      makeConversion(kYpCbCrPixelRange_Full);
  const auto& info = (range == YUVRange::FULL) ? cachedFull : cachedVideo;

  // ARGB→BGRA channel order (chroma already normalized to NV12 above).
  const uint8_t permuteMap[4] = {3, 2, 1, 0};

  // CPU fast path: scale Y/CbCr planes first, then convert at target size.
  // Eligible when ROI is the full image and post-resize dims are even.
  const bool fast_eligible = orientation == Orientation::UP && roi.x == 0.0f &&
      roi.y == 0.0f && roi.width == 1.0f && roi.height == 1.0f;
  if (fast_eligible) {
    GpuResizeDims dims;
    compute_gpu_dims(width, height, roi, orientation, config, dims);
    if ((dims.resize_w & 1) == 0 && (dims.resize_h & 1) == 0) {
      const int32_t rw = dims.resize_w;
      const int32_t rh = dims.resize_h;

      const size_t y_bytes = static_cast<size_t>(rw) * rh;
      const size_t uv_bytes = y_bytes / 2;
      uint8_t* yuv_planar = impl_->conv.resize(y_bytes + uv_bytes);
      uint8_t* y_small = yuv_planar;
      uint8_t* uv_small = yuv_planar + y_bytes;

      vImage_Buffer y_src = {
          const_cast<uint8_t*>(y_plane),
          static_cast<vImagePixelCount>(height),
          static_cast<vImagePixelCount>(width),
          static_cast<size_t>(y_stride)};
      vImage_Buffer y_dst = {
          y_small,
          static_cast<vImagePixelCount>(rh),
          static_cast<vImagePixelCount>(rw),
          static_cast<size_t>(rw)};
      vImage_Error verr =
          vImageScale_Planar8(&y_src, &y_dst, nullptr, kvImageNoFlags);
      ET_CHECK_OR_RETURN_ERROR(
          verr == kvImageNoError,
          Internal,
          "vImageScale_Planar8 (Y) failed: %zd",
          verr);

      vImage_Buffer uv_src = {
          const_cast<uint8_t*>(cbcr),
          static_cast<vImagePixelCount>((height + 1) / 2),
          static_cast<vImagePixelCount>((width + 1) / 2),
          static_cast<size_t>(cbcr_stride)};
      // Interleaved CbCr destination: rw/2 samples per row × 2 bytes = rw
      // bytes.
      const size_t uv_dst_stride = static_cast<size_t>(rw);
      vImage_Buffer uv_dst = {
          uv_small,
          static_cast<vImagePixelCount>(rh / 2),
          static_cast<vImagePixelCount>(rw / 2),
          uv_dst_stride};
      verr = vImageScale_CbCr8(&uv_src, &uv_dst, nullptr, kvImageNoFlags);
      ET_CHECK_OR_RETURN_ERROR(
          verr == kvImageNoError,
          Internal,
          "vImageScale_CbCr8 failed: %zd",
          verr);

      const size_t small_bgra_stride = static_cast<size_t>(rw) * 4;
      auto& bgra = impl_->bgra;
      uint8_t* bgra_small = bgra.resize(small_bgra_stride * rh);
      vImage_Buffer bgra_dst = {
          bgra_small,
          static_cast<vImagePixelCount>(rh),
          static_cast<vImagePixelCount>(rw),
          small_bgra_stride};
      verr = vImageConvert_420Yp8_CbCr8ToARGB8888(
          &y_dst, &uv_dst, &bgra_dst, &info, permuteMap, 255, kvImageNoFlags);
      ET_CHECK_OR_RETURN_ERROR(
          verr == kvImageNoError,
          Internal,
          "vImageConvert_420Yp8_CbCr8ToARGB8888 (fast) failed: %zd",
          verr);

      return normalize_bgra_into(
          *this,
          bgra_small,
          rw,
          rh,
          dims.final_w,
          dims.final_h,
          static_cast<int32_t>(small_bgra_stride),
          out_ptr);
    }
  }

  // CPU path: full-resolution YUV→BGRA conversion.
  vImage_Buffer yBuf = {
      const_cast<uint8_t*>(y_plane),
      static_cast<vImagePixelCount>(height),
      static_cast<vImagePixelCount>(width),
      static_cast<size_t>(y_stride)};
  vImage_Buffer uvBuf = {
      const_cast<uint8_t*>(cbcr),
      static_cast<vImagePixelCount>((height + 1) / 2),
      static_cast<vImagePixelCount>((width + 1) / 2),
      static_cast<size_t>(cbcr_stride)};

  const size_t bgra_stride = static_cast<size_t>(width) * 4;
  auto& bgra = impl_->bgra;
  bgra.resize(static_cast<size_t>(height) * bgra_stride);
  vImage_Buffer dstBuf = {
      bgra.data(),
      static_cast<vImagePixelCount>(height),
      static_cast<vImagePixelCount>(width),
      bgra_stride};

  auto vErr = vImageConvert_420Yp8_CbCr8ToARGB8888(
      &yBuf, &uvBuf, &dstBuf, &info, permuteMap, 255, kvImageNoFlags);
  ET_CHECK_OR_RETURN_ERROR(
      vErr == kvImageNoError,
      Internal,
      "vImageConvert_420Yp8_CbCr8ToARGB8888 failed: %zd",
      vErr);

  return process_bgra_cpu_only_into(
      *this, bgra.data(), width, height, orientation, roi, out);
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
      impl_->config.target_width > 0 && impl_->config.target_height > 0,
      InvalidArgument,
      "invalid target dimensions");

  const int32_t final_w = impl_->config.target_width;
  const int32_t final_h = impl_->config.target_height;
  const size_t total = static_cast<size_t>(3) * final_w * final_h;
  std::unique_ptr<float[]> output(new float[total]);

  std::vector<int32_t> shape = {1, 3, final_h, final_w};
  std::vector<executorch::aten::SizesType> tensor_shape(
      shape.begin(), shape.end());
  auto out = make_tensor_ptr(
      std::move(tensor_shape),
      static_cast<void*>(output.release()),
      executorch::aten::ScalarType::Float,
      executorch::aten::TensorShapeDynamism::DYNAMIC_BOUND,
      [](void* p) { delete[] static_cast<float*>(p); });

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
      impl_->config.target_width > 0 && impl_->config.target_height > 0,
      InvalidArgument,
      "invalid target dimensions");

  const int32_t final_w = impl_->config.target_width;
  const int32_t final_h = impl_->config.target_height;
  const size_t total = static_cast<size_t>(3) * final_w * final_h;
  std::unique_ptr<float[]> output(new float[total]);

  std::vector<int32_t> shape = {1, 3, final_h, final_w};
  std::vector<executorch::aten::SizesType> tensor_shape(
      shape.begin(), shape.end());
  auto out = make_tensor_ptr(
      std::move(tensor_shape),
      static_cast<void*>(output.release()),
      executorch::aten::ScalarType::Float,
      executorch::aten::TensorShapeDynamism::DYNAMIC_BOUND,
      [](void* p) { delete[] static_cast<float*>(p); });

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

// --- Apple-specific public helpers (declared in image_processor_apple.h) ---

// Run the pixel-buffer pipeline and write the normalized CHW float result into
// `out`, which must be a contiguous Float tensor shaped [1, 3, target_h,
// target_w]. GPU-enabled processors render directly into `out`; CPU processors
// route through the per-format CPU pipeline. No per-call output allocation.
Error process_pixelbuffer_into(
    const ImageProcessor& processor,
    CVPixelBufferRef pixelBuffer,
    Orientation orientation,
    executorch::aten::Tensor& out) {
  ET_CHECK_OR_RETURN_ERROR(
      pixelBuffer != nullptr, InvalidArgument, "pixelBuffer is null");

  const int32_t width =
      static_cast<int32_t>(CVPixelBufferGetWidth(pixelBuffer));
  const int32_t height =
      static_cast<int32_t>(CVPixelBufferGetHeight(pixelBuffer));
  const OSType pixelFormat = CVPixelBufferGetPixelFormatType(pixelBuffer);

  ET_CHECK_OR_RETURN_ERROR(
      width > 0 && height > 0, InvalidArgument, "invalid dimensions");
  ET_CHECK_OR_RETURN_ERROR(
      processor.config().target_width > 0 &&
          processor.config().target_height > 0,
      InvalidArgument,
      "invalid target dimensions");
  ET_CHECK_OR_RETURN_ERROR(
      is_supported_pixel_format(pixelFormat),
      InvalidArgument,
      "unsupported CVPixelBuffer format");
  ET_CHECK_OR_RETURN_ERROR(
      is_supported_orientation(orientation),
      InvalidArgument,
      "unsupported orientation");

  // Full-range buffers carry samples across the entire [0, 255]; everything
  // else is video range. The conversion must match to avoid color distortion.
  const YUVRange yuv_range =
      (pixelFormat == kCVPixelFormatType_420YpCbCr8BiPlanarFullRange ||
       pixelFormat == kCVPixelFormatType_420YpCbCr10BiPlanarFullRange)
      ? YUVRange::FULL
      : YUVRange::VIDEO;

  // Validate the caller-provided output tensor and obtain its buffer. Use the
  // shared helper so the contiguity check matches the CPU paths below; the GPU
  // branch writes `out` as tightly-packed CHW and would corrupt memory on a
  // non-contiguous tensor.
  if (Error err = check_out_tensor(processor.config(), out); err != Error::Ok) {
    return err;
  }
  float* out_ptr = out.mutable_data_ptr<float>();

  // GPU pixel-buffer-direct fast path. Core Image renders the resized image to
  // 8-bit BGRA (4 B/px, vs 16 B/px for float) to keep the GPU→CPU readback
  // small; normalize does the uint8->float conversion.
  if (should_use_gpu(processor.config(), width, height)) {
    int32_t resize_w, resize_h, final_w, final_h;
    const auto od = oriented_dims(width, height, orientation);
    compute_resize_dims(
        od.first,
        od.second,
        processor.config(),
        resize_w,
        resize_h,
        final_w,
        final_h);

    auto& gpu_resized = processor.impl().gpu_resized;
    gpu_resized.resize(static_cast<size_t>(resize_w) * resize_h * 4);
    const int32_t bgra_stride = resize_w * 4;

    // process_pixelbuffer processes the full image; kFullImage is the ROI
    // forwarded to the helper.
    static_assert(
        kFullImage.x == 0.0f && kFullImage.y == 0.0f &&
            kFullImage.width == 1.0f && kFullImage.height == 1.0f,
        "kFullImage must be {0,0,1,1}");
    int ret = ci_process_pixelbuffer_to_bgra(
        pixelBuffer,
        to_exif_orientation(orientation),
        kFullImage.x,
        kFullImage.y,
        kFullImage.width,
        kFullImage.height,
        resize_w,
        resize_h,
        gpu_resized.data(),
        bgra_stride);

    if (ret == 0) {
      return normalize_bgra_into(
          processor,
          gpu_resized.data(),
          resize_w,
          resize_h,
          final_w,
          final_h,
          bgra_stride,
          out_ptr);
    }
    ET_LOG(
        Debug,
        "GPU pixelbuffer resize failed (ret=%d), falling back to CPU",
        ret);
    // GPU failed — fall through to CPU path.
  }

  // CPU path. Lock the pixel buffer's base address and dispatch on format.
  // When the processor can use the GPU, route through a force-CPU proxy
  // (cached on the processor's pImpl) so process()/process_yuv() do not
  // re-attempt the GPU path on the bytes just locked into CPU memory.
  const ImageProcessor* cpu_processor = &processor;
  if (!is_cpu_only(processor.config())) {
    auto& cpu_proxy = processor.impl().cpu_proxy;
    if (!cpu_proxy) {
      ImageProcessorConfig cpu_config = processor.config();
      cpu_config.gpu_min_input_pixels = ImageProcessorConfig::kGpuNever;
      cpu_proxy = std::make_unique<ImageProcessor>(cpu_config);
    }
    cpu_processor = cpu_proxy.get();
  }

  if (CVPixelBufferLockBaseAddress(pixelBuffer, kCVPixelBufferLock_ReadOnly) !=
      kCVReturnSuccess) {
    return Error::AccessFailed;
  }
  Error result = [&]() -> Error {
    // BGRA / RGBA: hand bytes directly to the CPU pipeline.
    if (pixelFormat == kCVPixelFormatType_32BGRA ||
        pixelFormat == kCVPixelFormatType_32RGBA) {
      const ColorFormat fmt = (pixelFormat == kCVPixelFormatType_32BGRA)
          ? ColorFormat::BGRA
          : ColorFormat::RGBA;
      const auto* data =
          static_cast<const uint8_t*>(CVPixelBufferGetBaseAddress(pixelBuffer));
      const int32_t stride =
          static_cast<int32_t>(CVPixelBufferGetBytesPerRow(pixelBuffer));
      return cpu_processor->process_into(
          data, width, height, stride, fmt, out, orientation, kFullImage);
    }

    // 8-bit NV12 (semi-planar Y + interleaved CbCr).
    if (pixelFormat == kCVPixelFormatType_420YpCbCr8BiPlanarVideoRange ||
        pixelFormat == kCVPixelFormatType_420YpCbCr8BiPlanarFullRange) {
      const auto* y = static_cast<const uint8_t*>(
          CVPixelBufferGetBaseAddressOfPlane(pixelBuffer, 0));
      const int32_t y_stride = static_cast<int32_t>(
          CVPixelBufferGetBytesPerRowOfPlane(pixelBuffer, 0));
      const auto* uv = static_cast<const uint8_t*>(
          CVPixelBufferGetBaseAddressOfPlane(pixelBuffer, 1));
      const int32_t uv_stride = static_cast<int32_t>(
          CVPixelBufferGetBytesPerRowOfPlane(pixelBuffer, 1));
      return cpu_processor->process_yuv_into(
          y,
          y_stride,
          uv,
          uv_stride,
          width,
          height,
          YUVFormat::NV12,
          out,
          orientation,
          kFullImage,
          yuv_range);
    }

    // 10-bit P010: narrow each 16-bit sample to its high byte (8-bit NV12),
    // then dispatch to process_yuv.
    if (pixelFormat == kCVPixelFormatType_420YpCbCr10BiPlanarVideoRange ||
        pixelFormat == kCVPixelFormatType_420YpCbCr10BiPlanarFullRange) {
      const int32_t y_stride16 = static_cast<int32_t>(
          CVPixelBufferGetBytesPerRowOfPlane(pixelBuffer, 0));
      const int32_t uv_stride16 = static_cast<int32_t>(
          CVPixelBufferGetBytesPerRowOfPlane(pixelBuffer, 1));
      const auto* y16 = static_cast<const uint16_t*>(
          CVPixelBufferGetBaseAddressOfPlane(pixelBuffer, 0));
      const auto* uv16 = static_cast<const uint16_t*>(
          CVPixelBufferGetBaseAddressOfPlane(pixelBuffer, 1));

      const int32_t uv_height = (height + 1) / 2;
      const int32_t uv_width = (width + 1) / 2;
      const int32_t uv_samples_per_row = uv_width * 2;

      // Reuse per-processor scratch (no per-frame malloc/free) and narrow with
      // NEON instead of a scalar high-byte loop.
      auto& narrow_y = cpu_processor->impl().narrow_y;
      auto& narrow_uv = cpu_processor->impl().narrow_uv;
      uint8_t* y8 = narrow_y.resize(static_cast<size_t>(width) * height);
      uint8_t* uv8 =
          narrow_uv.resize(static_cast<size_t>(uv_samples_per_row) * uv_height);

      narrow_plane_p010_to_8bit(
          reinterpret_cast<const uint8_t*>(y16), y_stride16, y8, width, height);
      narrow_plane_p010_to_8bit(
          reinterpret_cast<const uint8_t*>(uv16),
          uv_stride16,
          uv8,
          uv_samples_per_row,
          uv_height);

      return cpu_processor->process_yuv_into(
          y8,
          width,
          uv8,
          uv_samples_per_row,
          width,
          height,
          YUVFormat::NV12,
          out,
          orientation,
          kFullImage,
          yuv_range);
    }

    return Error::InvalidArgument;
  }();
  CVPixelBufferUnlockBaseAddress(pixelBuffer, kCVPixelBufferLock_ReadOnly);
  return result;
}

// Allocate a CHW float tensor sized to the configured target and fill it via
// process_pixelbuffer_into.
Result<TensorPtr> process_pixelbuffer(
    const ImageProcessor& processor,
    CVPixelBufferRef pixelBuffer,
    Orientation orientation) {
  ET_CHECK_OR_RETURN_ERROR(
      processor.config().target_width > 0 &&
          processor.config().target_height > 0,
      InvalidArgument,
      "invalid target dimensions");

  const int32_t final_w = processor.config().target_width;
  const int32_t final_h = processor.config().target_height;
  const size_t total = static_cast<size_t>(3) * final_w * final_h;
  std::unique_ptr<float[]> output(new float[total]);

  std::vector<int32_t> shape = {1, 3, final_h, final_w};
  std::vector<executorch::aten::SizesType> tensor_shape(
      shape.begin(), shape.end());
  auto out = make_tensor_ptr(
      std::move(tensor_shape),
      static_cast<void*>(output.release()),
      executorch::aten::ScalarType::Float,
      executorch::aten::TensorShapeDynamism::DYNAMIC_BOUND,
      [](void* p) { delete[] static_cast<float*>(p); });

  auto err =
      process_pixelbuffer_into(processor, pixelBuffer, orientation, *out);
  if (err != Error::Ok) {
    return err;
  }
  return out;
}

} // namespace image
} // namespace extension
} // namespace executorch
