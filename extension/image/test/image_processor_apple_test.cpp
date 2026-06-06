/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

// Apple-specific ImageProcessor tests. These exercise the Core Image GPU paths
// and the CVPixelBuffer entry point, asserting they match the CPU pipeline for
// cases the portable tests cannot reach. The whole file is gated on __APPLE__
// so it is an empty translation unit on non-Apple platforms.

#ifdef __APPLE__

#include <executorch/extension/image/image_processor.h>
#include <executorch/extension/image/image_processor_apple.h>

#include <cstdint>
#include <vector>

#include <CoreFoundation/CoreFoundation.h>
#include <CoreVideo/CoreVideo.h>
#include <gtest/gtest.h>

#include <executorch/extension/tensor/tensor_ptr.h>
#include <executorch/runtime/platform/platform.h>

using namespace executorch::extension::image;
using executorch::extension::make_tensor_ptr;
using executorch::extension::TensorPtr;
using executorch::runtime::Error;

// Initialize PAL before running tests (needed for ET_LOG on error paths).
class AppleImageProcessorTestEnvironment : public ::testing::Environment {
 public:
  void SetUp() override {
    et_pal_init();
  }
};

const ::testing::Environment* const apple_image_processor_test_env =
    ::testing::AddGlobalTestEnvironment(new AppleImageProcessorTestEnvironment);

namespace {

// Build the {kCVPixelBufferIOSurfacePropertiesKey: {}} attributes dictionary
// that requests IOSurface-backed storage (needed for the zero-copy GPU path).
// Uses CoreFoundation rather than Objective-C literals so this file stays plain
// C++. Caller owns the returned dictionary and must CFRelease it.
CFDictionaryRef make_iosurface_attrs() {
  CFDictionaryRef empty = CFDictionaryCreate(
      kCFAllocatorDefault,
      nullptr,
      nullptr,
      0,
      &kCFTypeDictionaryKeyCallBacks,
      &kCFTypeDictionaryValueCallBacks);
  const void* keys[] = {kCVPixelBufferIOSurfacePropertiesKey};
  const void* values[] = {empty};
  CFDictionaryRef attrs = CFDictionaryCreate(
      kCFAllocatorDefault,
      keys,
      values,
      1,
      &kCFTypeDictionaryKeyCallBacks,
      &kCFTypeDictionaryValueCallBacks);
  CFRelease(empty);
  return attrs;
}

// Horizontally-split content: left half [0, w/2) one solid color, right half
// [w/2, w) another. Used to detect a wrong-region ROI crop.
std::vector<uint8_t> make_split_bgra(
    int32_t w,
    int32_t h,
    uint8_t lr,
    uint8_t lg,
    uint8_t lb,
    uint8_t rr,
    uint8_t rg,
    uint8_t rb) {
  std::vector<uint8_t> img(static_cast<size_t>(w) * h * 4);
  for (int32_t y = 0; y < h; ++y) {
    for (int32_t x = 0; x < w; ++x) {
      const size_t i = (static_cast<size_t>(y) * w + x) * 4;
      const bool right = x >= w / 2;
      img[i + 0] = right ? rb : lb; // B
      img[i + 1] = right ? rg : lg; // G
      img[i + 2] = right ? rr : lr; // R
      img[i + 3] = 255;
    }
  }
  return img;
}

// Vertically-split content: top half [0, h/2) one solid color, bottom half
// [h/2, h) another. Used to detect a wrong-region (or vertically-flipped) ROI
// crop along the y-axis.
std::vector<uint8_t> make_vsplit_bgra(
    int32_t w,
    int32_t h,
    uint8_t tr,
    uint8_t tg,
    uint8_t tb,
    uint8_t br,
    uint8_t bg,
    uint8_t bb) {
  std::vector<uint8_t> img(static_cast<size_t>(w) * h * 4);
  for (int32_t y = 0; y < h; ++y) {
    for (int32_t x = 0; x < w; ++x) {
      const size_t i = (static_cast<size_t>(y) * w + x) * 4;
      const bool bottom = y >= h / 2;
      img[i + 0] = bottom ? bb : tb; // B
      img[i + 1] = bottom ? bg : tg; // G
      img[i + 2] = bottom ? br : tr; // R
      img[i + 3] = 255;
    }
  }
  return img;
}

// Create a solid-color 32BGRA CVPixelBuffer (caller releases).
CVPixelBufferRef
make_bgra_pixelbuffer(int32_t w, int32_t h, uint8_t r, uint8_t g, uint8_t b) {
  CVPixelBufferRef pb = nullptr;
  const CVReturn status = CVPixelBufferCreate(
      kCFAllocatorDefault, w, h, kCVPixelFormatType_32BGRA, nullptr, &pb);
  if (status != kCVReturnSuccess || pb == nullptr) {
    return nullptr;
  }
  CVPixelBufferLockBaseAddress(pb, 0);
  auto* base = static_cast<uint8_t*>(CVPixelBufferGetBaseAddress(pb));
  const size_t stride = CVPixelBufferGetBytesPerRow(pb);
  for (int32_t y = 0; y < h; ++y) {
    for (int32_t x = 0; x < w; ++x) {
      uint8_t* px = base + static_cast<size_t>(y) * stride + x * 4;
      px[0] = b;
      px[1] = g;
      px[2] = r;
      px[3] = 255;
    }
  }
  CVPixelBufferUnlockBaseAddress(pb, 0);
  return pb;
}

// Create a 10-bit P010 (420YpCbCr10BiPlanar) CVPixelBuffer (caller releases).
CVPixelBufferRef
make_p010_pixelbuffer(int32_t w, int32_t h, uint8_t y_val, uint8_t uv_val) {
  CVPixelBufferRef pb = nullptr;
  CFDictionaryRef attrs = make_iosurface_attrs();
  const CVReturn status = CVPixelBufferCreate(
      kCFAllocatorDefault,
      w,
      h,
      kCVPixelFormatType_420YpCbCr10BiPlanarVideoRange,
      attrs,
      &pb);
  CFRelease(attrs);
  if (status != kCVReturnSuccess || pb == nullptr) {
    return nullptr;
  }

  CVPixelBufferLockBaseAddress(pb, 0);

  // Fill Y plane (16-bit values, upper 8 bits contain the 10-bit data)
  uint16_t* y_base =
      static_cast<uint16_t*>(CVPixelBufferGetBaseAddressOfPlane(pb, 0));
  const size_t y_stride = CVPixelBufferGetBytesPerRowOfPlane(pb, 0) / 2;
  const uint16_t y_val_10bit = static_cast<uint16_t>(y_val) << 8;
  for (int32_t y = 0; y < h; ++y) {
    for (int32_t x = 0; x < w; ++x) {
      y_base[y * y_stride + x] = y_val_10bit;
    }
  }

  // Fill UV plane (interleaved 16-bit CbCr values)
  uint16_t* uv_base =
      static_cast<uint16_t*>(CVPixelBufferGetBaseAddressOfPlane(pb, 1));
  const size_t uv_stride = CVPixelBufferGetBytesPerRowOfPlane(pb, 1) / 2;
  const int32_t uv_h = (h + 1) / 2;
  const int32_t uv_w = (w + 1) / 2;
  const uint16_t uv_val_10bit = static_cast<uint16_t>(uv_val) << 8;
  for (int32_t y = 0; y < uv_h; ++y) {
    for (int32_t x = 0; x < uv_w; ++x) {
      uv_base[y * uv_stride + x * 2] = uv_val_10bit; // Cb
      uv_base[y * uv_stride + x * 2 + 1] = uv_val_10bit; // Cr
    }
  }

  CVPixelBufferUnlockBaseAddress(pb, 0);
  return pb;
}

// Create an 8-bit NV12 (420YpCbCr8BiPlanar) CVPixelBuffer in the given range
// (pass kCVPixelFormatType_420YpCbCr8BiPlanar{Video,Full}Range). Plane 0 is the
// Y plane; plane 1 is interleaved CbCr. Caller releases.
CVPixelBufferRef make_nv12_pixelbuffer(
    int32_t w,
    int32_t h,
    uint8_t y_val,
    uint8_t cb_val,
    uint8_t cr_val,
    OSType format) {
  CVPixelBufferRef pb = nullptr;
  CFDictionaryRef attrs = make_iosurface_attrs();
  const CVReturn status =
      CVPixelBufferCreate(kCFAllocatorDefault, w, h, format, attrs, &pb);
  CFRelease(attrs);
  if (status != kCVReturnSuccess || pb == nullptr) {
    return nullptr;
  }

  CVPixelBufferLockBaseAddress(pb, 0);

  uint8_t* y_base =
      static_cast<uint8_t*>(CVPixelBufferGetBaseAddressOfPlane(pb, 0));
  const size_t y_stride = CVPixelBufferGetBytesPerRowOfPlane(pb, 0);
  for (int32_t y = 0; y < h; ++y) {
    for (int32_t x = 0; x < w; ++x) {
      y_base[y * y_stride + x] = y_val;
    }
  }

  uint8_t* uv_base =
      static_cast<uint8_t*>(CVPixelBufferGetBaseAddressOfPlane(pb, 1));
  const size_t uv_stride = CVPixelBufferGetBytesPerRowOfPlane(pb, 1);
  const int32_t uv_h = (h + 1) / 2;
  const int32_t uv_w = (w + 1) / 2;
  for (int32_t y = 0; y < uv_h; ++y) {
    for (int32_t x = 0; x < uv_w; ++x) {
      uv_base[y * uv_stride + x * 2] = cb_val; // Cb
      uv_base[y * uv_stride + x * 2 + 1] = cr_val; // Cr
    }
  }

  CVPixelBufferUnlockBaseAddress(pb, 0);
  return pb;
}

ImageProcessorConfig make_config(int32_t w, int32_t h) {
  ImageProcessorConfig config;
  config.target_width = w;
  config.target_height = h;
  return config;
}

// Solid-color semi-planar YUV in CPU memory (raw planes, no CVPixelBuffer).
// NV12 stores chroma as Cb,Cr; NV21 as Cr,Cb -- so the SAME logical (cb, cr)
// decodes to the same RGB under either format, letting a test assert NV21 ==
// NV12 to prove the Cr<->Cb correction is applied. `y` is w*h bytes (stride w);
// `uv` is (w/2 * h/2) interleaved chroma pairs (stride w). Requires even w, h.
struct PlanarYuv {
  std::vector<uint8_t> y;
  std::vector<uint8_t> uv;
};

PlanarYuv make_solid_yuv(
    int32_t w,
    int32_t h,
    uint8_t y_val,
    uint8_t cb,
    uint8_t cr,
    YUVFormat format) {
  PlanarYuv out;
  out.y.assign(static_cast<size_t>(w) * h, y_val);
  const int32_t cw = w / 2;
  const int32_t ch = h / 2;
  out.uv.resize(static_cast<size_t>(cw) * ch * 2);
  const bool nv12 = (format == YUVFormat::NV12);
  for (int32_t i = 0; i < cw * ch; ++i) {
    out.uv[i * 2 + 0] = nv12 ? cb : cr;
    out.uv[i * 2 + 1] = nv12 ? cr : cb;
  }
  return out;
}

// Config of the given target size forced onto the CPU path.
ImageProcessorConfig cpu_config(int32_t w, int32_t h) {
  auto config = make_config(w, h);
  config.gpu_min_input_pixels = ImageProcessorConfig::kGpuNever;
  return config;
}

// Config of the given target size forced onto the GPU path.
ImageProcessorConfig gpu_config(int32_t w, int32_t h) {
  auto config = make_config(w, h);
  config.gpu_min_input_pixels = ImageProcessorConfig::kGpuAlways;
  return config;
}

// Assert two CHW float result tensors are elementwise close.
void expect_tensors_near(
    const TensorPtr& a,
    const TensorPtr& b,
    float eps = 0.05f) {
  ASSERT_EQ(a->numel(), b->numel());
  const float* pa = a->const_data_ptr<float>();
  const float* pb = b->const_data_ptr<float>();
  for (int64_t i = 0; i < a->numel(); ++i) {
    EXPECT_NEAR(pa[i], pb[i], eps) << "mismatch at " << i;
  }
}

} // namespace

// Verifies the Core Image ROI crop is rebased to the coordinate-space origin
// so the render bounds {0,0,tw,th} sample the correct region.
TEST(AppleRoiTest, OffsetRoiCpuGpuEquivalence) {
  // Right-half ROI (x-offset only, full height) on horizontally-split content.
  // The x-only offset keeps this focused on the render-bounds origin and
  // sidesteps the separate y-axis convention question.
  const int32_t w = 8, h = 8;
  auto bgra =
      make_split_bgra(w, h, /*left*/ 30, 60, 90, /*right*/ 200, 150, 100);
  const NormalizedRect roi{0.5f, 0.0f, 0.5f, 1.0f};

  ImageProcessor cpu(cpu_config(4, 4));
  ImageProcessor gpu(gpu_config(4, 4));
  auto cpu_res = cpu.process(
      bgra.data(), w, h, w * 4, ColorFormat::BGRA, Orientation::UP, roi);
  auto gpu_res = gpu.process(
      bgra.data(), w, h, w * 4, ColorFormat::BGRA, Orientation::UP, roi);
  ASSERT_TRUE(cpu_res.ok());
  ASSERT_TRUE(gpu_res.ok());
  expect_tensors_near(cpu_res.get(), gpu_res.get());

  // The right-half ROI is the solid 'right' color, so the result must be that
  // color -- guards against selecting the wrong region even if cpu == gpu.
  EXPECT_NEAR(
      cpu_res.get()->const_data_ptr<float>()[0], 200.0f / 255.0f, 0.02f);
}

// Mirror of OffsetRoiCpuGpuEquivalence for the y-axis: a bottom-half ROI
// (y-offset only, full width) on vertically-split content. Core Image's
// coordinate origin is bottom-left while the CPU pipeline treats the buffer as
// top-origin, so a y-offset ROI is the case where the two conventions could
// diverge. Verifies they crop the same region and, via the anchor below,
// the correct (non-flipped) one.
TEST(AppleRoiTest, OffsetRoiYAxisCpuGpuEquivalence) {
  const int32_t w = 8, h = 8;
  auto bgra =
      make_vsplit_bgra(w, h, /*top*/ 30, 60, 90, /*bottom*/ 200, 150, 100);
  const NormalizedRect roi{0.0f, 0.5f, 1.0f, 0.5f};

  ImageProcessor cpu(cpu_config(4, 4));
  ImageProcessor gpu(gpu_config(4, 4));
  auto cpu_res = cpu.process(
      bgra.data(), w, h, w * 4, ColorFormat::BGRA, Orientation::UP, roi);
  auto gpu_res = gpu.process(
      bgra.data(), w, h, w * 4, ColorFormat::BGRA, Orientation::UP, roi);
  ASSERT_TRUE(cpu_res.ok());
  ASSERT_TRUE(gpu_res.ok());
  expect_tensors_near(cpu_res.get(), gpu_res.get());

  // The bottom-half ROI is the solid 'bottom' color, so the result must be that
  // color -- guards against selecting the wrong (e.g. vertically-flipped)
  // region even if cpu == gpu.
  EXPECT_NEAR(
      cpu_res.get()->const_data_ptr<float>()[0], 200.0f / 255.0f, 0.02f);
}

// Verifies RGBAf letterbox normalization follows the strided sub-rectangle
// rather than treating it as one contiguous block.
TEST(ApplePixelBufferTest, ImageNetLetterboxCpuGpuEquivalence) {
  // A tall (portrait) input letterboxed into a square target produces
  // horizontal padding (resize_w < target_width), routing the GPU RGBAf path
  // through the strided per-row normalize. With a non-identity (imagenet)
  // normalization, a single contiguous vDSP pass would corrupt the pad columns
  // between rows and skip trailing content rows. The GPU pixel-buffer path must
  // match the CPU pipeline (which normalizes BGRA per-row).
  CVPixelBufferRef pb = make_bgra_pixelbuffer(4, 12, 200, 100, 50);
  ASSERT_NE(pb, nullptr);

  auto make = [](bool cpu_only) {
    ImageProcessorConfig c = make_config(8, 8);
    c.resize_mode = ResizeMode::LETTERBOX;
    c.letterbox_anchor = LetterboxAnchor::CENTER;
    c.pad_value = 0.0f;
    c.normalization = Normalization::imagenet();
    c.gpu_min_input_pixels = cpu_only ? ImageProcessorConfig::kGpuNever
                                      : ImageProcessorConfig::kGpuAlways;
    return c;
  };

  ImageProcessor cpu(make(true));
  ImageProcessor gpu(make(false));
  auto cpu_res = process_pixelbuffer(cpu, pb);
  auto gpu_res = process_pixelbuffer(gpu, pb);
  CVPixelBufferRelease(pb);

  ASSERT_TRUE(cpu_res.ok());
  ASSERT_TRUE(gpu_res.ok());
  expect_tensors_near(cpu_res.get(), gpu_res.get());
}

// Verifies 10-bit P010 (420YpCbCr10BiPlanar) pixel buffer format works.
TEST(ApplePixelBufferTest, P010Format) {
  CVPixelBufferRef pb = make_p010_pixelbuffer(8, 6, 128, 128);
  ASSERT_NE(pb, nullptr);

  ImageProcessor processor(make_config(4, 4));
  auto result = process_pixelbuffer(processor, pb);
  CVPixelBufferRelease(pb);

  ASSERT_TRUE(result.ok());
  auto& tensor = result.get();
  EXPECT_EQ(tensor->size(0), 1);
  EXPECT_EQ(tensor->size(1), 3);
  EXPECT_EQ(tensor->size(2), 4);
  EXPECT_EQ(tensor->size(3), 4);

  const float* data = tensor->const_data_ptr<float>();
  // Y=128, U=128, V=128 should produce mid-range RGB values
  const float r0 = data[0];
  EXPECT_GT(r0, 0.3f);
  EXPECT_LT(r0, 0.7f);

  // All pixels should be consistent (solid color input)
  for (int i = 1; i < 16; ++i) {
    EXPECT_NEAR(data[i], r0, 0.03f) << "R at " << i;
  }
}

// Verifies P010 format produces similar results on CPU and GPU.
TEST(ApplePixelBufferTest, P010CpuGpuEquivalence) {
  CVPixelBufferRef pb = make_p010_pixelbuffer(8, 6, 128, 128);
  ASSERT_NE(pb, nullptr);

  ImageProcessor cpu(cpu_config(4, 4));
  ImageProcessor gpu(gpu_config(4, 4));
  auto cpu_res = process_pixelbuffer(cpu, pb);
  auto gpu_res = process_pixelbuffer(gpu, pb);
  CVPixelBufferRelease(pb);

  ASSERT_TRUE(cpu_res.ok());
  ASSERT_TRUE(gpu_res.ok());
  expect_tensors_near(cpu_res.get(), gpu_res.get());
}

// 8-bit NV12 carries its quantization range in its pixel-format type
// (...8BiPlanarVideoRange vs ...8BiPlanarFullRange). The decode must honor it:
// the GPU path (Core Image) reads the range from the buffer and decodes
// correctly, and the CPU pipeline must match.
//
// Neutral chroma (Cb=Cr=128) makes R=G=B a function of luma alone, so the
// matrix (601 vs 709) is irrelevant and only the range matters:
//   full range:  channel = Y / 255
//   video range: channel = clamp((Y - 16) / 219, 0, 1)
// At Y=235 these diverge maximally: full ~= 0.922, video clamps to 1.0
// (diff ~0.078, well beyond kEps). A CPU decode that assumes video range for a
// full-range buffer therefore reads ~1.0 and fails this comparison.
TEST(ApplePixelBufferTest, FullRangeNV12CpuGpuEquivalence) {
  const int32_t w = 8, h = 6;
  // Bright gray that is full-range white-ish but *above* the video-range white
  // point (235), so a video-range decode over-stretches it to clipping.
  const uint8_t y_val = 235;

  CVPixelBufferRef pb = make_nv12_pixelbuffer(
      w,
      h,
      y_val,
      /*cb*/ 128,
      /*cr*/ 128,
      kCVPixelFormatType_420YpCbCr8BiPlanarFullRange);
  ASSERT_NE(pb, nullptr);

  ImageProcessor cpu(cpu_config(4, 4));
  ImageProcessor gpu(gpu_config(4, 4));
  auto cpu_res = process_pixelbuffer(cpu, pb);
  auto gpu_res = process_pixelbuffer(gpu, pb);
  CVPixelBufferRelease(pb);

  ASSERT_TRUE(cpu_res.ok());
  ASSERT_TRUE(gpu_res.ok());

  // Anchor the correct answer: full-range neutral-chroma gray decodes to ~Y/255
  // per channel, with the GPU path as the reference.
  EXPECT_NEAR(
      gpu_res.get()->const_data_ptr<float>()[0],
      static_cast<float>(y_val) / 255.0f,
      0.03f);
  expect_tensors_near(cpu_res.get(), gpu_res.get());
}

// RGBA raw bytes take a separate route from BGRA on both backends (GPU uses
// CI_PIXEL_FORMAT_RGBA8; CPU permutes via to_bgra). Distinct R/G/B values make
// a wrong channel mapping detectable. The two backends must agree.
TEST(AppleColorFormatTest, RgbaRawBytesCpuGpuEquivalence) {
  const int32_t w = 8, h = 8;
  const uint8_t R = 200, G = 120, B = 40;
  std::vector<uint8_t> rgba(static_cast<size_t>(w) * h * 4);
  for (int32_t i = 0; i < w * h; ++i) {
    rgba[i * 4 + 0] = R;
    rgba[i * 4 + 1] = G;
    rgba[i * 4 + 2] = B;
    rgba[i * 4 + 3] = 255;
  }

  ImageProcessor cpu(cpu_config(4, 4));
  ImageProcessor gpu(gpu_config(4, 4));
  auto cpu_res = cpu.process(
      rgba.data(), w, h, w * 4, ColorFormat::RGBA, Orientation::UP, kFullImage);
  auto gpu_res = gpu.process(
      rgba.data(), w, h, w * 4, ColorFormat::RGBA, Orientation::UP, kFullImage);
  ASSERT_TRUE(cpu_res.ok());
  ASSERT_TRUE(gpu_res.ok());
  expect_tensors_near(cpu_res.get(), gpu_res.get());

  // Channel-order anchor: output is CHW (R, G, B planes). A BGRA/RGBA mixup
  // would swap the R and B planes.
  const float* cpu_data = cpu_res.get()->const_data_ptr<float>();
  const size_t spatial = static_cast<size_t>(4) * 4;
  EXPECT_NEAR(cpu_data[0], R / 255.0f, 0.02f); // R plane
  EXPECT_NEAR(cpu_data[2 * spatial], B / 255.0f, 0.02f); // B plane
}

// Combined x+y ROI offset (bottom-right quarter). The single-axis ROI tests
// cover x and y independently; this locks in both offsets together. Built
// inline as four red quadrants (TL=50, TR=100, BL=150, BR=200) so the selected
// region's color is unambiguous.
TEST(AppleRoiTest, OffsetRoiXYCpuGpuEquivalence) {
  const int32_t w = 8;
  const int32_t h = 8;
  std::vector<uint8_t> bgra(static_cast<size_t>(w) * h * 4);
  for (int32_t y = 0; y < h; ++y) {
    for (int32_t x = 0; x < w; ++x) {
      const size_t i = (static_cast<size_t>(y) * w + x) * 4;
      const bool bottom = y >= h / 2;
      const bool right = x >= w / 2;
      bgra[i + 0] = 0; // B
      bgra[i + 1] = 0; // G
      bgra[i + 2] = bottom ? (right ? 200 : 150) : (right ? 100 : 50); // R
      bgra[i + 3] = 255;
    }
  }
  // Bottom-right quarter -> the BR quadrant (R=200).
  const NormalizedRect roi{0.5f, 0.5f, 0.5f, 0.5f};

  ImageProcessor cpu(cpu_config(4, 4));
  ImageProcessor gpu(gpu_config(4, 4));
  auto cpu_res = cpu.process(
      bgra.data(), w, h, w * 4, ColorFormat::BGRA, Orientation::UP, roi);
  auto gpu_res = gpu.process(
      bgra.data(), w, h, w * 4, ColorFormat::BGRA, Orientation::UP, roi);
  ASSERT_TRUE(cpu_res.ok());
  ASSERT_TRUE(gpu_res.ok());
  expect_tensors_near(cpu_res.get(), gpu_res.get());

  // Bottom-right quadrant is solid R=200; guards against selecting the wrong
  // corner even if cpu == gpu.
  EXPECT_NEAR(
      cpu_res.get()->const_data_ptr<float>()[0], 200.0f / 255.0f, 0.02f);
}

// process_yuv() raw-planes GPU path (ci_process_yuv_to_bgra, which synthesizes
// a CVPixelBuffer from the planes) is otherwise untested -- the pixel-buffer
// tests go through a different helper (ci_process_pixelbuffer_to_bgra).
// Non-neutral chroma exercises the full YUV->RGB matrix; both backends use
// BT.601 and must agree.
TEST(AppleYuvTest, Nv12ProcessYuvCpuGpuEquivalence) {
  const int32_t w = 8, h = 6;
  const auto yuv =
      make_solid_yuv(w, h, /*y*/ 150, /*cb*/ 100, /*cr*/ 180, YUVFormat::NV12);

  ImageProcessor cpu(cpu_config(4, 4));
  ImageProcessor gpu(gpu_config(4, 4));
  auto cpu_res =
      cpu.process_yuv(yuv.y.data(), w, yuv.uv.data(), w, w, h, YUVFormat::NV12);
  auto gpu_res =
      gpu.process_yuv(yuv.y.data(), w, yuv.uv.data(), w, w, h, YUVFormat::NV12);
  ASSERT_TRUE(cpu_res.ok());
  ASSERT_TRUE(gpu_res.ok());
  expect_tensors_near(cpu_res.get(), gpu_res.get());
}

// NV21 reaches Apple only via process_yuv (CoreVideo has no NV21 pixel format),
// and its Cr<->Cb correction is implemented differently per backend (CPU
// permute vs GPU CIColorMatrix), so they can drift. Verify CPU == GPU, and that
// NV21 decodes identically to an NV12 buffer carrying the same logical chroma
// -- i.e. the swap is actually applied (a no-op swap would diverge under the
// non-neutral chroma used here).
TEST(AppleYuvTest, Nv21ProcessYuvCpuGpuEquivalence) {
  const int32_t w = 8;
  const int32_t h = 6;
  const uint8_t yv = 150, cb = 100, cr = 180;
  const auto nv21 = make_solid_yuv(w, h, yv, cb, cr, YUVFormat::NV21);
  const auto nv12 = make_solid_yuv(w, h, yv, cb, cr, YUVFormat::NV12);

  ImageProcessor cpu(cpu_config(4, 4));
  ImageProcessor gpu(gpu_config(4, 4));
  auto nv21_cpu = cpu.process_yuv(
      nv21.y.data(), w, nv21.uv.data(), w, w, h, YUVFormat::NV21);
  auto nv21_gpu = gpu.process_yuv(
      nv21.y.data(), w, nv21.uv.data(), w, w, h, YUVFormat::NV21);
  auto nv12_cpu = cpu.process_yuv(
      nv12.y.data(), w, nv12.uv.data(), w, w, h, YUVFormat::NV12);
  ASSERT_TRUE(nv21_cpu.ok());
  ASSERT_TRUE(nv21_gpu.ok());
  ASSERT_TRUE(nv12_cpu.ok());

  expect_tensors_near(nv21_cpu.get(), nv21_gpu.get()); // cpu matches gpu
  expect_tensors_near(nv21_cpu.get(), nv12_cpu.get()); // chroma swap applied
}

// process_pixelbuffer_into writes into a caller-owned tensor in place and must
// produce the same result as the allocating process_pixelbuffer variant.
// Verifies the result is written into `out`'s existing storage (no realloc).
TEST(ApplePixelBufferIntoTest, WritesIntoOutAndMatchesProcessPixelbuffer) {
  CVPixelBufferRef pb = make_bgra_pixelbuffer(8, 8, 200, 100, 50);
  ASSERT_NE(pb, nullptr);

  ImageProcessor processor(make_config(4, 4));
  auto ref = process_pixelbuffer(processor, pb);
  ASSERT_TRUE(ref.ok());

  auto out = make_tensor_ptr({1, 3, 4, 4}, std::vector<float>(3 * 4 * 4));
  const float* storage = out->const_data_ptr<float>();
  auto err = process_pixelbuffer_into(processor, pb, Orientation::UP, *out);
  CVPixelBufferRelease(pb);

  ASSERT_EQ(err, Error::Ok);
  // Result landed in the caller-provided buffer, not a freshly allocated one.
  EXPECT_EQ(out->const_data_ptr<float>(), storage);
  expect_tensors_near(out, ref.get());
}

// The same `out` tensor (and its backing allocation) can be reused across
// frames; each call overwrites it with the current frame's result.
TEST(ApplePixelBufferIntoTest, ReuseAcrossFrames) {
  ImageProcessor processor(make_config(4, 4));
  auto out = make_tensor_ptr({1, 3, 4, 4}, std::vector<float>(3 * 4 * 4));
  const float* storage = out->const_data_ptr<float>();

  CVPixelBufferRef pb1 = make_bgra_pixelbuffer(8, 8, 200, 100, 50);
  ASSERT_NE(pb1, nullptr);
  ASSERT_EQ(
      process_pixelbuffer_into(processor, pb1, Orientation::UP, *out),
      Error::Ok);
  auto ref1 = process_pixelbuffer(processor, pb1);
  CVPixelBufferRelease(pb1);
  ASSERT_TRUE(ref1.ok());
  expect_tensors_near(out, ref1.get());

  // A second, differently-colored frame written into the same tensor.
  CVPixelBufferRef pb2 = make_bgra_pixelbuffer(8, 8, 10, 220, 130);
  ASSERT_NE(pb2, nullptr);
  ASSERT_EQ(
      process_pixelbuffer_into(processor, pb2, Orientation::UP, *out),
      Error::Ok);
  auto ref2 = process_pixelbuffer(processor, pb2);
  CVPixelBufferRelease(pb2);
  ASSERT_TRUE(ref2.ok());
  expect_tensors_near(out, ref2.get());

  // Same backing storage reused across both frames (no per-call allocation).
  EXPECT_EQ(out->const_data_ptr<float>(), storage);
}

// process_pixelbuffer_into requires a contiguous Float [1, 3, target_h,
// target_w] output; a mismatched tensor must be rejected rather than corrupt
// memory. Mirrors ProcessIntoValidationTest in image_processor_test.cpp.
TEST(ApplePixelBufferIntoTest, RejectsMalformedOutputTensor) {
  CVPixelBufferRef pb = make_bgra_pixelbuffer(8, 8, 200, 100, 50);
  ASSERT_NE(pb, nullptr);
  ImageProcessor processor(make_config(4, 4));

  // Wrong spatial size (target is 4x4).
  auto wrong_size =
      make_tensor_ptr({1, 3, 8, 8}, std::vector<float>(3 * 8 * 8));
  EXPECT_EQ(
      process_pixelbuffer_into(processor, pb, Orientation::UP, *wrong_size),
      Error::InvalidArgument);

  // Wrong rank.
  auto wrong_rank = make_tensor_ptr({3, 4, 4}, std::vector<float>(3 * 4 * 4));
  EXPECT_EQ(
      process_pixelbuffer_into(processor, pb, Orientation::UP, *wrong_rank),
      Error::InvalidArgument);

  // Wrong dtype (Int, not Float).
  auto wrong_dtype =
      make_tensor_ptr({1, 3, 4, 4}, std::vector<int32_t>(3 * 4 * 4));
  EXPECT_EQ(
      process_pixelbuffer_into(processor, pb, Orientation::UP, *wrong_dtype),
      Error::InvalidArgument);

  CVPixelBufferRelease(pb);
}

#endif // __APPLE__
