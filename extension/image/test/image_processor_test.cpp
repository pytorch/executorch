/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <executorch/extension/image/image_processor.h>

#include <cmath>
#include <cstring>
#include <thread>
#include <vector>

#include <executorch/extension/tensor/tensor_ptr.h>
#include <executorch/runtime/platform/platform.h>
#include <gtest/gtest.h>

using namespace executorch::extension::image;
using executorch::extension::make_tensor_ptr;
using executorch::runtime::Error;

// Initialize PAL before running tests
class ImageProcessorTestEnvironment : public ::testing::Environment {
 public:
  void SetUp() override {
    et_pal_init();
  }
};

const ::testing::Environment* const image_processor_test_env =
    ::testing::AddGlobalTestEnvironment(new ImageProcessorTestEnvironment);

// --- Test helpers ---

namespace {

std::vector<uint8_t>
make_solid_bgra(int32_t w, int32_t h, uint8_t r, uint8_t g, uint8_t b) {
  std::vector<uint8_t> img(w * h * 4);
  for (int32_t i = 0; i < w * h; ++i) {
    img[i * 4] = b;
    img[i * 4 + 1] = g;
    img[i * 4 + 2] = r;
    img[i * 4 + 3] = 255;
  }
  return img;
}

// Four solid quadrants with fully distinct colors: top-left red, top-right
// green, bottom-left blue, bottom-right yellow. Every quadrant and every
// channel differs, so any spatial error (ROI region, resize flip/transpose,
// letterbox placement) or channel error (BGRA/RGBA swizzle) changes the output
// detectably. Width and height must be even.
std::vector<uint8_t> make_quadrant(int32_t w, int32_t h, ColorFormat format) {
  struct Rgb {
    uint8_t r, g, b;
  };
  const Rgb tl{255, 0, 0}, tr{0, 255, 0}, bl{0, 0, 255}, br{255, 255, 0};
  std::vector<uint8_t> img(static_cast<size_t>(w) * h * 4);
  for (int32_t y = 0; y < h; ++y) {
    for (int32_t x = 0; x < w; ++x) {
      const Rgb c = (y < h / 2) ? (x < w / 2 ? tl : tr) : (x < w / 2 ? bl : br);
      uint8_t* px = img.data() + (static_cast<size_t>(y) * w + x) * 4;
      if (format == ColorFormat::RGBA) {
        px[0] = c.r;
        px[1] = c.g;
        px[2] = c.b;
      } else {
        px[0] = c.b;
        px[1] = c.g;
        px[2] = c.r;
      }
      px[3] = 255;
    }
  }
  return img;
}

// Distinctive fill for the inter-row padding of a strided buffer. A pipeline
// that respects stride never reads it; a stage that assumes tight packing reads
// this value instead of real pixels, making its result diverge from the
// tight-stride result that the stride tests compare against.
constexpr uint8_t kStridePoison = 0xAB;

// Re-lay a tightly packed 4-byte-per-pixel image at a wider row stride, filling
// the extra bytes with kStridePoison.
std::vector<uint8_t> with_stride(
    const std::vector<uint8_t>& tight,
    int32_t w,
    int32_t h,
    int32_t pad_bytes) {
  const int32_t stride = w * 4 + pad_bytes;
  std::vector<uint8_t> out(static_cast<size_t>(stride) * h, kStridePoison);
  for (int32_t y = 0; y < h; ++y) {
    std::memcpy(
        out.data() + static_cast<size_t>(y) * stride,
        tight.data() + static_cast<size_t>(y) * w * 4,
        static_cast<size_t>(w) * 4);
  }
  return out;
}

ImageProcessorConfig make_config(int32_t w, int32_t h) {
  ImageProcessorConfig config;
  config.target_width = w;
  config.target_height = h;
  return config;
}

// Read channel `c` at (row, col) from a contiguous [1, C, H, W] CHW tensor.
float chw(
    const float* data,
    int32_t H,
    int32_t W,
    int32_t c,
    int32_t row,
    int32_t col) {
  return data[(static_cast<size_t>(c) * H + row) * W + col];
}

// Assert the R, G, B planes at (row, col) match expected channel values. The
// tolerance absorbs resampler differences between backends while staying far
// below the ~1.0 gap a wrong region, flip, or channel swap would produce.
void expect_rgb(
    const float* data,
    int32_t H,
    int32_t W,
    int32_t row,
    int32_t col,
    float r,
    float g,
    float b) {
  constexpr float kEps = 0.05f;
  EXPECT_NEAR(chw(data, H, W, 0, row, col), r, kEps)
      << "R at " << row << "," << col;
  EXPECT_NEAR(chw(data, H, W, 1, row, col), g, kEps)
      << "G at " << row << "," << col;
  EXPECT_NEAR(chw(data, H, W, 2, row, col), b, kEps)
      << "B at " << row << "," << col;
}

// Compare two CHW float buffers element-wise. Pass eps == 0 for bit-exact
// equality, used when two code paths (e.g. tight vs strided input, or the
// allocating vs caller-owned-tensor entry points) must produce identical
// output; pass a small eps when only the decoded color must agree.
void expect_tensor_near(
    const float* a,
    const float* b,
    size_t count,
    float eps,
    const char* msg) {
  for (size_t i = 0; i < count; ++i) {
    EXPECT_NEAR(a[i], b[i], eps) << msg << " at " << i;
  }
}

// Semi-planar YUV image with a solid luma and chroma. `cb`/`cr` are the logical
// chroma; the interleave order follows `format` (NV12 stores Cb,Cr; NV21 stores
// Cr,Cb), so the same cb/cr decodes to the same color in either format. The UV
// plane is tightly packed at a row stride of `width` bytes.
struct YuvImage {
  std::vector<uint8_t> y;
  std::vector<uint8_t> uv;
};

YuvImage make_yuv(
    int32_t w,
    int32_t h,
    uint8_t y_val,
    uint8_t cb,
    uint8_t cr,
    YUVFormat format) {
  YuvImage img;
  img.y.assign(static_cast<size_t>(w) * h, y_val);
  img.uv.resize(static_cast<size_t>(w / 2) * (h / 2) * 2);
  for (size_t pair = 0; pair < img.uv.size() / 2; ++pair) {
    if (format == YUVFormat::NV12) {
      img.uv[pair * 2] = cb;
      img.uv[pair * 2 + 1] = cr;
    } else {
      img.uv[pair * 2] = cr;
      img.uv[pair * 2 + 1] = cb;
    }
  }
  return img;
}

} // namespace

// Backend fixture: runs each pixel-processing test under both backend-selection
// policies. kGpuAlways uses the GPU where a platform backend provides one;
// kGpuNever forces the CPU path. The selected backend must satisfy the same
// invariants, so every TEST_P body is written to be backend-agnostic and
// tolerance-based (resamplers can differ slightly across backends).
class ProcessTest : public ::testing::TestWithParam<int64_t> {
 protected:
  ImageProcessorConfig cfg(int32_t w, int32_t h) {
    auto c = make_config(w, h);
    c.gpu_min_input_pixels = GetParam();
    return c;
  }
};

INSTANTIATE_TEST_SUITE_P(
    Backend,
    ProcessTest,
    ::testing::Values(
        ImageProcessorConfig::kGpuAlways,
        ImageProcessorConfig::kGpuNever),
    [](const ::testing::TestParamInfo<int64_t>& info) {
      return info.param == ImageProcessorConfig::kGpuAlways ? "Gpu" : "Cpu";
    });

// --- Output shape ---

TEST(ShapeTest, Stretch) {
  auto config = make_config(224, 224);
  config.resize_mode = ResizeMode::STRETCH;
  ImageProcessor p(config);
  EXPECT_EQ(
      p.compute_output_shape(640, 480), (std::vector<int32_t>{1, 3, 224, 224}));
}

TEST(ShapeTest, Letterbox) {
  auto config = make_config(224, 224);
  config.resize_mode = ResizeMode::LETTERBOX;
  ImageProcessor p(config);
  // Output shape is always target dims; padding is filled internally.
  EXPECT_EQ(
      p.compute_output_shape(640, 480), (std::vector<int32_t>{1, 3, 224, 224}));
}

// The output is always the target size: an ROI selects which content is sampled
// but never changes the reported shape. Exercises the non-default roi path.
TEST(ShapeTest, RoiDoesNotChangeOutputShape) {
  auto config = make_config(224, 224);
  config.resize_mode = ResizeMode::LETTERBOX;
  ImageProcessor p(config);
  const NormalizedRect roi{0.25f, 0.0f, 0.5f, 1.0f};
  EXPECT_EQ(
      p.compute_output_shape(640, 480, Orientation::UP, roi),
      (std::vector<int32_t>{1, 3, 224, 224}));
}

// A non-square target surfaces any row/col (width/height) transposition, both
// in the reported shape and the produced tensor.
TEST_P(ProcessTest, ShapeMatchesProcessOutput) {
  auto bgra = make_solid_bgra(8, 6, 10, 20, 30);
  auto config = cfg(/*w=*/5, /*h=*/3);
  config.resize_mode = ResizeMode::LETTERBOX;
  ImageProcessor p(config);
  auto shape = p.compute_output_shape(8, 6);
  auto result = p.process(bgra.data(), 8, 6, 8 * 4, ColorFormat::BGRA);
  ASSERT_TRUE(result.ok());
  const auto& out = result.get();
  ASSERT_EQ(shape, (std::vector<int32_t>{1, 3, 3, 5}));
  EXPECT_EQ(out->size(0), shape[0]);
  EXPECT_EQ(out->size(1), shape[1]);
  EXPECT_EQ(out->size(2), shape[2]);
  EXPECT_EQ(out->size(3), shape[3]);
}

// A target whose width and height differ must place each quadrant in the
// matching output cell; a width/height swap would scramble the layout. The
// target keeps width identical and halves height so the resampled corners stay
// inside their quadrants.
TEST_P(ProcessTest, NonSquareTargetPreservesLayout) {
  auto img = make_quadrant(8, 8, ColorFormat::BGRA);
  ImageProcessor p(cfg(/*w=*/8, /*h=*/4));
  auto result = p.process(img.data(), 8, 8, 8 * 4, ColorFormat::BGRA);
  ASSERT_TRUE(result.ok());
  const auto& out = result.get();
  EXPECT_EQ(out->size(2), 4); // height
  EXPECT_EQ(out->size(3), 8); // width
  const float* d = out->const_data_ptr<float>();
  expect_rgb(d, 4, 8, 0, 0, 1, 0, 0); // top-left red
  expect_rgb(d, 4, 8, 0, 7, 0, 1, 0); // top-right green
  expect_rgb(d, 4, 8, 3, 0, 0, 0, 1); // bottom-left blue
  expect_rgb(d, 4, 8, 3, 7, 1, 1, 0); // bottom-right yellow
}

// --- Letterbox padding ---

TEST(LetterboxPaddingTest, CenterSquareTarget) {
  auto config = make_config(224, 224);
  config.resize_mode = ResizeMode::LETTERBOX;
  config.letterbox_anchor = LetterboxAnchor::CENTER;
  ImageProcessor p(config);
  // 640x480 → scale = 224/640 = 0.35; resized 224x168; vertical pad per side
  // = (224 - 168) / 2 = 28, no horizontal pad.
  EXPECT_EQ(
      p.compute_letterbox_padding(640, 480),
      (std::pair<int32_t, int32_t>{0, 28}));
}

TEST(LetterboxPaddingTest, StretchHasNoPadding) {
  auto config = make_config(224, 224);
  config.resize_mode = ResizeMode::STRETCH;
  ImageProcessor p(config);
  EXPECT_EQ(
      p.compute_letterbox_padding(640, 480),
      (std::pair<int32_t, int32_t>{0, 0}));
}

TEST(LetterboxPaddingTest, TopLeftAnchorHasNoPadding) {
  auto config = make_config(224, 224);
  config.resize_mode = ResizeMode::LETTERBOX;
  config.letterbox_anchor = LetterboxAnchor::TOP_LEFT;
  ImageProcessor p(config);
  EXPECT_EQ(
      p.compute_letterbox_padding(640, 480),
      (std::pair<int32_t, int32_t>{0, 0}));
}

// The reported padding must match where content actually begins in the output,
// so callers can invert the geometry.
TEST_P(ProcessTest, LetterboxPaddingMatchesActualPlacement) {
  auto bgra = make_solid_bgra(8, 4, 100, 150, 200); // wide -> vertical padding
  auto config = cfg(4, 4);
  config.resize_mode = ResizeMode::LETTERBOX;
  config.pad_value = 0.0f;
  ImageProcessor p(config);
  const auto pad = p.compute_letterbox_padding(8, 4);
  ASSERT_EQ(pad.first, 0);
  ASSERT_GT(pad.second, 0);
  auto result = p.process(bgra.data(), 8, 4, 8 * 4, ColorFormat::BGRA);
  ASSERT_TRUE(result.ok());
  const float* d = result.get()->const_data_ptr<float>();
  // The row above the reported pad is padding; the first content row is at it.
  EXPECT_FLOAT_EQ(chw(d, 4, 4, 0, pad.second - 1, 0), 0.0f);
  EXPECT_NEAR(chw(d, 4, 4, 0, pad.second, 0), 100.0f / 255.0f, 0.02f);
}

// Letterbox fit is computed on the ROI'd region, so cropping to a square inside
// a wide image removes the padding the full image would need.
TEST(LetterboxPaddingTest, FollowsRoiAspect) {
  auto config = make_config(4, 4);
  config.resize_mode = ResizeMode::LETTERBOX;
  ImageProcessor p(config);
  EXPECT_GT(p.compute_letterbox_padding(8, 4).second, 0); // wide full image
  const NormalizedRect square_roi{0.0f, 0.0f, 0.5f, 1.0f}; // left 4x4 -> square
  EXPECT_EQ(
      p.compute_letterbox_padding(8, 4, square_roi),
      (std::pair<int32_t, int32_t>{0, 0}));
}

// --- Color channels and resize layout ---

// Downscaling the quadrant fixture to 4x4 must place each quadrant in its
// matching output cell with each channel in the correct plane. Catches resize
// flips/transposes and BGRA/RGBA channel swaps.
TEST_P(ProcessTest, PreservesQuadrantLayout) {
  for (ColorFormat fmt : {ColorFormat::BGRA, ColorFormat::RGBA}) {
    ImageProcessor p(cfg(4, 4));
    auto img = make_quadrant(8, 8, fmt);
    auto result = p.process(img.data(), 8, 8, 8 * 4, fmt);
    ASSERT_TRUE(result.ok());
    const float* d = result.get()->const_data_ptr<float>();
    // Corner cells sample a quadrant interior, away from the resampled edges.
    expect_rgb(d, 4, 4, 0, 0, 1, 0, 0); // top-left red
    expect_rgb(d, 4, 4, 0, 3, 0, 1, 0); // top-right green
    expect_rgb(d, 4, 4, 3, 0, 0, 0, 1); // bottom-left blue
    expect_rgb(d, 4, 4, 3, 3, 1, 1, 0); // bottom-right yellow
  }
}

// --- Normalization ---

TEST_P(ProcessTest, NormalizationZeroToOne) {
  auto bgra = make_solid_bgra(2, 2, 100, 150, 200);
  auto config = cfg(2, 2);
  config.normalization = Normalization::zeroToOne();
  ImageProcessor p(config);
  auto result = p.process(bgra.data(), 2, 2, 2 * 4, ColorFormat::BGRA);
  ASSERT_TRUE(result.ok());
  const float* data = result.get()->const_data_ptr<float>();
  const float kEps = 1e-5f;
  EXPECT_NEAR(data[0], 100.0f / 255.0f, kEps); // R
  EXPECT_NEAR(data[4], 150.0f / 255.0f, kEps); // G
  EXPECT_NEAR(data[8], 200.0f / 255.0f, kEps); // B
}

TEST_P(ProcessTest, NormalizationImageNet) {
  auto bgra = make_solid_bgra(2, 2, 128, 128, 128);
  auto config = cfg(2, 2);
  config.normalization = Normalization::imagenet();
  ImageProcessor p(config);
  auto result = p.process(bgra.data(), 2, 2, 2 * 4, ColorFormat::BGRA);
  ASSERT_TRUE(result.ok());
  const float* data = result.get()->const_data_ptr<float>();
  const float kEps = 1e-3f;
  // (128/255 - 0.485) / 0.229 = 0.0274
  EXPECT_NEAR(data[0], (128.0f / 255.0f - 0.485f) / 0.229f, kEps);
  EXPECT_NEAR(data[4], (128.0f / 255.0f - 0.456f) / 0.224f, kEps);
  EXPECT_NEAR(data[8], (128.0f / 255.0f - 0.406f) / 0.225f, kEps);
}

// --- Resize modes ---

TEST_P(ProcessTest, LetterboxTallInputPadsHorizontally) {
  // Tall source → letterbox should pad left and right (anchor=CENTER), the
  // mirror of the wide case below.
  auto bgra = make_solid_bgra(4, 8, 100, 150, 200);
  auto config = cfg(4, 4);
  config.resize_mode = ResizeMode::LETTERBOX;
  config.letterbox_anchor = LetterboxAnchor::CENTER;
  config.pad_value = 0.0f;
  ImageProcessor p(config);
  auto result = p.process(bgra.data(), 4, 8, 4 * 4, ColorFormat::BGRA);
  ASSERT_TRUE(result.ok());
  const float* d = result.get()->const_data_ptr<float>();
  // Source resizes to 2x4 → columns 1..2 hold content, columns 0 and 3 are pad.
  EXPECT_FLOAT_EQ(chw(d, 4, 4, 0, 0, 0), 0.0f); // left pad
  EXPECT_NEAR(chw(d, 4, 4, 0, 0, 1), 100.0f / 255.0f, 0.02f); // content
  EXPECT_FLOAT_EQ(chw(d, 4, 4, 0, 0, 3), 0.0f); // right pad
}

TEST_P(ProcessTest, LetterboxCenterPaddingHorizontal) {
  // Wide source → letterbox should pad top and bottom (anchor=CENTER).
  auto bgra = make_solid_bgra(8, 4, 100, 150, 200);
  auto config = cfg(4, 4);
  config.resize_mode = ResizeMode::LETTERBOX;
  config.letterbox_anchor = LetterboxAnchor::CENTER;
  config.pad_value = 0.0f;
  ImageProcessor p(config);
  auto result = p.process(bgra.data(), 8, 4, 8 * 4, ColorFormat::BGRA);
  ASSERT_TRUE(result.ok());
  const float* data = result.get()->const_data_ptr<float>();
  // Layout: 1×3×4×4. resize_w=4, resize_h=2 → padded with 1 row top + 1 row
  // bottom.
  // Top row of R plane should be pad_value (0.0).
  EXPECT_FLOAT_EQ(data[0 * 4 + 0], 0.0f);
  // Center row should have the actual color.
  const float kEps = 0.02f;
  EXPECT_NEAR(data[1 * 4 + 0], 100.0f / 255.0f, kEps);
  // Bottom row should be padded.
  EXPECT_FLOAT_EQ(data[3 * 4 + 0], 0.0f);
}

TEST_P(ProcessTest, LetterboxTopLeftAnchor) {
  // Wide source → with TOP_LEFT anchor, content goes to the top.
  auto bgra = make_solid_bgra(8, 4, 100, 150, 200);
  auto config = cfg(4, 4);
  config.resize_mode = ResizeMode::LETTERBOX;
  config.letterbox_anchor = LetterboxAnchor::TOP_LEFT;
  config.pad_value = 0.0f;
  ImageProcessor p(config);
  auto result = p.process(bgra.data(), 8, 4, 8 * 4, ColorFormat::BGRA);
  ASSERT_TRUE(result.ok());
  const float* data = result.get()->const_data_ptr<float>();
  // resize_w=4, resize_h=2 → content occupies rows 0..1, rows 2..3 are pad.
  const float kEps = 0.02f;
  EXPECT_NEAR(data[0 * 4 + 0], 100.0f / 255.0f, kEps);
  EXPECT_NEAR(data[1 * 4 + 0], 100.0f / 255.0f, kEps);
  EXPECT_FLOAT_EQ(data[2 * 4 + 0], 0.0f);
  EXPECT_FLOAT_EQ(data[3 * 4 + 0], 0.0f);
}

TEST_P(ProcessTest, LetterboxPadValue) {
  // pad_value should fill the unused area.
  auto bgra = make_solid_bgra(8, 4, 100, 150, 200);
  auto config = cfg(4, 4);
  config.resize_mode = ResizeMode::LETTERBOX;
  config.pad_value = 0.5f;
  ImageProcessor p(config);
  auto result = p.process(bgra.data(), 8, 4, 8 * 4, ColorFormat::BGRA);
  ASSERT_TRUE(result.ok());
  const float* data = result.get()->const_data_ptr<float>();
  EXPECT_FLOAT_EQ(data[0 * 4 + 0], 0.5f);
  EXPECT_FLOAT_EQ(data[3 * 4 + 0], 0.5f);
}

// Padding lives in output space: pad cells hold the raw pad_value while content
// is normalized, even under a non-identity normalization.
TEST_P(ProcessTest, LetterboxPadValueWithImagenet) {
  auto bgra = make_solid_bgra(8, 4, 255, 0, 0); // wide red -> vertical padding
  auto config = cfg(4, 4);
  config.resize_mode = ResizeMode::LETTERBOX;
  config.pad_value = 0.5f;
  config.normalization = Normalization::imagenet();
  ImageProcessor p(config);
  auto result = p.process(bgra.data(), 8, 4, 8 * 4, ColorFormat::BGRA);
  ASSERT_TRUE(result.ok());
  const float* d = result.get()->const_data_ptr<float>();
  EXPECT_FLOAT_EQ(chw(d, 4, 4, 0, 0, 0), 0.5f); // pad: raw value
  EXPECT_NEAR(
      chw(d, 4, 4, 0, 1, 0), (1.0f - 0.485f) / 0.229f, 1e-2f); // content
}

// --- ROI ---

// An ROI crops before resize, so the output must contain only the selected
// region. Distinct quadrants make a wrong region or a transposed x/y offset
// visible. Corner cells sample a region interior, away from resampled edges.
TEST_P(ProcessTest, RoiSelectsRegion) {
  auto img = make_quadrant(8, 8, ColorFormat::BGRA);
  ImageProcessor p(cfg(4, 4));

  // Right half: top-right (green) over bottom-right (yellow).
  auto right = p.process(
      img.data(),
      8,
      8,
      8 * 4,
      ColorFormat::BGRA,
      Orientation::UP,
      {0.5f, 0.0f, 0.5f, 1.0f});
  ASSERT_TRUE(right.ok());
  expect_rgb(right.get()->const_data_ptr<float>(), 4, 4, 0, 0, 0, 1, 0);
  expect_rgb(right.get()->const_data_ptr<float>(), 4, 4, 3, 0, 1, 1, 0);

  // Bottom half: bottom-left (blue) beside bottom-right (yellow).
  auto bottom = p.process(
      img.data(),
      8,
      8,
      8 * 4,
      ColorFormat::BGRA,
      Orientation::UP,
      {0.0f, 0.5f, 1.0f, 0.5f});
  ASSERT_TRUE(bottom.ok());
  expect_rgb(bottom.get()->const_data_ptr<float>(), 4, 4, 0, 0, 0, 0, 1);
  expect_rgb(bottom.get()->const_data_ptr<float>(), 4, 4, 0, 3, 1, 1, 0);

  // Bottom-right quarter: only yellow.
  auto corner = p.process(
      img.data(),
      8,
      8,
      8 * 4,
      ColorFormat::BGRA,
      Orientation::UP,
      {0.5f, 0.5f, 0.5f, 0.5f});
  ASSERT_TRUE(corner.ok());
  expect_rgb(corner.get()->const_data_ptr<float>(), 4, 4, 0, 0, 1, 1, 0);
}

// A sub-pixel ROI truncates below 1px in each dimension. The crop must clamp to
// at least one pixel rather than produce a zero-size resize, so the output
// keeps the target shape and contains no NaN.
TEST_P(ProcessTest, TinyRoiClampsToValidOutput) {
  auto config = cfg(4, 4);
  config.resize_mode = ResizeMode::LETTERBOX;
  ImageProcessor p(config);
  auto img = make_quadrant(8, 8, ColorFormat::BGRA);
  const NormalizedRect tiny{0.5f, 0.5f, 0.01f, 0.01f};
  auto r = p.process(
      img.data(), 8, 8, 8 * 4, ColorFormat::BGRA, Orientation::UP, tiny);
  ASSERT_TRUE(r.ok());
  const auto& out = r.get();
  EXPECT_EQ(out->size(2), 4);
  EXPECT_EQ(out->size(3), 4);
  const float* d = out->const_data_ptr<float>();
  for (int64_t i = 0; i < out->numel(); ++i) {
    EXPECT_FALSE(std::isnan(d[i])) << "NaN at " << i;
  }
}

// --- Stride ---

// A wider-than-tight row stride must produce the same output as tight packing.
// The padding is poisoned, so a stage that ignores stride reads poison and its
// result diverges from the tight run.
TEST_P(ProcessTest, StridedInputMatchesTight) {
  ImageProcessor p(cfg(2, 2));
  auto tight = make_quadrant(8, 8, ColorFormat::BGRA);
  auto padded = with_stride(tight, 8, 8, /*pad_bytes=*/11);

  auto a = p.process(tight.data(), 8, 8, 8 * 4, ColorFormat::BGRA);
  auto b = p.process(padded.data(), 8, 8, 8 * 4 + 11, ColorFormat::BGRA);
  ASSERT_TRUE(a.ok());
  ASSERT_TRUE(b.ok());
  expect_tensor_near(
      a.get()->const_data_ptr<float>(),
      b.get()->const_data_ptr<float>(),
      static_cast<size_t>(3) * 2 * 2,
      0.0f,
      "stride mismatch");
}

// --- Output tensor reuse ---

// process_into writes into a caller-owned tensor reused across frames; a later
// call must fully overwrite the previous result, including clearing letterbox
// padding back to pad_value.
TEST_P(ProcessTest, ProcessIntoReuseClearsPreviousResult) {
  ImageProcessor solid_proc(cfg(4, 4));
  auto solid = make_solid_bgra(4, 4, 200, 100, 50);
  auto out = solid_proc.process(solid.data(), 4, 4, 4 * 4, ColorFormat::BGRA);
  ASSERT_TRUE(out.ok());

  ImageProcessorConfig letterbox_cfg = cfg(4, 4);
  letterbox_cfg.resize_mode = ResizeMode::LETTERBOX;
  letterbox_cfg.pad_value = 0.0f;
  ImageProcessor letterbox_proc(letterbox_cfg);
  auto wide = make_solid_bgra(8, 4, 0, 0, 255); // wide -> top/bottom padding
  auto err = letterbox_proc.process_into(
      wide.data(),
      8,
      4,
      8 * 4,
      ColorFormat::BGRA,
      *out.get(),
      Orientation::UP,
      kFullImage);
  ASSERT_EQ(err, Error::Ok);

  const float* d = out.get()->const_data_ptr<float>();
  // Wide source resizes to 4x2, leaving rows 0 and 3 as padding.
  EXPECT_FLOAT_EQ(chw(d, 4, 4, 2, 0, 0), 0.0f); // pad, not stale 50/255
  EXPECT_NEAR(chw(d, 4, 4, 2, 1, 0), 1.0f, 0.02f); // content blue
}

// process() is documented as a thin allocating wrapper over process_into(), so
// both entry points must yield bit-identical output for the same input.
TEST_P(ProcessTest, ProcessIntoMatchesProcess) {
  auto bgra = make_solid_bgra(8, 6, 100, 150, 200);
  ImageProcessor p(cfg(4, 4));
  auto alloc = p.process(bgra.data(), 8, 6, 8 * 4, ColorFormat::BGRA);
  ASSERT_TRUE(alloc.ok());

  auto out = make_tensor_ptr({1, 3, 4, 4}, std::vector<float>(3 * 4 * 4));
  auto err = p.process_into(bgra.data(), 8, 6, 8 * 4, ColorFormat::BGRA, *out);
  ASSERT_EQ(err, Error::Ok);
  expect_tensor_near(
      alloc.get()->const_data_ptr<float>(),
      out->const_data_ptr<float>(),
      static_cast<size_t>(3) * 4 * 4,
      0.0f,
      "process vs process_into");
}

// --- Cross-stage integration ---

// Crop one quadrant, resize, then imagenet-normalize. A wrong stage order,
// coordinate space, or per-channel mismatch shifts the exact expected values.
TEST_P(ProcessTest, RoiResizeImagenetNormalize) {
  auto img = make_quadrant(8, 8, ColorFormat::BGRA);
  ImageProcessorConfig config = cfg(2, 2);
  config.normalization = Normalization::imagenet();
  ImageProcessor p(config);
  // Bottom-right quadrant is solid yellow (R=255, G=255, B=0).
  auto r = p.process(
      img.data(),
      8,
      8,
      8 * 4,
      ColorFormat::BGRA,
      Orientation::UP,
      {0.5f, 0.5f, 0.5f, 0.5f});
  ASSERT_TRUE(r.ok());
  const float* d = r.get()->const_data_ptr<float>();
  const float kEps = 1e-2f;
  EXPECT_NEAR(chw(d, 2, 2, 0, 0, 0), (1.0f - 0.485f) / 0.229f, kEps);
  EXPECT_NEAR(chw(d, 2, 2, 1, 0, 0), (1.0f - 0.456f) / 0.224f, kEps);
  EXPECT_NEAR(chw(d, 2, 2, 2, 0, 0), (0.0f - 0.406f) / 0.225f, kEps);
}

// --- YUV ---

// Padded Y and UV plane strides must produce the same result as tight planes.
// The padding is poisoned, so a stride-ignoring read diverges from the tight
// run.
TEST_P(ProcessTest, YuvStridedPlanesMatchTight) {
  const int32_t w = 8, h = 4;
  std::vector<uint8_t> y(w * h);
  for (int32_t i = 0; i < w * h; ++i) {
    y[i] = (i % w < w / 2) ? 200 : 60; // left bright, right dark
  }
  const int32_t uv_row = (w / 2) * 2;
  std::vector<uint8_t> uv(uv_row * (h / 2), 128);

  ImageProcessor p(cfg(4, 4));
  auto tight =
      p.process_yuv(y.data(), w, uv.data(), uv_row, w, h, YUVFormat::NV12);
  ASSERT_TRUE(tight.ok());

  const int32_t ys = w + 5, uvs = uv_row + 6;
  std::vector<uint8_t> yp(ys * h, kStridePoison);
  std::vector<uint8_t> uvp(uvs * (h / 2), kStridePoison);
  for (int32_t r = 0; r < h; ++r) {
    std::memcpy(yp.data() + r * ys, y.data() + r * w, w);
  }
  for (int32_t r = 0; r < h / 2; ++r) {
    std::memcpy(uvp.data() + r * uvs, uv.data() + r * uv_row, uv_row);
  }
  auto strided =
      p.process_yuv(yp.data(), ys, uvp.data(), uvs, w, h, YUVFormat::NV12);
  ASSERT_TRUE(strided.ok());

  expect_tensor_near(
      tight.get()->const_data_ptr<float>(),
      strided.get()->const_data_ptr<float>(),
      static_cast<size_t>(3) * 4 * 4,
      0.0f,
      "yuv stride mismatch");
}

TEST_P(ProcessTest, YuvNv21MatchesNv12ForNeutralChroma) {
  // For U=V=128, NV21 and NV12 should produce identical results since swapping
  // identical values has no effect.
  const int32_t w = 8, h = 6;
  auto nv12 = make_yuv(w, h, 128, 128, 128, YUVFormat::NV12);
  auto nv21 = make_yuv(w, h, 128, 128, 128, YUVFormat::NV21);
  ImageProcessor p(cfg(4, 4));
  auto r12 =
      p.process_yuv(nv12.y.data(), w, nv12.uv.data(), w, w, h, YUVFormat::NV12);
  auto r21 =
      p.process_yuv(nv21.y.data(), w, nv21.uv.data(), w, w, h, YUVFormat::NV21);
  ASSERT_TRUE(r12.ok());
  ASSERT_TRUE(r21.ok());
  expect_tensor_near(
      r12.get()->const_data_ptr<float>(),
      r21.get()->const_data_ptr<float>(),
      static_cast<size_t>(3) * 4 * 4,
      1e-5f,
      "neutral chroma NV12 vs NV21");
}

TEST_P(ProcessTest, YuvNv21MatchesNv12ForNonNeutralChroma) {
  // With non-neutral chroma the Cb<->Cr swap actually matters: a correct NV21
  // decode equals an NV12 decode of the SAME logical chroma. A no-op swap, or
  // the "decode as NV12 then swap R/B" shortcut, diverges here (BT.601 weights
  // Cr->R and Cb->B differently, and green mixes both). Neutral chroma cannot
  // catch that, so this is the test that guards the swap.
  const int32_t w = 8, h = 6;
  auto nv12 = make_yuv(w, h, 150, /*cb=*/100, /*cr=*/180, YUVFormat::NV12);
  auto nv21 = make_yuv(w, h, 150, /*cb=*/100, /*cr=*/180, YUVFormat::NV21);
  ImageProcessor p(cfg(4, 4));
  auto r12 =
      p.process_yuv(nv12.y.data(), w, nv12.uv.data(), w, w, h, YUVFormat::NV12);
  auto r21 =
      p.process_yuv(nv21.y.data(), w, nv21.uv.data(), w, w, h, YUVFormat::NV21);
  ASSERT_TRUE(r12.ok());
  ASSERT_TRUE(r21.ok());
  expect_tensor_near(
      r12.get()->const_data_ptr<float>(),
      r21.get()->const_data_ptr<float>(),
      static_cast<size_t>(3) * 4 * 4,
      0.02f,
      "non-neutral chroma NV12 vs NV21");
}

TEST_P(ProcessTest, YuvFullRangeVsVideoRange) {
  // Neutral chroma (U=V=128) makes R=G=B a function of luma alone, so only the
  // quantization range matters:
  //   full range:  channel = Y / 255
  //   video range: channel = clamp((Y - 16) / 219, 0, 1)
  // At Y=235 that is ~0.922 (full) vs 1.0 (video clamps), so decoding a
  // full-range frame as video range over-stretches it. Values are derived from
  // the BT.601 definition, not from the implementation.
  const int32_t w = 4, h = 4;
  auto img = make_yuv(w, h, 235, 128, 128, YUVFormat::NV12);
  ImageProcessor p(cfg(2, 2));

  auto full = p.process_yuv(
      img.y.data(),
      w,
      img.uv.data(),
      w,
      w,
      h,
      YUVFormat::NV12,
      Orientation::UP,
      kFullImage,
      YUVRange::FULL);
  auto video = p.process_yuv(
      img.y.data(),
      w,
      img.uv.data(),
      w,
      w,
      h,
      YUVFormat::NV12,
      Orientation::UP,
      kFullImage,
      YUVRange::VIDEO);
  ASSERT_TRUE(full.ok());
  ASSERT_TRUE(video.ok());

  const float* full_data = full.get()->const_data_ptr<float>();
  const float* video_data = video.get()->const_data_ptr<float>();

  // Full range maps Y=235 to ~0.922 on every channel.
  const float kExpectedFull = 235.0f / 255.0f;
  for (int c = 0; c < 3; ++c) {
    EXPECT_NEAR(full_data[c * 4], kExpectedFull, 0.02f) << "channel " << c;
  }
  // Video range over-stretches the same luma to the clamped maximum, so the two
  // ranges must visibly disagree (otherwise the range argument is a no-op).
  EXPECT_NEAR(video_data[0], 1.0f, 0.02f);
  EXPECT_GT(video_data[0] - full_data[0], 0.05f);
}

TEST_P(ProcessTest, YuvDefaultsToVideoRange) {
  // Y=235 neutral chroma decodes to ~1.0 under video range; the default range
  // must match an explicit VIDEO request.
  const int32_t w = 4, h = 4;
  auto img = make_yuv(w, h, 235, 128, 128, YUVFormat::NV12);
  ImageProcessor p(cfg(2, 2));

  auto def =
      p.process_yuv(img.y.data(), w, img.uv.data(), w, w, h, YUVFormat::NV12);
  auto video = p.process_yuv(
      img.y.data(),
      w,
      img.uv.data(),
      w,
      w,
      h,
      YUVFormat::NV12,
      Orientation::UP,
      kFullImage,
      YUVRange::VIDEO);
  ASSERT_TRUE(def.ok());
  ASSERT_TRUE(video.ok());
  expect_tensor_near(
      def.get()->const_data_ptr<float>(),
      video.get()->const_data_ptr<float>(),
      static_cast<size_t>(3) * 2 * 2,
      1e-5f,
      "default vs explicit video range");
}

// --- Thread safety ---

TEST(ThreadSafetyTest, ConcurrentProcessIsSafe) {
  // Different ImageProcessor instances are independent and may be used from
  // different threads concurrently.
  auto bgra = make_solid_bgra(64, 64, 100, 150, 200);
  std::vector<std::thread> threads;
  threads.reserve(4);
  for (int t = 0; t < 4; ++t) {
    threads.emplace_back([&]() {
      auto config = make_config(32, 32);
      ImageProcessor p(config);
      for (int i = 0; i < 8; ++i) {
        auto result = p.process(bgra.data(), 64, 64, 64 * 4, ColorFormat::BGRA);
        ASSERT_TRUE(result.ok());
      }
    });
  }
  for (auto& t : threads) {
    t.join();
  }
}

// --- Config ---

TEST(ConfigTest, ConfigRoundTrip) {
  ImageProcessorConfig in;
  in.target_width = 224;
  in.target_height = 224;
  in.resize_mode = ResizeMode::LETTERBOX;
  in.letterbox_anchor = LetterboxAnchor::TOP_LEFT;
  in.pad_value = 0.5f;
  in.normalization = Normalization::imagenet();
  in.gpu_min_input_pixels = ImageProcessorConfig::kGpuAlways;

  ImageProcessor p(in);
  const auto& out = p.config();
  EXPECT_EQ(out.target_width, 224);
  EXPECT_EQ(out.target_height, 224);
  EXPECT_EQ(out.resize_mode, ResizeMode::LETTERBOX);
  EXPECT_EQ(out.letterbox_anchor, LetterboxAnchor::TOP_LEFT);
  EXPECT_FLOAT_EQ(out.pad_value, 0.5f);
  EXPECT_FLOAT_EQ(out.normalization.mean[0], 0.485f);
  EXPECT_EQ(out.gpu_min_input_pixels, ImageProcessorConfig::kGpuAlways);
}

// --- Error handling ---

// Invalid configured target dimensions are rejected regardless of input.
TEST(ErrorTest, InvalidTargetDimensionsReturnError) {
  ImageProcessorConfig config;
  config.target_width = 0; // Invalid
  config.target_height = 4;
  ImageProcessor p(config);
  auto bgra = make_solid_bgra(8, 8, 100, 150, 200);
  auto result = p.process(bgra.data(), 8, 8, 32, ColorFormat::BGRA);
  EXPECT_FALSE(result.ok());
  EXPECT_EQ(result.error(), Error::InvalidArgument);
}

TEST(ErrorTest, ZeroStdDevReturnsError) {
  ImageProcessorConfig config;
  config.target_width = 4;
  config.target_height = 4;
  config.normalization = Normalization::zeroToOne();
  config.normalization.std_dev[1] = 0.0f; // Invalid: divide-by-zero channel.
  ImageProcessor p(config);
  auto bgra = make_solid_bgra(8, 8, 100, 150, 200);
  auto result = p.process(bgra.data(), 8, 8, 8 * 4, ColorFormat::BGRA);
  EXPECT_FALSE(result.ok());
  EXPECT_EQ(result.error(), Error::InvalidArgument);
}

// One invalid input argument per row; everything else is valid, so each row
// isolates a single rejection path of process().
struct ProcessErrorCase {
  const char* name;
  bool null_data;
  int32_t width;
  int32_t height;
  int32_t stride_bytes; // < 0 => use the tight stride width * 4
  NormalizedRect roi;
};

class ProcessErrorTest : public ::testing::TestWithParam<ProcessErrorCase> {};

TEST_P(ProcessErrorTest, RejectsInvalidInput) {
  const auto& c = GetParam();
  ImageProcessor p(make_config(4, 4));
  auto bgra = make_solid_bgra(8, 8, 100, 150, 200);
  const int32_t stride = c.stride_bytes < 0 ? 8 * 4 : c.stride_bytes;
  const uint8_t* data = c.null_data ? nullptr : bgra.data();
  auto result = p.process(
      data,
      c.width,
      c.height,
      stride,
      ColorFormat::BGRA,
      Orientation::UP,
      c.roi);
  EXPECT_FALSE(result.ok()) << c.name;
  EXPECT_EQ(result.error(), Error::InvalidArgument) << c.name;
}

INSTANTIATE_TEST_SUITE_P(
    BadInputs,
    ProcessErrorTest,
    ::testing::Values(
        ProcessErrorCase{"null_data", true, 8, 8, -1, kFullImage},
        ProcessErrorCase{"zero_width", false, 0, 8, -1, kFullImage},
        ProcessErrorCase{"zero_height", false, 8, 0, -1, kFullImage},
        ProcessErrorCase{"negative_width", false, -1, 8, -1, kFullImage},
        ProcessErrorCase{"negative_height", false, 8, -1, -1, kFullImage},
        // 16 bytes is too small for an 8px BGRA row (needs 32).
        ProcessErrorCase{"stride_too_small", false, 8, 8, 16, kFullImage},
        ProcessErrorCase{
            "roi_overflows_right",
            false,
            8,
            8,
            -1,
            NormalizedRect{0.5f, 0.0f, 0.6f, 1.0f}},
        ProcessErrorCase{
            "roi_zero_width",
            false,
            8,
            8,
            -1,
            NormalizedRect{0.0f, 0.0f, 0.0f, 1.0f}}),
    [](const ::testing::TestParamInfo<ProcessErrorCase>& i) {
      return i.param.name;
    });

// One invalid input argument per row for process_yuv().
struct YuvErrorCase {
  const char* name;
  bool null_y;
  bool null_uv;
  int32_t width;
  int32_t height;
  NormalizedRect roi;
  int32_t y_stride; // < 0 => tight (buffer width, 8)
  int32_t uv_stride; // < 0 => tight (buffer width, 8)
};

class YuvErrorTest : public ::testing::TestWithParam<YuvErrorCase> {};

TEST_P(YuvErrorTest, RejectsInvalidInput) {
  const auto& c = GetParam();
  ImageProcessor p(make_config(4, 4));
  std::vector<uint8_t> y(8 * 8, 128);
  std::vector<uint8_t> uv(8 * 8 / 2, 128);
  const uint8_t* yp = c.null_y ? nullptr : y.data();
  const uint8_t* uvp = c.null_uv ? nullptr : uv.data();
  const int32_t ys = c.y_stride < 0 ? 8 : c.y_stride;
  const int32_t uvs = c.uv_stride < 0 ? 8 : c.uv_stride;
  auto result = p.process_yuv(
      yp,
      ys,
      uvp,
      uvs,
      c.width,
      c.height,
      YUVFormat::NV12,
      Orientation::UP,
      c.roi);
  EXPECT_FALSE(result.ok()) << c.name;
  EXPECT_EQ(result.error(), Error::InvalidArgument) << c.name;
}

INSTANTIATE_TEST_SUITE_P(
    BadInputs,
    YuvErrorTest,
    ::testing::Values(
        YuvErrorCase{"null_y", true, false, 8, 8, kFullImage, -1, -1},
        YuvErrorCase{"null_uv", false, true, 8, 8, kFullImage, -1, -1},
        YuvErrorCase{"zero_width", false, false, 0, 8, kFullImage, -1, -1},
        YuvErrorCase{"zero_height", false, false, 8, 0, kFullImage, -1, -1},
        YuvErrorCase{"negative_width", false, false, -2, 8, kFullImage, -1, -1},
        YuvErrorCase{
            "negative_height",
            false,
            false,
            8,
            -2,
            kFullImage,
            -1,
            -1},
        // NV12/NV21 require even dimensions for 2x2 chroma subsampling.
        YuvErrorCase{"odd_width", false, false, 7, 8, kFullImage, -1, -1},
        YuvErrorCase{"odd_height", false, false, 8, 7, kFullImage, -1, -1},
        // Each Y/UV row needs at least `width` bytes.
        YuvErrorCase{
            "y_stride_too_small",
            false,
            false,
            8,
            8,
            kFullImage,
            4,
            -1},
        YuvErrorCase{
            "uv_stride_too_small",
            false,
            false,
            8,
            8,
            kFullImage,
            -1,
            4},
        YuvErrorCase{
            "roi_overflows_right",
            false,
            false,
            8,
            8,
            NormalizedRect{0.5f, 0.0f, 0.6f, 1.0f},
            -1,
            -1}),
    [](const ::testing::TestParamInfo<YuvErrorCase>& i) {
      return i.param.name;
    });

// process_into() requires a contiguous Float [1, 3, target_h, target_w] output;
// a mismatched tensor must be rejected rather than corrupt memory.
TEST(ProcessIntoValidationTest, RejectsMalformedOutputTensor) {
  ImageProcessor p(make_config(4, 4));
  auto bgra = make_solid_bgra(8, 8, 100, 150, 200);

  // Wrong spatial size (target is 4x4).
  auto wrong_size =
      make_tensor_ptr({1, 3, 8, 8}, std::vector<float>(3 * 8 * 8));
  EXPECT_EQ(
      p.process_into(bgra.data(), 8, 8, 32, ColorFormat::BGRA, *wrong_size),
      Error::InvalidArgument);

  // Wrong rank.
  auto wrong_rank = make_tensor_ptr({3, 4, 4}, std::vector<float>(3 * 4 * 4));
  EXPECT_EQ(
      p.process_into(bgra.data(), 8, 8, 32, ColorFormat::BGRA, *wrong_rank),
      Error::InvalidArgument);

  // Wrong dtype (Int, not Float).
  auto wrong_dtype =
      make_tensor_ptr({1, 3, 4, 4}, std::vector<int32_t>(3 * 4 * 4));
  EXPECT_EQ(
      p.process_into(bgra.data(), 8, 8, 32, ColorFormat::BGRA, *wrong_dtype),
      Error::InvalidArgument);

  // Non-contiguous: correct shape and dtype but a channels-last memory layout,
  // which the tightly-packed CHW write cannot target safely.
  auto non_contiguous = make_tensor_ptr<float>(
      {1, 3, 4, 4}, std::vector<float>(3 * 4 * 4), /*dim_order=*/{0, 2, 3, 1});
  EXPECT_EQ(
      p.process_into(bgra.data(), 8, 8, 32, ColorFormat::BGRA, *non_contiguous),
      Error::InvalidArgument);
}

// --- GPU path selection (pure predicates) ---

TEST(GpuSelectionTest, ShouldUseGpuThreshold) {
  ImageProcessorConfig config;
  config.gpu_min_input_pixels = 100;
  EXPECT_FALSE(should_use_gpu(config, 9, 10)); // 90 < 100
  EXPECT_TRUE(should_use_gpu(config, 10, 10)); // 100 >= 100
  EXPECT_TRUE(should_use_gpu(config, 20, 10)); // 200 >= 100
  EXPECT_FALSE(is_cpu_only(config));
}

TEST(GpuSelectionTest, AlwaysAndNeverSentinels) {
  ImageProcessorConfig always;
  always.gpu_min_input_pixels = ImageProcessorConfig::kGpuAlways;
  EXPECT_TRUE(should_use_gpu(always, 1, 1)); // even a 1px input uses GPU
  EXPECT_FALSE(is_cpu_only(always));

  ImageProcessorConfig never;
  never.gpu_min_input_pixels = ImageProcessorConfig::kGpuNever;
  EXPECT_FALSE(
      should_use_gpu(never, 100000, 100000)); // never crosses kGpuNever
  EXPECT_TRUE(is_cpu_only(never));
}

// --- Constructor tests ---

TEST(ConstructorTest, DefaultConstructor) {
  // Default constructor should create a valid processor
  ImageProcessor p;
  // Should have default config values
  const auto& config = p.config();
  EXPECT_GT(config.target_width, 0);
  EXPECT_GT(config.target_height, 0);
}

TEST(ConstructorTest, MoveConstructor) {
  ImageProcessor p1(make_config(4, 4));
  // Move construct p2 from p1
  ImageProcessor p2(std::move(p1));
  // p2 should be usable
  auto bgra = make_solid_bgra(8, 8, 100, 150, 200);
  auto result = p2.process(bgra.data(), 8, 8, 32, ColorFormat::BGRA);
  EXPECT_TRUE(result.ok());
}

TEST(ConstructorTest, MoveAssignment) {
  ImageProcessor p1(make_config(4, 4));
  ImageProcessor p2(make_config(8, 8));
  // Move assign p1 to p2
  p2 = std::move(p1);
  // p2 should now have p1's config (4x4)
  EXPECT_EQ(p2.config().target_width, 4);
  EXPECT_EQ(p2.config().target_height, 4);
  // p2 should be usable
  auto bgra = make_solid_bgra(8, 8, 100, 150, 200);
  auto result = p2.process(bgra.data(), 8, 8, 32, ColorFormat::BGRA);
  EXPECT_TRUE(result.ok());
}

// --- YUV ROI tests ---

TEST_P(ProcessTest, YuvNv12WithRoi) {
  auto config = cfg(4, 4);
  config.normalization = Normalization::zeroToOne();
  ImageProcessor processor(config);

  // Left half Y=76, right half Y=29 (neutral chroma), so the ROI selection is
  // visible as a luma difference in the output.
  const int32_t w = 8, h = 4;
  std::vector<uint8_t> y_plane(w * h);
  std::vector<uint8_t> uv_plane((w / 2) * (h / 2) * 2);
  for (int y = 0; y < h; ++y) {
    for (int x = 0; x < w; ++x) {
      y_plane[y * w + x] = (x < w / 2) ? 76 : 29;
    }
  }
  std::fill(uv_plane.begin(), uv_plane.end(), 128);

  // Process only right half (ROI: x=0.5, y=0, w=0.5, h=1.0)
  NormalizedRect right_half{0.5f, 0.0f, 0.5f, 1.0f};
  auto result = processor.process_yuv(
      y_plane.data(),
      w,
      uv_plane.data(),
      w,
      w,
      h,
      YUVFormat::NV12,
      Orientation::UP,
      right_half);
  ASSERT_TRUE(result.ok());

  auto& tensor = result.get();
  EXPECT_EQ(tensor->size(2), 4);
  EXPECT_EQ(tensor->size(3), 4);

  // Result should be from the right half (darker due to Y=29)
  const float* data = tensor->const_data_ptr<float>();
  const float r0 = data[0];
  // Y=29 with U=V=128 should give a darker value than Y=76
  EXPECT_LT(r0, 0.3f) << "Right half should be darker (Y=29)";
}

// process_yuv() is documented as a thin allocating wrapper over
// process_yuv_into(), so both entry points must yield bit-identical output.
// This is the only direct coverage of process_yuv_into().
TEST_P(ProcessTest, ProcessYuvIntoMatchesProcessYuv) {
  const int32_t w = 8, h = 6;
  auto img = make_yuv(w, h, 150, 100, 180, YUVFormat::NV12);
  ImageProcessor p(cfg(4, 4));
  auto alloc =
      p.process_yuv(img.y.data(), w, img.uv.data(), w, w, h, YUVFormat::NV12);
  ASSERT_TRUE(alloc.ok());

  auto out = make_tensor_ptr({1, 3, 4, 4}, std::vector<float>(3 * 4 * 4));
  auto err = p.process_yuv_into(
      img.y.data(), w, img.uv.data(), w, w, h, YUVFormat::NV12, *out);
  ASSERT_EQ(err, Error::Ok);
  expect_tensor_near(
      alloc.get()->const_data_ptr<float>(),
      out->const_data_ptr<float>(),
      static_cast<size_t>(3) * 4 * 4,
      0.0f,
      "process_yuv vs process_yuv_into");
}
