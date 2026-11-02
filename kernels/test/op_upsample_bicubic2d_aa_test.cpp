/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <executorch/kernels/test/FunctionHeaderWrapper.h> // Declares the operator
#include <executorch/kernels/test/TestUtil.h>
#include <executorch/runtime/core/exec_aten/exec_aten.h>
#include <executorch/runtime/core/exec_aten/testing_util/tensor_factory.h>
#include <executorch/runtime/core/exec_aten/testing_util/tensor_util.h>

#include <gtest/gtest.h>

#include <cmath>

using namespace ::testing;
using namespace executorch::aten;
using torch::executor::testing::TensorFactory;

namespace {

// Shape [1, 2, 6, 8]. Multiples of 1/4, so every floating type holds them.
const std::vector<float> kInput = {
    -2.75, 0.75,  -1.5,  2,     -0.25, -2.5,  1,     -1.25, 2.25,  0,     -2.25,
    1.25,  -1,    2.5,   0.25,  -2,    1.5,   -0.75, 2.75,  0.5,   -1.75, 1.75,
    -0.5,  -2.75, 0.75,  -1.5,  2,     -0.25, -2.5,  1,     -1.25, 2.25,  0,
    -2.25, 1.25,  -1,    2.5,   0.25,  -2,    1.5,   -0.75, 2.75,  0.5,   -1.75,
    1.75,  -0.5,  -2.75, 0.75,  -1.5,  2,     -0.25, -2.5,  1,     -1.25, 2.25,
    0,     -2.25, 1.25,  -1,    2.5,   0.25,  -2,    1.5,   -0.75, 2.75,  0.5,
    -1.75, 1.75,  -0.5,  -2.75, 0.75,  -1.5,  2,     -0.25, -2.5,  1,     -1.25,
    2.25,  0,     -2.25, 1.25,  -1,    2.5,   0.25,  -2,    1.5,   -0.75, 2.75,
    0.5,   -1.75, 1.75,  -0.5,  -2.75, 0.75,  -1.5,  2};

// Shape [1, 1, 3, 4].
const std::vector<float> kSmallInput =
    {-0.75, 0.5, 0, -0.5, 0.75, 0.25, -0.25, -0.75, 0.5, 0, -0.5, 0.75};

// The expected values below are what ATen's CPU kernel gives in float32.

// kInput resized to (3, 5).
const std::vector<float> kShrunk = {
    0.0906679854,   -0.475160062, 0.374634027,  0.143204734,  -0.922974527,
    0.203037277,    0.835856259,  -0.634037733, 0.384397358,  -0.519408643,
    -0.163436651,   0.299764723,  0.306681991,  -0.405743986, -0.157753274,
    -0.223817319,   0.0853600204, 0.205771729,  -0.521300793, 0.519143701,
    1.23301566,     -0.724639535, 0.290732175,  -0.220765993, -0.693045139,
    -0.00643681083, 0.718924642,  -1.07855761,  0.0471052192, 0.779319167};

// kInput resized to (3, 11) with align_corners.
const std::vector<float> kAlignCorners = {
    0.395212859, 0.270034224,  -0.177529454,   -0.820639431, 0.857328475,
    0.699784875, -1.11745334,  0.230914891,    0.775949657,  0.120987602,
    -1.70900011, 0.757739007,  -0.483611494,   -0.616503656, 1.71716607,
    0.306986153, -0.677421331, -0.310276628,   0.818704128,  -0.196790308,
    -1.37936914, 0.524807692,  -0.896195412,   0.633612096,  1.69014382,
    0.616090477, -1.3802079,   -0.500471592,   2.38540888,   0.339915723,
    -1.89413214, -2.42607069,  0.7296018,      -0.928531647, 0.535698175,
    1.00333369,  -1.07555234,  -0.00527923973, 0.776648641,  0.323259175,
    -1.72325718, -0.389089495, 1.5565505,      -0.577305675, 1.88765359,
    0.536147237, -0.666489065, -0.388005495,   0.688801527,  0.140282094,
    -1.49791741, 0.327211022,  0.468800664,    -0.297805637, -0.12769644,
    0.666050732, -0.949824452, -1.00370514,    2.31604958,   0.739088535,
    -1.57536066, -2.68413806,  0.481709063,    0.0198896192, -1.17022157,
    2.18622065};

// kInput resized to (6, 3) with scales 1.05 and 0.45: the height keeps its
// size, so it is copied whatever its scale.
const std::vector<float> kKeptHeight = {
    -1.0862571,     0.426353812,  -0.821281493,  0.500302792,  -0.473454088,
    0.912124515,    0.831802487,  0.638152242,   -0.020045612, 0.0818024501,
    -0.129340187,   -0.544154108, -0.832585812,  0.4716717,    0.18996644,
    1.00280011,     0.0817186236, -0.591454029,  0.252799958,  -0.848054767,
    0.30904001,     -0.586257041, 0.926353991,   -0.321281493, 1.00114834,
    -0.00857725739, -1.07128155,  0.250302762,   -0.723454058, 0.662124455,
    0.581802487,    0.370659858,  -0.0441541485, -0.168197602, -0.379340231,
    -0.794154167};

// kInput resized to (3, 8): the width keeps its size, so it is copied.
const std::vector<float> kKeptWidth = {
    -0.0784518421, 0.342050225,   -1.49895394,  1.59205019,    -0.730125546,
    0.171548143,   0.592050135,   -1.94665277,  1.32251918,    -1.32251906,
    1.93606877,    0.124999993,   -1.68606877,  1.57251906,    -0.87499994,
    -0.25,         -0.342050225,  0.0784519166, 0.980125546,   -1.34205019,
    1.74895406,    -0.0920502245, -2.34205031,  1.44665277,    -1.49895394,
    1.59205019,    -0.730125546,  0.171548143,  0.592050135,   -1.94665277,
    1.84205019,    -0.480125576,  1.93606877,   0.124999993,   -1.68606877,
    1.57251906,    -0.87499994,   -0.25,        0.37500003,    -1.43606877,
    0.980125546,   -1.34205019,   1.74895406,   -0.0920502245, -2.34205031,
    1.44665277,    -1.09205019,   1.99895406};

// Shape [1, 1, 2, 10]. The rows have opposite signs, so resizing to (1, 5)
// averages them into small values. A kernel that rounded its first pass to a
// half type would be off here by far more than the test tolerance.
const std::vector<float> kOppositeRows = {
    67,    66,    67,  63,  64.5,  63,    63.5, 67,  66,    66,
    -63.5, -64.5, -64, -62, -66.5, -63.5, -67,  -65, -66.5, -62};

// kOppositeRows resized to (1, 5).
const std::vector<float> kAveraged =
    {1.33681488, 0.797298431, -0.77734375, -0.417949677, 1.00313568};

// Shape [1, 1, 6, 8] of uint8 with sharp edges, so the width pass overshoots
// and its rows have to be clamped before the height pass, as ATen does.
const std::vector<float> kUint8Input = {
    38,  23,  25, 15,  225, 222, 220, 13,  247, 38,  223, 239,
    39,  0,   38, 251, 252, 250, 239, 251, 227, 254, 247, 15,
    249, 9,   17, 237, 255, 0,   252, 38,  26,  20,  4,   228,
    236, 244, 7,  14,  1,   7,   247, 2,   0,   247, 251, 9};

// kUint8Input resized to (2, 10). One fraction bit more or less in the
// fixed-point weights changes it.
const std::vector<float> kUint8Wide = {201, 115, 112, 163, 187, 170, 133,
                                       144, 150, 97,  109, 58,  68,  126,
                                       182, 176, 178, 173, 126, 13};

} // namespace

class OpUpsampleBicubic2dAAOutTest : public OperatorTest {
 protected:
  void SetUp() override {
    OperatorTest::SetUp();
    context_ = torch::executor::KernelRuntimeContext(nullptr, &temp_allocator_);
  }

  Tensor& op_upsample_bicubic2d_aa_out(
      const Tensor& input,
      const ArrayRef<int64_t> output_size,
      bool align_corners,
      const std::optional<double> scales_h,
      const std::optional<double> scales_w,
      Tensor& out) {
    // Like the runtime, frees the temporary memory between kernel calls.
    temp_allocator_.reset();
    return torch::executor::aten::_upsample_bicubic2d_aa_outf(
        context_, input, output_size, align_corners, scales_h, scales_w, out);
  }

  template <ScalarType DTYPE>
  void expect_resize(
      const std::vector<int32_t>& in_sizes,
      const std::vector<float>& in_data,
      const std::vector<int64_t>& output_size,
      bool align_corners,
      const std::optional<double> scales_h,
      const std::optional<double> scales_w,
      const std::vector<float>& expected,
      bool channels_last = false) {
    TensorFactory<DTYPE> tf;
    using CTYPE = typename TensorFactory<DTYPE>::ctype;
    const std::vector<int32_t> out_sizes = {
        in_sizes[0],
        in_sizes[1],
        static_cast<int32_t>(output_size[0]),
        static_cast<int32_t>(output_size[1])};
    Tensor in =
        tf.make(in_sizes, std::vector<CTYPE>(in_data.begin(), in_data.end()));
    Tensor want = tf.make(
        out_sizes, std::vector<CTYPE>(expected.begin(), expected.end()));
    Tensor out = tf.zeros(out_sizes);
    if (channels_last) {
      in = tf.channels_last_like(in);
      want = tf.channels_last_like(want);
      out = tf.zeros_channels_last(out_sizes);
    }
    op_upsample_bicubic2d_aa_out(
        in,
        ArrayRef<int64_t>(output_size.data(), output_size.size()),
        align_corners,
        scales_h,
        scales_w,
        out);
    // Half types are computed in float and rounded once, so they can be one
    // unit in the last place from the rounded float32 result. As a relative
    // tolerance, that unit is 2^-10 for half and 2^-7 for bfloat16 at a power
    // of two, so the check allows up to two units just below the next one.
    if constexpr (DTYPE == ScalarType::Half) {
      EXPECT_TENSOR_CLOSE_WITH_TOL(out, want, 1.0 / 1024, 1e-6);
    } else if constexpr (DTYPE == ScalarType::BFloat16) {
      EXPECT_TENSOR_CLOSE_WITH_TOL(out, want, 1.0 / 128, 1e-6);
    } else if constexpr (DTYPE == ScalarType::Byte) {
      EXPECT_TENSOR_EQ(out, want);
    } else {
      EXPECT_TENSOR_CLOSE_WITH_TOL(out, want, 1e-5, 1e-5);
    }
  }

 private:
  uint8_t temp_memory_[16 * 1024] = {};
  executorch::runtime::MemoryAllocator temp_allocator_{
      sizeof(temp_memory_),
      temp_memory_};
};

TEST_F(OpUpsampleBicubic2dAAOutTest, ShrinkMatchesATen) {
  // A second batch holding the negated input gives the negated result.
  std::vector<float> in = kInput;
  std::vector<float> expected = kShrunk;
  for (const float value : kInput) {
    in.push_back(-value);
  }
  for (const float value : kShrunk) {
    expected.push_back(-value);
  }
  expect_resize<ScalarType::Float>(
      {2, 2, 6, 8}, in, {3, 5}, false, std::nullopt, std::nullopt, expected);
}

TEST_F(OpUpsampleBicubic2dAAOutTest, AlignCornersMatchesATen) {
  expect_resize<ScalarType::Float>(
      {1, 2, 6, 8},
      kInput,
      {3, 11},
      true,
      std::nullopt,
      std::nullopt,
      kAlignCorners);
}

TEST_F(OpUpsampleBicubic2dAAOutTest, ScalesMatchATen) {
  // Like ATen, a scale that is not positive is ignored.
  expect_resize<ScalarType::Float>(
      {1, 2, 6, 8}, kInput, {3, 5}, false, -1.0, std::nan(""), kShrunk);
  // A scale this small puts the windows of the last two rows past the input,
  // so they read nothing and give 0. The width changes too, so this runs the
  // path that resizes both dims.
  expect_resize<ScalarType::Float>(
      {1, 1, 3, 4},
      kSmallInput,
      {5, 3},
      false,
      0.4,
      std::nullopt,
      {0.225569099,
       0.0334828235,
       -0.313413143,
       0.543530881,
       -0.355477989,
       0.389086157,
       0.421826512,
       -0.290672064,
       0.340768456,
       0,
       0,
       0,
       0,
       0,
       0});
}

TEST_F(OpUpsampleBicubic2dAAOutTest, CopiesDimsThatKeepTheirSize) {
  expect_resize<ScalarType::Float>(
      {1, 2, 6, 8}, kInput, {6, 8}, false, 1.05, 0.95, kInput);
}

TEST_F(OpUpsampleBicubic2dAAOutTest, ChannelsLastMatchesATen) {
  expect_resize<ScalarType::Float>(
      {1, 2, 6, 8},
      kInput,
      {3, 5},
      false,
      std::nullopt,
      std::nullopt,
      kShrunk,
      /*channels_last=*/true);
  expect_resize<ScalarType::Float>(
      {1, 2, 6, 8},
      kInput,
      {6, 3},
      false,
      1.05,
      0.45,
      kKeptHeight,
      /*channels_last=*/true);
  expect_resize<ScalarType::Float>(
      {1, 2, 6, 8},
      kInput,
      {3, 8},
      false,
      std::nullopt,
      0.95,
      kKeptWidth,
      /*channels_last=*/true);
}

// ATen's CPU kernel has no float16 or bfloat16 version, so these compare with
// its float32 results.
TEST_F(OpUpsampleBicubic2dAAOutTest, HalfTypesMatchFloat32ATen) {
  expect_resize<ScalarType::Half>(
      {1, 1, 2, 10},
      kOppositeRows,
      {1, 5},
      false,
      std::nullopt,
      std::nullopt,
      kAveraged);
  expect_resize<ScalarType::BFloat16>(
      {1, 1, 2, 10},
      kOppositeRows,
      {1, 5},
      false,
      std::nullopt,
      std::nullopt,
      kAveraged);
}

TEST_F(OpUpsampleBicubic2dAAOutTest, DoubleMatchesFloat64ATen) {
  TensorFactory<ScalarType::Double> tf;
  const Tensor in = tf.make(
      {1, 1, 3, 4},
      std::vector<double>(kSmallInput.begin(), kSmallInput.end()));
  Tensor out = tf.zeros({1, 1, 2, 3});
  const std::vector<int64_t> output_size = {2, 3};
  op_upsample_bicubic2d_aa_out(
      in,
      ArrayRef<int64_t>(output_size.data(), output_size.size()),
      false,
      std::nullopt,
      std::nullopt,
      out);
  // ATen's float64 result. A float32 computation is about 1e-7 away.
  const Tensor want = tf.make(
      {1, 1, 2, 3},
      {-0.02889796548038389,
       0.2150704092175421,
       -0.5353515689565878,
       0.5353515689565879,
       -0.2150704092175421,
       0.028897965480383748});
  EXPECT_TENSOR_CLOSE_WITH_TOL(out, want, 1e-12, 1e-12);
}

TEST_F(OpUpsampleBicubic2dAAOutTest, Uint8MatchesATenExactly) {
  expect_resize<ScalarType::Byte>(
      {1, 1, 6, 8},
      kUint8Input,
      {2, 10},
      false,
      std::nullopt,
      std::nullopt,
      kUint8Wide);
}

TEST_F(OpUpsampleBicubic2dAAOutTest, ResizesDynamicOutput) {
  TensorFactory<ScalarType::Float> tf;
  Tensor in = tf.make({1, 2, 6, 8}, kInput);
  Tensor out = tf.zeros(
      {1, 2, 6, 8}, torch::executor::TensorShapeDynamism::DYNAMIC_BOUND);
  const std::vector<int64_t> output_size = {3, 5};
  op_upsample_bicubic2d_aa_out(
      in,
      ArrayRef<int64_t>(output_size.data(), output_size.size()),
      false,
      std::nullopt,
      std::nullopt,
      out);
  EXPECT_TENSOR_CLOSE_WITH_TOL(out, tf.make({1, 2, 3, 5}, kShrunk), 1e-5, 1e-5);
}

TEST_F(OpUpsampleBicubic2dAAOutTest, RejectsIntegerInput) {
  TensorFactory<ScalarType::Int> tf;
  Tensor in = tf.zeros({1, 1, 4, 4});
  Tensor out = tf.zeros({1, 1, 2, 2});
  const std::vector<int64_t> output_size = {2, 2};
  ET_EXPECT_KERNEL_FAILURE(
      context_,
      op_upsample_bicubic2d_aa_out(
          in,
          ArrayRef<int64_t>(output_size.data(), output_size.size()),
          false,
          std::nullopt,
          std::nullopt,
          out));
}

TEST_F(OpUpsampleBicubic2dAAOutTest, FailsWithoutTempMemory) {
  const std::vector<int64_t> output_size = {6, 5};
  context_ = torch::executor::KernelRuntimeContext();
  TensorFactory<ScalarType::Float> tf;
  Tensor in = tf.make({1, 2, 6, 8}, kInput);
  Tensor out = tf.zeros({1, 2, 6, 5});
  ET_EXPECT_KERNEL_FAILURE(
      context_,
      op_upsample_bicubic2d_aa_out(
          in,
          ArrayRef<int64_t>(output_size.data(), output_size.size()),
          false,
          std::nullopt,
          std::nullopt,
          out));
  // uint8 computes its weights on its own path. A new context clears the
  // failure, so that this check can fail too.
  context_ = torch::executor::KernelRuntimeContext();
  TensorFactory<ScalarType::Byte> tb;
  Tensor in_u8 = tb.zeros({1, 2, 6, 8});
  Tensor out_u8 = tb.zeros({1, 2, 6, 5});
  ET_EXPECT_KERNEL_FAILURE(
      context_,
      op_upsample_bicubic2d_aa_out(
          in_u8,
          ArrayRef<int64_t>(output_size.data(), output_size.size()),
          false,
          std::nullopt,
          std::nullopt,
          out_u8));
}

TEST_F(OpUpsampleBicubic2dAAOutTest, FailsWhenTempMemoryRunsOut) {
  // The weights fit in the fixture's 16 KiB, but the rows between the passes
  // do not.
  TensorFactory<ScalarType::Float> tf;
  Tensor in = tf.zeros({1, 1, 3, 600});
  Tensor out = tf.zeros({1, 1, 2, 300});
  const std::vector<int64_t> output_size = {2, 300};
  ET_EXPECT_KERNEL_FAILURE(
      context_,
      op_upsample_bicubic2d_aa_out(
          in,
          ArrayRef<int64_t>(output_size.data(), output_size.size()),
          false,
          std::nullopt,
          std::nullopt,
          out));
}

TEST_F(OpUpsampleBicubic2dAAOutTest, RejectsEmptyInput) {
  TensorFactory<ScalarType::Float> tf;
  Tensor in = tf.zeros({1, 1, 0, 4});
  Tensor out = tf.zeros({1, 1, 2, 2});
  const std::vector<int64_t> output_size = {2, 2};
  ET_EXPECT_KERNEL_FAILURE(
      context_,
      op_upsample_bicubic2d_aa_out(
          in,
          ArrayRef<int64_t>(output_size.data(), output_size.size()),
          false,
          std::nullopt,
          std::nullopt,
          out));
}

TEST_F(OpUpsampleBicubic2dAAOutTest, RejectsScaleTooSmallForInput) {
  TensorFactory<ScalarType::Float> tf;
  // The width changes too, so computing its weights must not hide the
  // height's error.
  Tensor in = tf.zeros({1, 1, 64, 8});
  Tensor out = tf.zeros({1, 1, 2, 4});
  const std::vector<int64_t> output_size = {2, 4};
  ET_EXPECT_KERNEL_FAILURE(
      context_,
      op_upsample_bicubic2d_aa_out(
          in,
          ArrayRef<int64_t>(output_size.data(), output_size.size()),
          false,
          0.01,
          std::nullopt,
          out));
}

TEST_F(OpUpsampleBicubic2dAAOutTest, RejectsUint8WeightsTooLargeForFixedPoint) {
  // With this scale, the window of the middle output nearly sums to zero, so
  // its largest normalized weight leaves no fraction bit in 16-bit fixed
  // point, although every weight still fits in 16 bits.
  TensorFactory<ScalarType::Byte> tf;
  Tensor in = tf.make({1, 1, 1, 4}, {10, 20, 30, 40});
  Tensor out = tf.zeros({1, 1, 1, 5});
  const std::vector<int64_t> output_size = {1, 5};
  ET_EXPECT_KERNEL_FAILURE(
      context_,
      op_upsample_bicubic2d_aa_out(
          in,
          ArrayRef<int64_t>(output_size.data(), output_size.size()),
          false,
          std::nullopt,
          0.4638698,
          out));
}

TEST_F(OpUpsampleBicubic2dAAOutTest, RejectsUint8WeightsOutsideInt16) {
  // With this scale, the last output's window sums to a negative value, so
  // its most negative weight is larger than any positive one and does not fit
  // in 16 bits at the precision the positive ones allow.
  TensorFactory<ScalarType::Byte> tf;
  Tensor in = tf.full({1, 1, 1, 19}, 100);
  Tensor out = tf.zeros({1, 1, 1, 6});
  const std::vector<int64_t> output_size = {1, 6};
  ET_EXPECT_KERNEL_FAILURE(
      context_,
      op_upsample_bicubic2d_aa_out(
          in,
          ArrayRef<int64_t>(output_size.data(), output_size.size()),
          false,
          std::nullopt,
          0.25,
          out));
}

TEST_F(OpUpsampleBicubic2dAAOutTest, ResizesBothDimsLikeTwoOneDimResizes) {
  // Inputs this tall make the kernel keep only a few resized rows between its
  // passes. All 64 of them would not fit in the fixture's 16 KiB of temporary
  // memory.
  TensorFactory<ScalarType::Float> tf;
  std::vector<float> data(64 * 128);
  for (const auto i : c10::irange(data.size())) {
    data[i] = std::sin(0.37f * i);
  }
  const Tensor in = tf.make({1, 1, 64, 128}, data);
  const auto resize = [&](const Tensor& from, int64_t h, int64_t w) {
    Tensor to =
        tf.zeros({1, 1, static_cast<int32_t>(h), static_cast<int32_t>(w)});
    const std::vector<int64_t> output_size = {h, w};
    op_upsample_bicubic2d_aa_out(
        from,
        ArrayRef<int64_t>(output_size.data(), output_size.size()),
        false,
        std::nullopt,
        std::nullopt,
        to);
    return to;
  };
  const Tensor wide = resize(in, 64, 64);
  for (const int64_t height : {48, 80}) {
    EXPECT_TENSOR_CLOSE_WITH_TOL(
        resize(in, height, 64), resize(wide, height, 64), 1e-6, 1e-6);
  }
}
