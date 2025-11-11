/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <executorch/kernels/test/FunctionHeaderWrapper.h> // Declares the operator
#include <executorch/kernels/test/TestUtil.h>
#include <executorch/kernels/test/supported_features.h>
#include <executorch/runtime/core/exec_aten/exec_aten.h>
#include <executorch/runtime/core/exec_aten/testing_util/tensor_factory.h>
#include <executorch/runtime/core/exec_aten/testing_util/tensor_util.h>

#include <gtest/gtest.h>

using executorch::aten::ScalarType;
using executorch::aten::Tensor;
using std::optional;
using torch::executor::testing::SupportedFeatures;
using torch::executor::testing::TensorFactory;

#ifdef USE_ATEN_LIB
template <class T>
using OptionalArrayRef = std::optional<c10::ArrayRef<T>>;
#else
using executorch::aten::OptionalArrayRef;
#endif

class OpUpsampleBilinear2dTest : public OperatorTest {
 protected:
  Tensor& op_upsample_bilinear2d_vec_out(
      const Tensor& in,
      const OptionalArrayRef<int64_t>& output_size,
      bool align_corners,
      const OptionalArrayRef<double>& scale_factors,
      Tensor& out) {
    return torch::executor::aten::upsample_bilinear2d_outf(
        context_, in, output_size, align_corners, scale_factors, out);
  }

  template <class CTYPE, executorch::aten::ScalarType DTYPE>
  void test_upsample_bilinear2d_dtype() {
    TensorFactory<DTYPE> tf;

    if (torch::executor::testing::SupportedFeatures::get()->is_aten &&
        (DTYPE == ScalarType::Char || DTYPE == ScalarType::Short ||
         DTYPE == ScalarType::Int || DTYPE == ScalarType::Long)) {
      // not supported.
      return;
    }
    const auto input = tf.make({1, 1, 1, 2}, {1, 4});
    std::array<int64_t, 2> output_size = {1, 4};
    auto out = tf.zeros({1, 1, 1, 4});

    op_upsample_bilinear2d_vec_out(
        input,
        OptionalArrayRef<int64_t>({output_size.data(), output_size.size()}),
        true,
        {},
        out);

    const auto expected = tf.make({1, 1, 1, 4}, {1, 2, 3, 4});

    EXPECT_TENSOR_CLOSE(out, expected);
  }
};

TEST_F(OpUpsampleBilinear2dTest, Simple1x2To1x4) {
  TensorFactory<ScalarType::Float> tf;

  const auto input = tf.make({1, 1, 1, 2}, {1.0, 4.0});
  std::array<int64_t, 2> output_size = {1, 4};
  auto out = tf.zeros({1, 1, 1, 4});

  op_upsample_bilinear2d_vec_out(
      input,
      OptionalArrayRef<int64_t>({output_size.data(), output_size.size()}),
      false,
      {},
      out);

  const auto expected = tf.make({1, 1, 1, 4}, {1.0, 1.75, 3.25, 4.0});

  EXPECT_TENSOR_EQ(out, expected);
}

TEST_F(OpUpsampleBilinear2dTest, Simple1x2To1x4AlignCorners) {
  TensorFactory<ScalarType::Float> tf;

  const auto input = tf.make({1, 1, 2, 1}, {1.0, 4.0});
  std::array<int64_t, 2> output_size = {4, 1};
  auto out = tf.zeros({1, 1, 4, 1});

  op_upsample_bilinear2d_vec_out(
      input,
      OptionalArrayRef<int64_t>({output_size.data(), output_size.size()}),
      true,
      {},
      out);

  const auto expected = tf.make({1, 1, 4, 1}, {1.0, 2.0, 3.0, 4.0});

  EXPECT_TENSOR_EQ(out, expected);
}

TEST_F(OpUpsampleBilinear2dTest, Simple2x1To4x1) {
  TensorFactory<ScalarType::Float> tf;

  const auto input = tf.make({1, 1, 2, 1}, {1.0, 4.0});
  std::array<int64_t, 2> output_size = {4, 1};
  auto out = tf.zeros({1, 1, 4, 1});

  op_upsample_bilinear2d_vec_out(
      input,
      OptionalArrayRef<int64_t>({output_size.data(), output_size.size()}),
      false,
      {},
      out);

  const auto expected = tf.make({1, 1, 4, 1}, {1.0, 1.75, 3.25, 4.0});

  EXPECT_TENSOR_EQ(out, expected);
}

TEST_F(OpUpsampleBilinear2dTest, Simple2x1To4x1AlignCorners) {
  TensorFactory<ScalarType::Float> tf;

  const auto input = tf.make({1, 1, 2, 1}, {1.0, 4.0});
  std::array<int64_t, 2> output_size = {4, 1};
  auto out = tf.zeros({1, 1, 4, 1});

  op_upsample_bilinear2d_vec_out(
      input,
      OptionalArrayRef<int64_t>({output_size.data(), output_size.size()}),
      true,
      {},
      out);

  const auto expected = tf.make({1, 1, 4, 1}, {1.0, 2.0, 3.0, 4.0});

  EXPECT_TENSOR_EQ(out, expected);
}

TEST_F(OpUpsampleBilinear2dTest, SmokeTest) {
  TensorFactory<ScalarType::Float> tf;

  const auto input = tf.make(
      {1, 1, 2, 3},
      {
          1.0,
          2.0,
          3.0,
          4.0,
          5.0,
          6.0,
      });
  std::array<int64_t, 2> output_size = {3, 4};
  auto out = tf.zeros({1, 1, 3, 4});

  op_upsample_bilinear2d_vec_out(
      input,
      OptionalArrayRef<int64_t>({output_size.data(), output_size.size()}),
      false,
      {},
      out);

  const auto expected = tf.make(
      {1, 1, 3, 4},
      {1.0000,
       1.6250,
       2.3750,
       3.0000,
       2.5000,
       3.1250,
       3.8750,
       4.5000,
       4.0000,
       4.6250,
       5.3750,
       6.0000});

  EXPECT_TENSOR_CLOSE(out, expected);
}

TEST_F(OpUpsampleBilinear2dTest, SmokeTestAlignCorners) {
  TensorFactory<ScalarType::Float> tf;

  const auto input = tf.make(
      {1, 1, 2, 3},
      {
          1.0,
          2.0,
          3.0,
          4.0,
          5.0,
          6.0,
      });
  std::array<int64_t, 2> output_size = {3, 4};
  auto out = tf.zeros({1, 1, 3, 4});

  op_upsample_bilinear2d_vec_out(
      input,
      OptionalArrayRef<int64_t>({output_size.data(), output_size.size()}),
      true,
      {},
      out);

  const auto expected = tf.make(
      {1, 1, 3, 4},
      {1.0000,
       1.6667,
       2.3333,
       3.0000,
       2.5000,
       3.1667,
       3.8333,
       4.5000,
       4.0000,
       4.6667,
       5.3333,
       6.0000});

  EXPECT_TENSOR_CLOSE_WITH_TOL(out, expected, 0, 0.0001);
}

TEST_F(OpUpsampleBilinear2dTest, SmokeTestScales) {
  TensorFactory<ScalarType::Float> tf;

  const auto input = tf.make(
      {1, 1, 2, 3},
      {
          1.0,
          2.0,
          3.0,
          4.0,
          5.0,
          6.0,
      });
  auto out = tf.zeros({1, 1, 3, 4});

  const std::array<double, 2> scale_factors = {3.0 / 2, 4.0 / 3};
  op_upsample_bilinear2d_vec_out(
      input,
      {},
      false,
      OptionalArrayRef<double>({scale_factors.data(), scale_factors.size()}),
      out);

  const auto expected = tf.make(
      {1, 1, 3, 4},
      {1.0000,
       1.6250,
       2.3750,
       3.0000,
       2.5000,
       3.1250,
       3.8750,
       4.5000,
       4.0000,
       4.6250,
       5.3750,
       6.0000});

  EXPECT_TENSOR_CLOSE(out, expected);
}

TEST_F(OpUpsampleBilinear2dTest, SmokeTestAlignCornersScales) {
  TensorFactory<ScalarType::Float> tf;

  const auto input = tf.make(
      {1, 1, 2, 3},
      {
          1.0,
          2.0,
          3.0,
          4.0,
          5.0,
          6.0,
      });
  auto out = tf.zeros({1, 1, 3, 4});

  const std::array<double, 2> scale_factors = {3.0 / 2, 4.0 / 3};
  op_upsample_bilinear2d_vec_out(
      input,
      {},
      true,
      OptionalArrayRef<double>({scale_factors.data(), scale_factors.size()}),
      out);

  const auto expected = tf.make(
      {1, 1, 3, 4},
      {1.0000,
       1.6667,
       2.3333,
       3.0000,
       2.5000,
       3.1667,
       3.8333,
       4.5000,
       4.0000,
       4.6667,
       5.3333,
       6.0000});

  EXPECT_TENSOR_CLOSE_WITH_TOL(out, expected, 0, 0.0001);
}

TEST_F(OpUpsampleBilinear2dTest, DType) {
#define TEST_ENTRY(ctype, dtype) \
  test_upsample_bilinear2d_dtype<ctype, ScalarType::dtype>();
  ET_FORALL_REALHBF16_TYPES(TEST_ENTRY);
#undef TEST_ENTRY
}

TEST_F(OpUpsampleBilinear2dTest, MismatchedOutputSizeDies) {
  if (SupportedFeatures::get()->output_resize) {
    GTEST_SKIP()
        << "The current kernel supports implicitly resizing output tensor";
  }
  TensorFactory<ScalarType::Float> tf;

  const auto input = tf.ones({1, 1, 1, 2});
  std::array<int64_t, 2> output_size = {1, 4};
  auto out = tf.zeros({1, 1, 1, 5});

  ET_EXPECT_KERNEL_FAILURE(
      context_,
      op_upsample_bilinear2d_vec_out(
          input,
          OptionalArrayRef<int64_t>({output_size.data(), output_size.size()}),
          false,
          {},
          out));
}

TEST_F(OpUpsampleBilinear2dTest, InvalidInputRankDies) {
  TensorFactory<ScalarType::Float> tf;

  const auto input = tf.ones({1, 1, 2});
  std::array<int64_t, 2> output_size = {1, 4};
  auto out = tf.zeros({1, 1, 1, 4});

  ET_EXPECT_KERNEL_FAILURE(
      context_,
      op_upsample_bilinear2d_vec_out(
          input,
          OptionalArrayRef<int64_t>({output_size.data(), output_size.size()}),
          false,
          {},
          out));
}

TEST_F(OpUpsampleBilinear2dTest, InvalidOutputRankDies) {
  TensorFactory<ScalarType::Float> tf;

  const auto input = tf.ones({1, 1, 2});
  std::array<int64_t, 2> output_size = {1, 4};
  auto out = tf.zeros({1, 1, 4});

  ET_EXPECT_KERNEL_FAILURE(
      context_,
      op_upsample_bilinear2d_vec_out(
          input,
          OptionalArrayRef<int64_t>({output_size.data(), output_size.size()}),
          false,
          {},
          out));
}

TEST_F(OpUpsampleBilinear2dTest, MissingOutputSizeOrScaleDies) {
  TensorFactory<ScalarType::Float> tf;

  const auto input = tf.ones({1, 1, 2});
  auto out = tf.zeros({1, 1, 4});

  ET_EXPECT_KERNEL_FAILURE(
      context_, op_upsample_bilinear2d_vec_out(input, {}, false, {}, out));
}

TEST_F(OpUpsampleBilinear2dTest, BothOutputSizeAndScaleDies) {
  TensorFactory<ScalarType::Float> tf;

  const auto input = tf.ones({1, 1, 2});
  const std::array<int64_t, 2> output_size = {1, 4};
  const std::array<double, 2> scale_factors = {2, 1};
  auto out = tf.zeros({1, 1, 4});

  ET_EXPECT_KERNEL_FAILURE(
      context_,
      op_upsample_bilinear2d_vec_out(
          input,
          OptionalArrayRef<int64_t>({output_size.data(), output_size.size()}),
          false,
          OptionalArrayRef<double>(
              {scale_factors.data(), scale_factors.size()}),
          out));
}

TEST_F(OpUpsampleBilinear2dTest, MismatchedDTypeDies) {
  TensorFactory<ScalarType::Float> tf;
  TensorFactory<ScalarType::Long> tf2;

  const auto input = tf.ones({1, 1, 1, 2});
  std::array<int64_t, 2> output_size = {1, 4};
  auto out = tf2.zeros({1, 1, 1, 4});

  ET_EXPECT_KERNEL_FAILURE(
      context_,
      op_upsample_bilinear2d_vec_out(
          input,
          OptionalArrayRef<int64_t>({output_size.data(), output_size.size()}),
          false,
          {},
          out));
}

TEST_F(OpUpsampleBilinear2dTest, ComputedOutputSizeMatchesExpected) {
  // Computed output sizes (from input size * scales) must match PyTorch
  // eager-mode - multiplied as double and cast (truncated) to an integral type.
  // See
  // https://github.com/pytorch/pytorch/blob/main/aten/src/ATen/native/UpSample.cpp
  TensorFactory<ScalarType::Float> tf;

  // Test case format: { in_h, in_w, scale_h, scale_w, out_h, out_w }
  std::vector<std::tuple<int32_t, int32_t, double, double, int32_t, int32_t>>
      test_cases = {
          {10, 10, 9.99999, 9.55, 99, 95},
          {10, 10, 9.99999999, 0.1, 99, 1},
      };

  for (const auto& test_case : test_cases) {
    const auto [in_h, in_w, scale_h, scale_w, out_h, out_w] = test_case;

    const auto input = tf.ones({1, 1, in_h, in_w});
    auto out = tf.zeros({1, 1, out_h, out_w});
    std::array<double, 2> scale_factors = {scale_h, scale_w};

    op_upsample_bilinear2d_vec_out(
        input,
        {},
        false,
        OptionalArrayRef<double>({scale_factors.data(), scale_factors.size()}),
        out);

    const auto expected = tf.ones({1, 1, out_h, out_w});

    EXPECT_TENSOR_CLOSE(out, expected);
  }
}

TEST_F(OpUpsampleBilinear2dTest, ZeroComputedOutputSizeDies) {
  TensorFactory<ScalarType::Float> tf;

  const auto input = tf.ones({1, 1, 1, 2});
  auto out = tf.zeros({1, 1, 1, 4});
  std::array<double, 2> scale_factors = {1, 0.25};

  ET_EXPECT_KERNEL_FAILURE(
      context_,
      op_upsample_bilinear2d_vec_out(
          input,
          {},
          false,
          OptionalArrayRef<double>(
              {scale_factors.data(), scale_factors.size()}),
          out));
}

TEST_F(OpUpsampleBilinear2dTest, MismatchedDimOrderDies) {
  TensorFactory<ScalarType::Float> tf;

  if (torch::executor::testing::SupportedFeatures::get()->is_aten) {
    GTEST_SKIP() << "ATen kernel can implicitly convert dim order";
  }

  const auto input = tf.ones({1, 1, 1, 2});
  auto out = tf.zeros_channels_last({1, 1, 1, 4});
  std::array<double, 2> scale_factors = {2, 2};

  ET_EXPECT_KERNEL_FAILURE(
      context_,
      op_upsample_bilinear2d_vec_out(
          input,
          {},
          false,
          OptionalArrayRef<double>(
              {scale_factors.data(), scale_factors.size()}),
          out));
}

TEST_F(OpUpsampleBilinear2dTest, NumericsCheck) {
  TensorFactory<ScalarType::Float> tf;

  const auto input = tf.ones({3, 7, 47, 99});
  auto out = tf.zeros({3, 7, 291, 512});
  std::array<int64_t, 2> output_size = {291, 512};

  auto input_ptr = static_cast<float*>(input.mutable_data_ptr());
  for (auto i = 0ul; i < input.numel(); i++) {
    input_ptr[i] = static_cast<float>(i);
  }

  op_upsample_bilinear2d_vec_out(
      input,
      OptionalArrayRef<int64_t>({output_size.data(), output_size.size()}),
      false,
      {},
      out);

  // Indices and expected values to evaluate.
  std::vector<std::tuple<int, int, int, int, float>> test_values = {
      {0, 2, 60, 200, 10262.14453125},
      {1, 6, 5, 503, 60624.30078125},
      {2, 0, 111, 300, 66932.953125},
  };

  const auto output_data = static_cast<const float*>(out.const_data_ptr());
  for (const auto& test_case : test_values) {
    const auto [n, c, h, w, expected] = test_case;
    const auto actual = output_data
        [n * out.strides()[0] + c * out.strides()[1] + h * out.strides()[2] +
         w * out.strides()[3]];
    EXPECT_FLOAT_EQ(expected, actual);
  }
}

TEST_F(OpUpsampleBilinear2dTest, NumericsCheckAlignCorners) {
  TensorFactory<ScalarType::Float> tf;

  const auto input = tf.ones({3, 7, 47, 99});
  auto out = tf.zeros({3, 7, 291, 512});
  std::array<int64_t, 2> output_size = {291, 512};

  auto input_ptr = static_cast<float*>(input.mutable_data_ptr());
  for (auto i = 0ul; i < input.numel(); i++) {
    input_ptr[i] = static_cast<float>(i);
  }

  op_upsample_bilinear2d_vec_out(
      input,
      OptionalArrayRef<int64_t>({output_size.data(), output_size.size()}),
      true,
      {},
      out);

  // Indices and expected values to evaluate.
  std::vector<std::tuple<int, int, int, int, float>> test_values = {
      {0, 2, 60, 200, 10286.5634765625},
      {1, 6, 5, 503, 60663.98046875},
      {2, 0, 111, 300, 66942.625},
  };

  const auto output_data = static_cast<const float*>(out.const_data_ptr());
  for (const auto& test_case : test_values) {
    const auto [n, c, h, w, expected] = test_case;
    const auto actual = output_data
        [n * out.strides()[0] + c * out.strides()[1] + h * out.strides()[2] +
         w * out.strides()[3]];
    EXPECT_FLOAT_EQ(expected, actual);
  }
}

TEST_F(OpUpsampleBilinear2dTest, Simple5x1To4x1) {
  TensorFactory<ScalarType::Float> tf;

  const auto input = tf.make({1, 1, 5, 1}, {1.0, 2.0, 3.0, 4.0, 5.0});
  std::array<int64_t, 2> output_size = {4, 1};
  auto out = tf.zeros({1, 1, 4, 1});

  op_upsample_bilinear2d_vec_out(
      input,
      OptionalArrayRef<int64_t>({output_size.data(), output_size.size()}),
      false,
      {},
      out);

  const auto expected = tf.make({1, 1, 4, 1}, {1.1250, 2.3750, 3.6250, 4.8750});

  EXPECT_TENSOR_EQ(out, expected);
}

TEST_F(OpUpsampleBilinear2dTest, Simple5x1To4x1AlignCorners) {
  TensorFactory<ScalarType::Float> tf;

  const auto input = tf.make({1, 1, 5, 1}, {1.0, 2.0, 3.0, 4.0, 5.0});
  std::array<int64_t, 2> output_size = {4, 1};
  auto out = tf.zeros({1, 1, 4, 1});

  op_upsample_bilinear2d_vec_out(
      input,
      OptionalArrayRef<int64_t>({output_size.data(), output_size.size()}),
      true,
      {},
      out);

  const auto expected = tf.make({1, 1, 4, 1}, {1.0, 2.333333, 3.666667, 5.0});

  EXPECT_TENSOR_CLOSE(out, expected);
}

TEST_F(OpUpsampleBilinear2dTest, Simple1x2To1x4ChannelsLast) {
  TensorFactory<ScalarType::Float> tf;

  const auto input = tf.make_channels_last({1, 1, 1, 2}, {1.0, 4.0});
  std::array<int64_t, 2> output_size = {1, 4};
  auto out = tf.zeros_channels_last({1, 1, 1, 4});

  op_upsample_bilinear2d_vec_out(
      input,
      OptionalArrayRef<int64_t>({output_size.data(), output_size.size()}),
      false,
      {},
      out);

  const auto expected =
      tf.make_channels_last({1, 1, 1, 4}, {1.0, 1.75, 3.25, 4.0});

  EXPECT_TENSOR_EQ(out, expected);
}

TEST_F(OpUpsampleBilinear2dTest, SmokeTestChannelsLast) {
  TensorFactory<ScalarType::Float> tf;

  const auto input = tf.make_channels_last(
      {1, 2, 3, 4}, {0.0, 12, 1, 13, 2, 14, 3, 15, 4,  16, 5,  17,
                     6,   18, 7, 19, 8, 20, 9, 21, 10, 22, 11, 23});
  std::array<int64_t, 2> output_size = {6, 8};
  auto out = tf.zeros_channels_last({1, 2, 6, 8});

  op_upsample_bilinear2d_vec_out(
      input,
      OptionalArrayRef<int64_t>({output_size.data(), output_size.size()}),
      false,
      {},
      out);

  const auto expected = tf.make_channels_last(
      {1, 2, 6, 8},
      {0.0000, 12.0000, 0.2500,  12.2500, 0.7500,  12.7500, 1.2500,  13.2500,
       1.7500, 13.7500, 2.2500,  14.2500, 2.7500,  14.7500, 3.0000,  15.0000,
       1.0000, 13.0000, 1.2500,  13.2500, 1.7500,  13.7500, 2.2500,  14.2500,
       2.7500, 14.7500, 3.2500,  15.2500, 3.7500,  15.7500, 4.0000,  16.0000,
       3.0000, 15.0000, 3.2500,  15.2500, 3.7500,  15.7500, 4.2500,  16.2500,
       4.7500, 16.7500, 5.2500,  17.2500, 5.7500,  17.7500, 6.0000,  18.0000,
       5.0000, 17.0000, 5.2500,  17.2500, 5.7500,  17.7500, 6.2500,  18.2500,
       6.7500, 18.7500, 7.2500,  19.2500, 7.7500,  19.7500, 8.0000,  20.0000,
       7.0000, 19.0000, 7.2500,  19.2500, 7.7500,  19.7500, 8.2500,  20.2500,
       8.7500, 20.7500, 9.2500,  21.2500, 9.7500,  21.7500, 10.0000, 22.0000,
       8.0000, 20.0000, 8.2500,  20.2500, 8.7500,  20.7500, 9.2500,  21.2500,
       9.7500, 21.7500, 10.2500, 22.2500, 10.7500, 22.7500, 11.0000, 23.0000});

  EXPECT_TENSOR_CLOSE(out, expected);
}

TEST_F(OpUpsampleBilinear2dTest, NumericsCheckChannelsLast) {
  TensorFactory<ScalarType::Float> tf;

  const auto input = tf.zeros_channels_last({3, 7, 47, 99});
  auto out = tf.zeros_channels_last({3, 7, 291, 512});
  std::array<int64_t, 2> output_size = {291, 512};

  auto input_ptr = static_cast<float*>(input.mutable_data_ptr());
  for (auto i = 0ul; i < input.numel(); i++) {
    input_ptr[i] = static_cast<float>(i);
  }

  op_upsample_bilinear2d_vec_out(
      input,
      OptionalArrayRef<int64_t>({output_size.data(), output_size.size()}),
      false,
      {},
      out);

  // Indices and expected values to evaluate.
  std::vector<std::tuple<int, int, int, int, float>> test_values = {
      {0, 2, 60, 200, 6695.0137},
      {1, 6, 5, 503, 33524.098},
      {2, 0, 111, 300, 77678.68},
  };

  const auto output_data = static_cast<const float*>(out.const_data_ptr());
  for (const auto& test_case : test_values) {
    const auto [n, c, h, w, expected] = test_case;
    const auto actual = output_data
        [n * out.strides()[0] + c * out.strides()[1] + h * out.strides()[2] +
         w * out.strides()[3]];
    EXPECT_FLOAT_EQ(expected, actual);
  }
}

TEST_F(OpUpsampleBilinear2dTest, NumericsCheckAlignCornersChannelsLast) {
  TensorFactory<ScalarType::Float> tf;

  const auto input = tf.zeros_channels_last({3, 7, 47, 99});
  auto out = tf.zeros_channels_last({3, 7, 291, 512});
  std::array<int64_t, 2> output_size = {291, 512};

  auto input_ptr = static_cast<float*>(input.mutable_data_ptr());
  for (auto i = 0ul; i < input.numel(); i++) {
    input_ptr[i] = static_cast<float>(i);
  }

  op_upsample_bilinear2d_vec_out(
      input,
      OptionalArrayRef<int64_t>({output_size.data(), output_size.size()}),
      true,
      {},
      out);

  // Indices and expected values to evaluate.
  std::vector<std::tuple<int, int, int, int, float>> test_values = {
      {0, 2, 60, 200, 6865.9414},
      {1, 6, 5, 503, 33801.883},
      {2, 0, 111, 300, 77746.32},
  };

  const auto output_data = static_cast<const float*>(out.const_data_ptr());
  for (const auto& test_case : test_values) {
    const auto [n, c, h, w, expected] = test_case;
    const auto actual = output_data
        [n * out.strides()[0] + c * out.strides()[1] + h * out.strides()[2] +
         w * out.strides()[3]];
    EXPECT_FLOAT_EQ(expected, actual);
  }
}
