/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <executorch/kernels/test/FunctionHeaderWrapper.h> // Declares the operator
#include <executorch/kernels/test/supported_features.h>
#include <executorch/runtime/core/exec_aten/exec_aten.h>
#include <executorch/runtime/core/exec_aten/testing_util/tensor_factory.h>
#include <executorch/runtime/core/exec_aten/testing_util/tensor_util.h>
#include <executorch/runtime/core/exec_aten/util/tensor_util.h>

#include <executorch/kernels/test/TestUtil.h>

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

class OpUpsampleNearest2dTest : public OperatorTest {
 protected:
  Tensor& op_upsample_nearest2d_out(
      const Tensor& in,
      const OptionalArrayRef<int64_t> output_size,
      const OptionalArrayRef<double> scale_factors,
      Tensor& out) {
    return torch::executor::aten::upsample_nearest2d_outf(
        context_, in, output_size, scale_factors, out);
  }

  template <class CTYPE, executorch::aten::ScalarType DTYPE>
  void test_upsample_nearest2d_dtype() {
    TensorFactory<DTYPE> tf;

    if (torch::executor::testing::SupportedFeatures::get()->is_aten &&
        (DTYPE == ScalarType::Char || DTYPE == ScalarType::Short ||
         DTYPE == ScalarType::Int || DTYPE == ScalarType::Long)) {
      // not supported.
      return;
    }
    const auto input = tf.make({1, 1, 2, 2}, {1, 2, 3, 4});
    std::array<int64_t, 2> output_size = {4, 4};
    auto out = tf.zeros({1, 1, 4, 4});

    op_upsample_nearest2d_out(
        input,
        OptionalArrayRef<int64_t>({output_size.data(), output_size.size()}),
        {},
        out);

    const auto expected =
        tf.make({1, 1, 4, 4}, {1, 1, 2, 2, 1, 1, 2, 2, 3, 3, 4, 4, 3, 3, 4, 4});

    EXPECT_TENSOR_CLOSE(out, expected);
  }
};

TEST_F(OpUpsampleNearest2dTest, SmokeTest) {
  TensorFactory<ScalarType::Float> tf;

  const auto input = tf.make(
      {1, 1, 2, 2},
      {
          0.1,
          0.2,
          1.1,
          1.2,
      });
  std::array<int64_t, 2> output_size = {4, 4};
  auto out = tf.zeros({1, 1, 4, 4});

  op_upsample_nearest2d_out(
      input,
      OptionalArrayRef<int64_t>({output_size.data(), output_size.size()}),
      {},
      out);

  const auto expected = tf.make(
      {1, 1, 4, 4},
      {
          0.1,
          0.1,
          0.2,
          0.2,
          0.1,
          0.1,
          0.2,
          0.2,
          1.1,
          1.1,
          1.2,
          1.2,
          1.1,
          1.1,
          1.2,
          1.2,
      });

  EXPECT_TENSOR_EQ(out, expected);
}

TEST_F(OpUpsampleNearest2dTest, SmokeTestScale) {
  TensorFactory<ScalarType::Float> tf;

  const auto input = tf.make(
      {1, 1, 2, 2},
      {
          0.1,
          0.2,
          1.1,
          1.2,
      });
  auto out = tf.zeros({1, 1, 4, 4});
  std::array<double, 2> scale_factors = {2, 2};

  op_upsample_nearest2d_out(
      input,
      {},
      OptionalArrayRef<double>({scale_factors.data(), scale_factors.size()}),
      out);

  const auto expected = tf.make(
      {1, 1, 4, 4},
      {
          0.1,
          0.1,
          0.2,
          0.2,
          0.1,
          0.1,
          0.2,
          0.2,
          1.1,
          1.1,
          1.2,
          1.2,
          1.1,
          1.1,
          1.2,
          1.2,
      });

  EXPECT_TENSOR_EQ(out, expected);
}

TEST_F(OpUpsampleNearest2dTest, UpsampleSimpleFractional) {
  TensorFactory<ScalarType::Float> tf;

  const auto input = tf.make(
      {1, 1, 2, 2},
      {
          0.1,
          0.2,
          1.1,
          1.2,
      });
  std::array<int64_t, 2> output_size = {5, 9};
  auto out = tf.zeros({1, 1, 5, 9});

  op_upsample_nearest2d_out(
      input,
      OptionalArrayRef<int64_t>({output_size.data(), output_size.size()}),
      {},
      out);

  const auto expected = tf.make(
      {1, 1, 5, 9}, {0.1, 0.1, 0.1, 0.1, 0.1, 0.2, 0.2, 0.2, 0.2, 0.1, 0.1, 0.1,
                     0.1, 0.1, 0.2, 0.2, 0.2, 0.2, 0.1, 0.1, 0.1, 0.1, 0.1, 0.2,
                     0.2, 0.2, 0.2, 1.1, 1.1, 1.1, 1.1, 1.1, 1.2, 1.2, 1.2, 1.2,
                     1.1, 1.1, 1.1, 1.1, 1.1, 1.2, 1.2, 1.2, 1.2});

  EXPECT_TENSOR_EQ(out, expected);
}

TEST_F(OpUpsampleNearest2dTest, UpsampleSimpleFractionalScale) {
  TensorFactory<ScalarType::Float> tf;

  const auto input = tf.make(
      {1, 1, 2, 2},
      {
          0.1,
          0.2,
          1.1,
          1.2,
      });
  auto out = tf.zeros({1, 1, 5, 9});
  std::array<double, 2> scale_factors = {5 / 2.0, 9 / 2.0};

  op_upsample_nearest2d_out(
      input,
      {},
      OptionalArrayRef<double>({scale_factors.data(), scale_factors.size()}),
      out);

  const auto expected = tf.make(
      {1, 1, 5, 9}, {0.1, 0.1, 0.1, 0.1, 0.1, 0.2, 0.2, 0.2, 0.2, 0.1, 0.1, 0.1,
                     0.1, 0.1, 0.2, 0.2, 0.2, 0.2, 0.1, 0.1, 0.1, 0.1, 0.1, 0.2,
                     0.2, 0.2, 0.2, 1.1, 1.1, 1.1, 1.1, 1.1, 1.2, 1.2, 1.2, 1.2,
                     1.1, 1.1, 1.1, 1.1, 1.1, 1.2, 1.2, 1.2, 1.2});

  EXPECT_TENSOR_EQ(out, expected);
}

TEST_F(OpUpsampleNearest2dTest, MultiBatchAndChannel) {
  TensorFactory<ScalarType::Float> tf;
  const auto input = tf.make(
      {2, 2, 2, 2},
      {
          0.1,
          0.2,
          1.1,
          1.2,
          2.1,
          2.2,
          3.1,
          3.2,
          4.1,
          4.2,
          5.1,
          5.2,
          6.1,
          6.2,
          7.1,
          7.2,
      });
  std::array<int64_t, 2> output_size = {4, 4};
  auto out = tf.zeros({2, 2, 4, 4});

  op_upsample_nearest2d_out(
      input,
      OptionalArrayRef<int64_t>({output_size.data(), output_size.size()}),
      {},
      out);

  const auto expected = tf.make(
      {2, 2, 4, 4},
      {
          0.1, 0.1, 0.2, 0.2, 0.1, 0.1, 0.2, 0.2, 1.1, 1.1, 1.2, 1.2, 1.1,
          1.1, 1.2, 1.2, 2.1, 2.1, 2.2, 2.2, 2.1, 2.1, 2.2, 2.2, 3.1, 3.1,
          3.2, 3.2, 3.1, 3.1, 3.2, 3.2, 4.1, 4.1, 4.2, 4.2, 4.1, 4.1, 4.2,
          4.2, 5.1, 5.1, 5.2, 5.2, 5.1, 5.1, 5.2, 5.2, 6.1, 6.1, 6.2, 6.2,
          6.1, 6.1, 6.2, 6.2, 7.1, 7.1, 7.2, 7.2, 7.1, 7.1, 7.2, 7.2,
      });

  EXPECT_TENSOR_EQ(out, expected);
}

TEST_F(OpUpsampleNearest2dTest, DType) {
#define TEST_ENTRY(ctype, dtype) \
  test_upsample_nearest2d_dtype<ctype, ScalarType::dtype>();
  ET_FORALL_REALHBF16_TYPES(TEST_ENTRY);
#undef TEST_ENTRY
}

TEST_F(OpUpsampleNearest2dTest, MismatchedOutputSizeDies) {
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
      op_upsample_nearest2d_out(
          input,
          OptionalArrayRef<int64_t>({output_size.data(), output_size.size()}),
          {},
          out));
}

TEST_F(OpUpsampleNearest2dTest, InvalidInputRankDies) {
  TensorFactory<ScalarType::Float> tf;

  const auto input = tf.ones({1, 1, 2});
  std::array<int64_t, 2> output_size = {1, 4};
  auto out = tf.zeros({1, 1, 1, 4});

  ET_EXPECT_KERNEL_FAILURE(
      context_,
      op_upsample_nearest2d_out(
          input,
          OptionalArrayRef<int64_t>({output_size.data(), output_size.size()}),
          {},
          out));
}

TEST_F(OpUpsampleNearest2dTest, InvalidOutputRankDies) {
  TensorFactory<ScalarType::Float> tf;

  const auto input = tf.ones({1, 1, 2});
  std::array<int64_t, 2> output_size = {1, 4};
  auto out = tf.zeros({1, 1, 4});

  ET_EXPECT_KERNEL_FAILURE(
      context_,
      op_upsample_nearest2d_out(
          input,
          OptionalArrayRef<int64_t>({output_size.data(), output_size.size()}),
          {},
          out));
}

TEST_F(OpUpsampleNearest2dTest, MissingOutputSizeOrScaleDies) {
  TensorFactory<ScalarType::Float> tf;

  const auto input = tf.ones({1, 1, 2});
  auto out = tf.zeros({1, 1, 4});

  ET_EXPECT_KERNEL_FAILURE(
      context_, op_upsample_nearest2d_out(input, {}, {}, out));
}

TEST_F(OpUpsampleNearest2dTest, BothOutputSizeAndScaleDies) {
  TensorFactory<ScalarType::Float> tf;

  const auto input = tf.ones({1, 1, 1, 2});
  std::array<int64_t, 2> output_size = {1, 4};
  std::array<double, 2> scale_factors = {1, 2};
  auto out = tf.zeros({1, 1, 1, 4});

  ET_EXPECT_KERNEL_FAILURE(
      context_,
      op_upsample_nearest2d_out(
          input,
          OptionalArrayRef<int64_t>({output_size.data(), output_size.size()}),
          OptionalArrayRef<double>(
              {scale_factors.data(), scale_factors.size()}),
          out));
}

TEST_F(OpUpsampleNearest2dTest, MismatchedDTypeDies) {
  TensorFactory<ScalarType::Float> tf;
  TensorFactory<ScalarType::Long> tf2;

  const auto input = tf.ones({1, 1, 2});
  std::array<int64_t, 2> output_size = {1, 4};
  auto out = tf2.zeros({1, 1, 4});

  ET_EXPECT_KERNEL_FAILURE(
      context_,
      op_upsample_nearest2d_out(
          input,
          OptionalArrayRef<int64_t>({output_size.data(), output_size.size()}),
          {},
          out));
}

TEST_F(OpUpsampleNearest2dTest, ComputedOutputSizeMatchesExpected) {
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
    std::array<double, 2> scale_factors = {scale_h, scale_w};
    auto out = tf.zeros({1, 1, out_h, out_w});

    op_upsample_nearest2d_out(
        input,
        {},
        OptionalArrayRef<double>({scale_factors.data(), scale_factors.size()}),
        out);

    const auto expected = tf.ones({1, 1, out_h, out_w});

    EXPECT_TENSOR_EQ(out, expected);
  }
}

TEST_F(OpUpsampleNearest2dTest, ZeroComputedOutputSizeDies) {
  TensorFactory<ScalarType::Float> tf;

  const auto input = tf.ones({1, 1, 1, 2});
  std::array<double, 2> scale_factors = {1, 0.25};
  auto out = tf.zeros({1, 1, 1, 4});

  ET_EXPECT_KERNEL_FAILURE(
      context_,
      op_upsample_nearest2d_out(
          input,
          {},
          OptionalArrayRef<double>(
              {scale_factors.data(), scale_factors.size()}),
          out));
}

TEST_F(OpUpsampleNearest2dTest, SmokeTestChannelsLast) {
  TensorFactory<ScalarType::Float> tf;

  const auto input = tf.make_channels_last(
      {1, 2, 2, 2},
      {
          0.1,
          2.1,
          0.2,
          2.2,
          1.1,
          3.1,
          1.2,
          3.2,
      });
  std::array<int64_t, 2> output_size = {4, 4};
  auto out = tf.zeros_channels_last({1, 2, 4, 4});

  op_upsample_nearest2d_out(
      input,
      OptionalArrayRef<int64_t>({output_size.data(), output_size.size()}),
      {},
      out);

  const auto expected = tf.make_channels_last(
      {1, 2, 4, 4},
      {0.1000, 2.1000, 0.1000, 2.1000, 0.2000, 2.2000, 0.2000, 2.2000,
       0.1000, 2.1000, 0.1000, 2.1000, 0.2000, 2.2000, 0.2000, 2.2000,
       1.1000, 3.1000, 1.1000, 3.1000, 1.2000, 3.2000, 1.2000, 3.2000,
       1.1000, 3.1000, 1.1000, 3.1000, 1.2000, 3.2000, 1.2000, 3.2000});

  EXPECT_TENSOR_EQ(out, expected);
}
