/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <executorch/kernels/test/FunctionHeaderWrapper.h> // Declares the operator
#include <executorch/kernels/test/ScalarOverflowTestMacros.h>
#include <executorch/kernels/test/TestUtil.h>
#include <executorch/kernels/test/supported_features.h>
#include <executorch/kernels/test/supported_features_skip.h>
#include <executorch/runtime/core/exec_aten/exec_aten.h>
#include <executorch/runtime/core/exec_aten/testing_util/tensor_factory.h>
#include <executorch/runtime/core/exec_aten/testing_util/tensor_util.h>

#include <gtest/gtest.h>
#include <initializer_list>
#include <limits>

using namespace ::testing;
using executorch::aten::IntArrayRef;
using executorch::aten::Scalar;
using executorch::aten::ScalarType;
using executorch::aten::Tensor;
using torch::executor::testing::SupportedFeatures;
using torch::executor::testing::TensorFactory;

class OpConstantPadNDOutTest : public OperatorTest {
 protected:
  Tensor& op_constant_pad_nd_out(
      const Tensor& self,
      const IntArrayRef padding,
      const Scalar& value,
      Tensor& out) {
    return torch::executor::aten::constant_pad_nd_outf(
        context_, self, padding, value, out);
  }

  Tensor& op_constant_pad_nd_out(
      const Tensor& self,
      std::initializer_list<int64_t> padding,
      const Scalar& value,
      Tensor& out) {
    return op_constant_pad_nd_out(
        self, IntArrayRef(padding.begin(), padding.size()), value, out);
  }

  template <ScalarType DTYPE>
  void test_constant_pad_nd_out_dim2() {
    TensorFactory<DTYPE> tf;

    const std::vector<int32_t> sizes = {2, 4, 4};
    const std::vector<int32_t> sizes_out = {2, 4, 6};
    const std::vector<int64_t> padding = {1, 1};

    // clang-format off
    Tensor self = tf.make(
        sizes,
        {
           1,  2,  3,  4,
           5,  6,  7,  8,
           1,  2,  3,  4,
           5,  6,  7,  8,

           1,  2,  3,  4,
           5,  6,  7,  8,
           1,  2,  3,  4,
           5,  6,  7,  8,
        });
    // clang-format on

    // clang-format off
    Tensor expected = tf.make(
        sizes_out,
        {
           7,  1,  2,  3,  4,  7,
           7,  5,  6,  7,  8,  7,
           7,  1,  2,  3,  4,  7,
           7,  5,  6,  7,  8,  7,

           7,  1,  2,  3,  4,  7,
           7,  5,  6,  7,  8,  7,
           7,  1,  2,  3,  4,  7,
           7,  5,  6,  7,  8,  7,
        });
    // clang-format on

    IntArrayRef padding_ref = IntArrayRef(padding.data(), padding.size());
    Tensor out = tf.zeros(sizes_out);

    // Valid input should give the expected output
    op_constant_pad_nd_out(self, padding_ref, 7, out);
    EXPECT_TENSOR_CLOSE(out, expected);
  }

  template <ScalarType DTYPE>
  void test_constant_pad_nd_out_dim1() {
    TensorFactory<DTYPE> tf;

    const std::vector<int32_t> sizes = {2, 4, 4};
    const std::vector<int32_t> sizes_out = {2, 6, 4};
    const std::vector<int64_t> padding = {0, 0, 2, 0};

    // clang-format off
    Tensor self = tf.make(
        sizes,
        {
           1,  2,  3,  4,
           5,  6,  7,  8,
           1,  2,  3,  4,
           5,  6,  7,  8,

           1,  2,  3,  4,
           5,  6,  7,  8,
           1,  2,  3,  4,
           5,  6,  7,  8,
        });
    // clang-format on

    // clang-format off
    Tensor expected = tf.make(
        sizes_out,
        {
           7,  7,  7,  7,
           7,  7,  7,  7,
           1,  2,  3,  4,
           5,  6,  7,  8,
           1,  2,  3,  4,
           5,  6,  7,  8,

           7,  7,  7,  7,
           7,  7,  7,  7,
           1,  2,  3,  4,
           5,  6,  7,  8,
           1,  2,  3,  4,
           5,  6,  7,  8,
        });
    // clang-format on

    IntArrayRef padding_ref = IntArrayRef(padding.data(), padding.size());
    Tensor out = tf.zeros(sizes_out);

    // Valid input should give the expected output
    op_constant_pad_nd_out(self, padding_ref, 7, out);
    EXPECT_TENSOR_CLOSE(out, expected);
  }

  template <ScalarType DTYPE>
  void test_constant_pad_nd_out_dim0() {
    TensorFactory<DTYPE> tf;

    const std::vector<int32_t> sizes = {2, 4, 4};
    const std::vector<int32_t> sizes_out = {3, 4, 4};
    const std::vector<int64_t> padding = {0, 0, 0, 0, 1, 0};

    // clang-format off
    Tensor self = tf.make(
        sizes,
        {
           1,  2,  3,  4,
           5,  6,  7,  8,
           1,  2,  3,  4,
           5,  6,  7,  8,

           1,  2,  3,  4,
           5,  6,  7,  8,
           1,  2,  3,  4,
           5,  6,  7,  8,
        });
    // clang-format on

    // clang-format off
    Tensor expected = tf.make(
        sizes_out,
        {
           7,  7,  7,  7,
           7,  7,  7,  7,
           7,  7,  7,  7,
           7,  7,  7,  7,

           1,  2,  3,  4,
           5,  6,  7,  8,
           1,  2,  3,  4,
           5,  6,  7,  8,

           1,  2,  3,  4,
           5,  6,  7,  8,
           1,  2,  3,  4,
           5,  6,  7,  8,
        });
    // clang-format on

    IntArrayRef padding_ref = IntArrayRef(padding.data(), padding.size());
    Tensor out = tf.zeros(sizes_out);

    // Valid input should give the expected output
    op_constant_pad_nd_out(self, padding_ref, 7, out);
    EXPECT_TENSOR_CLOSE(out, expected);
  }

  template <ScalarType DTYPE>
  void test_constant_pad_nd_out_dim12() {
    TensorFactory<DTYPE> tf;

    const std::vector<int32_t> sizes = {2, 4, 4};
    const std::vector<int32_t> sizes_out = {2, 6, 7};
    const std::vector<int64_t> padding = {2, 1, 0, 2};

    // clang-format off
    Tensor self = tf.make(
        sizes,
        {
           1,  2,  3,  4,
           5,  6,  7,  8,
           1,  2,  3,  4,
           5,  6,  7,  8,

           1,  2,  3,  4,
           5,  6,  7,  8,
           1,  2,  3,  4,
           5,  6,  7,  8,
        });
    // clang-format on

    // clang-format off
    Tensor expected = tf.make(
        sizes_out,
        {
           7,  7,  1,  2,  3,  4,  7,
           7,  7,  5,  6,  7,  8,  7,
           7,  7,  1,  2,  3,  4,  7,
           7,  7,  5,  6,  7,  8,  7,
           7,  7,  7,  7,  7,  7,  7,
           7,  7,  7,  7,  7,  7,  7,

           7,  7,  1,  2,  3,  4,  7,
           7,  7,  5,  6,  7,  8,  7,
           7,  7,  1,  2,  3,  4,  7,
           7,  7,  5,  6,  7,  8,  7,
           7,  7,  7,  7,  7,  7,  7,
           7,  7,  7,  7,  7,  7,  7,
        });
    // clang-format on

    IntArrayRef padding_ref = IntArrayRef(padding.data(), padding.size());
    Tensor out = tf.zeros(sizes_out);

    // Valid input should give the expected output
    op_constant_pad_nd_out(self, padding_ref, 7, out);
    EXPECT_TENSOR_CLOSE(out, expected);
  }

  template <ScalarType DTYPE>
  void test_constant_pad_nd_out_dim02() {
    TensorFactory<DTYPE> tf;

    const std::vector<int32_t> sizes = {2, 4, 4};
    const std::vector<int32_t> sizes_out = {3, 4, 7};
    const std::vector<int64_t> padding = {2, 1, 0, 0, 0, 1};

    // clang-format off
    Tensor self = tf.make(
        sizes,
        {
           1,  2,  3,  4,
           5,  6,  7,  8,
           1,  2,  3,  4,
           5,  6,  7,  8,

           1,  2,  3,  4,
           5,  6,  7,  8,
           1,  2,  3,  4,
           5,  6,  7,  8,
        });
    // clang-format on

    // clang-format off
    Tensor expected = tf.make(
        sizes_out,
        {
           7,  7,  1,  2,  3,  4,  7,
           7,  7,  5,  6,  7,  8,  7,
           7,  7,  1,  2,  3,  4,  7,
           7,  7,  5,  6,  7,  8,  7,

           7,  7,  1,  2,  3,  4,  7,
           7,  7,  5,  6,  7,  8,  7,
           7,  7,  1,  2,  3,  4,  7,
           7,  7,  5,  6,  7,  8,  7,

           7,  7,  7,  7,  7,  7,  7,
           7,  7,  7,  7,  7,  7,  7,
           7,  7,  7,  7,  7,  7,  7,
           7,  7,  7,  7,  7,  7,  7,
        });
    // clang-format on

    IntArrayRef padding_ref = IntArrayRef(padding.data(), padding.size());
    Tensor out = tf.zeros(sizes_out);

    // Valid input should give the expected output
    op_constant_pad_nd_out(self, padding_ref, 7, out);
    EXPECT_TENSOR_CLOSE(out, expected);
  }

  template <ScalarType DTYPE>
  void test_constant_pad_nd_out_dim012() {
    TensorFactory<DTYPE> tf;

    const std::vector<int32_t> sizes = {2, 4, 4};
    const std::vector<int32_t> sizes_out = {3, 5, 7};
    const std::vector<int64_t> padding = {2, 1, 1, 0, 0, 1};

    // clang-format off
    Tensor self = tf.make(
        sizes,
        {
           1,  2,  3,  4,
           5,  6,  7,  8,
           1,  2,  3,  4,
           5,  6,  7,  8,

           1,  2,  3,  4,
           5,  6,  7,  8,
           1,  2,  3,  4,
           5,  6,  7,  8,
        });
    // clang-format on

    // clang-format off
    Tensor expected = tf.make(
        sizes_out,
        {
           7,  7,  7,  7,  7,  7,  7,
           7,  7,  1,  2,  3,  4,  7,
           7,  7,  5,  6,  7,  8,  7,
           7,  7,  1,  2,  3,  4,  7,
           7,  7,  5,  6,  7,  8,  7,

           7,  7,  7,  7,  7,  7,  7,
           7,  7,  1,  2,  3,  4,  7,
           7,  7,  5,  6,  7,  8,  7,
           7,  7,  1,  2,  3,  4,  7,
           7,  7,  5,  6,  7,  8,  7,

           7,  7,  7,  7,  7,  7,  7,
           7,  7,  7,  7,  7,  7,  7,
           7,  7,  7,  7,  7,  7,  7,
           7,  7,  7,  7,  7,  7,  7,
           7,  7,  7,  7,  7,  7,  7,
        });
    // clang-format on

    IntArrayRef padding_ref = IntArrayRef(padding.data(), padding.size());
    Tensor out = tf.zeros(sizes_out);

    // Valid input should give the expected output
    op_constant_pad_nd_out(self, padding_ref, 7, out);
    EXPECT_TENSOR_CLOSE(out, expected);
  }

  template <ScalarType DTYPE>
  void test_mixed_padding() {
    TensorFactory<DTYPE> tf;
    Tensor self =
        tf.make({2, 3, 4}, {1,  2,  3,  4,  5,  6,  7,  8,  9,  10, 11, 12,
                            13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24});
    Tensor expected =
        tf.make({2, 4, 3}, {0, 5,  6,  0, 9,  10, 0, 0, 0, 0, 0, 0,
                            0, 17, 18, 0, 21, 22, 0, 0, 0, 0, 0, 0});
    Tensor out = tf.zeros_like(expected);

    op_constant_pad_nd_out(self, {1, -2, -1, 2}, 0, out);
    EXPECT_TENSOR_EQ(out, expected);
  }

  template <ScalarType DTYPE>
  void expect_bad_scalar_value_dies(const Scalar& bad_value) {
    TensorFactory<DTYPE> tf;
    const std::vector<int32_t> sizes = {2, 2};
    const std::vector<int32_t> sizes_out = {2, 4};
    const std::vector<int64_t> padding = {1, 1};

    IntArrayRef padding_ref = IntArrayRef(padding.data(), padding.size());
    Tensor self = tf.ones(sizes);
    Tensor out = tf.zeros(sizes_out);

    ET_EXPECT_KERNEL_FAILURE(
        context_, op_constant_pad_nd_out(self, padding_ref, bad_value, out));
  }
};

TEST_F(OpConstantPadNDOutTest, TestPadDim2) {
#define TEST_ENTRY(ctype, dtype) \
  test_constant_pad_nd_out_dim2<ScalarType::dtype>();

  ET_FORALL_REALHBF16_TYPES(TEST_ENTRY);
#undef TEST_ENTRY
}

TEST_F(OpConstantPadNDOutTest, TestPadDim1) {
#define TEST_ENTRY(ctype, dtype) \
  test_constant_pad_nd_out_dim1<ScalarType::dtype>();

  ET_FORALL_REALHBF16_TYPES(TEST_ENTRY);
#undef TEST_ENTRY
}

TEST_F(OpConstantPadNDOutTest, TestPadDim0) {
#define TEST_ENTRY(ctype, dtype) \
  test_constant_pad_nd_out_dim0<ScalarType::dtype>();

  ET_FORALL_REALHBF16_TYPES(TEST_ENTRY);
#undef TEST_ENTRY
}

TEST_F(OpConstantPadNDOutTest, TestPadDim1And2) {
#define TEST_ENTRY(ctype, dtype) \
  test_constant_pad_nd_out_dim12<ScalarType::dtype>();

  ET_FORALL_REALHBF16_TYPES(TEST_ENTRY);
#undef TEST_ENTRY
}

TEST_F(OpConstantPadNDOutTest, TestPadDim0And2) {
#define TEST_ENTRY(ctype, dtype) \
  test_constant_pad_nd_out_dim02<ScalarType::dtype>();

  ET_FORALL_REALHBF16_TYPES(TEST_ENTRY);
#undef TEST_ENTRY
}

TEST_F(OpConstantPadNDOutTest, TestPadDim0And1And2) {
#define TEST_ENTRY(ctype, dtype) \
  test_constant_pad_nd_out_dim012<ScalarType::dtype>();

  ET_FORALL_REALHBF16_TYPES(TEST_ENTRY);
#undef TEST_ENTRY
}

TEST_F(OpConstantPadNDOutTest, MixedPositiveAndNegativePadding) {
#define TEST_ENTRY(ctype, dtype) test_mixed_padding<ScalarType::dtype>();
  ET_FORALL_REALHBF16_TYPES(TEST_ENTRY);
#undef TEST_ENTRY
}

TEST_F(OpConstantPadNDOutTest, CropMultipleDimensions) {
  TensorFactory<ScalarType::Int> tf;
  Tensor self =
      tf.make({2, 3, 4}, {1,  2,  3,  4,  5,  6,  7,  8,  9,  10, 11, 12,
                          13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24});
  Tensor out = tf.zeros({1, 1, 2});

  op_constant_pad_nd_out(self, {-1, -1, -1, -1, -1, 0}, 0, out);
  EXPECT_TENSOR_EQ(out, tf.make({1, 1, 2}, {18, 19}));
}

TEST_F(OpConstantPadNDOutTest, PaddingWithUnchangedShape) {
  TensorFactory<ScalarType::Float> tf;
  Tensor self = tf.make({4}, {1, 2, 3, 4});
  Tensor out = tf.zeros({4});

  op_constant_pad_nd_out(self, {2, -2}, 9, out);
  EXPECT_TENSOR_EQ(out, tf.make({4}, {9, 9, 1, 2}));
  op_constant_pad_nd_out(self, {-2, 2}, 9, out);
  EXPECT_TENSOR_EQ(out, tf.make({4}, {3, 4, 9, 9}));
}

TEST_F(OpConstantPadNDOutTest, NegativePaddingBool) {
  TensorFactory<ScalarType::Bool> tf;
  Tensor self = tf.make({4}, {true, false, true, false});
  Tensor out = tf.zeros({4});

  op_constant_pad_nd_out(self, {-1, 1}, true, out);
  EXPECT_TENSOR_EQ(out, tf.make({4}, {false, true, false, true}));
}

TEST_F(OpConstantPadNDOutTest, FullyCroppedInputWithPadding) {
  TensorFactory<ScalarType::Int> tf;
  Tensor self = tf.make({3}, {1, 2, 3});
  Tensor out = tf.zeros({2});

  op_constant_pad_nd_out(self, {-3, 2}, 7, out);
  EXPECT_TENSOR_EQ(out, tf.full({2}, 7));
}

TEST_F(OpConstantPadNDOutTest, EmptyOutput) {
  TensorFactory<ScalarType::Float> tf;
  Tensor self = tf.ones({3});
  Tensor out = tf.zeros({0});

  op_constant_pad_nd_out(self, {-2, -1}, 7, out);
  EXPECT_TENSOR_EQ(out, tf.zeros({0}));
}

TEST_F(OpConstantPadNDOutTest, EmptyOutputWithPositivePadding) {
  TensorFactory<ScalarType::Float> tf;
  Tensor self = tf.ones({2, 3});
  Tensor out = tf.zeros({3, 0});

  op_constant_pad_nd_out(self, {-3, 0, 1, 0}, 7, out);
  EXPECT_TENSOR_EQ(out, tf.zeros({3, 0}));
}

TEST_F(OpConstantPadNDOutTest, PadEmptyInput) {
  TensorFactory<ScalarType::Float> tf;
  Tensor self = tf.zeros({0});
  Tensor out = tf.zeros({2});

  op_constant_pad_nd_out(self, {1, 1}, 7, out);
  EXPECT_TENSOR_EQ(out, tf.full({2}, 7));
}

TEST_F(OpConstantPadNDOutTest, CropIgnoresUnusedScalarOverflow) {
  TensorFactory<ScalarType::Float> tf;
  Tensor self = tf.make({3}, {1, 2, 3});
  Tensor out = tf.zeros({1});

  op_constant_pad_nd_out(
      self, {-1, -1}, std::numeric_limits<double>::max(), out);
  EXPECT_TENSOR_EQ(out, tf.make({1}, {2}));
}

TEST_F(OpConstantPadNDOutTest, ScalarWithNoPadding) {
  TensorFactory<ScalarType::Float> tf;
  Tensor self = tf.full({}, 3);
  Tensor out = tf.zeros({});

  op_constant_pad_nd_out(self, {}, std::numeric_limits<double>::max(), out);
  EXPECT_TENSOR_EQ(out, self);
}

TEST_F(OpConstantPadNDOutTest, MixedPaddingDynamicOutput) {
  TensorFactory<ScalarType::Int> tf;
  Tensor self = tf.make({2, 3}, {1, 2, 3, 4, 5, 6});
  Tensor out =
      tf.zeros({4, 6}, torch::executor::TensorShapeDynamism::DYNAMIC_BOUND);

  op_constant_pad_nd_out(self, {-1, 2}, 7, out);
  EXPECT_TENSOR_EQ(out, tf.make({2, 4}, {2, 3, 7, 7, 5, 6, 7, 7}));
}

TEST_F(OpConstantPadNDOutTest, MixedPaddingChannelsLast) {
  TensorFactory<ScalarType::Float> tf;
  Tensor self = tf.channels_last_like(
      tf.make({1, 2, 2, 3}, {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12}));
  Tensor expected = tf.channels_last_like(
      tf.make({1, 2, 2, 3}, {5, 6, 0, 0, 0, 0, 11, 12, 0, 0, 0, 0}));
  Tensor out = tf.zeros_channels_last({1, 2, 2, 3});

  op_constant_pad_nd_out(self, {-1, 1, -1, 1}, 0, out);
  EXPECT_TENSOR_EQ(out, expected);
}

TEST_F(OpConstantPadNDOutTest, Issue13554) {
  TensorFactory<ScalarType::Double> tf;
  Tensor self = tf.ones({6, 19, 7, 8, 8});
  Tensor out = tf.zeros({6, 1, 17, 13, 5});
  const double value = std::numeric_limits<double>::max();

  op_constant_pad_nd_out(self, {5, -8, -4, 9, 1, 9, -9, -9}, value, out);
  EXPECT_TENSOR_EQ(out, tf.full({6, 1, 17, 13, 5}, value));
}

TEST_F(OpConstantPadNDOutTest, ExcessiveCroppingFails) {
  TensorFactory<ScalarType::Float> tf;
  Tensor self = tf.ones({3});
  Tensor out = tf.zeros({3});
  const int64_t min = std::numeric_limits<int64_t>::min();
  const std::vector<std::vector<int64_t>> paddings = {
      {-4, 5}, {5, -4}, {-2, -2}, {min, 0}, {0, min}};

  for (const auto& padding : paddings) {
    context_ = torch::executor::KernelRuntimeContext();
    ET_EXPECT_KERNEL_FAILURE(
        context_,
        op_constant_pad_nd_out(
            self, IntArrayRef(padding.data(), padding.size()), 0, out));
  }
}

TEST_F(OpConstantPadNDOutTest, OutputDimensionOverflowFails) {
  ET_SKIP_IF(
      SupportedFeatures::get()->is_aten, "ATen supports int64 tensor sizes");
  TensorFactory<ScalarType::Float> tf;
  Tensor self = tf.ones({3});
  Tensor out = tf.zeros({3});
  const std::vector<std::vector<int64_t>> paddings = {
      {std::numeric_limits<int32_t>::max(), 0},
      {std::numeric_limits<int64_t>::max(), 1}};

  for (const auto& padding : paddings) {
    context_ = torch::executor::KernelRuntimeContext();
    ET_EXPECT_KERNEL_FAILURE(
        context_,
        op_constant_pad_nd_out(
            self, IntArrayRef(padding.data(), padding.size()), 0, out));
  }
}

TEST_F(OpConstantPadNDOutTest, DifferentInputOutputTypesFail) {
  TensorFactory<ScalarType::Float> tf;
  TensorFactory<ScalarType::Double> tf_out;

  const std::vector<int32_t> sizes = {1, 4, 4};
  const std::vector<int32_t> sizes_out = {1, 4, 6};
  const std::vector<int64_t> padding = {1, 1};

  IntArrayRef padding_ref = IntArrayRef(padding.data(), padding.size());

  Tensor self = tf.ones(sizes);
  Tensor out = tf_out.zeros(sizes_out);

  ET_EXPECT_KERNEL_FAILURE(
      context_, op_constant_pad_nd_out(self, padding_ref, 0, out));
}

TEST_F(OpConstantPadNDOutTest, OddNumberOfPaddingElementsFail) {
  TensorFactory<ScalarType::Float> tf;

  const std::vector<int32_t> sizes = {1, 4, 4};
  const std::vector<int32_t> sizes_out = {1, 4, 4};
  const std::vector<int64_t> padding = {1, 1, 0};

  IntArrayRef padding_ref = IntArrayRef(padding.data(), padding.size());

  Tensor self = tf.ones(sizes);
  Tensor out = tf.zeros(sizes_out);

  ET_EXPECT_KERNEL_FAILURE(
      context_, op_constant_pad_nd_out(self, padding_ref, 0, out));
}

TEST_F(OpConstantPadNDOutTest, TooManyPaddingElementsFail) {
  TensorFactory<ScalarType::Float> tf;

  const std::vector<int32_t> sizes = {1, 4, 4};
  const std::vector<int32_t> sizes_out = {1, 4, 4};
  const std::vector<int64_t> padding = {3, 2, 1, 1, 2, 1, 1, 0};

  IntArrayRef padding_ref = IntArrayRef(padding.data(), padding.size());

  Tensor self = tf.ones(sizes);
  Tensor out = tf.zeros(sizes_out);

  ET_EXPECT_KERNEL_FAILURE(
      context_, op_constant_pad_nd_out(self, padding_ref, 0, out));
}

TEST_F(OpConstantPadNDOutTest, IncorrectOutputShapeFail) {
  ET_SKIP_IF(
      SupportedFeatures::get()->is_aten,
      "ATen kernel can handle reshape output");

  TensorFactory<ScalarType::Float> tf;

  const std::vector<int32_t> sizes = {1, 4, 4};
  const std::vector<int32_t> sizes_out = {1, 4, 4};
  const std::vector<int64_t> padding = {1, 1};

  IntArrayRef padding_ref = IntArrayRef(padding.data(), padding.size());

  Tensor self = tf.ones(sizes);
  Tensor out = tf.zeros(sizes_out);

  ET_EXPECT_KERNEL_FAILURE(
      context_, op_constant_pad_nd_out(self, padding_ref, 0, out));
}

GENERATE_SCALAR_OVERFLOW_TESTS(OpConstantPadNDOutTest)

TEST_F(OpConstantPadNDOutTest, PositivePaddingChannelsLast) {
  TensorFactory<ScalarType::Float> tf;

  Tensor self = tf.channels_last_like(
      tf.make({1, 3, 2, 2}, {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12}));
  Tensor out = tf.zeros_channels_last({1, 3, 2, 4});
  const std::vector<int64_t> padding = {1, 1};

  op_constant_pad_nd_out(
      self, IntArrayRef(padding.data(), padding.size()), 0.0, out);
  Tensor expected = tf.channels_last_like(
      tf.make({1, 3, 2, 4}, {0, 1, 2, 0, 0, 3, 4,  0, 0, 5,  6,  0,
                             0, 7, 8, 0, 0, 9, 10, 0, 0, 11, 12, 0}));
  EXPECT_TENSOR_EQ(out, expected);
}
