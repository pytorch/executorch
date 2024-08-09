/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <executorch/runtime/core/exec_aten/exec_aten.h>
#include <executorch/runtime/core/exec_aten/testing_util/tensor_factory.h>
#include <executorch/runtime/core/exec_aten/util/tensor_util.h>
#include <executorch/runtime/platform/runtime.h>
#include <executorch/test/utils/DeathTest.h>
#include <cmath>
#include <limits>

#include <gtest/gtest.h>

using namespace ::testing;
using exec_aten::ScalarType;
using exec_aten::Tensor;
using executorch::runtime::extract_scalar_tensor;
using executorch::runtime::testing::TensorFactory;

class TensorUtilTest : public ::testing::Test {
 protected:
  // Factories for tests to use. These will be torn down and recreated for each
  // test case.
  TensorFactory<ScalarType::Byte> tf_byte_;
  TensorFactory<ScalarType::Int> tf_int_;
  TensorFactory<ScalarType::Float> tf_float_;
  TensorFactory<ScalarType::Double> tf_double_;
  TensorFactory<ScalarType::Bool> tf_bool_;

  void SetUp() override {
    // As some of these tests cause ET_LOG to be called, the PAL must be
    // initialized first by calling runtime_init();
    executorch::runtime::runtime_init();
  }
};

TEST_F(TensorUtilTest, IdentityChecks) {
  Tensor t = tf_byte_.ones({2, 2});

  // A tensor is the same shape as itself.
  ET_CHECK_SAME_SHAPE2(t, t);
  ET_CHECK_SAME_SHAPE3(t, t, t);

  // A tensor is the same dtype as itself.
  ET_CHECK_SAME_DTYPE2(t, t);
  ET_CHECK_SAME_DTYPE3(t, t, t);

  // A tensor is the same shape and dtype as itself.
  ET_CHECK_SAME_SHAPE_AND_DTYPE2(t, t);
  ET_CHECK_SAME_SHAPE_AND_DTYPE3(t, t, t);
}

TEST_F(TensorUtilTest, SameShapesDifferentDtypes) {
  // Three different tensors with the same shape but different dtypes.
  Tensor a = tf_byte_.ones({2, 2});
  Tensor b = tf_int_.ones({2, 2});
  Tensor c = tf_float_.ones({2, 2});

  // The tensors have the same shapes.
  ET_CHECK_SAME_SHAPE2(a, b);
  ET_CHECK_SAME_SHAPE3(a, b, c);

  // Not the same dtypes. Check both positions.
  ET_EXPECT_DEATH(ET_CHECK_SAME_DTYPE2(a, b), "");
  ET_EXPECT_DEATH(ET_CHECK_SAME_DTYPE2(b, a), "");
  ET_EXPECT_DEATH(ET_CHECK_SAME_SHAPE_AND_DTYPE2(a, b), "");
  ET_EXPECT_DEATH(ET_CHECK_SAME_SHAPE_AND_DTYPE2(b, a), "");

  // Test with a mismatching tensor in all positions, where the other two agree.
  ET_EXPECT_DEATH(ET_CHECK_SAME_DTYPE3(a, b, b), "");
  ET_EXPECT_DEATH(ET_CHECK_SAME_DTYPE3(b, a, b), "");
  ET_EXPECT_DEATH(ET_CHECK_SAME_DTYPE3(b, b, a), "");
  ET_EXPECT_DEATH(ET_CHECK_SAME_SHAPE_AND_DTYPE3(a, b, b), "");
  ET_EXPECT_DEATH(ET_CHECK_SAME_SHAPE_AND_DTYPE3(b, a, b), "");
  ET_EXPECT_DEATH(ET_CHECK_SAME_SHAPE_AND_DTYPE3(b, b, a), "");
}

TEST_F(TensorUtilTest, DifferentShapesSameDtypes) {
  // Two different tensors with different shapes but the same dtypes,
  // dimensions, and number of elements.
  Tensor a = tf_int_.ones({1, 4});
  Tensor b = tf_int_.ones({2, 2});
  // A third tensor with the same shape and dtype as b.
  Tensor b2 = tf_int_.ones({2, 2});

  // The different tensors are not the same shape. Check both positions.
  ET_EXPECT_DEATH(ET_CHECK_SAME_SHAPE2(a, b), "");
  ET_EXPECT_DEATH(ET_CHECK_SAME_SHAPE2(b, a), "");

  // Test with the different tensor in all positions.
  ET_EXPECT_DEATH(ET_CHECK_SAME_SHAPE3(a, b, b2), "");
  ET_EXPECT_DEATH(ET_CHECK_SAME_SHAPE3(b, a, b2), "");
  ET_EXPECT_DEATH(ET_CHECK_SAME_SHAPE3(b, b2, a), "");

  // They are the same dtypes.
  ET_CHECK_SAME_DTYPE2(a, b);
  ET_CHECK_SAME_DTYPE2(b, a);
  ET_CHECK_SAME_DTYPE3(a, b, b2);
  ET_CHECK_SAME_DTYPE3(b, a, b2);
  ET_CHECK_SAME_DTYPE3(b, b2, a);

  // But not the same shape-and-dtype.
  ET_EXPECT_DEATH(ET_CHECK_SAME_SHAPE_AND_DTYPE2(a, b), "");
  ET_EXPECT_DEATH(ET_CHECK_SAME_SHAPE_AND_DTYPE2(b, a), "");
  ET_EXPECT_DEATH(ET_CHECK_SAME_SHAPE_AND_DTYPE3(a, b, b2), "");
  ET_EXPECT_DEATH(ET_CHECK_SAME_SHAPE_AND_DTYPE3(b, a, b2), "");
  ET_EXPECT_DEATH(ET_CHECK_SAME_SHAPE_AND_DTYPE3(b, b2, a), "");
}

TEST_F(TensorUtilTest, ZeroDimensionalTensor) {
  // Create a zero-dimensional tensor.
  Tensor t = tf_int_.ones({});

  // Demonstrate that the tensor has zero dimensions.
  EXPECT_EQ(t.dim(), 0);

  // Make sure nothing blows up when the tensor has zero dimensions.
  ET_CHECK_SAME_SHAPE2(t, t);
  ET_CHECK_SAME_SHAPE3(t, t, t);
  ET_CHECK_SAME_DTYPE2(t, t);
  ET_CHECK_SAME_DTYPE3(t, t, t);
  ET_CHECK_SAME_SHAPE_AND_DTYPE2(t, t);
  ET_CHECK_SAME_SHAPE_AND_DTYPE3(t, t, t);
}

TEST_F(TensorUtilTest, EmptyTensor) {
  // Create a tensor with no elements by providing a zero-width dimension.
  Tensor t = tf_int_.ones({0});

  // Demonstrate that the tensor has no elements.
  EXPECT_EQ(t.nbytes(), 0);
  EXPECT_EQ(t.numel(), 0);

  // Make sure nothing blows up when the tensor has no elements.
  ET_CHECK_SAME_SHAPE2(t, t);
  ET_CHECK_SAME_SHAPE3(t, t, t);
  ET_CHECK_SAME_DTYPE2(t, t);
  ET_CHECK_SAME_DTYPE3(t, t, t);
  ET_CHECK_SAME_SHAPE_AND_DTYPE2(t, t);
  ET_CHECK_SAME_SHAPE_AND_DTYPE3(t, t, t);
}

TEST_F(TensorUtilTest, GetLeadingDimsSmokeTest) {
  // Create a tensor with some dimensions
  Tensor t = tf_int_.ones({2, 3, 4});

  // getLeadingDims(t, 1) => t.size(0)
  EXPECT_EQ(executorch::runtime::getLeadingDims(t, 1), 2);

  // getLeadingDims(t, 2) => t.size(0) * t.size(1)
  EXPECT_EQ(executorch::runtime::getLeadingDims(t, 2), 6);

  // getLeadingDims(t, 3) => t.size(0) * t.size(1) * t.size(2)
  EXPECT_EQ(executorch::runtime::getLeadingDims(t, 3), 24);
}

TEST_F(TensorUtilTest, GetLeadingDimsInputOutOfBoundDies) {
  // Create a tensor with some dimensions
  Tensor t = tf_int_.ones({2, 3, 4});

  // dim needs to be in the range [0, t.dim()]
  ET_EXPECT_DEATH(executorch::runtime::getLeadingDims(t, -2), "");
  ET_EXPECT_DEATH(executorch::runtime::getLeadingDims(t, -1), "");
  ET_EXPECT_DEATH(executorch::runtime::getLeadingDims(t, 4), "");
}

TEST_F(TensorUtilTest, GetTrailingDimsSmokeTest) {
  // Create a tensor with some dimensions
  Tensor t = tf_int_.ones({2, 3, 4});

  // getTrailingDims(t, 1) => t.size(2)
  EXPECT_EQ(executorch::runtime::getTrailingDims(t, 1), 4);

  // getTrailingDims(t, 0) => t.size(1) * t.size(2)
  EXPECT_EQ(executorch::runtime::getTrailingDims(t, 0), 12);

  // getTrailingDims(t, -1) => t.size(0) * t.size(1) * t.size(2)
  EXPECT_EQ(executorch::runtime::getTrailingDims(t, -1), 24);
}

TEST_F(TensorUtilTest, GetTrailingDimsInputOutOfBoundDies) {
  // Create a tensor with some dimensions
  Tensor t = tf_int_.ones({2, 3, 4});

  // dim needs to be in the range [-1, t.dim() - 1)
  ET_EXPECT_DEATH(executorch::runtime::getTrailingDims(t, -2), "");
  ET_EXPECT_DEATH(executorch::runtime::getTrailingDims(t, 3), "");
  ET_EXPECT_DEATH(executorch::runtime::getTrailingDims(t, 4), "");
}

TEST_F(TensorUtilTest, ContiguousCheckSupported) {
  std::vector<float> data = {1, 2, 3, 4, 5, 6};
  std::vector<int32_t> sizes = {1, 2, 3};

  Tensor t_contiguous = tf_float_.make(sizes, data);

  // t_incontiguous = tf.make(sizes=(1, 2, 3)).permute(2, 0, 1)
  // {3, 1, 2}
  // changed stride {1, 3, 1} => {2, 1, 2} because {1, 3, 1} is not
  // the right value.
  Tensor t_incontiguous = tf_float_.make(sizes, data, /*strides=*/{2, 1, 2});

  // Assert t_contiguous is contiguous.
  ET_CHECK_CONTIGUOUS(t_contiguous);

  // Assert t_incontiguous is incontiguous.
  ET_EXPECT_DEATH(ET_CHECK_CONTIGUOUS(t_incontiguous), "");
}

TEST_F(TensorUtilTest, CheckSameContiguousStrideSupported) {
  // Tensors in the following list share same stride.
  std::vector<Tensor> same_stride_tensor_list = {
      tf_float_.ones(/*sizes=*/{1, 2, 3, 4}),
      tf_byte_.ones(/*sizes=*/{4, 2, 3, 4}),
      tf_int_.ones(/*sizes=*/{10, 2, 3, 4}),
      tf_float_.make(
          /*sizes=*/{0, 2, 3, 4}, /*data=*/{}, /*strides=*/{24, 12, 4, 1})};

  // different_stride = tensor(size=(0,2,3,4)).permute(0, 2, 3, 1)
  // {0, 3, 4, 2}
  // stride for (0, 2, 3, 4) with permute = (24, 1, 8, 2)
  // So change stride from {24, 3, 1, 6} => {24, 1, 8, 2}
  Tensor different_stride = tf_float_.make(
      /*sizes=*/{0, 2, 3, 4}, /*data=*/{}, /*strides=*/{24, 1, 8, 2});

  // Any two tensors in `same_stride_tensor_list` have same strides. The two
  // could contain duplicate tensors.
  for (int i = 0; i < same_stride_tensor_list.size(); i++) {
    for (int j = i; j < same_stride_tensor_list.size(); j++) {
      auto ti = same_stride_tensor_list[i];
      auto tj = same_stride_tensor_list[j];
      ET_CHECK_SAME_STRIDES2(ti, tj);
    }
  }

  // Any tensor in `same_stride_tensor_list` shall not have same stride with
  // `different_stride`.
  for (int i = 0; i < same_stride_tensor_list.size(); i++) {
    auto ti = same_stride_tensor_list[i];
    ET_EXPECT_DEATH(ET_CHECK_SAME_STRIDES2(ti, different_stride), "");
  }

  // Any three tensors in same_stride_tensor_list have same strides. The three
  // could contain duplicate tensors.
  for (size_t i = 0; i < same_stride_tensor_list.size(); i++) {
    for (size_t j = i; j < same_stride_tensor_list.size(); j++) {
      for (size_t k = j; k < same_stride_tensor_list.size(); k++) {
        auto ti = same_stride_tensor_list[i];
        auto tj = same_stride_tensor_list[j];
        auto tk = same_stride_tensor_list[k];
        ET_CHECK_SAME_STRIDES3(ti, tj, tk);
      }
    }
  }

  // Any two tensors in same_stride_tensor_list shall not have same strides with
  // `different_stride`. The two could contain duplicate tensors.
  for (int i = 0; i < same_stride_tensor_list.size(); i++) {
    for (int j = i; j < same_stride_tensor_list.size(); j++) {
      auto ti = same_stride_tensor_list[i];
      auto tj = same_stride_tensor_list[j];
      ET_EXPECT_DEATH(ET_CHECK_SAME_STRIDES3(ti, tj, different_stride), "");
    }
  }
}

TEST_F(TensorUtilTest, ExtractIntScalarTensorSmoke) {
  Tensor t = tf_int_.ones({1});
  bool ok;
#define CASE_INT_DTYPE(ctype, unused)          \
  ctype out_##ctype;                           \
  ok = extract_scalar_tensor(t, &out_##ctype); \
  ASSERT_TRUE(ok);                             \
  EXPECT_EQ(out_##ctype, 1);

  ET_FORALL_INT_TYPES(CASE_INT_DTYPE);
#undef CASE_INT_DTYPE
}

TEST_F(TensorUtilTest, ExtractFloatScalarTensorFloatingTypeSmoke) {
  Tensor t = tf_float_.ones({1});
  bool ok;
#define CASE_FLOAT_DTYPE(ctype, unused)        \
  ctype out_##ctype;                           \
  ok = extract_scalar_tensor(t, &out_##ctype); \
  ASSERT_TRUE(ok);                             \
  EXPECT_EQ(out_##ctype, 1.0);

  ET_FORALL_FLOAT_TYPES(CASE_FLOAT_DTYPE);
#undef CASE_FLOAT_DTYPE
}

TEST_F(TensorUtilTest, ExtractFloatScalarTensorIntegralTypeSmoke) {
  Tensor t = tf_int_.ones({1});
  bool ok;
#define CASE_FLOAT_DTYPE(ctype, unused)        \
  ctype out_##ctype;                           \
  ok = extract_scalar_tensor(t, &out_##ctype); \
  ASSERT_TRUE(ok);                             \
  EXPECT_EQ(out_##ctype, 1.0);

  ET_FORALL_INT_TYPES(CASE_FLOAT_DTYPE);
#undef CASE_FLOAT_DTYPE
}

TEST_F(TensorUtilTest, ExtractBoolScalarTensorSmoke) {
  Tensor t = tf_bool_.ones({1});
  bool out;
  bool ok;
  ok = extract_scalar_tensor(t, &out);
  ASSERT_TRUE(ok);
  EXPECT_EQ(out, true);
}

TEST_F(TensorUtilTest, FloatScalarTensorStressTests) {
  float value;
  bool ok;

  // Case: Positive Infinity
  Tensor t_pos_inf = tf_double_.make({1}, {INFINITY});
  ok = extract_scalar_tensor(t_pos_inf, &value);
  EXPECT_TRUE(ok);
  EXPECT_TRUE(std::isinf(value));

  // Case: Negative Infinity
  Tensor t_neg_inf = tf_double_.make({1}, {-INFINITY});
  ok = extract_scalar_tensor(t_neg_inf, &value);
  EXPECT_TRUE(ok);
  EXPECT_TRUE(std::isinf(value));

  // Case: Not a Number (NaN) - ex: sqrt(-1.0)
  Tensor t_nan = tf_double_.make({1}, {NAN});
  ok = extract_scalar_tensor(t_nan, &value);
  EXPECT_TRUE(ok);
  EXPECT_TRUE(std::isnan(value));
}

TEST_F(TensorUtilTest, IntScalarTensorNotIntegralTypeFails) {
  Tensor t = tf_float_.ones({1});
  int64_t out;
  // Fails since tensor is floating type but attempting to extract integer
  // value.
  bool ok = extract_scalar_tensor(t, &out);
  EXPECT_FALSE(ok);
}

TEST_F(TensorUtilTest, FloatScalarTensorNotFloatingTypeFails) {
  Tensor t = tf_bool_.ones({1});
  double out;
  // Fails since tensor is boolean type but attempting to extract float value.
  bool ok = extract_scalar_tensor(t, &out);
  EXPECT_FALSE(ok);
}

TEST_F(TensorUtilTest, IntTensorNotScalarFails) {
  Tensor t = tf_int_.ones({2, 3});
  int64_t out;
  // Fails since tensor has multiple dims and values.
  bool ok = extract_scalar_tensor(t, &out);
  EXPECT_FALSE(ok);
}

TEST_F(TensorUtilTest, FloatTensorNotScalarFails) {
  Tensor t = tf_float_.ones({2, 3});
  double out;
  // Fails since tensor has multiple dims and values.
  bool ok = extract_scalar_tensor(t, &out);
  EXPECT_FALSE(ok);
}

TEST_F(TensorUtilTest, IntTensorOutOfBoundFails) {
  Tensor t = tf_int_.make({1}, {256});
  int8_t out;
  // Fails since 256 is out of bounds for `int8_t` (-128 to 127).
  bool ok = extract_scalar_tensor(t, &out);
  EXPECT_FALSE(ok);
}

TEST_F(TensorUtilTest, FloatTensorOutOfBoundFails) {
  Tensor t = tf_double_.make({1}, {1.0}); // Placeholder value.
  float out;
  bool ok;

#define CASE_FLOAT(value)              \
  t = tf_double_.make({1}, {value});   \
  ok = extract_scalar_tensor(t, &out); \
  EXPECT_FALSE(ok);

  // Float tensor can't handle double's largest negative value (note the use of
  // `lowest` rather than `min`).
  CASE_FLOAT(std::numeric_limits<double>::lowest());

  // Float tensor can't handle double's largest positive value.
  CASE_FLOAT(std::numeric_limits<double>::max());

#undef CASE_FLOAT
}

TEST_F(TensorUtilTest, BoolScalarTensorNotBooleanTypeFails) {
  Tensor c = tf_byte_.ones({1});
  bool out;
  // Fails since tensor is integral type but attempting to extract boolean
  // value.
  bool ok = extract_scalar_tensor(c, &out);
  EXPECT_FALSE(ok);
}

TEST_F(TensorUtilTest, BoolTensorNotScalarFails) {
  Tensor c = tf_bool_.ones({2, 3});
  bool out;
  // Fails since tensor has multiple dims and values.
  bool ok = extract_scalar_tensor(c, &out);
  EXPECT_FALSE(ok);
}

//
// Tests for utility functions that check tensor attributes
//

TEST_F(TensorUtilTest, TensorIsRankTest) {
  using executorch::runtime::tensor_is_rank;
  Tensor a = tf_float_.ones({2, 3, 5});

  EXPECT_TRUE(tensor_is_rank(a, 3));
  EXPECT_FALSE(tensor_is_rank(a, 0));
  EXPECT_FALSE(tensor_is_rank(a, 5));
}

TEST_F(TensorUtilTest, TensorHasDimTest) {
  using executorch::runtime::tensor_has_dim;
  Tensor a = tf_float_.ones({2, 3, 5});

  EXPECT_TRUE(tensor_has_dim(a, 2));
  EXPECT_TRUE(tensor_has_dim(a, 1));
  EXPECT_TRUE(tensor_has_dim(a, 0));
  EXPECT_TRUE(tensor_has_dim(a, -1));
  EXPECT_TRUE(tensor_has_dim(a, -2));
  EXPECT_TRUE(tensor_has_dim(a, -3));

  EXPECT_FALSE(tensor_has_dim(a, -4));
  EXPECT_FALSE(tensor_has_dim(a, 3));
}

TEST_F(TensorUtilTest, TensorsHaveSameDtypeTest) {
  using executorch::runtime::tensors_have_same_dtype;
  Tensor a = tf_float_.ones({2, 3});
  Tensor b = tf_float_.ones({2, 3});
  Tensor c = tf_float_.ones({3, 3});
  Tensor d = tf_int_.ones({4, 3});

  EXPECT_TRUE(tensors_have_same_dtype(a, b));
  EXPECT_FALSE(tensors_have_same_dtype(a, d));
  EXPECT_TRUE(tensors_have_same_dtype(a, b, c));
  EXPECT_FALSE(tensors_have_same_dtype(a, b, d));
}

TEST_F(TensorUtilTest, TensorsHaveSameSizeAtDimTest) {
  using executorch::runtime::tensors_have_same_size_at_dims;
  Tensor a = tf_float_.ones({2, 3, 4, 5});
  Tensor b = tf_float_.ones({5, 4, 3, 2});

  EXPECT_TRUE(tensors_have_same_size_at_dims(a, 0, b, 3));
  EXPECT_TRUE(tensors_have_same_size_at_dims(a, 1, b, 2));
  EXPECT_FALSE(tensors_have_same_size_at_dims(a, 1, b, 0));
  EXPECT_FALSE(tensors_have_same_size_at_dims(a, 4, b, 0));
  EXPECT_FALSE(tensors_have_same_size_at_dims(a, 2, b, 3));
}

TEST_F(TensorUtilTest, TensorsHaveSameShapeTest) {
  using executorch::runtime::tensors_have_same_shape;
  Tensor a = tf_float_.ones({2, 3});
  Tensor b = tf_int_.ones({2, 3});
  Tensor c = tf_byte_.ones({2, 3});
  Tensor d = tf_double_.ones({3, 2});
  Tensor e = tf_bool_.ones({3, 2});

  EXPECT_TRUE(tensors_have_same_shape(a, b));
  EXPECT_FALSE(tensors_have_same_shape(a, d));
  EXPECT_TRUE(tensors_have_same_shape(d, e));
  EXPECT_TRUE(tensors_have_same_shape(a, b, c));
  EXPECT_FALSE(tensors_have_same_shape(a, b, d));
  EXPECT_FALSE(tensors_have_same_shape(a, d, e));

  Tensor scalar_a = tf_float_.ones({1, 1});
  Tensor scalar_b = tf_double_.ones({1});
  Tensor scalar_c = tf_int_.ones({1, 1, 1, 1});

  EXPECT_TRUE(tensors_have_same_shape(scalar_a, scalar_b));
  EXPECT_TRUE(tensors_have_same_shape(scalar_a, scalar_b, scalar_c));
}

TEST_F(TensorUtilTest, TensorsHaveSameShapeAndDtypeTest) {
  using executorch::runtime::tensors_have_same_shape_and_dtype;
  Tensor a = tf_float_.ones({2, 3});
  Tensor b = tf_float_.ones({2, 3});
  Tensor c = tf_float_.ones({2, 3});
  Tensor d = tf_double_.ones({2, 3});
  Tensor e = tf_float_.ones({3, 2});

  EXPECT_TRUE(tensors_have_same_shape_and_dtype(a, b));
  EXPECT_FALSE(tensors_have_same_shape_and_dtype(a, d));
  EXPECT_TRUE(tensors_have_same_shape_and_dtype(a, b, c));
  EXPECT_FALSE(tensors_have_same_shape_and_dtype(a, b, d));
  EXPECT_FALSE(tensors_have_same_shape_and_dtype(a, d, e));

  Tensor scalar_a = tf_float_.ones({1, 1});
  Tensor scalar_b = tf_float_.ones({1});
  Tensor scalar_c = tf_float_.ones({1, 1, 1, 1});

  EXPECT_TRUE(tensors_have_same_shape_and_dtype(scalar_a, scalar_b));
  EXPECT_TRUE(tensors_have_same_shape_and_dtype(scalar_a, scalar_b, scalar_c));
}

TEST_F(TensorUtilTest, TensorsHaveSameStridesTest) {
  using executorch::runtime::tensors_have_same_strides;
  Tensor a = tf_float_.full_channels_last({4, 5, 2, 3}, 1);
  Tensor b = tf_float_.full_channels_last({4, 5, 2, 3}, 2);
  Tensor c = tf_float_.full_channels_last({4, 5, 2, 3}, 3);
  Tensor d = tf_double_.ones({4, 5, 2, 3});
  Tensor e = tf_float_.ones({4, 5, 2, 3});

  EXPECT_TRUE(tensors_have_same_strides(a, b));
  EXPECT_FALSE(tensors_have_same_strides(a, d));
  EXPECT_TRUE(tensors_have_same_strides(a, b, c));
  EXPECT_FALSE(tensors_have_same_strides(a, b, d));
  EXPECT_FALSE(tensors_have_same_strides(a, d, e));
}

TEST_F(TensorUtilTest, TensorIsContiguous) {
  using executorch::runtime::tensor_is_contiguous;
  // Note that the strides.size() == 0 case is not tested, since
  Tensor a = tf_float_.full_channels_last({4, 5, 2, 3}, 1);
  Tensor b = tf_float_.ones({4, 5, 2, 3});
  Tensor c = tf_float_.full_channels_last({1, 1, 1, 1}, 1);
  Tensor d = tf_float_.ones({});

  EXPECT_FALSE(tensor_is_contiguous(a));
  EXPECT_TRUE(tensor_is_contiguous(b));
  EXPECT_TRUE(tensor_is_contiguous(c));
  EXPECT_TRUE(tensor_is_contiguous(d));
}

TEST_F(TensorUtilTest, ResizeZeroDimTensor) {
  Tensor a = tf_float_.ones({});

  EXPECT_EQ(
      executorch::runtime::resize_tensor(a, {}),
      executorch::runtime::Error::Ok);
  EXPECT_EQ(a.dim(), 0);
}
