/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <cmath>
#include <limits>

#include <executorch/kernels/test/FunctionHeaderWrapper.h> // Declares the operator
#include <executorch/kernels/test/TestUtil.h>
#include <executorch/kernels/test/supported_features.h>
#include <executorch/kernels/test/supported_features_skip.h>
#include <executorch/runtime/core/exec_aten/exec_aten.h>
#include <executorch/runtime/core/exec_aten/testing_util/tensor_factory.h>
#include <executorch/runtime/core/exec_aten/testing_util/tensor_util.h>
#include <executorch/runtime/core/exec_aten/util/scalar_type_util.h>

#include <gtest/gtest.h>

using namespace ::testing;
using executorch::aten::Scalar;
using executorch::aten::ScalarType;
using executorch::aten::Tensor;
using torch::executor::testing::TensorFactory;

class OpBucketizeTest : public OperatorTest {
 protected:
  Tensor& op_bucketize_tensor_out(
      const Tensor& self,
      const Tensor& boundaries,
      bool out_int32,
      bool right,
      Tensor& out) {
    return torch::executor::aten::bucketize_outf(
        context_, self, boundaries, out_int32, right, out);
  }

  Tensor& op_bucketize_scalar_out(
      const Scalar& self,
      const Tensor& boundaries,
      bool out_int32,
      bool right,
      Tensor& out) {
    return torch::executor::aten::bucketize_outf(
        context_, self, boundaries, out_int32, right, out);
  }
};

TEST_F(OpBucketizeTest, Basic1DFloatLongOut) {
  TensorFactory<ScalarType::Float> tf_float;
  TensorFactory<ScalarType::Long> tf_long;

  // boundaries = [1.0, 3.0, 5.0, 7.0, 9.0]
  Tensor boundaries = tf_float.make({5}, {1.0f, 3.0f, 5.0f, 7.0f, 9.0f});
  // self = [0.0, 1.0, 3.0, 5.0, 8.0, 10.0]
  Tensor self = tf_float.make({6}, {0.0f, 1.0f, 3.0f, 5.0f, 8.0f, 10.0f});
  Tensor out = tf_long.zeros({6});

  // right = false: boundaries[i-1] <= val < boundaries[i]
  // 0.0 -> 0 (0.0 < 1.0)
  // 1.0 -> 1 (1.0 <= 1.0 < 3.0)
  // 3.0 -> 2 (3.0 <= 3.0 < 5.0)
  // 5.0 -> 3 (5.0 <= 5.0 < 7.0)
  // 8.0 -> 4 (7.0 <= 8.0 < 9.0)
  // 10.0 -> 5 (9.0 <= 10.0)
  op_bucketize_tensor_out(
      self, boundaries, /*out_int32=*/false, /*right=*/false, out);
  EXPECT_TENSOR_EQ(out, tf_long.make({6}, {0, 1, 2, 3, 4, 5}));

  // right = true: boundaries[i-1] < val <= boundaries[i]
  // 0.0 -> 0 (0.0 <= 1.0)
  // 1.0 -> 0 (1.0 <= 1.0)
  // 3.0 -> 1 (1.0 < 3.0 <= 3.0)
  // 5.0 -> 2 (3.0 < 5.0 <= 5.0)
  // 8.0 -> 4 (7.0 < 8.0 <= 9.0)
  // 10.0 -> 5 (9.0 < 10.0)
  op_bucketize_tensor_out(
      self, boundaries, /*out_int32=*/false, /*right=*/true, out);
  EXPECT_TENSOR_EQ(out, tf_long.make({6}, {0, 0, 1, 2, 4, 5}));
}

TEST_F(OpBucketizeTest, Basic1DFloatInt32Out) {
  TensorFactory<ScalarType::Float> tf_float;
  TensorFactory<ScalarType::Int> tf_int;

  Tensor boundaries = tf_float.make({5}, {1.0f, 3.0f, 5.0f, 7.0f, 9.0f});
  Tensor self = tf_float.make({6}, {0.0f, 1.0f, 3.0f, 5.0f, 8.0f, 10.0f});
  Tensor out = tf_int.zeros({6});

  op_bucketize_tensor_out(
      self, boundaries, /*out_int32=*/true, /*right=*/false, out);
  EXPECT_TENSOR_EQ(out, tf_int.make({6}, {0, 1, 2, 3, 4, 5}));

  op_bucketize_tensor_out(
      self, boundaries, /*out_int32=*/true, /*right=*/true, out);
  EXPECT_TENSOR_EQ(out, tf_int.make({6}, {0, 0, 1, 2, 4, 5}));
}

TEST_F(OpBucketizeTest, MultiDimTensorInput) {
  TensorFactory<ScalarType::Double> tf_double;
  TensorFactory<ScalarType::Long> tf_long;

  Tensor boundaries = tf_double.make({3}, {2.0, 4.0, 6.0});
  // 2 x 3 tensor
  Tensor self = tf_double.make({2, 3}, {1.0, 2.0, 3.0, 4.0, 5.0, 7.0});
  Tensor out = tf_long.zeros({2, 3});

  op_bucketize_tensor_out(
      self, boundaries, /*out_int32=*/false, /*right=*/false, out);
  EXPECT_TENSOR_EQ(out, tf_long.make({2, 3}, {0, 1, 1, 2, 2, 3}));

  op_bucketize_tensor_out(
      self, boundaries, /*out_int32=*/false, /*right=*/true, out);
  EXPECT_TENSOR_EQ(out, tf_long.make({2, 3}, {0, 0, 1, 1, 2, 3}));
}

TEST_F(OpBucketizeTest, MixedDtypesIntegerAndFloat) {
  TensorFactory<ScalarType::Int> tf_int;
  TensorFactory<ScalarType::Float> tf_float;
  TensorFactory<ScalarType::Long> tf_long;

  // boundaries are float, self is int
  Tensor boundaries = tf_float.make({4}, {1.5f, 3.5f, 5.5f, 7.5f});
  Tensor self = tf_int.make({5}, {1, 2, 4, 6, 8});
  Tensor out = tf_long.zeros({5});

  op_bucketize_tensor_out(
      self, boundaries, /*out_int32=*/false, /*right=*/false, out);
  EXPECT_TENSOR_EQ(out, tf_long.make({5}, {0, 1, 2, 3, 4}));

  op_bucketize_tensor_out(
      self, boundaries, /*out_int32=*/false, /*right=*/true, out);
  EXPECT_TENSOR_EQ(out, tf_long.make({5}, {0, 1, 2, 3, 4}));
}

TEST_F(OpBucketizeTest, NaNHandling) {
  TensorFactory<ScalarType::Float> tf_float;
  TensorFactory<ScalarType::Long> tf_long;

  float nan = std::numeric_limits<float>::quiet_NaN();
  Tensor boundaries = tf_float.make({3}, {1.0f, 2.0f, 3.0f});
  Tensor self = tf_float.make({4}, {0.0f, nan, 2.5f, nan});
  Tensor out = tf_long.zeros({4});

  // NaN elements should be assigned index equal to boundaries.numel() (3)
  op_bucketize_tensor_out(
      self, boundaries, /*out_int32=*/false, /*right=*/false, out);
  EXPECT_TENSOR_EQ(out, tf_long.make({4}, {0, 3, 2, 3}));

  op_bucketize_tensor_out(
      self, boundaries, /*out_int32=*/false, /*right=*/true, out);
  EXPECT_TENSOR_EQ(out, tf_long.make({4}, {0, 3, 2, 3}));
}

TEST_F(OpBucketizeTest, EmptyInputTensor) {
  TensorFactory<ScalarType::Float> tf_float;
  TensorFactory<ScalarType::Long> tf_long;

  Tensor boundaries = tf_float.make({3}, {1.0f, 2.0f, 3.0f});
  Tensor self = tf_float.make({0}, {});
  Tensor out = tf_long.zeros({0});

  op_bucketize_tensor_out(
      self, boundaries, /*out_int32=*/false, /*right=*/false, out);
  EXPECT_EQ(out.numel(), 0);
}

TEST_F(OpBucketizeTest, EmptyBoundaries) {
  TensorFactory<ScalarType::Float> tf_float;
  TensorFactory<ScalarType::Long> tf_long;

  Tensor boundaries = tf_float.make({0}, {});
  Tensor self = tf_float.make({3}, {1.0f, 2.0f, 3.0f});
  Tensor out = tf_long.zeros({3});

  op_bucketize_tensor_out(
      self, boundaries, /*out_int32=*/false, /*right=*/false, out);
  EXPECT_TENSOR_EQ(out, tf_long.make({3}, {0, 0, 0}));
}

TEST_F(OpBucketizeTest, ScalarOverloadFloatLongOut) {
  TensorFactory<ScalarType::Float> tf_float;
  TensorFactory<ScalarType::Long> tf_long;

  Tensor boundaries = tf_float.make({5}, {1.0f, 3.0f, 5.0f, 7.0f, 9.0f});
  Tensor out = tf_long.zeros({});

  // self = 3.0, right = false -> 2
  op_bucketize_scalar_out(
      Scalar(3.0f), boundaries, /*out_int32=*/false, /*right=*/false, out);
  EXPECT_TENSOR_EQ(out, tf_long.make({}, {2}));

  // self = 3.0, right = true -> 1
  op_bucketize_scalar_out(
      Scalar(3.0f), boundaries, /*out_int32=*/false, /*right=*/true, out);
  EXPECT_TENSOR_EQ(out, tf_long.make({}, {1}));

  // self = 0.0 -> 0
  op_bucketize_scalar_out(
      Scalar(0.0f), boundaries, /*out_int32=*/false, /*right=*/false, out);
  EXPECT_TENSOR_EQ(out, tf_long.make({}, {0}));

  // self = 10.0 -> 5
  op_bucketize_scalar_out(
      Scalar(10.0f), boundaries, /*out_int32=*/false, /*right=*/false, out);
  EXPECT_TENSOR_EQ(out, tf_long.make({}, {5}));
}

TEST_F(OpBucketizeTest, ScalarOverloadIntInt32Out) {
  TensorFactory<ScalarType::Float> tf_float;
  TensorFactory<ScalarType::Int> tf_int;

  Tensor boundaries = tf_float.make({5}, {1.0f, 3.0f, 5.0f, 7.0f, 9.0f});
  Tensor out = tf_int.zeros({});

  // self = 3 (int), right = false -> 2 (int32)
  op_bucketize_scalar_out(
      Scalar(static_cast<int64_t>(3)),
      boundaries,
      /*out_int32=*/true,
      /*right=*/false,
      out);
  EXPECT_TENSOR_EQ(out, tf_int.make({}, {2}));

  // self = 3 (int), right = true -> 1 (int32)
  op_bucketize_scalar_out(
      Scalar(static_cast<int64_t>(3)),
      boundaries,
      /*out_int32=*/true,
      /*right=*/true,
      out);
  EXPECT_TENSOR_EQ(out, tf_int.make({}, {1}));
}

TEST_F(OpBucketizeTest, InvalidBoundariesDimThrows) {
  TensorFactory<ScalarType::Float> tf_float;
  TensorFactory<ScalarType::Long> tf_long;

  // 2D boundaries (invalid)
  Tensor boundaries_2d = tf_float.make({2, 2}, {1.0f, 2.0f, 3.0f, 4.0f});
  Tensor self = tf_float.make({2}, {1.5f, 2.5f});
  Tensor out = tf_long.zeros({2});

  ET_EXPECT_KERNEL_FAILURE(
      context_,
      op_bucketize_tensor_out(
          self, boundaries_2d, /*out_int32=*/false, /*right=*/false, out));
}

TEST_F(OpBucketizeTest, InvalidOutputTypeMismatchThrows) {
  TensorFactory<ScalarType::Float> tf_float;
  TensorFactory<ScalarType::Int> tf_int;
  TensorFactory<ScalarType::Long> tf_long;

  Tensor boundaries = tf_float.make({3}, {1.0f, 2.0f, 3.0f});
  Tensor self = tf_float.make({2}, {1.5f, 2.5f});

  // out_int32=true but output is Long
  Tensor out_long = tf_long.zeros({2});
  ET_EXPECT_KERNEL_FAILURE(
      context_,
      op_bucketize_tensor_out(
          self, boundaries, /*out_int32=*/true, /*right=*/false, out_long));

  // out_int32=false but output is Int
  Tensor out_int = tf_int.zeros({2});
  ET_EXPECT_KERNEL_FAILURE(
      context_,
      op_bucketize_tensor_out(
          self, boundaries, /*out_int32=*/false, /*right=*/false, out_int));
}

