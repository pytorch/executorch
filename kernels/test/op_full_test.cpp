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
#include <executorch/runtime/core/exec_aten/exec_aten.h>
#include <executorch/runtime/core/exec_aten/testing_util/tensor_factory.h>
#include <executorch/runtime/core/exec_aten/testing_util/tensor_util.h>
#include <executorch/test/utils/DeathTest.h>

#include <gtest/gtest.h>

using namespace ::testing;
using executorch::aten::IntArrayRef;
using executorch::aten::MemoryFormat;
using executorch::aten::Scalar;
using executorch::aten::ScalarType;
using executorch::aten::Tensor;
using std::optional;
using torch::executor::testing::TensorFactory;

class OpFullOutTest : public OperatorTest {
 protected:
  Tensor&
  op_full_out(const IntArrayRef sizes, const Scalar& fill_value, Tensor& out) {
    return torch::executor::aten::full_outf(context_, sizes, fill_value, out);
  }

  template <ScalarType DTYPE>
  void test_ones_out(std::vector<int32_t>&& size_int32_t) {
    TensorFactory<DTYPE> tf;
    std::vector<int64_t> size_int64_t(size_int32_t.begin(), size_int32_t.end());
    auto aref = IntArrayRef(size_int64_t.data(), size_int64_t.size());

    // Boolean Scalar
    // Before: `out` consists of 0s.
    Tensor out = tf.zeros(size_int32_t);
    // After: `out` consists of 1s.
    op_full_out(aref, true, out);
    EXPECT_TENSOR_EQ(out, tf.ones(size_int32_t));

    // Integral Scalar
    // Before: `out` consists of 0s.
    out = tf.zeros(size_int32_t);
    // After: `out` consists of 1s.
    op_full_out(aref, 1, out);
    EXPECT_TENSOR_EQ(out, tf.ones(size_int32_t));

    // Floating Point Scalar
    // Before: `out` consists of 0s.
    out = tf.zeros(size_int32_t);
    // After: `out` consists of 1s.
    op_full_out(aref, 1.0, out);
    EXPECT_TENSOR_EQ(out, tf.ones(size_int32_t));
  }

  template <ScalarType DTYPE>
  void expect_bad_scalar_value_dies(const Scalar& bad_value) {
    TensorFactory<DTYPE> tf;
    std::vector<int32_t> sizes = {2, 2};
    std::vector<int64_t> sizes_int64_t(sizes.begin(), sizes.end());
    auto aref = IntArrayRef(sizes_int64_t.data(), sizes_int64_t.size());
    Tensor out = tf.zeros(sizes);

    ET_EXPECT_KERNEL_FAILURE(context_, op_full_out(aref, bad_value, out));
  }
};

#define GENERATE_TEST(_, DTYPE)                  \
  TEST_F(OpFullOutTest, DTYPE##Tensors) {        \
    test_ones_out<ScalarType::DTYPE>({});        \
    test_ones_out<ScalarType::DTYPE>({1});       \
    test_ones_out<ScalarType::DTYPE>({1, 1, 1}); \
    test_ones_out<ScalarType::DTYPE>({2, 0, 4}); \
    test_ones_out<ScalarType::DTYPE>({2, 3, 4}); \
  }

ET_FORALL_REALHBF16_TYPES(GENERATE_TEST)

GENERATE_SCALAR_OVERFLOW_TESTS(OpFullOutTest)

TEST_F(OpFullOutTest, HalfSupport) {
  TensorFactory<ScalarType::Half> tf;

  std::vector<int64_t> sizes_int64_t_vec = {2, 3};
  std::vector<int32_t> sizes_in32_t_vec = {2, 3};
  auto sizes = IntArrayRef(sizes_int64_t_vec.data(), sizes_int64_t_vec.size());

  // Boolean Scalar
  Tensor out = tf.zeros(sizes_in32_t_vec);
  op_full_out(sizes, true, out);
  EXPECT_TENSOR_EQ(out, tf.ones(sizes_in32_t_vec));

  // Integral Scalar
  out = tf.zeros(sizes_in32_t_vec);
  op_full_out(sizes, 1, out);
  EXPECT_TENSOR_EQ(out, tf.ones(sizes_in32_t_vec));

  // Floating Point Scalar
  out = tf.zeros(sizes_in32_t_vec);
  op_full_out(sizes, 3.1415926535, out);
  EXPECT_TENSOR_EQ(out, tf.full(sizes_in32_t_vec, 3.1415926535));
}

TEST_F(OpFullOutTest, ZeroDim) {
  TensorFactory<ScalarType::Half> tf;

  std::vector<int64_t> sizes_int64_t_vec = {};
  std::vector<int32_t> sizes_in32_t_vec = {};
  auto sizes = IntArrayRef(sizes_int64_t_vec.data(), sizes_int64_t_vec.size());

  // Boolean Scalar
  Tensor out = tf.zeros(sizes_in32_t_vec);
  op_full_out(sizes, true, out);
  EXPECT_TENSOR_EQ(out, tf.ones(sizes_in32_t_vec));
}
