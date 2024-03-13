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

using namespace ::testing;
using exec_aten::IntArrayRef;
using exec_aten::ScalarType;
using exec_aten::Tensor;
using torch::executor::testing::TensorFactory;

class OpOnesOutTest : public OperatorTest {
 protected:
  Tensor& op_ones_out(IntArrayRef size, Tensor& out) {
    return torch::executor::aten::ones_outf(context_, size, out);
  }

  template <ScalarType DTYPE>
  void test_ones_out(std::vector<int32_t>&& size_int32_t) {
    TensorFactory<DTYPE> tf;
    std::vector<int64_t> size_int64_t(size_int32_t.begin(), size_int32_t.end());
    auto aref = IntArrayRef(size_int64_t.data(), size_int64_t.size());

    // Before: `out` consists of 0s.
    Tensor out = tf.zeros(size_int32_t);

    // After: `out` consists of 1s.
    op_ones_out(aref, out);

    EXPECT_TENSOR_EQ(out, tf.ones(size_int32_t));
  }
};

#define GENERATE_TEST(_, DTYPE)                  \
  TEST_F(OpOnesOutTest, DTYPE##Tensors) {        \
    test_ones_out<ScalarType::DTYPE>({});        \
    test_ones_out<ScalarType::DTYPE>({1});       \
    test_ones_out<ScalarType::DTYPE>({1, 1, 1}); \
    test_ones_out<ScalarType::DTYPE>({2, 0, 4}); \
    test_ones_out<ScalarType::DTYPE>({2, 3, 4}); \
  }

ET_FORALL_REAL_TYPES_AND(Bool, GENERATE_TEST)
