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
#include <executorch/runtime/core/exec_aten/util/scalar_type_util.h>

#include <gtest/gtest.h>

using namespace ::testing;
using exec_aten::IntArrayRef;
using exec_aten::ScalarType;
using exec_aten::Tensor;
using torch::executor::testing::TensorFactory;

Tensor& op_zeros_out(IntArrayRef size, Tensor& out) {
  exec_aten::RuntimeContext context{};
  return torch::executor::aten::zeros_outf(context, size, out);
}

template <ScalarType DTYPE>
void test_zeros_out(std::vector<int32_t>&& size_int32_t) {
  TensorFactory<DTYPE> tf;
  std::vector<int64_t> sizes(size_int32_t.begin(), size_int32_t.end());
  auto aref = exec_aten::ArrayRef<int64_t>(sizes.data(), sizes.size());
  Tensor out = tf.ones(size_int32_t);

  op_zeros_out(aref, out);

  EXPECT_TENSOR_EQ(out, tf.zeros(size_int32_t));
}

#define GENERATE_TEST(_, DTYPE)                   \
  TEST(OpZerosOutKernelTest, DTYPE##Tensors) {    \
    test_zeros_out<ScalarType::DTYPE>({2, 3, 4}); \
    test_zeros_out<ScalarType::DTYPE>({2, 0, 4}); \
    test_zeros_out<ScalarType::DTYPE>({});        \
  }

ET_FORALL_REAL_TYPES_AND(Bool, GENERATE_TEST)

TEST(OpZerosOutKernelTest, DynamicShapeUpperBoundSameAsExpected) {
  TensorFactory<ScalarType::Float> tf;
  Tensor expected = tf.zeros({3, 2});

  int64_t sizes[2] = {3, 2};
  auto sizes_aref = exec_aten::ArrayRef<int64_t>(sizes);
  Tensor out =
      tf.ones({3, 2}, torch::executor::TensorShapeDynamism::DYNAMIC_BOUND);
  op_zeros_out(sizes_aref, out);
  EXPECT_TENSOR_EQ(out, expected);
}

TEST(OpZerosOutKernelTest, DynamicShapeUpperBoundLargerThanExpected) {
  TensorFactory<ScalarType::Float> tf;
  Tensor expected = tf.zeros({3, 2});

  int64_t sizes[2] = {3, 2};
  auto sizes_aref = exec_aten::ArrayRef<int64_t>(sizes);
  Tensor out =
      tf.ones({10, 10}, torch::executor::TensorShapeDynamism::DYNAMIC_BOUND);
  op_zeros_out(sizes_aref, out);
  EXPECT_TENSOR_EQ(out, expected);
}

TEST(OpZerosOutKernelTest, DynamicShapeUnbound) {
  if (!torch::executor::testing::SupportedFeatures::get()->output_resize) {
    GTEST_SKIP() << "Dynamic shape unbound not supported";
  }
  TensorFactory<ScalarType::Float> tf;
  Tensor expected = tf.zeros({3, 2});

  int64_t sizes[2] = {3, 2};
  auto sizes_aref = exec_aten::ArrayRef<int64_t>(sizes);
  Tensor out =
      tf.ones({1, 1}, torch::executor::TensorShapeDynamism::DYNAMIC_UNBOUND);
  op_zeros_out(sizes_aref, out);
  EXPECT_TENSOR_EQ(out, expected);
}
