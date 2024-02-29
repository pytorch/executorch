/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <gtest/gtest.h>

#include <executorch/runtime/core/evalue.h>
#include <executorch/runtime/core/exec_aten/exec_aten.h>
#include <executorch/runtime/core/exec_aten/testing_util/tensor_factory.h>
#include <executorch/runtime/core/exec_aten/testing_util/tensor_util.h>
#include <executorch/runtime/core/exec_aten/util/tensor_util.h>
#include <executorch/runtime/kernel/kernel_runtime_context.h>
#include <executorch/runtime/kernel/operator_registry.h>
#include <executorch/runtime/platform/runtime.h>
#include <executorch/test/utils/DeathTest.h>

using exec_aten::SizesType;
using torch::executor::Error;
using torch::executor::resize_tensor;

namespace torch {
namespace executor {

class RegisterPrimOpsTest : public ::testing::Test {
 protected:
  RuntimeContext context;
  void SetUp() override {
    torch::executor::runtime_init();
    context = RuntimeContext();
  }
};

TEST_F(RegisterPrimOpsTest, OpRegistered) {
  EXPECT_TRUE(hasOpsFn("aten::sym_size.int"));
  EXPECT_TRUE(hasOpsFn("aten::sym_numel"));
}

TEST_F(RegisterPrimOpsTest, SymSizeReturnsCorrectValue) {
  testing::TensorFactory<ScalarType::Int> tf;

  Tensor self_tensor = tf.ones({3, 5});
  EValue values[3];
  int64_t dim = 1;
  int64_t out = 0;
  values[0] = EValue(self_tensor);
  values[1] = EValue(dim);
  values[2] = EValue(out);

  EValue* stack[3];
  for (size_t i = 0; i < 3; i++) {
    stack[i] = &values[i];
  }

  getOpsFn("aten::sym_size.int")(context, stack);

  int64_t expected = 5;
  EXPECT_EQ(stack[2]->toInt(), expected);
}

TEST_F(RegisterPrimOpsTest, SymNumelReturnsCorrectValue) {
  testing::TensorFactory<ScalarType::Int> tf;

  Tensor self_tensor = tf.ones({3, 5});
  EValue values[2];
  int64_t out = 0;
  values[0] = EValue(self_tensor);
  values[1] = EValue(out);

  EValue* stack[2];
  for (size_t i = 0; i < 2; i++) {
    stack[i] = &values[i];
  }

  getOpsFn("aten::sym_numel")(context, stack);

  int64_t expected = 15;
  EXPECT_EQ(stack[1]->toInt(), expected);
}

TEST_F(RegisterPrimOpsTest, TestAlgebraOps) {
  EValue values[3];
  int64_t a = 3;
  int64_t b = 4;
  int64_t out = 0;
  values[0] = EValue(a);
  values[1] = EValue(b);
  values[2] = EValue(out);

  EValue* stack[3];
  for (size_t i = 0; i < 3; i++) {
    stack[i] = &values[i];
  }

  getOpsFn("executorch_prim::add.Scalar")(context, stack);
  EXPECT_EQ(stack[2]->toInt(), 7);

  getOpsFn("executorch_prim::sub.Scalar")(context, stack);
  EXPECT_EQ(stack[2]->toInt(), -1);

  getOpsFn("executorch_prim::mul.Scalar")(context, stack);
  EXPECT_EQ(stack[2]->toInt(), 12);

  getOpsFn("executorch_prim::floordiv.Scalar")(context, stack);
  EXPECT_EQ(stack[2]->toInt(), 0);

  getOpsFn("executorch_prim::truediv.Scalar")(context, stack);
  EXPECT_FLOAT_EQ(stack[2]->toDouble(), 0.75);
}

TEST_F(RegisterPrimOpsTest, TestETCopyIndex) {
  EXPECT_TRUE(hasOpsFn("executorch_prim::et_copy_index.tensor"));

  int64_t index = 0;
  testing::TensorFactory<ScalarType::Int> tf;

#ifdef USE_ATEN_LIB
  // ATen mode tensors don't need dynamism specification.
  Tensor copy_to = tf.make({2, 2}, {0, 0, 0, 0});
#else
  std::vector<int> buf(4);
  SizesType expected_output_size[2] = {0, 0};
  Tensor copy_to =
      tf.make({2, 2}, {0, 0, 0, 0}, {}, TensorShapeDynamism::DYNAMIC_BOUND);
  // Resize the tensor to 0 size for the tests.
  Error err = resize_tensor(copy_to, {expected_output_size, 2});
  EXPECT_EQ(err, Error::Ok);
#endif

  Tensor to_copy = tf.make({2}, {3, 4});

  EValue values[3];
  EValue* stack[3];

  values[0] = EValue(copy_to);
  values[1] = EValue(to_copy);
  values[2] = EValue(index);

  stack[0] = &values[0];
  stack[1] = &values[1];
  stack[2] = &values[2];

  // Simple test to copy to index 0.
  getOpsFn("executorch_prim::et_copy_index.tensor")(context, stack);

  EXPECT_EQ(copy_to.sizes()[0], 1);
  EXPECT_EQ(copy_to.sizes()[1], 2);
  EXPECT_TENSOR_EQ(copy_to, tf.make({1, 2}, {3, 4}));

  values[1] = tf.make({2}, {5, 6});
  values[2] = EValue((int64_t)1);
  // Copy to the next index, 1.
  getOpsFn("executorch_prim::et_copy_index.tensor")(context, stack);

  EXPECT_EQ(copy_to.sizes()[0], 2);
  EXPECT_EQ(copy_to.sizes()[1], 2);
  EXPECT_TENSOR_EQ(copy_to, tf.make({2, 2}, {3, 4, 5, 6}));
}

TEST_F(RegisterPrimOpsTest, TestETCopyIndexMismatchShape) {
  int64_t index = 1;
  testing::TensorFactory<ScalarType::Int> tf;

  EValue values[3];
  EValue* stack[3];

  auto copy_to = tf.make({2, 3}, {1, 2, 3, 4, 5, 6});
  auto to_copy = tf.make({2}, {1, 2});

  values[0] = EValue(copy_to);
  values[1] = EValue(to_copy);
  values[2] = EValue(index);

  stack[0] = &values[0];
  stack[1] = &values[1];
  stack[2] = &values[2];

  // Try to copy and replace at index 1. This will fail because
  // copy_to.sizes[1:] and to_copy.sizes[:] don't match each other
  // which is a pre-requisite for this operator.
  ET_EXPECT_DEATH(
      getOpsFn("executorch_prim::et_copy_index.tensor")(context, stack), "");
}

TEST_F(RegisterPrimOpsTest, TestETCopyIndexStaticShape) {
  int64_t index = 1;
  testing::TensorFactory<ScalarType::Int> tf;

  EValue values[3];
  EValue* stack[3];

  // Test with static shape tensors.
  const std::vector<int> buf = {1, 2, 3, 4};
  auto copy_to = tf.make({2, 2}, buf);
  auto to_copy = tf.make({2}, {5, 6});

  values[0] = EValue(copy_to);
  values[1] = EValue(to_copy);
  values[2] = EValue(index);

  stack[0] = &values[0];
  stack[1] = &values[1];
  stack[2] = &values[2];

  // Copy and replace at index 1.
  getOpsFn("executorch_prim::et_copy_index.tensor")(context, stack);
  EXPECT_EQ(copy_to.sizes()[0], 2);
  EXPECT_EQ(copy_to.sizes()[1], 2);
  EXPECT_TENSOR_EQ(copy_to, tf.make({2, 2}, {1, 2, 5, 6}));

#ifndef USE_ATEN_LIB
  // Copy and replace at index 2. This should trigger an EXPECT
  // in lean mode.
  index = 2;
  values[2] = EValue(index);
  ET_EXPECT_DEATH(
      getOpsFn("executorch_prim::et_copy_index.tensor")(context, stack), "");
#endif
}

TEST_F(RegisterPrimOpsTest, TestBooleanOps) {
  EValue values[3];
  double a = 3;
  double b = 4;
  bool out = false;
  values[0] = EValue(a);
  values[1] = EValue(b);
  values[2] = EValue(out);

  EValue* stack[3];
  for (size_t i = 0; i < 3; i++) {
    stack[i] = &values[i];
  }

  getOpsFn("executorch_prim::ge.Scalar")(context, stack);
  EXPECT_EQ(stack[2]->toBool(), false);

  getOpsFn("executorch_prim::gt.Scalar")(context, stack);
  EXPECT_EQ(stack[2]->toBool(), false);

  getOpsFn("executorch_prim::le.Scalar")(context, stack);
  EXPECT_EQ(stack[2]->toBool(), true);

  getOpsFn("executorch_prim::lt.Scalar")(context, stack);
  EXPECT_EQ(stack[2]->toBool(), true);

  getOpsFn("executorch_prim::eq.Scalar")(context, stack);
  EXPECT_EQ(stack[2]->toBool(), false);
}

TEST_F(RegisterPrimOpsTest, LocalScalarDenseReturnsCorrectValue) {
  testing::TensorFactory<ScalarType::Int> tf;

  Tensor self_tensor = tf.ones({1});
  const int64_t num_vals = 2;
  EValue values[num_vals];
  int64_t out = 0;
  values[0] = EValue(self_tensor);
  values[1] = EValue(out);

  EValue* stack[num_vals];
  for (size_t i = 0; i < num_vals; i++) {
    stack[i] = &values[i];
  }

  getOpsFn("aten::_local_scalar_dense")(context, stack);

  int64_t expected = 1;
  EXPECT_EQ(stack[1]->toInt(), expected);
}

} // namespace executor
} // namespace torch
