/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <gtest/gtest.h>

#include <executorch/kernels/test/TestUtil.h>
#include <executorch/runtime/core/evalue.h>
#include <executorch/runtime/core/exec_aten/exec_aten.h>
#include <executorch/runtime/core/exec_aten/testing_util/tensor_factory.h>
#include <executorch/runtime/core/exec_aten/testing_util/tensor_util.h>
#include <executorch/runtime/core/exec_aten/util/tensor_util.h>
#include <executorch/runtime/kernel/kernel_runtime_context.h>
#include <executorch/runtime/kernel/operator_registry.h>
#include <executorch/runtime/platform/runtime.h>
#include <cstdint>
#include <cstdio>

using executorch::aten::SizesType;
using torch::executor::Error;
using torch::executor::resize_tensor;

namespace torch {
namespace executor {

class RegisterPrimOpsTest : public OperatorTest {
 protected:
  void SetUp() override {
    context_ = KernelRuntimeContext();
  }
};

TEST_F(RegisterPrimOpsTest, OpRegistered) {
  EXPECT_TRUE(hasOpsFn("aten::sym_size.int"));
  EXPECT_TRUE(hasOpsFn("aten::sym_numel"));
  EXPECT_TRUE(hasOpsFn("executorch_prim::sym_max.Scalar"));
  EXPECT_TRUE(hasOpsFn("executorch_prim::sym_min.Scalar"));
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

  getOpsFn("aten::sym_size.int")(context_, stack);

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

  getOpsFn("aten::sym_numel")(context_, stack);

  int64_t expected = 15;
  EXPECT_EQ(stack[1]->toInt(), expected);
}

TEST_F(RegisterPrimOpsTest, SymMaxReturnsCorrectValue) {
  EValue values[3];
  int64_t a = 5;
  int64_t b = 3;
  int64_t out = 0;
  values[0] = EValue(a);
  values[1] = EValue(b);
  values[2] = EValue(out);

  EValue* stack[3];
  for (size_t i = 0; i < 3; i++) {
    stack[i] = &values[i];
  }

  getOpsFn("executorch_prim::sym_max.Scalar")(context_, stack);
  EXPECT_EQ(stack[2]->toInt(), 5);

  // Test with swapped values
  values[0] = EValue(b);
  values[1] = EValue(a);
  values[2] = EValue(out);
  getOpsFn("executorch_prim::sym_max.Scalar")(context_, stack);
  EXPECT_EQ(stack[2]->toInt(), 5);

  // Test with equal values
  values[0] = EValue(a);
  values[1] = EValue(a);
  values[2] = EValue(out);
  getOpsFn("executorch_prim::sym_max.Scalar")(context_, stack);
  EXPECT_EQ(stack[2]->toInt(), 5);

  // Test with negative values
  a = -2;
  b = -5;
  values[0] = EValue(a);
  values[1] = EValue(b);
  values[2] = EValue(out);
  getOpsFn("executorch_prim::sym_max.Scalar")(context_, stack);
  EXPECT_EQ(stack[2]->toInt(), -2);
}

TEST_F(RegisterPrimOpsTest, SymMinReturnsCorrectValue) {
  EValue values[3];
  int64_t a = 5;
  int64_t b = 3;
  int64_t out = 0;
  values[0] = EValue(a);
  values[1] = EValue(b);
  values[2] = EValue(out);

  EValue* stack[3];
  for (size_t i = 0; i < 3; i++) {
    stack[i] = &values[i];
  }

  getOpsFn("executorch_prim::sym_min.Scalar")(context_, stack);
  EXPECT_EQ(stack[2]->toInt(), 3);

  // Test with swapped values
  values[0] = EValue(b);
  values[1] = EValue(a);
  values[2] = EValue(out);
  getOpsFn("executorch_prim::sym_min.Scalar")(context_, stack);
  EXPECT_EQ(stack[2]->toInt(), 3);

  // Test with equal values
  values[0] = EValue(a);
  values[1] = EValue(a);
  values[2] = EValue(out);
  getOpsFn("executorch_prim::sym_min.Scalar")(context_, stack);
  EXPECT_EQ(stack[2]->toInt(), 5);

  // Test with negative values
  a = -2;
  b = -5;
  values[0] = EValue(a);
  values[1] = EValue(b);
  values[2] = EValue(out);
  getOpsFn("executorch_prim::sym_min.Scalar")(context_, stack);
  EXPECT_EQ(stack[2]->toInt(), -5);
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

  EValue* stack2[2] = {&values[0], &values[1]};

  getOpsFn("executorch_prim::add.Scalar")(context_, stack);
  EXPECT_EQ(stack[2]->toInt(), 7);

  getOpsFn("executorch_prim::sub.Scalar")(context_, stack);
  EXPECT_EQ(stack[2]->toInt(), -1);

  getOpsFn("executorch_prim::mul.Scalar")(context_, stack);
  EXPECT_EQ(stack[2]->toInt(), 12);

  getOpsFn("executorch_prim::floordiv.Scalar")(context_, stack);
  EXPECT_EQ(stack[2]->toInt(), 0);

  getOpsFn("executorch_prim::truediv.Scalar")(context_, stack);
  EXPECT_FLOAT_EQ(stack[2]->toDouble(), 0.75);

  getOpsFn("executorch_prim::mod.int")(context_, stack);
  EXPECT_EQ(stack[2]->toInt(), 3);

  getOpsFn("executorch_prim::mod.Scalar")(context_, stack);
  EXPECT_EQ(stack[2]->toInt(), 3);

  getOpsFn("executorch_prim::sym_float.Scalar")(context_, stack2);
  EXPECT_FLOAT_EQ(stack[1]->toDouble(), 3.0);
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
  SizesType expected_output_size[2] = {0, 2};
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
  getOpsFn("executorch_prim::et_copy_index.tensor")(context_, stack);

  EXPECT_EQ(copy_to.sizes()[0], 1);
  EXPECT_EQ(copy_to.sizes()[1], 2);
  EXPECT_TENSOR_EQ(copy_to, tf.make({1, 2}, {3, 4}));

  values[1] = tf.make({2}, {5, 6});
  values[2] = EValue((int64_t)1);
  // Copy to the next index, 1.
  getOpsFn("executorch_prim::et_copy_index.tensor")(context_, stack);

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
  ET_EXPECT_KERNEL_FAILURE(
      context_,
      getOpsFn("executorch_prim::et_copy_index.tensor")(context_, stack));
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
  getOpsFn("executorch_prim::et_copy_index.tensor")(context_, stack);
  EXPECT_EQ(copy_to.sizes()[0], 2);
  EXPECT_EQ(copy_to.sizes()[1], 2);
  EXPECT_TENSOR_EQ(copy_to, tf.make({2, 2}, {1, 2, 5, 6}));

#ifndef USE_ATEN_LIB
  // Copy and replace at index 2. This should trigger an EXPECT
  // in lean mode.
  index = 2;
  values[2] = EValue(index);
  ET_EXPECT_DEATH(
      getOpsFn("executorch_prim::et_copy_index.tensor")(context_, stack), "");
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

  getOpsFn("executorch_prim::ge.Scalar")(context_, stack);
  EXPECT_EQ(stack[2]->toBool(), false);

  getOpsFn("executorch_prim::gt.Scalar")(context_, stack);
  EXPECT_EQ(stack[2]->toBool(), false);

  getOpsFn("executorch_prim::le.Scalar")(context_, stack);
  EXPECT_EQ(stack[2]->toBool(), true);

  getOpsFn("executorch_prim::lt.Scalar")(context_, stack);
  EXPECT_EQ(stack[2]->toBool(), true);

  getOpsFn("executorch_prim::eq.Scalar")(context_, stack);
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

  getOpsFn("aten::_local_scalar_dense")(context_, stack);

  int64_t expected = 1;
  EXPECT_EQ(stack[1]->toInt(), expected);
}

TEST_F(RegisterPrimOpsTest, NegScalarReturnsCorrectValue) {
  EValue values[2];

  // Test with float
  values[0] = EValue(5.0f);
  values[1] = EValue(0.0f);

  EValue* stack[2];
  for (size_t i = 0; i < 2; i++) {
    stack[i] = &values[i];
  }

  getOpsFn("executorch_prim::neg.Scalar")(context_, stack);

  EXPECT_EQ(stack[1]->toDouble(), -5.0f);

  // Test with int
  int64_t a = 5;
  int64_t b = 0;
  values[0] = EValue(a);
  values[1] = EValue(b);

  getOpsFn("executorch_prim::neg.Scalar")(context_, stack);

  EXPECT_EQ(stack[1]->toInt(), -5l);
}

TEST_F(RegisterPrimOpsTest, TestNegScalarWithTensorFails) {
  testing::TensorFactory<ScalarType::Int> tf;

  EValue values[2];

  auto tensor = tf.make({2, 3}, {1, 2, 3, 4, 5, 6});

  int64_t zero = 0;
  values[0] = EValue(tensor);
  values[1] = EValue(zero);

  EValue* stack[2];
  for (size_t i = 0; i < 2; i++) {
    stack[i] = &values[i];
  }

  // Try to negate a tensor, which should cause a runtime error.
  ET_EXPECT_KERNEL_FAILURE(
      context_, getOpsFn("executorch_prim::neg.Scalar")(context_, stack));
}

TEST_F(RegisterPrimOpsTest, TestETView) {
  EXPECT_TRUE(hasOpsFn("executorch_prim::et_view.default"));

  testing::TensorFactory<ScalarType::Int> tf;

  // ***************************************************************************
  // Make self for tests
  // ***************************************************************************
  auto self = tf.make({3, 2}, {1, 2, 3, 4, 5, 6});
  auto self_evalue = EValue(self);

  // ***************************************************************************
  // Make size for tests
  // ***************************************************************************
  int64_t size[3] = {1, 3, -1};
  EValue size_as_evals[3] = {EValue(size[0]), EValue(size[1]), EValue(size[2])};
  EValue* size_wrapped_vals[3] = {
      &size_as_evals[0], &size_as_evals[1], &size_as_evals[2]};
  int64_t size_unwrapped_vals[3] = {0, 0, 0};
  BoxedEvalueList<int64_t> size_boxed_list(
      size_wrapped_vals, size_unwrapped_vals, 3);
  EValue size_int_list_evalue = EValue(&size_boxed_list);

  int64_t bad_size1[3] = {-1, 3, -1}; // two inferred dimensions
  EValue bad_size_as_evals1[3] = {
      EValue(bad_size1[0]), EValue(bad_size1[1]), EValue(bad_size1[2])};
  EValue* bad_size_wrapped_vals1[3] = {
      &bad_size_as_evals1[0], &bad_size_as_evals1[1], &bad_size_as_evals1[2]};
  int64_t bad_size_unwrapped_vals1[3] = {0, 0, 0};
  BoxedEvalueList<int64_t> bad_size_boxed_list1(
      bad_size_wrapped_vals1, bad_size_unwrapped_vals1, 3);
  EValue bad_size_int_list_evalue1 = EValue(&bad_size_boxed_list1);

  int64_t bad_size2[3] = {-2, -3, 1}; // negative size not supported
  EValue bad_size_as_evals2[3] = {
      EValue(bad_size2[0]), EValue(bad_size2[1]), EValue(bad_size2[2])};
  EValue* bad_size_wrapped_vals2[3] = {
      &bad_size_as_evals2[0], &bad_size_as_evals2[1], &bad_size_as_evals2[2]};
  int64_t bad_size_unwrapped_vals2[3] = {0, 0, 0};
  BoxedEvalueList<int64_t> bad_size_boxed_list2(
      bad_size_wrapped_vals2, bad_size_unwrapped_vals2, 3);
  EValue bad_size_int_list_evalue2 = EValue(&bad_size_boxed_list2);

  // ***************************************************************************
  // Make outs for tests
  // ***************************************************************************
  constexpr int N_GOOD_OUTS = 2;
  Tensor good_outs[N_GOOD_OUTS] = {
      tf.ones({1, 3, 2}), // correct size with nullptr
      tf.ones({1, 3, 2}), // correct size with self data_ptr
  };
  internal::reset_data_ptr(good_outs[0]);
  ET_CHECK(
      internal::set_tensor_data(
          good_outs[1], self.mutable_data_ptr(), good_outs[1].nbytes()) ==
      Error::Ok);
  EValue good_out_evalues[N_GOOD_OUTS] = {
      EValue(good_outs[0]), EValue(good_outs[1])};

  // bad outs expect death
  constexpr int N_BAD_OUTS = 2;
  Tensor bad_outs[N_BAD_OUTS] = {
      tf.ones({1, 3, 2, 1}), // wrong rank
      tf.ones({1, 3, 3}) // wrong size
  };
  EValue bad_out_evalues[N_BAD_OUTS] = {
      EValue(bad_outs[0]), EValue(bad_outs[1])};

  // ***************************************************************************
  // Run tests
  // ***************************************************************************

  constexpr int N_BAD_STACKS = N_BAD_OUTS + 2;
  EValue* bad_stacks[N_BAD_STACKS][3] = {
      // Bad out stacks
      {&self_evalue, &size_int_list_evalue, &bad_out_evalues[0]},
      {&self_evalue, &size_int_list_evalue, &bad_out_evalues[1]},
      // Bad size stacks
      {&self_evalue, &bad_size_int_list_evalue1, &good_out_evalues[0]},
      {&self_evalue, &bad_size_int_list_evalue2, &good_out_evalues[0]}};

  // Bad stacks expect death
  for (int i = 0; i < N_BAD_STACKS; i++) {
    ET_EXPECT_KERNEL_FAILURE(
        context_,
        getOpsFn("executorch_prim::et_view.default")(context_, bad_stacks[i]));
  }

  constexpr int N_GOOD_STACKS = N_GOOD_OUTS;
  EValue* good_out_stacks[N_GOOD_STACKS][3] = {
      {&self_evalue, &size_int_list_evalue, &good_out_evalues[0]},
      {&self_evalue, &size_int_list_evalue, &good_out_evalues[1]}};

  // Good outs expect no death and correct output
  for (int i = 0; i < N_GOOD_STACKS; i++) {
    getOpsFn("executorch_prim::et_view.default")(context_, good_out_stacks[i]);
    EXPECT_TENSOR_EQ(good_outs[i], tf.make({1, 3, 2}, {1, 2, 3, 4, 5, 6}));
    EXPECT_EQ(good_outs[i].const_data_ptr(), self.const_data_ptr());
  }
}

TEST_F(RegisterPrimOpsTest, TestETViewDynamic) {
  testing::TensorFactory<ScalarType::Int> tf;

  auto self = tf.make({3, 1}, {1, 2, 3});
  auto self_evalue = EValue(self);

  int64_t size[3] = {1, 3, -1}; // inferred size should be {1, 3, 1}
  // Construct the size as an EValue int_list
  EValue size_as_evals[3] = {EValue(size[0]), EValue(size[1]), EValue(size[2])};
  EValue* size_wrapped_vals[3] = {
      &size_as_evals[0], &size_as_evals[1], &size_as_evals[2]};
  int64_t size_unwrapped_vals[3] = {0, 0, 0};
  BoxedEvalueList<int64_t> size_boxed_list_2(
      size_wrapped_vals, size_unwrapped_vals, 3);
  EValue size_int_list_evalue = EValue(&size_boxed_list_2);

#ifdef USE_ATEN_LIB
  // ATen mode tensors don't need dynamism specification.
  auto out = tf.make({3, 2, 1}, {0, 0, 0, 0, 0, 0});
#else
  auto out = tf.make(
      {3, 2, 1}, {0, 0, 0, 0, 0, 0}, {}, TensorShapeDynamism::DYNAMIC_BOUND);
#endif

  internal::reset_data_ptr(out);
  EValue out_evalue = EValue(out);

  EValue* stack[3] = {&self_evalue, &size_int_list_evalue, &out_evalue};

  getOpsFn("executorch_prim::et_view.default")(context_, stack);

  EXPECT_TENSOR_EQ(out, tf.make({1, 3, 1}, {1, 2, 3}));
  EXPECT_EQ(out.const_data_ptr(), self.const_data_ptr());
}

TEST_F(RegisterPrimOpsTest, TestETViewEmpty) {
  testing::TensorFactory<ScalarType::Int> tf;

  auto self = tf.make({3, 1, 0}, {});
  auto self_evalue = EValue(self);
  EXPECT_EQ(self.const_data_ptr(), nullptr); // empty tensor has null data

  // Construct the sizes
  int64_t size[3] = {3, 1, -1};
  EValue size_as_evals[3] = {EValue(size[0]), EValue(size[1]), EValue(size[2])};
  EValue* size_wrapped_vals[3] = {
      &size_as_evals[0], &size_as_evals[1], &size_as_evals[2]};
  int64_t size_unwrapped_vals[3] = {0, 0, 0};
  BoxedEvalueList<int64_t> size_boxed_list_3(
      size_wrapped_vals, size_unwrapped_vals, 3);
  EValue size_int_list_evalue = EValue(&size_boxed_list_3);

  int64_t bad_size[3] = {0, 1, -1}; // bad size: cannot infer with 0
  EValue bad_size_as_evals[3] = {
      EValue(bad_size[0]), EValue(bad_size[1]), EValue(bad_size[2])};
  EValue* bad_size_wrapped_vals[3] = {
      &bad_size_as_evals[0], &bad_size_as_evals[1], &bad_size_as_evals[2]};
  int64_t bad_size_unwrapped_vals[3] = {0, 0, 0};
  BoxedEvalueList<int64_t> bad_size_boxed_list(
      bad_size_wrapped_vals, bad_size_unwrapped_vals, 3);
  EValue bad_size_int_list_evalue = EValue(&bad_size_boxed_list);

  auto out = tf.make({3, 1, 0}, {}, {});
  EValue out_evalue = EValue(out);
  EXPECT_EQ(out.const_data_ptr(), nullptr);

  // good size test
  EValue* stack[3] = {&self_evalue, &size_int_list_evalue, &out_evalue};
  getOpsFn("executorch_prim::et_view.default")(context_, stack);
  EXPECT_TENSOR_EQ(out, tf.make({3, 1, 0}, {}));
  EXPECT_EQ(out.const_data_ptr(), self.const_data_ptr());

  // bad size test
  EValue* bad_stack[3] = {&self_evalue, &bad_size_int_list_evalue, &out_evalue};
  ET_EXPECT_KERNEL_FAILURE(
      context_,
      getOpsFn("executorch_prim::et_view.default")(context_, bad_stack));
}

TEST_F(RegisterPrimOpsTest, TestCeil) {
  std::array<double, 10> inputs = {
      0.0, 0.25, 0.5, 0.75, 1.0, 1.75, -0.5, -1.0, -1.5, 9.999999};
  std::array<int64_t, 10> expected = {0, 1, 1, 1, 1, 2, 0, -1, -1, 10};

  for (auto i = 0; i < inputs.size(); i++) {
    EValue values[2];
    values[0] = EValue(inputs[i]);
    values[1] = EValue(0.0);

    EValue* stack[2];
    for (size_t j = 0; j < 2; j++) {
      stack[j] = &values[j];
    }

    getOpsFn("executorch_prim::ceil.Scalar")(context_, stack);
    EXPECT_EQ(stack[1]->toInt(), expected[i]);
  }
}

TEST_F(RegisterPrimOpsTest, TestRound) {
  // Note that Python uses round-to-even for halfway values.
  std::array<double, 10> inputs = {
      0.0, 0.25, 0.5, 0.75, 1.0, 1.5, -0.5, -1.0, -1.5, 9.999999};
  std::array<int64_t, 10> expected = {0, 0, 0, 1, 1, 2, 0, -1, -2, 10};

  for (auto i = 0; i < inputs.size(); i++) {
    EValue values[2];
    values[0] = EValue(inputs[i]);
    values[1] = EValue(0.0);

    EValue* stack[2];
    for (size_t j = 0; j < 2; j++) {
      stack[j] = &values[j];
    }

    getOpsFn("executorch_prim::round.Scalar")(context_, stack);
    EXPECT_EQ(stack[1]->toInt(), expected[i]);
  }
}

TEST_F(RegisterPrimOpsTest, TestTrunc) {
  std::array<double, 10> inputs = {
      0.0, 0.25, 0.5, 0.75, 1.0, 1.75, -0.5, -1.0, -1.5, 9.999999};
  std::array<int64_t, 10> expected = {0, 0, 0, 0, 1, 1, 0, -1, -1, 9};

  for (auto i = 0; i < inputs.size(); i++) {
    EValue values[2];
    values[0] = EValue(inputs[i]);
    values[1] = EValue(0.0);

    EValue* stack[2];
    for (size_t j = 0; j < 2; j++) {
      stack[j] = &values[j];
    }

    getOpsFn("executorch_prim::trunc.Scalar")(context_, stack);
    EXPECT_EQ(stack[1]->toInt(), expected[i]);
  }
}

// Test that each prim op returns InvalidProgram error when given a stack that's
// one element shorter than expected
TEST_F(RegisterPrimOpsTest, TestInvalidProgramErrorOnShortStack) {
  // Test aten::sym_size.int with a stack of size 2 (missing output)
  {
    testing::TensorFactory<ScalarType::Int> tf;
    Tensor self_tensor = tf.ones({3, 5});
    EValue values[2];
    int64_t dim = 1;
    values[0] = EValue(self_tensor);
    values[1] = EValue(dim);

    EValue* stack[2];
    for (size_t i = 0; i < 2; i++) {
      stack[i] = &values[i];
    }

    ET_EXPECT_KERNEL_FAILURE(
        context_, getOpsFn("aten::sym_size.int")(context_, stack));
    EXPECT_EQ(context_.failure_state(), torch::executor::Error::InvalidProgram);
  }

  // Test aten::sym_numel with a stack of size 1 (missing output)
  {
    testing::TensorFactory<ScalarType::Int> tf;
    Tensor self_tensor = tf.ones({3, 5});
    EValue values[1];
    values[0] = EValue(self_tensor);

    EValue* stack[1];
    stack[0] = &values[0];

    ET_EXPECT_KERNEL_FAILURE(
        context_, getOpsFn("aten::sym_numel")(context_, stack));
    EXPECT_EQ(context_.failure_state(), torch::executor::Error::InvalidProgram);
  }

  // Test executorch_prim::sym_max.Scalar with a stack of size 2 (missing
  // output)
  {
    EValue values[2];
    int64_t a = 5;
    int64_t b = 3;
    values[0] = EValue(a);
    values[1] = EValue(b);

    EValue* stack[2];
    for (size_t i = 0; i < 2; i++) {
      stack[i] = &values[i];
    }

    ET_EXPECT_KERNEL_FAILURE(
        context_, getOpsFn("executorch_prim::sym_max.Scalar")(context_, stack));
    EXPECT_EQ(context_.failure_state(), Error::InvalidProgram);
  }

  // Test executorch_prim::sym_min.Scalar with a stack of size 2 (missing
  // output)
  {
    EValue values[2];
    int64_t a = 5;
    int64_t b = 3;
    values[0] = EValue(a);
    values[1] = EValue(b);

    EValue* stack[2];
    for (size_t i = 0; i < 2; i++) {
      stack[i] = &values[i];
    }

    ET_EXPECT_KERNEL_FAILURE(
        context_, getOpsFn("executorch_prim::sym_min.Scalar")(context_, stack));
    EXPECT_EQ(context_.failure_state(), Error::InvalidProgram);
  }

  // Test algebra ops with a stack of size 2 (missing output)
  {
    EValue values[2];
    int64_t a = 3;
    int64_t b = 4;
    values[0] = EValue(a);
    values[1] = EValue(b);

    EValue* stack[2];
    for (size_t i = 0; i < 2; i++) {
      stack[i] = &values[i];
    }

    ET_EXPECT_KERNEL_FAILURE(
        context_, getOpsFn("executorch_prim::add.Scalar")(context_, stack));
    EXPECT_EQ(context_.failure_state(), Error::InvalidProgram);

    ET_EXPECT_KERNEL_FAILURE(
        context_, getOpsFn("executorch_prim::sub.Scalar")(context_, stack));
    EXPECT_EQ(context_.failure_state(), Error::InvalidProgram);

    ET_EXPECT_KERNEL_FAILURE(
        context_, getOpsFn("executorch_prim::mul.Scalar")(context_, stack));
    EXPECT_EQ(context_.failure_state(), Error::InvalidProgram);

    ET_EXPECT_KERNEL_FAILURE(
        context_,
        getOpsFn("executorch_prim::floordiv.Scalar")(context_, stack));
    EXPECT_EQ(context_.failure_state(), Error::InvalidProgram);

    ET_EXPECT_KERNEL_FAILURE(
        context_, getOpsFn("executorch_prim::truediv.Scalar")(context_, stack));
    EXPECT_EQ(context_.failure_state(), Error::InvalidProgram);

    ET_EXPECT_KERNEL_FAILURE(
        context_, getOpsFn("executorch_prim::mod.int")(context_, stack));
    EXPECT_EQ(context_.failure_state(), Error::InvalidProgram);

    ET_EXPECT_KERNEL_FAILURE(
        context_, getOpsFn("executorch_prim::mod.Scalar")(context_, stack));
    EXPECT_EQ(context_.failure_state(), Error::InvalidProgram);
  }

  // Test executorch_prim::sym_float.Scalar with a stack of size 1 (missing
  // output)
  {
    EValue values[1];
    int64_t a = 3;
    values[0] = EValue(a);

    EValue* stack[1];
    stack[0] = &values[0];

    ET_EXPECT_KERNEL_FAILURE(
        context_,
        getOpsFn("executorch_prim::sym_float.Scalar")(context_, stack));
    EXPECT_EQ(context_.failure_state(), Error::InvalidProgram);
  }

  // Test boolean ops with a stack of size 2 (missing output)
  {
    EValue values[2];
    double a = 3;
    double b = 4;
    values[0] = EValue(a);
    values[1] = EValue(b);

    EValue* stack[2];
    for (size_t i = 0; i < 2; i++) {
      stack[i] = &values[i];
    }

    ET_EXPECT_KERNEL_FAILURE(
        context_, getOpsFn("executorch_prim::ge.Scalar")(context_, stack));
    EXPECT_EQ(context_.failure_state(), Error::InvalidProgram);

    ET_EXPECT_KERNEL_FAILURE(
        context_, getOpsFn("executorch_prim::gt.Scalar")(context_, stack));
    EXPECT_EQ(context_.failure_state(), Error::InvalidProgram);

    ET_EXPECT_KERNEL_FAILURE(
        context_, getOpsFn("executorch_prim::le.Scalar")(context_, stack));
    EXPECT_EQ(context_.failure_state(), Error::InvalidProgram);

    ET_EXPECT_KERNEL_FAILURE(
        context_, getOpsFn("executorch_prim::lt.Scalar")(context_, stack));
    EXPECT_EQ(context_.failure_state(), Error::InvalidProgram);

    ET_EXPECT_KERNEL_FAILURE(
        context_, getOpsFn("executorch_prim::eq.Scalar")(context_, stack));
    EXPECT_EQ(context_.failure_state(), Error::InvalidProgram);
  }

  // Test aten::_local_scalar_dense with a stack of size 1 (missing output)
  {
    testing::TensorFactory<ScalarType::Int> tf;
    Tensor self_tensor = tf.ones({1});
    EValue values[1];
    values[0] = EValue(self_tensor);

    EValue* stack[1];
    stack[0] = &values[0];

    ET_EXPECT_KERNEL_FAILURE(
        context_, getOpsFn("aten::_local_scalar_dense")(context_, stack));
    EXPECT_EQ(context_.failure_state(), Error::InvalidProgram);
  }

  // Test executorch_prim::neg.Scalar with a stack of size 1 (missing output)
  {
    EValue values[1];
    values[0] = EValue(5.0f);

    EValue* stack[1];
    stack[0] = &values[0];

    ET_EXPECT_KERNEL_FAILURE(
        context_, getOpsFn("executorch_prim::neg.Scalar")(context_, stack));
    EXPECT_EQ(context_.failure_state(), Error::InvalidProgram);
  }

  // Test executorch_prim::et_copy_index.tensor with a stack of size 2 (missing
  // index)
  {
    testing::TensorFactory<ScalarType::Int> tf;
    auto copy_to = tf.make({2, 2}, {0, 0, 0, 0});
    auto to_copy = tf.make({2}, {3, 4});

    EValue values[2];
    values[0] = EValue(copy_to);
    values[1] = EValue(to_copy);

    EValue* stack[2];
    stack[0] = &values[0];
    stack[1] = &values[1];

    ET_EXPECT_KERNEL_FAILURE(
        context_,
        getOpsFn("executorch_prim::et_copy_index.tensor")(context_, stack));
    EXPECT_EQ(context_.failure_state(), Error::InvalidProgram);
  }

  // Test executorch_prim::et_view.default with a stack of size 2 (missing
  // output)
  {
    testing::TensorFactory<ScalarType::Int> tf;
    auto self = tf.make({3, 2}, {1, 2, 3, 4, 5, 6});
    auto self_evalue = EValue(self);

    int64_t size[3] = {1, 3, -1};
    EValue size_as_evals[3] = {
        EValue(size[0]), EValue(size[1]), EValue(size[2])};
    EValue* size_wrapped_vals[3] = {
        &size_as_evals[0], &size_as_evals[1], &size_as_evals[2]};
    int64_t size_unwrapped_vals[3] = {0, 0, 0};
    BoxedEvalueList<int64_t> size_boxed_list_4(
        size_wrapped_vals, size_unwrapped_vals, 3);
    EValue size_int_list_evalue = EValue(&size_boxed_list_4);

    EValue* stack[2] = {&self_evalue, &size_int_list_evalue};

    ET_EXPECT_KERNEL_FAILURE(
        context_,
        getOpsFn("executorch_prim::et_view.default")(context_, stack));
    EXPECT_EQ(context_.failure_state(), Error::InvalidProgram);
  }

  // Test ceil, round, trunc with a stack of size 1 (missing output)
  {
    EValue values[1];
    values[0] = EValue(5.5);

    EValue* stack[1];
    stack[0] = &values[0];

    ET_EXPECT_KERNEL_FAILURE(
        context_, getOpsFn("executorch_prim::ceil.Scalar")(context_, stack));
    EXPECT_EQ(context_.failure_state(), Error::InvalidProgram);

    ET_EXPECT_KERNEL_FAILURE(
        context_, getOpsFn("executorch_prim::round.Scalar")(context_, stack));
    EXPECT_EQ(context_.failure_state(), Error::InvalidProgram);

    ET_EXPECT_KERNEL_FAILURE(
        context_, getOpsFn("executorch_prim::trunc.Scalar")(context_, stack));
    EXPECT_EQ(context_.failure_state(), Error::InvalidProgram);
  }
}

} // namespace executor
} // namespace torch
