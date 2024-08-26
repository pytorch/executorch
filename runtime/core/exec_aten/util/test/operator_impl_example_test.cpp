/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

/**
 * @file
 *
 * Demonstrates the implementation of a simple operator using the utilities in
 * this package.
 */

#include <executorch/runtime/core/exec_aten/testing_util/tensor_factory.h>
#include <executorch/runtime/core/exec_aten/testing_util/tensor_util.h>
#include <executorch/runtime/core/exec_aten/util/scalar_type_util.h>
#include <executorch/runtime/core/exec_aten/util/tensor_util.h>
#include <executorch/test/utils/DeathTest.h>
#include <gtest/gtest.h>

using namespace ::testing;
using exec_aten::ScalarType;
using exec_aten::Tensor;
using executorch::runtime::testing::TensorFactory;

//
// A sample implementation of a PyTorch operator using the utilities in this
// package.
//

/**
 * Adds the elements of `a` and `b`, overwriting `out`.
 *
 * Assumes that the tensors are contiguous, are the same shape, and have the
 * same dtype. CTYPE should be the C type (like `float` or `int`) that matches
 * the dtype of the tensors.
 */
template <class CTYPE>
void add_tensors_impl(const Tensor& a, const Tensor& b, Tensor& out) {
  ET_DCHECK(a.numel() == b.numel() && b.numel() == out.numel());
  const size_t n = a.numel();
  const auto data_a = a.const_data_ptr<CTYPE>();
  const auto data_b = b.const_data_ptr<CTYPE>();
  auto data_out = out.mutable_data_ptr<CTYPE>();
  for (size_t i = 0; i < n; ++i) {
    data_out[i] = data_a[i] + data_b[i];
  }
}

/**
 * Element-wise sum of `a` and `b`, overwriting `out`.
 *
 * Asserts if the tensors are not all the same dtype and shape.
 */
Tensor& add_tensors_op(const Tensor& a, const Tensor& b, Tensor& out) {
  ET_CHECK_SAME_SHAPE_AND_DTYPE3(a, b, out);

// ET_FORALL_REAL_TYPES() will call this macro for every ScalarType backed by a
// real-number C type. `ctype` will be that C type (e.g. `float`), and `dtype`
// will be the unqualified ScalarType enumeration name (e.g., `Float`).
//
// Use this to create a switch statement that dispatches to the impl template
// based on the types of the provided tensors.
#define ADD_TENSORS(ctype, dtype)       \
  case ScalarType::dtype:               \
    add_tensors_impl<ctype>(a, b, out); \
    break;

  switch (a.scalar_type()) {
    ET_FORALL_REAL_TYPES(ADD_TENSORS)
    default:
      ET_CHECK_MSG(false, "Unhandled dtype %hhd", a.scalar_type());
  }

#undef ADD_TENSORS

  return out;
}

//
// Some basic unit tests demonstrating that the operator implementation works as
// expected.
//

TEST(OperatorImplExampleTest, AddIntTensors) {
  TensorFactory<ScalarType::Int> tf;

  const std::vector<int32_t> sizes = {2, 2};

  // Destination for the sum.
  Tensor out = tf.zeros(sizes);

  // Add two tensors.
  add_tensors_op(tf.make(sizes, /*data=*/{1, 2, 4, 8}), tf.ones(sizes), out);

  // Check that it matches the expected output.
  EXPECT_TENSOR_EQ(out, tf.make(sizes, /*data=*/{2, 3, 5, 9}));
}

TEST(OperatorImplExampleTest, AddDoubleTensors) {
  TensorFactory<ScalarType::Double> tf;

  const std::vector<int32_t> sizes = {2, 2};

  // Destination for the sum.
  Tensor out = tf.zeros(sizes);

  // Add two tensors.
  add_tensors_op(
      tf.make(sizes, /*data=*/{1.1, 2.2, 4.4, 8.8}), tf.ones(sizes), out);

  // Check that it matches the expected output.
  EXPECT_TENSOR_CLOSE(out, tf.make(sizes, /*data=*/{2.1, 3.2, 5.4, 9.8}));
}

TEST(OperatorImplExampleTest, UnhandledDtypeDies) {
  // add_tensors_op() doesn't handle Bool.
  TensorFactory<ScalarType::Bool> tf;

  const std::vector<int32_t> sizes = {2, 2};

  // Addends.
  Tensor a = tf.make(sizes, /*data=*/{false, true, false, true});
  Tensor b = tf.make(sizes, /*data=*/{true, false, true, false});

  // Destination for the sum.
  Tensor out = tf.zeros(sizes);

  // Adding the two boolean tensors should cause an assertion and kill the
  // test process.
  ET_EXPECT_DEATH(add_tensors_op(a, b, out), "");
}

TEST(OpAddOutKernelTest, MismatchedInputDimsDies) {
  TensorFactory<ScalarType::Int> tf;

  // Addends with the same number of elements but different dimensions.
  Tensor a = tf.ones(/*sizes=*/{4});
  Tensor b = tf.ones(/*sizes=*/{2, 2});

  // Destination for the sum; matches the shape of one of the inputs.
  Tensor out = tf.zeros(/*sizes=*/{4});

  // Adding the two mismatched tensors should cause an assertion and kill the
  // test process.
  ET_EXPECT_DEATH(add_tensors_op(a, b, out), "");
}

TEST(OpAddOutKernelTest, MismatchedInputDtypesDies) {
  TensorFactory<ScalarType::Int> tf_int;
  TensorFactory<ScalarType::Float> tf_float;

  const std::vector<int32_t> sizes = {2, 2};

  // Addends with the same shape but different dtypes.
  Tensor a = tf_int.ones(sizes);
  Tensor b = tf_float.ones(sizes);

  // Destination for the sum; matches the dtype of one of the inputs.
  Tensor out = tf_float.zeros(sizes);

  // Adding the two mismatched tensors should cause an assertion and kill the
  // test process.
  ET_EXPECT_DEATH(add_tensors_op(a, b, out), "");
}

TEST(OpAddOutKernelTest, MixingUnhandledDtypeDies) {
  TensorFactory<ScalarType::Int> tf_int;
  TensorFactory<ScalarType::Bool> tf_bool;

  const std::vector<int32_t> sizes = {2, 2};

  // Addends with the same shape but different dtypes, one of which (bool) is
  // not supported by add_tensors_op().
  Tensor a = tf_int.ones(sizes);
  Tensor b = tf_bool.ones(sizes);

  // Destination for the sum; matches the input with a valid dtype.
  Tensor out = tf_int.zeros(sizes);

  // Adding the two mismatched tensors should cause an assertion and kill the
  // test process.
  ET_EXPECT_DEATH(add_tensors_op(a, b, out), "");
}
