/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <executorch/kernels/optimized/NativeFunctions.h>
#include <executorch/kernels/test/TestUtil.h>
#include <executorch/runtime/core/exec_aten/exec_aten.h>
#include <executorch/runtime/core/exec_aten/testing_util/tensor_factory.h>
#include <executorch/runtime/core/exec_aten/testing_util/tensor_util.h>

#include <gtest/gtest.h>

#include <vector>

using executorch::aten::ScalarType;
using executorch::aten::Tensor;
using torch::executor::Error;
using torch::executor::KernelRuntimeContext;
using torch::executor::TensorShapeDynamism;
using torch::executor::native::opt_add_out;
using torch::executor::native::opt_div_out;
using torch::executor::native::opt_mul_out;
using torch::executor::native::opt_sub_out;
using torch::executor::testing::TensorFactory;

// Calls the optimized kernels directly, since portable sub lacks complex
// support and portable div lacks complex broadcasting. Complex add and mul
// broadcasting are covered by the shared op tests. The sub helper uses a
// non-unit alpha.
class ComplexBinaryBroadcastTest : public OperatorTest {
 protected:
  // Outputs are taken by value so callers can pass freshly made tensors.
  Tensor sub_out(const Tensor& a, const Tensor& b, Tensor out) {
    return opt_sub_out(context_, a, b, /*alpha=*/2, out);
  }
  Tensor div_out(const Tensor& a, const Tensor& b, Tensor out) {
    return opt_div_out(context_, a, b, out);
  }

  // Uses a fresh context per op so that one rejection cannot mask another.
  void expect_rejected(const Tensor& a, const Tensor& b, Tensor out) {
    KernelRuntimeContext add_context, sub_context, mul_context, div_context;
    opt_add_out(add_context, a, b, /*alpha=*/2, out);
    opt_sub_out(sub_context, a, b, /*alpha=*/2, out);
    opt_mul_out(mul_context, a, b, out);
    opt_div_out(div_context, a, b, out);
    EXPECT_NE(add_context.failure_state(), Error::Ok);
    EXPECT_NE(sub_context.failure_state(), Error::Ok);
    EXPECT_NE(mul_context.failure_state(), Error::Ok);
    EXPECT_NE(div_context.failure_state(), Error::Ok);
  }

  template <typename CTYPE, ScalarType DTYPE>
  void test_complex_sub_broadcast() {
    TensorFactory<DTYPE> tf;
    // Bounded outputs are resized to the broadcast shape.
    const auto make_out = [&tf](const std::vector<int32_t>& sizes) {
      return tf.full(sizes, CTYPE(0, 0), TensorShapeDynamism::DYNAMIC_BOUND);
    };

    // Values from issue #24408, with zero-dimensional and single-element
    // scalars on either side.
    const Tensor x = tf.make({2}, {CTYPE(1, 1), CTYPE(2, -2)});
    for (const auto& scalar_sizes :
         {std::vector<int32_t>{}, std::vector<int32_t>{1}}) {
      const Tensor s = tf.make(scalar_sizes, {CTYPE(2, 2)});
      EXPECT_TENSOR_EQ(
          sub_out(x, s, make_out({4})),
          tf.make({2}, {CTYPE(-3, -3), CTYPE(-2, -6)}));
      EXPECT_TENSOR_EQ(
          sub_out(s, x, make_out({4})),
          tf.make({2}, {CTYPE(0, 0), CTYPE(-2, 6)}));
    }

    // Both operands are scalars.
    EXPECT_TENSOR_EQ(
        sub_out(
            tf.make({}, {CTYPE(5, 1)}),
            tf.make({1}, {CTYPE(1, 1)}),
            make_out({1})),
        tf.make({1}, {CTYPE(3, -1)}));

    // Both operands are broadcast.
    EXPECT_TENSOR_EQ(
        sub_out(
            tf.make({2, 1}, {CTYPE(5, 1), CTYPE(3, 5)}),
            tf.make({1, 2}, {CTYPE(1, 1), CTYPE(1, -1)}),
            make_out({2, 2})),
        tf.make({2, 2}, {CTYPE(3, -1), CTYPE(3, 3), CTYPE(1, 3), CTYPE(1, 7)}));

    // Empty output.
    const Tensor empty = tf.make({0}, {});
    EXPECT_TENSOR_EQ(
        sub_out(empty, tf.make({}, {CTYPE(2, 2)}), make_out({0})), empty);
  }

  template <typename CTYPE, ScalarType DTYPE>
  void test_complex_div_broadcast() {
    TensorFactory<DTYPE> tf;
    // Bounded outputs are resized to the broadcast shape.
    const auto make_out = [&tf](const std::vector<int32_t>& sizes) {
      return tf.full(sizes, CTYPE(0, 0), TensorShapeDynamism::DYNAMIC_BOUND);
    };

    // Values from issue #24408, with zero-dimensional and single-element
    // scalars on either side.
    const Tensor x = tf.make({2}, {CTYPE(1, 1), CTYPE(2, -2)});
    for (const auto& scalar_sizes :
         {std::vector<int32_t>{}, std::vector<int32_t>{1}}) {
      const Tensor s = tf.make(scalar_sizes, {CTYPE(2, 2)});
      EXPECT_TENSOR_EQ(
          div_out(x, s, make_out({4})),
          tf.make({2}, {CTYPE(0.5f, 0), CTYPE(0, -1)}));
      EXPECT_TENSOR_EQ(
          div_out(s, x, make_out({4})),
          tf.make({2}, {CTYPE(2, 0), CTYPE(0, 1)}));
    }

    // Both operands are scalars.
    EXPECT_TENSOR_EQ(
        div_out(
            tf.make({}, {CTYPE(5, 1)}),
            tf.make({1}, {CTYPE(1, 1)}),
            make_out({1})),
        tf.make({1}, {CTYPE(3, -2)}));

    // Both operands are broadcast.
    EXPECT_TENSOR_EQ(
        div_out(
            tf.make({2, 1}, {CTYPE(5, 1), CTYPE(3, 5)}),
            tf.make({1, 2}, {CTYPE(1, 1), CTYPE(1, -1)}),
            make_out({2, 2})),
        tf.make(
            {2, 2}, {CTYPE(3, -2), CTYPE(2, 3), CTYPE(4, 1), CTYPE(-1, 4)}));

    // Empty output.
    const Tensor empty = tf.make({0}, {});
    EXPECT_TENSOR_EQ(
        div_out(empty, tf.make({}, {CTYPE(2, 2)}), make_out({0})), empty);
  }
};

TEST_F(ComplexBinaryBroadcastTest, ComplexSubBroadcast) {
#define TEST_ENTRY(ctype, dtype) \
  test_complex_sub_broadcast<ctype, ScalarType::dtype>();
  ET_FORALL_COMPLEXH_TYPES(TEST_ENTRY);
#undef TEST_ENTRY
}

TEST_F(ComplexBinaryBroadcastTest, ComplexDivBroadcast) {
#define TEST_ENTRY(ctype, dtype) \
  test_complex_div_broadcast<ctype, ScalarType::dtype>();
  ET_FORALL_COMPLEX_TYPES(TEST_ENTRY);
#undef TEST_ENTRY
}

TEST_F(ComplexBinaryBroadcastTest, RejectsInvalidComplexInputs) {
  TensorFactory<ScalarType::Float> tf_float;
  TensorFactory<ScalarType::ComplexFloat> tf_cfloat;
  TensorFactory<ScalarType::ComplexDouble> tf_cdouble;
  using CFloat = TensorFactory<ScalarType::ComplexFloat>::ctype;
  using CDouble = TensorFactory<ScalarType::ComplexDouble>::ctype;

  const Tensor x = tf_cfloat.make({2}, {CFloat(1, 1), CFloat(2, -2)});
  const Tensor s = tf_cfloat.make({}, {CFloat(2, 2)});
  const Tensor s_double = tf_cdouble.make({}, {CDouble(2, 2)});

  // Incompatible shapes.
  expect_rejected(x, tf_cfloat.make({3}, {1, 2, 3}), tf_cfloat.zeros({2}));
  // Mixed complex input dtypes, with the scalar on either side.
  expect_rejected(x, s_double, tf_cdouble.zeros({2}));
  expect_rejected(s_double, x, tf_cdouble.zeros({2}));
  // Complex and real inputs, with the real scalar on either side.
  expect_rejected(x, tf_float.ones({}), tf_cfloat.zeros({2}));
  expect_rejected(tf_float.ones({}), x, tf_cfloat.zeros({2}));
  // Output dtype differs from the complex inputs.
  expect_rejected(x, s, tf_cdouble.zeros({2}));
  expect_rejected(s, x, tf_cdouble.zeros({2}));
  expect_rejected(x, s, tf_float.zeros({2}));
  // Real inputs with a complex output.
  expect_rejected(tf_float.ones({2}), tf_float.ones({}), tf_cfloat.zeros({2}));
}
