/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <executorch/kernels/test/FunctionHeaderWrapper.h> // Declares the operator
#include <executorch/kernels/test/TestUtil.h>
#include <executorch/kernels/test/supported_features.h>
#include <executorch/runtime/core/exec_aten/exec_aten.h>
#include <executorch/runtime/core/exec_aten/testing_util/tensor_factory.h>
#include <executorch/runtime/core/exec_aten/testing_util/tensor_util.h>

#include <gtest/gtest.h>

using namespace ::testing;
using exec_aten::optional;
using exec_aten::ScalarType;
using exec_aten::Tensor;
using torch::executor::testing::TensorFactory;

namespace {
Tensor& lift_fresh_copy_out(const Tensor& self, Tensor& out) {
  exec_aten::RuntimeContext context{};
  return torch::executor::aten::lift_fresh_copy_outf(context, self, out);
}
} // namespace

// regular test for lift_fresh_copy.out
// test if lift_fresh_copy.out works well under all kinds of legal input type.
template <class CTYPE, exec_aten::ScalarType DTYPE>
void test_dtype() {
  TensorFactory<DTYPE> tf;
  Tensor self = tf.ones(/*sizes=*/{2, 4});
  Tensor out = tf.zeros(/*sizes=*/{2, 4});

  ::lift_fresh_copy_out(self, out);
  EXPECT_TENSOR_EQ(self, out);

  Tensor self_empty = tf.make(/*sizes=*/{}, /*data=*/{1});
  Tensor out_empty = tf.make(/*sizes=*/{}, /*data=*/{0});

  ::lift_fresh_copy_out(self_empty, out_empty);
  EXPECT_TENSOR_EQ(self_empty, out_empty);
}

TEST(OpLiftFreshCopyTest, AllDtypesSupported) {
#define TEST_ENTRY(ctype, dtype) test_dtype<ctype, ScalarType::dtype>();
  ET_FORALL_REAL_TYPES_AND(Bool, TEST_ENTRY);
#undef TEST_ENTRY
}

template <class CTYPE, ScalarType DTYPE>
void test_empty_input() {
  TensorFactory<DTYPE> tf;
  Tensor self = tf.make(/*sizes=*/{3, 0, 1, 2}, /*data=*/{});
  Tensor out = tf.zeros({3, 0, 1, 2});
  ::lift_fresh_copy_out(self, out);
  EXPECT_TENSOR_EQ(self, out);
}

TEST(OpLiftFreshCopyTest, EmptyInputSupported) {
#define TEST_ENTRY(ctype, dtype) test_empty_input<ctype, ScalarType::dtype>();
  ET_FORALL_REAL_TYPES_AND(Bool, TEST_ENTRY);
#undef TEST_ENTRY
}

TEST(OpLiftFreshCopyTest, MismatchedSizesDie) {
  if (torch::executor::testing::SupportedFeatures::get()->is_aten) {
    GTEST_SKIP() << "ATen kernel can handle mismatched sizes";
  }
  TensorFactory<ScalarType::Int> tf;
  Tensor self = tf.make(/*sizes=*/{3, 1, 1, 2}, /*data=*/{1, 2, 3, 4, 5, 6});
  Tensor out = tf.zeros({3, 2, 1, 1});
  ET_EXPECT_KERNEL_FAILURE(::lift_fresh_copy_out(self, out));
}

TEST(OpLiftFreshCopyTest, MismatchedDTypeDie) {
  TensorFactory<ScalarType::Int> tf_in;
  TensorFactory<ScalarType::Float> tf_out;
  Tensor self = tf_in.make(/*sizes=*/{3, 1, 1, 2}, /*data=*/{1, 2, 3, 4, 5, 6});
  Tensor out = tf_out.zeros({3, 1, 1, 2});
  ET_EXPECT_KERNEL_FAILURE(::lift_fresh_copy_out(self, out));
}
