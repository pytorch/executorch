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
using executorch::aten::ScalarType;
using executorch::aten::Tensor;
using torch::executor::testing::TensorFactory;

class OpCeilTest : public OperatorTest {
 protected:
  Tensor& op_ceil_out(const Tensor& self, Tensor& out) {
    return torch::executor::aten::ceil_outf(context_, self, out);
  }

  template <ScalarType DTYPE>
  void test_ceil_float_dtype() {
    TensorFactory<DTYPE> tf;

    Tensor in = tf.make({1, 7}, {-3.0, -2.99, -1.01, 0.0, 1.01, 2.99, 3.0});
    Tensor out = tf.zeros({1, 7});
    Tensor expected = tf.make({1, 7}, {-3.0, -2.0, -1.0, 0.0, 2.0, 3.0, 3.0});

    Tensor ret = op_ceil_out(in, out);

    EXPECT_TENSOR_EQ(out, ret);
    EXPECT_TENSOR_EQ(out, expected);
  }
};

TEST_F(OpCeilTest, AllFloatDtypeSupport) {
#define TEST_ENTRY(ctype, dtype) test_ceil_float_dtype<ScalarType::dtype>();
  if (torch::executor::testing::SupportedFeatures::get()->is_aten) {
    ET_FORALL_FLOAT_TYPES(TEST_ENTRY);
  } else {
    ET_FORALL_FLOATHBF16_TYPES(TEST_ENTRY);
  }
#undef TEST_ENTRY
}
