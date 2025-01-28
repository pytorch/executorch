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
using exec_aten::ScalarType;
using exec_aten::Tensor;
using torch::executor::testing::SupportedFeatures;
using torch::executor::testing::TensorFactory;

class OpMaskedSelectOutTest : public OperatorTest {
 protected:
  Tensor&
  op_masked_select_out(const Tensor& in, const Tensor& mask, Tensor& out) {
    return torch::executor::aten::masked_select_outf(context_, in, mask, out);
  }
};

TEST_F(OpMaskedSelectOutTest, SmokeTest) {
  TensorFactory<ScalarType::Int> tf;
  TensorFactory<ScalarType::Bool> tfBool;

  Tensor in = tf.make({2, 3}, {1, 2, 3, 4, 5, 6});
  Tensor mask = tfBool.make({2, 3}, {true, false, false, true, false, true});
  Tensor out = tf.zeros({3});

  op_masked_select_out(in, mask, out);
  EXPECT_TENSOR_EQ(out, tf.make({3}, {1, 4, 6}));
}

TEST_F(OpMaskedSelectOutTest, BroadcastInput) {
  TensorFactory<ScalarType::Int> tf;
  TensorFactory<ScalarType::Bool> tfBool;

  Tensor in = tf.make({3}, {1, 2, 3});
  Tensor mask = tfBool.make({2, 3}, {true, false, false, true, false, true});
  Tensor out = tf.zeros({3});

  op_masked_select_out(in, mask, out);
  EXPECT_TENSOR_EQ(out, tf.make({3}, {1, 1, 3}));
}

TEST_F(OpMaskedSelectOutTest, BroadcastMask) {
  TensorFactory<ScalarType::Int> tf;
  TensorFactory<ScalarType::Bool> tfBool;

  Tensor in = tf.make({2, 3}, {1, 2, 3, 4, 5, 6});
  Tensor mask = tfBool.make({3}, {false, true, false});

  Tensor out = tf.zeros({2});

  op_masked_select_out(in, mask, out);
  EXPECT_TENSOR_EQ(out, tf.make({2}, {2, 5}));
}

TEST_F(OpMaskedSelectOutTest, BroadcastInputAndMask) {
  TensorFactory<ScalarType::Int> tf;
  TensorFactory<ScalarType::Bool> tfBool;

  Tensor in = tf.ones({2, 3, 4, 1});
  Tensor mask = tfBool.ones({2, 1, 1, 5});
  Tensor out = tf.zeros({120});

  op_masked_select_out(in, mask, out);
  EXPECT_TENSOR_EQ(out, tf.ones({120}));
}

TEST_F(OpMaskedSelectOutTest, EmptyInput) {
  TensorFactory<ScalarType::Int> tf;
  TensorFactory<ScalarType::Bool> tfBool;

  Tensor in = tf.make({2, 0}, {});
  Tensor mask = tfBool.make({2, 1}, {true, true});
  Tensor out = tf.zeros({0});

  op_masked_select_out(in, mask, out);
  EXPECT_TENSOR_EQ(out, tf.make({0}, {}));
}

TEST_F(OpMaskedSelectOutTest, EmptyMask) {
  TensorFactory<ScalarType::Int> tf;
  TensorFactory<ScalarType::Bool> tfBool;

  Tensor in = tf.make({2, 1}, {100, 200});
  Tensor mask = tfBool.make({2, 0}, {});
  Tensor out = tf.zeros({0});

  op_masked_select_out(in, mask, out);
  EXPECT_TENSOR_EQ(out, tf.make({0}, {}));
}

TEST_F(OpMaskedSelectOutTest, EmptyInputAndMask) {
  TensorFactory<ScalarType::Int> tf;
  TensorFactory<ScalarType::Bool> tfBool;

  Tensor in = tf.make({2, 0}, {});
  Tensor mask = tfBool.make({0}, {});
  Tensor out = tf.zeros({0});

  op_masked_select_out(in, mask, out);
  EXPECT_TENSOR_EQ(out, tf.make({0}, {}));
}
