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

class OpMaskedScatterOutTest : public OperatorTest {
 protected:
  Tensor& op_masked_scatter_out(
      const Tensor& in,
      const Tensor& mask,
      const Tensor& src,
      Tensor& out) {
    return torch::executor::aten::masked_scatter_outf(
        context_, in, mask, src, out);
  }
};

TEST_F(OpMaskedScatterOutTest, SmokeTest) {
  TensorFactory<ScalarType::Int> tf;
  TensorFactory<ScalarType::Bool> tfBool;

  Tensor in = tf.make({2, 3}, {1, 2, 3, 4, 5, 6});
  Tensor mask = tfBool.make({2, 3}, {true, false, false, true, false, true});
  Tensor src = tf.make({3}, {10, 20, 30});

  Tensor out = tf.zeros({2, 3});

  op_masked_scatter_out(in, mask, src, out);
  EXPECT_TENSOR_EQ(out, tf.make({2, 3}, {10, 2, 3, 20, 5, 30}));
}

TEST_F(OpMaskedScatterOutTest, BroadcastInput) {
  TensorFactory<ScalarType::Int> tf;
  TensorFactory<ScalarType::Bool> tfBool;

  Tensor in = tf.make({3}, {1, 2, 3});
  Tensor mask = tfBool.make({2, 3}, {true, false, false, true, false, true});
  Tensor src = tf.make({3}, {10, 20, 30});

  Tensor out = tf.zeros({2, 3});

  op_masked_scatter_out(in, mask, src, out);
  EXPECT_TENSOR_EQ(out, tf.make({2, 3}, {10, 2, 3, 20, 2, 30}));
}

TEST_F(OpMaskedScatterOutTest, BroadcastMask) {
  TensorFactory<ScalarType::Int> tf;
  TensorFactory<ScalarType::Bool> tfBool;

  Tensor in = tf.make({2, 3}, {1, 2, 3, 4, 5, 6});
  Tensor mask = tfBool.make({3}, {false, true, false});
  Tensor src = tf.make({2}, {10, 20});

  Tensor out = tf.zeros({2, 3});

  op_masked_scatter_out(in, mask, src, out);
  EXPECT_TENSOR_EQ(out, tf.make({2, 3}, {1, 10, 3, 4, 20, 6}));
}

TEST_F(OpMaskedScatterOutTest, SrcWithMoreElements) {
  TensorFactory<ScalarType::Int> tf;
  TensorFactory<ScalarType::Bool> tfBool;

  Tensor in = tf.make({2, 3}, {1, 2, 3, 4, 5, 6});
  Tensor mask = tfBool.make({2, 3}, {true, false, false, true, false, true});
  Tensor src = tf.make({4}, {10, 20, 30, 40});

  Tensor out = tf.zeros({2, 3});

  op_masked_scatter_out(in, mask, src, out);
  EXPECT_TENSOR_EQ(out, tf.make({2, 3}, {10, 2, 3, 20, 5, 30}));
}

TEST_F(OpMaskedScatterOutTest, SrcWithLessElementsFails) {
  TensorFactory<ScalarType::Int> tf;
  TensorFactory<ScalarType::Bool> tfBool;

  Tensor in = tf.make({2, 3}, {1, 2, 3, 4, 5, 6});
  Tensor mask = tfBool.make({2, 3}, {true, false, false, true, false, true});
  Tensor src = tf.make({2}, {10, 20});

  Tensor out = tf.zeros({2, 3});

  ET_EXPECT_KERNEL_FAILURE(context_, op_masked_scatter_out(in, mask, src, out));
}

TEST_F(OpMaskedScatterOutTest, EmptyMask) {
  TensorFactory<ScalarType::Int> tf;
  TensorFactory<ScalarType::Bool> tfBool;

  Tensor in = tf.make({2, 1}, {100, 200});
  Tensor mask = tfBool.make({2, 0}, {});
  Tensor src = tf.make({4}, {10, 20, 30, 40});

  Tensor out = tf.zeros({2, 0});

  op_masked_scatter_out(in, mask, src, out);
  EXPECT_TENSOR_EQ(out, tf.make({2, 0}, {}));
}

TEST_F(OpMaskedScatterOutTest, EmptySrc) {
  TensorFactory<ScalarType::Int> tf;
  TensorFactory<ScalarType::Bool> tfBool;

  Tensor in = tf.make({2, 1}, {100, 200});
  Tensor mask = tfBool.make({2, 1}, {false, false});
  Tensor src = tf.make({0}, {});

  Tensor out = tf.zeros({2, 1});

  op_masked_scatter_out(in, mask, src, out);
  EXPECT_TENSOR_EQ(out, tf.make({2, 1}, {100, 200}));
}

TEST_F(OpMaskedScatterOutTest, EmptyMaskAndSrc) {
  TensorFactory<ScalarType::Int> tf;
  TensorFactory<ScalarType::Bool> tfBool;

  Tensor in = tf.make({2, 1}, {100, 200});
  Tensor mask = tfBool.make({0}, {});
  Tensor src = tf.make({0}, {});

  Tensor out = tf.zeros({2, 0});

  op_masked_scatter_out(in, mask, src, out);
  EXPECT_TENSOR_EQ(out, tf.make({2, 0}, {}));
}
