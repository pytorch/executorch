/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <executorch/kernels/test/FunctionHeaderWrapper.h> // Declares the operator
#include <executorch/kernels/test/TestUtil.h>
#include <executorch/runtime/core/exec_aten/testing_util/tensor_factory.h>
#include <executorch/runtime/core/exec_aten/testing_util/tensor_util.h>
#include <cmath>

using namespace executorch::aten;
using torch::executor::testing::TensorFactory;

class OpCumminOutTest : public OperatorTest {
 protected:
  std::tuple<Tensor&, Tensor&> op_cummin_out(
      const Tensor& self,
      int64_t dim,
      Tensor& values,
      Tensor& indices) {
    return torch::executor::aten::cummin_outf(
        context_, self, dim, values, indices);
  }

  template <ScalarType DTYPE>
  void test_dtype() {
    TensorFactory<DTYPE> tf;
    TensorFactory<ScalarType::Long> tf_long;
    // clang-format off
    Tensor in = tf.make({2, 4}, {
      3, 1, 2, 1,
      0, 5, 0, 4});
    // clang-format on
    Tensor values = tf.zeros({2, 4});
    Tensor indices = tf_long.zeros({2, 4});

    op_cummin_out(in, /*dim=*/1, values, indices);
    // A tie moves the index to the later element, as in PyTorch.
    EXPECT_TENSOR_EQ(values, tf.make({2, 4}, {3, 1, 1, 1, 0, 0, 0, 0}));
    EXPECT_TENSOR_EQ(indices, tf_long.make({2, 4}, {0, 1, 1, 3, 0, 0, 2, 2}));

    op_cummin_out(in, /*dim=*/-2, values, indices);
    EXPECT_TENSOR_EQ(values, tf.make({2, 4}, {3, 1, 2, 1, 0, 1, 0, 1}));
    EXPECT_TENSOR_EQ(indices, tf_long.make({2, 4}, {0, 0, 0, 0, 1, 0, 1, 0}));
  }
};

TEST_F(OpCumminOutTest, AllRealDtypesSupported) {
#define TEST_ENTRY(ctype, dtype) test_dtype<ScalarType::dtype>();
  ET_FORALL_REALHBF16_TYPES(TEST_ENTRY);
#undef TEST_ENTRY
}

TEST_F(OpCumminOutTest, NanBecomesTheMinimum) {
  TensorFactory<ScalarType::Float> tf;
  TensorFactory<ScalarType::Long> tf_long;
  const float nan = NAN;
  Tensor in = tf.make({5}, {2, nan, 1, nan, 0});
  Tensor values = tf.zeros({5});
  Tensor indices = tf_long.zeros({5});

  op_cummin_out(in, /*dim=*/0, values, indices);
  EXPECT_TENSOR_CLOSE(values, tf.make({5}, {2, nan, nan, nan, nan}));
  EXPECT_TENSOR_EQ(indices, tf_long.make({5}, {0, 1, 1, 3, 3}));
}

TEST_F(OpCumminOutTest, Bool) {
  TensorFactory<ScalarType::Bool> tf;
  TensorFactory<ScalarType::Long> tf_long;
  Tensor in = tf.make({4}, {true, false, true, false});
  Tensor values = tf.zeros({4});
  Tensor indices = tf_long.zeros({4});

  op_cummin_out(in, /*dim=*/0, values, indices);
  EXPECT_TENSOR_EQ(values, tf.make({4}, {true, false, false, false}));
  EXPECT_TENSOR_EQ(indices, tf_long.make({4}, {0, 1, 1, 3}));
}

TEST_F(OpCumminOutTest, ZeroDim) {
  TensorFactory<ScalarType::Float> tf;
  TensorFactory<ScalarType::Long> tf_long;
  Tensor in = tf.make({}, {2});
  Tensor values = tf.zeros({});
  Tensor indices = tf_long.ones({});

  op_cummin_out(in, /*dim=*/-1, values, indices);
  EXPECT_TENSOR_EQ(values, in);
  EXPECT_TENSOR_EQ(indices, tf_long.zeros({}));
}

TEST_F(OpCumminOutTest, InvalidDimDies) {
  TensorFactory<ScalarType::Float> tf;
  TensorFactory<ScalarType::Long> tf_long;
  Tensor in = tf.ones({2, 3});
  Tensor values = tf.zeros({2, 3});
  Tensor indices = tf_long.zeros({2, 3});
  ET_EXPECT_KERNEL_FAILURE(context_, op_cummin_out(in, 2, values, indices));
}
