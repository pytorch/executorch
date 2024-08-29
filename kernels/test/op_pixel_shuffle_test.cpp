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
using exec_aten::Scalar;
using exec_aten::ScalarType;
using exec_aten::Tensor;
using torch::executor::testing::SupportedFeatures;
using torch::executor::testing::TensorFactory;

class OpPixelShuffleOutTest : public OperatorTest {
 protected:
  Tensor& op_pixel_shuffle_out(
      const Tensor& self,
      int64_t upscale_factor,
      Tensor& out) {
    return torch::executor::aten::pixel_shuffle_outf(
        context_, self, upscale_factor, out);
  }

  template <ScalarType DTYPE_IN>
  void test_pixel_shuffle() {
    TensorFactory<DTYPE_IN> tf_in;

    const std::vector<int32_t> sizes = {1, 4, 2, 2};
    const std::vector<int32_t> out_sizes = {1, 1, 4, 4};

    // Destination for the pixel_shuffle.
    Tensor out = tf_in.zeros(out_sizes);

    op_pixel_shuffle_out(
        tf_in.make(
            sizes, {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15}),
        2,
        out);
    EXPECT_TENSOR_EQ(
        out,
        // Pixel shuffle distributes channels amongst the spatial dimensions.
        tf_in.make(
            out_sizes, {0, 4, 1, 5, 8, 12, 9, 13, 2, 6, 3, 7, 10, 14, 11, 15}));
  }
};

//
// Correctness Tests
//

/**
 * Uses the function templates above to test all input dtypes.
 */
TEST_F(OpPixelShuffleOutTest, AllRealDtypesSupported) {
#define ENUMERATE_TEST_ENTRY(ctype, dtype) \
  test_pixel_shuffle<ScalarType::dtype>();

  ET_FORALL_REAL_TYPES(ENUMERATE_TEST_ENTRY)

#undef ENUMERATE_TEST_ENTRY
}

TEST_F(OpPixelShuffleOutTest, LargerInputRank) {
  TensorFactory<ScalarType::Int> tf;

  // Pixel shuffle allows a 3D (or higher) input tensor, make sure the extra
  // dimensions don't cause issues.
  Tensor a = tf.ones(/*sizes=*/{1, 4, 1, 4, 2, 2});

  const std::vector<int32_t> out_sizes = {1, 4, 1, 1, 4, 4};
  Tensor out = tf.zeros(out_sizes);

  op_pixel_shuffle_out(a, 2, out);
  EXPECT_TENSOR_EQ(out, tf.ones(out_sizes));
}

// Mismatched shape tests.
TEST_F(OpPixelShuffleOutTest, InvalidInputChannelsDies) {
  TensorFactory<ScalarType::Int> tf;

  // Input tensors with invalid shapes. 7 is not divisible by upsample_factor
  // ** 2.
  Tensor a = tf.ones(/*sizes=*/{1, 7, 4, 4});

  Tensor out = tf.zeros(/*sizes=*/{1, 1, 8, 8});

  // Using the wrong input shape should exit with an error code.
  ET_EXPECT_KERNEL_FAILURE(context_, op_pixel_shuffle_out(a, 2, out));
}

TEST_F(OpPixelShuffleOutTest, WrongInputRankDies) {
  TensorFactory<ScalarType::Int> tf;

  // Pixel shuffle requires a 3D or higher input tensor.
  Tensor a = tf.ones(/*sizes=*/{1, 2});
  Tensor out = tf.zeros(/*sizes=*/{1, 2});

  // Using the wrong input shape should exit with an error code.
  ET_EXPECT_KERNEL_FAILURE(context_, op_pixel_shuffle_out(a, 2, out));
}

TEST_F(OpPixelShuffleOutTest, DifferentDtypeDies) {
  TensorFactory<ScalarType::Int> tf;
  TensorFactory<ScalarType::Float> tf_float;

  Tensor a = tf.ones(/*sizes=*/{1, 18, 4, 4});

  // Pixel shuffle requires two tensors with the same dtype.
  Tensor out = tf_float.zeros(/*sizes=*/{1, 2, 12, 12});

  // Using the wrong output shape should exit with an error code.
  ET_EXPECT_KERNEL_FAILURE(context_, op_pixel_shuffle_out(a, 3, out));
}

TEST_F(OpPixelShuffleOutTest, NegativeUpscaleFactorDies) {
  TensorFactory<ScalarType::Int> tf;
  Tensor a = tf.ones(/*sizes=*/{1, 18, 4, 4});
  Tensor out = tf.zeros(/*sizes=*/{1, 2, 12, 12});
  // Using a negative upscale factor should exit with an error code.
  ET_EXPECT_KERNEL_FAILURE(context_, op_pixel_shuffle_out(a, -3, out));
}
