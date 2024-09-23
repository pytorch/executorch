/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <executorch/extension/llm/custom_ops/op_tile_crop.h>
#include <executorch/kernels/test/TestUtil.h>
#include <executorch/runtime/core/exec_aten/testing_util/tensor_factory.h>
#include <executorch/runtime/core/exec_aten/testing_util/tensor_util.h>
#include <gtest/gtest.h>

using namespace ::testing;
using exec_aten::ScalarType;
using exec_aten::Tensor;
using executorch::runtime::testing::TensorFactory;

class OpTileCropOutTest : public OperatorTest {
 protected:
  Tensor& op_tile_crop_out(const Tensor& self, int64_t tile_size, Tensor& out) {
    return torch::executor::native::tile_crop_out_impl(
        context_, self, tile_size, out);
  }

  template <ScalarType DTYPE_IN>
  void test_tile_crop() {
    TensorFactory<DTYPE_IN> tf_in;

    const std::vector<int32_t> sizes = {1, 4, 4};
    const std::vector<int32_t> out_sizes = {4, 1, 2, 2};

    Tensor out = tf_in.zeros(out_sizes);

    // clang-format off
    op_tile_crop_out(
        tf_in.make(
            sizes, { 0,  1,  2,  3,
                     4,  5,  6,  7,
                     8,  9, 10, 11,
                    12, 13, 14, 15}),
        2,
        out);
    EXPECT_TENSOR_EQ(
        out,
        tf_in.make(
            out_sizes, {0,  1,  4,  5,
                        2,  3,  6,  7,
                        8,  9, 12, 13,
                       10, 11, 14, 15}));
    // clang-format on
  }
};

//
// Correctness Tests
//

/**
 * Uses the function templates above to test all input dtypes.
 */
TEST_F(OpTileCropOutTest, AllRealDtypesSupported){
#define ENUMERATE_TEST_ENTRY(ctype, dtype) test_tile_crop<ScalarType::dtype>();
    ET_FORALL_REAL_TYPES(ENUMERATE_TEST_ENTRY)
#undef ENUMERATE_TEST_ENTRY
}

// Mismatched shape tests.
TEST_F(OpTileCropOutTest, InvalidInputShapeDies) {
  TensorFactory<ScalarType::Int> tf;

  // Input tensors with invalid shapes. 7 is not divisible by tile_size
  Tensor in = tf.ones(/*sizes=*/{1, 7, 8});
  Tensor out = tf.zeros(/*sizes=*/{16, 1, 2, 2});

  ET_EXPECT_KERNEL_FAILURE(context_, op_tile_crop_out(in, 2, out));
}

TEST_F(OpTileCropOutTest, WrongInputRankDies) {
  TensorFactory<ScalarType::Int> tf;

  // Tile crop requires a 3D input tensor.
  Tensor in = tf.ones(/*sizes=*/{1, 2});
  Tensor out = tf.zeros(/*sizes=*/{1, 2});

  ET_EXPECT_KERNEL_FAILURE(context_, op_tile_crop_out(in, 2, out));
}

TEST_F(OpTileCropOutTest, DifferentDtypeDies) {
  TensorFactory<ScalarType::Int> tf;
  TensorFactory<ScalarType::Float> tf_float;

  Tensor in = tf.ones(/*sizes=*/{2, 12, 12});

  // Tile crop requires two tensors with the same dtype.
  Tensor out = tf_float.zeros(/*sizes=*/{9, 2, 4, 4});

  ET_EXPECT_KERNEL_FAILURE(context_, op_tile_crop_out(in, 3, out));
}

TEST_F(OpTileCropOutTest, NegativeTileSizeDies) {
  TensorFactory<ScalarType::Int> tf;
  Tensor in = tf.ones(/*sizes=*/{2, 12, 12});
  Tensor out = tf.zeros(/*sizes=*/{9, 2, 4, 4});
  ET_EXPECT_KERNEL_FAILURE(context_, op_tile_crop_out(in, -3, out));
}
