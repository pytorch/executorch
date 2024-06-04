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

class OpSplitWithSizesCopyOutTest : public OperatorTest {
 protected:
  void op_split_with_sizes_copy_out(
      const exec_aten::Tensor& self,
      exec_aten::ArrayRef<int64_t> split_sizes,
      int64_t dim,
      exec_aten::TensorList out) {
    return torch::executor::aten::split_with_sizes_copy_outf(
        context_, self, split_sizes, dim, out);
  }

  void test_tensor_shape_dynamism(exec_aten::TensorShapeDynamism dynamism) {
    torch::executor::testing::TensorFactory<exec_aten::ScalarType::Float>
        tfFloat;

    exec_aten::Tensor self = tfFloat.make(
        {2, 6, 3},
        {-31.25,  -92.75,  -39.75,  -3.25,   53.875,  88.25,   -0.625,  -1.125,
         14.75,   42.0,    89.875,  -21.125, -8.0,    -64.125, 23.0,    37.0,
         46.125,  -83.25,  -58.125, 19.625,  -71.125, 64.75,   -1.375,  -83.5,
         -61.375, 13.125,  28.625,  -94.0,   -67.0,   -8.625,  -88.875, -79.125,
         0.375,   -61.375, 65.0,    -99.375});
    ::std::vector<int64_t> split_sizes_vec = {3, 1, 2};
    exec_aten::ArrayRef<int64_t> split_sizes = exec_aten::ArrayRef<int64_t>(
        split_sizes_vec.data(), split_sizes_vec.size());
    int64_t dim = 1;

    ::std::vector<exec_aten::Tensor> out_vec;
    if (dynamism == exec_aten::TensorShapeDynamism::STATIC) {
      out_vec = {
          tfFloat.zeros({2, 3, 3}),
          tfFloat.zeros({2, 1, 3}),
          tfFloat.zeros({2, 2, 3})};
    } else { // dynamism == exec_aten::TensorShapeDynamism::DYNAMIC_BOUND
      out_vec = {
          tfFloat.zeros(
              {2, 3, 10}, torch::executor::TensorShapeDynamism::DYNAMIC_BOUND),
          tfFloat.zeros(
              {2, 1, 10}, torch::executor::TensorShapeDynamism::DYNAMIC_BOUND),
          tfFloat.zeros(
              {2, 2, 10}, torch::executor::TensorShapeDynamism::DYNAMIC_BOUND)};
    }

    exec_aten::TensorList out =
        exec_aten::TensorList(out_vec.data(), out_vec.size());
    ::std::vector<exec_aten::Tensor> out_expected_vec = {
        tfFloat.make(
            {2, 3, 3},
            {-31.25,
             -92.75,
             -39.75,
             -3.25,
             53.875,
             88.25,
             -0.625,
             -1.125,
             14.75,
             -58.125,
             19.625,
             -71.125,
             64.75,
             -1.375,
             -83.5,
             -61.375,
             13.125,
             28.625}),
        tfFloat.make({2, 1, 3}, {42.0, 89.875, -21.125, -94.0, -67.0, -8.625}),
        tfFloat.make(
            {2, 2, 3},
            {-8.0,
             -64.125,
             23.0,
             37.0,
             46.125,
             -83.25,
             -88.875,
             -79.125,
             0.375,
             -61.375,
             65.0,
             -99.375})};
    exec_aten::TensorList out_expected =
        exec_aten::TensorList(out_expected_vec.data(), out_expected_vec.size());
    op_split_with_sizes_copy_out(self, split_sizes, dim, out);
    EXPECT_TENSOR_LISTS_CLOSE(out, out_expected);
  }
};

TEST_F(OpSplitWithSizesCopyOutTest, SanityCheckDim1) {
  test_tensor_shape_dynamism(exec_aten::TensorShapeDynamism::STATIC);
}

TEST_F(OpSplitWithSizesCopyOutTest, DynamicShape) {
  test_tensor_shape_dynamism(exec_aten::TensorShapeDynamism::DYNAMIC_BOUND);
}
