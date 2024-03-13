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
#include <executorch/runtime/core/exec_aten/util/scalar_type_util.h>

#include <gtest/gtest.h>
#include <cstdint>

using namespace ::testing;
using exec_aten::ArrayRef;
using exec_aten::ScalarType;
using exec_aten::Tensor;
using exec_aten::TensorList;
using torch::executor::testing::TensorFactory;

class OpStackOutTest : public OperatorTest {
 protected:
  Tensor& op_stack_out(TensorList tensors, int64_t dim, Tensor& out) {
    return torch::executor::aten::stack_outf(context_, tensors, dim, out);
  }

  template <class CTYPE, exec_aten::ScalarType DTYPE>
  void test_dtype() {
    TensorFactory<DTYPE> tf;

    // Will be stackd along out.dim(1). Use different input values so we can see
    // where each output value came from.
    Tensor x = tf.ones({3, 4});
    Tensor y = tf.zeros({3, 4});
    std::vector<Tensor> inputs = {x, y};

    Tensor out = tf.ones({3, 2, 4});
    op_stack_out(
        ArrayRef<Tensor>(inputs.data(), inputs.size()), /*dim=*/1, out);

    // The two tensors x and y  are stacked along the 1st dimension with the
    // order [x, y], so the x and y should be equal to expected[:, 0, :] and
    // expected[:, 1, :] e.g. expected[i, 0, j] = x[i, j] and expected[i, 1, j]
    // = y[i, j] for any i in [-x.size(0), x.size(0)-1] and j in [-x.size(1),
    // x.size(1)-1]
    // clang-format off
    Tensor expected = tf.make(
        {3, 2, 4},
        {
          // all ones below are from x,
          // and all zeros are from y.
          // [0, :, :]
          1, 1, 1, 1, // [0, 0, :]
          0, 0, 0, 0, // [0, 1, :]

          // [1, :, :]
          1, 1, 1, 1, // [1, 0, :]
          0, 0, 0, 0, // [1, 1, :]

          // [2, :, :]
          1, 1, 1, 1, // [2, 0, :]
          0, 0, 0, 0, // [2, 1, :]
        });
    // clang-format on

    EXPECT_TENSOR_EQ(out, expected);
  }

  // Running stacking experiments along given dim.
  void run_stack_tests(
      const std::vector<Tensor>& inputs,
      int64_t dim,
      const Tensor& expected) {
    ArrayRef<Tensor> inputs_array(inputs.data(), inputs.size());

    TensorFactory<ScalarType::Double> tf;
    const std::vector<int32_t> out_size(
        expected.sizes().begin(), expected.sizes().end());
    Tensor out = tf.zeros(out_size);

    // Should always return the provided out Tensor.
    Tensor ret = op_stack_out(inputs_array, dim, out);
    EXPECT_TENSOR_EQ(out, ret);
    EXPECT_TENSOR_EQ(out, expected);

    ret = op_stack_out(inputs_array, /*dim=*/dim - out.dim(), out);
    EXPECT_TENSOR_EQ(out, ret);
    EXPECT_TENSOR_EQ(out, expected);
  }
};

TEST_F(OpStackOutTest, InsertFront) {
  TensorFactory<ScalarType::Double> tf;

  // clang-format off
  Tensor x = tf.make(
      {3, 4},
      {
           1.,  2.,  3.,  4., // [0, :]
           5.,  6.,  7.,  8., // [1, :]
           9., 10., 11., 12., // [2, :]

      });
  Tensor y = tf.make(
      {3, 4},
      {
           -1.,  -2.,  -3.,  -4., // [0, :]
           -5.,  -6.,  -7.,  -8., // [1, :]
           -9., -10., -11., -12., // [2, :]
      });
  // clang-format on

  const std::vector<Tensor> inputs = {x, y};

  // Try to stack the two tensors in the front (dim = 0)
  // The size of expected output tensor should follow the below rules:
  // - For any input tensor, its size(i) == output.size(i) if i < dim, and its
  //   size(i) == output.size(i+1) if i >= dim
  // - For the stack dimension (output[dim]), its size should be the number of
  //   input tensors
  std::vector<int32_t> expected_size = {2, 3, 4};

  // The two tensors x and y  are stacked along the 0th dimension with the order
  // [x, y], so the x and y should be equal to expected[0, :, :] and expected[1,
  // :, :] e.g. expected[0, i, j] = x[i, j] and expected[1, i, j] = y[i, j] for
  // any i in [-x.size(0), x.size(0)-1] and j in [-x.size(1), x.size(1)-1]
  // clang-format off
  Tensor expected = tf.make(
    expected_size,
    {
        // [0, :, :] equals to x
        1.,   2.,   3.,   4., // [0, 0, :]
        5.,   6.,   7.,   8., // [0, 1, :]
        9.,  10.,  11.,  12., // [0, 2, :]

        // [1, :, :] equals to y
        -1.,  -2.,  -3.,  -4., // [1, 0, :]
        -5.,  -6.,  -7.,  -8., // [1, 1, :]
        -9., -10., -11., -12., // [1, 2, :]
    });
  // clang-format on

  run_stack_tests(inputs, /*dim=*/0, expected);
}

TEST_F(OpStackOutTest, InsertMiddle) {
  TensorFactory<ScalarType::Double> tf;

  // Two tensors with same size. Stack them on multiple dimensions
  // clang-format off
  Tensor x = tf.make(
      {3, 4},
      {
           1.,  2.,  3.,  4., // [0, :]
           5.,  6.,  7.,  8., // [1, :]
           9., 10., 11., 12., // [2, :]

      });
  Tensor y = tf.make(
      {3, 4},
      {
           -1.,  -2.,  -3.,  -4., // [0, :]
           -5.,  -6.,  -7.,  -8., // [1, :]
           -9., -10., -11., -12., // [2, :]
      });
  // clang-format on

  const std::vector<Tensor> inputs = {x, y};

  // Try to stack the two tensors in the middle (dim = 1)
  // The size of expected output tensor should follow the below rules:
  // - For any input tensor, its size(i) == output.size(i) if i < dim, and its
  //   size(i) == output.size(i+1) if i >= dim
  // - For the stack dimension (output[dim]), its size should be the number of
  //   input tensors
  std::vector<int32_t> expected_size = {3, 2, 4};

  // The two tensors x and y  are stacked along the 1st dimension with the order
  // [x, y], so the x and y should be equal to expected[:, 0, :] and expected[:,
  // 1, :] e.g. expected[i, 0, j] = x[i, j] and expected[i, 1, j] = y[i, j] for
  // any i in [-x.size(0), x.size(0)-1] and j in [-x.size(1), x.size(1)-1]
  // clang-format off
  Tensor expected = tf.make(
      expected_size,
      {
         // [0, :, :]
         1.,   2.,   3.,   4., // [0, 0, :] = x[0, :]
        -1.,  -2.,  -3.,  -4., // [0, 1, :] = y[0, :]

         // [1, :, :]
         5.,   6.,   7.,   8., // [1, 0, :] = x[1, :]
        -5.,  -6.,  -7.,  -8., // [1, 1, :] = y[1, :]

         // [2, :, :]
         9.,  10.,  11.,  12., // [2, 0, :] = x[2, :]
        -9., -10., -11., -12., // [2, 1, :] = y[2, :]
      });
  // clang-format on

  run_stack_tests(inputs, /*dim=*/1, expected);
}

TEST_F(OpStackOutTest, InsertEnd) {
  TensorFactory<ScalarType::Double> tf;

  // Two tensors with same size. Stack them on multiple dimensions
  // clang-format off
  Tensor x = tf.make(
      {3, 4},
      {
           1.,  2.,  3.,  4., // [0, :]
           5.,  6.,  7.,  8., // [1, :]
           9., 10., 11., 12., // [2, :]

      });
  Tensor y = tf.make(
      {3, 4},
      {
           -1.,  -2.,  -3.,  -4., // [0, :]
           -5.,  -6.,  -7.,  -8., // [1, :]
           -9., -10., -11., -12., // [2, :]
      });
  // clang-format on

  const std::vector<Tensor> inputs = {x, y};

  // Try to stack the two tensors at the end (dim = 2)
  // The size of expected output tensor should follow the below rules:
  // - For any input tensor, its size(i) == output.size(i) if i < dim, and its
  //   size(i) == output.size(i+1) if i >= dim
  // - For the stack dimension (output[dim]), its size should be the number of
  //   input tensors
  std::vector<int32_t> expected_size = {3, 4, 2};

  // The two tensors x and y are stacked along the 2nd dimension with the order
  // [x, y], so the x and y should be equal to expected[:, :, 0] and expected[:,
  // :, 1] e.g. expected[i, j, 0] = x[i, j] and expected[i, j, 1] = y[i, j] for
  // any i in [-x.size(0), x.size(0)-1] and j in [-x.size(1), x.size(1)-1]
  // clang-format off
  Tensor expected = tf.make(
      expected_size,
      {
          // All values in the first column are from x,
          // and the second column are from y

          // [0, :, :]
          1.,  -1., // [0, 0, :]
          2.,  -2., // [0, 1, :]
          3.,  -3., // [0, 2, :]
          4.,  -4., // [0, 3, :]

          // [1, :, :]
          5.,  -5., // [1, 0, :]
          6.,  -6., // [1, 1, :]
          7.,  -7., // [1, 2, :]
          8.,  -8., // [1, 3, :]

          // [2, :, :]
          9.,  -9., // [2, 0, :]
         10., -10., // [2, 1, :]
         11., -11., // [2, 2, :]
         12., -12., // [2, 3, :]
      });
  // clang-format on

  run_stack_tests(inputs, /*dim=*/2, expected);
}

/// A generic smoke test that works for any dtype that supports ones() and
/// zeros().
TEST_F(OpStackOutTest, AllDtypesSupported) {
#define TEST_ENTRY(ctype, dtype) test_dtype<ctype, ScalarType::dtype>();
  ET_FORALL_REAL_TYPES_AND(Bool, TEST_ENTRY);
#undef TEST_ENTRY
  // TODO: Also add tests for half, complex, quantized, and other types. Easiest
  // way to do that would be to make TensorFactory support zeros() and ones()
  // for those types.
}

TEST_F(OpStackOutTest, NoInputTensorsWithEmptyOutTensorFails) {
  TensorFactory<ScalarType::Int> tf;

  // Make an empty out tensor and demonstrate that it's empty.
  Tensor out = tf.make({0}, {});
  EXPECT_EQ(out.numel(), 0);

  // Pass an empty list of input tensors.
  ET_EXPECT_KERNEL_FAILURE(
      context_, op_stack_out(ArrayRef<Tensor>(), /*dim=*/0, out));
}

TEST_F(OpStackOutTest, AllEmptyInputTensors) {
  TensorFactory<ScalarType::Int> tf;

  // Using empty tensors as input.
  Tensor empty = tf.make({0, 10, 3}, {});
  EXPECT_EQ(empty.numel(), 0);
  std::vector<Tensor> inputs = {empty, empty, empty};

  Tensor x = tf.ones({2, 2});

  // Output whose shape is appropriate for stacking along out.dim(0).
  Tensor out = tf.make({3, 0, 10, 3}, {});
  EXPECT_EQ(out.numel(), 0);

  Tensor ret = op_stack_out(
      ArrayRef<Tensor>(inputs.data(), inputs.size()), /*dim=*/0, out);
  EXPECT_EQ(ret.numel(), 0);
  // Success if it doesn't assert on the weird-shaped empty input and the
  // empty_out is still a empty array
}

TEST_F(OpStackOutTest, DimOutOfBoundDies) {
  TensorFactory<ScalarType::Int> tf;

  // Stack a single tensor with size [1, 1]. The size of output would always be
  // [1, 1, 1] no matter stack on which dimension.
  Tensor x = tf.ones({1, 1});
  ArrayRef<Tensor> inputs(&x, 1);

  Tensor out = tf.zeros({1, 1, 1});

  // Some invalid dim values.
  const std::vector<int64_t> invalid_dims = {3, 4, 5, -4, -5, -6};
  for (int64_t dim : invalid_dims) {
    ET_EXPECT_KERNEL_FAILURE(context_, op_stack_out(inputs, dim, out));
  }
}

TEST_F(OpStackOutTest, MismatchedDtypesDies) {
  TensorFactory<ScalarType::Int> tf_int;
  TensorFactory<ScalarType::Float> tf_float;
  Tensor out = tf_int.zeros({1, 2, 2});

  // Size is compatible to the output, but a mismatched dtype.
  std::vector<Tensor> inputs = {tf_float.ones({2, 2})};

  ET_EXPECT_KERNEL_FAILURE(
      context_,
      op_stack_out(
          ArrayRef<Tensor>(inputs.data(), inputs.size()), /*dim=*/0, out));
}

TEST_F(OpStackOutTest, OutMatchNumelWithExtraDimAtEndDies) {
  if (torch::executor::testing::SupportedFeatures::get()->is_aten) {
    GTEST_SKIP() << "ATen kernel can handle out with mismatched dimensions";
  }
  TensorFactory<ScalarType::Int> tf;
  Tensor out = tf.zeros({1, 2, 2, 1});

  // Same dtype and numel as the output, but a mixmatched size (output.dim()
  // should always one greater than input.dim())
  std::vector<Tensor> inputs = {tf.ones({2, 2})};

  ET_EXPECT_KERNEL_FAILURE(
      context_,
      op_stack_out(
          ArrayRef<Tensor>(inputs.data(), inputs.size()), /*dim=*/0, out));
}

TEST_F(OpStackOutTest, OutMatchNumelLackDimAtFrontDies) {
  if (torch::executor::testing::SupportedFeatures::get()->is_aten) {
    GTEST_SKIP() << "ATen kernel can handle out with mismatched dimensions";
  }
  TensorFactory<ScalarType::Int> tf;
  Tensor out = tf.zeros({2, 2});

  // Same dtype and numel as the output, but a mixmatched size (output.dim()
  // should always one greater than input.dim())
  std::vector<Tensor> inputs = {tf.ones({2, 2})};

  ET_EXPECT_KERNEL_FAILURE(
      context_,
      op_stack_out(
          ArrayRef<Tensor>(inputs.data(), inputs.size()), /*dim=*/0, out));
}

TEST_F(OpStackOutTest, OutRegularMismatchDimDies) {
  if (torch::executor::testing::SupportedFeatures::get()->is_aten) {
    GTEST_SKIP() << "ATen kernel can handle out with mismatched dimensions";
  }
  TensorFactory<ScalarType::Int> tf;

  // Should be {2, 2, 3} to match the inputs when calling stack() with dim 0.
  Tensor out = tf.zeros({2, 4, 5});

  std::vector<Tensor> inputs = {
      tf.ones({2, 3}),
      tf.ones({2, 3}),
  };

  ET_EXPECT_KERNEL_FAILURE(
      context_,
      op_stack_out(
          ArrayRef<Tensor>(inputs.data(), inputs.size()), /*dim=*/0, out));
}

/* %python
import torch
torch.manual_seed(0)
x = [torch.randint(10, (2, 3)),
     torch.randint(10, (2, 3)),
     torch.randint(10, (2, 3)),
     torch.randint(10, (2, 3))]
res = torch.stack(x, 0)
op = "op_stack_out"
opt_extra_params = "0,"
dtype = "ScalarType::Int"
check = "EXPECT_TENSOR_EQ" */

TEST_F(OpStackOutTest, DynamicShapeUpperBoundSameAsExpected) {
  /* %python
  out_args = "{4, 2, 3}, torch::executor::TensorShapeDynamism::DYNAMIC_BOUND"
  %rewrite(unary_op_tensor_list_in) */

  TensorFactory<ScalarType::Int> tf;

  std::vector<Tensor> xv = {
      tf.make({2, 3}, {4, 9, 3, 0, 3, 9}),
      tf.make({2, 3}, {7, 3, 7, 3, 1, 6}),
      tf.make({2, 3}, {6, 9, 8, 6, 6, 8}),
      tf.make({2, 3}, {4, 3, 6, 9, 1, 4})};
  TensorList x(xv.data(), xv.size());
  Tensor expected = tf.make({4, 2, 3}, {4, 9, 3, 0, 3, 9, 7, 3, 7, 3, 1, 6,
                                        6, 9, 8, 6, 6, 8, 4, 3, 6, 9, 1, 4});

  Tensor out =
      tf.zeros({4, 2, 3}, torch::executor::TensorShapeDynamism::DYNAMIC_BOUND);
  op_stack_out(x, 0, out);
  EXPECT_TENSOR_EQ(out, expected);
}

TEST_F(OpStackOutTest, DynamicShapeUpperBoundLargerThanExpected) {
  if (!torch::executor::testing::SupportedFeatures::get()->output_resize) {
    GTEST_SKIP() << "Dynamic shape not supported";
  }
  /* %python
  out_args = "{5, 5, 5}, torch::executor::TensorShapeDynamism::DYNAMIC_BOUND"
  %rewrite(unary_op_tensor_list_in) */

  TensorFactory<ScalarType::Int> tf;

  std::vector<Tensor> xv = {
      tf.make({2, 3}, {4, 9, 3, 0, 3, 9}),
      tf.make({2, 3}, {7, 3, 7, 3, 1, 6}),
      tf.make({2, 3}, {6, 9, 8, 6, 6, 8}),
      tf.make({2, 3}, {4, 3, 6, 9, 1, 4})};
  TensorList x(xv.data(), xv.size());
  Tensor expected = tf.make({4, 2, 3}, {4, 9, 3, 0, 3, 9, 7, 3, 7, 3, 1, 6,
                                        6, 9, 8, 6, 6, 8, 4, 3, 6, 9, 1, 4});

  Tensor out =
      tf.zeros({5, 5, 5}, torch::executor::TensorShapeDynamism::DYNAMIC_BOUND);
  op_stack_out(x, 0, out);
  EXPECT_TENSOR_EQ(out, expected);
}

TEST_F(OpStackOutTest, DynamicShapeUnbound) {
  if (!torch::executor::testing::SupportedFeatures::get()->output_resize) {
    GTEST_SKIP() << "Dynamic shape not supported";
  }
  /* %python
  out_args = "{1, 1, 1}, torch::executor::TensorShapeDynamism::DYNAMIC_UNBOUND"
  %rewrite(unary_op_tensor_list_in) */

  TensorFactory<ScalarType::Int> tf;

  std::vector<Tensor> xv = {
      tf.make({2, 3}, {4, 9, 3, 0, 3, 9}),
      tf.make({2, 3}, {7, 3, 7, 3, 1, 6}),
      tf.make({2, 3}, {6, 9, 8, 6, 6, 8}),
      tf.make({2, 3}, {4, 3, 6, 9, 1, 4})};
  TensorList x(xv.data(), xv.size());
  Tensor expected = tf.make({4, 2, 3}, {4, 9, 3, 0, 3, 9, 7, 3, 7, 3, 1, 6,
                                        6, 9, 8, 6, 6, 8, 4, 3, 6, 9, 1, 4});

  Tensor out = tf.zeros(
      {1, 1, 1}, torch::executor::TensorShapeDynamism::DYNAMIC_UNBOUND);
  op_stack_out(x, 0, out);
  EXPECT_TENSOR_EQ(out, expected);
}
