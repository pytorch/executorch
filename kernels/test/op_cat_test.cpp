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

using namespace ::testing;
using exec_aten::ArrayRef;
using exec_aten::ScalarType;
using exec_aten::Tensor;
using exec_aten::TensorList;
using torch::executor::testing::TensorFactory;

class OpCatOutTest : public OperatorTest {
 protected:
  Tensor& op_cat_out(TensorList tensors, int64_t dim, Tensor& out) {
    return torch::executor::aten::cat_outf(context_, tensors, dim, out);
  }

  template <class CTYPE, exec_aten::ScalarType DTYPE>
  void test_dtype() {
    TensorFactory<DTYPE> tf;

    // Will be concatenated along dim[1]. Use different input values so we can
    // see where each output value came from.
    Tensor x = tf.ones({2, 1});
    Tensor y = tf.zeros({2, 1});
    std::vector<Tensor> inputs = {x, y};

    Tensor out = tf.ones({2, 2});
    op_cat_out(ArrayRef<Tensor>(inputs.data(), inputs.size()), /*dim=*/1, out);

    // clang-format off
    Tensor expected = tf.make(
        {2, 2},
        {
            1, 0,
            1, 0,
        });
    // clang-format on

    EXPECT_TENSOR_EQ(out, expected);
  }
};

TEST_F(OpCatOutTest, SmokeDim1) {
  TensorFactory<ScalarType::Int> tf;

  // Two tensors with the same number of dimensions and the same dim[0]
  // size, but different dim[1] sizes. These will be concatenated along dim[1].
  // clang-format off
  Tensor x = tf.make(
      {2, 3},
      {
          1, 2, 3,
          4, 5, 6,
      });
  Tensor y = tf.make(
      {2, 1},
      {
          10,
          20,
      });
  // clang-format on

  std::vector<Tensor> inputs = {x, y};

  // Output tensor with the shape of the two input tensors concatenated along
  // dim[1].
  // - It should have the same number of dimensions as each input.
  // - For non-cat dimensions (dim[0]), it should have the same size as the
  //   input tensors.
  // - For the cat dimension (dim[1]), its size should be the sum of the cat
  //   dimensions of the inputs: in this case, 3 + 1.
  Tensor out = tf.zeros({2, 4});

  // Concatenate along dim[1].
  Tensor ret = op_cat_out(
      ArrayRef<Tensor>(inputs.data(), inputs.size()), /*dim=*/1, out);

  // Should always return the provided out Tensor.
  EXPECT_TENSOR_EQ(ret, out);

  // clang-format off
  Tensor expected = tf.make(
      {2, 4},
      {
          1, 2, 3, 10,
          4, 5, 6, 20,
      });
  // clang-format on

  EXPECT_TENSOR_EQ(out, expected);
}

TEST_F(OpCatOutTest, HalfSupport) {
  if (torch::executor::testing::SupportedFeatures::get()->is_aten) {
    GTEST_SKIP() << "Test Half support only for ExecuTorch mode";
  }
  TensorFactory<ScalarType::Half> tf;

  Tensor x = tf.make({2, 3}, {1.5, -2.0, 3.25, 4.0, -5.5, 6.5});
  Tensor y = tf.make({2, 1}, {10.0, 20.0});

  std::vector<Tensor> inputs = {x, y};

  Tensor out = tf.zeros({2, 4});

  // Concatenate along dim[1].
  Tensor ret = op_cat_out(
      ArrayRef<Tensor>(inputs.data(), inputs.size()), /*dim=*/1, out);

  Tensor expected =
      tf.make({2, 4}, {1.5, -2.0, 3.25, 10.0, 4.0, -5.5, 6.5, 20.0});
  EXPECT_TENSOR_EQ(out, expected);
}

TEST_F(OpCatOutTest, NegativeDims) {
  TensorFactory<ScalarType::Int> tf;

  // Symmetrical input tensors can can be concatenated along any dimension.
  // clang-format off
  Tensor x = tf.make(
      {2, 2},
      {
          1, 2,
          3, 4,
      });
  Tensor y = tf.make(
      {2, 2},
      {
          10, 20,
          30, 40,
      });
  // clang-format on

  std::vector<Tensor> inputs = {x, y};

  // Cat along dim[-1], which should be the same as dim[1].
  Tensor out_neg1 = tf.zeros({2, 4});
  op_cat_out(
      ArrayRef<Tensor>(inputs.data(), inputs.size()), /*dim=*/-1, out_neg1);

  Tensor out_1 = tf.zeros({2, 4});
  op_cat_out(ArrayRef<Tensor>(inputs.data(), inputs.size()), /*dim=*/1, out_1);

  EXPECT_TENSOR_EQ(out_neg1, out_1);

  // Cat along dim[-2], which should be the same as dim[0].
  Tensor out_neg2 = tf.zeros({4, 2});
  op_cat_out(
      ArrayRef<Tensor>(inputs.data(), inputs.size()), /*dim=*/-2, out_neg2);

  Tensor out_0 = tf.zeros({4, 2});
  op_cat_out(ArrayRef<Tensor>(inputs.data(), inputs.size()), /*dim=*/0, out_0);

  EXPECT_TENSOR_EQ(out_neg2, out_0);
}

/// A generic smoke test that works for any dtype that supports ones() and
/// zeros().
TEST_F(OpCatOutTest, AllDtypesSupported) {
#define TEST_ENTRY(ctype, dtype) test_dtype<ctype, ScalarType::dtype>();
  ET_FORALL_REAL_TYPES_AND(Bool, TEST_ENTRY);
#undef TEST_ENTRY
  // TODO: Also add tests for half, complex, quantized, and other types. Easiest
  // way to do that would be to make TensorFactory support zeros() and ones()
  // for those types.
}

TEST_F(OpCatOutTest, EmptyInputTensorShapeIgnored) {
  if (torch::executor::testing::SupportedFeatures::get()->is_aten) {
    GTEST_SKIP() << "ATen kernel doesn't ignore empty input tensor shape";
  }
  TensorFactory<ScalarType::Int> tf;

  // An empty tensor with a shape totally different from the non-empty inputs.
  Tensor empty = tf.make({0, 10, 3}, {});
  EXPECT_EQ(empty.numel(), 0);

  Tensor x = tf.ones({2, 2});

  std::vector<Tensor> inputs = {x, empty, x};

  // Output whose shape is appropriate for concatenating along dim[0].
  Tensor out = tf.zeros({4, 2});

  op_cat_out(ArrayRef<Tensor>(inputs.data(), inputs.size()), /*dim=*/0, out);
  // Success if it doesn't assert on the weird-shaped empty input.
}

TEST_F(OpCatOutTest, DimBounds) {
  TensorFactory<ScalarType::Int> tf;

  // Cat a single tensor, which can be done across any dimension and still
  // produces the same output shape.
  Tensor x = tf.ones({2, 2});
  ArrayRef<Tensor> inputs(&x, 1);

  Tensor out = tf.zeros({2, 2});

  // Some valid dim values.
  // Negative values work like python indices: -1 is the rightmost element,
  // -2 the second-from-rightmost, etc.
  const std::vector<int64_t> valid_dims = {0, 1, -1, -2};
  for (int64_t dim : valid_dims) {
    op_cat_out(inputs, dim, out);
    // Success if it doesn't assert.
  }

  // Some invalid dim values.
  const std::vector<int64_t> invalid_dims = {2, -3};
  for (int64_t dim : invalid_dims) {
    ET_EXPECT_KERNEL_FAILURE(context_, op_cat_out(inputs, dim, out));
  }
}

TEST_F(OpCatOutTest, NoInputTensorsWithNonEmptyOutputDies) {
  TensorFactory<ScalarType::Int> tf;
  Tensor out = tf.ones({1});

  // Providing an empty list of input tensors should
  // cause an assertion and kill the test process.
  ET_EXPECT_KERNEL_FAILURE(
      context_, op_cat_out(ArrayRef<Tensor>(), /*dim=*/0, out));
}

TEST_F(OpCatOutTest, NoInputTensorsWithEmptyOutputDies) {
  TensorFactory<ScalarType::Int> tf;

  // Make an empty out tensor and demonstrate that it's empty.
  Tensor out = tf.make({0}, {});
  EXPECT_EQ(out.numel(), 0);

  // Providing an empty list of input tensors should
  // cause an assertion and kill the test process.
  ET_EXPECT_KERNEL_FAILURE(
      context_, op_cat_out(ArrayRef<Tensor>(), /*dim=*/0, out));
}

TEST_F(OpCatOutTest, MismatchedDtypesDies) {
  TensorFactory<ScalarType::Int> tf_int;
  TensorFactory<ScalarType::Float> tf_float;
  Tensor out = tf_int.zeros({4, 2});

  // Same shape as the output, but a different dtype.
  std::vector<Tensor> inputs = {tf_float.ones({2, 2})};

  ET_EXPECT_KERNEL_FAILURE(
      context_,
      op_cat_out(
          ArrayRef<Tensor>(inputs.data(), inputs.size()), /*dim=*/0, out));
}

TEST_F(OpCatOutTest, MismatchedDimensionsDies) {
  if (torch::executor::testing::SupportedFeatures::get()->is_aten) {
    GTEST_SKIP() << "ATen kernel can handle mismatched dimensions";
  }
  TensorFactory<ScalarType::Int> tf;
  Tensor out = tf.zeros({2, 2});

  // Same dtype and numel as the output, but a different number of dimensions.
  std::vector<Tensor> inputs = {tf.ones({1, 1, 1, 1})};

  ET_EXPECT_KERNEL_FAILURE(
      context_,
      op_cat_out(
          ArrayRef<Tensor>(inputs.data(), inputs.size()), /*dim=*/0, out));
}

TEST_F(OpCatOutTest, MismatchedDimensionSizeDies) {
  if (torch::executor::testing::SupportedFeatures::get()->is_aten) {
    GTEST_SKIP() << "ATen kernel can handle mismatched dimension size";
  }
  TensorFactory<ScalarType::Int> tf;
  Tensor out = tf.zeros({2, 2});

  // Same dtype and number of dimensions as the output, but a different-sized 1
  // dimension.
  std::vector<Tensor> inputs = {tf.ones({2, 3})};

  ET_EXPECT_KERNEL_FAILURE(
      context_,
      op_cat_out(
          ArrayRef<Tensor>(inputs.data(), inputs.size()), /*dim=*/0, out));
}

TEST_F(OpCatOutTest, WrongOutShapeDies) {
  if (torch::executor::testing::SupportedFeatures::get()->is_aten) {
    GTEST_SKIP() << "ATen kernel can handle wrong out shape";
  }
  TensorFactory<ScalarType::Int> tf;

  // Should be {4, 3} to match the inputs when calling cat() with dim 0.
  Tensor out = tf.zeros({4, 5});

  std::vector<Tensor> inputs = {
      tf.ones({2, 3}),
      tf.ones({2, 3}),
  };

  ET_EXPECT_KERNEL_FAILURE(
      context_,
      op_cat_out(
          ArrayRef<Tensor>(inputs.data(), inputs.size()), /*dim=*/0, out));
}

/* %python
import torch
torch.manual_seed(0)
x = [torch.randint(10, (2, 3)),
     torch.randint(10, (2, 3)),
     torch.randint(10, (2, 3)),
     torch.randint(10, (2, 3))]
res = torch.cat(x, 0)
op = "op_cat_out"
opt_extra_params = "0,"
dtype = "ScalarType::Int"
check = "EXPECT_TENSOR_EQ" */

TEST_F(OpCatOutTest, DynamicShapeUpperBoundSameAsExpected) {
  /* %python
  out_args = "{8, 3}, torch::executor::TensorShapeDynamism::DYNAMIC_BOUND"
  %rewrite(unary_op_tensor_list_in) */

  TensorFactory<ScalarType::Int> tf;

  std::vector<Tensor> xv = {
      tf.make({2, 3}, {4, 9, 3, 0, 3, 9}),
      tf.make({2, 3}, {7, 3, 7, 3, 1, 6}),
      tf.make({2, 3}, {6, 9, 8, 6, 6, 8}),
      tf.make({2, 3}, {4, 3, 6, 9, 1, 4})};
  TensorList x(xv.data(), xv.size());
  Tensor expected = tf.make({8, 3}, {4, 9, 3, 0, 3, 9, 7, 3, 7, 3, 1, 6,
                                     6, 9, 8, 6, 6, 8, 4, 3, 6, 9, 1, 4});

  Tensor out =
      tf.zeros({8, 3}, torch::executor::TensorShapeDynamism::DYNAMIC_BOUND);
  op_cat_out(x, 0, out);
  EXPECT_TENSOR_EQ(out, expected);
}

TEST_F(OpCatOutTest, DynamicShapeUpperBoundLargerThanExpected) {
  /* %python
  out_args = "{10, 10}, torch::executor::TensorShapeDynamism::DYNAMIC_BOUND"
  %rewrite(unary_op_tensor_list_in) */

  TensorFactory<ScalarType::Int> tf;

  std::vector<Tensor> xv = {
      tf.make({2, 3}, {4, 9, 3, 0, 3, 9}),
      tf.make({2, 3}, {7, 3, 7, 3, 1, 6}),
      tf.make({2, 3}, {6, 9, 8, 6, 6, 8}),
      tf.make({2, 3}, {4, 3, 6, 9, 1, 4})};
  TensorList x(xv.data(), xv.size());
  Tensor expected = tf.make({8, 3}, {4, 9, 3, 0, 3, 9, 7, 3, 7, 3, 1, 6,
                                     6, 9, 8, 6, 6, 8, 4, 3, 6, 9, 1, 4});

  Tensor out =
      tf.zeros({10, 10}, torch::executor::TensorShapeDynamism::DYNAMIC_BOUND);
  op_cat_out(x, 0, out);
  EXPECT_TENSOR_EQ(out, expected);
}

TEST_F(OpCatOutTest, DynamicShapeUnbound) {
  if (!torch::executor::testing::SupportedFeatures::get()->output_resize) {
    GTEST_SKIP() << "Dynamic shape unbound not supported";
  }
  /* %python
  out_args = "{1, 1}, torch::executor::TensorShapeDynamism::DYNAMIC_UNBOUND"
  %rewrite(unary_op_tensor_list_in) */

  TensorFactory<ScalarType::Int> tf;

  std::vector<Tensor> xv = {
      tf.make({2, 3}, {4, 9, 3, 0, 3, 9}),
      tf.make({2, 3}, {7, 3, 7, 3, 1, 6}),
      tf.make({2, 3}, {6, 9, 8, 6, 6, 8}),
      tf.make({2, 3}, {4, 3, 6, 9, 1, 4})};
  TensorList x(xv.data(), xv.size());
  Tensor expected = tf.make({8, 3}, {4, 9, 3, 0, 3, 9, 7, 3, 7, 3, 1, 6,
                                     6, 9, 8, 6, 6, 8, 4, 3, 6, 9, 1, 4});

  Tensor out =
      tf.zeros({1, 1}, torch::executor::TensorShapeDynamism::DYNAMIC_UNBOUND);
  op_cat_out(x, 0, out);
  EXPECT_TENSOR_EQ(out, expected);
}
