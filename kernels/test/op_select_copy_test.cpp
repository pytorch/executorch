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
#include <sys/types.h>

using namespace ::testing;
using exec_aten::ArrayRef;
using exec_aten::ScalarType;
using exec_aten::Tensor;
using torch::executor::testing::TensorFactory;

class OpSelectCopyIntOutTest : public OperatorTest {
 protected:
  Tensor& op_select_copy_int_out(
      const Tensor& self,
      int64_t dim,
      int64_t index,
      Tensor& out) {
    return torch::executor::aten::select_copy_outf(
        context_, self, dim, index, out);
  }

  template <class CTYPE, exec_aten::ScalarType DTYPE>
  void test_dtype() {
    TensorFactory<DTYPE> tf;

    // Based on the split defintion, if we split any dim()=3 and size(1)=2
    // tensor along first dim to two tensors [ret_0, ret_1], the ret_0 and ret_1
    // shall be equal to x[:, 0, :] and x[:, 1, :] e.g. x[i, 0, j] = ret_0[i, j]
    // and x[i, 1, j] = ret_1[i, j] for any i in [-x.size(0), x.size(0)) and j
    // in
    // [-x.size(2), x.size(2))
    // Therefore we design the following tensor x for test easily: it is a
    // tensor formed by stacking tensors ones(3, 4) and zeros(3,4) along the
    // first dim. So if we select the tensor along the first dim by the above
    // rules, the ret_0 should be ones(3, 4) and ret_1 should be zeros(3, 4)

    // clang-format off
    Tensor x = tf.make(
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

    // Expected values for out_0 and ret_0 after the test are all ones(3, 4)
    // based on the above rules. So here we set the default value of out_0 as
    // zeros(3, 4) on purpose, to eliminate the influence to the final result
    // from initial value. Same for out_1 and ret_1.

    Tensor out_0 = tf.zeros({3, 4});
    Tensor out_1 = tf.ones({3, 4});
    Tensor ret_0 = op_select_copy_int_out(x, /*dim=*/1, /*index=*/0, out_0);
    Tensor ret_1 = op_select_copy_int_out(x, /*dim=*/1, /*index=*/1, out_1);

    EXPECT_TENSOR_EQ(ret_0, out_0);
    EXPECT_TENSOR_EQ(ret_1, out_1);

    EXPECT_TENSOR_EQ(ret_0, tf.ones({3, 4}));
    EXPECT_TENSOR_EQ(ret_1, tf.zeros({3, 4}));
  }

  // Run the test by selecting Tensor x on given dim and all available indexes
  // on that dimension
  void run_test_cases(
      const Tensor& x,
      ssize_t dim,
      const std::vector<Tensor>& expected) {
    // Generated out tensor sharing same size and dtype with expected tensor
    TensorFactory<ScalarType::Double> tf;

    const std::vector<int32_t> out_size(
        expected[0].sizes().begin(), expected[0].sizes().end());
    Tensor out = tf.ones(out_size);

    for (ssize_t idx = 0; idx < x.size(dim); idx++) {
      // Should always return the provided out Tensor.
      // The ret shall meet the expectation.
      Tensor ret = op_select_copy_int_out(x, dim, idx, out);
      EXPECT_TENSOR_EQ(out, ret);
      EXPECT_TENSOR_EQ(out, expected[idx]);

      ret = op_select_copy_int_out(x, dim, /*index=*/idx - x.size(dim), out);
      EXPECT_TENSOR_EQ(out, ret);

      EXPECT_TENSOR_EQ(out, expected[idx]);
    }
  }
};

TEST_F(OpSelectCopyIntOutTest, SelectFrontDimAllIndexes) {
  TensorFactory<ScalarType::Double> tf;

  // clang-format off
  Tensor x = tf.make(
      {2, 3, 4},
      {
          // [0, :, :]
          1.,   2.,   3.,   4., // [0, 0, :]
          5.,   6.,   7.,   8., // [0, 1, :]
          9.,  10.,  11.,  12., // [0, 2, :]

          // [1, :, :]
         -1.,  -2.,  -3.,  -4., // [1, 0, :]
         -5.,  -6.,  -7.,  -8., // [1, 1, :]
         -9., -10., -11., -12., // [1, 2, :]
      });
  // clang-format on

  // Try to select the tensor from the input front (0th dimension)
  // The size of output tensor should follow these rules:
  // - output.size(i) shall equal input.size(i) if i < dim,
  // - output.size(i) shall equal input.size(i+1) if i >= dim
  const std::vector<int32_t> out_size = {3, 4};

  Tensor out = tf.zeros(out_size);

  // clang-format off
  std::vector<Tensor> expected_rets = {
    // Expected result when choosing from the 0th dimension and 0th index
    // The result should equal x[0，:, :]
    tf.make(
      out_size,
      {
        1.,   2.,   3.,   4., // [0, :]
        5.,   6.,   7.,   8., // [1, :]
        9.,  10.,  11.,  12., // [2, :]
      }),

    // Expected result when choosing from the 0th dimension and 1st index
    // The result should euqal x[1, :, :]
    tf.make(
      out_size,
      {
        -1.,  -2.,  -3.,  -4., // [0, :]
        -5.,  -6.,  -7.,  -8., // [1, :]
        -9., -10., -11., -12., // [2, :]
      })
  };
  // clang-format on

  run_test_cases(x, /*dim=*/0, expected_rets);
}

TEST_F(OpSelectCopyIntOutTest, SelectMiddleDimAllIndexes) {
  TensorFactory<ScalarType::Double> tf;

  // clang-format off
  Tensor x = tf.make(
      {2, 3, 4},
      {
          // [0, :, :]
          1.,   2.,   3.,   4., // [0, 0, :]
          5.,   6.,   7.,   8., // [0, 1, :]
          9.,  10.,  11.,  12., // [0, 2, :]

          // [1, :, :]
         -1.,  -2.,  -3.,  -4., // [1, 0, :]
         -5.,  -6.,  -7.,  -8., // [1, 1, :]
         -9., -10., -11., -12., // [1, 2, :]
      });
  // clang-format on

  // Try to select the tensor from the input front (0th dimension)
  // The size of output tensor should follow these rules:
  // - output.size(i) shall equal input.size(i) if i < dim,
  // - output.size(i) shall equal input.size(i+1) if i >= dim
  const std::vector<int32_t> out_size = {2, 4};

  Tensor out = tf.zeros(out_size);

  // clang-format off
  std::vector<Tensor> expected_rets = {
    // Expected result when choosing from the 1st dimension and 0th index
    // The result should equal x[:，0, :]
    tf.make(
      out_size,
      {
         1.,   2.,   3.,   4., // [0, :]
        -1.,  -2.,  -3.,  -4., // [1, :]
      }),
    // Expected result when choosing from the 1st dimension and 1st index
    // The result should equal x[:, 1, :]
    tf.make(
      out_size,
      {
         5.,   6.,   7.,   8., // [0, :]
        -5.,  -6.,  -7.,  -8., // [1, :]
      }),
    // Expected result when choosing from the 1st dimension and 2th index
    // The result should equal x[:，2, :]
    tf.make(
      out_size,
      {
         9.,  10.,  11.,  12., // [0, :]
        -9., -10., -11., -12., // [1, :]
      })
  };
  // clang-format on

  run_test_cases(x, /*dim=*/1, expected_rets);
}

TEST_F(OpSelectCopyIntOutTest, SelectEndDimAllIndexes) {
  TensorFactory<ScalarType::Double> tf;

  // clang-format off
  Tensor x = tf.make(
    {2, 3, 4},
    {
      // [0, :, :]
      1.,   2.,   3.,   4., // [0, 0, :]
      5.,   6.,   7.,   8., // [0, 1, :]
      9.,  10.,  11.,  12., // [0, 2, :]

      // [1, :, :]
      -1.,  -2.,  -3.,  -4., // [1, 0, :]
      -5.,  -6.,  -7.,  -8., // [1, 1, :]
      -9., -10., -11., -12., // [1, 2, :]
    });
  // clang-format on

  // Try to select the tensor from the input front (0th dimension)
  // The size of output tensor should follow these rules:
  // - output.size(i) shall equal input.size(i) if i < dim,
  // - output.size(i) shall equal input.size(i+1) if i >= dim
  const std::vector<int32_t> out_size = {2, 3};

  Tensor out = tf.zeros(out_size);

  // clang-format off
  std::vector<Tensor> expected_rets = {
    // Expected result when choosing from the 2nd dimension and 0th index
    // The result should equal x[:，:, 0] (a.k.a 0th column of x data layout)
    tf.make(
      out_size,
      {
         1.,  5.,  9.,  // [0, :]
        -1., -5., -9.,  // [1, :]
      }),
    // Expected result when choosing from the 2nd dimension and 1st index
    // The result should equal x[:，:, 1] (a.k.a 1st column of x data layout)
    tf.make(
      out_size,
      {
         2.,  6.,  10.,  // [0, :]
        -2., -6., -10.,  // [1, :]
      }),
    // Expected result when choosing from the 2nd dimension and 2nd index
    // The result should equal x[:，:, 2] (a.k.a 2nd column of x data layout)
    tf.make(
      out_size,
      {
         3.,  7.,  11.,  // [0, :]
        -3., -7., -11.,  // [1, :]
      }),
    // Expected result when choosing from the 2nd dimension and 3rd index
    // The result should equal x[:，:, 3] (a.k.a 3rd column of x data layout)
    tf.make(
      out_size,
      {
         4.,  8.,  12.,  // [0, :]
        -4., -8., -12.,  // [1, :]
      })
  };
  // clang-format on

  run_test_cases(x, /*dim=*/2, expected_rets);
}

/// A generic smoke test that works for any dtype that supports ones() and
/// zeros().
TEST_F(OpSelectCopyIntOutTest, AllDtypesSupported) {
#define TEST_ENTRY(ctype, dtype) test_dtype<ctype, ScalarType::dtype>();
  ET_FORALL_REAL_TYPES_AND(Bool, TEST_ENTRY);
#undef TEST_ENTRY
  // TODO: Also add tests for half, complex, quantized, and other types. Easiest
  // way to do that would be to make TensorFactory support zeros() and ones()
  // for those types.
}

//////////////////////////////////////////////////////////////////////////////
// The following tests focus on empty-size tensor and empty tensor.
// Here we first define the term:
// empty-size tensor: size is [] but do have data (e.g.tensor(5))
// empty tensor: size is not [] and the size of at least one
// dim is zero, and does not have data in it (e.g ones(1,0,2,3))

// In this test we are gonnna find if our select function support vector tensor
// input and empty-size tensor output. Such combination is quite normal in real
// world (e.g. select(torch.range(10), 0, 5, out) == tensor(5))
TEST_F(OpSelectCopyIntOutTest, VectorInputSupported) {
  TensorFactory<ScalarType::Int> tf;

  Tensor x = tf.make({10}, {0, 1, 2, 3, 4, 5, 6, 7, 8, 9});

  // Make an empty-size out tensor and demonstrate that it has data.
  Tensor out = tf.make({}, {0});
  EXPECT_EQ(out.numel(), 1);

  // pass the empty-size tensor to the function,
  Tensor expect = tf.make({}, {5});
  op_select_copy_int_out(x, /*dim=*/0, /*index=*/5, out);
  EXPECT_TENSOR_EQ(out, expect);
}

// This test focuses on the support for empty tensor (dim() > 0) input and empty
// tensor output
TEST_F(OpSelectCopyIntOutTest, EmptyTensorNonZeroNDimsInputSupported) {
  TensorFactory<ScalarType::Int> tf;

  // Using empty tensors as input.
  Tensor x = tf.make({3, 0, 10, 3}, {});
  EXPECT_EQ(x.numel(), 0);

  // Output whose shape is appropriate for selecting along dim(2)
  Tensor out = tf.make({3, 0, 3}, {});
  EXPECT_EQ(out.numel(), 0);

  Tensor ret = op_select_copy_int_out(x, /*dim=*/2, /*index=*/3, out);
  EXPECT_EQ(ret.numel(), 0);
  // Success if it doesn't assert on the weird-shaped empty input and the
  // ret is still a empty array
}

// Apply select on dim() == 0 empty tensor input and empty tensor output
TEST_F(OpSelectCopyIntOutTest, EmptyTensorZeroNDimsInputDies) {
  TensorFactory<ScalarType::Int> tf;

  // Using empty tensors as input.
  Tensor x = tf.make({0}, {});
  EXPECT_EQ(x.numel(), 0);

  // Output whose shape is appropriate for selecting along dim(0)
  Tensor out = tf.make({}, {0});
  EXPECT_EQ(out.numel(), 1);

  // Expected failure when slicing on the dimension with length 0 since no space
  // on the dimension could be sliced. (out of bound error)
  ET_EXPECT_KERNEL_FAILURE(
      context_, op_select_copy_int_out(x, /*dim=*/0, /*index=*/0, out));
}
///////////////////////////////////////////////////////////////////////

TEST_F(OpSelectCopyIntOutTest, DimOutOfBoundDies) {
  TensorFactory<ScalarType::Int> tf;

  Tensor x = tf.ones({1, 1, 1});
  Tensor out = tf.zeros({1, 1});

  // Some invalid dim values.
  const std::vector<int32_t> invalid_dims = {3, 4, 5, -4, -5, -6};
  for (ssize_t dim : invalid_dims) {
    ET_EXPECT_KERNEL_FAILURE(
        context_, op_select_copy_int_out(x, dim, /*index=*/0, out));
  }
}

TEST_F(OpSelectCopyIntOutTest, MismatchedDtypesDies) {
  TensorFactory<ScalarType::Int> tf_int;
  TensorFactory<ScalarType::Float> tf_float;
  Tensor x = tf_int.zeros({1, 2, 2});

  // Size is compatible to the output, but a mismatched dtype.
  Tensor out = tf_float.ones({2, 2});

  ET_EXPECT_KERNEL_FAILURE(
      context_, op_select_copy_int_out(x, /*dim=*/0, /*index=*/0, out));
}

TEST_F(OpSelectCopyIntOutTest, OutMatchNumelLackDimAtEndDies) {
  if (torch::executor::testing::SupportedFeatures::get()->is_aten) {
    GTEST_SKIP() << "ATen kernel can handle out with mismatched dimensions";
  }
  TensorFactory<ScalarType::Int> tf;
  Tensor x = tf.zeros({1, 2, 2, 1});

  // Out shares the same dtype and numel as the expected output, but a
  // mixmatched size (out.dim() should always one lower than x.dim())
  Tensor out = tf.ones({2, 2});

  ET_EXPECT_KERNEL_FAILURE(
      context_, op_select_copy_int_out(x, /*dim=*/0, /*index=*/0, out));
}

TEST_F(OpSelectCopyIntOutTest, OutMatchNumelExtraDimAtFrontDies) {
  if (torch::executor::testing::SupportedFeatures::get()->is_aten) {
    GTEST_SKIP() << "ATen kernel can handle out with mismatched dimensions";
  }
  TensorFactory<ScalarType::Int> tf;
  Tensor x = tf.zeros({2, 2});

  // Out shares the same dtype and numel as the expected output, but a
  // mixmatched size (out.dim() should always one lower than x.dim())
  Tensor out = tf.ones({1, 2});

  ET_EXPECT_KERNEL_FAILURE(
      context_, op_select_copy_int_out(x, /*dim=*/0, /*index=*/0, out));
}

TEST_F(OpSelectCopyIntOutTest, OutSizeMismatchDimDies) {
  if (torch::executor::testing::SupportedFeatures::get()->is_aten) {
    GTEST_SKIP() << "ATen kernel can handle out with mismatched dimensions";
  }
  TensorFactory<ScalarType::Int> tf;

  Tensor x = tf.zeros({2, 4, 7, 5});

  // Should be {2, 4, 5} to match the x when calling select() with dim 2.
  Tensor out = tf.zeros({2, 4, 7});

  ET_EXPECT_KERNEL_FAILURE(
      context_, op_select_copy_int_out(x, /*dim=*/2, /*index=*/3, out));
}

/* %python
import torch
torch.manual_seed(0)
x = torch.rand(2, 3, 4)
res = torch.select(x, 1, 2)
op = "op_select_copy_int_out"
opt_extra_params = "1, 2,"
dtype = "ScalarType::Float"
check = "EXPECT_TENSOR_EQ" */

TEST_F(OpSelectCopyIntOutTest, DynamicShapeUpperBoundSameAsExpected) {
  /* %python
  out_args = "{2, 4}, torch::executor::TensorShapeDynamism::DYNAMIC_BOUND"
  %rewrite(unary_op) */

  TensorFactory<ScalarType::Float> tf;

  Tensor x = tf.make(
      {2, 3, 4},
      {0.49625658988952637,  0.7682217955589294,  0.08847743272781372,
       0.13203048706054688,  0.30742281675338745, 0.6340786814689636,
       0.4900934100151062,   0.8964447379112244,  0.455627977848053,
       0.6323062777519226,   0.3488934636116028,  0.40171730518341064,
       0.022325754165649414, 0.16885894536972046, 0.2938884496688843,
       0.518521785736084,    0.6976675987243652,  0.800011396408081,
       0.16102945804595947,  0.28226858377456665, 0.6816085577011108,
       0.9151939749717712,   0.39709991216659546, 0.8741558790206909});
  Tensor expected = tf.make(
      {2, 4},
      {0.455627977848053,
       0.6323062777519226,
       0.3488934636116028,
       0.40171730518341064,
       0.6816085577011108,
       0.9151939749717712,
       0.39709991216659546,
       0.8741558790206909});

  Tensor out =
      tf.zeros({2, 4}, torch::executor::TensorShapeDynamism::DYNAMIC_BOUND);
  op_select_copy_int_out(x, 1, 2, out);
  EXPECT_TENSOR_EQ(out, expected);
}

TEST_F(OpSelectCopyIntOutTest, DynamicShapeUpperBoundLargerThanExpected) {
  if (!torch::executor::testing::SupportedFeatures::get()->output_resize) {
    GTEST_SKIP() << "Dynamic shape not supported";
  }
  /* %python
  out_args = "{5, 5}, torch::executor::TensorShapeDynamism::DYNAMIC_BOUND"
  %rewrite(unary_op) */

  TensorFactory<ScalarType::Float> tf;

  Tensor x = tf.make(
      {2, 3, 4},
      {0.49625658988952637,  0.7682217955589294,  0.08847743272781372,
       0.13203048706054688,  0.30742281675338745, 0.6340786814689636,
       0.4900934100151062,   0.8964447379112244,  0.455627977848053,
       0.6323062777519226,   0.3488934636116028,  0.40171730518341064,
       0.022325754165649414, 0.16885894536972046, 0.2938884496688843,
       0.518521785736084,    0.6976675987243652,  0.800011396408081,
       0.16102945804595947,  0.28226858377456665, 0.6816085577011108,
       0.9151939749717712,   0.39709991216659546, 0.8741558790206909});
  Tensor expected = tf.make(
      {2, 4},
      {0.455627977848053,
       0.6323062777519226,
       0.3488934636116028,
       0.40171730518341064,
       0.6816085577011108,
       0.9151939749717712,
       0.39709991216659546,
       0.8741558790206909});

  Tensor out =
      tf.zeros({5, 5}, torch::executor::TensorShapeDynamism::DYNAMIC_BOUND);
  op_select_copy_int_out(x, 1, 2, out);
  EXPECT_TENSOR_EQ(out, expected);
}

TEST_F(OpSelectCopyIntOutTest, DynamicShapeUnbound) {
  if (!torch::executor::testing::SupportedFeatures::get()->output_resize) {
    GTEST_SKIP() << "Dynamic shape not supported";
  }
  /* %python
  out_args = "{1, 1}, torch::executor::TensorShapeDynamism::DYNAMIC_UNBOUND"
  %rewrite(unary_op) */

  TensorFactory<ScalarType::Float> tf;

  Tensor x = tf.make(
      {2, 3, 4},
      {0.49625658988952637,  0.7682217955589294,  0.08847743272781372,
       0.13203048706054688,  0.30742281675338745, 0.6340786814689636,
       0.4900934100151062,   0.8964447379112244,  0.455627977848053,
       0.6323062777519226,   0.3488934636116028,  0.40171730518341064,
       0.022325754165649414, 0.16885894536972046, 0.2938884496688843,
       0.518521785736084,    0.6976675987243652,  0.800011396408081,
       0.16102945804595947,  0.28226858377456665, 0.6816085577011108,
       0.9151939749717712,   0.39709991216659546, 0.8741558790206909});
  Tensor expected = tf.make(
      {2, 4},
      {0.455627977848053,
       0.6323062777519226,
       0.3488934636116028,
       0.40171730518341064,
       0.6816085577011108,
       0.9151939749717712,
       0.39709991216659546,
       0.8741558790206909});

  Tensor out =
      tf.zeros({1, 1}, torch::executor::TensorShapeDynamism::DYNAMIC_UNBOUND);
  op_select_copy_int_out(x, 1, 2, out);
  EXPECT_TENSOR_EQ(out, expected);
}
