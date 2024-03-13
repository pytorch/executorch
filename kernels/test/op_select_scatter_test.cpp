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

class OpSelectScatterOutTest : public OperatorTest {
 protected:
  Tensor& op_select_scatter_out(
      const Tensor& self,
      const Tensor& src,
      int64_t dim,
      int64_t index,
      Tensor& out) {
    return torch::executor::aten::select_scatter_outf(
        context_, self, src, dim, index, out);
  }

  template <class CTYPE, exec_aten::ScalarType DTYPE>
  void test_dtype() {
    TensorFactory<DTYPE> tf;

    // Using the following tensors, inserting a tensor of either ones or zeros
    // into the appropriate selected slice should result in a tensor of all ones
    // or all zeros.

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

    // clang-format off
    Tensor src_ones = tf.make(
        {3, 4},
        {
            // [:, :]
            1,  1,  1,  1, // [0, :]
            1,  1,  1,  1, // [1, :]
            1,  1,  1,  1, // [2, :]
        });
    // clang-format on

    // clang-format off
    Tensor src_zeros = tf.make(
        {3, 4},
        {
            // [:, :]
            0,  0,  0,  0, // [0, :]
            0,  0,  0,  0, // [1, :]
            0,  0,  0,  0, // [2, :]
        });
    // clang-format on

    // Expected outs should be all ones or all zeros depending on which src
    // tensor is used.

    Tensor out_0 = tf.zeros({3, 2, 4});
    Tensor out_1 = tf.ones({3, 2, 4});
    Tensor ret_0 =
        op_select_scatter_out(x, src_zeros, /*dim=*/1, /*index=*/0, out_0);
    Tensor ret_1 =
        op_select_scatter_out(x, src_ones, /*dim=*/1, /*index=*/1, out_1);

    EXPECT_TENSOR_EQ(ret_0, out_0);
    EXPECT_TENSOR_EQ(ret_1, out_1);

    EXPECT_TENSOR_EQ(ret_0, tf.zeros({3, 2, 4}));
    EXPECT_TENSOR_EQ(ret_1, tf.ones({3, 2, 4}));
  }

  // Run the test by selecting Tensor x on given dim and all available indexes
  // on that dimension
  void run_test_cases(
      const Tensor& x,
      const Tensor& src,
      ssize_t dim,
      const std::vector<Tensor>& expected) {
    // Generated out tensor sharing same size and dtype with expected tensor
    TensorFactory<ScalarType::Double> tf;

    const std::vector<int32_t> out_size(
        expected[0].sizes().begin(), expected[0].sizes().end());
    Tensor out = tf.zeros(out_size);

    for (ssize_t idx = 0; idx < x.size(dim); idx++) {
      // Should always return the provided out Tensor.
      // The ret shall meet the expectation.
      Tensor ret = op_select_scatter_out(x, src, dim, idx, out);
      EXPECT_TENSOR_EQ(out, ret);
      EXPECT_TENSOR_EQ(out, expected[idx]);

      ret =
          op_select_scatter_out(x, src, dim, /*index=*/idx - x.size(dim), out);
      EXPECT_TENSOR_EQ(out, ret);
      EXPECT_TENSOR_EQ(out, expected[idx]);
    }
  }

  /* %python
  import torch
  torch.manual_seed(0)
  x = torch.randint(10, (2, 3, 2))
  y = torch.randint(10, (3, 2))
  dim = 0
  index = 1
  res = torch.select_scatter(x, y, dim, index)
  op = "op_select_scatter_out"
  opt_extra_params = f"""{dim}, {index},"""
  out_args = "out_shape, dynamism"
  dtype = "ScalarType::Int"
  check = "EXPECT_TENSOR_CLOSE" */

  void test_dynamic_shape(
      const std::vector<int32_t>& out_shape,
      enum torch::executor::TensorShapeDynamism dynamism) {
    /* %python
    %rewrite(binary_op) */

    TensorFactory<ScalarType::Int> tf;

    Tensor x = tf.make({2, 3, 2}, {4, 9, 3, 0, 3, 9, 7, 3, 7, 3, 1, 6});
    Tensor y = tf.make({3, 2}, {6, 9, 8, 6, 6, 8});
    Tensor expected = tf.make({2, 3, 2}, {4, 9, 3, 0, 3, 9, 6, 9, 8, 6, 6, 8});

    Tensor out = tf.zeros(out_shape, dynamism);
    op_select_scatter_out(x, y, 0, 1, out);
    EXPECT_TENSOR_CLOSE(out, expected);
  }
};

TEST_F(OpSelectScatterOutTest, SelectFrontDimAllIndexes) {
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

  // clang-format off
  Tensor src = tf.make(
      {3, 4},
      {
          // [0, :, :]
          1.,  4.,  1.,  4., // [0, 0, :]
          1.,  4.,  1.,  4., // [0, 1, :]
          1.,  4.,  1.,  4., // [0, 2, :]
      });
  // clang-format on

  // Try to select the tensor from the input front (0th dimension)
  // The size of output tensor should follow these rules:
  // - output.size(i) shall equal input.size(i) if i < dim,
  // - output.size(i) shall equal input.size(i+1) if i >= dim
  const std::vector<int32_t> out_size = {2, 3, 4};

  Tensor out = tf.zeros(out_size);

  // clang-format off
  std::vector<Tensor> expected_rets = {
    // Expected result when choosing from the 0th dimension and 0th index
    // The result should equal x[0，:, :]
    tf.make(
      out_size,
      {
         // [0, :, :]
         1.,  4.,  1.,  4., // [0, 0, :]
         1.,  4.,  1.,  4., // [0, 1, :]
         1.,  4.,  1.,  4., // [0, 2, :]

         // [1, :, :]
        -1.,  -2.,  -3.,  -4., // [1, 0, :]
        -5.,  -6.,  -7.,  -8., // [1, 1, :]
        -9., -10., -11., -12., // [1, 2, :]
      }),

    // Expected result when choosing from the 0th dimension and 1st index
    // The result should euqal x[1, :, :]
    tf.make(
      out_size,
      {
        // [0, :, :]
        1.,   2.,   3.,   4., // [0, 0, :]
        5.,   6.,   7.,   8., // [0, 1, :]
        9.,  10.,  11.,  12., // [0, 2, :]

        // [1, :, :]
        1.,  4.,  1.,  4., // [1, 0, :]
        1.,  4.,  1.,  4., // [1, 1, :]
        1.,  4.,  1.,  4., // [1, 2, :]
      })
  };
  // clang-format on

  run_test_cases(x, src, /*dim=*/0, expected_rets);
}

TEST_F(OpSelectScatterOutTest, SelectMiddleDimAllIndexes) {
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

  // clang-format off
  Tensor src = tf.make(
      {2, 4},
      {
          // [0, :, :]
          1.,  4.,  1.,  4., // [0, 0, :]
          1.,  4.,  1.,  4., // [0, 2, :]
      });
  // clang-format on

  // Try to select the tensor from the input front (0th dimension)
  // The size of output tensor should follow these rules:
  // - output.size(i) shall equal input.size(i) if i < dim,
  // - output.size(i) shall equal input.size(i+1) if i >= dim
  const std::vector<int32_t> out_size = {2, 3, 4};

  Tensor out = tf.zeros(out_size);

  // clang-format off
  std::vector<Tensor> expected_rets = {
    // Expected result when choosing from the 1st dimension and 0th index
    // The result should equal x[:，0, :]
    tf.make(
      out_size,
      {
         // [0, :, :]
         1.,   4.,   1.,   4., // [0, 0, :]
         5.,   6.,   7.,   8., // [0, 1, :]
         9.,  10.,  11.,  12., // [0, 2, :]

         // [1, :, :]
         1.,   4.,   1.,   4., // [1, 0, :]
        -5.,  -6.,  -7.,  -8., // [1, 1, :]
        -9., -10., -11., -12., // [1, 2, :]
      }),
    // Expected result when choosing from the 1st dimension and 1st index
    // The result should equal x[:, 1, :]
    tf.make(
      out_size,
      {
         // [0, :, :]
         1.,   2.,   3.,   4., // [0, 0, :]
         1.,   4.,   1.,   4., // [0, 1, :]
         9.,  10.,  11.,  12., // [0, 2, :]

         // [1, :, :]
        -1.,  -2.,  -3.,  -4., // [1, 0, :]
         1.,   4.,   1.,   4., // [1, 1, :]
        -9., -10., -11., -12., // [1, 2, :]
      }),
    // Expected result when choosing from the 1st dimension and 2th index
    // The result should equal x[:，2, :]
    tf.make(
      out_size,
      {
         // [0, :, :]
         1.,   2.,   3.,   4., // [0, 0, :]
         5.,   6.,   7.,   8., // [0, 1, :]
         1.,   4.,   1.,   4., // [0, 2, :]

         // [1, :, :]
        -1.,  -2.,  -3.,  -4., // [1, 0, :]
        -5.,  -6.,  -7.,  -8., // [1, 1, :]
         1.,   4.,   1.,   4., // [1, 2, :]
      })
  };
  // clang-format on

  run_test_cases(x, src, /*dim=*/1, expected_rets);
}

TEST_F(OpSelectScatterOutTest, SelectEndDimAllIndexes) {
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

  // clang-format off
  Tensor src = tf.make(
    {2, 3},
    {
        // [0, :, :]
        1.,  4.,  1., // [0, 0, :]
        1.,  4.,  1., // [0, 1, :]
    });
  // clang-format on

  // Try to select the tensor from the input front (0th dimension)
  // The size of output tensor should follow these rules:
  // - output.size(i) shall equal input.size(i) if i < dim,
  // - output.size(i) shall equal input.size(i+1) if i >= dim
  const std::vector<int32_t> out_size = {2, 3, 4};

  Tensor out = tf.zeros(out_size);

  // clang-format off
  std::vector<Tensor> expected_rets = {
    // Expected result when choosing from the 2nd dimension and 0th index
    // The result should equal x[:，:, 0] (a.k.a 0th column of x data layout)
    tf.make(
      out_size,
      {
        // [0, :, :]
        1.,   2.,   3.,   4., // [0, 0, :]
        4.,   6.,   7.,   8., // [0, 1, :]
        1.,  10.,  11.,  12., // [0, 2, :]

        // [1, :, :]
        1.,  -2.,  -3.,  -4., // [1, 0, :]
        4.,  -6.,  -7.,  -8., // [1, 1, :]
        1., -10., -11., -12., // [1, 2, :]
      }),
    // Expected result when choosing from the 2nd dimension and 1st index
    // The result should equal x[:，:, 1] (a.k.a 1st column of x data layout)
    tf.make(
      out_size,
      {
         // [0, :, :]
         1.,  1.,   3.,   4., // [0, 0, :]
         5.,  4.,   7.,   8., // [0, 1, :]
         9.,  1.,  11.,  12., // [0, 2, :]

         // [1, :, :]
        -1.,  1.,  -3.,  -4., // [1, 0, :]
        -5.,  4.,  -7.,  -8., // [1, 1, :]
        -9.,  1., -11., -12., // [1, 2, :]
      }),
    // Expected result when choosing from the 2nd dimension and 2nd index
    // The result should equal x[:，:, 2] (a.k.a 2nd column of x data layout)
    tf.make(
      out_size,
      {
         // [0, :, :]
         1.,   2.,  1.,   4., // [0, 0, :]
         5.,   6.,  4.,   8., // [0, 1, :]
         9.,  10.,  1.,  12., // [0, 2, :]

         // [1, :, :]
        -1.,  -2.,  1.,  -4., // [1, 0, :]
        -5.,  -6.,  4.,  -8., // [1, 1, :]
        -9., -10.,  1., -12., // [1, 2, :]
      }),
    // Expected result when choosing from the 2nd dimension and 3rd index
    // The result should equal x[:，:, 3] (a.k.a 3rd column of x data layout)
    tf.make(
      out_size,
      {
         // [0, :, :]
         1.,   2.,   3.,  1., // [0, 0, :]
         5.,   6.,   7.,  4., // [0, 1, :]
         9.,  10.,  11.,  1., // [0, 2, :]

         // [1, :, :]
        -1.,  -2.,  -3.,  1., // [1, 0, :]
        -5.,  -6.,  -7.,  4., // [1, 1, :]
        -9., -10., -11.,  1., // [1, 2, :]
      })
  };
  // clang-format on

  run_test_cases(x, src, /*dim=*/2, expected_rets);
}

#ifndef USE_ATEN_LIB
// Same test as above, but this time the output size is slightly off
TEST_F(OpSelectScatterOutTest, OutputDynamicShape) {
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

  // clang-format off
  Tensor src = tf.make(
    {2, 3},
    {
        // [0, :, :]
        1.,  4.,  1., // [0, 0, :]
        1.,  4.,  1., // [0, 1, :]
    });
  // clang-format on

  // In this case, the output starts off with a different shape than is
  // expected. We are checking to see that dynamic shape support is working
  // correctly and that the output will be resized to the correct shape inside
  // the kernel.
  const std::vector<int32_t> out_size = {2, 6, 2};
  const std::vector<int32_t> actual_out_size = {2, 3, 4};

  Tensor out =
      tf.zeros(out_size, torch::executor::TensorShapeDynamism::DYNAMIC_BOUND);

  // clang-format off
  Tensor expected_ret = tf.make(
    actual_out_size,
    {
      // [0, :, :]
      1.,   2.,   3.,   4., // [0, 0, :]
      4.,   6.,   7.,   8., // [0, 1, :]
      1.,  10.,  11.,  12., // [0, 2, :]

      // [1, :, :]
      1.,  -2.,  -3.,  -4., // [1, 0, :]
      4.,  -6.,  -7.,  -8., // [1, 1, :]
      1., -10., -11., -12., // [1, 2, :]
    });
  // clang-format on

  Tensor ret = op_select_scatter_out(x, src, 2, 0, out);
  EXPECT_TENSOR_EQ(out, ret);
  EXPECT_TENSOR_EQ(out, expected_ret);
}
#endif

/// A generic smoke test that works for any dtype that supports ones() and
/// zeros().
TEST_F(OpSelectScatterOutTest, AllDtypesSupported) {
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

// This test focuses on the support for empty tensor (dim() > 0) input and empty
// tensor output
TEST_F(OpSelectScatterOutTest, EmptyTensorNonZeroNDimsInputSupported) {
  TensorFactory<ScalarType::Int> tf;

  // Using empty tensors as input.
  Tensor x = tf.make({3, 0, 10, 3}, {});
  EXPECT_EQ(x.numel(), 0);

  // src tensor whose shape is appropriate to place in dim(2) of x
  Tensor src = tf.make({3, 0, 3}, {});

  // Output whose shape is equal to the input shape
  Tensor out = tf.make({3, 0, 10, 3}, {});
  EXPECT_EQ(out.numel(), 0);

  Tensor ret = op_select_scatter_out(x, src, /*dim=*/2, /*index=*/3, out);
  EXPECT_EQ(ret.numel(), 0);
  // Success if it doesn't assert on the weird-shaped empty input and the
  // ret is still a empty array
}

// Apply select on dim() == 0 empty tensor input and empty tensor output
TEST_F(OpSelectScatterOutTest, EmptyTensorZeroNDimsInputDies) {
  TensorFactory<ScalarType::Int> tf;

  // Using empty tensors as input.
  Tensor x = tf.make({0}, {});
  EXPECT_EQ(x.numel(), 0);

  // Using empty src tensor
  Tensor src = tf.make({0}, {});
  EXPECT_EQ(src.numel(), 0);

  // Output whose shape is equal to the input shape
  Tensor out = tf.make({}, {0});
  EXPECT_EQ(out.numel(), 1);

  // Expected failure when slicing on the dimension with length 0 since no space
  // on the dimension could be sliced. (out of bound error)
  ET_EXPECT_KERNEL_FAILURE(
      context_, op_select_scatter_out(x, src, /*dim=*/0, /*index=*/0, out));
}
///////////////////////////////////////////////////////////////////////

TEST_F(OpSelectScatterOutTest, DimOutOfBoundDies) {
  TensorFactory<ScalarType::Int> tf;

  Tensor x = tf.ones({1, 1, 1});
  Tensor src = tf.ones({1, 1});

  Tensor out = tf.zeros({1, 1, 1});

  // Some invalid dim values.
  const std::vector<int32_t> invalid_dims = {3, 4, 5, -4, -5, -6};
  for (ssize_t dim : invalid_dims) {
    ET_EXPECT_KERNEL_FAILURE(
        context_, op_select_scatter_out(x, src, dim, /*index=*/0, out));
  }
}

TEST_F(OpSelectScatterOutTest, IndexOutOfBoundDies) {
  TensorFactory<ScalarType::Int> tf;

  Tensor x = tf.ones({1, 1, 1});
  Tensor src = tf.ones({1, 1});

  Tensor out = tf.zeros({1, 1, 1});

  // Some invalid dim values.
  const std::vector<int32_t> invalid_indices = {3, 4, 5, -4, -5, -6};
  for (ssize_t idx : invalid_indices) {
    ET_EXPECT_KERNEL_FAILURE(
        context_, op_select_scatter_out(x, src, /*dim=*/0, idx, out));
  }
}

TEST_F(OpSelectScatterOutTest, MismatchedDtypesDies) {
  TensorFactory<ScalarType::Int> tf_int;
  TensorFactory<ScalarType::Float> tf_float;
  Tensor x = tf_int.zeros({1, 2, 2});
  Tensor src = tf_int.zeros({2, 2});

  // Size is compatible to the output, but a mismatched dtype.
  Tensor out = tf_float.ones({1, 2, 2});

  ET_EXPECT_KERNEL_FAILURE(
      context_, op_select_scatter_out(x, src, /*dim=*/0, /*index=*/0, out));
}

TEST_F(OpSelectScatterOutTest, SrcMatchNumelLackDimAtEndDies) {
  TensorFactory<ScalarType::Int> tf;
  Tensor x = tf.zeros({1, 2, 2, 1});
  // src shares the same dtype and numel as the selected slice, but the wrong
  // size (src.dim() should always one lower than x.dim())
  Tensor src = tf.zeros({2, 2});

  Tensor out = tf.ones({1, 2, 2, 1});

  ET_EXPECT_KERNEL_FAILURE(
      context_, op_select_scatter_out(x, src, /*dim=*/0, /*index=*/0, out));
}

TEST_F(OpSelectScatterOutTest, SrcMatchNumelExtraDimAtFrontDies) {
  TensorFactory<ScalarType::Int> tf;
  Tensor x = tf.zeros({2, 2});
  // src shares the same dtype and numel as the selected slice, but the wrong
  // size (src.dim() should always one lower than x.dim())
  Tensor src = tf.zeros({1, 2});

  Tensor out = tf.ones({2, 2});

  ET_EXPECT_KERNEL_FAILURE(
      context_, op_select_scatter_out(x, src, /*dim=*/0, /*index=*/0, out));
}

TEST_F(OpSelectScatterOutTest, SrcSizeMismatchDimDies) {
  TensorFactory<ScalarType::Int> tf;

  Tensor x = tf.zeros({2, 4, 7, 5});
  // Should be {2, 4, 5} to match the selected slice of x when calling select()
  // with dim 2.
  Tensor src = tf.zeros({2, 4, 7});

  Tensor out = tf.zeros({2, 4, 7, 5});

  ET_EXPECT_KERNEL_FAILURE(
      context_, op_select_scatter_out(x, src, /*dim=*/2, /*index=*/3, out));
}

TEST_F(OpSelectScatterOutTest, DynamicShapeUpperBoundSameAsExpected) {
  test_dynamic_shape(
      {2, 3, 2}, torch::executor::TensorShapeDynamism::DYNAMIC_BOUND);
}

TEST_F(OpSelectScatterOutTest, DynamicShapeUpperBoundLargerThanExpected) {
  if (!torch::executor::testing::SupportedFeatures::get()->output_resize) {
    GTEST_SKIP() << "Dynamic shape not supported";
  }
  test_dynamic_shape(
      {10, 10, 10}, torch::executor::TensorShapeDynamism::DYNAMIC_BOUND);
}

TEST_F(OpSelectScatterOutTest, DynamicShapeUnbound) {
  if (!torch::executor::testing::SupportedFeatures::get()->output_resize) {
    GTEST_SKIP() << "Dynamic shape not supported";
  }
  test_dynamic_shape(
      {1, 1, 1}, torch::executor::TensorShapeDynamism::DYNAMIC_UNBOUND);
}
