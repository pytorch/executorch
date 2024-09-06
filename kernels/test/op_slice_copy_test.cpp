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
using exec_aten::optional;
using exec_aten::ScalarType;
using exec_aten::Tensor;
using torch::executor::testing::TensorFactory;

class OpSliceCopyTensorOutTest : public OperatorTest {
 protected:
  Tensor& op_slice_copy_tensor_out(
      const Tensor& self,
      int64_t dim,
      optional<int64_t> start,
      optional<int64_t> end,
      int64_t step,
      Tensor& out) {
    return torch::executor::aten::slice_copy_outf(
        context_, self, dim, start, end, step, out);
  }

  template <class CTYPE, exec_aten::ScalarType DTYPE>
  void test_dtype() {
    TensorFactory<DTYPE> tf;

    // clang-format off
    Tensor input = tf.make(
      /*sizes=*/{3, 4},
      /*data=*/{
        1,   2,   3,   4, // [0, :]
        5,   6,   7,   8, // [1, :]
        9,  10,  11,  12, // [2, :]
      });
  
    // op_slice_copy_tensor_out(input, /*dim=*/0, /*start=*/0, /*end=*/2, /*step=*/1, out),
    // The result should equal to input[0:2:1, :]
    Tensor expect_ret = tf.make(
      /*sizes=*/{2, 4},
      /*data=*/{
        1,   2,   3,   4, // [0, :]
        5,   6,   7,   8, // [1, :]
      });
    // clang-format on

    Tensor out = tf.zeros({2, 4});
    Tensor ret = op_slice_copy_tensor_out(
        input, /*dim=*/0, /*start=*/0, /*end=*/2, /*step=*/1, out);

    EXPECT_TENSOR_EQ(out, ret);
    EXPECT_TENSOR_EQ(ret, expect_ret);
  }
};

TEST_F(OpSliceCopyTensorOutTest, LegalDimSupported) {
  TensorFactory<ScalarType::Double> tf;

  // clang-format off
  Tensor input = tf.make(
    /*sizes=*/{2, 3, 4},
    /*data=*/{
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
  // The size of expected output tensor should follow these rules:
  // - output.size(i) shall equal input.size(i) if i != dim,
  // - output.size(i) shall equal num_values if i == dim
  //   The definition of num_values could be found at https://fburl.com/code/mnnxkowm

  // op_slice_copy_tensor_out(input, /*dim=*/0, /*start=*/0, /*end=*/1, /*step=*/1, out),
  // The result should equal to input[0:1:1，:, :]
  Tensor expected_dim_0 = tf.make(
    /*sizes=*/{1, 3, 4},
    /*data=*/{
      1.,   2.,   3.,   4., // [0, :]
      5.,   6.,   7.,   8., // [1, :]
      9.,  10.,  11.,  12., // [2, :]
    });
  // op_slice_copy_tensor_out(input, /*dim=*/1, /*start=*/0, /*end=*/1, /*step=*/1, out),
  // The result should equal to input[:，0:1:1, :]
  Tensor expected_dim_1 = tf.make(
    /*sizes=*/{2, 1, 4},
    /*data=*/{
        1.,   2.,   3.,   4., // [0, :, :]
      -1.,  -2.,  -3.,  -4., // [1, :, :]
    });
  // op_slice_copy_tensor_out(input, /*dim=*/2, /*start=*/0, /*end=*/1, /*step=*/1, out),
  // The result should equal to input[:，:, 0:1:1]
  Tensor expected_dim_2 = tf.make(
    /*sizes=*/{2, 3, 1},
    /*data=*/{
        1.,   5.,   9., // [0, :, :]
      -1.,  -5.,  -9., // [1, :, :]
    });
  // clang-format on
  std::vector<Tensor> expected_rets = {
      // Groud truth for dim=-3
      expected_dim_0,
      // Groud truth for dim=-2
      expected_dim_1,
      // Groud truth for dim=-1
      expected_dim_2,
      // Groud truth for dim=0
      expected_dim_0,
      // Groud truth for dim=1
      expected_dim_1,
      // Groud truth for dim=2
      expected_dim_2,
  };

  for (int64_t dim = -3; dim < 3; dim++) {
    int64_t testcase_idx = dim + 3;
    auto expected_ret = expected_rets[testcase_idx];
    Tensor out = tf.zeros_like(expected_ret);

    // Slice input on dim with start=0, end = 0 and step = 1
    // Should always return the provided out Tensor.
    // The ret shall meet the expectation.
    Tensor ret = op_slice_copy_tensor_out(
        input, dim, /*start=*/0, /*end=*/1, /*step=*/1, out);
    EXPECT_TENSOR_EQ(out, ret);
    EXPECT_TENSOR_EQ(ret, expected_rets[testcase_idx]);
  }
}

TEST_F(OpSliceCopyTensorOutTest, AllStartValsSupported) {
  TensorFactory<ScalarType::Double> tf;

  // clang-format off
  Tensor input = tf.make(
    /*sizes=*/{2, 3, 4},
    /*data=*/{
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
  // Set the end large enough to hold any start

  // The size of expected output tensor should follow these rules:
  // - output.size(i) shall equal input.size(i) if i != dim,
  // - output.size(i) shall equal num_values if i == dim
  //   The definition of num_values could be found at https://fburl.com/code/mnnxkowm

  // op_slice_copy_tensor_out(input, /*dim=*/1, /*start=*/ <= 0, /*end=*/10, /*step=*/1, out),
  // The result shall equal to input[:，0:3:1, :]
  Tensor expected_start_0_or_below = tf.make(
    /*sizes=*/{2, 3, 4},
    /*data=*/{
      // [0, :, :]
      1.,   2.,   3.,   4., // [0, 0, :]
      5.,   6.,   7.,   8., // [0, 1, :]
      9.,  10.,  11.,  12., // [0, 2, :]

      // [1, :, :]
      -1.,  -2.,  -3.,  -4., // [1, 0, :]
      -5.,  -6.,  -7.,  -8., // [1, 1, :]
      -9., -10., -11., -12., // [1, 2, :]
    });
  // op_slice_copy_tensor_out(input, /*dim=*/1, /*start=*/1, /*end=*/10, /*step=*/1, out),
  // The result shall equal to input[:，1:3:1, :]
  Tensor expected_start_1 = tf.make(
    /*sizes=*/{2, 2, 4},
    /*data=*/{
      // [0, :, :]
      5.,   6.,   7.,   8., // [0, 0, :]
      9.,  10.,  11.,  12., // [0, 1, :]

      // [1, :, :]
      -5.,  -6.,  -7.,  -8., // [1, 0, :]
      -9., -10., -11., -12., // [1, 1, :]
    });
  // op_slice_copy_tensor_out(input, /*dim=*/1, /*start=*/2, /*end=*/10, /*step=*/1, out),
  // The result shall equal to input[:，2:3:1, :] = input
  Tensor expected_start_2 = tf.make(
    /*sizes=*/{2, 1, 4},
    /*data=*/{
       9.,  10.,  11.,  12., // [0, 0, :]
      -9., -10., -11., -12., // [1, 0, :]
    });

  // op_slice_copy_tensor_out(input, /*dim=*/1, /*start=*/ > input.size(1) = 2, /*end=*/10, /*step=*/1, out),
  // The result shall equal to input[:, 3:3:1, :], which is an empty tensor
  Tensor expected_start_3_or_above = tf.make({2, 0, 4}, {});
  // clang-format on
  std::vector<Tensor> expected_rets = {// start = -3
                                       expected_start_0_or_below,
                                       // start = -2
                                       expected_start_1,
                                       // start = -1
                                       expected_start_2,
                                       // start = 0
                                       expected_start_0_or_below,
                                       // start = 1
                                       expected_start_1,
                                       // start = 2
                                       expected_start_2,
                                       // start = 3
                                       expected_start_3_or_above};

  // In this test, we maintain dim and step as 1 and 1, also set the end
  // large enough to hold any start
  int64_t dim = 1;
  int64_t end = 10;
  int64_t step = 1;
  for (int64_t start = -3; start < 4; start++) {
    int64_t testcase_idx = start + 3;
    auto expected_ret = expected_rets[testcase_idx];
    Tensor out = tf.zeros_like(expected_ret);

    // Should always return the provided out Tensor.
    // The ret shall meet the expectation.
    Tensor ret = op_slice_copy_tensor_out(input, dim, start, end, step, out);
    EXPECT_TENSOR_EQ(out, ret);
    EXPECT_TENSOR_EQ(ret, expected_ret);
  }
}

TEST_F(OpSliceCopyTensorOutTest, AllEndValsSupported) {
  TensorFactory<ScalarType::Double> tf;

  // clang-format off
  Tensor input = tf.make(
    /*sizes=*/{2, 3, 4},
    /*data=*/{
      // [0, :, :]
      1.,   2.,   3.,   4., // [0, 0, :]
      5.,   6.,   7.,   8., // [0, 1, :]
      9.,  10.,  11.,  12., // [0, 2, :]

      // [1, :, :]
      -1.,  -2.,  -3.,  -4., // [1, 0, :]
      -5.,  -6.,  -7.,  -8., // [1, 1, :]
      -9., -10., -11., -12., // [1, 2, :]
    });

  // The size of expected output tensor should follow these rules:
  // - output.size(i) shall equal input.size(i) if i != dim,
  // - output.size(i) shall equal num_values if i == dim
  //   The definition of num_values could be found at https://fburl.com/code/mnnxkowm

  // op_slice_copy_tensor_out(input, /*dim=*/1, /*start=*/0, /*end=*/ <= 0, /*step=*/1, out),
  // The result should equal input[:，0:0:1, :], which should be an empty tensor
  Tensor expected_end_0_or_below = tf.make({2, 0, 4}, {});

  // op_slice_copy_tensor_out(input, /*dim=*/1, /*start=*/0, /*end=*/1, /*step=*/1, out),
  // The result should equal to input[:，0:1:1, :]
  Tensor expected_end_1 = tf.make(
    /*sizes=*/{2, 1, 4},
    /*data=*/{
      1.,   2.,   3.,   4., // [0, :, :]
      -1.,  -2.,  -3.,  -4., // [1, :, :]
    });

  // op_slice_copy_tensor_out(input, /*dim=*/1, /*start=*/0, /*end=*/2, /*step=*/1, out),
  // The result should equal input[:，0:2:1, :]
  Tensor expected_end_2 = tf.make(
    /*sizes=*/{2, 2, 4},
    /*data=*/{
      // [0, :, :]
      1.,   2.,   3.,   4., // [0, 0, :]
      5.,   6.,   7.,   8., // [0, 1, :]

      // [1, :, :]
      -1.,  -2.,  -3.,  -4., // [1, 0, :]
      -5.,  -6.,  -7.,  -8., // [1, 1, :]
    });
  // op_slice_copy_tensor_out(input, /*dim=*/1, /*start=*/0, /*end=*/ >= 3, /*step=*/1, out),
  // The result should equal input[:，0:3:1, :] = input for any end >= 3
  Tensor expected_end_3_or_above = tf.make(
    /*sizes=*/{2, 3, 4},
    /*data=*/{
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
  std::vector<Tensor> expected_rets = {// end = -3
                                       expected_end_0_or_below,
                                       // end = -2
                                       expected_end_1,
                                       // end = -1
                                       expected_end_2,
                                       // end = 0
                                       expected_end_0_or_below,
                                       // end = 1
                                       expected_end_1,
                                       // end = 2
                                       expected_end_2,
                                       // end = 3
                                       expected_end_3_or_above};

  int64_t dim = 1;
  int64_t start = 0;
  int64_t step = 1;
  for (int64_t end = -3; end < 4; end++) {
    int64_t testcase_idx = end + 3;

    auto expected_ret = expected_rets[testcase_idx];
    Tensor out = tf.zeros_like(expected_ret);

    // Should always return the provided out Tensor.
    // The ret shall meet the expectation.
    Tensor ret = op_slice_copy_tensor_out(input, dim, start, end, step, out);
    EXPECT_TENSOR_EQ(out, ret);
    EXPECT_TENSOR_EQ(ret, expected_ret);
  }
}

TEST_F(OpSliceCopyTensorOutTest, LegalStepsSupported) {
  TensorFactory<ScalarType::Double> tf;

  // clang-format off
  Tensor input = tf.make(
    /*sizes=*/{2, 3, 4},
    /*data=*/{
      // [0, :, :]
      1.,   2.,   3.,   4., // [0, 0, :]
      5.,   6.,   7.,   8., // [0, 1, :]
      9.,  10.,  11.,  12., // [0, 2, :]

      // [1, :, :]
      -1.,  -2.,  -3.,  -4., // [1, 0, :]
      -5.,  -6.,  -7.,  -8., // [1, 1, :]
      -9., -10., -11., -12., // [1, 2, :]
    });

  // Set the end large enough to hold any step

  // Expected ret for op_slice_copy_tensor_out(input, /*dim=*/1, /*start=*/0, /*end=*/10, /*step=*/1, out),
  // The result should equal to input[:，0:3:1, :]
  Tensor expected_0 = tf.make(
    /*sizes=*/{2, 3, 4},
    /*data=*/{
      // [0, :, :]
      1.,   2.,   3.,   4., // [0, 0, :]
      5.,   6.,   7.,   8., // [0, 1, :]
      9.,  10.,  11.,  12., // [0, 2, :]

      // [1, :, :]
      -1.,  -2.,  -3.,  -4., // [1, 0, :]
      -5.,  -6.,  -7.,  -8., // [1, 1, :]
      -9., -10., -11., -12., // [1, 2, :]
    });
  // Expected ret for op_slice_copy_tensor_out(input, /*dim=*/1, /*start=*/0, /*end=*/10, /*step=*/2, out),
  // The result should equal to input[:，0:3:2, :]
  Tensor expected_1 = tf.make(
    /*sizes=*/{2, 2, 4},
    /*data=*/{
      // [0, :, :]
      1.,   2.,   3.,   4., // [0, 0, :]
      9.,  10.,  11.,  12., // [0, 1, :]

      // [1, :, :]
      -1.,  -2.,  -3.,  -4., // [1, 0, :]
      -9., -10., -11., -12., // [1, 1, :]
    });
  // Expected ret for op_slice_copy_tensor_out(input, /*dim=*/1, /*start=*/0, /*end=*/10, /*step=*/3, out),
  // The result should equal to input[:，0:3:3, :] = input
  Tensor expected_2 = tf.make(
    /*sizes=*/{2, 1, 4},
    /*data=*/{
      1.,   2.,   3.,   4., // [0, 0, :]
      -1.,  -2.,  -3.,  -4., // [1, 0, :]
    });
  // clang-format on
  std::vector<Tensor> expected_rets = {expected_0, expected_1, expected_2};

  // In this test, we maintain start and dim as 0 and 1, also set the
  // end large enough to hold any step
  int64_t start = 0;
  int64_t dim = 1;
  int64_t end = 10;
  for (int64_t step = 1; step < 4; step++) {
    int64_t testcase_idx = step - 1;

    auto expected_ret = expected_rets[testcase_idx];
    Tensor out = tf.zeros_like(expected_ret);

    // Should always return the provided out Tensor.
    // The ret shall meet the expectation.
    Tensor ret = op_slice_copy_tensor_out(input, dim, start, end, step, out);
    EXPECT_TENSOR_EQ(out, ret);
    EXPECT_TENSOR_EQ(ret, expected_ret);
  }
}

/// A generic smoke test that works for any dtype that supports ones() and
/// zeros().
TEST_F(OpSliceCopyTensorOutTest, AllDtypesSupported) {
  if (torch::executor::testing::SupportedFeatures::get()->is_aten) {
    GTEST_SKIP() << "ATen kernel test fails";
  }
#define TEST_ENTRY(ctype, dtype) test_dtype<ctype, ScalarType::dtype>();
  ET_FORALL_REAL_TYPES_AND(Bool, TEST_ENTRY);
#undef TEST_ENTRY
  // TODO: Also add tests for half, complex, quantized, and other types. Easiest
  // way to do that would be to make TensorFactory support zeros() and ones()
  // for those types.
}

TEST_F(OpSliceCopyTensorOutTest, EmptyInputSupported) {
  TensorFactory<ScalarType::Int> tf;

  Tensor input = tf.ones({1, 0, 1});
  Tensor out = tf.zeros({1, 0, 1});

  Tensor expect = tf.ones({1, 0, 1});

  // Some invalid dim values.
  for (int64_t dim = 0; dim > input.dim(); dim++) {
    Tensor ret = op_slice_copy_tensor_out(
        input, dim, /*start=*/0, /*end=*/1, /*step=*/1, out);
    EXPECT_TENSOR_EQ(ret, out);

    // All operations in this test share same ground truth
    EXPECT_TENSOR_EQ(ret, expect);
  }
}

TEST_F(OpSliceCopyTensorOutTest, EmptySizeInputDies) {
  TensorFactory<ScalarType::Int> tf;

  Tensor input = tf.ones({});
  Tensor out = tf.ones({});

  // The operation shall die whatever the end is.
  ET_EXPECT_KERNEL_FAILURE(
      context_,
      op_slice_copy_tensor_out(
          input, /*dim=*/0, /*start=*/0, /*end=*/0, /*step=*/1, out));
  ET_EXPECT_KERNEL_FAILURE(
      context_,
      op_slice_copy_tensor_out(
          input, /*dim=*/0, /*start=*/0, /*end=*/1, /*step=*/1, out));
}

TEST_F(OpSliceCopyTensorOutTest, ZeroLengthSupported) {
  TensorFactory<ScalarType::Int> tf;

  Tensor input = tf.ones({2, 3});
  Tensor out = tf.ones({2, 0});

  Tensor expect = tf.ones({2, 0});

  Tensor ret = op_slice_copy_tensor_out(
      input, /*dim=*/1, /*start=*/1, /*end=*/1, /*step=*/1, out);
  EXPECT_TENSOR_EQ(ret, out);
  EXPECT_TENSOR_EQ(ret, expect);

  ret = op_slice_copy_tensor_out(
      input, /*dim=*/1, /*start=*/-1, /*end=*/-1, /*step=*/1, out);
  EXPECT_TENSOR_EQ(ret, out);
  EXPECT_TENSOR_EQ(ret, expect);
}

TEST_F(OpSliceCopyTensorOutTest, NonPostiveStepsDies) {
  TensorFactory<ScalarType::Int> tf;

  Tensor input = tf.ones({1, 1, 1});
  Tensor out = tf.zeros({1, 1, 1});

  // Some invalid step values.
  const std::vector<int64_t> invalid_steps = {-2, -1, 0};
  for (int64_t step : invalid_steps) {
    ET_EXPECT_KERNEL_FAILURE(
        context_,
        op_slice_copy_tensor_out(
            input, /*dim=*/0, /*start=*/0, /*end=*/1, /*step=*/step, out));
  }
}

TEST_F(OpSliceCopyTensorOutTest, DimOutOfBoundDies) {
  TensorFactory<ScalarType::Int> tf;

  Tensor input = tf.ones({1, 1, 1});
  Tensor out = tf.zeros({1, 1, 1});

  // Some invalid dim values.
  const std::vector<int64_t> invalid_dims = {3, 4, 5, -4, -5, -6};
  for (int64_t dim : invalid_dims) {
    ET_EXPECT_KERNEL_FAILURE(
        context_,
        op_slice_copy_tensor_out(
            input, dim, /*start=*/0, /*end=*/1, /*step=*/1, out));
  }
}

TEST_F(OpSliceCopyTensorOutTest, MismatchedDtypesDies) {
  TensorFactory<ScalarType::Int> tf_int;
  TensorFactory<ScalarType::Float> tf_float;
  Tensor input = tf_int.zeros({1, 2, 2});

  // Size is compatible to the output, but a mismatched dtype.
  Tensor out = tf_float.ones({1, 2, 2});

  ET_EXPECT_KERNEL_FAILURE(
      context_,
      op_slice_copy_tensor_out(
          input, /*dim=*/0, /*start=*/0, /*end=*/1, /*step=*/1, out));
}

TEST_F(OpSliceCopyTensorOutTest, OutSizeMismatchDimDies) {
  if (torch::executor::testing::SupportedFeatures::get()->is_aten) {
    GTEST_SKIP() << "ATen kernel can handle out with mismatched dimensions";
  }
  TensorFactory<ScalarType::Int> tf;

  Tensor input = tf.zeros({2, 4, 7, 5});

  // Should be {2, 4, 7, 5}
  Tensor out = tf.zeros({2, 4, 7});

  ET_EXPECT_KERNEL_FAILURE(
      context_,
      op_slice_copy_tensor_out(
          input, /*dim=*/0, /*start=*/0, /*end=*/2, /*step=*/1, out));
}

TEST_F(OpSliceCopyTensorOutTest, DefaultStartValSupported) {
  TensorFactory<ScalarType::Int> tf;

  Tensor input = tf.zeros({2, 4, 7, 5});

  Tensor out = tf.ones({2, 4, 7, 5});
  Tensor expected = tf.zeros({2, 4, 7, 5});

  Tensor ret_default_start = op_slice_copy_tensor_out(
      input,
      /*dim=*/0,
      /*start=*/exec_aten::nullopt,
      /*end=*/2,
      /*step=*/1,
      out);
  EXPECT_TENSOR_EQ(ret_default_start, out);
  EXPECT_TENSOR_EQ(ret_default_start, expected);
}

TEST_F(OpSliceCopyTensorOutTest, DefaultEndValSupported) {
  TensorFactory<ScalarType::Int> tf;

  Tensor input = tf.zeros({2, 4, 7, 5});

  Tensor out = tf.ones({2, 4, 7, 5});
  Tensor expected = tf.zeros({2, 4, 7, 5});

  Tensor ret_default_end = op_slice_copy_tensor_out(
      input,
      /*dim=*/0,
      /*start=*/0,
      /*end=*/exec_aten::nullopt,
      /*step=*/1,
      out);
  EXPECT_TENSOR_EQ(ret_default_end, out);
  EXPECT_TENSOR_EQ(ret_default_end, expected);
}

/* %python
import torch
torch.manual_seed(0)
x = torch.rand(2, 6, 3)
res = x[:, 1:5:2, :]
print(res.size())
op = "op_slice_copy_tensor_out"
opt_extra_params = "1, 1, 5, 2,"
dtype = "ScalarType::Float"
check = "EXPECT_TENSOR_EQ" */

TEST_F(OpSliceCopyTensorOutTest, DynamicShapeUpperBoundSameAsExpected) {
  /* %python
  out_args = "{2, 2, 3}, torch::executor::TensorShapeDynamism::DYNAMIC_BOUND"
  %rewrite(unary_op) */

  TensorFactory<ScalarType::Float> tf;

  Tensor x = tf.make(
      {2, 6, 3},
      {0.49625658988952637,  0.7682217955589294,  0.08847743272781372,
       0.13203048706054688,  0.30742281675338745, 0.6340786814689636,
       0.4900934100151062,   0.8964447379112244,  0.455627977848053,
       0.6323062777519226,   0.3488934636116028,  0.40171730518341064,
       0.022325754165649414, 0.16885894536972046, 0.2938884496688843,
       0.518521785736084,    0.6976675987243652,  0.800011396408081,
       0.16102945804595947,  0.28226858377456665, 0.6816085577011108,
       0.9151939749717712,   0.39709991216659546, 0.8741558790206909,
       0.41940832138061523,  0.5529070496559143,  0.9527381062507629,
       0.036164820194244385, 0.1852310299873352,  0.37341737747192383,
       0.3051000237464905,   0.9320003986358643,  0.17591017484664917,
       0.2698335647583008,   0.15067976713180542, 0.03171950578689575});
  Tensor expected = tf.make(
      {2, 2, 3},
      {0.13203048706054688,
       0.30742281675338745,
       0.6340786814689636,
       0.6323062777519226,
       0.3488934636116028,
       0.40171730518341064,
       0.9151939749717712,
       0.39709991216659546,
       0.8741558790206909,
       0.036164820194244385,
       0.1852310299873352,
       0.37341737747192383});

  Tensor out =
      tf.zeros({2, 2, 3}, torch::executor::TensorShapeDynamism::DYNAMIC_BOUND);
  op_slice_copy_tensor_out(x, 1, 1, 5, 2, out);
  EXPECT_TENSOR_EQ(out, expected);
}

TEST_F(OpSliceCopyTensorOutTest, DynamicShapeUpperBoundLargerThanExpected) {
  if (!torch::executor::testing::SupportedFeatures::get()->output_resize) {
    GTEST_SKIP() << "Dynamic shape not supported";
  }
  /* %python
  out_args = "{10, 10, 10}, torch::executor::TensorShapeDynamism::DYNAMIC_BOUND"
  %rewrite(unary_op) */

  TensorFactory<ScalarType::Float> tf;

  Tensor x = tf.make(
      {2, 6, 3},
      {0.49625658988952637,  0.7682217955589294,  0.08847743272781372,
       0.13203048706054688,  0.30742281675338745, 0.6340786814689636,
       0.4900934100151062,   0.8964447379112244,  0.455627977848053,
       0.6323062777519226,   0.3488934636116028,  0.40171730518341064,
       0.022325754165649414, 0.16885894536972046, 0.2938884496688843,
       0.518521785736084,    0.6976675987243652,  0.800011396408081,
       0.16102945804595947,  0.28226858377456665, 0.6816085577011108,
       0.9151939749717712,   0.39709991216659546, 0.8741558790206909,
       0.41940832138061523,  0.5529070496559143,  0.9527381062507629,
       0.036164820194244385, 0.1852310299873352,  0.37341737747192383,
       0.3051000237464905,   0.9320003986358643,  0.17591017484664917,
       0.2698335647583008,   0.15067976713180542, 0.03171950578689575});
  Tensor expected = tf.make(
      {2, 2, 3},
      {0.13203048706054688,
       0.30742281675338745,
       0.6340786814689636,
       0.6323062777519226,
       0.3488934636116028,
       0.40171730518341064,
       0.9151939749717712,
       0.39709991216659546,
       0.8741558790206909,
       0.036164820194244385,
       0.1852310299873352,
       0.37341737747192383});

  Tensor out = tf.zeros(
      {10, 10, 10}, torch::executor::TensorShapeDynamism::DYNAMIC_BOUND);
  op_slice_copy_tensor_out(x, 1, 1, 5, 2, out);
  EXPECT_TENSOR_EQ(out, expected);
}

TEST_F(OpSliceCopyTensorOutTest, DynamicShapeUnbound) {
  if (!torch::executor::testing::SupportedFeatures::get()->output_resize) {
    GTEST_SKIP() << "Dynamic shape not supported";
  }
  /* %python
  out_args = "{1, 1, 1}, torch::executor::TensorShapeDynamism::DYNAMIC_UNBOUND"
  %rewrite(unary_op) */

  TensorFactory<ScalarType::Float> tf;

  Tensor x = tf.make(
      {2, 6, 3},
      {0.49625658988952637,  0.7682217955589294,  0.08847743272781372,
       0.13203048706054688,  0.30742281675338745, 0.6340786814689636,
       0.4900934100151062,   0.8964447379112244,  0.455627977848053,
       0.6323062777519226,   0.3488934636116028,  0.40171730518341064,
       0.022325754165649414, 0.16885894536972046, 0.2938884496688843,
       0.518521785736084,    0.6976675987243652,  0.800011396408081,
       0.16102945804595947,  0.28226858377456665, 0.6816085577011108,
       0.9151939749717712,   0.39709991216659546, 0.8741558790206909,
       0.41940832138061523,  0.5529070496559143,  0.9527381062507629,
       0.036164820194244385, 0.1852310299873352,  0.37341737747192383,
       0.3051000237464905,   0.9320003986358643,  0.17591017484664917,
       0.2698335647583008,   0.15067976713180542, 0.03171950578689575});
  Tensor expected = tf.make(
      {2, 2, 3},
      {0.13203048706054688,
       0.30742281675338745,
       0.6340786814689636,
       0.6323062777519226,
       0.3488934636116028,
       0.40171730518341064,
       0.9151939749717712,
       0.39709991216659546,
       0.8741558790206909,
       0.036164820194244385,
       0.1852310299873352,
       0.37341737747192383});

  Tensor out = tf.zeros(
      {1, 1, 1}, torch::executor::TensorShapeDynamism::DYNAMIC_UNBOUND);
  op_slice_copy_tensor_out(x, 1, 1, 5, 2, out);
  EXPECT_TENSOR_EQ(out, expected);
}
