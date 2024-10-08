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

class OpSliceScatterTensorOutTest : public OperatorTest {
 protected:
  Tensor& op_slice_scatter_out(
      const Tensor& self,
      const Tensor& src,
      int64_t dim,
      optional<int64_t> start,
      optional<int64_t> end,
      int64_t step,
      Tensor& out) {
    return torch::executor::aten::slice_scatter_outf(
        context_, self, src, dim, start, end, step, out);
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

    // op_slice_scatter_out(input, src, /*dim=*/0, /*start=*/0, /*end=*/2, /*step=*/1, out),
    // src shape should equal to input[0:2:1, :]
    Tensor src = tf.make(
      /*sizes=*/{2, 4},
      /*data=*/{
        5,   6,   7,   8, // [0, :]
        1,   2,   3,   4, // [1, :]
      });
    Tensor expect_ret = tf.make(
      /*sizes=*/{3, 4},
      /*data=*/{
        5,   6,   7,   8, // [0, :]
        1,   2,   3,   4, // [1, :]
        9,  10,  11,  12, // [2, :]
      });
    // clang-format on

    Tensor out = tf.zeros({3, 4});
    Tensor ret = op_slice_scatter_out(
        input, src, /*dim=*/0, /*start=*/0, /*end=*/2, /*step=*/1, out);

    EXPECT_TENSOR_EQ(out, ret);
    EXPECT_TENSOR_EQ(ret, expect_ret);
  }
};

TEST_F(OpSliceScatterTensorOutTest, LegalDimSupported) {
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
  // The size of the src tensor should follow these rules:
  // - src.size(i) shall equal input.size(i) if i != dim,
  // - src.size(i) shall equal num_values if i == dim
  //   The definition of num_values could be found at https://fburl.com/code/mnnxkowm

  // op_slice_scatter_out(input, src, /*dim=*/0, /*start=*/0, /*end=*/1, /*step=*/1, out),
  // src shape should equal to input[0:1:1，:, :]
  Tensor src_dim_0 = tf.make(
    /*sizes=*/{1, 3, 4},
    /*data=*/{
      8.,   7.,   6.,   5., // [1, :]
      4.,   3.,   2.,   1., // [0, :]
      1.,  14.,  18.,  19., // [2, :]
    });
  Tensor expected_dim_0 = tf.make(
    /*sizes=*/{2, 3, 4},
    /*data=*/{
      // [0, :, :]
       8.,   7.,   6.,   5., // [0, 1, :]
       4.,   3.,   2.,   1., // [0, 0, :]
       1.,  14.,  18.,  19., // [0, 2, :]

      // [1, :, :]
      -1.,  -2.,  -3.,  -4., // [1, 0, :]
      -5.,  -6.,  -7.,  -8., // [1, 1, :]
      -9., -10., -11., -12., // [1, 2, :]
    });
  // op_slice_scatter_out(input, src, /*dim=*/1, /*start=*/0, /*end=*/1, /*step=*/1, out),
  // src shape should equal to input[:，0:1:1, :]
  Tensor src_dim_1 = tf.make(
    /*sizes=*/{2, 1, 4},
    /*data=*/{
       4.,   3.,   2.,   1., // [0, :, :]
      -4.,  -3.,  -2.,  -1., // [1, :, :]
    });
  Tensor expected_dim_1 = tf.make(
    /*sizes=*/{2, 3, 4},
    /*data=*/{
      // [0, :, :]
       4.,   3.,   2.,   1., // [0, 0, :]
       5.,   6.,   7.,   8., // [0, 1, :]
       9.,  10.,  11.,  12., // [0, 2, :]

      // [1, :, :]
      -4.,  -3.,  -2.,  -1., // [1, 0, :]
      -5.,  -6.,  -7.,  -8., // [1, 1, :]
      -9., -10., -11., -12., // [1, 2, :]
    });
  // op_slice_scatter_out(input, src, /*dim=*/2, /*start=*/0, /*end=*/1, /*step=*/1, out),
  // src shape should equal to input[:，:, 0:1:1]
  Tensor src_dim_2 = tf.make(
    /*sizes=*/{2, 3, 1},
    /*data=*/{
       7.,   1.,   6., // [0, :, :]
      -5.,  -9.,  -2., // [1, :, :]
    });
  Tensor expected_dim_2 = tf.make(
    /*sizes=*/{2, 3, 4},
    /*data=*/{
      // [0, :, :]
       7.,   2.,   3.,   4., // [0, 0, :]
       1.,   6.,   7.,   8., // [0, 1, :]
       6.,  10.,  11.,  12., // [0, 2, :]

      // [1, :, :]
      -5.,  -2.,  -3.,  -4., // [1, 0, :]
      -9.,  -6.,  -7.,  -8., // [1, 1, :]
      -2., -10., -11., -12., // [1, 2, :]
    });
  // clang-format on
  std::vector<Tensor> src_tensors = {
      // Source tensor for dim=-3
      src_dim_0,
      // Source tensor for dim=-2
      src_dim_1,
      // Source tensor for dim=-1
      src_dim_2,
      // Source tensor for dim=0
      src_dim_0,
      // Source tensor for dim=1
      src_dim_1,
      // Source tensor for dim=2
      src_dim_2,
  };
  std::vector<Tensor> expected_rets = {
      // Ground truth for dim=-3
      expected_dim_0,
      // Ground truth for dim=-2
      expected_dim_1,
      // Ground truth for dim=-1
      expected_dim_2,
      // Ground truth for dim=0
      expected_dim_0,
      // Ground truth for dim=1
      expected_dim_1,
      // Ground truth for dim=2
      expected_dim_2,
  };

  for (int64_t dim = -3; dim < 3; dim++) {
    int64_t testcase_idx = dim + 3;
    auto src = src_tensors[testcase_idx];
    auto expected_ret = expected_rets[testcase_idx];

    Tensor out = tf.zeros_like(expected_ret);

    // Slice input on dim with start=0, end = 0 and step = 1
    // Should always return the provided out Tensor.
    // The ret shall meet the expectation.
    Tensor ret = op_slice_scatter_out(
        input, src, dim, /*start=*/0, /*end=*/1, /*step=*/1, out);
    EXPECT_TENSOR_EQ(out, ret);
    EXPECT_TENSOR_EQ(ret, expected_rets[testcase_idx]);
  }
}

TEST_F(OpSliceScatterTensorOutTest, AllStartValsSupported) {
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

  // The size of the src tensor should follow these rules:
  // - src.size(i) shall equal input.size(i) if i != dim,
  // - src.size(i) shall equal num_values if i == dim
  //   The definition of num_values could be found at https://fburl.com/code/mnnxkowm

  // op_slice_scatter_out(input, src, /*dim=*/1, /*start=*/ <= 0, /*end=*/10, /*step=*/1, out),
  // src shape shall equal to input[:，0:3:1, :]
  Tensor src_start_0_or_below = tf.make(
    /*sizes=*/{2, 3, 4},
    /*data=*/{
      // [0, :, :]
      -1.,  -2.,  -3.,  -4., // [0, 0, :]
      -5.,  -6.,  -7.,  -8., // [0, 1, :]
      -9., -10., -11., -12., // [0, 2, :]

      // [1, :, :]
       1.,   2.,   3.,   4., // [1, 0, :]
       5.,   6.,   7.,   8., // [1, 1, :]
       9.,  10.,  11.,  12., // [1, 2, :]
    });
  Tensor expected_start_0_or_below = tf.make(
    /*sizes=*/{2, 3, 4},
    /*data=*/{
      // [0, :, :]
      -1.,  -2.,  -3.,  -4., // [0, 0, :]
      -5.,  -6.,  -7.,  -8., // [0, 1, :]
      -9., -10., -11., -12., // [0, 2, :]

      // [1, :, :]
       1.,   2.,   3.,   4., // [1, 0, :]
       5.,   6.,   7.,   8., // [1, 1, :]
       9.,  10.,  11.,  12., // [1, 2, :]
    });
  // op_slice_scatter_out(input, src, /*dim=*/1, /*start=*/1, /*end=*/10, /*step=*/1, out),
  // src shape shall equal to input[:，1:3:1, :]
  Tensor src_start_1 = tf.make(
    /*sizes=*/{2, 2, 4},
    /*data=*/{
      // [0, :, :]
      -9., -10., -11., -12., // [0, 1, :]
      -5.,  -6.,  -7.,  -8., // [0, 0, :]

      // [1, :, :]
       9.,  10.,  11.,  12., // [1, 1, :]
       5.,   6.,   7.,   8., // [1, 0, :]
    });
  Tensor expected_start_1 = tf.make(
    /*sizes=*/{2, 3, 4},
    /*data=*/{
      // [0, :, :]
       1.,   2.,   3.,   4., // [0, 0, :]
      -9., -10., -11., -12., // [0, 1, :]
      -5.,  -6.,  -7.,  -8., // [0, 2, :]

      // [1, :, :]
      -1.,  -2.,  -3.,  -4., // [1, 0, :]
       9.,  10.,  11.,  12., // [1, 1, :]
       5.,   6.,   7.,   8., // [1, 0, :]
    });
  // op_slice_scatter_out(input, src, /*dim=*/1, /*start=*/2, /*end=*/10, /*step=*/1, out),
  // src shape shall equal to input[:，2:3:1, :] = input
  Tensor src_start_2 = tf.make(
    /*sizes=*/{2, 1, 4},
    /*data=*/{
       1.,  19.,  18.,  17., // [0, 0, :]
      -1., -19., -18., -17., // [1, 0, :]
    });
  Tensor expected_start_2 = tf.make(
    /*sizes=*/{2, 3, 4},
    /*data=*/{
      // [0, :, :]
       1.,   2.,   3.,   4., // [0, 0, :]
       5.,   6.,   7.,   8., // [0, 1, :]
       1.,  19.,  18.,  17., // [0, 2, :]

      // [1, :, :]
      -1.,  -2.,  -3.,  -4., // [1, 0, :]
      -5.,  -6.,  -7.,  -8., // [1, 1, :]
      -1., -19., -18., -17., // [1, 2, :]
    });
  // op_slice_scatter_out(input, src, /*dim=*/1, /*start=*/ > input.size(1) = 2, /*end=*/10, /*step=*/1, out),
  // src_shape shall equal to input[:, 3:3:1, :], which is an empty tensor
  Tensor src_start_3_or_above = tf.make({2, 0, 4}, {});
  Tensor expected_start_3_or_above = tf.make(
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

  std::vector<Tensor> src_tensors = {// start = -3
                                     src_start_0_or_below,
                                     // start = -2
                                     src_start_1,
                                     // start = -1
                                     src_start_2,
                                     // start = 0
                                     src_start_0_or_below,
                                     // start = 1
                                     src_start_1,
                                     // start = 2
                                     src_start_2,
                                     // start = 3
                                     src_start_3_or_above};
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
    auto src = src_tensors[testcase_idx];
    auto expected_ret = expected_rets[testcase_idx];
    Tensor out = tf.zeros_like(expected_ret);

    // Should always return the provided out Tensor.
    // The ret shall meet the expectation.
    Tensor ret = op_slice_scatter_out(input, src, dim, start, end, step, out);
    EXPECT_TENSOR_EQ(out, ret);
    EXPECT_TENSOR_EQ(ret, expected_ret);
  }
}

TEST_F(OpSliceScatterTensorOutTest, AllEndValsSupported) {
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

  // op_slice_scatter_out(input, src, /*dim=*/1, /*start=*/0, /*end=*/ <= 0, /*step=*/1, out),
  // src shape should equal input[:，0:0:1, :], which should be an empty tensor
  Tensor src_end_0_or_below = tf.make({2, 0, 4}, {});
  Tensor expected_end_0_or_below = tf.make(
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

  // op_slice_scatter_out(input, src, /*dim=*/1, /*start=*/0, /*end=*/1, /*step=*/1, out),
  // src shape should equal to input[:，0:1:1, :]
  Tensor src_end_1 = tf.make(
    /*sizes=*/{2, 1, 4},
    /*data=*/{
      -4.,  -3.,  -2.,  -1., // [0, :, :]
       4.,   3.,   2.,   1., // [1, :, :]
    });
  Tensor expected_end_1 = tf.make(
    /*sizes=*/{2, 3, 4},
    /*data=*/{
      // [0, :, :]
      -4.,  -3.,  -2.,  -1., // [0, 0, :]
       5.,   6.,   7.,   8., // [0, 1, :]
       9.,  10.,  11.,  12., // [0, 2, :]

      // [1, :, :]
       4.,   3.,   2.,   1., // [1, 0, :]
      -5.,  -6.,  -7.,  -8., // [1, 1, :]
      -9., -10., -11., -12., // [1, 2, :]
    });

  // op_slice_scatter_out(input, src, /*dim=*/1, /*start=*/0, /*end=*/2, /*step=*/1, out),
  // src shape should equal input[:，0:2:1, :]
  Tensor src_end_2 = tf.make(
    /*sizes=*/{2, 2, 4},
    /*data=*/{
      // [0, :, :]
      -8.,  -7.,  -6.,  -5., // [0, 0, :]
      -4.,  -3.,  -2.,  -1., // [0, :, :]

      // [1, :, :]
       8.,   7.,   6.,   5., // [1, 0, :]
       4.,   3.,   2.,   1., // [1, 1, :]
    });
  Tensor expected_end_2 = tf.make(
    /*sizes=*/{2, 3, 4},
    /*data=*/{
      // [0, :, :]
      -8.,  -7.,  -6.,  -5., // [0, 0, :]
      -4.,  -3.,  -2.,  -1., // [0, 1, :]
       9.,  10.,  11.,  12., // [0, 2, :]

      // [1, :, :]
       8.,   7.,   6.,   5., // [1, 0, :]
       4.,   3.,   2.,   1., // [1, 1, :]
      -9., -10., -11., -12., // [1, 2, :]
    });
  // op_slice_scatter_out(input, src, /*dim=*/1, /*start=*/0, /*end=*/ >= 3, /*step=*/1, out),
  // src shape should equal input[:，0:3:1, :] = input for any end >= 3
  Tensor src_end_3_or_above = tf.make(
    /*sizes=*/{2, 3, 4},
    /*data=*/{
      // [0, :, :]
      -1.,  -2.,  -3.,  -4., // [0, 0, :]
      -5.,  -6.,  -7.,  -8., // [0, 1, :]
      -9., -10., -11., -12., // [0, 2, :]

      // [1, :, :]
       1.,   2.,   3.,   4., // [1, 0, :]
       5.,   6.,   7.,   8., // [1, 1, :]
       9.,  10.,  11.,  12., // [1, 2, :]
    });
  Tensor expected_end_3_or_above = tf.make(
    /*sizes=*/{2, 3, 4},
    /*data=*/{
      // [0, :, :]
      -1.,  -2.,  -3.,  -4., // [0, 0, :]
      -5.,  -6.,  -7.,  -8., // [0, 1, :]
      -9., -10., -11., -12., // [0, 2, :]

      // [1, :, :]
       1.,   2.,   3.,   4., // [1, 0, :]
       5.,   6.,   7.,   8., // [1, 1, :]
       9.,  10.,  11.,  12., // [1, 2, :]
    });
  // clang-format on

  std::vector<Tensor> src_tensors = {// end = -3
                                     src_end_0_or_below,
                                     // end = -2
                                     src_end_1,
                                     // end = -1
                                     src_end_2,
                                     // end = 0
                                     src_end_0_or_below,
                                     // end = 1
                                     src_end_1,
                                     // end = 2
                                     src_end_2,
                                     // end = 3
                                     src_end_3_or_above};

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

    auto src = src_tensors[testcase_idx];
    auto expected_ret = expected_rets[testcase_idx];
    Tensor out = tf.zeros_like(expected_ret);

    // Should always return the provided out Tensor.
    // The ret shall meet the expectation.
    Tensor ret = op_slice_scatter_out(input, src, dim, start, end, step, out);
    EXPECT_TENSOR_EQ(out, ret);
    EXPECT_TENSOR_EQ(ret, expected_ret);
  }
}

TEST_F(OpSliceScatterTensorOutTest, LegalStepsSupported) {
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

  // Expected ret for op_slice_scatter_out(input, src, /*dim=*/1, /*start=*/0, /*end=*/10, /*step=*/1, out),
  // src shape should equal to input[:，0:3:1, :]
  Tensor src_0 = tf.make(
    /*sizes=*/{2, 3, 4},
    /*data=*/{
      // [0, :, :]
      -1.,  -2.,  -3.,  -4., // [0, 0, :]
      -5.,  -6.,  -7.,  -8., // [0, 1, :]
      -9., -10., -11., -12., // [0, 2, :]

      // [1, :, :]
       1.,   2.,   3.,   4., // [1, 0, :]
       5.,   6.,   7.,   8., // [1, 1, :]
       9.,  10.,  11.,  12., // [1, 2, :]
    });
  Tensor expected_0 = tf.make(
    /*sizes=*/{2, 3, 4},
    /*data=*/{
      // [0, :, :]
      -1.,  -2.,  -3.,  -4., // [0, 0, :]
      -5.,  -6.,  -7.,  -8., // [0, 1, :]
      -9., -10., -11., -12., // [0, 2, :]

      // [1, :, :]
       1.,   2.,   3.,   4., // [1, 0, :]
       5.,   6.,   7.,   8., // [1, 1, :]
       9.,  10.,  11.,  12., // [1, 2, :]
    });
  // Expected ret for op_slice_scatter_out(input, src, /*dim=*/1, /*start=*/0, /*end=*/10, /*step=*/2, out),
  // src shape should equal to input[:，0:3:2, :]
  Tensor src_1 = tf.make(
    /*sizes=*/{2, 2, 4},
    /*data=*/{
      // [0, :, :]
      -1.,  -2.,  -3.,  -4., // [0, 0, :]
      -9., -10., -11., -12., // [0, 1, :]

      // [1, :, :]
       1.,   2.,   3.,   4., // [1, 0, :]
       9.,  10.,  11.,  12., // [1, 1, :]
    });
  Tensor expected_1 = tf.make(
    /*sizes=*/{2, 3, 4},
    /*data=*/{
      // [0, :, :]
      -1.,  -2.,  -3.,  -4., // [0, 0, :]
       5.,   6.,   7.,   8., // [0, 1, :]
      -9., -10., -11., -12., // [0, 2, :]

      // [1, :, :]
       1.,   2.,   3.,   4., // [1, 0, :]
      -5.,  -6.,  -7.,  -8., // [1, 1, :]
       9.,  10.,  11.,  12., // [1, 2, :]
    });
  // Expected ret for op_slice_scatter_out(input, src, /*dim=*/1, /*start=*/0, /*end=*/10, /*step=*/3, out),
  // src shape should equal to input[:，0:3:3, :] = input
  Tensor src_2 = tf.make(
    /*sizes=*/{2, 1, 4},
    /*data=*/{
      -1.,  -2.,  -3.,  -4., // [0, 0, :]
       1.,   2.,   3.,   4., // [1, 0, :]
    });
  Tensor expected_2 = tf.make(
    /*sizes=*/{2, 3, 4},
    /*data=*/{
      // [0, :, :]
      -1.,  -2.,  -3.,  -4., // [0, 0, :]
       5.,   6.,   7.,   8., // [0, 1, :]
       9.,  10.,  11.,  12., // [0, 2, :]

      // [1, :, :]
       1.,   2.,   3.,   4., // [1, 0, :]
      -5.,  -6.,  -7.,  -8., // [1, 1, :]
      -9., -10., -11., -12., // [1, 2, :]
    });
  // clang-format on

  std::vector<Tensor> src_tensors = {src_0, src_1, src_2};
  std::vector<Tensor> expected_rets = {expected_0, expected_1, expected_2};

  // In this test, we maintain start and dim as 0 and 1, also set the
  // end large enough to hold any step
  int64_t start = 0;
  int64_t dim = 1;
  int64_t end = 10;
  for (int64_t step = 1; step < 4; step++) {
    int64_t testcase_idx = step - 1;

    auto src = src_tensors[testcase_idx];
    auto expected_ret = expected_rets[testcase_idx];
    Tensor out = tf.zeros_like(expected_ret);

    // Should always return the provided out Tensor.
    // The ret shall meet the expectation.
    Tensor ret = op_slice_scatter_out(input, src, dim, start, end, step, out);
    EXPECT_TENSOR_EQ(out, ret);
    EXPECT_TENSOR_EQ(ret, expected_ret);
  }
}

/// A generic smoke test that works for any dtype that supports ones() and
/// zeros().
TEST_F(OpSliceScatterTensorOutTest, AllRealDtypesSupported) {
#define TEST_ENTRY(ctype, dtype) test_dtype<ctype, ScalarType::dtype>();
  ET_FORALL_REALHBF16_TYPES(TEST_ENTRY);
#undef TEST_ENTRY
  // TODO: Also add tests for half, complex, quantized, and other types. Easiest
  // way to do that would be to make TensorFactory support zeros() and ones()
  // for those types.
}

TEST_F(OpSliceScatterTensorOutTest, EmptyInputSupported) {
  TensorFactory<ScalarType::Int> tf;

  Tensor input = tf.ones({1, 0, 1});
  Tensor src = tf.zeros({1, 0, 1});
  Tensor out = tf.zeros({1, 0, 1});

  Tensor expect = tf.ones({1, 0, 1});

  // Some invalid dim values.
  for (int64_t dim = 0; dim > input.dim(); dim++) {
    Tensor ret = op_slice_scatter_out(
        input, src, dim, /*start=*/0, /*end=*/1, /*step=*/1, out);
    EXPECT_TENSOR_EQ(ret, out);

    // All operations in this test share same ground truth
    EXPECT_TENSOR_EQ(ret, expect);
  }
}

TEST_F(OpSliceScatterTensorOutTest, EmptySizeInputDies) {
  TensorFactory<ScalarType::Int> tf;

  Tensor input = tf.ones({});
  Tensor src = tf.ones({});
  Tensor out = tf.ones({});

  // The operation shall die whatever the end is.
  ET_EXPECT_KERNEL_FAILURE(
      context_,
      op_slice_scatter_out(
          input, src, /*dim=*/0, /*start=*/0, /*end=*/0, /*step=*/1, out));
  ET_EXPECT_KERNEL_FAILURE(
      context_,
      op_slice_scatter_out(
          input, src, /*dim=*/0, /*start=*/0, /*end=*/1, /*step=*/1, out));
}

TEST_F(OpSliceScatterTensorOutTest, NonPostiveStepsDies) {
  TensorFactory<ScalarType::Int> tf;

  Tensor input = tf.ones({1, 1, 1});
  Tensor src = tf.zeros({1, 1, 1});
  Tensor out = tf.zeros({1, 1, 1});

  // Some invalid step values.
  const std::vector<int64_t> invalid_steps = {-2, -1, 0};
  for (int64_t step : invalid_steps) {
    ET_EXPECT_KERNEL_FAILURE(
        context_,
        op_slice_scatter_out(
            input, src, /*dim=*/0, /*start=*/0, /*end=*/1, /*step=*/step, out));
  }
}

TEST_F(OpSliceScatterTensorOutTest, DimOutOfBoundDies) {
  TensorFactory<ScalarType::Int> tf;

  Tensor input = tf.ones({1, 1, 1});
  Tensor src = tf.zeros({1, 1, 1});
  Tensor out = tf.zeros({1, 1, 1});

  // Some invalid dim values.
  const std::vector<int64_t> invalid_dims = {3, 4, 5, -4, -5, -6};
  for (int64_t dim : invalid_dims) {
    ET_EXPECT_KERNEL_FAILURE(
        context_,
        op_slice_scatter_out(
            input, src, dim, /*start=*/0, /*end=*/1, /*step=*/1, out));
  }
}

TEST_F(OpSliceScatterTensorOutTest, MismatchedOutDtypesDies) {
  TensorFactory<ScalarType::Int> tf_int;
  TensorFactory<ScalarType::Float> tf_float;
  Tensor input = tf_int.zeros({1, 2, 2});
  Tensor src = tf_int.zeros({1, 2, 2});

  // Size is compatible to the output, but a mismatched dtype.
  Tensor out = tf_float.ones({1, 2, 2});

  ET_EXPECT_KERNEL_FAILURE(
      context_,
      op_slice_scatter_out(
          input, src, /*dim=*/0, /*start=*/0, /*end=*/1, /*step=*/1, out));
}

TEST_F(OpSliceScatterTensorOutTest, OutSizeMismatchDimDies) {
  if (torch::executor::testing::SupportedFeatures::get()->is_aten) {
    GTEST_SKIP() << "ATen kernel can handle out with mismatched dimensions";
  }
  TensorFactory<ScalarType::Int> tf;

  Tensor input = tf.zeros({2, 4, 7, 5});
  Tensor src = tf.zeros({2, 4, 7, 5});

  // Should be {2, 4, 7, 5}
  Tensor out = tf.zeros({2, 4, 7});

  ET_EXPECT_KERNEL_FAILURE(
      context_,
      op_slice_scatter_out(
          input, src, /*dim=*/0, /*start=*/0, /*end=*/2, /*step=*/1, out));
}

TEST_F(OpSliceScatterTensorOutTest, SrcSizeMismatchDimDies) {
  if (torch::executor::testing::SupportedFeatures::get()->is_aten) {
    GTEST_SKIP() << "ATen kernel can handle out with mismatched dimensions";
  }
  TensorFactory<ScalarType::Int> tf;

  Tensor input = tf.zeros({2, 4, 7, 5});
  Tensor src = tf.zeros({2, 4, 7});

  // Should be {2, 4, 7, 5}
  Tensor out = tf.zeros({2, 4, 7, 5});

  ET_EXPECT_KERNEL_FAILURE(
      context_,
      op_slice_scatter_out(
          input, src, /*dim=*/0, /*start=*/0, /*end=*/2, /*step=*/1, out));
}

TEST_F(OpSliceScatterTensorOutTest, DefaultStartValSupported) {
  TensorFactory<ScalarType::Int> tf;

  Tensor input = tf.zeros({2, 4, 7, 5});
  Tensor src = tf.ones({2, 4, 7, 5});

  Tensor out = tf.zeros({2, 4, 7, 5});
  Tensor expected = tf.ones({2, 4, 7, 5});

  Tensor ret_default_start = op_slice_scatter_out(
      input,
      src,
      /*dim=*/0,
      /*start=*/exec_aten::nullopt,
      /*end=*/2,
      /*step=*/1,
      out);
  EXPECT_TENSOR_EQ(ret_default_start, out);
  EXPECT_TENSOR_EQ(ret_default_start, expected);
}

TEST_F(OpSliceScatterTensorOutTest, DefaultEndValSupported) {
  TensorFactory<ScalarType::Int> tf;

  Tensor input = tf.zeros({2, 4, 7, 5});
  Tensor src = tf.ones({2, 4, 7, 5});

  Tensor out = tf.zeros({2, 4, 7, 5});
  Tensor expected = tf.ones({2, 4, 7, 5});

  Tensor ret_default_end = op_slice_scatter_out(
      input,
      src,
      /*dim=*/0,
      /*start=*/0,
      /*end=*/exec_aten::nullopt,
      /*step=*/1,
      out);
  EXPECT_TENSOR_EQ(ret_default_end, out);
  EXPECT_TENSOR_EQ(ret_default_end, expected);
}

TEST_F(OpSliceScatterTensorOutTest, DynamicShapeTest) {
  TensorFactory<ScalarType::Int> tf;

  Tensor input = tf.zeros({1, 4, 4});
  Tensor src = tf.ones({1, 4, 4});

  Tensor out =
      tf.zeros({1, 2, 8}, torch::executor::TensorShapeDynamism::DYNAMIC_BOUND);
  Tensor expected = tf.ones({1, 4, 4});

  Tensor ret_default_end = op_slice_scatter_out(
      input,
      src,
      /*dim=*/0,
      /*start=*/0,
      /*end=*/exec_aten::nullopt,
      /*step=*/1,
      out);
  EXPECT_TENSOR_EQ(ret_default_end, out);
  EXPECT_TENSOR_EQ(ret_default_end, expected);
}

TEST_F(OpSliceScatterTensorOutTest, LargeEndValue) {
  TensorFactory<ScalarType::Int> tf;

  Tensor input = tf.zeros({1, 1, 2, 5, 3, 3});
  Tensor src = tf.ones({1, 1, 2, 5, 3, 3});

  Tensor out = tf.zeros({1, 1, 2, 5, 3, 3});
  Tensor expected = tf.ones({1, 1, 2, 5, 3, 3});

  Tensor ret = op_slice_scatter_out(
      input,
      src,
      /*dim=*/1,
      /*start=*/0,
      /*end=*/9223372036854775807,
      /*step=*/1,
      out);
  EXPECT_TENSOR_EQ(ret, out);
  EXPECT_TENSOR_EQ(ret, expected);
}
