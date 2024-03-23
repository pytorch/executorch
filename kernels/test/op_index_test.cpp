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
using exec_aten::optional;
using exec_aten::ScalarType;
using exec_aten::Tensor;
using torch::executor::testing::TensorFactory;

using OptTensorArrayRef = ArrayRef<optional<Tensor>>;

class OpIndexTensorOutTest : public OperatorTest {
 protected:
  Tensor& op_index_tensor_out(
      const Tensor& input,
      OptTensorArrayRef indices,
      Tensor& out) {
#ifdef USE_ATEN_LIB
    c10::List<c10::optional<at::Tensor>> indices_list(indices);
    return torch::executor::aten::index_outf(
        context_, input, indices_list, out);
#else
    return torch::executor::aten::index_outf(context_, input, indices, out);
#endif
  }

  template <
      exec_aten::ScalarType INPUT_DTYPE,
      exec_aten::ScalarType INDEX_DTYPE,
      exec_aten::ScalarType OUTPUT_DTYPE>
  void test_dtype() {
    TensorFactory<INPUT_DTYPE> tf;
    TensorFactory<INDEX_DTYPE> tfl;
    TensorFactory<OUTPUT_DTYPE> tfo;
    TensorFactory<ScalarType::Bool> tfb;

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

    // indices [0, 1, 2], [1, 0, 3], expressed two different ways
    optional<Tensor> indices[] = {
        optional<Tensor>(tfl.make({2}, {0, 1})),
        optional<Tensor>(tfl.make({2}, {1, 0})),
        optional<Tensor>(tfl.make({2}, {2, 3}))};

    optional<Tensor> indices_mixed[] = {
        optional<Tensor>(tfl.make({2}, {0, 1})),
        optional<Tensor>(tfb.make({2}, {false, true})),
        optional<Tensor>(tfl.make({2}, {2, 3}))};

    std::vector<int32_t> out_size{2};

    Tensor out_0 = tfo.zeros(out_size);
    Tensor ret_0 = op_index_tensor_out(x, /*indices=*/indices, out_0);

    EXPECT_TENSOR_EQ(ret_0, out_0);
    EXPECT_TENSOR_EQ(ret_0, tfo.make(out_size, {0, 1}));

    // Repeat the test with alternative indices representation

    Tensor out_0_with_mixed = tfo.zeros(out_size);
    Tensor ret_0_with_mixed =
        op_index_tensor_out(x, /*indices=*/indices, out_0_with_mixed);

    EXPECT_TENSOR_EQ(ret_0_with_mixed, out_0_with_mixed);
    EXPECT_TENSOR_EQ(ret_0_with_mixed, tfo.make(out_size, {0, 1}));
  }

  /**
   * Generic test for integral index lists
   */
  void test_dtype_enumerate_in_types() {
#define TEST_ENTRY(ctype, dtype) \
  test_dtype<ScalarType::dtype, ScalarType::Long, ScalarType::dtype>();

    ET_FORALL_REAL_TYPES_AND(Bool, TEST_ENTRY);

#undef TEST_ENTRY
  }

  // Run the test by selecting elements in input
  void run_test_cases(
      const Tensor& x,
      OptTensorArrayRef indices,
      const Tensor& expected) {
    // Generated out tensor sharing same size and dtype with expected tensor
    TensorFactory<ScalarType::Double> tf;

    const std::vector<int32_t> out_size(
        expected.sizes().begin(), expected.sizes().end());
    Tensor out = tf.ones(out_size);

    Tensor ret = op_index_tensor_out(x, indices, out);
    EXPECT_TENSOR_EQ(out, ret);
    EXPECT_TENSOR_EQ(ret, expected);
  }
};

//
// Correctness Tests
//

TEST_F(OpIndexTensorOutTest, IndexMask) {
  TensorFactory<ScalarType::Double> tf;
  TensorFactory<ScalarType::Bool> tfb;
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
  Tensor indices = tfb.make(
      {2, 3, 4},
      {
         // [0, :, :]
          true, false, false, false, // [0, 0, :]
         false, false,  true, false, // [0, 1, :]
         false, false, false, false, // [0, 2, :]

         // [1, :, :]
         false,  true, false, false, // [1, 0, :]
         false, false, false, false, // [1, 1, :]
         false, false,  true, false, // [1, 2, :]
      });
  // clang-format on

  // clang-format off
  Tensor expected = tf.make(
    {4},
    {1., 7., -2., -11.}
  );
  // clang-format on

  run_test_cases(x, {indices}, expected);
}

TEST_F(OpIndexTensorOutTest, SelectFrontDimAllIndexes) {
  TensorFactory<ScalarType::Double> tf;
  TensorFactory<ScalarType::Int> tfi;
  TensorFactory<ScalarType::Long> tfl;
  TensorFactory<ScalarType::Bool> tfb;
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

  // Try to select the input value at indices
  // [1, 0, 1], [1, 0, 2]. This is expressed in various ways to test different
  // indexing expressions.
  optional<Tensor> indices[] = {
      optional<Tensor>(tfl.make({1}, {1})),
      optional<Tensor>(tfl.make({1}, {0})),
      optional<Tensor>(tfl.make({2}, {1, 2}))};

  optional<Tensor> indices_int[] = {
      optional<Tensor>(tfi.make({1}, {1})),
      optional<Tensor>(tfi.make({1}, {0})),
      optional<Tensor>(tfi.make({2}, {1, 2}))};

  optional<Tensor> indices_negative[] = {
      optional<Tensor>(tfl.make({1}, {-1})),
      optional<Tensor>(tfl.make({1}, {0})),
      optional<Tensor>(tfl.make({2}, {-3, -2}))};

  optional<Tensor> indices_bool[] = {
      optional<Tensor>(tfb.make({2}, {false, true})),
      optional<Tensor>(tfb.make({3}, {true, false, false})),
      optional<Tensor>(tfl.make({2}, {-3, -2}))};

  optional<Tensor> indices_mixed[] = {
      optional<Tensor>(tfb.make({2}, {false, true})),
      optional<Tensor>(tfl.make({1}, {0})),
      optional<Tensor>(tfl.make({2}, {-3, -2}))};

  std::vector<int32_t> out_size{2};

  // clang-format off
  Tensor expected = tf.make(
    out_size,
    {-2., -3.,}
  );
  // clang-format on

  run_test_cases(x, /*indices=*/indices, expected);
  run_test_cases(x, /*indices=*/indices_int, expected);
  run_test_cases(x, /*indices=*/indices_negative, expected);
  run_test_cases(x, /*indices=*/indices_bool, expected);
  run_test_cases(x, /*indices=*/indices_mixed, expected);
}

TEST_F(OpIndexTensorOutTest, SelectTwoValuesAtSameIndex) {
  TensorFactory<ScalarType::Double> tf;
  TensorFactory<ScalarType::Long> tfl;
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

  // Try to select the value at the same index
  optional<Tensor> indices[] = {
      optional<Tensor>(tfl.make({1, 2}, {0, 0})),
      optional<Tensor>(tfl.make({1, 2}, {1, 1})),
      optional<Tensor>(tfl.make({1, 2}, {2, 2}))};

  std::vector<int32_t> out_size{1, 2}; // In ATen the size is (1, 2)

  // clang-format off
  Tensor expected = tf.make(
    out_size,
    {7., 7.,}
  );
  // clang-format on

  run_test_cases(x, /*indices=*/indices, expected);
}

TEST_F(OpIndexTensorOutTest, IndicesFewerThanInputDimSupported) {
  TensorFactory<ScalarType::Double> tf;
  TensorFactory<ScalarType::Int> tfi;
  TensorFactory<ScalarType::Long> tfl;
  TensorFactory<ScalarType::Bool> tfb;
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

  // Try to select the input value at indices
  // [1, 0, :], [1, 1, :]. This is expressed in various ways to test different
  // indexing expressions.

  optional<Tensor> indices[] = {
      optional<Tensor>(tfl.make({1}, {1})),
      optional<Tensor>(tfl.make({2}, {0, 1}))};

  optional<Tensor> indices_mixed[] = {
      optional<Tensor>(tfi.make({1}, {-1})),
      optional<Tensor>(tfb.make({3}, {true, true, false}))};

  std::vector<int32_t> out_size{2, 4};

  // clang-format off
  Tensor expected = tf.make(
    out_size,
    {
      -1.,  -2.,  -3.,  -4.,
      -5.,  -6.,  -7.,  -8.,
    }
  );
  // clang-format on

  run_test_cases(x, /*indices=*/indices, expected);
  run_test_cases(x, /*indices=*/indices_mixed, expected);
}

TEST_F(OpIndexTensorOutTest, IndicesWithNullTensorsSupported) {
  TensorFactory<ScalarType::Double> tf;
  TensorFactory<ScalarType::Long> tfl;
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

  optional<Tensor> indices0[] = {
      optional<Tensor>(),
      optional<Tensor>(tfl.make({1}, {1})),
      optional<Tensor>(tfl.make({2}, {0, 1}))};

  // clang-format off
  Tensor expected0 = tf.make(
    {2, 2},
    {
       5.,   6.,
      -5.,  -6.,
    }
  );
  // clang-format on

  run_test_cases(x, /*indices=*/indices0, expected0);

  optional<Tensor> indices1[] = {
      optional<Tensor>(tfl.make({1}, {1})),
      optional<Tensor>(),
      optional<Tensor>(tfl.make({2}, {0, 1}))};

  // clang-format off
  Tensor expected1 = tf.make(
    {2, 3},
    {
      -1.,  -5.,  -9.,
      -2.,  -6., -10.,
    }
  );
  // clang-format on

  run_test_cases(x, /*indices=*/indices1, expected1);

  optional<Tensor> indices2[] = {
      optional<Tensor>(tfl.make({1}, {1})),
      optional<Tensor>(tfl.make({2}, {0, 1})),
      optional<Tensor>()};

  // clang-format off
  Tensor expected2 = tf.make(
    {2, 4},
    {
      -1.,  -2.,  -3.,  -4.,
      -5.,  -6.,  -7.,  -8.,
    }
  );
  // clang-format on

  run_test_cases(x, /*indices=*/indices2, expected2);
}

TEST_F(OpIndexTensorOutTest, IndicesWithOnlyNullTensorsSupported) {
  if (torch::executor::testing::SupportedFeatures::get()->is_aten) {
    GTEST_SKIP() << "ATen kernel test fails";
  }
  TensorFactory<ScalarType::Double> tf;

  Tensor x = tf.make({2, 3}, {1., 2., 3., 4., 5., 6.});
  optional<Tensor> indices0[] = {optional<Tensor>()};
  run_test_cases(x, indices0, x);

  optional<Tensor> indices1[] = {optional<Tensor>(), optional<Tensor>()};
  run_test_cases(x, indices1, x);

  optional<Tensor> indices2[] = {
      optional<Tensor>(), optional<Tensor>(), optional<Tensor>()};
  Tensor out = tf.ones({2, 3});
  ET_EXPECT_KERNEL_FAILURE_WITH_MSG(
      context_, op_index_tensor_out(x, indices2, out), "");
}

TEST_F(OpIndexTensorOutTest, EmptyIndicesSupported) {
  if (torch::executor::testing::SupportedFeatures::get()->is_aten) {
    GTEST_SKIP() << "ATen kernel test fails";
  }
  TensorFactory<ScalarType::Float> tf;

  // Using empty tensors as input.
  Tensor x = tf.make({2}, {1., 2.});

  Tensor out = tf.zeros({2});

  op_index_tensor_out(x, /*indices=*/{}, out);
  EXPECT_TENSOR_EQ(out, x);
  // Success if it doesn't assert on the weird-shaped empty input and the
  // ret is still a empty array
}

//
// Test that all dtypes are supported
//

TEST_F(OpIndexTensorOutTest, AllDtypesSupportedForInput) {
  test_dtype_enumerate_in_types();
}

TEST_F(OpIndexTensorOutTest, AllDtypesSupportedForIndex) {
  test_dtype<ScalarType::Double, ScalarType::Long, ScalarType::Double>();
  test_dtype<ScalarType::Double, ScalarType::Int, ScalarType::Double>();
}

//
// Death Tests
//

TEST_F(OpIndexTensorOutTest, IndexOutOfBoundDies) {
  TensorFactory<ScalarType::Int> tf;
  TensorFactory<ScalarType::Long> tfl;

  Tensor x = tf.ones({1, 1, 1});
  Tensor out = tf.zeros({1, 1, 1});
  Tensor index = tfl.make({1}, {5});

  ET_EXPECT_KERNEL_FAILURE_WITH_MSG(
      context_, op_index_tensor_out(x, /*indices=*/{index}, out), "");
}

TEST_F(OpIndexTensorOutTest, NegativeIndexOutOfBoundDies) {
  TensorFactory<ScalarType::Int> tf;
  TensorFactory<ScalarType::Long> tfl;

  Tensor x = tf.ones({1, 1, 1});
  Tensor out = tf.zeros({1, 1, 1});
  Tensor index = tfl.make({1}, {-5});

  ET_EXPECT_KERNEL_FAILURE_WITH_MSG(
      context_, op_index_tensor_out(x, /*indices=*/{index}, out), "");
}

TEST_F(OpIndexTensorOutTest, TooManyBooleanIndexCountDies) {
  TensorFactory<ScalarType::Float> tf;
  TensorFactory<ScalarType::Bool> tfb;

  Tensor x = tf.ones({1, 1, 1});
  Tensor out = tf.zeros({1, 1, 1});
  Tensor index = tfb.make({3}, {true, false, false});

  ET_EXPECT_KERNEL_FAILURE_WITH_MSG(
      context_, op_index_tensor_out(x, /*indices=*/{index}, out), "");
}

TEST_F(OpIndexTensorOutTest, TooFewBooleanIndexCountDies) {
  TensorFactory<ScalarType::Float> tf;
  TensorFactory<ScalarType::Bool> tfb;

  Tensor x = tf.ones({4});
  Tensor out = tf.zeros({1});
  Tensor index = tfb.make({1}, {true});

  // ATen kernel will throw exception instead of death
  ET_EXPECT_KERNEL_FAILURE_WITH_MSG(
      context_, op_index_tensor_out(x, /*indices=*/{index}, out), "");
}

TEST_F(OpIndexTensorOutTest, MismatchedIndexMaskDies) {
  TensorFactory<ScalarType::Float> tf;
  TensorFactory<ScalarType::Bool> tfb;

  Tensor x = tf.ones({4, 4});
  Tensor out = tf.zeros({9});
  Tensor index = tfb.ones({3, 3});

  // ATen kernel will throw exception instead of death
  ET_EXPECT_KERNEL_FAILURE_WITH_MSG(
      context_, op_index_tensor_out(x, /*indices=*/{index}, out), "");
}

TEST_F(OpIndexTensorOutTest, MismatchedOutputDimDies) {
  TensorFactory<ScalarType::Int> tf;
  TensorFactory<ScalarType::Long> tfl;

  Tensor x = tf.zeros({2, 4, 7, 5});
  Tensor index = tfl.make({1}, {3});

  // Should be {1, 4, 7, 5}
  Tensor out = tf.zeros({2, 4});

  ET_EXPECT_KERNEL_FAILURE_WITH_MSG(
      context_, op_index_tensor_out(x, /*indices=*/{index}, out), "");
}

TEST_F(OpIndexTensorOutTest, InvalidIndicesDtypeDies) {
  TensorFactory<ScalarType::Int> tf;
  TensorFactory<ScalarType::Float> tff;

  Tensor x = tf.zeros({2, 4, 7, 5});
  Tensor index = tff.make({1}, {3});

  Tensor out = tf.zeros({1, 4, 7, 5});

  ET_EXPECT_KERNEL_FAILURE_WITH_MSG(
      context_, op_index_tensor_out(x, /*indices=*/{index}, out), "");
}

TEST_F(OpIndexTensorOutTest, InvalidIndicesShapesDies) {
  TensorFactory<ScalarType::Float> tf;
  TensorFactory<ScalarType::Long> tfl;

  Tensor x = tf.zeros({2, 4, 7, 5});
  // clang-format off
  optional<Tensor> indices[] = {
      optional<Tensor>(tfl.make({3}, {1, 1, 1,})),
      optional<Tensor>(tfl.make({2}, {1, 2}))};

  Tensor out = tf.ones({3, 7, 5});
  // clang-format on

  ET_EXPECT_KERNEL_FAILURE_WITH_MSG(
      context_, op_index_tensor_out(x, indices, out), "");
}

TEST_F(OpIndexTensorOutTest, InvalidIndicesShapeDies2) {
  if (torch::executor::testing::SupportedFeatures::get()->is_aten) {
    GTEST_SKIP() << "";
  }
  TensorFactory<ScalarType::Float> tf;
  TensorFactory<ScalarType::Long> tfl;

  Tensor x = tf.zeros({4, 4});
  // clang-format off
  optional<Tensor> indices[] = {
      optional<Tensor>(tfl.make({2, 2}, {1, 1, 1, 1,})),
      optional<Tensor>(tfl.make({1, 2}, {3, 0,}))};

  Tensor out = tf.ones({4});
  // clang-format on

  ET_EXPECT_KERNEL_FAILURE_WITH_MSG(
      context_, op_index_tensor_out(x, indices, out), "");
}

//
// Dynamic Shape Tests
//

// Test whether resize works when out is having larger size
TEST_F(OpIndexTensorOutTest, UpperBoundOutTensor) {
  TensorFactory<ScalarType::Double> tf;
  TensorFactory<ScalarType::Long> tfl;
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

  // Try to select the tensor from the input
  // indices [0, 2, 2], [1, 1, 2]
  optional<Tensor> indices[] = {
      optional<Tensor>(tfl.make({1, 2}, {0, 1})),
      optional<Tensor>(tfl.make({1, 2}, {2, 1})),
      optional<Tensor>(tfl.make({1, 2}, {2, 2}))};

  Tensor out =
      tf.zeros({5, 5}, torch::executor::TensorShapeDynamism::DYNAMIC_BOUND);
  // clang-format off
  Tensor expected = tf.make(
    {1, 2},
    {
          11.,  -7.
    }
  );
  // clang-format on

  Tensor ret = op_index_tensor_out(x, indices, out);
  EXPECT_TENSOR_EQ(out, ret);
  EXPECT_TENSOR_EQ(ret, expected);
}
