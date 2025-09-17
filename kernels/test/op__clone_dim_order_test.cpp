/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <cstdint>

#include <executorch/kernels/test/FunctionHeaderWrapper.h> // Declares the operator.
#include <executorch/kernels/test/TestUtil.h>
#include <executorch/kernels/test/supported_features.h>
#include <executorch/runtime/core/exec_aten/exec_aten.h>
#include <executorch/runtime/core/exec_aten/testing_util/tensor_factory.h>
#include <executorch/runtime/core/exec_aten/testing_util/tensor_util.h>

#include <gtest/gtest.h>

using namespace ::testing;
using executorch::aten::ArrayRef;
using executorch::aten::ScalarType;
using executorch::aten::Tensor;
using std::optional;
using torch::executor::testing::TensorFactory;

class OpDimOrderCloneTest : public OperatorTest {
 protected:
  Tensor& op__clone_dim_order_out(
      const Tensor& self,
      bool non_blocking,
      std::optional<ArrayRef<int64_t>> dim_order,
      Tensor& out) {
    return torch::executor::dim_order_ops::_clone_dim_order_outf(
        context_, self, non_blocking, dim_order, out);
  }

  template <typename INPUT_CTYPE, typename OUTPUT_CTYPE>
  std::vector<OUTPUT_CTYPE> vector_type_cast(std::vector<INPUT_CTYPE> input) {
    std::vector<OUTPUT_CTYPE> output(input.size());
    std::transform(
        input.begin(), input.end(), output.begin(), [](INPUT_CTYPE x) {
          return static_cast<OUTPUT_CTYPE>(x);
        });
    return output;
  }

  template <typename INPUT_CTYPE, typename OUTPUT_CTYPE>
  struct ToTestCase {
    const std::vector<int32_t> sizes;
    const std::vector<INPUT_CTYPE> data_in;
    const std::vector<OUTPUT_CTYPE> data_out;
  };

  template <typename CTYPE, ScalarType DTYPE>
  void test_runner_clone(std::vector<ToTestCase<double, double>> test_cases) {
    TensorFactory<DTYPE> tf_in;
    TensorFactory<DTYPE> tf_out;

    for (const auto& test_case : test_cases) {
      auto data_in = vector_type_cast<double, CTYPE>(test_case.data_in);

      Tensor input = tf_in.make(test_case.sizes, data_in);
      Tensor output = tf_out.zeros_like(input);

      std::vector<int64_t> dim_order_vec;
      for (int64_t i = 0; i < input.dim(); i++) {
        dim_order_vec.push_back(i);
      }
      ArrayRef<int64_t> dim_order(dim_order_vec.data(), dim_order_vec.size());

      Tensor ret = op__clone_dim_order_out(
          /*self=*/input,
          /*non_blocking=*/false,
          dim_order,
          output);

      Tensor expected = tf_out.make(test_case.sizes, data_in);

      // Verifies that the returned and output tensor from _clone_dim_order both
      // match the original input (expected).
      EXPECT_TENSOR_EQ(ret, output);
      EXPECT_TENSOR_EQ(ret, expected);
    }
  }

  // Helper for testing dynamic shape outputs.
  void test_dynamic_shape(
      const std::vector<int32_t>& out_shape,
      enum torch::executor::TensorShapeDynamism dynamism) {
    TensorFactory<ScalarType::Float> tf;

    Tensor x = tf.make(
        {2, 3},
        {0.49625658988952637,
         0.7682217955589294,
         0.08847743272781372,
         0.13203048706054688,
         0.30742281675338745,
         0.6340786814689636});
    Tensor expected = tf.make(
        {2, 3},
        {0.49625658988952637,
         0.7682217955589294,
         0.08847743272781372,
         0.13203048706054688,
         0.30742281675338745,
         0.6340786814689636});

    bool non_blocking = false;

    Tensor out = tf.zeros(out_shape, dynamism);

    std::vector<int64_t> dim_order_vec;
    for (int64_t i = 0; i < x.dim(); i++) {
      dim_order_vec.push_back(i);
    }
    ArrayRef<int64_t> dim_order(dim_order_vec.data(), dim_order_vec.size());

    Tensor ret = op__clone_dim_order_out(
        /*self=*/x, non_blocking, dim_order, out);

    EXPECT_TENSOR_EQ(out, expected);
    EXPECT_TENSOR_EQ(ret, expected);
  }
};

// Clones tensors of all real dtypes.
TEST_F(OpDimOrderCloneTest, AllDtypesSupported) {
  std::vector<ToTestCase<double, double>> test_cases = {
      {
          /*sizes=*/{2, 4},
          /*data_in=*/{2.11, 3.2, 2.3, 4.0, 1.1, 5.2, 1.1, 6.3},
          /*data_out=*/{}, // data_out shouldn't be used in test_runner_clone
      },
      {
          /*sizes=*/{3, 4, 0, 5},
          /*data_in=*/{},
          /*data_out=*/{},
      },
      {
          /*sizes=*/{},
          /*data_in=*/{10.0},
          /*data_out=*/{}, // data_out shouldn't be used in test_runner_clone
      },
  };

#define TEST_KERNEL(CTYPE, DTYPE) \
  test_runner_clone<CTYPE, ScalarType::DTYPE>(test_cases);

  ET_FORALL_REAL_TYPES(TEST_KERNEL);

#undef TEST_KERNEL
}

// Cloning with mismatched input and output tensor shapes should fail.
TEST_F(OpDimOrderCloneTest, MismatchedSizesDie) {
  if (torch::executor::testing::SupportedFeatures::get()->is_aten) {
    GTEST_SKIP() << "Skipping: ATen kernel supports mismatched sizes.";
  }
  TensorFactory<ScalarType::Int> tf;
  Tensor input = tf.make(/*sizes=*/{3, 1, 1, 2}, /*data=*/{1, 2, 3, 4, 5, 6});
  Tensor out = tf.zeros({3, 2, 1, 1});
  std::vector<int64_t> dim_order_vec;
  for (int64_t i = 0; i < input.dim(); i++) {
    dim_order_vec.push_back(i);
  }
  ArrayRef<int64_t> dim_order(dim_order_vec.data(), dim_order_vec.size());

  ET_EXPECT_KERNEL_FAILURE(
      context_,
      op__clone_dim_order_out(
          /*self=*/input,
          /*non_blocking=*/false,
          dim_order,
          out));
}

// Cloning with an unsupported memory format should fail.
TEST_F(OpDimOrderCloneTest, MismatchedMemoryFormatDies) {
  if (torch::executor::testing::SupportedFeatures::get()->is_aten) {
    GTEST_SKIP()
        << "Skipping: ATen kernel supports non-contiguous memory formats.";
  }
  TensorFactory<ScalarType::Float> tf_in;
  TensorFactory<ScalarType::Float> tf_out;
  Tensor input =
      tf_in.make(/*sizes=*/{3, 1, 1, 2}, /*data=*/{1, 2, 3, 4, 5, 6});
  Tensor out = tf_out.zeros({3, 1, 1, 2});

  std::vector<int64_t> dim_order_vec;
  for (int64_t i = 0; i < input.dim(); i++) {
    dim_order_vec.push_back(i);
  }

  // Mutate dim_order_vec to create an illegal dim_order.
  dim_order_vec[1] = 3;
  dim_order_vec[3] = 1;
  ArrayRef<int64_t> dim_order(dim_order_vec.data(), dim_order_vec.size());

  ET_EXPECT_KERNEL_FAILURE(
      context_,
      op__clone_dim_order_out(
          /*self=*/input,
          /*non_blocking=*/false,
          dim_order,
          out));
}

// Cloning with non‑blocking=true should fail because portable kernels only
// support blocking.
TEST_F(OpDimOrderCloneTest, MismatchedBlockingDie) {
  if (torch::executor::testing::SupportedFeatures::get()->is_aten) {
    GTEST_SKIP()
        << "Skipping: ATen kernel supports non-blocking data transfer.";
  }
  TensorFactory<ScalarType::Int> tf;
  Tensor input = tf.make(/*sizes=*/{3, 1, 1, 2}, /*data=*/{1, 2, 3, 4, 5, 6});
  Tensor out = tf.zeros(/*sizes=*/{3, 1, 1, 2});

  std::vector<int64_t> dim_order_vec;
  for (int64_t i = 0; i < input.dim(); i++) {
    dim_order_vec.push_back(i);
  }
  ArrayRef<int64_t> dim_order(dim_order_vec.data(), dim_order_vec.size());

  ET_EXPECT_KERNEL_FAILURE(
      context_,
      op__clone_dim_order_out(
          /*self=*/input,
          /*non_blocking=*/true,
          dim_order,
          out));
}

TEST_F(OpDimOrderCloneTest, DynamicShapeUpperBoundSameAsExpected) {
  test_dynamic_shape(
      {2, 3}, torch::executor::TensorShapeDynamism::DYNAMIC_BOUND);
}

TEST_F(OpDimOrderCloneTest, DynamicShapeUpperBoundLargerThanExpected) {
  test_dynamic_shape(
      {10, 10}, torch::executor::TensorShapeDynamism::DYNAMIC_BOUND);
}

TEST_F(OpDimOrderCloneTest, DynamicShapeUnbound) {
  if (!torch::executor::testing::SupportedFeatures::get()->output_resize) {
    GTEST_SKIP() << "Skipping: Dynamic shape unbound not supported.";
  }
  test_dynamic_shape(
      {1, 1}, torch::executor::TensorShapeDynamism::DYNAMIC_UNBOUND);
}

TEST_F(OpDimOrderCloneTest, ContiguousToChannelsLast) {
  TensorFactory<ScalarType::Float> tf;

  // x is in contiguous dim order {0, 1, 2, 3}.
  // make_with_dimorder() defaults to contiguous when dim_order isn't specified.
  Tensor x = tf.make_with_dimorder(
      {3, 5, 2, 2},
      {0.2432, 0.5248, 0.5361, 0.8513, 0.8184, 0.8206, 0.7357, 0.9655, 0.6138,
       0.1112, 0.2799, 0.1079, 0.9680, 0.2548, 0.0393, 0.6002, 0.2257, 0.8766,
       0.2715, 0.1595, 0.2029, 0.7026, 0.6982, 0.8529, 0.4405, 0.6560, 0.9217,
       0.6372, 0.2446, 0.6590, 0.3866, 0.7185, 0.4439, 0.5346, 0.3179, 0.4492,
       0.3491, 0.6970, 0.8456, 0.2516, 0.2345, 0.2924, 0.7695, 0.0911, 0.8530,
       0.8560, 0.6909, 0.7719, 0.8923, 0.5546, 0.6978, 0.8151, 0.3007, 0.3961,
       0.8416, 0.4296, 0.7203, 0.8963, 0.3597, 0.5552});

  Tensor out = tf.full_channels_last({3, 5, 2, 2}, 0.0);
  Tensor expected = tf.make_with_dimorder(
      {3, 5, 2, 2},
      {0.2432, 0.8184, 0.6138, 0.9680, 0.2257, 0.5248, 0.8206, 0.1112, 0.2548,
       0.8766, 0.5361, 0.7357, 0.2799, 0.0393, 0.2715, 0.8513, 0.9655, 0.1079,
       0.6002, 0.1595, 0.2029, 0.4405, 0.2446, 0.4439, 0.3491, 0.7026, 0.6560,
       0.6590, 0.5346, 0.6970, 0.6982, 0.9217, 0.3866, 0.3179, 0.8456, 0.8529,
       0.6372, 0.7185, 0.4492, 0.2516, 0.2345, 0.8530, 0.8923, 0.3007, 0.7203,
       0.2924, 0.8560, 0.5546, 0.3961, 0.8963, 0.7695, 0.6909, 0.6978, 0.8416,
       0.3597, 0.0911, 0.7719, 0.8151, 0.4296, 0.5552},
      /*dim_order=*/{0, 2, 3, 1});

  std::vector<int64_t> dim_order_vec = {0, 2, 3, 1};
  executorch::aten::ArrayRef<int64_t> dim_order(
      dim_order_vec.data(), dim_order_vec.size());
  Tensor ret = op__clone_dim_order_out(
      /*self*/ x, /*non_blocking*/ false, /*dim_order*/ dim_order, out);

  EXPECT_TENSOR_EQ(out, expected);
  EXPECT_TENSOR_EQ(ret, expected);
}

TEST_F(OpDimOrderCloneTest, ChannelsLastToContiguous) {
  TensorFactory<ScalarType::Float> tf;

  Tensor out = tf.full({3, 5, 2, 2}, 0.0);

  // x is in channels_last dim order {0, 2, 3, 1}.
  Tensor x = tf.make_with_dimorder(
      {3, 5, 2, 2},
      {0.2432, 0.8184, 0.6138, 0.9680, 0.2257, 0.5248, 0.8206, 0.1112, 0.2548,
       0.8766, 0.5361, 0.7357, 0.2799, 0.0393, 0.2715, 0.8513, 0.9655, 0.1079,
       0.6002, 0.1595, 0.2029, 0.4405, 0.2446, 0.4439, 0.3491, 0.7026, 0.6560,
       0.6590, 0.5346, 0.6970, 0.6982, 0.9217, 0.3866, 0.3179, 0.8456, 0.8529,
       0.6372, 0.7185, 0.4492, 0.2516, 0.2345, 0.8530, 0.8923, 0.3007, 0.7203,
       0.2924, 0.8560, 0.5546, 0.3961, 0.8963, 0.7695, 0.6909, 0.6978, 0.8416,
       0.3597, 0.0911, 0.7719, 0.8151, 0.4296, 0.5552},
      /*dim_order=*/{0, 2, 3, 1});

  Tensor expected = tf.make_with_dimorder(
      {3, 5, 2, 2},
      {0.2432, 0.5248, 0.5361, 0.8513, 0.8184, 0.8206, 0.7357, 0.9655, 0.6138,
       0.1112, 0.2799, 0.1079, 0.9680, 0.2548, 0.0393, 0.6002, 0.2257, 0.8766,
       0.2715, 0.1595, 0.2029, 0.7026, 0.6982, 0.8529, 0.4405, 0.6560, 0.9217,
       0.6372, 0.2446, 0.6590, 0.3866, 0.7185, 0.4439, 0.5346, 0.3179, 0.4492,
       0.3491, 0.6970, 0.8456, 0.2516, 0.2345, 0.2924, 0.7695, 0.0911, 0.8530,
       0.8560, 0.6909, 0.7719, 0.8923, 0.5546, 0.6978, 0.8151, 0.3007, 0.3961,
       0.8416, 0.4296, 0.7203, 0.8963, 0.3597, 0.5552});

  std::vector<int64_t> dim_order_vec = {0, 1, 2, 3};
  executorch::aten::ArrayRef<int64_t> dim_order(
      dim_order_vec.data(), dim_order_vec.size());
  Tensor ret = op__clone_dim_order_out(
      /*self*/ x, /*non_blocking*/ false, /*dim_order*/ dim_order, out);

  EXPECT_TENSOR_EQ(out, expected);
  EXPECT_TENSOR_EQ(ret, expected);
}

TEST_F(OpDimOrderCloneTest, PreserveChannelsLast) {
  TensorFactory<ScalarType::Float> tf;

  Tensor out = tf.full_channels_last({3, 5, 2, 2}, 0.0);
  Tensor x = tf.make_with_dimorder(
      {3, 5, 2, 2},
      {0.2432, 0.8184, 0.6138, 0.9680, 0.2257, 0.5248, 0.8206, 0.1112, 0.2548,
       0.8766, 0.5361, 0.7357, 0.2799, 0.0393, 0.2715, 0.8513, 0.9655, 0.1079,
       0.6002, 0.1595, 0.2029, 0.4405, 0.2446, 0.4439, 0.3491, 0.7026, 0.6560,
       0.6590, 0.5346, 0.6970, 0.6982, 0.9217, 0.3866, 0.3179, 0.8456, 0.8529,
       0.6372, 0.7185, 0.4492, 0.2516, 0.2345, 0.8530, 0.8923, 0.3007, 0.7203,
       0.2924, 0.8560, 0.5546, 0.3961, 0.8963, 0.7695, 0.6909, 0.6978, 0.8416,
       0.3597, 0.0911, 0.7719, 0.8151, 0.4296, 0.5552},
      /*dim_order=*/{0, 2, 3, 1});

  Tensor expected = tf.make_with_dimorder(
      {3, 5, 2, 2},
      {0.2432, 0.8184, 0.6138, 0.9680, 0.2257, 0.5248, 0.8206, 0.1112, 0.2548,
       0.8766, 0.5361, 0.7357, 0.2799, 0.0393, 0.2715, 0.8513, 0.9655, 0.1079,
       0.6002, 0.1595, 0.2029, 0.4405, 0.2446, 0.4439, 0.3491, 0.7026, 0.6560,
       0.6590, 0.5346, 0.6970, 0.6982, 0.9217, 0.3866, 0.3179, 0.8456, 0.8529,
       0.6372, 0.7185, 0.4492, 0.2516, 0.2345, 0.8530, 0.8923, 0.3007, 0.7203,
       0.2924, 0.8560, 0.5546, 0.3961, 0.8963, 0.7695, 0.6909, 0.6978, 0.8416,
       0.3597, 0.0911, 0.7719, 0.8151, 0.4296, 0.5552},
      /*dim_order=*/{0, 2, 3, 1});

  Tensor ret = op__clone_dim_order_out(
      /*self*/ x,
      /*non_blocking*/ false,
      /*dim_order*/ executorch::aten::nullopt,
      out);

  EXPECT_TENSOR_EQ(out, expected);
  EXPECT_TENSOR_EQ(ret, expected);
}
