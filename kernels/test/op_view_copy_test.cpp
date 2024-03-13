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
#include <executorch/runtime/core/exec_aten/util/tensor_util.h>

#include <gtest/gtest.h>

using namespace ::testing;
using exec_aten::IntArrayRef;
using exec_aten::ScalarType;
using exec_aten::Tensor;
using torch::executor::testing::TensorFactory;

class OpViewTest : public OperatorTest {
 protected:
  Tensor& op_view_copy_out(const Tensor& self, IntArrayRef size, Tensor& out) {
    return torch::executor::aten::view_copy_outf(context_, self, size, out);
  }

  template <class CTYPE, exec_aten::ScalarType DTYPE>
  void run_view_test_cases(
      const Tensor& input,
      const std::vector<std::vector<int32_t>>& out_shapes) {
    TensorFactory<DTYPE> tf;
    for (std::vector<int32_t> size : out_shapes) {
      Tensor out = tf.ones(size);

      // The interface of op_view_copy_out should use int64_t as int, while
      // tensor size needs int32_t so we need to transfrom from int32_t to
      // int64_t to pass the size to op_view_copy_out function
      std::vector<int64_t> size_int64_t(size.size());
      std::transform(
          size.begin(), size.end(), size_int64_t.begin(), [](int32_t x) {
            return (int64_t)x;
          });

      Tensor ret = op_view_copy_out(
          input,
          exec_aten::ArrayRef<int64_t>(
              size_int64_t.data(), size_int64_t.size()),
          out);
      EXPECT_TENSOR_EQ(out, ret);
      EXPECT_TENSOR_DATA_EQ(input, out);
    }
  }

  // Test if op_view_copy_out works well under all kinds of legal input type.
  template <class CTYPE, exec_aten::ScalarType DTYPE>
  void test_dtype() {
    TensorFactory<DTYPE> tf;
    Tensor input = tf.make(/*sizes=*/{2, 4}, /*data=*/{0, 1, 1, 1, 0, 1, 0, 1});

    // Differne kinds of output shape meet the requirement (have same numel as
    // input)
    std::vector<std::vector<int32_t>> out_shapes = {
        {8},
        {8, 1},
        {1, 8},
        {2, 4},
        {4, 2},
        {2, 2, 2},
        {1, 2, 1, 2, 1, 2, 1},
    };

    run_view_test_cases<CTYPE, DTYPE>(input, out_shapes);
  }

  template <class CTYPE, ScalarType DTYPE>
  void test_empty_input() {
    TensorFactory<DTYPE> tf;
    Tensor input = tf.make(/*sizes=*/{3, 0, 1, 2}, /*data=*/{});
    // Differnet kinds of output shape meet the requirement (have same numel as
    // input)
    std::vector<std::vector<int32_t>> out_shapes = {
        {6, 0}, {6, 0, 0}, {3, 0, 1, 2}, {1, 0, 2, 3}};
    run_view_test_cases<CTYPE, DTYPE>(input, out_shapes);
  }

  /* %python
  import torch
  torch.manual_seed(0)
  x = torch.randint(10, (3, 4))
  res = x.view(2, 6)
  op = "op_view_copy_out"
  opt_setup_params = """
    int64_t size[] = {2, 6};
  """
  opt_extra_params = "size,"
  out_args = "out_shape, dynamism"
  dtype = "ScalarType::Int"
  check = "EXPECT_TENSOR_EQ" */

  void test_dynamic_shape(
      const std::vector<int32_t>& out_shape,
      enum torch::executor::TensorShapeDynamism dynamism) {
    /* %python
    %rewrite(unary_op) */

    TensorFactory<ScalarType::Int> tf;

    Tensor x = tf.make({3, 4}, {4, 9, 3, 0, 3, 9, 7, 3, 7, 3, 1, 6});
    Tensor expected = tf.make({2, 6}, {4, 9, 3, 0, 3, 9, 7, 3, 7, 3, 1, 6});

    int64_t size[] = {2, 6};

    Tensor out = tf.zeros(out_shape, dynamism);
    op_view_copy_out(x, size, out);
    EXPECT_TENSOR_EQ(out, expected);
  }
};

namespace {
std::vector<int64_t> vector_32_to_64(std::vector<int32_t> vector_32) {
  std::vector<int64_t> vector_64(vector_32.size());
  std::transform(
      vector_32.begin(), vector_32.end(), vector_64.begin(), [](int32_t x) {
        return (int64_t)x;
      });
  return vector_64;
}

} // namespace

// Regular test for op_view_copy_out.
TEST_F(OpViewTest, AllDtypesSupported) {
#define TEST_ENTRY(ctype, dtype) test_dtype<ctype, ScalarType::dtype>();
  ET_FORALL_REAL_TYPES_AND(Bool, TEST_ENTRY);
#undef TEST_ENTRY
}

TEST_F(OpViewTest, EmptyInputSupported) {
#define TEST_ENTRY(ctype, dtype) test_empty_input<ctype, ScalarType::dtype>();
  ET_FORALL_REAL_TYPES_AND(Bool, TEST_ENTRY);
#undef TEST_ENTRY
}

TEST_F(OpViewTest, InputOutputMismatchedSizesDie) {
  TensorFactory<ScalarType::Int> tf;
  std::vector<int32_t> size_in = {3, 1, 1, 2};
  std::vector<int32_t> size_out = {3, 2, 1, 2};

  Tensor input = tf.make(size_in, /*data=*/{1, 2, 3, 4, 5, 6});
  Tensor out = tf.ones(size_out);

  // The interface of op_view_copy_out should use int64_t as int, while tensor
  // size needs int32_t so we need to transfrom from int32_t to int64_t to pass
  // the size to op_view_copy_out function
  std::vector<int64_t> size_int64_t = vector_32_to_64(size_out);

  // The numel of input and output tensor should be same
  ET_EXPECT_KERNEL_FAILURE(
      context_,
      op_view_copy_out(
          input,
          exec_aten::ArrayRef<int64_t>(
              size_int64_t.data(), size_int64_t.size()),
          out));
}

TEST_F(OpViewTest, SizeOutputMismatchedSizesDie) {
  TensorFactory<ScalarType::Int> tf;
  std::vector<int32_t> size = {3, 1, 1, 2};
  std::vector<int32_t> size_target = {3, 2, 1, 2};
  Tensor input = tf.make(size, /*data=*/{1, 2, 3, 4, 5, 6});
  Tensor out = tf.ones(size);

  // The interface of op_view_copy_out should use int64_t as int, while tensor
  // size needs int32_t. So we need to transfrom from int32_t to int64_t to pass
  // the size to op_view_copy_out function
  std::vector<int64_t> size_int64_t = vector_32_to_64(size_target);

  // The target size and out.size() should be same
  ET_EXPECT_KERNEL_FAILURE(
      context_,
      op_view_copy_out(
          input,
          exec_aten::ArrayRef<int64_t>(
              size_int64_t.data(), size_int64_t.size()),
          out));
}

TEST_F(OpViewTest, MismatchedTypesDie) {
  TensorFactory<ScalarType::Int> tf_in;
  TensorFactory<ScalarType::Float> tf_out;
  std::vector<int32_t> size = {3, 1, 1, 2};

  Tensor input = tf_in.make(size, /*data=*/{1, 2, 3, 4, 5, 6});
  Tensor out = tf_out.ones(size);

  // The interface of op_view_copy_out should use int64_t as int, while tensor
  // size needs int32_t. So we need to transfrom from int32_t to int64_t to pass
  // the size to op_view_copy_out function
  std::vector<int64_t> size_int64_t = vector_32_to_64(size);

  // DTYPE of input and output should be same.
  ET_EXPECT_KERNEL_FAILURE(
      context_,
      op_view_copy_out(
          input,
          exec_aten::ArrayRef<int64_t>(
              size_int64_t.data(), size_int64_t.size()),
          out));
}

TEST_F(OpViewTest, SizeInfer) {
  TensorFactory<ScalarType::Float> tf_in;
  TensorFactory<ScalarType::Float> tf_out_valid, tf_out_invalid;
  std::vector<int32_t> in_size = {2, 2, 2};
  std::vector<int32_t> out_size_view = {4, 2};
  std::vector<int32_t> out_size_valid = {-1, 2};
  std::vector<int32_t> out_size_invalid = {-1, -1};

  Tensor input = tf_in.make(in_size, /*data=*/{1, 2, 3, 4, 5, 6, 7, 8});
  Tensor out = tf_out_valid.ones(out_size_view);

  // The interface of op_view_copy_out should use int64_t as int, while tensor
  // size needs int32_t. So we need to transfrom from int32_t to int64_t to pass
  // the size to op_view_copy_out function
  std::vector<int64_t> valid_size_int64_t = vector_32_to_64(out_size_valid);
  std::vector<int64_t> invalid_size_int64_t = vector_32_to_64(out_size_invalid);

  // Inferring one dimension is valid.
  op_view_copy_out(
      input,
      exec_aten::ArrayRef<int64_t>(
          valid_size_int64_t.data(), valid_size_int64_t.size()),
      out);
  EXPECT_TENSOR_DATA_EQ(input, out);
  // Inferring two dimensions is invalid.
  ET_EXPECT_KERNEL_FAILURE(
      context_,
      op_view_copy_out(
          input,
          exec_aten::ArrayRef<int64_t>(
              invalid_size_int64_t.data(), invalid_size_int64_t.size()),
          out));
}

#if !defined(USE_ATEN_LIB)
TEST_F(OpViewTest, UpperBoundOutTensor) {
  TensorFactory<ScalarType::Float> tf;
  Tensor input = tf.make(/*sizes=*/{2, 4}, /*data=*/{0, 1, 1, 1, 0, 1, 0, 1});
  Tensor output = tf.zeros(
      /*sizes=*/{2, 2, 2}, torch::executor::TensorShapeDynamism::DYNAMIC_BOUND);

  std::vector<int32_t> size = {2, 2, 2};
  Tensor ref_output = tf.make(size, /*data=*/{0, 1, 1, 1, 0, 1, 0, 1});
  std::vector<int64_t> size_int64_t(size.size());
  std::transform(size.begin(), size.end(), size_int64_t.begin(), [](int32_t x) {
    return (int64_t)x;
  });

  op_view_copy_out(
      input,
      exec_aten::ArrayRef<int64_t>(size_int64_t.data(), size_int64_t.size()),
      output);
  EXPECT_TENSOR_EQ(ref_output, output);

  output = tf.zeros(
      /*sizes=*/{1, 4, 2}, torch::executor::TensorShapeDynamism::DYNAMIC_BOUND);
  size = std::vector<int32_t>({1, 4, 2});
  ref_output = tf.make(size, /*data=*/{0, 1, 1, 1, 0, 1, 0, 1});
  size_int64_t = std::vector<int64_t>(size.size());
  std::transform(size.begin(), size.end(), size_int64_t.begin(), [](int32_t x) {
    return (int64_t)x;
  });
  size_int64_t[1] = -1;

  op_view_copy_out(
      input,
      exec_aten::ArrayRef<int64_t>(size_int64_t.data(), size_int64_t.size()),
      output);
  EXPECT_TENSOR_EQ(ref_output, output);
}
#endif

TEST_F(OpViewTest, DynamicShapeUpperBoundSameAsExpected) {
  test_dynamic_shape(
      {2, 6}, torch::executor::TensorShapeDynamism::DYNAMIC_BOUND);
}

TEST_F(OpViewTest, DynamicShapeUpperBoundLargerThanExpected) {
  if (!torch::executor::testing::SupportedFeatures::get()->output_resize) {
    GTEST_SKIP() << "Dynamic shape not supported";
  }
  test_dynamic_shape(
      {10, 10}, torch::executor::TensorShapeDynamism::DYNAMIC_BOUND);
}

TEST_F(OpViewTest, DynamicShapeUnbound) {
  if (!torch::executor::testing::SupportedFeatures::get()->output_resize) {
    GTEST_SKIP() << "Dynamic shape not supported";
  }
  test_dynamic_shape(
      {1, 1}, torch::executor::TensorShapeDynamism::DYNAMIC_UNBOUND);
}
