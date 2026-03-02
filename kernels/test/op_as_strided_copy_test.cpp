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

#include <executorch/kernels/portable/cpu/util/copy_ops_util.h>
#include <executorch/test/utils/DeathTest.h>

#include <gtest/gtest.h>
#include <cmath>
#include <cstdint>

using namespace ::testing;
using executorch::aten::ArrayRef;
using executorch::aten::Scalar;
using executorch::aten::ScalarType;
using executorch::aten::Tensor;
using std::optional;
using torch::executor::testing::TensorFactory;

class OpAsStridedCopyOutTest : public OperatorTest {
 protected:
  Tensor& op_as_strided_copy_out(
      const Tensor& self,
      ArrayRef<int64_t> size,
      ArrayRef<int64_t> stride,
      optional<int64_t> storage_offset,
      Tensor& out) {
    return torch::executor::aten::as_strided_copy_outf(
        context_, self, size, stride, storage_offset, out);
  }

  // Common testing for eq operator
  template <ScalarType DTYPE>
  void test_detach_copy_out() {
    TensorFactory<DTYPE> tf;
    const std::vector<int32_t> in_sizes = {3, 3};
    const std::vector<int32_t> out_sizes = {2, 2, 2};

    Tensor in = tf.make(in_sizes, {1, 2, 3, 4, 5, 6, 7, 8, 9});
    Tensor out = tf.zeros(out_sizes);

    // Valid input should give the expected output
    optional<int64_t> storage_offset;
    int64_t sizes[3] = {2, 2, 2};
    int64_t stride[3] = {1, 2, 3};
    op_as_strided_copy_out(
        /*self=*/in,
        /*size=*/ArrayRef<int64_t>{sizes, 3},
        /*stride=*/ArrayRef<int64_t>{stride, 3},
        storage_offset,
        out);
    EXPECT_TENSOR_EQ(out, tf.make(out_sizes, {1, 4, 3, 6, 2, 5, 4, 7}));

    // With storage offset
    op_as_strided_copy_out(
        /*self=*/in,
        /*size=*/ArrayRef<int64_t>{sizes, 3},
        /*stride=*/ArrayRef<int64_t>{stride, 3},
        /*storage_offset=*/2,
        out);
    EXPECT_TENSOR_EQ(out, tf.make(out_sizes, {3, 6, 5, 8, 4, 7, 6, 9}));
  }

  template <ScalarType DTYPE>
  void test_as_strided_copy_out_invalid_parameters() {
    TensorFactory<DTYPE> tf;

    const std::vector<int32_t> in_sizes = {3, 3};
    const std::vector<int32_t> out_sizes = {2, 2, 2};

    Tensor in = tf.ones(in_sizes);
    Tensor out = tf.zeros(out_sizes);
    optional<int64_t> storage_offset;
    int64_t sizes[3] = {2, 2, 2};
    int64_t stride[3] = {1, 2, 3};

    // Mismatch strides and shape should die
    int64_t stride_short[2] = {1, 2};
    ET_EXPECT_KERNEL_FAILURE(
        context_,
        op_as_strided_copy_out(
            /*self=*/in,
            /*size=*/ArrayRef<int64_t>{sizes, 3},
            /*stride=*/ArrayRef<int64_t>{stride_short, 2},
            storage_offset,
            out));

    // Negative strides should die
    int64_t stride_negative[3] = {1, 2, -1};
    ET_EXPECT_KERNEL_FAILURE(
        context_,
        op_as_strided_copy_out(
            /*self=*/in,
            /*size=*/ArrayRef<int64_t>{sizes, 3},
            /*stride=*/ArrayRef<int64_t>{stride_negative, 3},
            storage_offset,
            out));

    // Mismatch output tensor shape and size should die
    int64_t size_invalid[3] = {2, 2, 1};
    ET_EXPECT_KERNEL_FAILURE(
        context_,
        op_as_strided_copy_out(
            /*self=*/in,
            /*size=*/ArrayRef<int64_t>{size_invalid, 3},
            /*stride=*/ArrayRef<int64_t>{stride, 3},
            storage_offset,
            out));

    // Invalid storage offset should die
    storage_offset = -1;
    ET_EXPECT_KERNEL_FAILURE(
        context_,
        op_as_strided_copy_out(
            /*self=*/in,
            /*size=*/ArrayRef<int64_t>{sizes, 3},
            /*stride=*/ArrayRef<int64_t>{stride, 3},
            storage_offset,
            out));

    // Out of bound storage access of `in` should die
    storage_offset = 3;
    ET_EXPECT_KERNEL_FAILURE(
        context_,
        op_as_strided_copy_out(
            /*self=*/in,
            /*size=*/ArrayRef<int64_t>{sizes, 3},
            /*stride=*/ArrayRef<int64_t>{stride, 3},
            storage_offset,
            out));
  }
};

template <>
void OpAsStridedCopyOutTest::test_detach_copy_out<ScalarType::Bool>() {
  TensorFactory<ScalarType::Bool> tf;
  const std::vector<int32_t> in_sizes = {3, 3};
  const std::vector<int32_t> out_sizes = {2, 2, 2};
  Tensor in = tf.make(
      in_sizes, {false, true, false, true, false, true, false, true, false});
  Tensor out = tf.zeros(out_sizes);

  // Valid input should give the expected output
  optional<int64_t> storage_offset = 2;
  int64_t sizes[3] = {2, 2, 2};
  int64_t stride[3] = {1, 2, 3};
  op_as_strided_copy_out(
      /*self=*/in,
      /*size=*/ArrayRef<int64_t>{sizes, 3},
      /*stride=*/ArrayRef<int64_t>{stride, 3},
      storage_offset,
      out);
  EXPECT_TENSOR_EQ(
      out,
      tf.make(out_sizes, {false, true, false, true, true, false, true, false}));
}

template <>
void OpAsStridedCopyOutTest::test_detach_copy_out<ScalarType::Float>() {
  TensorFactory<ScalarType::Float> tf;
  const std::vector<int32_t> in_sizes = {3, 3};
  const std::vector<int32_t> out_sizes = {2, 2, 2};

  Tensor in = tf.make(
      in_sizes, {3.14, 2.33, 42, INFINITY, -INFINITY, NAN, -3.14, -2.33, -42});
  Tensor out = tf.zeros(out_sizes);

  // Valid input should give the expected output
  optional<int64_t> storage_offset = 2;
  int64_t sizes[3] = {2, 2, 2};
  int64_t stride[3] = {1, 2, 3};
  op_as_strided_copy_out(
      /*self=*/in,
      /*size=*/ArrayRef<int64_t>{sizes, 3},
      /*stride=*/ArrayRef<int64_t>{stride, 3},
      storage_offset,
      out);
  EXPECT_TENSOR_CLOSE(
      out,
      tf.make(
          out_sizes,
          {42.0, NAN, -INFINITY, 2.33, INFINITY, -3.14, NAN, -42.0}));
}

TEST_F(OpAsStridedCopyOutTest, AllScalarInputOutputSupport) {
#define TEST_ENTRY(ctype, dtype) test_detach_copy_out<ScalarType::dtype>();
  ET_FORALL_INT_TYPES(TEST_ENTRY);
#undef TEST_ENTRY
}

TEST_F(OpAsStridedCopyOutTest, InvalidParametersDies) {
  if (torch::executor::testing::SupportedFeatures::get()->is_aten) {
    GTEST_SKIP() << "ATen kernel can handle invalid parameter";
  }
#define TEST_ENTRY(ctype, dtype) \
  test_as_strided_copy_out_invalid_parameters<ScalarType::dtype>();
  ET_FORALL_REAL_TYPES_AND(Bool, TEST_ENTRY);
#undef TEST_ENTRY
}

TEST_F(OpAsStridedCopyOutTest, StrideDrivenOutOfBoundsDies) {
  // This test calls as_strided_copy<T>(), bypassing check_as_strided_copy_args
  // which has its own validation. This is important because internal callers
  // (like diagonal_copy) may call as_strided_copy directly.
  TensorFactory<ScalarType::Int> tf;

  // Create a 3x3 input tensor (9 elements total)
  const std::vector<int32_t> in_sizes = {3, 3};
  Tensor in = tf.ones(in_sizes);

  // Create output tensor with shape {2, 2}
  const std::vector<int32_t> out_sizes = {2, 2};
  Tensor out = tf.zeros(out_sizes);

  // Case 1: offset is within range (0), but stride causes OOB
  // With size={2, 2}, stride={1, 100}, the maximum index accessed is:
  // offset + (2-1)*1 + (2-1)*100 = 0 + 1 + 100 = 101, which exceeds numel=9
  int64_t sizes[2] = {2, 2};
  int64_t stride_oob[2] = {1, 100};

  // Call as_strided_copy directly, bypassing check_as_strided_copy_args
  // Use brace-initialization to construct ArrayRef compatible with
  // torch::executor
  ET_EXPECT_DEATH(
      torch::executor::as_strided_copy<int32_t>(
          in,
          {sizes, 2},
          {stride_oob, 2},
          /*offset=*/0,
          out),
      "");

  // Case 2: Valid offset but combined with strides exceeds bounds
  // offset=5, size={2, 2}, stride={1, 3}
  // max_index = 5 + (2-1)*1 + (2-1)*3 = 5 + 1 + 3 = 9
  // This equals numel (9), but the check requires max_index < numel
  int64_t stride_boundary[2] = {1, 3};
  ET_EXPECT_DEATH(
      torch::executor::as_strided_copy<int32_t>(
          in,
          {sizes, 2},
          {stride_boundary, 2},
          /*offset=*/5,
          out),
      "");
}

TEST_F(OpAsStridedCopyOutTest, StrideDrivenOutOfBoundsPublicApiDies) {
  // This test verifies that stride-driven out-of-bounds access is rejected
  // via the public API (as_strided_copy_out), which validates through
  // check_as_strided_copy_args before calling as_strided_copy.
  TensorFactory<ScalarType::Int> tf;

  // Create a 3x3 input tensor (9 elements total)
  const std::vector<int32_t> in_sizes = {3, 3};
  Tensor in = tf.ones(in_sizes);

  // Create output tensor with shape {2, 2}
  const std::vector<int32_t> out_sizes = {2, 2};
  Tensor out = tf.zeros(out_sizes);

  // Case 1: offset is within range (0), but stride causes OOB
  // With size={2, 2}, stride={1, 100}, the maximum index accessed is:
  // offset + (2-1)*1 + (2-1)*100 = 0 + 1 + 100 = 101, which exceeds numel=9
  int64_t sizes[2] = {2, 2};
  int64_t stride_oob[2] = {1, 100};
  optional<int64_t> storage_offset = 0;
  ET_EXPECT_KERNEL_FAILURE(
      context_,
      op_as_strided_copy_out(
          /*self=*/in,
          /*size=*/ArrayRef<int64_t>{sizes, 2},
          /*stride=*/ArrayRef<int64_t>{stride_oob, 2},
          storage_offset,
          out));

  // Case 2: Valid offset but combined with strides exceeds bounds
  // offset=5, size={2, 2}, stride={1, 3}
  // max_index = 5 + (2-1)*1 + (2-1)*3 = 5 + 1 + 3 = 9
  // This equals numel (9), but the check requires max_index < numel
  int64_t stride_boundary[2] = {1, 3};
  storage_offset = 5;
  ET_EXPECT_KERNEL_FAILURE(
      context_,
      op_as_strided_copy_out(
          /*self=*/in,
          /*size=*/ArrayRef<int64_t>{sizes, 2},
          /*stride=*/ArrayRef<int64_t>{stride_boundary, 2},
          storage_offset,
          out));
}

TEST_F(OpAsStridedCopyOutTest, MismatchedInputDtypesDies) {
  TensorFactory<ScalarType::Byte> tf_byte;
  TensorFactory<ScalarType::Char> tf_char;
  const std::vector<int32_t> in_sizes = {3, 3};
  const std::vector<int32_t> out_sizes = {2, 2, 2};

  Tensor in = tf_byte.make(in_sizes, {1, 2, 3, 4, 5, 6, 7, 8, 9});
  Tensor out = tf_char.zeros(out_sizes);
  optional<int64_t> storage_offset;
  int64_t sizes[3] = {2, 2, 2};
  int64_t stride[3] = {1, 2, 3};

  ET_EXPECT_KERNEL_FAILURE(
      context_,
      op_as_strided_copy_out(
          /*self=*/in,
          /*size=*/ArrayRef<int64_t>{sizes, 3},
          /*stride=*/ArrayRef<int64_t>{stride, 3},
          storage_offset,
          out));
}

/* %python
import torch
torch.manual_seed(0)
x = torch.rand(3, 3)
res = torch.as_strided(x, (2, 2, 2), (1, 2, 3))
op = "op_as_strided_copy_out"
opt_setup_params = f"""
  {declare_array_ref([2, 2, 2], "int64_t", "size")}
  {declare_array_ref([1, 2, 3], "int64_t", "stride")}
  optional<int64_t> storage_offset;
"""
opt_extra_params = "size, stride, storage_offset,"
dtype = "ScalarType::Float"
check = "EXPECT_TENSOR_EQ" */

TEST_F(OpAsStridedCopyOutTest, DynamicShapeUpperBoundSameAsExpected) {
  /* %python
  out_args = "{2, 2, 2}, torch::executor::TensorShapeDynamism::DYNAMIC_BOUND"
  %rewrite(unary_op) */

  TensorFactory<ScalarType::Float> tf;

  Tensor x = tf.make(
      {3, 3},
      {0.49625658988952637,
       0.7682217955589294,
       0.08847743272781372,
       0.13203048706054688,
       0.30742281675338745,
       0.6340786814689636,
       0.4900934100151062,
       0.8964447379112244,
       0.455627977848053});
  Tensor expected = tf.make(
      {2, 2, 2},
      {0.49625658988952637,
       0.13203048706054688,
       0.08847743272781372,
       0.6340786814689636,
       0.7682217955589294,
       0.30742281675338745,
       0.13203048706054688,
       0.4900934100151062});

  std::vector<int64_t> sizev = {2, 2, 2};
  ArrayRef<int64_t> size(sizev.data(), sizev.size());
  std::vector<int64_t> stridev = {1, 2, 3};
  ArrayRef<int64_t> stride(stridev.data(), stridev.size());
  optional<int64_t> storage_offset;

  Tensor out =
      tf.zeros({2, 2, 2}, torch::executor::TensorShapeDynamism::DYNAMIC_BOUND);
  op_as_strided_copy_out(x, size, stride, storage_offset, out);
  EXPECT_TENSOR_EQ(out, expected);
}

TEST_F(OpAsStridedCopyOutTest, DynamicShapeUpperBoundLargerThanExpected) {
  /* %python
  out_args = "{5, 5, 5}, torch::executor::TensorShapeDynamism::DYNAMIC_BOUND"
  %rewrite(unary_op) */

  TensorFactory<ScalarType::Float> tf;

  Tensor x = tf.make(
      {3, 3},
      {0.49625658988952637,
       0.7682217955589294,
       0.08847743272781372,
       0.13203048706054688,
       0.30742281675338745,
       0.6340786814689636,
       0.4900934100151062,
       0.8964447379112244,
       0.455627977848053});
  Tensor expected = tf.make(
      {2, 2, 2},
      {0.49625658988952637,
       0.13203048706054688,
       0.08847743272781372,
       0.6340786814689636,
       0.7682217955589294,
       0.30742281675338745,
       0.13203048706054688,
       0.4900934100151062});

  std::vector<int64_t> sizev = {2, 2, 2};
  ArrayRef<int64_t> size(sizev.data(), sizev.size());
  std::vector<int64_t> stridev = {1, 2, 3};
  ArrayRef<int64_t> stride(stridev.data(), stridev.size());
  optional<int64_t> storage_offset;

  Tensor out =
      tf.zeros({5, 5, 5}, torch::executor::TensorShapeDynamism::DYNAMIC_BOUND);
  op_as_strided_copy_out(x, size, stride, storage_offset, out);
  EXPECT_TENSOR_EQ(out, expected);
}

TEST_F(OpAsStridedCopyOutTest, DynamicShapeUnbound) {
  if (!torch::executor::testing::SupportedFeatures::get()->output_resize) {
    GTEST_SKIP() << "Dynamic shape unbound not supported";
  }
  /* %python
  out_args = "{1, 1, 1}, torch::executor::TensorShapeDynamism::DYNAMIC_UNBOUND"
  %rewrite(unary_op) */

  TensorFactory<ScalarType::Float> tf;

  Tensor x = tf.make(
      {3, 3},
      {0.49625658988952637,
       0.7682217955589294,
       0.08847743272781372,
       0.13203048706054688,
       0.30742281675338745,
       0.6340786814689636,
       0.4900934100151062,
       0.8964447379112244,
       0.455627977848053});
  Tensor expected = tf.make(
      {2, 2, 2},
      {0.49625658988952637,
       0.13203048706054688,
       0.08847743272781372,
       0.6340786814689636,
       0.7682217955589294,
       0.30742281675338745,
       0.13203048706054688,
       0.4900934100151062});

  std::vector<int64_t> sizev = {2, 2, 2};
  ArrayRef<int64_t> size(sizev.data(), sizev.size());
  std::vector<int64_t> stridev = {1, 2, 3};
  ArrayRef<int64_t> stride(stridev.data(), stridev.size());
  optional<int64_t> storage_offset;

  Tensor out = tf.zeros(
      {1, 1, 1}, torch::executor::TensorShapeDynamism::DYNAMIC_UNBOUND);
  op_as_strided_copy_out(x, size, stride, storage_offset, out);
  EXPECT_TENSOR_EQ(out, expected);
}
