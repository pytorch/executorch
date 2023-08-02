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

#include <gtest/gtest.h>

using namespace ::testing;
using exec_aten::ArrayRef;
using exec_aten::IntArrayRef;
using exec_aten::ScalarType;
using exec_aten::Tensor;
using torch::executor::testing::TensorFactory;

Tensor& op_permute_copy_out(const Tensor& self, IntArrayRef dims, Tensor& out) {
  exec_aten::RuntimeContext context{};
  return torch::executor::aten::permute_copy_outf(context, self, dims, out);
}

TEST(OpPermuteCopyKernelTest, OneDPermute) {
  TensorFactory<ScalarType::Int> tf;

  const std::vector<int64_t> new_dim = {0};

  const std::vector<int32_t> sizes = {2};
  Tensor t_int = tf.make(sizes, {1, 2});

  Tensor out = tf.zeros(sizes);

  op_permute_copy_out(
      t_int, ArrayRef<int64_t>(new_dim.data(), new_dim.size()), out);
  EXPECT_TENSOR_EQ(out, tf.make(sizes, {1, 2}));
}

TEST(OpPermuteCopyKernelTest, PermuteWithNoDataReorder) {
  TensorFactory<ScalarType::Int> tf;

  const std::vector<int64_t> new_dim = {1, 0, 2};

  // clang-format off
  Tensor t_int = tf.make({1,4,5}, {
    0,  1,  2,  3,  4,
    5,  6,  7,  8,  9,
    10, 11, 12, 13, 14,
    15, 16, 17,18, 19});
  // clang-format on

  const std::vector<int32_t> new_sizes = {4, 1, 5};
  Tensor out = tf.zeros(new_sizes);

  op_permute_copy_out(
      t_int, ArrayRef<int64_t>(new_dim.data(), new_dim.size()), out);
  // clang-format off
  EXPECT_TENSOR_EQ(out, tf.make(new_sizes, {
    0,  1,  2,  3,  4,
    5,  6,  7,  8,  9,
    10, 11, 12, 13, 14,
    15, 16, 17, 18, 19}));
  // clang-format on
}

TEST(OpPermuteCopyKernelTest, TwoDPermute) {
  TensorFactory<ScalarType::Int> tf;

  const std::vector<int64_t> new_dim = {1, 0};

  // clang-format off
  Tensor t_int = tf.make({2, 3}, {
    // 2x3 data block
    0, 1, 2,
    3, 4, 5
  });
  // clang-format on

  const std::vector<int32_t> new_sizes = {3, 2};
  Tensor out = tf.zeros(new_sizes);

  op_permute_copy_out(
      t_int, ArrayRef<int64_t>(new_dim.data(), new_dim.size()), out);
  // clang-format off
  EXPECT_TENSOR_EQ(out, tf.make(new_sizes, {
    // 3x2 data block
    0, 3,
    1, 4,
    2, 5
  }));
  // clang-format on
}

TEST(OpPermuteCopyKernelTest, ThreeDPermute) {
  TensorFactory<ScalarType::Int> tf;

  const std::vector<int64_t> new_dim = {2, 0, 1};

  // clang-format off
  Tensor t_int = tf.make({2, 1, 3}, {
    // 2 1x3 data blocks
    0, 1, 2,

    3, 4, 5
  });
  // clang-format on

  const std::vector<int32_t> new_sizes = {3, 2, 1};
  Tensor out = tf.zeros(new_sizes);

  op_permute_copy_out(
      t_int, ArrayRef<int64_t>(new_dim.data(), new_dim.size()), out);
  // clang-format off
  EXPECT_TENSOR_EQ(out, tf.make(new_sizes, {
  // 3 2x1 data blocks
    0,
    3,

    1,
    4,

    2,
    5
  }));
  // clang-format on
}

TEST(OpPermuteCopyKernelTest, FourDPermute) {
  TensorFactory<ScalarType::Int> tf;

  const std::vector<int64_t> new_dim = {0, 3, 2, 1};

  // clang-format off
  Tensor t_int = tf.make(
      {2, 3, 3, 4},
      // 2 groupings of 3 3x4 data blocks
      {0,  1,  2,  3,
       4,  5,  6,  7,
       8,  9,  10, 11,

       12, 13, 14, 15,
       16, 17, 18, 19,
       20, 21, 22, 23,

       24, 25, 26, 27,
       28, 29, 30, 31,
       32, 33, 34, 35,


       36, 37, 38, 39,
       40, 41, 42, 43,
       44, 45, 46, 47,

       48, 49, 50, 51,
       52, 53, 54, 55,
       56, 57, 58, 59,

       60, 61, 62, 63,
       64, 65, 66, 67,
       68, 69, 70, 71});
  // clang-format on

  const std::vector<int32_t> new_sizes = {2, 4, 3, 3};
  Tensor out = tf.zeros(new_sizes);

  // Long results like this are gotten by running torch.permute in a notebook
  // and copy pasting the result. Ex:
  //   import torch
  //   x = torch.arange(0, 72, 1).view(2, 3, 3, 4).contiguous()
  //   print(x.flatten().contiguous())
  //   z = torch.permute(x, (0, 3, 2, 1))
  //   print(z.flatten().contiguous())
  op_permute_copy_out(
      t_int, ArrayRef<int64_t>(new_dim.data(), new_dim.size()), out);
  EXPECT_TENSOR_EQ(
      out,
      // clang-format off
      tf.make(new_sizes, {
        // 2 groupings of 4 3x3 data blocks
        0, 12, 24,
        4, 16, 28,
        8, 20, 32,

        1, 13, 25,
        5, 17, 29,
        9, 21, 33,

        2, 14, 26,
        6, 18, 30,
        10, 22, 34,

        3, 15, 27,
        7, 19, 31,
        11, 23, 35,


        36, 48, 60,
        40, 52, 64,
        44, 56, 68,

        37, 49, 61,
        41, 53, 65,
        45, 57, 69,

        38, 50, 62,
        42, 54, 66,
        46, 58, 70,

        39, 51, 63,
        43, 55, 67,
        47, 59, 71}));
  // clang-format on
}

TEST(OpPermuteCopyKernelTest, FiveDPermute) {
  TensorFactory<ScalarType::Int> tf;

  const std::vector<int64_t> new_dim = {4, 3, 2, 1, 0};

  const std::vector<int32_t> sizes = {2, 2, 2, 2, 2};
  // clang-format off
  Tensor t_int = tf.make(
      sizes, {
        0,  1,
        2,  3,

        4,  5,
        6,  7,


        8,  9,
        10, 11,

        12, 13,
        14, 15,


        16, 17,
        18, 19,

        20, 21,
        22, 23,


        24, 25,
        26, 27,

        28, 29,
        30, 31});
  // clang-format on

  Tensor out = tf.zeros(sizes);

  op_permute_copy_out(
      t_int, ArrayRef<int64_t>(new_dim.data(), new_dim.size()), out);
  // Long results like this are gotten by running torch.permute in a notebook
  // and copy pasting the result. Ex:
  //   import torch
  //   x = torch.arange(0, 32, 1).view(2, 2, 2, 2, 2).contiguous()
  //   print(x.flatten().contiguous())
  //   z = torch.permute(x, (4, 3, 2, 1, 0))
  //   print(z.flatten().contiguous())
  // clang-format off
  EXPECT_TENSOR_EQ(
      out, tf.make(sizes, {
        0,  16,
        8,  24,

        4,  20,
        12, 28,


        2,  18,
        10, 26,

        6,  22,
        14, 30,


        1,  17,
        9,  25,

        5,  21,
        13, 29,


        3,  19,
        11, 27,

        7,  23,
        15, 31}));
  // clang-format on
}

TEST(OpPermuteCopyKernelTest, AllDimensionsSizeOne) {
  TensorFactory<ScalarType::Int> tf;

  const std::vector<int64_t> new_dim = {4, 3, 2, 1, 0};

  const std::vector<int32_t> sizes = {1, 1, 1, 1, 1};
  Tensor t_int = tf.make(sizes, {1});

  Tensor out = tf.zeros(sizes);

  op_permute_copy_out(
      t_int, ArrayRef<int64_t>(new_dim.data(), new_dim.size()), out);
  EXPECT_TENSOR_EQ(out, tf.make(sizes, {1}));
}

TEST(OpPermuteCopyKernelTest, DupeDimensionPos) {
  TensorFactory<ScalarType::Int> tf;

  const std::vector<int64_t> new_dim = {0, 1, 1};

  const std::vector<int32_t> sizes = {1, 1, 1};
  Tensor t_int = tf.make(sizes, {1});

  Tensor out = tf.zeros(sizes);

  ET_EXPECT_KERNEL_FAILURE(op_permute_copy_out(
      t_int, ArrayRef<int64_t>(new_dim.data(), new_dim.size()), out));
}

TEST(OpPermuteCopyKernelTest, DupeDimensionNeg) {
  TensorFactory<ScalarType::Int> tf;

  const std::vector<int64_t> new_dim = {0, 1, -2};

  const std::vector<int32_t> sizes = {1, 1, 1};
  Tensor t_int = tf.make(sizes, {1});

  Tensor out = tf.zeros(sizes);

  ET_EXPECT_KERNEL_FAILURE(op_permute_copy_out(
      t_int, ArrayRef<int64_t>(new_dim.data(), new_dim.size()), out));
}

TEST(OpPermuteCopyKernelTest, MismatchDim) {
  TensorFactory<ScalarType::Int> tf;

  const std::vector<int64_t> new_dim = {0, 1, 2};

  const std::vector<int32_t> sizes = {1, 1};
  Tensor t_int = tf.make(sizes, {1});

  Tensor out = tf.zeros(sizes);

  ET_EXPECT_KERNEL_FAILURE(op_permute_copy_out(
      t_int, ArrayRef<int64_t>(new_dim.data(), new_dim.size()), out));
}

/* %python
import torch
torch.manual_seed(0)
x = torch.randint(10, (2, 3, 4))
res = torch.permute(x, (2, 0, 1))
op = "op_permute_copy_out"
opt_setup_params = f"""
  {declare_array_ref([2, 0, 1], "int64_t", "perm_aref")}
"""
opt_extra_params = "perm_aref,"
dtype = "ScalarType::Int"
check = "EXPECT_TENSOR_EQ" */

TEST(OpPermuteCopyKernelTest, DynamicShapeUpperBoundSameAsExpected) {
  /* %python
  out_args = "{4, 2, 3}, torch::executor::TensorShapeDynamism::DYNAMIC_BOUND"
  %rewrite(unary_op) */

  TensorFactory<ScalarType::Int> tf;

  Tensor x = tf.make({2, 3, 4}, {4, 9, 3, 0, 3, 9, 7, 3, 7, 3, 1, 6,
                                 6, 9, 8, 6, 6, 8, 4, 3, 6, 9, 1, 4});
  Tensor expected = tf.make({4, 2, 3}, {4, 3, 7, 6, 6, 6, 9, 9, 3, 9, 8, 9,
                                        3, 7, 1, 8, 4, 1, 0, 3, 6, 6, 3, 4});

  std::vector<int64_t> perm_arefv = {2, 0, 1};
  ArrayRef<int64_t> perm_aref(perm_arefv.data(), perm_arefv.size());

  Tensor out =
      tf.zeros({4, 2, 3}, torch::executor::TensorShapeDynamism::DYNAMIC_BOUND);
  op_permute_copy_out(x, perm_aref, out);
  EXPECT_TENSOR_EQ(out, expected);
}

TEST(OpPermuteCopyKernelTest, DynamicShapeUpperBoundLargerThanExpected) {
  /* %python
  out_args = "{5, 5, 5}, torch::executor::TensorShapeDynamism::DYNAMIC_BOUND"
  %rewrite(unary_op) */

  TensorFactory<ScalarType::Int> tf;

  Tensor x = tf.make({2, 3, 4}, {4, 9, 3, 0, 3, 9, 7, 3, 7, 3, 1, 6,
                                 6, 9, 8, 6, 6, 8, 4, 3, 6, 9, 1, 4});
  Tensor expected = tf.make({4, 2, 3}, {4, 3, 7, 6, 6, 6, 9, 9, 3, 9, 8, 9,
                                        3, 7, 1, 8, 4, 1, 0, 3, 6, 6, 3, 4});

  std::vector<int64_t> perm_arefv = {2, 0, 1};
  ArrayRef<int64_t> perm_aref(perm_arefv.data(), perm_arefv.size());

  Tensor out =
      tf.zeros({5, 5, 5}, torch::executor::TensorShapeDynamism::DYNAMIC_BOUND);
  op_permute_copy_out(x, perm_aref, out);
  EXPECT_TENSOR_EQ(out, expected);
}

TEST(OpPermuteCopyKernelTest, DynamicShapeUnbound) {
  if (!torch::executor::testing::SupportedFeatures::get()->output_resize) {
    GTEST_SKIP() << "Dynamic shape unbound not supported";
  }
  /* %python
  out_args = "{1, 1, 1}, torch::executor::TensorShapeDynamism::DYNAMIC_UNBOUND"
  %rewrite(unary_op) */

  TensorFactory<ScalarType::Int> tf;

  Tensor x = tf.make({2, 3, 4}, {4, 9, 3, 0, 3, 9, 7, 3, 7, 3, 1, 6,
                                 6, 9, 8, 6, 6, 8, 4, 3, 6, 9, 1, 4});
  Tensor expected = tf.make({4, 2, 3}, {4, 3, 7, 6, 6, 6, 9, 9, 3, 9, 8, 9,
                                        3, 7, 1, 8, 4, 1, 0, 3, 6, 6, 3, 4});

  std::vector<int64_t> perm_arefv = {2, 0, 1};
  ArrayRef<int64_t> perm_aref(perm_arefv.data(), perm_arefv.size());

  Tensor out = tf.zeros(
      {1, 1, 1}, torch::executor::TensorShapeDynamism::DYNAMIC_UNBOUND);
  op_permute_copy_out(x, perm_aref, out);
  EXPECT_TENSOR_EQ(out, expected);
}

TEST(OpPermuteCopyOutTest, DtypeTest_float32_float32) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Float> tfFloat;

  exec_aten::Tensor self = tfFloat.make({2, 2}, {1.3125, 2.625, 3.5, 4.875});
  ::std::vector<int64_t> dims_vec = {1, 0};
  exec_aten::ArrayRef<int64_t> dims =
      exec_aten::ArrayRef<int64_t>(dims_vec.data(), dims_vec.size());
  exec_aten::Tensor out = tfFloat.zeros({2, 2});
  exec_aten::Tensor out_expected =
      tfFloat.make({2, 2}, {1.3125, 3.5, 2.625, 4.875});
  op_permute_copy_out(self, dims, out);
  EXPECT_TENSOR_CLOSE(out, out_expected);
}

TEST(OpPermuteCopyOutTest, DtypeTest_float32_float64) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Double>
      tfDouble;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Float> tfFloat;

  exec_aten::Tensor self = tfFloat.make({2, 2}, {1.3125, 2.625, 3.5, 4.875});
  ::std::vector<int64_t> dims_vec = {1, 0};
  exec_aten::ArrayRef<int64_t> dims =
      exec_aten::ArrayRef<int64_t>(dims_vec.data(), dims_vec.size());
  exec_aten::Tensor out = tfDouble.zeros({2, 2});
  ET_EXPECT_KERNEL_FAILURE(op_permute_copy_out(self, dims, out));
}

TEST(OpPermuteCopyOutTest, DtypeTest_float32_uint8) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Byte> tfByte;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Float> tfFloat;

  exec_aten::Tensor self = tfFloat.make({2, 2}, {1.3125, 2.625, 3.5, 4.875});
  ::std::vector<int64_t> dims_vec = {1, 0};
  exec_aten::ArrayRef<int64_t> dims =
      exec_aten::ArrayRef<int64_t>(dims_vec.data(), dims_vec.size());
  exec_aten::Tensor out = tfByte.zeros({2, 2});
  ET_EXPECT_KERNEL_FAILURE(op_permute_copy_out(self, dims, out));
}

TEST(OpPermuteCopyOutTest, DtypeTest_float32_int8) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Char> tfChar;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Float> tfFloat;

  exec_aten::Tensor self = tfFloat.make({2, 2}, {1.3125, 2.625, 3.5, 4.875});
  ::std::vector<int64_t> dims_vec = {1, 0};
  exec_aten::ArrayRef<int64_t> dims =
      exec_aten::ArrayRef<int64_t>(dims_vec.data(), dims_vec.size());
  exec_aten::Tensor out = tfChar.zeros({2, 2});
  ET_EXPECT_KERNEL_FAILURE(op_permute_copy_out(self, dims, out));
}

TEST(OpPermuteCopyOutTest, DtypeTest_float32_int16) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Float> tfFloat;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Short> tfShort;

  exec_aten::Tensor self = tfFloat.make({2, 2}, {1.3125, 2.625, 3.5, 4.875});
  ::std::vector<int64_t> dims_vec = {1, 0};
  exec_aten::ArrayRef<int64_t> dims =
      exec_aten::ArrayRef<int64_t>(dims_vec.data(), dims_vec.size());
  exec_aten::Tensor out = tfShort.zeros({2, 2});
  ET_EXPECT_KERNEL_FAILURE(op_permute_copy_out(self, dims, out));
}

TEST(OpPermuteCopyOutTest, DtypeTest_float32_int32) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Float> tfFloat;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Int> tfInt;

  exec_aten::Tensor self = tfFloat.make({2, 2}, {1.3125, 2.625, 3.5, 4.875});
  ::std::vector<int64_t> dims_vec = {1, 0};
  exec_aten::ArrayRef<int64_t> dims =
      exec_aten::ArrayRef<int64_t>(dims_vec.data(), dims_vec.size());
  exec_aten::Tensor out = tfInt.zeros({2, 2});
  ET_EXPECT_KERNEL_FAILURE(op_permute_copy_out(self, dims, out));
}

TEST(OpPermuteCopyOutTest, DtypeTest_float32_int64) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Float> tfFloat;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Long> tfLong;

  exec_aten::Tensor self = tfFloat.make({2, 2}, {1.3125, 2.625, 3.5, 4.875});
  ::std::vector<int64_t> dims_vec = {1, 0};
  exec_aten::ArrayRef<int64_t> dims =
      exec_aten::ArrayRef<int64_t>(dims_vec.data(), dims_vec.size());
  exec_aten::Tensor out = tfLong.zeros({2, 2});
  ET_EXPECT_KERNEL_FAILURE(op_permute_copy_out(self, dims, out));
}

TEST(OpPermuteCopyOutTest, DtypeTest_float32_bool) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Bool> tfBool;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Float> tfFloat;

  exec_aten::Tensor self = tfFloat.make({2, 2}, {1.3125, 2.625, 3.5, 4.875});
  ::std::vector<int64_t> dims_vec = {1, 0};
  exec_aten::ArrayRef<int64_t> dims =
      exec_aten::ArrayRef<int64_t>(dims_vec.data(), dims_vec.size());
  exec_aten::Tensor out = tfBool.zeros({2, 2});
  ET_EXPECT_KERNEL_FAILURE(op_permute_copy_out(self, dims, out));
}

TEST(OpPermuteCopyOutTest, DtypeTest_float64_float32) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Double>
      tfDouble;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Float> tfFloat;

  exec_aten::Tensor self = tfDouble.make({2, 2}, {1.3125, 2.625, 3.5, 4.875});
  ::std::vector<int64_t> dims_vec = {1, 0};
  exec_aten::ArrayRef<int64_t> dims =
      exec_aten::ArrayRef<int64_t>(dims_vec.data(), dims_vec.size());
  exec_aten::Tensor out = tfFloat.zeros({2, 2});
  ET_EXPECT_KERNEL_FAILURE(op_permute_copy_out(self, dims, out));
}

TEST(OpPermuteCopyOutTest, DtypeTest_float64_float64) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Double>
      tfDouble;

  exec_aten::Tensor self = tfDouble.make({2, 2}, {1.3125, 2.625, 3.5, 4.875});
  ::std::vector<int64_t> dims_vec = {1, 0};
  exec_aten::ArrayRef<int64_t> dims =
      exec_aten::ArrayRef<int64_t>(dims_vec.data(), dims_vec.size());
  exec_aten::Tensor out = tfDouble.zeros({2, 2});
  exec_aten::Tensor out_expected =
      tfDouble.make({2, 2}, {1.3125, 3.5, 2.625, 4.875});
  op_permute_copy_out(self, dims, out);
  EXPECT_TENSOR_CLOSE(out, out_expected);
}

TEST(OpPermuteCopyOutTest, DtypeTest_float64_uint8) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Byte> tfByte;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Double>
      tfDouble;

  exec_aten::Tensor self = tfDouble.make({2, 2}, {1.3125, 2.625, 3.5, 4.875});
  ::std::vector<int64_t> dims_vec = {1, 0};
  exec_aten::ArrayRef<int64_t> dims =
      exec_aten::ArrayRef<int64_t>(dims_vec.data(), dims_vec.size());
  exec_aten::Tensor out = tfByte.zeros({2, 2});
  ET_EXPECT_KERNEL_FAILURE(op_permute_copy_out(self, dims, out));
}

TEST(OpPermuteCopyOutTest, DtypeTest_float64_int8) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Char> tfChar;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Double>
      tfDouble;

  exec_aten::Tensor self = tfDouble.make({2, 2}, {1.3125, 2.625, 3.5, 4.875});
  ::std::vector<int64_t> dims_vec = {1, 0};
  exec_aten::ArrayRef<int64_t> dims =
      exec_aten::ArrayRef<int64_t>(dims_vec.data(), dims_vec.size());
  exec_aten::Tensor out = tfChar.zeros({2, 2});
  ET_EXPECT_KERNEL_FAILURE(op_permute_copy_out(self, dims, out));
}

TEST(OpPermuteCopyOutTest, DtypeTest_float64_int16) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Double>
      tfDouble;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Short> tfShort;

  exec_aten::Tensor self = tfDouble.make({2, 2}, {1.3125, 2.625, 3.5, 4.875});
  ::std::vector<int64_t> dims_vec = {1, 0};
  exec_aten::ArrayRef<int64_t> dims =
      exec_aten::ArrayRef<int64_t>(dims_vec.data(), dims_vec.size());
  exec_aten::Tensor out = tfShort.zeros({2, 2});
  ET_EXPECT_KERNEL_FAILURE(op_permute_copy_out(self, dims, out));
}

TEST(OpPermuteCopyOutTest, DtypeTest_float64_int32) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Double>
      tfDouble;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Int> tfInt;

  exec_aten::Tensor self = tfDouble.make({2, 2}, {1.3125, 2.625, 3.5, 4.875});
  ::std::vector<int64_t> dims_vec = {1, 0};
  exec_aten::ArrayRef<int64_t> dims =
      exec_aten::ArrayRef<int64_t>(dims_vec.data(), dims_vec.size());
  exec_aten::Tensor out = tfInt.zeros({2, 2});
  ET_EXPECT_KERNEL_FAILURE(op_permute_copy_out(self, dims, out));
}

TEST(OpPermuteCopyOutTest, DtypeTest_float64_int64) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Double>
      tfDouble;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Long> tfLong;

  exec_aten::Tensor self = tfDouble.make({2, 2}, {1.3125, 2.625, 3.5, 4.875});
  ::std::vector<int64_t> dims_vec = {1, 0};
  exec_aten::ArrayRef<int64_t> dims =
      exec_aten::ArrayRef<int64_t>(dims_vec.data(), dims_vec.size());
  exec_aten::Tensor out = tfLong.zeros({2, 2});
  ET_EXPECT_KERNEL_FAILURE(op_permute_copy_out(self, dims, out));
}

TEST(OpPermuteCopyOutTest, DtypeTest_float64_bool) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Bool> tfBool;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Double>
      tfDouble;

  exec_aten::Tensor self = tfDouble.make({2, 2}, {1.3125, 2.625, 3.5, 4.875});
  ::std::vector<int64_t> dims_vec = {1, 0};
  exec_aten::ArrayRef<int64_t> dims =
      exec_aten::ArrayRef<int64_t>(dims_vec.data(), dims_vec.size());
  exec_aten::Tensor out = tfBool.zeros({2, 2});
  ET_EXPECT_KERNEL_FAILURE(op_permute_copy_out(self, dims, out));
}

TEST(OpPermuteCopyOutTest, DtypeTest_uint8_float32) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Byte> tfByte;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Float> tfFloat;

  exec_aten::Tensor self = tfByte.make({2, 2}, {1, 2, 3, 4});
  ::std::vector<int64_t> dims_vec = {1, 0};
  exec_aten::ArrayRef<int64_t> dims =
      exec_aten::ArrayRef<int64_t>(dims_vec.data(), dims_vec.size());
  exec_aten::Tensor out = tfFloat.zeros({2, 2});
  ET_EXPECT_KERNEL_FAILURE(op_permute_copy_out(self, dims, out));
}

TEST(OpPermuteCopyOutTest, DtypeTest_uint8_float64) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Byte> tfByte;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Double>
      tfDouble;

  exec_aten::Tensor self = tfByte.make({2, 2}, {1, 2, 3, 4});
  ::std::vector<int64_t> dims_vec = {1, 0};
  exec_aten::ArrayRef<int64_t> dims =
      exec_aten::ArrayRef<int64_t>(dims_vec.data(), dims_vec.size());
  exec_aten::Tensor out = tfDouble.zeros({2, 2});
  ET_EXPECT_KERNEL_FAILURE(op_permute_copy_out(self, dims, out));
}

TEST(OpPermuteCopyOutTest, DtypeTest_uint8_uint8) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Byte> tfByte;

  exec_aten::Tensor self = tfByte.make({2, 2}, {1, 2, 3, 4});
  ::std::vector<int64_t> dims_vec = {1, 0};
  exec_aten::ArrayRef<int64_t> dims =
      exec_aten::ArrayRef<int64_t>(dims_vec.data(), dims_vec.size());
  exec_aten::Tensor out = tfByte.zeros({2, 2});
  exec_aten::Tensor out_expected = tfByte.make({2, 2}, {1, 3, 2, 4});
  op_permute_copy_out(self, dims, out);
  EXPECT_TENSOR_CLOSE(out, out_expected);
}

TEST(OpPermuteCopyOutTest, DtypeTest_uint8_int8) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Byte> tfByte;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Char> tfChar;

  exec_aten::Tensor self = tfByte.make({2, 2}, {1, 2, 3, 4});
  ::std::vector<int64_t> dims_vec = {1, 0};
  exec_aten::ArrayRef<int64_t> dims =
      exec_aten::ArrayRef<int64_t>(dims_vec.data(), dims_vec.size());
  exec_aten::Tensor out = tfChar.zeros({2, 2});
  ET_EXPECT_KERNEL_FAILURE(op_permute_copy_out(self, dims, out));
}

TEST(OpPermuteCopyOutTest, DtypeTest_uint8_int16) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Byte> tfByte;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Short> tfShort;

  exec_aten::Tensor self = tfByte.make({2, 2}, {1, 2, 3, 4});
  ::std::vector<int64_t> dims_vec = {1, 0};
  exec_aten::ArrayRef<int64_t> dims =
      exec_aten::ArrayRef<int64_t>(dims_vec.data(), dims_vec.size());
  exec_aten::Tensor out = tfShort.zeros({2, 2});
  ET_EXPECT_KERNEL_FAILURE(op_permute_copy_out(self, dims, out));
}

TEST(OpPermuteCopyOutTest, DtypeTest_uint8_int32) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Byte> tfByte;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Int> tfInt;

  exec_aten::Tensor self = tfByte.make({2, 2}, {1, 2, 3, 4});
  ::std::vector<int64_t> dims_vec = {1, 0};
  exec_aten::ArrayRef<int64_t> dims =
      exec_aten::ArrayRef<int64_t>(dims_vec.data(), dims_vec.size());
  exec_aten::Tensor out = tfInt.zeros({2, 2});
  ET_EXPECT_KERNEL_FAILURE(op_permute_copy_out(self, dims, out));
}

TEST(OpPermuteCopyOutTest, DtypeTest_uint8_int64) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Byte> tfByte;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Long> tfLong;

  exec_aten::Tensor self = tfByte.make({2, 2}, {1, 2, 3, 4});
  ::std::vector<int64_t> dims_vec = {1, 0};
  exec_aten::ArrayRef<int64_t> dims =
      exec_aten::ArrayRef<int64_t>(dims_vec.data(), dims_vec.size());
  exec_aten::Tensor out = tfLong.zeros({2, 2});
  ET_EXPECT_KERNEL_FAILURE(op_permute_copy_out(self, dims, out));
}

TEST(OpPermuteCopyOutTest, DtypeTest_uint8_bool) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Bool> tfBool;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Byte> tfByte;

  exec_aten::Tensor self = tfByte.make({2, 2}, {1, 2, 3, 4});
  ::std::vector<int64_t> dims_vec = {1, 0};
  exec_aten::ArrayRef<int64_t> dims =
      exec_aten::ArrayRef<int64_t>(dims_vec.data(), dims_vec.size());
  exec_aten::Tensor out = tfBool.zeros({2, 2});
  ET_EXPECT_KERNEL_FAILURE(op_permute_copy_out(self, dims, out));
}

TEST(OpPermuteCopyOutTest, DtypeTest_int8_float32) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Char> tfChar;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Float> tfFloat;

  exec_aten::Tensor self = tfChar.make({2, 2}, {1, 2, 3, 4});
  ::std::vector<int64_t> dims_vec = {1, 0};
  exec_aten::ArrayRef<int64_t> dims =
      exec_aten::ArrayRef<int64_t>(dims_vec.data(), dims_vec.size());
  exec_aten::Tensor out = tfFloat.zeros({2, 2});
  ET_EXPECT_KERNEL_FAILURE(op_permute_copy_out(self, dims, out));
}

TEST(OpPermuteCopyOutTest, DtypeTest_int8_float64) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Char> tfChar;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Double>
      tfDouble;

  exec_aten::Tensor self = tfChar.make({2, 2}, {1, 2, 3, 4});
  ::std::vector<int64_t> dims_vec = {1, 0};
  exec_aten::ArrayRef<int64_t> dims =
      exec_aten::ArrayRef<int64_t>(dims_vec.data(), dims_vec.size());
  exec_aten::Tensor out = tfDouble.zeros({2, 2});
  ET_EXPECT_KERNEL_FAILURE(op_permute_copy_out(self, dims, out));
}

TEST(OpPermuteCopyOutTest, DtypeTest_int8_uint8) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Byte> tfByte;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Char> tfChar;

  exec_aten::Tensor self = tfChar.make({2, 2}, {1, 2, 3, 4});
  ::std::vector<int64_t> dims_vec = {1, 0};
  exec_aten::ArrayRef<int64_t> dims =
      exec_aten::ArrayRef<int64_t>(dims_vec.data(), dims_vec.size());
  exec_aten::Tensor out = tfByte.zeros({2, 2});
  ET_EXPECT_KERNEL_FAILURE(op_permute_copy_out(self, dims, out));
}

TEST(OpPermuteCopyOutTest, DtypeTest_int8_int8) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Char> tfChar;

  exec_aten::Tensor self = tfChar.make({2, 2}, {1, 2, 3, 4});
  ::std::vector<int64_t> dims_vec = {1, 0};
  exec_aten::ArrayRef<int64_t> dims =
      exec_aten::ArrayRef<int64_t>(dims_vec.data(), dims_vec.size());
  exec_aten::Tensor out = tfChar.zeros({2, 2});
  exec_aten::Tensor out_expected = tfChar.make({2, 2}, {1, 3, 2, 4});
  op_permute_copy_out(self, dims, out);
  EXPECT_TENSOR_CLOSE(out, out_expected);
}

TEST(OpPermuteCopyOutTest, DtypeTest_int8_int16) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Char> tfChar;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Short> tfShort;

  exec_aten::Tensor self = tfChar.make({2, 2}, {1, 2, 3, 4});
  ::std::vector<int64_t> dims_vec = {1, 0};
  exec_aten::ArrayRef<int64_t> dims =
      exec_aten::ArrayRef<int64_t>(dims_vec.data(), dims_vec.size());
  exec_aten::Tensor out = tfShort.zeros({2, 2});
  ET_EXPECT_KERNEL_FAILURE(op_permute_copy_out(self, dims, out));
}

TEST(OpPermuteCopyOutTest, DtypeTest_int8_int32) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Char> tfChar;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Int> tfInt;

  exec_aten::Tensor self = tfChar.make({2, 2}, {1, 2, 3, 4});
  ::std::vector<int64_t> dims_vec = {1, 0};
  exec_aten::ArrayRef<int64_t> dims =
      exec_aten::ArrayRef<int64_t>(dims_vec.data(), dims_vec.size());
  exec_aten::Tensor out = tfInt.zeros({2, 2});
  ET_EXPECT_KERNEL_FAILURE(op_permute_copy_out(self, dims, out));
}

TEST(OpPermuteCopyOutTest, DtypeTest_int8_int64) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Char> tfChar;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Long> tfLong;

  exec_aten::Tensor self = tfChar.make({2, 2}, {1, 2, 3, 4});
  ::std::vector<int64_t> dims_vec = {1, 0};
  exec_aten::ArrayRef<int64_t> dims =
      exec_aten::ArrayRef<int64_t>(dims_vec.data(), dims_vec.size());
  exec_aten::Tensor out = tfLong.zeros({2, 2});
  ET_EXPECT_KERNEL_FAILURE(op_permute_copy_out(self, dims, out));
}

TEST(OpPermuteCopyOutTest, DtypeTest_int8_bool) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Bool> tfBool;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Char> tfChar;

  exec_aten::Tensor self = tfChar.make({2, 2}, {1, 2, 3, 4});
  ::std::vector<int64_t> dims_vec = {1, 0};
  exec_aten::ArrayRef<int64_t> dims =
      exec_aten::ArrayRef<int64_t>(dims_vec.data(), dims_vec.size());
  exec_aten::Tensor out = tfBool.zeros({2, 2});
  ET_EXPECT_KERNEL_FAILURE(op_permute_copy_out(self, dims, out));
}

TEST(OpPermuteCopyOutTest, DtypeTest_int16_float32) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Float> tfFloat;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Short> tfShort;

  exec_aten::Tensor self = tfShort.make({2, 2}, {1, 2, 3, 4});
  ::std::vector<int64_t> dims_vec = {1, 0};
  exec_aten::ArrayRef<int64_t> dims =
      exec_aten::ArrayRef<int64_t>(dims_vec.data(), dims_vec.size());
  exec_aten::Tensor out = tfFloat.zeros({2, 2});
  ET_EXPECT_KERNEL_FAILURE(op_permute_copy_out(self, dims, out));
}

TEST(OpPermuteCopyOutTest, DtypeTest_int16_float64) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Double>
      tfDouble;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Short> tfShort;

  exec_aten::Tensor self = tfShort.make({2, 2}, {1, 2, 3, 4});
  ::std::vector<int64_t> dims_vec = {1, 0};
  exec_aten::ArrayRef<int64_t> dims =
      exec_aten::ArrayRef<int64_t>(dims_vec.data(), dims_vec.size());
  exec_aten::Tensor out = tfDouble.zeros({2, 2});
  ET_EXPECT_KERNEL_FAILURE(op_permute_copy_out(self, dims, out));
}

TEST(OpPermuteCopyOutTest, DtypeTest_int16_uint8) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Byte> tfByte;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Short> tfShort;

  exec_aten::Tensor self = tfShort.make({2, 2}, {1, 2, 3, 4});
  ::std::vector<int64_t> dims_vec = {1, 0};
  exec_aten::ArrayRef<int64_t> dims =
      exec_aten::ArrayRef<int64_t>(dims_vec.data(), dims_vec.size());
  exec_aten::Tensor out = tfByte.zeros({2, 2});
  ET_EXPECT_KERNEL_FAILURE(op_permute_copy_out(self, dims, out));
}

TEST(OpPermuteCopyOutTest, DtypeTest_int16_int8) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Char> tfChar;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Short> tfShort;

  exec_aten::Tensor self = tfShort.make({2, 2}, {1, 2, 3, 4});
  ::std::vector<int64_t> dims_vec = {1, 0};
  exec_aten::ArrayRef<int64_t> dims =
      exec_aten::ArrayRef<int64_t>(dims_vec.data(), dims_vec.size());
  exec_aten::Tensor out = tfChar.zeros({2, 2});
  ET_EXPECT_KERNEL_FAILURE(op_permute_copy_out(self, dims, out));
}

TEST(OpPermuteCopyOutTest, DtypeTest_int16_int16) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Short> tfShort;

  exec_aten::Tensor self = tfShort.make({2, 2}, {1, 2, 3, 4});
  ::std::vector<int64_t> dims_vec = {1, 0};
  exec_aten::ArrayRef<int64_t> dims =
      exec_aten::ArrayRef<int64_t>(dims_vec.data(), dims_vec.size());
  exec_aten::Tensor out = tfShort.zeros({2, 2});
  exec_aten::Tensor out_expected = tfShort.make({2, 2}, {1, 3, 2, 4});
  op_permute_copy_out(self, dims, out);
  EXPECT_TENSOR_CLOSE(out, out_expected);
}

TEST(OpPermuteCopyOutTest, DtypeTest_int16_int32) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Int> tfInt;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Short> tfShort;

  exec_aten::Tensor self = tfShort.make({2, 2}, {1, 2, 3, 4});
  ::std::vector<int64_t> dims_vec = {1, 0};
  exec_aten::ArrayRef<int64_t> dims =
      exec_aten::ArrayRef<int64_t>(dims_vec.data(), dims_vec.size());
  exec_aten::Tensor out = tfInt.zeros({2, 2});
  ET_EXPECT_KERNEL_FAILURE(op_permute_copy_out(self, dims, out));
}

TEST(OpPermuteCopyOutTest, DtypeTest_int16_int64) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Long> tfLong;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Short> tfShort;

  exec_aten::Tensor self = tfShort.make({2, 2}, {1, 2, 3, 4});
  ::std::vector<int64_t> dims_vec = {1, 0};
  exec_aten::ArrayRef<int64_t> dims =
      exec_aten::ArrayRef<int64_t>(dims_vec.data(), dims_vec.size());
  exec_aten::Tensor out = tfLong.zeros({2, 2});
  ET_EXPECT_KERNEL_FAILURE(op_permute_copy_out(self, dims, out));
}

TEST(OpPermuteCopyOutTest, DtypeTest_int16_bool) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Bool> tfBool;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Short> tfShort;

  exec_aten::Tensor self = tfShort.make({2, 2}, {1, 2, 3, 4});
  ::std::vector<int64_t> dims_vec = {1, 0};
  exec_aten::ArrayRef<int64_t> dims =
      exec_aten::ArrayRef<int64_t>(dims_vec.data(), dims_vec.size());
  exec_aten::Tensor out = tfBool.zeros({2, 2});
  ET_EXPECT_KERNEL_FAILURE(op_permute_copy_out(self, dims, out));
}

TEST(OpPermuteCopyOutTest, DtypeTest_int32_float32) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Float> tfFloat;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Int> tfInt;

  exec_aten::Tensor self = tfInt.make({2, 2}, {1, 2, 3, 4});
  ::std::vector<int64_t> dims_vec = {1, 0};
  exec_aten::ArrayRef<int64_t> dims =
      exec_aten::ArrayRef<int64_t>(dims_vec.data(), dims_vec.size());
  exec_aten::Tensor out = tfFloat.zeros({2, 2});
  ET_EXPECT_KERNEL_FAILURE(op_permute_copy_out(self, dims, out));
}

TEST(OpPermuteCopyOutTest, DtypeTest_int32_float64) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Double>
      tfDouble;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Int> tfInt;

  exec_aten::Tensor self = tfInt.make({2, 2}, {1, 2, 3, 4});
  ::std::vector<int64_t> dims_vec = {1, 0};
  exec_aten::ArrayRef<int64_t> dims =
      exec_aten::ArrayRef<int64_t>(dims_vec.data(), dims_vec.size());
  exec_aten::Tensor out = tfDouble.zeros({2, 2});
  ET_EXPECT_KERNEL_FAILURE(op_permute_copy_out(self, dims, out));
}

TEST(OpPermuteCopyOutTest, DtypeTest_int32_uint8) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Byte> tfByte;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Int> tfInt;

  exec_aten::Tensor self = tfInt.make({2, 2}, {1, 2, 3, 4});
  ::std::vector<int64_t> dims_vec = {1, 0};
  exec_aten::ArrayRef<int64_t> dims =
      exec_aten::ArrayRef<int64_t>(dims_vec.data(), dims_vec.size());
  exec_aten::Tensor out = tfByte.zeros({2, 2});
  ET_EXPECT_KERNEL_FAILURE(op_permute_copy_out(self, dims, out));
}

TEST(OpPermuteCopyOutTest, DtypeTest_int32_int8) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Char> tfChar;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Int> tfInt;

  exec_aten::Tensor self = tfInt.make({2, 2}, {1, 2, 3, 4});
  ::std::vector<int64_t> dims_vec = {1, 0};
  exec_aten::ArrayRef<int64_t> dims =
      exec_aten::ArrayRef<int64_t>(dims_vec.data(), dims_vec.size());
  exec_aten::Tensor out = tfChar.zeros({2, 2});
  ET_EXPECT_KERNEL_FAILURE(op_permute_copy_out(self, dims, out));
}

TEST(OpPermuteCopyOutTest, DtypeTest_int32_int16) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Int> tfInt;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Short> tfShort;

  exec_aten::Tensor self = tfInt.make({2, 2}, {1, 2, 3, 4});
  ::std::vector<int64_t> dims_vec = {1, 0};
  exec_aten::ArrayRef<int64_t> dims =
      exec_aten::ArrayRef<int64_t>(dims_vec.data(), dims_vec.size());
  exec_aten::Tensor out = tfShort.zeros({2, 2});
  ET_EXPECT_KERNEL_FAILURE(op_permute_copy_out(self, dims, out));
}

TEST(OpPermuteCopyOutTest, DtypeTest_int32_int32) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Int> tfInt;

  exec_aten::Tensor self = tfInt.make({2, 2}, {1, 2, 3, 4});
  ::std::vector<int64_t> dims_vec = {1, 0};
  exec_aten::ArrayRef<int64_t> dims =
      exec_aten::ArrayRef<int64_t>(dims_vec.data(), dims_vec.size());
  exec_aten::Tensor out = tfInt.zeros({2, 2});
  exec_aten::Tensor out_expected = tfInt.make({2, 2}, {1, 3, 2, 4});
  op_permute_copy_out(self, dims, out);
  EXPECT_TENSOR_CLOSE(out, out_expected);
}

TEST(OpPermuteCopyOutTest, DtypeTest_int32_int64) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Int> tfInt;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Long> tfLong;

  exec_aten::Tensor self = tfInt.make({2, 2}, {1, 2, 3, 4});
  ::std::vector<int64_t> dims_vec = {1, 0};
  exec_aten::ArrayRef<int64_t> dims =
      exec_aten::ArrayRef<int64_t>(dims_vec.data(), dims_vec.size());
  exec_aten::Tensor out = tfLong.zeros({2, 2});
  ET_EXPECT_KERNEL_FAILURE(op_permute_copy_out(self, dims, out));
}

TEST(OpPermuteCopyOutTest, DtypeTest_int32_bool) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Bool> tfBool;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Int> tfInt;

  exec_aten::Tensor self = tfInt.make({2, 2}, {1, 2, 3, 4});
  ::std::vector<int64_t> dims_vec = {1, 0};
  exec_aten::ArrayRef<int64_t> dims =
      exec_aten::ArrayRef<int64_t>(dims_vec.data(), dims_vec.size());
  exec_aten::Tensor out = tfBool.zeros({2, 2});
  ET_EXPECT_KERNEL_FAILURE(op_permute_copy_out(self, dims, out));
}

TEST(OpPermuteCopyOutTest, DtypeTest_int64_float32) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Float> tfFloat;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Long> tfLong;

  exec_aten::Tensor self = tfLong.make({2, 2}, {1, 2, 3, 4});
  ::std::vector<int64_t> dims_vec = {1, 0};
  exec_aten::ArrayRef<int64_t> dims =
      exec_aten::ArrayRef<int64_t>(dims_vec.data(), dims_vec.size());
  exec_aten::Tensor out = tfFloat.zeros({2, 2});
  ET_EXPECT_KERNEL_FAILURE(op_permute_copy_out(self, dims, out));
}

TEST(OpPermuteCopyOutTest, DtypeTest_int64_float64) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Double>
      tfDouble;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Long> tfLong;

  exec_aten::Tensor self = tfLong.make({2, 2}, {1, 2, 3, 4});
  ::std::vector<int64_t> dims_vec = {1, 0};
  exec_aten::ArrayRef<int64_t> dims =
      exec_aten::ArrayRef<int64_t>(dims_vec.data(), dims_vec.size());
  exec_aten::Tensor out = tfDouble.zeros({2, 2});
  ET_EXPECT_KERNEL_FAILURE(op_permute_copy_out(self, dims, out));
}

TEST(OpPermuteCopyOutTest, DtypeTest_int64_uint8) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Byte> tfByte;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Long> tfLong;

  exec_aten::Tensor self = tfLong.make({2, 2}, {1, 2, 3, 4});
  ::std::vector<int64_t> dims_vec = {1, 0};
  exec_aten::ArrayRef<int64_t> dims =
      exec_aten::ArrayRef<int64_t>(dims_vec.data(), dims_vec.size());
  exec_aten::Tensor out = tfByte.zeros({2, 2});
  ET_EXPECT_KERNEL_FAILURE(op_permute_copy_out(self, dims, out));
}

TEST(OpPermuteCopyOutTest, DtypeTest_int64_int8) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Char> tfChar;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Long> tfLong;

  exec_aten::Tensor self = tfLong.make({2, 2}, {1, 2, 3, 4});
  ::std::vector<int64_t> dims_vec = {1, 0};
  exec_aten::ArrayRef<int64_t> dims =
      exec_aten::ArrayRef<int64_t>(dims_vec.data(), dims_vec.size());
  exec_aten::Tensor out = tfChar.zeros({2, 2});
  ET_EXPECT_KERNEL_FAILURE(op_permute_copy_out(self, dims, out));
}

TEST(OpPermuteCopyOutTest, DtypeTest_int64_int16) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Long> tfLong;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Short> tfShort;

  exec_aten::Tensor self = tfLong.make({2, 2}, {1, 2, 3, 4});
  ::std::vector<int64_t> dims_vec = {1, 0};
  exec_aten::ArrayRef<int64_t> dims =
      exec_aten::ArrayRef<int64_t>(dims_vec.data(), dims_vec.size());
  exec_aten::Tensor out = tfShort.zeros({2, 2});
  ET_EXPECT_KERNEL_FAILURE(op_permute_copy_out(self, dims, out));
}

TEST(OpPermuteCopyOutTest, DtypeTest_int64_int32) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Int> tfInt;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Long> tfLong;

  exec_aten::Tensor self = tfLong.make({2, 2}, {1, 2, 3, 4});
  ::std::vector<int64_t> dims_vec = {1, 0};
  exec_aten::ArrayRef<int64_t> dims =
      exec_aten::ArrayRef<int64_t>(dims_vec.data(), dims_vec.size());
  exec_aten::Tensor out = tfInt.zeros({2, 2});
  ET_EXPECT_KERNEL_FAILURE(op_permute_copy_out(self, dims, out));
}

TEST(OpPermuteCopyOutTest, DtypeTest_int64_int64) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Long> tfLong;

  exec_aten::Tensor self = tfLong.make({2, 2}, {1, 2, 3, 4});
  ::std::vector<int64_t> dims_vec = {1, 0};
  exec_aten::ArrayRef<int64_t> dims =
      exec_aten::ArrayRef<int64_t>(dims_vec.data(), dims_vec.size());
  exec_aten::Tensor out = tfLong.zeros({2, 2});
  exec_aten::Tensor out_expected = tfLong.make({2, 2}, {1, 3, 2, 4});
  op_permute_copy_out(self, dims, out);
  EXPECT_TENSOR_CLOSE(out, out_expected);
}

TEST(OpPermuteCopyOutTest, DtypeTest_int64_bool) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Bool> tfBool;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Long> tfLong;

  exec_aten::Tensor self = tfLong.make({2, 2}, {1, 2, 3, 4});
  ::std::vector<int64_t> dims_vec = {1, 0};
  exec_aten::ArrayRef<int64_t> dims =
      exec_aten::ArrayRef<int64_t>(dims_vec.data(), dims_vec.size());
  exec_aten::Tensor out = tfBool.zeros({2, 2});
  ET_EXPECT_KERNEL_FAILURE(op_permute_copy_out(self, dims, out));
}

TEST(OpPermuteCopyOutTest, DtypeTest_bool_float32) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Bool> tfBool;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Float> tfFloat;

  exec_aten::Tensor self = tfBool.make({2, 2}, {true, false, false, true});
  ::std::vector<int64_t> dims_vec = {1, 0};
  exec_aten::ArrayRef<int64_t> dims =
      exec_aten::ArrayRef<int64_t>(dims_vec.data(), dims_vec.size());
  exec_aten::Tensor out = tfFloat.zeros({2, 2});
  ET_EXPECT_KERNEL_FAILURE(op_permute_copy_out(self, dims, out));
}

TEST(OpPermuteCopyOutTest, DtypeTest_bool_float64) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Bool> tfBool;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Double>
      tfDouble;

  exec_aten::Tensor self = tfBool.make({2, 2}, {true, false, false, true});
  ::std::vector<int64_t> dims_vec = {1, 0};
  exec_aten::ArrayRef<int64_t> dims =
      exec_aten::ArrayRef<int64_t>(dims_vec.data(), dims_vec.size());
  exec_aten::Tensor out = tfDouble.zeros({2, 2});
  ET_EXPECT_KERNEL_FAILURE(op_permute_copy_out(self, dims, out));
}

TEST(OpPermuteCopyOutTest, DtypeTest_bool_uint8) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Bool> tfBool;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Byte> tfByte;

  exec_aten::Tensor self = tfBool.make({2, 2}, {true, false, false, true});
  ::std::vector<int64_t> dims_vec = {1, 0};
  exec_aten::ArrayRef<int64_t> dims =
      exec_aten::ArrayRef<int64_t>(dims_vec.data(), dims_vec.size());
  exec_aten::Tensor out = tfByte.zeros({2, 2});
  ET_EXPECT_KERNEL_FAILURE(op_permute_copy_out(self, dims, out));
}

TEST(OpPermuteCopyOutTest, DtypeTest_bool_int8) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Bool> tfBool;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Char> tfChar;

  exec_aten::Tensor self = tfBool.make({2, 2}, {true, false, false, true});
  ::std::vector<int64_t> dims_vec = {1, 0};
  exec_aten::ArrayRef<int64_t> dims =
      exec_aten::ArrayRef<int64_t>(dims_vec.data(), dims_vec.size());
  exec_aten::Tensor out = tfChar.zeros({2, 2});
  ET_EXPECT_KERNEL_FAILURE(op_permute_copy_out(self, dims, out));
}

TEST(OpPermuteCopyOutTest, DtypeTest_bool_int16) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Bool> tfBool;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Short> tfShort;

  exec_aten::Tensor self = tfBool.make({2, 2}, {true, false, false, true});
  ::std::vector<int64_t> dims_vec = {1, 0};
  exec_aten::ArrayRef<int64_t> dims =
      exec_aten::ArrayRef<int64_t>(dims_vec.data(), dims_vec.size());
  exec_aten::Tensor out = tfShort.zeros({2, 2});
  ET_EXPECT_KERNEL_FAILURE(op_permute_copy_out(self, dims, out));
}

TEST(OpPermuteCopyOutTest, DtypeTest_bool_int32) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Bool> tfBool;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Int> tfInt;

  exec_aten::Tensor self = tfBool.make({2, 2}, {true, false, false, true});
  ::std::vector<int64_t> dims_vec = {1, 0};
  exec_aten::ArrayRef<int64_t> dims =
      exec_aten::ArrayRef<int64_t>(dims_vec.data(), dims_vec.size());
  exec_aten::Tensor out = tfInt.zeros({2, 2});
  ET_EXPECT_KERNEL_FAILURE(op_permute_copy_out(self, dims, out));
}

TEST(OpPermuteCopyOutTest, DtypeTest_bool_int64) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Bool> tfBool;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Long> tfLong;

  exec_aten::Tensor self = tfBool.make({2, 2}, {true, false, false, true});
  ::std::vector<int64_t> dims_vec = {1, 0};
  exec_aten::ArrayRef<int64_t> dims =
      exec_aten::ArrayRef<int64_t>(dims_vec.data(), dims_vec.size());
  exec_aten::Tensor out = tfLong.zeros({2, 2});
  ET_EXPECT_KERNEL_FAILURE(op_permute_copy_out(self, dims, out));
}

TEST(OpPermuteCopyOutTest, DtypeTest_bool_bool) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Bool> tfBool;

  exec_aten::Tensor self = tfBool.make({2, 2}, {true, false, false, true});
  ::std::vector<int64_t> dims_vec = {1, 0};
  exec_aten::ArrayRef<int64_t> dims =
      exec_aten::ArrayRef<int64_t>(dims_vec.data(), dims_vec.size());
  exec_aten::Tensor out = tfBool.zeros({2, 2});
  exec_aten::Tensor out_expected =
      tfBool.make({2, 2}, {true, false, false, true});
  op_permute_copy_out(self, dims, out);
  EXPECT_TENSOR_CLOSE(out, out_expected);
}
